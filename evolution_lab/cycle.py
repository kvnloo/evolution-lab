"""Outer cycle around autoresearch: evolve the production fly without stopping.

Each session is one bounded `run_autoresearch_loop`. The league champion is
the global best; sessions resume from it. SearchDriver-only — do not import
bench internals beyond the public run/select API.
"""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .autoresearch import _promote_bundle, run_autoresearch_loop
from .sweep import run_sweep
from .select import BenchResult, load_champion, load_config, save_champion


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def league_dir(root: Path | None = None) -> Path:
    return (root or _root()) / "runs" / "autoresearch" / "league"


def seed_league(root: Path) -> Path:
    dest = league_dir(root)
    dest.mkdir(parents=True, exist_ok=True)
    champ = dest / "champion.json"
    if champ.is_file():
        return dest
    prior = root / "runs" / "autoresearch" / "20260917" / "champion.json"
    if prior.is_file():
        shutil.copy2(prior, champ)
    return dest


def _write_overlay(cfg: dict[str, Any], epsilon: float, path: Path) -> Path:
    overlay = json.loads(json.dumps(cfg))
    overlay.setdefault("objective", {})["improve_epsilon"] = float(epsilon)
    space = overlay.setdefault("search_space", {})
    space.pop("history", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(overlay, indent=2) + "\n", encoding="utf-8")
    return path


def _better(challenger: BenchResult | None, incumbent: BenchResult | None) -> bool:
    if challenger is None or not challenger.gates_pass:
        return False
    if incumbent is None:
        return True
    return challenger.latency_score + 1e-12 < incumbent.latency_score


def run_cycle(
    *,
    forever: bool = True,
    max_sessions: int | None = None,
    max_hours: float | None = None,
    max_experiments: int | None = None,
    patience: int | None = None,
    skip_unit_tests: bool = True,
    promote_bundle: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    root = _root()
    cfg = load_config()
    cycle_cfg = cfg.get("cycle") or {}
    league = seed_league(root)
    epsilon = float(cycle_cfg.get("epsilon_start", 0.05))
    eps_min = float(cycle_cfg.get("epsilon_min", 0.01))
    shrink = float(cycle_cfg.get("epsilon_shrink", 0.5))
    idle_limit = int(cycle_cfg.get("shrink_after_idle_sessions", 2))
    stride = int(cycle_cfg.get("seed_stride", 17))
    per_exp = int(max_experiments if max_experiments is not None else cycle_cfg.get("max_experiments_per_session", 25))
    per_pat = int(patience if patience is not None else cycle_cfg.get("patience_per_session", 12))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    t0 = time.time()
    session = 0
    idle = 0
    keeps = 0
    print("cycle_session: 0 starting", flush=True)
    status: dict[str, Any] = {
        "schema": "flyforge.cycle.v1",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "sessions": 0,
        "keeps": 0,
        "epsilon": epsilon,
        "stopped_reason": None,
    }
    while True:
        if max_sessions is not None and session >= max_sessions:
            status["stopped_reason"] = "max_sessions"
            break
        if max_hours is not None and (time.time() - t0) >= max_hours * 3600.0:
            status["stopped_reason"] = "max_hours"
            break
        if not forever and max_sessions is None and max_hours is None:
            status["stopped_reason"] = "not_forever"
            break
        session += 1
        tag = f"cycle-{stamp}-{session:03d}"
        print(f"cycle_session: {session} tag={tag} epsilon={epsilon:.4f}", flush=True)
        overlay = _write_overlay(cfg, epsilon, league / "overlay.json")
        sess_dir = root / "runs" / "autoresearch" / tag
        sess_dir.mkdir(parents=True, exist_ok=True)
        league_champ_path = league / "champion.json"
        if league_champ_path.is_file():
            shutil.copy2(league_champ_path, sess_dir / "champion.json")
        summary = run_autoresearch_loop(
            config_path=overlay,
            run_dir=sess_dir,
            max_experiments=per_exp,
            patience=per_pat,
            promote_bundle=False,
            skip_unit_tests=skip_unit_tests,
            seed=int(seed) + session * stride,
        )
        sess_champ = load_champion(sess_dir)
        league_champ = load_champion(league)
        if _better(sess_champ, league_champ):
            assert sess_champ is not None
            save_champion(league, sess_champ)
            if promote_bundle:
                dest = _promote_bundle(sess_champ, root)
                print(json.dumps({"promoted_bundle": str(dest), "tag": tag}, indent=2), flush=True)
            keeps += 1
            idle = 0
            epsilon = float(cycle_cfg.get("epsilon_start", 0.05))
        else:
            idle += 1
            if idle >= idle_limit:
                epsilon = max(eps_min, epsilon * shrink)
                if bool(cycle_cfg.get("idle_sweep", False)):
                    print(json.dumps({"cycle": "idle_sweep", "session": session, "epsilon": epsilon}), flush=True)
                    sweep = run_sweep()
                    print(json.dumps({"cycle": "idle_sweep_done", "keeps": len(sweep.get("keeps") or [])}), flush=True)
        lc = load_champion(league)
        status.update(
            {
                "sessions": session,
                "keeps": keeps,
                "idle_sessions": idle,
                "epsilon": epsilon,
                "last_tag": tag,
                "last_session_summary": summary,
                "league_experiment_id": None if lc is None else lc.experiment_id,
                "league_latency_score": None if lc is None else lc.latency_score,
                "league_description": None if lc is None else lc.candidate.description,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        (league / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: status[k] for k in ("sessions", "keeps", "epsilon", "last_tag", "league_latency_score") if k in status}, indent=2), flush=True)
        if not forever and max_sessions is None and max_hours is None:
            status["stopped_reason"] = "single_session"
            break
    (league / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return status
