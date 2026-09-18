"""Parallel fly experiment pack. 75% of logical CPUs, OMP_NUM_THREADS=1."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any

from .bench import run_fly_bench
from .select import FlyCandidate, load_champion, load_config, save_champion
from .autoresearch import _promote_bundle


def _workers() -> int:
    n = os.cpu_count() or 4
    return max(1, int(n * 0.75))


def _from_league(root: Path) -> FlyCandidate:
    champ = load_champion(root / "runs" / "autoresearch" / "league")
    if champ is None:
        raise SystemExit("no league champion; run autoresearch first")
    return champ.candidate


def pack_candidates(base: FlyCandidate) -> list[FlyCandidate]:
    out: list[FlyCandidate] = []

    def add(**kwargs: Any) -> None:
        d = base.to_dict()
        d.update(kwargs)
        desc = kwargs.get("description") or json.dumps(kwargs, sort_keys=True)
        d["description"] = desc
        g = deepcopy(d["genome"])
        if "hidden" in kwargs:
            g["architecture"] = dict(g["architecture"])
            g["architecture"]["hidden"] = int(kwargs["hidden"])
            g["id"] = f"{g.get('lineage','mb')}-sw-{len(out)}"
        if "seed" in kwargs:
            g["training"] = dict(g["training"])
            g["training"]["seed"] = int(kwargs["seed"])
        d["genome"] = g
        out.append(FlyCandidate.from_dict(d))

    add(description="control-champion")
    for kw in (5, 8, 10, 12, 15, 19):
        add(k_winners=kw, description=f"k_winners={kw}")
    for h in (128,):
        for kw in (8, 10, 13):
            add(hidden=h, k_winners=kw, description=f"hidden={h},k_winners={kw}")
    for lr in (0.15, 0.25, 0.45, 0.55):
        for ep in (10, 15, 30, 40):
            add(plasticity_lr=lr, plasticity_epochs=ep, description=f"lr={lr},epochs={ep}")
    for seed in (0, 2):
        add(seed=seed, description=f"seed={seed}")
    for d in (1, 2):
        add(dagger_rounds=d, description=f"dagger_rounds={d}")
    return out


def _init_worker() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def _run_one(payload: dict[str, Any]) -> dict[str, Any]:
    _init_worker()
    from .select import FlyCandidate, load_config
    from .bench import run_fly_bench

    cand = FlyCandidate.from_dict(payload["candidate"])
    cfg = load_config()
    cfg = json.loads(json.dumps(cfg))
    cfg["bench"]["omp_smoke"] = False
    root = Path(payload["run_dir"])
    t0 = time.perf_counter()
    result = run_fly_bench(
        cand,
        config=cfg,
        run_dir=root,
        experiment_id=payload["experiment_id"],
        skip_unit_tests=True,
    )
    return {
        "experiment_id": result.experiment_id,
        "description": cand.description,
        "latency_score": result.latency_score,
        "gates_pass": result.gates_pass,
        "gate_reasons": result.gate_reasons,
        "closed_loop_reward": result.metrics.closed_loop_reward,
        "confirm_success": result.metrics.confirm_success,
        "n_params": result.metrics.n_params,
        "advise_warm_p99_ms": result.metrics.advise_warm_p99_ms,
        "closed_loop_ms": result.metrics.closed_loop_ms,
        "wall_s": time.perf_counter() - t0,
        "k_winners": cand.k_winners,
        "dagger_rounds": cand.dagger_rounds,
    }


def run_sweep(*, workers: int | None = None) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    n = workers or _workers()
    base = _from_league(root)
    cands = pack_candidates(base)
    run_dir = root / "runs" / "sweep-parallel"
    run_dir.mkdir(parents=True, exist_ok=True)
    league = load_champion(root / "runs" / "autoresearch" / "league")
    champ_score = league.latency_score if league else 1e9
    jobs = []
    for i, cand in enumerate(cands):
        exp_id = f"sw-{i:03d}"
        jobs.append(
            {
                "experiment_id": exp_id,
                "candidate": cand.to_dict(),
                "run_dir": str(run_dir),
            }
        )
    print(json.dumps({"sweep": "start", "workers": n, "n_jobs": len(jobs), "champion": champ_score}), flush=True)
    rows: list[dict[str, Any]] = []
    keeps: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n, initializer=_init_worker) as pool:
        futs = {pool.submit(_run_one, job): job["experiment_id"] for job in jobs}
        for fut in as_completed(futs):
            exp = futs[fut]
            try:
                row = fut.result()
            except Exception as exc:  # noqa: BLE001
                row = {"experiment_id": exp, "error": str(exc), "gates_pass": False, "latency_score": 0.0}
            rows.append(row)
            keep = bool(row.get("gates_pass")) and float(row.get("latency_score") or 1e9) < champ_score * 0.99
            row["keep"] = keep
            if keep:
                keeps.append(row)
            print(json.dumps({"done": len(rows), "total": len(jobs), **{k: row.get(k) for k in ("experiment_id", "description", "latency_score", "gates_pass", "closed_loop_reward", "keep", "wall_s")}}, default=str), flush=True)
    rows.sort(key=lambda r: (not r.get("gates_pass", False), r.get("latency_score") or 1e9))
    summary = {
        "workers": n,
        "n_jobs": len(jobs),
        "elapsed_s": time.perf_counter() - t0,
        "champion_before": champ_score,
        "keeps": keeps,
        "best_gated": next((r for r in rows if r.get("gates_pass")), None),
        "rows": rows,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"sweep": "done", "elapsed_s": summary["elapsed_s"], "keeps": len(keeps), "best": summary["best_gated"]}, default=str), flush=True)
    return summary
