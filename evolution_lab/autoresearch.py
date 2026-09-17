"""FlyForge autoresearch loop — optimize recovery kernel like a profiler."""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .autoresearch_propose import default_champion_candidate, propose_candidate
from .bench import candidate_to_genome, run_fly_bench, train_candidate
from .select import (
    BenchResult,
    append_results_tsv,
    decide_status,
    format_summary_block,
    load_champion,
    load_config,
    save_champion,
)
from .student_bundle import default_bundle_path, save_bundle


def _run_dir(config: dict[str, Any], tag: str | None) -> Path:
    root = Path(__file__).resolve().parents[1]
    base = root / str((config.get("paths") or {}).get("run_dir", "runs/autoresearch"))
    if tag:
        return base / tag
    return base


def run_baseline(
    *,
    config_path: Path | None = None,
    run_dir: Path | None = None,
    tag: str | None = None,
    skip_unit_tests: bool = False,
) -> BenchResult:
    cfg = load_config(config_path)
    rd = run_dir or _run_dir(cfg, tag)
    cand = default_champion_candidate()
    cand.description = "baseline champion"
    result = run_fly_bench(
        cand,
        config=cfg,
        run_dir=rd,
        experiment_id="baseline",
        skip_unit_tests=skip_unit_tests,
    )
    result.status = "keep" if result.gates_pass else "discard"
    append_results_tsv(rd, result)
    if result.gates_pass:
        save_champion(rd, result)
    print(format_summary_block(result))
    return result


def _promote_bundle(result: BenchResult, repo_root: Path) -> Path:
    work = repo_root / "runs/autoresearch/promote-tmp"
    work.mkdir(parents=True, exist_ok=True)
    genome = candidate_to_genome(result.candidate)
    student, aux = train_candidate(result.candidate, work_dir=work)
    dest = default_bundle_path(repo_root)
    save_bundle(
        genome,
        student,
        dest,
        metrics={"closed_loop": aux["closed_loop_reward"], "autoresearch": result.experiment_id},
    )
    shutil.rmtree(work, ignore_errors=True)
    return dest


def run_autoresearch_loop(
    *,
    config_path: Path | None = None,
    run_dir: Path | None = None,
    tag: str | None = None,
    max_experiments: int | None = None,
    patience: int | None = None,
    promote_bundle: bool = False,
    skip_unit_tests: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    stop = cfg.get("stop") or {}
    max_exp = int(max_experiments if max_experiments is not None else stop.get("max_experiments", 200))
    patience_n = int(patience if patience is not None else stop.get("patience", 50))
    max_hours = float(stop.get("max_hours", 8))
    rd = run_dir or _run_dir(cfg, tag)
    rd.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[1]
    champion = load_champion(rd)
    if champion is None:
        champion = run_baseline(config_path=config_path, run_dir=rd, skip_unit_tests=skip_unit_tests)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    no_keep = 0
    experiments = 0
    last_keep: BenchResult | None = champion if champion.status == "keep" else None
    summary: dict[str, Any] = {
        "run_dir": str(rd),
        "experiments": 0,
        "keeps": 0,
        "stopped_reason": None,
        "champion_experiment_id": champion.experiment_id,
        "champion_latency_score": champion.latency_score,
    }
    while experiments < max_exp and (time.time() - t0) < max_hours * 3600.0:
        experiments += 1
        cand = propose_candidate(champion.candidate, rng, cfg)
        result = run_fly_bench(
            cand,
            config=cfg,
            run_dir=rd,
            skip_unit_tests=skip_unit_tests,
        )
        result.status = decide_status(result, champion, cfg)
        append_results_tsv(rd, result)
        print(format_summary_block(result))
        if result.status == "keep":
            champion = result
            save_champion(rd, champion)
            no_keep = 0
            summary["keeps"] += 1
            summary["champion_experiment_id"] = champion.experiment_id
            summary["champion_latency_score"] = champion.latency_score
            last_keep = result
            if promote_bundle:
                path = _promote_bundle(result, repo_root)
                print(json.dumps({"promoted_bundle": str(path)}, indent=2))
        else:
            no_keep += 1
        if no_keep >= patience_n:
            summary["stopped_reason"] = "patience"
            break
        rebench_every = int(stop.get("rebench_champion_every", 10))
        if rebench_every > 0 and experiments % rebench_every == 0:
            check = run_fly_bench(
                champion.candidate,
                config=cfg,
                run_dir=rd,
                experiment_id=f"rebench-{experiments}",
                skip_unit_tests=True,
            )
            if not check.gates_pass:
                summary["stopped_reason"] = "champion_regression"
                break
    else:
        if experiments >= max_exp:
            summary["stopped_reason"] = "max_experiments"
        elif (time.time() - t0) >= max_hours * 3600.0:
            summary["stopped_reason"] = "max_hours"
    summary["experiments"] = experiments
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    (rd / "session_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
