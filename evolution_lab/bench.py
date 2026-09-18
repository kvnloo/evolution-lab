"""Fly recovery kernel benchmark — frozen judge for autoresearch."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .dagger import aggregate_dagger_episodes, mean_closed_loop_reward
from .engine import seed_genomes
from .gym import make_env
from .models import fit_student
from .schema import ExperimentGenome, genome_from_dict
from .select import (
    BenchMetrics,
    BenchResult,
    FlyCandidate,
    check_gates,
    latency_score,
    load_config,
)
from .splits import load_splits
from .student_bundle import save_bundle


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def _success(pred: np.ndarray, episodes) -> float:
    y = np.stack([ep.labels[-1] for ep in episodes])
    return float((pred == y).mean()) if len(y) else 0.0


def candidate_to_genome(candidate: FlyCandidate) -> ExperimentGenome:
    genome = genome_from_dict(candidate.genome)
    tr = replace(
        genome.training,
        plasticity_lr=float(candidate.plasticity_lr),
        plasticity_epochs=int(candidate.plasticity_epochs),
    )
    kw = int(getattr(candidate, "k_winners", 0) or 0)
    arch = genome.architecture if kw <= 0 else replace(genome.architecture, k_winners=kw)
    return replace(genome, training=tr, architecture=arch)


def default_champion_candidate() -> FlyCandidate:
    genome = next(g for g in seed_genomes() if g.architecture.family == "local_plasticity")
    return FlyCandidate(
        genome_id=genome.id,
        genome=genome.to_dict(),
        description="seed local_plasticity champion",
    )


def train_candidate(candidate: FlyCandidate, *, work_dir: Path) -> tuple[Any, dict[str, float]]:
    genome = candidate_to_genome(candidate)
    data = load_splits()
    env = make_env("hermes_recovery", delayed_cue=True, history=genome.architecture.history)
    t0 = time.perf_counter()
    train_pool = list(data.train)
    student = fit_student(genome, train_pool)
    for _ in range(int(candidate.dagger_rounds)):
        train_pool = train_pool + aggregate_dagger_episodes(env, student, list(range(32, 56)))
        student = fit_student(genome, train_pool, init_extras=student.extras)
    fit_ms = (time.perf_counter() - t0) * 1000.0
    closed_loop = mean_closed_loop_reward(env, student, list(range(24)))
    bundle_dir = work_dir / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    t1 = time.perf_counter()
    bundle_path = save_bundle(
        genome,
        student,
        bundle_dir / "recovery_student.npz",
        metrics={"closed_loop": closed_loop},
    )
    return student, {
        "fit_train_ms": fit_ms,
        "bundle_save_ms": (time.perf_counter() - t1) * 1000.0,
        "closed_loop_reward": closed_loop,
        "bundle_path": str(bundle_path),
    }


def score_accuracy(student, genome: ExperimentGenome) -> dict[str, float]:
    data = load_splits()
    return {
        "confirm_success": _success(student.predict_fn(data.confirm), data.confirm),
        "val_success": _success(student.predict_fn(data.val), data.val),
        "ood_success": _success(student.predict_fn(data.ood), data.ood),
        "violations": 0.0,
        "n_params": int(student.n_params),
    }


def bench_advise_timings(genome: ExperimentGenome, bundle_path: Path, *, warm_iters: int) -> dict[str, float]:
    from .advise import advise, clear_student_cache

    fields = {"sandbox_alive": 1.0, "transient": 1.0, "retry": 0.0, "budget": 0.8}
    prev = os.environ.get("FLYFORGE_RECOVERY_BUNDLE")
    os.environ["FLYFORGE_RECOVERY_BUNDLE"] = str(bundle_path)
    try:
        clear_student_cache()
        t0 = time.perf_counter()
        advise(fields, family="local_plasticity")
        cold_ms = (time.perf_counter() - t0) * 1000.0
        clear_student_cache()
        advise(fields, family="local_plasticity")
        hot: list[float] = []
        for _ in range(warm_iters):
            t2 = time.perf_counter()
            advise(fields, family="local_plasticity")
            hot.append((time.perf_counter() - t2) * 1000.0)
        return {
            "advise_cold_ms": cold_ms,
            "advise_warm_p50_ms": _percentile(hot, 50),
            "advise_warm_p99_ms": _percentile(hot, 99),
        }
    finally:
        clear_student_cache()
        if prev is None:
            os.environ.pop("FLYFORGE_RECOVERY_BUNDLE", None)
        else:
            os.environ["FLYFORGE_RECOVERY_BUNDLE"] = prev


def bench_closed_loop_ms(student, genome: ExperimentGenome, *, seeds: list[int]) -> float:
    env = make_env("hermes_recovery", delayed_cue=True, history=genome.architecture.history)
    t0 = time.perf_counter()
    mean_closed_loop_reward(env, student, seeds)
    return (time.perf_counter() - t0) * 1000.0


def bench_omp_bridge_ms(repo_root: Path) -> float:
    recovery_py = Path.home() / ".omp/agent/extensions/flyforge-recovery/recovery.py"
    if not recovery_py.is_file():
        return 0.0
    payload = json.dumps(
        {"action": "plan", "event": {"kind": "tool_error", "tool": "bash", "message": "connection reset by peer"}}
    )
    env = os.environ.copy()
    env.setdefault("FLYFORGE_EVOLUTION_LAB", str(repo_root))
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(recovery_py), payload],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        check=False,
    )
    return (time.perf_counter() - t0) * 1000.0 if proc.returncode == 0 else (time.perf_counter() - t0) * 1000.0


def run_fly_bench(
    candidate: FlyCandidate | None = None,
    *,
    config: dict[str, Any] | None = None,
    run_dir: Path | None = None,
    experiment_id: str | None = None,
    skip_unit_tests: bool = False,
) -> BenchResult:
    cfg = config or load_config()
    root = Path(__file__).resolve().parents[1]
    bench_cfg = cfg.get("bench") or {}
    work_root = run_dir or (root / str((cfg.get("paths") or {}).get("run_dir", "runs/autoresearch")))
    exp_id = experiment_id or f"fly-{uuid.uuid4().hex[:8]}"
    work_dir = work_root / "candidates" / exp_id
    work_dir.mkdir(parents=True, exist_ok=True)
    cand = candidate or default_champion_candidate()
    t_wall = time.perf_counter()
    unit_ok = True
    if not skip_unit_tests:
        proc = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_gym.py"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        unit_ok = proc.returncode == 0
    metrics = BenchMetrics(unit_tests_ok=unit_ok)
    try:
        student, aux = train_candidate(cand, work_dir=work_dir)
        genome = candidate_to_genome(cand)
        acc = score_accuracy(student, genome)
        metrics.closed_loop_reward = float(aux["closed_loop_reward"])
        metrics.confirm_success = acc["confirm_success"]
        metrics.val_success = acc["val_success"]
        metrics.ood_success = acc["ood_success"]
        metrics.violations = acc["violations"]
        metrics.n_params = acc["n_params"]
        metrics.fit_train_ms = float(aux["fit_train_ms"])
        metrics.bundle_save_ms = float(aux["bundle_save_ms"])
        timings = bench_advise_timings(
            genome,
            Path(aux["bundle_path"]),
            warm_iters=int(bench_cfg.get("advise_warm_iters", 200)),
        )
        metrics.advise_cold_ms = timings["advise_cold_ms"]
        metrics.advise_warm_p50_ms = timings["advise_warm_p50_ms"]
        metrics.advise_warm_p99_ms = timings["advise_warm_p99_ms"]
        seeds = list(range(int(bench_cfg.get("closed_loop_seeds", 24))))
        metrics.closed_loop_ms = bench_closed_loop_ms(student, genome, seeds=seeds)
        if bench_cfg.get("omp_smoke", True):
            metrics.omp_bridge_ms = bench_omp_bridge_ms(root)
    except Exception as exc:
        metrics.error = str(exc)
    gates_pass, gate_reasons = check_gates(metrics, cfg)
    result = BenchResult(
        experiment_id=exp_id,
        candidate=cand,
        metrics=metrics,
        latency_score=latency_score(metrics, cfg),
        gates_pass=gates_pass,
        gate_reasons=gate_reasons,
        wall_s=time.perf_counter() - t_wall,
    )
    (work_dir / "bench_result.json").write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    return result


def load_candidate(path: Path) -> FlyCandidate:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "genome" in data:
        return FlyCandidate.from_dict(data)
    return FlyCandidate.from_dict({"genome_id": data["id"], "genome": data})
