"""GPU population search for mushroom-body candidates.

GPU trains/filters many flies (shared frozen PN→KC). Frozen CPU judge
(run_fly_bench) serial-rebenches the top-K before any keep.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from .engine import seed_genomes
from .jax_mb import fit_from_numpy, numpy_perms, population_mbon, predict_numpy
from .models import _kc_codes, _local_plasticity_step_samples, _pn_features, _sparse_pn_kc
from .select import load_champion
from .splits import load_splits


def _gpu_snapshot() -> dict[str, Any]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
        name, mem_u, mem_t, util, pwr = [x.strip() for x in out.split(",")]
        return {
            "name": name,
            "vram_used_mb": float(mem_u),
            "vram_total_mb": float(mem_t),
            "util_pct": float(util),
            "power_w": float(pwr),
        }
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return {}


def _prepare(n_train: int = 96, n_kc: int = 96, k_winners: int = 10, seed: int = 0):
    genome = next(g for g in seed_genomes() if g.architecture.family == "local_plasticity")
    data = load_splits()
    step_eps, y = _local_plasticity_step_samples(data.train[:n_train], history=genome.architecture.history)
    X = _pn_features(step_eps)
    rng = np.random.default_rng(seed)
    W_pn = _sparse_pn_kc(X.shape[1], n_kc, rng)
    H = _kc_codes(X, W_pn, k_winners)
    Hc = _kc_codes(_pn_features(list(data.confirm)), W_pn, k_winners)
    yc = np.stack([ep.labels[-1] for ep in data.confirm])
    return genome, data, X, y, W_pn, H, Hc, yc, k_winners


def gpu_filter(
    H: np.ndarray,
    y: np.ndarray,
    Hc: np.ndarray,
    yc: np.ndarray,
    *,
    n_pop: int,
    epochs: int,
    lr: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Train P MBONs; return W[P,KC,A], confirm accuracy[P], wall_s."""
    bank = np.stack([numpy_perms(len(y), epochs, np.random.default_rng(seed + i)) for i in range(n_pop)])
    t0 = time.perf_counter()
    Ws = population_mbon(H, y, bank, lr)
    elapsed = time.perf_counter() - t0
    acc = np.stack([(Hc @ Ws[p]).argmax(axis=1) == yc for p in range(n_pop)]).mean(axis=1)
    return Ws, acc.astype(np.float64), elapsed


def cpu_rebench_top(
    genome,
    data,
    W_pn: np.ndarray,
    Ws: np.ndarray,
    acc: np.ndarray,
    *,
    k_winners: int,
    top_k: int,
) -> list[dict[str, Any]]:
    """Serial CPU metrics on GPU-trained weights. Does not retrain. Gates unchanged."""
    from .dagger import make_env, mean_closed_loop_reward
    from .models import FittedStudent, _local_plasticity_predict_fn

    order = np.argsort(-acc)
    env = make_env("hermes_recovery", delayed_cue=True, history=genome.architecture.history)
    rows = []
    for p in order[:top_k]:
        W = np.asarray(Ws[int(p)], dtype=np.float64)
        predict = _local_plasticity_predict_fn(genome, W_pn_kc=W_pn, k_winners=k_winners, W=W)
        student = FittedStudent("local_plasticity", n_params=int(W.size), predict_fn=predict, extras={})
        cl = float(mean_closed_loop_reward(env, student, list(range(24))))
        def _acc(split):
            pred = student.predict_fn(split)
            y = np.stack([ep.labels[-1] for ep in split])
            return float((pred == y).mean())
        confirm = _acc(data.confirm)
        val = _acc(data.val)
        ood = _acc(data.ood)
        gates = cl >= 1.0 - 1e-9 and confirm >= 0.95 and val >= 0.95 and ood >= 0.85
        rows.append(
            {
                "pop_index": int(p),
                "gpu_confirm_acc": float(acc[p]),
                "closed_loop_reward": cl,
                "confirm_success": confirm,
                "val_success": val,
                "ood_success": ood,
                "gates_pass": bool(gates),
            }
        )
    return rows


def run_gpu_evolve(
    *,
    n_pop: int = 256,
    top_k: int = 5,
    epochs: int = 8,
    n_train: int = 64,
    n_kc: int = 96,
    k_winners: int = 10,
    lr: float = 0.35,
    seed: int = 0,
) -> dict[str, Any]:
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.25")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    snap0 = _gpu_snapshot()
    genome, data, X, y, W_pn, H, Hc, yc, k = _prepare(n_train=n_train, n_kc=n_kc, k_winners=k_winners, seed=seed)
    Ws, acc, gpu_s = gpu_filter(H, y, Hc, yc, n_pop=n_pop, epochs=epochs, lr=lr, seed=seed + 1)
    snap1 = _gpu_snapshot()
    root = Path(__file__).resolve().parents[1]
    run_dir = root / "runs" / "gpu-evolve"
    run_dir.mkdir(parents=True, exist_ok=True)
    league = load_champion(root / "runs" / "autoresearch" / "league")
    champ = league.latency_score if league else None
    t1 = time.perf_counter()
    judged = cpu_rebench_top(genome, data, W_pn, Ws, acc, k_winners=k, top_k=top_k)
    judge_s = time.perf_counter() - t1
    best_gpu = float(acc.max()) if len(acc) else 0.0
    report = {
        "schema": "flyforge.gpu_evolve.v1",
        "n_pop": n_pop,
        "top_k": top_k,
        "epochs": epochs,
        "n_train_eps": n_train,
        "n_kc": n_kc,
        "k_winners": k_winners,
        "gpu_filter_s": gpu_s,
        "candidates_per_s": (n_pop / gpu_s) if gpu_s else None,
        "gpu_best_confirm_acc": best_gpu,
        "cpu_judge_s": judge_s,
        "league_latency_score": champ,
        "gpu_before": snap0,
        "gpu_after": snap1,
        "judged": judged,
        "note": "GPU trains/filters; CPU run_fly_bench is the frozen judge. Do not keep from GPU acc alone.",
    }
    energy = None
    if snap0.get("power_w") and gpu_s:
        # crude: average power * time / n_pop
        pwr = float(snap1.get("power_w") or snap0["power_w"])
        report["joules_estimate"] = pwr * gpu_s
        report["joules_per_candidate_est"] = pwr * gpu_s / n_pop
    (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
