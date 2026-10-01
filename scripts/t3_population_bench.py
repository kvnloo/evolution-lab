#!/usr/bin/env python3
"""Throughput of CPU population training vs population size (one backend per process).

Every candidate is the seed genome's shape (hidden=128 -> 640 plastic weights, k=13,
20 epochs, lr=0.35) with its own seed, trained on the full locked P0 train split
(192 episodes -> 1536 per-step prefixes). That makes one candidate exactly the work of
one serial `fit_student(local_plasticity)` call, so candidates/s is comparable.

Pin cores from outside (taskset -c 0 / 0-1 / 0-3) and set OMP/XLA threads to match.

  taskset -c 0-3 python scripts/t3_population_bench.py --backend jax32 --pops 1,16,256 \
      --label t4 --out runs/t3-pop-bench.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from evolution_lab import cpu_population as cp  # noqa: E402
from evolution_lab.engine import seed_genomes  # noqa: E402
from evolution_lab.models import fit_student  # noqa: E402
from evolution_lab.splits import load_splits  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["reference", "numpy64", "numpy32", "jax32", "jax64", "jax32_shard"])
    ap.add_argument("--pops", default="1,4,16,64,256")
    ap.add_argument("--label", default="")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    data = load_splits()
    ts = cp.train_set(data.train)
    base = next(g for g in seed_genomes() if g.architecture.family == "local_plasticity")
    for P in [int(x) for x in a.pops.split(",")]:
        genes = [cp.Gene(seed=1000 + i, epochs=a.epochs) for i in range(P)]
        t0 = time.perf_counter()
        if a.backend == "jax32_shard":
            # warm: one tiny call per worker compiles nothing reusable across processes, so the
            # timed number INCLUDES per-process setup + compile + train (end-to-end).
            accs, _, stats = cp.evaluate_sharded(genes, data.train, data.val, workers=a.workers, backend="jax")
            setup_s = max(s["setup_s"] for s in stats)
            compile_s = 0.0
            train_s = time.perf_counter() - t0
        elif a.backend == "reference":
            for g in genes:
                gen = replace(base, training=replace(base.training, seed=g.seed, plasticity_epochs=g.epochs))
                fit_student(gen, list(data.train))
            setup_s, train_s, compile_s = 0.0, time.perf_counter() - t0, 0.0
        else:
            st = cp.setup_population(genes, ts)
            setup_s = time.perf_counter() - t0
            compile_s = 0.0
            if a.backend.startswith("jax"):
                # warm/compile on the same shapes, then time a second call
                t1 = time.perf_counter()
                cp.train_jax(st, ts.y, x64=a.backend == "jax64")
                compile_s = time.perf_counter() - t1
            t1 = time.perf_counter()
            if a.backend == "numpy64":
                cp.train_numpy(st, ts.y)
            elif a.backend == "numpy32":
                cp.train_numpy(st, ts.y, dtype=np.float32)
            else:
                cp.train_jax(st, ts.y, x64=a.backend == "jax64")
            train_s = time.perf_counter() - t1
        row = {
            "backend": a.backend,
            "label": a.label,
            "cpus": sorted(os.sched_getaffinity(0)),
            "P": P,
            "epochs": a.epochs,
            "n_samples": int(ts.X.shape[0]),
            "setup_s": setup_s,
            "first_call_s": compile_s,
            "train_s": train_s,
            "cand_per_s": P / train_s,
            "cand_per_s_incl_setup": P / (train_s + setup_s),
            "maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        }
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        if a.out:
            with open(a.out, "a") as f:
                f.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
