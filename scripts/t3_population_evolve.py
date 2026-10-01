#!/usr/bin/env python3
"""P0 population evolution on CPU: filter on val, frozen serial judge before any keep.

1. Filter  (cpu_population.evolve): jax-cpu float32 trainer, population P, G generations,
           objectives = val per-step accuracy (max) and plastic params (min). Val ONLY.
2. Judge   the val Pareto front (+ top by val) is re-trained serially by the unchanged
           frozen judge `bench.run_fly_bench` (fit_student float64, repo gates: closed-loop
           24 seeds == 1.0, confirm/val >= 0.95, ood >= 0.85, test_gym.py green).
           We also check population-vs-judge parity on the judge's own bundle.
3. Held-out  on the judge's weights only: confirm/ood per-step accuracy and a 400-seed gym
           battery (seeds 10_000..10_399) that nothing in selection touched.
4. Keep    gates pass AND parity >= 0.999 decision agreement AND the candidate Pareto-
           dominates the incumbent (local-plasticity-000) on (held-out CL, -params).

--raw evolves without the hand-written delayed-cue protocol: the plastic readout alone must
carry the delayed cue. The repo judge always applies the protocol, so for --raw we report
its gates AND the raw held-out metrics on the judge-retrained weights.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from evolution_lab import cpu_population as cp  # noqa: E402
from evolution_lab.bench import run_fly_bench  # noqa: E402
from evolution_lab.dagger import mean_closed_loop_reward  # noqa: E402
from evolution_lab.engine import seed_genomes  # noqa: E402
from evolution_lab.gym import make_env  # noqa: E402
from evolution_lab.models import FittedStudent, _apply_delayed_cue_protocol, _kc_codes, _pn_features  # noqa: E402
from evolution_lab.select import FlyCandidate, load_config  # noqa: E402
from evolution_lab.splits import load_splits  # noqa: E402

HELDOUT = list(range(10_000, 10_400))


def candidate_for(g: cp.Gene) -> FlyCandidate:
    base = next(x for x in seed_genomes() if x.architecture.family == "local_plasticity")
    genome = replace(
        base,
        id=f"t3-lp-h{g.n_kc}-k{g.k}-s{g.seed}",
        architecture=replace(base.architecture, hidden=g.hidden),
        training=replace(base.training, seed=g.seed),
    )
    return FlyCandidate(
        genome_id=genome.id,
        genome=genome.to_dict(),
        plasticity_lr=g.lr,
        plasticity_epochs=g.epochs,
        k_winners=g.k,
        description="t3 cpu population survivor",
    )


def student_from_bundle(path: Path, *, protocol: bool) -> FittedStudent:
    b = np.load(path, allow_pickle=True)
    W, W_pn = b["W_kc_mbon"], b["W_pn_kc"]
    k = int(np.asarray(b["k_winners"]).reshape(-1)[0])

    def predict(eps):
        raw = (_kc_codes(_pn_features(eps), W_pn, k) @ W).argmax(1)
        return _apply_delayed_cue_protocol(eps, raw) if protocol else raw

    return FittedStudent("local_plasticity", int(W.size), predict, {"W": W, "W_pn": W_pn, "k": k})


def lat_us(student, eps, reps=300):
    one = [eps[0]]
    student.predict_fn(one)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        student.predict_fn(one)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1e6)


def heldout_metrics(student, data, env, pref):
    out = {}
    for s in ("val", "confirm", "ood"):
        eps, y = pref[s]
        out[f"{s}_step"] = float((student.predict_fn(eps) == y).mean())
        last = list(getattr(data, s))
        out[f"{s}_last"] = float((student.predict_fn(last) == np.stack([e.labels[-1] for e in last])).mean())
    out["cl_repo24"] = float(mean_closed_loop_reward(env, student, list(range(24))))
    out["cl_heldout400"] = float(mean_closed_loop_reward(env, student, HELDOUT))
    out["lat_single_us_p50"] = lat_us(student, pref["confirm"][0])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=128)
    ap.add_argument("--generations", type=int, default=6)
    ap.add_argument("--judge-top", type=int, default=12)
    ap.add_argument("--raw", action="store_true", help="evolve without the delayed-cue protocol")
    ap.add_argument("--backend", default="jax", choices=["jax", "numpy"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    protocol = not a.raw
    tag = "raw" if a.raw else "protocol"

    data = load_splits()
    ts = cp.train_set(data.train)
    env = make_env("hermes_recovery", delayed_cue=True, history=8)
    pref = {s: cp.prefix_episodes(getattr(data, s)) for s in ("val", "confirm", "ood")}

    t0 = time.perf_counter()
    res = cp.evolve(ts, data.val, pop=a.pop, generations=a.generations, protocol=protocol,
                    backend=a.backend, seed=a.seed, seed_genes=(cp.Gene(),),
                    workers=a.workers, train_eps=data.train)
    filter_s = time.perf_counter() - t0
    scored = res["scored"]
    front = res["front"]
    by_val = sorted(scored, key=lambda g: (-scored[g]["val_step"], g.n_params))
    to_judge = list(dict.fromkeys([cp.Gene()] + front + by_val))[: a.judge_top]
    print(f"filter: {len(scored)} unique candidates in {filter_s:.1f}s; judging {len(to_judge)}", flush=True)

    # population weights for parity (retrain the judged set once more with the filter backend)
    st = cp.setup_population(to_judge, ts)
    Wpop = cp.train_jax(st, ts.y) if a.backend == "jax" else cp.train_numpy(st, ts.y)
    ppop = cp.predict_population(st, Wpop, pref["confirm"][0] + pref["val"][0], protocol=protocol)

    cfg = load_config()
    judge_dir = ROOT / "runs" / f"t3-judge-{tag}-s{a.seed}"
    rows = []
    for p, g in enumerate(to_judge):
        t1 = time.perf_counter()
        br = run_fly_bench(candidate_for(g), config=cfg, run_dir=judge_dir,
                           experiment_id=f"{tag}-{p:02d}-h{g.n_kc}-k{g.k}-s{g.seed}")
        judge_s = time.perf_counter() - t1
        bundle = judge_dir / "candidates" / br.experiment_id / "bundle" / "recovery_student.npz"
        judge_student = student_from_bundle(bundle, protocol=protocol)
        W_judge = judge_student.extras["W"]
        Wp = cp.dense_weights(st, Wpop, p)
        pj = judge_student.predict_fn(pref["confirm"][0] + pref["val"][0])
        parity = {
            "max_abs_dW": float(np.abs(W_judge - Wp).max()),
            "rel_dW": float(np.abs(W_judge - Wp).max() / max(1e-12, np.abs(W_judge).max())),
            "decision_agree": float((pj == ppop[p]).mean()),
        }
        held = heldout_metrics(judge_student, data, env, pref)
        m = br.metrics
        row = {
            "gene": g.to_dict(),
            "n_kc": g.n_kc,
            "k": g.k,
            "params": g.n_params,
            "filter_val_step": scored[g]["val_step"],
            "judge": {
                "gates_pass": br.gates_pass,
                "gate_reasons": br.gate_reasons,
                "closed_loop_reward": m.closed_loop_reward,
                "confirm_success": m.confirm_success,
                "val_success": m.val_success,
                "ood_success": m.ood_success,
                "advise_warm_p50_ms": m.advise_warm_p50_ms,
                "advise_warm_p99_ms": m.advise_warm_p99_ms,
                "latency_score": br.latency_score,
                "wall_s": judge_s,
            },
            "parity": parity,
            "heldout": held,
        }
        rows.append(row)
        print(json.dumps({"p": p, "gene": g.to_dict(), "val": round(scored[g]["val_step"], 4),
                          "gates": br.gates_pass, "agree": round(parity["decision_agree"], 4),
                          "cl400": round(held["cl_heldout400"], 4), "confirm_step": round(held["confirm_step"], 4),
                          "reasons": br.gate_reasons}), flush=True)

    inc = next(r for r in rows if r["gene"] == cp.Gene().to_dict())
    for r in rows:
        dominates = (
            r["heldout"]["cl_heldout400"] >= inc["heldout"]["cl_heldout400"]
            and r["params"] <= inc["params"]
            and (r["heldout"]["cl_heldout400"] > inc["heldout"]["cl_heldout400"] or r["params"] < inc["params"])
        )
        r["keep"] = bool(r["judge"]["gates_pass"] and r["parity"]["decision_agree"] >= 0.999 and dominates
                         and r is not inc)
        r["dominates_incumbent"] = bool(dominates)
    kept = [r for r in rows if r["keep"]]
    print(f"kept {len(kept)} / {len(rows)} (incumbent cl400={inc['heldout']['cl_heldout400']:.4f})")
    for r in kept:
        print("  keep", r["gene"], "params", r["params"], "cl400", round(r["heldout"]["cl_heldout400"], 4))

    if a.out:
        out = {
            "schema": "flyforge.t3-population-evolve.v1",
            "evidence_class": "exploratory_beta",
            "promotion_eligible": False,
            "mode": tag,
            "backend": f"{a.backend}-cpu float32 filter; judge = bench.run_fly_bench (fit_student float64)",
            "cpus": sorted(os.sched_getaffinity(0)),
            "pop": a.pop,
            "generations": a.generations,
            "seed": a.seed,
            "selection_split": "val (per-step prefixes) only",
            "heldout": "confirm, ood, gym seeds 10000..10399",
            "filter_s": filter_s,
            "n_unique_trained": len(scored),
            "history": res["history"],
            "judged": rows,
            "kept": [r["gene"] for r in kept],
            "joules": "unknown (no NVML/RAPL read on this host)",
        }
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
