#!/usr/bin/env python3
"""Final held-out Pareto: evolved mushroom bodies vs mandatory controls on a FRESH battery.

Pre-registered picks (no held-out data used to choose them): from every
results/t3-evolve-*.json take judged[1], the first val-Pareto-front candidate after the
incumbent. The 400-seed battery was used by the keep rule, so it is NOT reused here:
this script scores on a new disjoint gym battery (seeds 20_000..21_999) plus the locked
confirm split. Every mushroom body is retrained by fit_student (float64, the judge path).

Axes: accuracy (fresh closed loop, higher better), single-decision latency (median of
2000 predict calls on one pinned core, lower better), params (lower better).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

import t3_control_repro as ctl  # noqa: E402
from evolution_lab import cpu_population as cp  # noqa: E402
from evolution_lab.dagger import mean_closed_loop_reward  # noqa: E402
from evolution_lab.engine import seed_genomes  # noqa: E402
from evolution_lab.gym import make_env  # noqa: E402
from evolution_lab.models import FittedStudent, fit_student  # noqa: E402
from evolution_lab.splits import load_splits  # noqa: E402

FRESH = list(range(20_000, 22_000))


def lp_student(g: cp.Gene, data, *, protocol: bool) -> FittedStudent:
    base = next(x for x in seed_genomes() if x.architecture.family == "local_plasticity")
    genome = replace(
        base,
        architecture=replace(base.architecture, hidden=g.hidden, k_winners=g.k),
        training=replace(base.training, seed=g.seed, plasticity_lr=g.lr, plasticity_epochs=g.epochs),
    )
    st = fit_student(genome, list(data.train))
    return st if protocol else ctl.lp_raw_from(st)


def latency(student, eps, reps=2000):
    one = [eps[0]]
    for _ in range(50):
        student.predict_fn(one)
    ts = np.empty(reps)
    for i in range(reps):
        t0 = time.perf_counter_ns()
        student.predict_fn(one)
        ts[i] = time.perf_counter_ns() - t0
    return float(np.median(ts) / 1e3), float(np.percentile(ts, 99) / 1e3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="results/t3-pareto.json")
    a = ap.parse_args()
    data = load_splits()
    env = make_env("hermes_recovery", delayed_cue=True, history=8)
    conf_eps, conf_y = cp.prefix_episodes(data.confirm)

    models: dict[str, FittedStudent] = {
        "rule_expert (0p)": ctl.build_rule_expert(data, 0),
        "rule_repo": ctl.build_rule(data, 0),
        "majority": ctl.build_majority(data, 0),
        "ridge_last (direct_input)": ctl.build_repo_family("direct_input")(data, 0),
        "ridge_pn": ctl.build_ridge_pn(data, 0),
        "mlp_pn seed0": ctl.build_mlp_pn(data, 0),
        "mlp_repo seed0": ctl.build_repo_family("mlp")(data, 0),
        "lp incumbent (proto)": lp_student(cp.Gene(), data, protocol=True),
        "lp incumbent (raw)": lp_student(cp.Gene(), data, protocol=False),
    }
    picks = {}
    for f in sorted(glob.glob(str(ROOT / "results" / "t3-evolve-*-s*.json"))):
        d = json.load(open(f))
        r = d["judged"][1]
        g = cp.Gene(**r["gene"])
        name = f"evolved {d['mode']} s{d['seed']} (kc{g.n_kc},k{g.k})"
        models[name] = lp_student(g, data, protocol=d["mode"] == "protocol")
        picks[name] = {"file": Path(f).name, "gene": r["gene"], "filter_val_step": r["filter_val_step"]}

    rows = {}
    for name, st in models.items():
        t0 = time.perf_counter()
        cl = float(mean_closed_loop_reward(env, st, FRESH))
        conf = float((st.predict_fn(conf_eps) == conf_y).mean())
        p50, p99 = latency(st, conf_eps)
        rows[name] = {"cl_fresh2000": cl, "confirm_step": conf, "lat_p50_us": p50, "lat_p99_us": p99,
                      "params": int(st.n_params), "eval_s": time.perf_counter() - t0}
        print(f"{name:40} cl2000={cl:.4f} conf_step={conf:.4f} p50={p50:7.1f}us p99={p99:7.1f}us params={st.n_params}",
              flush=True)

    names = list(rows)
    pts = np.array([[rows[n]["cl_fresh2000"], -rows[n]["lat_p50_us"], -rows[n]["params"]] for n in names])
    front = cp.pareto_front(pts)
    for n, f in zip(names, front):
        rows[n]["pareto"] = bool(f)
    print("pareto front (acc up, latency down, params down):", [n for n, f in zip(names, front) if f])
    # learned-only front (drop the oracle rule)
    learned = [i for i, n in enumerate(names) if not n.startswith("rule") and n != "majority"]
    lf = cp.pareto_front(pts[learned])
    print("learned-only front:", [names[i] for i, f in zip(learned, lf) if f])
    out = {
        "schema": "flyforge.t3-pareto.v1",
        "evidence_class": "exploratory_beta",
        "promotion_eligible": False,
        "fresh_battery": [FRESH[0], FRESH[-1]],
        "cpus": sorted(os.sched_getaffinity(0)),
        "picks_rule": "judged[1] of each evolve run (first val-front candidate); chosen without held-out data",
        "picks": picks,
        "rows": rows,
        "pareto": [n for n, f in zip(names, front) if f],
        "pareto_learned_only": [names[i] for i, f in zip(learned, lf) if f],
    }
    Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
