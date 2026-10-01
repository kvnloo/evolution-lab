#!/usr/bin/env python3
"""T3 reproduction: Track A recovery fly + mushroom-body learner vs mandatory controls.

Everything is fitted on the locked P0 train split (data/p0, seed 20260912) and scored on
the locked val / confirm / ood splits, plus a held-out closed-loop gym battery on fresh
seeds that no split used. Nothing is re-sampled.

Three metrics, never merged:
  last_step   L1 success_rate: accuracy of the final-step action (what the archive logs)
  per_step    accuracy over all 8 prefixes of every episode (prefixes built exactly like
              closed-loop predict: episode_from_prefix, first-frame padding)
  closed_loop mean per-step reward in the hermes_recovery gym (24 repo seeds and a
              disjoint 400-seed battery, seeds 10_000..10_399)

Controls (mandatory):
  rule_expert deterministic gym expert: teacher_action(last) + escalate at final step if
              the t=0 cue was seen (0 params) -- this IS the label generator
  rule        repo `rule` family (escalates whenever the cue was seen: exact on last-step,
              wrong at intermediate prefixes)
  majority    most frequent train label
  ridge_last  repo direct_input (ridge on last frame)
  ridge_pn    ridge on the SAME PN features the mushroom body sees, per-step prefixes
  ridge_pn_proto  ridge_pn + the same hand-written delayed-cue protocol lp gets
  mlp_pn      sklearn MLP (trained, 64 hidden) on the same PN features, per-step prefixes
  mlp / gru / fixed_reservoir / rewired_reservoir  repo Track A families (fit_student)

Mushroom body:
  lp          repo local_plasticity (frozen PN->KC, plastic KC->MBON, delayed-cue protocol)
  lp_raw      same fitted weights WITHOUT the hand-written delayed-cue protocol (ablation)
  ridge_kc    ridge readout on the same frozen KC codes (closed form instead of plasticity)
  mb_bundle   committed data/p0/recovery_student.npz, evaluated per prefix

Stochastic learners run over --seeds seeds; we report mean, min and max.

Usage: python scripts/t3_control_repro.py --seeds 5 --json results/t3-control-repro.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evolution_lab.dagger import mean_closed_loop_reward  # noqa: E402
from evolution_lab.engine import seed_genomes  # noqa: E402
from evolution_lab.gym import make_env  # noqa: E402
from evolution_lab.models import (  # noqa: E402
    FittedStudent,
    _apply_delayed_cue_protocol,
    _kc_codes,
    _local_plasticity_step_samples,
    _pn_features,
    fit_student,
    ridge_fit,
)
from evolution_lab.splits import load_splits  # noqa: E402
from evolution_lab.task import ACTIONS, N_ACTIONS, episode_from_prefix  # noqa: E402

HISTORY = 8
REPO_CL_SEEDS = list(range(24))
HELDOUT_CL_SEEDS = list(range(10_000, 10_400))


def genome_for(family: str):
    return next(g for g in seed_genomes() if g.architecture.family == family)


def prefix_set(episodes):
    eps, ys = [], []
    for ep in episodes:
        for t in range(ep.frames.shape[0]):
            eps.append(episode_from_prefix(ep.frames[: t + 1], history=HISTORY, env=ep.env))
            ys.append(int(ep.labels[t]))
    return eps, np.asarray(ys, dtype=np.int64)


def last_labels(episodes):
    return np.stack([ep.labels[-1] for ep in episodes])


def l2_select(X, y, Xv, yv, grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)):
    best = None
    for l2 in grid:
        W = ridge_fit(X, y, N_ACTIONS, l2)
        acc = float(((Xv @ W).argmax(1) == yv).mean())
        if best is None or acc > best[0] + 1e-12:
            best = (acc, l2, W)
    return best[1], best[2]


def with_bias(X):
    return np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)


# ----------------------------------------------------------------------------- builders


def build_rule(data, seed):
    return fit_student(genome_for("rule"), list(data.train))


def build_rule_expert(data, seed):
    """The gym expert as a deterministic rule: teacher_action on the last frame, plus
    'escalate at the final step if the t=0 cue was seen' (elapsed feature == 1). 0 params."""
    from evolution_lab.task import teacher_action

    def predict(eps):
        out = []
        for ep in eps:
            last = ep.frames[-1]
            if float(ep.frames[:, 9].max()) > 0.5 and float(last[3]) > 0.99:
                out.append(ACTIONS.index("escalate"))
            else:
                out.append(teacher_action(last))
        return np.asarray(out, dtype=np.int64)

    return FittedStudent("rule_expert", 0, predict, {})


def build_majority(data, seed):
    _, y = _local_plasticity_step_samples(list(data.train), history=HISTORY)
    top = int(np.bincount(y, minlength=N_ACTIONS).argmax())
    return FittedStudent("majority", 0, lambda eps: np.full(len(eps), top, dtype=np.int64), {})


def build_repo_family(family):
    def build(data, seed):
        g = genome_for(family)
        g = replace(g, training=replace(g.training, seed=seed))
        return fit_student(g, list(data.train))

    return build


def _pn_prefix_train(data):
    step_eps, y = _local_plasticity_step_samples(list(data.train), history=HISTORY)
    Xv_eps, yv = prefix_set(data.val)
    return _pn_features(step_eps), y, _pn_features(Xv_eps), yv


def build_ridge_pn(data, seed):
    X, y, Xv, yv = _pn_prefix_train(data)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    l2, W = l2_select(with_bias((X - mu) / sd), y, with_bias((Xv - mu) / sd), yv)

    def predict(eps):
        return (with_bias((_pn_features(eps) - mu) / sd) @ W).argmax(1)

    return FittedStudent("ridge_pn", int(W.size), predict, {"l2": l2})


def build_ridge_pn_proto(data, seed):
    """ridge_pn plus the same hand-written delayed-cue protocol the mushroom body gets."""
    base = build_ridge_pn(data, seed)

    def predict(eps):
        return _apply_delayed_cue_protocol(eps, base.predict_fn(eps))

    return FittedStudent("ridge_pn_proto", base.n_params, predict, base.extras)


def build_mlp_pn(data, seed):
    from sklearn.neural_network import MLPClassifier

    X, y, _, _ = _pn_prefix_train(data)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=3000, random_state=seed, alpha=1e-4)
    clf.fit((X - mu) / sd, y)
    n_params = sum(int(w.size) for w in clf.coefs_) + sum(int(b.size) for b in clf.intercepts_)

    def predict(eps):
        return clf.predict((_pn_features(eps) - mu) / sd).astype(np.int64)

    return FittedStudent("mlp_pn", n_params, predict, {})


def build_lp(data, seed):
    g = genome_for("local_plasticity")
    g = replace(g, training=replace(g.training, seed=seed))
    return fit_student(g, list(data.train))


def lp_raw_from(student):
    ex = student.extras
    W_pn, k, W = ex["W_pn_kc"], ex["k_winners"], ex["W_kc_mbon"]

    def predict(eps):
        return (_kc_codes(_pn_features(eps), W_pn, k) @ W).argmax(1)

    return FittedStudent("lp_raw", int(W.size), predict, {})


def ridge_kc_from(student, data):
    ex = student.extras
    W_pn, k = ex["W_pn_kc"], ex["k_winners"]
    step_eps, y = _local_plasticity_step_samples(list(data.train), history=HISTORY)
    H = _kc_codes(_pn_features(step_eps), W_pn, k)
    v_eps, yv = prefix_set(data.val)
    Hv = _kc_codes(_pn_features(v_eps), W_pn, k)
    l2, W = l2_select(H, y, Hv, yv)

    def predict(eps):
        return (_kc_codes(_pn_features(eps), W_pn, k) @ W).argmax(1)

    return FittedStudent("ridge_kc", int(W.size), predict, {"l2": l2})


def build_bundle(data, seed):
    b = np.load(ROOT / "data" / "p0" / "recovery_student.npz", allow_pickle=True)
    W, W_pn = b["W_kc_mbon"], b["W_pn_kc"]
    k = int(np.asarray(b["k_winners"]).reshape(-1)[0])

    def predict(eps):
        raw = (_kc_codes(_pn_features(eps), W_pn, k) @ W).argmax(1)
        return _apply_delayed_cue_protocol(eps, raw)

    return FittedStudent("mb_bundle", int(W.size), predict, {})


# ----------------------------------------------------------------------------- scoring


def latency_us(student, episodes, reps=200):
    one = [episodes[0]]
    student.predict_fn(one)
    t0 = time.perf_counter()
    for _ in range(reps):
        student.predict_fn(one)
    single = (time.perf_counter() - t0) / reps * 1e6
    t0 = time.perf_counter()
    for _ in range(5):
        student.predict_fn(episodes)
    batched = (time.perf_counter() - t0) / 5 / len(episodes) * 1e6
    return single, batched


def score(student, data, env, prefix_cache):
    out = {}
    for split in ("val", "confirm", "ood"):
        eps = list(getattr(data, split))
        out[f"{split}_last"] = float((student.predict_fn(eps) == last_labels(eps)).mean())
        p_eps, p_y = prefix_cache[split]
        out[f"{split}_step"] = float((student.predict_fn(p_eps) == p_y).mean())
    out["cl_repo24"] = float(mean_closed_loop_reward(env, student, REPO_CL_SEEDS))
    out["cl_heldout400"] = float(mean_closed_loop_reward(env, student, HELDOUT_CL_SEEDS))
    single, batched = latency_us(student, prefix_cache["confirm"][0])
    out["lat_single_us"] = single
    out["lat_batched_us"] = batched
    out["params"] = int(student.n_params)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    data = load_splits()
    env = make_env("hermes_recovery", delayed_cue=True, history=HISTORY)
    prefix_cache = {s: prefix_set(getattr(data, s)) for s in ("val", "confirm", "ood")}

    deterministic = {"rule_expert": build_rule_expert, "rule": build_rule, "majority": build_majority, "ridge_last": build_repo_family("direct_input"),
                     "ridge_pn": build_ridge_pn, "ridge_pn_proto": build_ridge_pn_proto, "mb_bundle": build_bundle}
    stochastic = {"mlp_pn": build_mlp_pn, "mlp_repo": build_repo_family("mlp"), "gru_repo": build_repo_family("gru"),
                  "fixed_reservoir": build_repo_family("fixed_reservoir"),
                  "rewired_reservoir": build_repo_family("rewired_reservoir")}

    rows: dict[str, list[dict]] = {}
    t_all = time.perf_counter()
    for name, build in deterministic.items():
        t0 = time.perf_counter()
        st = build(data, 0)
        r = score(st, data, env, prefix_cache)
        r["fit_s"] = time.perf_counter() - t0
        rows[name] = [r]
        print(f"{name:18} {json.dumps({k: round(v, 4) for k, v in r.items()})}", flush=True)
    for seed in range(a.seeds):
        for name, build in stochastic.items():
            t0 = time.perf_counter()
            st = build(data, seed)
            fit_s = time.perf_counter() - t0
            r = score(st, data, env, prefix_cache)
            r["fit_s"] = fit_s
            rows.setdefault(name, []).append(r)
        t0 = time.perf_counter()
        lp = build_lp(data, seed)
        fit_s = time.perf_counter() - t0
        for name, st in (("lp", lp), ("lp_raw", lp_raw_from(lp)), ("ridge_kc", ridge_kc_from(lp, data))):
            r = score(st, data, env, prefix_cache)
            r["fit_s"] = fit_s
            rows.setdefault(name, []).append(r)
        print(f"seed {seed} done  lp={json.dumps({k: round(v, 3) for k, v in rows['lp'][-1].items()})}", flush=True)

    summary = {}
    for name, rs in rows.items():
        keys = rs[0].keys()
        summary[name] = {
            k: {"mean": float(np.mean([r[k] for r in rs])), "min": float(np.min([r[k] for r in rs])),
                "max": float(np.max([r[k] for r in rs]))}
            for k in keys
        }
        summary[name]["n_seeds"] = len(rs)

    cols = ["val_last", "confirm_last", "ood_last", "val_step", "confirm_step", "cl_repo24", "cl_heldout400",
            "params", "lat_single_us"]
    print()
    print(f"{'model':18}" + "".join(f"{c:>14}" for c in cols))
    for name, s in summary.items():
        print(f"{name:18}" + "".join(f"{s[c]['mean']:>14.4f}" for c in cols))
    print(f"total {time.perf_counter() - t_all:.1f}s")

    if a.json:
        out = {
            "schema": "flyforge.t3-control-repro.v1",
            "evidence_class": "exploratory_beta",
            "promotion_eligible": False,
            "splits": "data/p0 (seed 20260912, bit-identical re-lock verified)",
            "host": "i7-4870HQ CPU only, threads capped at 4",
            "threads": os.environ.get("OMP_NUM_THREADS"),
            "seeds": a.seeds,
            "heldout_closed_loop_seeds": [HELDOUT_CL_SEEDS[0], HELDOUT_CL_SEEDS[-1]],
            "actions": list(ACTIONS),
            "summary": summary,
            "raw": rows,
        }
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
