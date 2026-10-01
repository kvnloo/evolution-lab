"""T3b: held-out evaluation on REAL Hermes recovery episodes.

Controls: gym rule (task.teacher_action on the real gym frame), majority, ridge, MLP,
incumbent local_plasticity MB (zero-shot from synthetic P0 and retrained on real train),
and population-evolved MB candidates (cpu_population trainer; selection on val ONLY;
frozen judge = unchanged float64 reference models._plasticity_train, parity-checked on val
before the single test read).

Reads private episodes/split from ~/.z0int/research/t3b; writes aggregate metrics only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from evolution_lab import cpu_population as cp  # noqa: E402
from evolution_lab.models import _kc_codes, _plasticity_train, _pn_features, _sparse_pn_kc  # noqa: E402
from evolution_lab.real_recovery import ACTIONS, MAPPINGS, history_episodes, label_of  # noqa: E402
from evolution_lab.task import N_ACTIONS, teacher_action  # noqa: E402

PRIV = Path.home() / ".z0int" / "research" / "t3b"
A = len(ACTIONS)


# ----------------------------------------------------------------------------- metrics


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    conf = probs.max(1)
    pred = probs.argmax(1)
    e = 0.0
    for b in range(bins):
        m = (conf > b / bins) & (conf <= (b + 1) / bins)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def metrics(pred: np.ndarray, y: np.ndarray, probs: np.ndarray | None = None) -> dict:
    out = {"n": int(len(y)), "acc": float((pred == y).mean()) if len(y) else None}
    rec = {}
    for a in range(A):
        m = y == a
        if m.any():
            rec[ACTIONS[a]] = round(float((pred[m] == a).mean()), 4)
    out["recall"] = rec
    out["balanced_acc"] = float(np.mean(list(rec.values()))) if rec else None
    out["pred_dist"] = {ACTIONS[k]: int(v) for k, v in sorted(Counter(pred.tolist()).items())}
    if probs is not None:
        out["ece"] = ece(probs, y)
        out["nll"] = float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-9, 1))))
    return out


def grouped_boot(pred: np.ndarray, base: np.ndarray, y: np.ndarray, groups: np.ndarray, n: int = 2000, seed: int = 0):
    """Lineage-grouped bootstrap CI of acc and of acc(pred) - acc(base)."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in ug}
    accs, diffs = [], []
    for _ in range(n):
        pick = rng.choice(ug, size=len(ug), replace=True)
        ii = np.concatenate([idx_by[g] for g in pick])
        accs.append((pred[ii] == y[ii]).mean())
        diffs.append((pred[ii] == y[ii]).mean() - (base[ii] == y[ii]).mean())
    q = lambda v: [round(float(np.quantile(v, 0.025)), 4), round(float(np.quantile(v, 0.975)), 4)]  # noqa: E731
    return {"acc_ci95": q(accs), "diff_vs_rule_ci95": q(diffs), "p_diff_le_0": float(np.mean(np.asarray(diffs) <= 0))}


def latency_us(fn, x_one, reps: int = 300) -> float:
    fn(x_one)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        fn(x_one)
        ts.append(time.perf_counter_ns() - t0)
    return float(np.median(ts) / 1000.0)


# ----------------------------------------------------------------------------- models


def softmax(s: np.ndarray) -> np.ndarray:
    s = s - s.max(1, keepdims=True)
    e = np.exp(np.clip(s, -30, 30))
    return e / e.sum(1, keepdims=True)


class MB:
    """local_plasticity MB trained by the frozen reference trainer (float64)."""

    def __init__(self, g: cp.Gene, eps_train, y_train):
        rng = np.random.default_rng(int(g.seed))
        n_pn = _pn_features(eps_train[:1]).shape[1]
        self.W_pn = _sparse_pn_kc(n_pn, g.n_kc, rng)
        self.k = g.k
        self.W = _plasticity_train(
            list(eps_train), y_train, W_pn_kc=self.W_pn, k_winners=g.k,
            W=np.zeros((g.n_kc, N_ACTIONS)), rng=rng, epochs=int(g.epochs), lr=float(g.lr),
        )
        self.params = int(self.W.size)

    def scores(self, X):
        return _kc_codes(X, self.W_pn, self.k) @ self.W


def fit_sklearn(kind: str, Xtr, ytr, Xva, yva, seed: int = 0):
    from sklearn.linear_model import RidgeClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(Xtr)
    Xt, Xv = sc.transform(Xtr), sc.transform(Xva)
    best = None
    grid = [0.1, 1.0, 10.0, 100.0] if kind == "ridge" else [1e-4, 1e-3, 1e-2, 1e-1]
    for a in grid:
        if kind == "ridge":
            m = RidgeClassifier(alpha=a).fit(Xt, ytr)
        else:
            m = MLPClassifier(hidden_layer_sizes=(64,), alpha=a, max_iter=600, random_state=seed).fit(Xt, ytr)
        v = (m.predict(Xv) == yva).mean()
        if best is None or v > best[0]:
            best = (v, a, m)
    _, a, m = best
    if kind == "ridge":
        params = int(m.coef_.size + m.intercept_.size)
    else:
        params = int(sum(w.size for w in m.coefs_) + sum(b.size for b in m.intercepts_))
    return sc, m, a, params


# ----------------------------------------------------------------------------- evolution


def evolve_real(Xtr, ytr, Xva, yva, *, pop: int, gens: int, seed: int, fitness: str, backend: str, log):
    rng = np.random.default_rng(seed)
    ts = cp.TrainSet(Xtr, ytr)
    genes = [cp.Gene()] + [cp.random_gene(rng) for _ in range(pop - 1)]
    scored: dict = {}
    n_trained = 0
    t_all = time.perf_counter()

    def fit_score(pv):
        if fitness == "acc":
            return (pv == yva[None, :]).mean(1)
        cls = [a for a in range(A) if (yva == a).any()]
        return np.mean([(pv[:, yva == a] == a).mean(1) for a in cls], axis=0)

    for gen in range(gens):
        todo = [g for g in dict.fromkeys(genes) if g not in scored]
        if todo:
            st = cp.setup_population(todo, ts)
            W = cp.train_jax(st, ts.y) if backend == "jax" else cp.train_numpy(st, ts.y)
            pv = np.empty((len(todo), Xva.shape[0]), dtype=np.int64)
            for p, g in enumerate(todo):
                i, v = cp.sparse_kc(Xva, st.W_pn[p], g.k)
                pv[p] = np.einsum("nk,nka->na", v, W[p][i].astype(np.float64)).argmax(1)
            f = fit_score(pv)
            for g, a, row in zip(todo, f, pv):
                scored[g] = {"val_fit": float(a), "params": g.n_params, "val_pred": row}
            n_trained += len(todo)
        uniq = list(dict.fromkeys(genes))
        pts = np.asarray([[scored[g]["val_fit"], -scored[g]["params"]] for g in uniq])
        rank = cp.nondominated_rank(pts)
        order = np.lexsort((pts[:, 1] * -1, -pts[:, 0], rank))
        parents = [uniq[i] for i in order[: max(2, pop // 2)]]
        best = max(uniq, key=lambda g: (scored[g]["val_fit"], -g.n_params))
        log(f"  gen {gen}: trained {len(todo)} best val_{fitness}={scored[best]['val_fit']:.4f} params={best.n_params}")
        genes = parents + [cp.mutate(parents[int(rng.integers(0, len(parents)))], rng) for _ in range(pop - len(parents))]
    return scored, n_trained, time.perf_counter() - t_all


# ----------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", default="loose", choices=list(MAPPINGS))
    ap.add_argument("--pop", type=int, default=256)
    ap.add_argument("--gens", type=int, default=8)
    ap.add_argument("--evo-seeds", default="0,1")
    ap.add_argument("--backend", default="jax")
    ap.add_argument("--judge-top", type=int, default=6)
    ap.add_argument("--no-evolve", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    log = lambda *a: print(*a, flush=True)  # noqa: E731

    eps = [json.loads(l) for l in (PRIV / "episodes.jsonl").read_text().splitlines() if l.strip()]
    split = json.loads((PRIV / "split_v1.json").read_text())["lineage_split"]
    sp = np.asarray([split[e["lineage"]] for e in eps])
    y_all = np.asarray([label_of(e, args.mapping) for e in eps])
    lin = np.asarray([e["lineage"] for e in eps])
    src = np.asarray([e["source"] for e in eps])
    fam = np.asarray([e["extra"]["family"] for e in eps])
    H = {"gym": history_episodes(eps, rich=False), "rich": history_episodes(eps, rich=True)}
    X = {k: _pn_features(v) for k, v in H.items()}
    last_gym = np.stack([h.frames[-1] for h in H["gym"]])

    res: dict = {"mapping": args.mapping, "models": {}, "evolution": {}}
    masks = {s: (sp == s) & (y_all >= 0) for s in ("train", "val", "test")}
    res["n_labeled"] = {s: int(m.sum()) for s, m in masks.items()}
    res["n_features"] = {k: int(v.shape[1]) for k, v in X.items()}
    log(f"mapping={args.mapping} labeled={res['n_labeled']} feats={res['n_features']}")
    tr, va, te = masks["train"], masks["val"], masks["test"]
    ytr, yva, yte = y_all[tr], y_all[va], y_all[te]
    if len(np.unique(ytr)) < 2 or te.sum() < 20:
        res["verdict_note"] = "too few labeled episodes / classes for a meaningful fit"
        log(res["verdict_note"])

    preds_test: dict[str, np.ndarray] = {}
    predictors: dict = {}

    def record(name, pv_val, pv_test, *, params, lat, probs_test=None, probs_val=None, extra=None, fn=None, fs=None):
        preds_test[name] = pv_test
        if fn is not None:
            predictors[name] = (fs, fn)
        res["models"][name] = {
            "params": params,
            "latency_us": round(lat, 2),
            "val": metrics(pv_val, yva, probs_val),
            "test": metrics(pv_test, yte, probs_test),
            **(extra or {}),
        }
        log(f"{name:28s} val {res['models'][name]['val']['acc']:.4f}  test {res['models'][name]['test']['acc']:.4f}  "
            f"bal {res['models'][name]['test']['balanced_acc']:.4f}  params {params}")

    # rule
    rule_fn = lambda F: np.asarray([teacher_action(f) for f in F])  # noqa: E731
    record("rule_gym", rule_fn(last_gym[va]), rule_fn(last_gym[te]), params=0, lat=latency_us(rule_fn, last_gym[te][:1]),
           fn=rule_fn, fs="last")
    # majority
    maj = Counter(ytr.tolist()).most_common(1)[0][0]
    record("majority", np.full(va.sum(), maj), np.full(te.sum(), maj), params=0, lat=0.0,
           fn=lambda Z: np.full(len(Z), maj), fs="last")

    # sklearn controls
    for kind in ("ridge", "mlp"):
        for fs in ("gym", "rich"):
            seeds = [0] if kind == "ridge" else [0, 1, 2, 3, 4]
            accs = []
            for s in seeds:
                sc, m, a, params = fit_sklearn(kind, X[fs][tr], ytr, X[fs][va], yva, seed=s)
                pvv, pvt = m.predict(sc.transform(X[fs][va])), m.predict(sc.transform(X[fs][te]))
                accs.append(float((pvt == yte).mean()))
                if s == 0:
                    pt = pv = None
                    if kind == "mlp":
                        full = lambda Z: np.pad(m.predict_proba(Z), ((0, 0), (0, 0)))  # noqa: E731
                        P = np.zeros((te.sum(), A)); P[:, m.classes_] = m.predict_proba(sc.transform(X[fs][te])); pt = P
                        Pv = np.zeros((va.sum(), A)); Pv[:, m.classes_] = m.predict_proba(sc.transform(X[fs][va])); pv = Pv
                        del full
                    fn = lambda Z, m=m, sc=sc: m.predict(sc.transform(Z))  # noqa: E731
                    record(f"{kind}_{fs}", pvv, pvt, params=params, lat=latency_us(fn, X[fs][te][:1]),
                           probs_test=pt, probs_val=pv, extra={"hp": a}, fn=fn, fs=fs)
            if len(seeds) > 1:
                res["models"][f"{kind}_{fs}"]["test_acc_seeds"] = [round(x, 4) for x in accs]

    # incumbent MB zero-shot: Gene() trained on synthetic P0 train (gym layout)
    from evolution_lab.splits import load_splits

    syn = load_splits()
    syn_train = syn["train"] if isinstance(syn, dict) else syn.train
    ts_syn = cp.train_set(syn_train)
    g0 = cp.Gene()
    step_eps, y_syn = cp.prefix_episodes(syn_train)
    mb_syn = MB(g0, step_eps, y_syn)
    fn = lambda Z: mb_syn.scores(Z).argmax(1)  # noqa: E731
    record("lp_incumbent_synthetic_zeroshot", fn(X["gym"][va]), fn(X["gym"][te]), params=mb_syn.params,
           lat=latency_us(fn, X["gym"][te][:1]), probs_test=softmax(mb_syn.scores(X["gym"][te])),
           probs_val=softmax(mb_syn.scores(X["gym"][va])), fn=fn, fs="gym")
    del ts_syn

    # incumbent MB retrained on real train (reference trainer)
    H_tr = {k: [h for h, m in zip(v, tr) if m] for k, v in H.items()}
    for fs in ("gym", "rich"):
        accs = []
        for s in range(5):
            mb = MB(cp.Gene(seed=s), H_tr[fs], ytr)
            accs.append(float((mb.scores(X[fs][te]).argmax(1) == yte).mean()))
            if s == 0:
                fn = lambda Z, mb=mb: mb.scores(Z).argmax(1)  # noqa: E731
                record(f"lp_incumbent_real_{fs}", fn(X[fs][va]), fn(X[fs][te]), params=mb.params,
                       lat=latency_us(fn, X[fs][te][:1]), probs_test=softmax(mb.scores(X[fs][te])),
                       probs_val=softmax(mb.scores(X[fs][va])), fn=fn, fs=fs)
        res["models"][f"lp_incumbent_real_{fs}"]["test_acc_seeds"] = [round(x, 4) for x in accs]

    # population evolution: val-only selection, frozen reference judge, single test read
    if not args.no_evolve:
        for fs in ("gym", "rich"):
            for fitness in ("acc", "bal"):
                for es in [int(s) for s in args.evo_seeds.split(",")]:
                    tag = f"evolved_{fs}_{fitness}_s{es}"
                    log(f"evolve {tag}")
                    scored, n_tr, secs = evolve_real(X[fs][tr], ytr, X[fs][va], yva, pop=args.pop, gens=args.gens,
                                                     seed=es, fitness=fitness, backend=args.backend, log=log)
                    ranked = sorted(scored, key=lambda g: (-scored[g]["val_fit"], g.n_params))[: args.judge_top]
                    judged = []
                    for g in ranked:
                        mb = MB(g, H_tr[fs], ytr)
                        jv = mb.scores(X[fs][va]).argmax(1)
                        parity = float((jv == scored[g]["val_pred"]).mean())
                        judged.append((g, mb, jv, parity))
                    # pre-registered pick: best JUDGE val fitness among parity >= 0.99, tie -> fewer params
                    def jfit(jv):
                        if fitness == "acc":
                            return float((jv == yva).mean())
                        return float(np.mean([(jv[yva == a] == a).mean() for a in range(A) if (yva == a).any()]))
                    ok = [j for j in judged if j[3] >= 0.99] or judged
                    g, mb, jv, parity = max(ok, key=lambda j: (jfit(j[2]), -j[0].n_params))
                    fn = lambda Z, mb=mb: mb.scores(Z).argmax(1)  # noqa: E731
                    record(tag, jv, fn(X[fs][te]), params=mb.params, lat=latency_us(fn, X[fs][te][:1]),
                           probs_test=softmax(mb.scores(X[fs][te])), probs_val=softmax(mb.scores(X[fs][va])),
                           extra={"gene": g.to_dict(), "judge_val_parity": parity}, fn=fn, fs=fs)
                    res["evolution"][tag] = {
                        "candidates_trained": n_tr, "seconds": round(secs, 1),
                        "cand_per_s": round(n_tr / max(secs, 1e-9), 2),
                        "judged": [{"gene": j[0].to_dict(), "filter_val": scored[j[0]]["val_fit"],
                                    "judge_val": jfit(j[2]), "parity": j[3]} for j in judged],
                    }

    # grouped bootstrap vs rule on test + slices
    g_te, s_te, f_te = lin[te], src[te], fam[te]
    base = preds_test["rule_gym"]
    dom = Counter(fam.tolist()).most_common(1)[0][0]
    for name, pv in preds_test.items():
        r = res["models"][name]
        r["test_boot"] = grouped_boot(pv, base, yte, g_te)
        r["test_slices"] = {
            "cron": metrics(pv[s_te == "cron"], yte[s_te == "cron"])["acc"] if (s_te == "cron").any() else None,
            "non_cron": metrics(pv[s_te != "cron"], yte[s_te != "cron"])["acc"] if (s_te != "cron").any() else None,
            "n_non_cron": int((s_te != "cron").sum()),
            "dominant_job": metrics(pv[f_te == dom], yte[f_te == dom])["acc"] if (f_te == dom).any() else None,
            "n_dominant_job": int((f_te == dom).sum()),
        }

    # logged-policy replay on ALL test episodes with a mapped logged action and an observed
    # outcome (good or bad): success rate of the logged outcome where policy == logged action.
    logged = np.asarray([
        (ACTIONS.index(MAPPINGS[args.mapping][e["raw_action"]]) if MAPPINGS[args.mapping].get(e["raw_action"]) else -1)
        for e in eps
    ])
    outc = np.asarray([
        (-1 if e["local_recovered"] is None else int(bool(e["local_recovered"]) and e["session_complete"] is not False))
        for e in eps
    ])
    ope_m = (sp == "test") & (logged >= 0) & (outc >= 0)
    feats = {"last": last_gym, **X}
    ope = {"n": int(ope_m.sum()), "logged_success_rate": float(outc[ope_m].mean()) if ope_m.any() else None}
    for name, (fs, fn) in predictors.items():
        pa = fn(feats[fs][ope_m])
        match = pa == logged[ope_m]
        ope[name] = {"n_match": int(match.sum()),
                     "success_when_match": float(outc[ope_m][match].mean()) if match.any() else None,
                     "success_when_mismatch": float(outc[ope_m][~match].mean()) if (~match).any() else None}
    res["test_logged_replay"] = ope
    log("logged replay:", json.dumps(ope))

    # sensitivity: cross-job split (train on all non-dominant-family labeled, test on dominant family)
    lab = y_all >= 0
    sens = {}
    trj, tej = lab & (fam != dom), lab & (fam == dom)
    if trj.sum() > 50 and tej.sum() > 50 and len(np.unique(y_all[trj])) > 1:
        yj, yt = y_all[trj], y_all[tej]
        sens["n_train"], sens["n_test"] = int(trj.sum()), int(tej.sum())
        sens["rule_gym"] = float((rule_fn(last_gym[tej]) == yt).mean())
        sens["majority_of_train"] = float((Counter(yj.tolist()).most_common(1)[0][0] == yt).mean())
        for kind in ("ridge", "mlp"):
            for fs in ("gym", "rich"):
                sc, m, _, _ = fit_sklearn(kind, X[fs][trj], yj, X[fs][trj], yj)
                sens[f"{kind}_{fs}"] = float((m.predict(sc.transform(X[fs][tej])) == yt).mean())
        Hj = {k: [h for h, m in zip(v, trj) if m] for k, v in H.items()}
        for fs in ("gym", "rich"):
            mb = MB(cp.Gene(), Hj[fs], yj)
            sens[f"lp_incumbent_real_{fs}"] = float((mb.scores(X[fs][tej]).argmax(1) == yt).mean())
        sens["test_label_dist"] = {ACTIONS[k]: int(v) for k, v in Counter(yt.tolist()).items()}
    res["sensitivity_cross_job"] = sens
    log("cross-job sensitivity:", json.dumps(sens))

    out = Path(args.out or ROOT / "results" / f"t3b-eval-{args.mapping}.json")
    out.write_text(json.dumps(res, indent=1, sort_keys=True, default=str) + "\n")
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
