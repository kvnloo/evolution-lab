"""T3b action-space v1: pre-registered comparison (docs/t3b-action-space-v1-prereg.md).

Reads private v1 episodes + frozen split from ~/.z0int/research/t3b; writes aggregate
metrics only. `--dry` scores val only (for debugging) and never touches test labels.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from evolution_lab.action_space_v1 import (  # noqa: E402
    A1,
    ACTIONS_V1,
    COST,
    label_v1,
    mean_cost,
    min_cost_decode,
    plasticity_train_k,
    rule_v1,
    v1_history,
)
from evolution_lab.cpu_population import Gene  # noqa: E402
from evolution_lab.models import _kc_codes, _pn_features, _sparse_pn_kc  # noqa: E402
from evolution_lab.task import Episode  # noqa: E402

PRIV = Path.home() / ".z0int" / "research" / "t3b"
TEMPS = (0.25, 0.5, 1.0, 2.0, 4.0)
N_BOOT = 2000


def softmax(s: np.ndarray) -> np.ndarray:
    s = s - s.max(1, keepdims=True)
    e = np.exp(np.clip(s, -30, 30))
    return e / e.sum(1, keepdims=True)


def fit_temp(S_val: np.ndarray, y_val: np.ndarray) -> float:
    best = None
    for T in TEMPS:
        P = softmax(S_val / T)
        nll = -np.mean(np.log(np.clip(P[np.arange(len(y_val)), y_val], 1e-9, 1)))
        if best is None or nll < best[0]:
            best = (nll, T)
    return best[1]


def macro_f1(y, p) -> float:
    f = []
    for a in sorted(set(y.tolist()) | set(p.tolist())):
        tp = ((p == a) & (y == a)).sum()
        pr = tp / max(1, (p == a).sum())
        rc = tp / max(1, (y == a).sum())
        f.append(0.0 if pr + rc == 0 else 2 * pr * rc / (pr + rc))
    return float(np.mean(f))


def metrics(y, p) -> dict:
    rec = {}
    for a in range(A1):
        m = y == a
        rec[ACTIONS_V1[a]] = round(float((p[m] == a).mean()), 4) if m.sum() >= 5 else ("n/a" if not m.any() else f"n/a(n={int(m.sum())})")
    num = [v for v in rec.values() if isinstance(v, float)]
    return {
        "n": int(len(y)),
        "cost": round(mean_cost(y, p), 4),
        "acc": round(float((p == y).mean()), 4),
        "macro_f1": round(macro_f1(y, p), 4),
        "balanced_acc": round(float(np.mean(num)), 4) if num else None,
        "recall": rec,
        "pred_dist": {ACTIONS_V1[k]: int(v) for k, v in sorted(Counter(p.tolist()).items())},
        "n_pred_abort": int((p == ACTIONS_V1.index("abort")).sum()),
    }


def boot(preds: dict, y, groups, refs: dict[str, str], pairs: list[tuple[str, str]], seed=0) -> dict:
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in ug}
    names = list(preds)
    cost_i = {n: COST[y, preds[n]] for n in names}
    err_i = {n: (preds[n] != y).astype(float) for n in names}
    draws = {n: [] for n in names}
    errd = {n: [] for n in names}
    for _ in range(N_BOOT):
        ii = np.concatenate([idx_by[g] for g in rng.choice(ug, size=len(ug), replace=True)])
        for n in names:
            draws[n].append(cost_i[n][ii].mean())
            errd[n].append(err_i[n][ii].mean())
    q = lambda v: [round(float(np.quantile(v, 0.025)), 4), round(float(np.quantile(v, 0.975)), 4)]  # noqa: E731
    out = {}
    for n in names:
        d = {"cost_ci95": q(draws[n]), "err_ci95": q(errd[n])}
        for tag, ref in refs.items():
            if ref in draws and ref != n:
                dc = np.asarray(draws[n]) - np.asarray(draws[ref])
                de = np.asarray(errd[n]) - np.asarray(errd[ref])
                d[f"dcost_vs_{tag}_ci95"] = q(dc)
                d[f"dcost_vs_{tag}_p_ge_0"] = round(float(np.mean(dc >= 0)), 4)
                d[f"derr_vs_{tag}_ci95"] = q(de)
        out[n] = d
    for a, b in pairs:
        if a in draws and b in draws:
            dc = np.asarray(draws[a]) - np.asarray(draws[b])
            out[f"{a}__minus__{b}"] = {"dcost_ci95": q(dc), "p_ge_0": round(float(np.mean(dc >= 0)), 4)}
    return out


class MB:
    """Incumbent local-plasticity mushroom body with 6 MBON outputs."""

    def __init__(self, g: Gene, Xtr, ytr):
        rng = np.random.default_rng(int(g.seed))
        self.W_pn = _sparse_pn_kc(Xtr.shape[1], g.n_kc, rng)
        self.k = g.k
        self.W = plasticity_train_k(_kc_codes(Xtr, self.W_pn, g.k), ytr, W=np.zeros((g.n_kc, A1)), rng=rng,
                                    epochs=int(g.epochs), lr=float(g.lr))
        self.params = int(self.W.size)

    def scores(self, X):
        return _kc_codes(X, self.W_pn, self.k) @ self.W


def run(eps, sp, lin, y_all, X, *, dry: bool, log) -> dict:
    from sklearn.linear_model import RidgeClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    m = {s: (sp == s) & (y_all >= 0) for s in ("train", "val", "test")}
    tr, va, te = m["train"], m["val"], m["test"]
    ytr, yva, yte = y_all[tr], y_all[va], y_all[te]
    res: dict = {"n": {s: int(v.sum()) for s, v in m.items()},
                 "label_dist": {s: {ACTIONS_V1[k]: int(v) for k, v in sorted(Counter(y_all[mm].tolist()).items())}
                                for s, mm in m.items()},
                 "models": {}}
    log(f"n={res['n']}")
    P_val: dict[str, np.ndarray] = {}
    P_te: dict[str, np.ndarray] = {}

    def rec(name, pv, pt, **extra):
        P_val[name] = pv
        P_te[name] = pt
        res["models"][name] = {"val": metrics(yva, pv), **({} if dry else {"test": metrics(yte, pt)}), **extra}
        r = res["models"][name]
        log(f"{name:26s} val cost {r['val']['cost']:.3f} acc {r['val']['acc']:.3f}"
            + ("" if dry else f" | test cost {r['test']['cost']:.3f} acc {r['test']['acc']:.3f} f1 {r['test']['macro_f1']:.3f}"))

    def rec_scores(name, Sv, St, **extra):
        rec(f"{name}__argmax", Sv.argmax(1), St.argmax(1), **extra)
        T = fit_temp(Sv, yva)
        rec(f"{name}__cost", min_cost_decode(softmax(Sv / T)), min_cost_decode(softmax(St / T)), temp=T, **extra)

    nva, nte = int(va.sum()), int(te.sum())
    maj = Counter(ytr.tolist()).most_common(1)[0][0]
    rec("majority", np.full(nva, maj), np.full(nte, maj), action=ACTIONS_V1[maj])
    bc = int(np.argmin([COST[ytr, a].mean() for a in range(A1)]))
    rec("best_constant", np.full(nva, bc), np.full(nte, bc), action=ACTIONS_V1[bc])
    rule = np.asarray([rule_v1(e) for e in eps])
    rec("rule_v1", rule[va], rule[te])

    for fs in ("gym", "rich"):
        sc = StandardScaler().fit(X[fs][tr])
        Xt, Xv, Xe = sc.transform(X[fs][tr]), sc.transform(X[fs][va]), sc.transform(X[fs][te])
        best = None
        for a in (0.1, 1.0, 10.0, 100.0):
            mdl = RidgeClassifier(alpha=a).fit(Xt, ytr)
            c = mean_cost(yva, mdl.predict(Xv))
            if best is None or c < best[0]:
                best = (c, a, mdl)
        _, a, mdl = best

        def full(S, mdl=mdl):
            out = np.full((S.shape[0], A1), -1e3)
            out[:, mdl.classes_] = S
            return out

        rec_scores(f"ridge_{fs}", full(mdl.decision_function(Xv)), full(mdl.decision_function(Xe)), alpha=a,
                   params=int(mdl.coef_.size + mdl.intercept_.size))
        seed_costs = []
        for s in range(5):
            best = None
            for a in (1e-4, 1e-3, 1e-2, 1e-1):
                mdl = MLPClassifier(hidden_layer_sizes=(64,), alpha=a, max_iter=600, random_state=s).fit(Xt, ytr)
                c = mean_cost(yva, mdl.predict(Xv))
                if best is None or c < best[0]:
                    best = (c, a, mdl)
            _, a, mdl = best

            def lp(Z, mdl=mdl):
                out = np.full((Z.shape[0], A1), -30.0)
                out[:, mdl.classes_] = np.log(np.clip(mdl.predict_proba(Z), 1e-12, 1))
                return out

            Sv, St = lp(Xv), lp(Xe)
            if s == 0:
                rec_scores(f"mlp_{fs}", Sv, St, alpha=a,
                           params=int(sum(w.size for w in mdl.coefs_) + sum(b.size for b in mdl.intercepts_)))
            if not dry:
                T = fit_temp(Sv, yva)
                seed_costs.append(round(mean_cost(yte, min_cost_decode(softmax(St / T))), 4))
        if not dry:
            res["models"][f"mlp_{fs}__cost"]["test_cost_seeds"] = seed_costs

        seed_costs = []
        for s in range(5):
            t0 = time.perf_counter()
            mb = MB(Gene(seed=s), X[fs][tr], ytr)
            secs = time.perf_counter() - t0
            Sv, St = mb.scores(X[fs][va]), mb.scores(X[fs][te])
            if s == 0:
                rec_scores(f"mb_{fs}", Sv, St, params=mb.params, gene=Gene(seed=0).to_dict(), train_s=round(secs, 2))
            if not dry:
                T = fit_temp(Sv, yva)
                seed_costs.append(round(mean_cost(yte, min_cost_decode(softmax(St / T))), 4))
        if not dry:
            res["models"][f"mb_{fs}__cost"]["test_cost_seeds"] = seed_costs

        best = None
        for hid in (128, 512):
            for lr in (0.1, 0.35):
                for ep in (10, 20):
                    g = Gene(seed=0, hidden=hid, lr=lr, epochs=ep)
                    mb = MB(g, X[fs][tr], ytr)
                    c = mean_cost(yva, mb.scores(X[fs][va]).argmax(1))
                    if best is None or c < best[0]:
                        best = (c, g, mb)
        _, g, mb = best
        rec_scores(f"mbtuned_{fs}", mb.scores(X[fs][va]), mb.scores(X[fs][te]), params=mb.params, gene=g.to_dict())

    if not dry:
        refs = {"best_constant": "best_constant", "rule": "rule_v1", "majority": "majority"}
        pairs = [("mb_rich__cost", "mlp_rich__cost"), ("mb_rich__cost", "ridge_rich__cost"),
                 ("mbtuned_rich__cost", "mlp_rich__cost"), ("mb_rich__cost", "best_constant")]
        res["bootstrap"] = boot(P_te, yte, lin[te], refs, pairs)
        pb = res["bootstrap"]["mb_rich__cost__minus__best_constant"]
        res["primary_endpoint"] = {
            "model": "mb_rich__cost", "comparator": "best_constant",
            "test_cost_mb": res["models"]["mb_rich__cost"]["test"]["cost"],
            "test_cost_const": res["models"]["best_constant"]["test"]["cost"],
            "dcost_ci95": pb["dcost_ci95"],
            "verdict": "MB beats trivial controller" if pb["dcost_ci95"][1] < 0 else "null (CI includes 0 or favours constant)",
        }
        mm = res["bootstrap"]["mb_rich__cost__minus__mlp_rich__cost"]
        res["secondary_mb_vs_mlp"] = {"dcost_ci95": mm["dcost_ci95"],
                                      "competitive (upper < 0.05)": bool(mm["dcost_ci95"][1] < 0.05)}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="val only; test labels untouched")
    ap.add_argument("--out", default=str(ROOT / "results" / "t3b-v1as-eval.json"))
    args = ap.parse_args()
    log = lambda *a: print(*a, flush=True)  # noqa: E731
    t0 = time.time()
    split = json.loads((PRIV / "split_v1.json").read_text())["lineage_split"]
    eps = [json.loads(l) for l in (PRIV / "episodes_v1as.jsonl").read_text().splitlines() if l.strip()]
    eps = [e for e in eps if e["lineage"] in split]
    sp = np.asarray([split[e["lineage"]] for e in eps])
    lin = np.asarray([e["lineage"] for e in eps])
    H = v1_history(eps)
    X = {"rich": _pn_features(H), "gym": _pn_features([Episode(h.frames[:, :16], h.labels, h.env) for h in H])}
    out = {"prereg": "docs/t3b-action-space-v1-prereg.md", "actions": list(ACTIONS_V1), "cost_matrix": COST.tolist(),
           "n_features": {k: int(v.shape[1]) for k, v in X.items()}, "dry": args.dry}
    for tag, filtered in (("primary_filtered", True), ("sensitivity_behavioral", False)):
        log(f"=== {tag}")
        y = np.asarray([label_v1(e, filtered=filtered) for e in eps])
        out[tag] = run(eps, sp, lin, y, X, dry=args.dry, log=log)
    out["seconds"] = round(time.time() - t0, 1)
    if args.dry:
        log(json.dumps({k: {m: v["val"]["cost"] for m, v in out[k]["models"].items()}
                        for k in ("primary_filtered", "sensitivity_behavioral")}, indent=1))
        return
    Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
    log(json.dumps(out["primary_filtered"]["primary_endpoint"], indent=1))
    log(json.dumps(out["primary_filtered"]["secondary_mb_vs_mlp"], indent=1))
    log(f"wrote {args.out} in {out['seconds']}s")


if __name__ == "__main__":
    main()
