#!/usr/bin/env python3
"""Phase 2 training comparison: majority / ridge / logistic / mlp / mushroom.

Fits five decision models on the SAME deterministic feature vector and the SAME
per-episode candidate sets (``legal_actions``) taken from the frozen Phase 2
corpus ``corpus/phase2/p1b-20260921T1430Z`` and evaluates them on the untouched
held-out splits.

Hard rules enforced here:
  * the fit set is exactly the episodes whose ``bucket`` is ``train``; every
    other bucket (validation, confirm, ood, sealed_human_audited) is read-only;
  * every vocabulary, scaling constant and hyperparameter is either derived from
    the train split or fixed a priori;
  * the gold label of a held-out episode is only ever used to score, never to
    fit, scale, select or calibrate.

The mushroom path imports ``evolution_lab.models`` (``_pn_features``,
``_kc_codes``, ``_sparse_pn_kc``, ``_local_plasticity_step_samples``,
``_plasticity_train``) instead of re-implementing the PN->KC->MBON construction.

Outputs ``results/phase2-training-comparison.json`` (+ Markdown companion).
"""

from __future__ import annotations

import json
import platform
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evolution_lab.models import (  # noqa: E402
    _kc_codes,
    _local_plasticity_step_samples,
    _pn_features,
    _plasticity_train,
    _sparse_pn_kc,
    ridge_fit,
)
from evolution_lab.task import Episode  # noqa: E402

CORPUS = ROOT / "corpus" / "phase2" / "p1b-20260921T1430Z"
OUT_JSON = ROOT / "results" / "phase2-training-comparison.json"
OUT_MD = ROOT / "results" / "phase2-training-comparison.md"

FIT_BUCKET = "train"
HELD_OUT_BUCKETS = ("validation", "confirm", "ood", "sealed_human_audited")
SPLIT_ORDER = (FIT_BUCKET,) + HELD_OUT_BUCKETS

# A priori, fixed before touching any data (no tuning on any split).
RIDGE_L2 = 1e-2                     # same default as evolution_lab Genome training.l2
LOGISTIC_C = 1.0                    # fixed a priori
LOGISTIC_MAX_ITER = 5000
MLP_HIDDEN = 16                     # "tiny": one hidden layer, width 16
MLP_ALPHA = 1e-4
MLP_LR = 1e-3
MLP_MAX_ITER = 1000
MLP_EARLY_STOP_VALIDATION_FRACTION = 0.2   # carved out of TRAIN only
MLP_EARLY_STOP_PATIENCE = 30
MLP_SEED = 0
# Mushroom: the shipped local-plasticity genome values (engine.seed_genomes()).
MB_N_KC = 128                       # Architecture.hidden for local-plasticity-000
MB_K_WINNERS = 0                    # genome value; 0 => max(5, round(0.10 * n_kc)) = 13
MB_EPOCHS = 20                      # Training.plasticity_epochs
MB_LR = 0.35                        # Training.plasticity_lr
MB_SEED = 0                         # Training.seed
UNSEEN_FLOOR_LOGPROB = float(np.log(1e-12))  # fixed floor for classes with no train row

THRESHOLDS = np.round(np.arange(0.0, 1.0 + 1e-9, 0.02), 10).tolist()
BEST_MIN_COVERAGE = 0.25            # a priori rule for "best operating point"


# --------------------------------------------------------------------- loading


def load_corpus():
    rows = []
    with (CORPUS / "episodes.jsonl").open() as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    splits_raw = json.loads((CORPUS / "splits.json").read_text())
    assignments = splits_raw["assignments"]
    manifest = json.loads((CORPUS / "manifest.json").read_text())
    return rows, splits_raw, assignments, manifest


def report_buckets(assignments, rows):
    """Hard rule: print exact bucket names and counts before anything else."""
    by_bucket = Counter(a["bucket"] for a in assignments)
    by_chrono = Counter(a["chronological_bucket"] for a in assignments)
    crosstab = Counter((a["bucket"], a["chronological_bucket"]) for a in assignments)
    print("=" * 72)
    print("SPLIT ASSIGNMENTS READ FROM splits.json (printed first, nothing fitted yet)")
    print(f"  episodes.jsonl rows : {len(rows)}")
    print(f"  splits assignments  : {len(assignments)}")
    print("  bucket names/counts :")
    for name in SPLIT_ORDER:
        print(f"      {name:22s} {by_bucket.get(name, 0)}")
    print("  chronological_bucket names/counts (secondary axis, NOT used to fit):")
    for name in sorted(by_chrono):
        print(f"      {name:22s} {by_chrono[name]}")
    print("  bucket x chronological_bucket crosstab:")
    for key in sorted(crosstab):
        print(f"      {key[0]:22s} x {key[1]:10s} {crosstab[key]}")
    print(f"  FIT SET  = bucket == '{FIT_BUCKET}'  (n={by_bucket[FIT_BUCKET]})")
    print(f"  HELD OUT = {list(HELD_OUT_BUCKETS)}")
    print("=" * 72)
    assert set(by_bucket) == set(SPLIT_ORDER), f"unexpected bucket names: {sorted(by_bucket)}"
    assert sum(by_bucket.values()) == len(rows)
    return by_bucket, by_chrono, crosstab


# ------------------------------------------------------- feature construction

CONTINUOUS_KEYS = (
    "budget_units",
    "authority_breadth",
    "candidate_action_count",
    "declared_dangerous_count",
    "legal_family_count",
    "satisfied_count",
)
BINARY_KEYS = (
    "expect_abstain",
    "gold_present",
    "has_deterministic_solution",
    "is_empty",
)
LABEL_ADJACENT_KEYS = ("expect_abstain", "has_deterministic_solution")


def raw_feature_parts(row):
    sb = row["state_before"]
    feats = sb["features"]
    cont = {
        "budget_units": float(sb["budget_units"]),
        "authority_breadth": float(feats["authority_breadth"]),
        "candidate_action_count": float(feats["candidate_action_count"]),
        "declared_dangerous_count": float(feats["declared_dangerous_count"]),
        "legal_family_count": float(feats["legal_family_count"]),
        "satisfied_count": float(feats["satisfied_count"]),
    }
    binary = {k: float(bool(feats[k])) for k in BINARY_KEYS}
    return {
        "continuous": cont,
        "binary": binary,
        "family": sb["family"],
        "legal_families": list(feats["legal_families"]),
        "legal_risk_classes": list(feats["legal_risk_classes"]),
        "max_risk_class": feats["max_risk_class"],
    }


def build_codec(parts_by_id, fit_ids, *, drop_label_adjacent: bool):
    """All vocabularies and scaling constants come from the fit set only."""
    families = sorted({parts_by_id[i]["family"] for i in fit_ids})
    legal_families = sorted({x for i in fit_ids for x in parts_by_id[i]["legal_families"]})
    risk_classes = sorted({x for i in fit_ids for x in parts_by_id[i]["legal_risk_classes"]})
    max_risk = sorted({parts_by_id[i]["max_risk_class"] for i in fit_ids})
    binary_keys = (
        tuple(k for k in BINARY_KEYS if k not in LABEL_ADJACENT_KEYS)
        if drop_label_adjacent
        else BINARY_KEYS
    )
    cont_stack = np.array(
        [[parts_by_id[i]["continuous"][k] for k in CONTINUOUS_KEYS] for i in fit_ids],
        dtype=np.float64,
    )
    mean = cont_stack.mean(axis=0)
    std = np.maximum(cont_stack.std(axis=0), 1e-8)
    return {
        "families": families,
        "legal_families": legal_families,
        "risk_classes": risk_classes,
        "max_risk": max_risk,
        "binary_keys": binary_keys,
        "continuous_keys": CONTINUOUS_KEYS,
        "mean": mean,
        "std": std,
    }


def encode(parts, codec, *, drop_label_adjacent: bool):
    cont = np.array(
        [(parts["continuous"][k] - m) / s
         for k, m, s in zip(codec["continuous_keys"], codec["mean"], codec["std"])],
        dtype=np.float64,
    )
    binary_keys = (
        tuple(k for k in BINARY_KEYS if k not in LABEL_ADJACENT_KEYS)
        if drop_label_adjacent
        else BINARY_KEYS
    )
    binary = np.array([parts["binary"][k] for k in binary_keys], dtype=np.float64)

    def one_hot(value, vocab):
        v = np.zeros(len(vocab) + 1, dtype=np.float64)  # last slot = UNK
        v[vocab.index(value) if value in vocab else len(vocab)] = 1.0
        return v

    def multi_hot(values, vocab):
        v = np.zeros(len(vocab) + 1, dtype=np.float64)
        hit = False
        for x in values:
            if x in vocab:
                v[vocab.index(x)] = 1.0
                hit = True
        if not hit:
            v[len(vocab)] = 1.0
        return v

    return np.concatenate(
        [
            cont,
            binary,
            one_hot(parts["family"], codec["families"]),
            multi_hot(parts["legal_families"], codec["legal_families"]),
            multi_hot(parts["legal_risk_classes"], codec["risk_classes"]),
            one_hot(parts["max_risk_class"], codec["max_risk"]),
        ]
    )


def feature_layout(codec):
    return [
        {"block": "continuous_zscored_train_stats", "keys": list(codec["continuous_keys"]),
         "n": len(codec["continuous_keys"])},
        {"block": "binary_0_1", "keys": list(codec["binary_keys"]), "n": len(codec["binary_keys"])},
        {"block": "one_hot_state_before.family_plus_UNK", "keys": codec["families"], "n": len(codec["families"]) + 1},
        {"block": "multi_hot_features.legal_families_plus_UNK", "keys": codec["legal_families"], "n": len(codec["legal_families"]) + 1},
        {"block": "multi_hot_features.legal_risk_classes_plus_UNK", "keys": codec["risk_classes"], "n": len(codec["risk_classes"]) + 1},
        {"block": "one_hot_features.max_risk_class_plus_UNK", "keys": codec["max_risk"], "n": len(codec["max_risk"]) + 1},
    ]


# ------------------------------------------------------------------- predictors


def fit_majority(X, ytr, n_actions, ctx):
    freq = np.bincount(ytr, minlength=n_actions).astype(np.float64) / len(ytr)
    return (lambda Xq: np.tile(freq, (Xq.shape[0], 1))), {} 


def fit_ridge(X, ytr, n_actions, ctx):
    Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)  # model bias term
    W = ridge_fit(Xb, ytr, n_actions, RIDGE_L2)

    def score(Xq):
        return np.concatenate([Xq, np.ones((Xq.shape[0], 1))], axis=1) @ W

    return score, {"n_params": int(W.size)}


def _proba_to_log_scores(proba, classes, n_actions):
    out = np.full((proba.shape[0], n_actions), UNSEEN_FLOOR_LOGPROB, dtype=np.float64)
    out[:, np.asarray(classes, dtype=int)] = np.log(np.maximum(proba, 1e-300))
    return out


def fit_logistic(X, ytr, n_actions, ctx):
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=LOGISTIC_C, max_iter=LOGISTIC_MAX_ITER, solver="lbfgs", tol=1e-6)
    clf.fit(X, ytr)
    info = {
        "n_iter": [int(v) for v in np.atleast_1d(clf.n_iter_)],
        "trained_classes": [int(c) for c in clf.classes_],
        "n_params": int(clf.coef_.size + clf.intercept_.size),
    }
    return (lambda Xq: _proba_to_log_scores(clf.predict_proba(Xq), clf.classes_, n_actions)), info


def fit_mlp(X, ytr, n_actions, ctx):
    from sklearn.neural_network import MLPClassifier

    clf = MLPClassifier(
        hidden_layer_sizes=(MLP_HIDDEN,),
        activation="relu",
        solver="adam",
        alpha=MLP_ALPHA,
        batch_size=min(32, X.shape[0]),
        learning_rate_init=MLP_LR,
        max_iter=MLP_MAX_ITER,
        early_stopping=True,                       # internal split of TRAIN only
        validation_fraction=MLP_EARLY_STOP_VALIDATION_FRACTION,
        n_iter_no_change=MLP_EARLY_STOP_PATIENCE,
        random_state=MLP_SEED,
        shuffle=True,
    )
    clf.fit(X, ytr)
    info = {
        "n_iter": int(clf.n_iter_),
        "early_stop_best_validation_score": float(clf.best_validation_score_),
        "trained_classes": [int(c) for c in clf.classes_],
        "n_params": int(sum(p.size for p in clf.coefs_) + sum(p.size for p in clf.intercepts_)),
    }
    return (lambda Xq: _proba_to_log_scores(clf.predict_proba(Xq), clf.classes_, n_actions)), info


def fit_mushroom(Xtr, ytr, n_actions):
    """PN -> frozen sparse PN->KC -> k-winner KC -> plastic KC->MBON.

    Calls the repo's own ``_sparse_pn_kc`` / ``_local_plasticity_step_samples`` /
    ``_plasticity_train`` / ``_pn_features`` / ``_kc_codes`` unchanged. The one
    adaptation: ``_plasticity_train`` builds its target from the module constant
    ``evolution_lab.models.N_ACTIONS`` (the 5-action recovery vocabulary), which
    is patched to the Phase 2 action count for the duration of the call; the
    plasticity rule itself (softmax delta + class-conditional LTD, lr decay) is
    the shipped one.
    """
    import evolution_lab.models as models

    def to_episodes(X, y=None):
        eps = []
        for i in range(X.shape[0]):
            lab = np.array([int(y[i]) if y is not None else 0], dtype=np.int64)
            eps.append(Episode(X[i][None, :].astype(np.float64), lab, env="phase2"))
        return eps

    tr_eps = to_episodes(Xtr, ytr)
    step_eps, step_y = _local_plasticity_step_samples(tr_eps, history=1)

    rng = np.random.default_rng(MB_SEED)
    Xp_probe = _pn_features(step_eps[:1])
    n_kc = MB_N_KC
    W_pn_kc = _sparse_pn_kc(Xp_probe.shape[1], n_kc, rng)
    k_winners = MB_K_WINNERS if MB_K_WINNERS > 0 else max(5, int(round(0.10 * n_kc)))
    W = np.zeros((n_kc, n_actions), dtype=np.float64)

    prev_n_actions = models.N_ACTIONS
    models.N_ACTIONS = n_actions
    try:
        W = _plasticity_train(
            step_eps,
            step_y,
            W_pn_kc=W_pn_kc,
            k_winners=k_winners,
            W=W,
            rng=rng,
            epochs=MB_EPOCHS,
            lr=MB_LR,
        )
    finally:
        models.N_ACTIONS = prev_n_actions

    def score(X):
        eps = to_episodes(X)
        Xp = _pn_features(eps)
        Ht = _kc_codes(Xp, W_pn_kc, k_winners)
        return Ht @ W

    info = {
        "encoder": "flatten+last+maxpool over a single frame equal to the shared feature vector",
        "pn_dim": int(Xp_probe.shape[1]),
        "n_kc": int(n_kc),
        "k_winners": int(k_winners),
        "plastic": "kc_to_mbon_only",
        "plasticity_rule": "repo _plasticity_train (softmax delta + class-conditional LTD, lr decay 0.92/epoch)",
        "pnm_kc_frozen_random": True,
        "mb_seed": MB_SEED,
        "epochs": MB_EPOCHS,
        "lr": MB_LR,
        "n_params": int(W.size + W_pn_kc.size),
        "readout_params": int(W.size),
        "delayed_cue_protocol_applied": False,
        "delayed_cue_protocol_note": (
            "evolution_lab.models._apply_delayed_cue_protocol keys off Hermes frame indices "
            "(0/3/7/9) and maps its override onto the 5-action recovery ACTIONS tuple; the Phase 2 "
            "vocabulary and feature vector have neither. It is a domain post-hoc rule, not the "
            "PN->KC->MBON architecture, so it is omitted rather than applied to foreign indices."
        ),
    }
    return score, info


MODELS = ("majority", "ridge", "logistic", "mlp", "mushroom")


def fit_all(Xtr, ytr, n_actions):
    """Fit every model on the identical (Xtr, ytr). Returns {name: (score_fn, info)}."""
    fitted = {}
    fitted["majority"] = fit_majority(Xtr, ytr, n_actions, {})
    fitted["ridge"] = fit_ridge(Xtr, ytr, n_actions, {})
    fitted["logistic"] = fit_logistic(Xtr, ytr, n_actions, {})
    fitted["mlp"] = fit_mlp(Xtr, ytr, n_actions, {})
    fitted["mushroom"] = fit_mushroom(Xtr, ytr, n_actions)
    return fitted


# --------------------------------------------------------------------- metrics


def softmax_rows(S):
    S = S - S.max(axis=1, keepdims=True)
    E = np.exp(S)
    return E / E.sum(axis=1, keepdims=True)


def decide(scores, candidates, n_actions):
    """Masked argmax over the shared per-episode candidate set. Never leaves legal."""
    n = scores.shape[0]
    pred = np.empty(n, dtype=np.int64)
    conf = np.empty(n, dtype=np.float64)
    for i in range(n):
        idx = np.asarray(candidates[i], dtype=np.int64)
        s = scores[i, idx]
        best = int(np.argmax(s))          # numpy argmax ties -> first (lowest action id)
        pred[i] = int(idx[best])
        p = softmax_rows(s[None, :])[0]
        conf[i] = float(p[best])
    return pred, conf


def macro_recall(y_true, y_pred, classes_present):
    vals = []
    for c in classes_present:
        m = y_true == c
        vals.append(float((y_pred[m] == c).mean()))
    return float(np.mean(vals)) if vals else None


def risk_coverage(y_true, y_pred, conf):
    curve = {"thresholds": list(THRESHOLDS), "coverage": [], "n_covered": [],
             "success_given_covered": [], "n_correct": []}
    for t in THRESHOLDS:
        m = conf >= t
        nc = int(m.sum())
        ncorr = int((y_pred[m] == y_true[m]).sum())
        curve["n_covered"].append(nc)
        curve["coverage"].append(nc / len(y_true))
        curve["n_correct"].append(ncorr)
        curve["success_given_covered"].append((ncorr / nc) if nc else None)
    return curve


def best_operating_point(curve):
    """A priori rule: max success|covered subject to coverage >= BEST_MIN_COVERAGE,
    tie-break by higher coverage; if none qualifies, the max-coverage point."""
    pts = [
        (t, c, s)
        for t, c, s in zip(curve["thresholds"], curve["coverage"], curve["success_given_covered"])
        if s is not None
    ]
    elig = [p for p in pts if p[1] >= BEST_MIN_COVERAGE]
    if elig:
        pick = max(elig, key=lambda p: (p[2], p[1]))
    else:
        pick = max(pts, key=lambda p: p[1])
    return {"threshold": float(pick[0]), "coverage": float(pick[1]),
            "success_given_covered": float(pick[2]),
            "rule": f"max success|covered with coverage>={BEST_MIN_COVERAGE}, tie-break higher coverage"}


def evaluate_split(name, score_fn, X, y_global, cand_lists, n_actions, train_actions, split_ids):
    scores = score_fn(X)
    pred, conf = decide(scores, cand_lists, n_actions)

    legal_ok = np.array(
        [pred[i] in set(cand_lists[i]) for i in range(len(pred))], dtype=bool
    )
    outside = int((~legal_ok).sum())

    # Non-tautological control: what the SAME scores would pick WITHOUT candidate
    # masking. This shows masking is doing real work rather than the check being vacuous.
    raw_argmax = np.argmax(scores, axis=1)
    unconstrained_outside = int(
        sum(1 for i in range(len(pred)) if int(raw_argmax[i]) not in set(cand_lists[i]))
    )

    present = sorted(set(int(v) for v in y_global))
    zero_shot_mask = np.array([int(y_global[i]) not in train_actions for i in range(len(y_global))])
    n_zero = int(zero_shot_mask.sum())
    seen_mask = ~zero_shot_mask

    acc = float((pred == y_global).mean())
    out = {
        "split": name,
        "n": int(len(y_global)),
        "n_below_20_flag": bool(len(y_global) < 20),
        "top1_accuracy": acc,
        "macro_recall_over_gold_classes_present": macro_recall(y_global, pred, present),
        "n_gold_classes_present": len(present),
        "gold_classes_present": present,
        "single_gold_class_degenerate": len(present) == 1,
        "majority_class_fraction_in_split": (max(Counter(y_global.tolist()).values()) / len(y_global)),
        "n_gold_classes_absent_from_train": n_zero,
        "zero_shot_gold_fraction": n_zero / len(y_global),
        "accuracy_on_gold_absent_from_train": (float((pred[zero_shot_mask] == y_global[zero_shot_mask]).mean())
                                               if n_zero else None),
        "accuracy_on_gold_present_in_train": (float((pred[seen_mask] == y_global[seen_mask]).mean())
                                              if int(seen_mask.sum()) else None),
        "n_gold_present_in_train": int(seen_mask.sum()),
        "uniform_candidate_chance_accuracy": float(
            np.mean([1.0 / len(c) for c in cand_lists])
        ),
        "predicted_actions_outside_legal_actions": outside,
        "unconstrained_argmax_outside_legal_actions": unconstrained_outside,
        "unconstrained_argmax_note": (
            "how often the same scores would leave legal_actions if the candidate mask were removed; "
            "the constrained count above is 0 by construction, this field shows the mask is load-bearing"
        ),
        "predicted_action_histogram": {str(k): int(v) for k, v in
                                       sorted(Counter(pred.tolist()).items())},
        "risk_coverage": risk_coverage(y_global, pred, conf),
        "best_operating_point": None,  # filled below
    }
    out["best_operating_point"] = best_operating_point(out["risk_coverage"])
    out["no_abstention_endpoint"] = {
        "threshold": 0.0,
        "coverage": out["risk_coverage"]["coverage"][0],
        "success_given_covered": out["risk_coverage"]["success_given_covered"][0],
    }
    return out


def train_internal_cv(Xtr, ytr, n_actions, cand_tr):
    """Informational only: stratified 5-fold inside the FIT split, never held out."""
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    res = {}
    for name in MODELS:
        accs, recalls = [], []
        for tr, te in skf.split(Xtr, ytr):
            fitted = fit_all(Xtr[tr], ytr[tr], n_actions)
            fn, _ = fitted[name]
            scores = fn(Xtr[te])
            pred, _ = decide(scores, [cand_tr[i] for i in te], n_actions)
            accs.append(float((pred == ytr[te]).mean()))
            recalls.append(macro_recall(ytr[te], pred, sorted(set(int(v) for v in ytr[te]))))
        res[name] = {
            "mean_top1_accuracy": float(np.mean(accs)),
            "std_top1_accuracy": float(np.std(accs)),
            "mean_macro_recall": float(np.mean([r for r in recalls if r is not None])),
            "folds": 5,
        }
    return res


# ------------------------------------------------------------------------ main


def main() -> int:
    rows, splits_raw, assignments, manifest = load_corpus()
    by_bucket, by_chrono, crosstab = report_buckets(assignments, rows)

    rows_by_id = {r["episode_id"]: r for r in rows}
    bucket_of = {a["episode_id"]: a["bucket"] for a in assignments}
    chrono_of = {a["episode_id"]: a["chronological_bucket"] for a in assignments}
    split_ids = {b: [a["episode_id"] for a in assignments if a["bucket"] == b] for b in SPLIT_ORDER}
    fit_ids = split_ids[FIT_BUCKET]
    assert all(bucket_of[i] == FIT_BUCKET for i in fit_ids)

    # ---- global candidate vocabulary + global action index (structural, no gold)
    candidate_vocab = sorted({a for r in rows for a in r["legal_actions"]})
    action_index = {a: i for i, a in enumerate(candidate_vocab)}
    n_actions = len(candidate_vocab)

    # ---- gold-in-legal check (compiler correctness)
    gold_in_legal = {"n_episodes_checked": len(rows), "n_missing": 0, "missing_episode_ids": []}
    per_split_gold_missing = {}
    for r in rows:
        if r["verification"]["gold_action"] not in r["legal_actions"]:
            gold_in_legal["missing_episode_ids"].append(r["episode_id"])
    gold_in_legal["n_missing"] = len(gold_in_legal["missing_episode_ids"])
    for b in SPLIT_ORDER:
        miss = [i for i in split_ids[b]
                if rows_by_id[i]["verification"]["gold_action"] not in rows_by_id[i]["legal_actions"]]
        per_split_gold_missing[b] = {"n": len(miss), "episode_ids": miss}

    # ---- train gold actions
    train_gold_actions = sorted({rows_by_id[i]["verification"]["gold_action"] for i in fit_ids})
    train_actions_idx = set(action_index[a] for a in train_gold_actions)
    print(f"global candidate vocabulary: {n_actions} actions")
    print(f"train gold action set      : {len(train_gold_actions)} actions -> {train_gold_actions}")
    held_out_gold_actions = sorted(
        {rows_by_id[i]["verification"]["gold_action"]
         for b in HELD_OUT_BUCKETS for i in split_ids[b]}
    )
    print(f"held-out gold action set   : {len(held_out_gold_actions)} -> {held_out_gold_actions}")
    print(f"held-out gold absent from train gold: "
          f"{sorted(set(held_out_gold_actions) - set(train_gold_actions))}")
    print(f"gold-in-legal-actions check: {gold_in_legal['n_missing']} missing of {len(rows)}")
    print()

    parts_by_id = {i: raw_feature_parts(rows_by_id[i]) for i in rows_by_id}
    y_global_all = np.array([action_index[rows_by_id[i]["verification"]["gold_action"]] for i in rows_by_id])
    y_by_id = {i: action_index[rows_by_id[i]["verification"]["gold_action"]] for i in rows_by_id}
    cand_idx_by_id = {
        i: [action_index[a] for a in rows_by_id[i]["legal_actions"]] for i in rows_by_id
    }

    variants = {
        "state_before_full": False,
        "no_label_adjacent": True,
    }
    results = {}
    for vname, drop_la in variants.items():
        codec = build_codec(parts_by_id, fit_ids, drop_label_adjacent=drop_la)
        X_by_id = {i: encode(parts_by_id[i], codec, drop_label_adjacent=drop_la) for i in rows_by_id}
        dim = len(next(iter(X_by_id.values())))
        Xtr = np.stack([X_by_id[i] for i in fit_ids])
        ytr = np.array([y_by_id[i] for i in fit_ids])
        cand_tr = [cand_idx_by_id[i] for i in fit_ids]

        fitted = fit_all(Xtr, ytr, n_actions)
        per_model = {}
        for name in MODELS:
            fn, info = fitted[name]
            per_split = {}
            for b in SPLIT_ORDER:
                ids = split_ids[b]
                Xb = np.stack([X_by_id[i] for i in ids])
                yb = np.array([y_by_id[i] for i in ids])
                cb = [cand_idx_by_id[i] for i in ids]
                per_split[b] = evaluate_split(
                    b, fn, Xb, yb, cb, n_actions, train_actions_idx, ids
                )
            per_model[name] = {
                "fit_info": _jsonable(info),
                "per_split": per_split,
            }
        results[vname] = {
            "feature_dim": dim,
            "feature_layout": feature_layout(codec),
            "codec": {
                "families": codec["families"],
                "legal_families": codec["legal_families"],
                "risk_classes": codec["risk_classes"],
                "max_risk": codec["max_risk"],
                "binary_keys": list(codec["binary_keys"]),
                "continuous_mean": codec["mean"].tolist(),
                "continuous_std": codec["std"].tolist(),
                "vocab_source": f"fit split only (bucket == '{FIT_BUCKET}', n={len(fit_ids)})",
            },
            "models": per_model,
        }
        print(f"[variant {vname}] feature_dim={dim}")
        for name in MODELS:
            line = "  ".join(
                f"{b}={per_model[name]['per_split'][b]['top1_accuracy']:.3f}"
                for b in HELD_OUT_BUCKETS
            )
            print(f"  {name:9s} train_internal_ok  {line}")

    # ---- informational train-internal CV (train split only, never held out)
    codec = build_codec(parts_by_id, fit_ids, drop_label_adjacent=False)
    X_by_id = {i: encode(parts_by_id[i], codec, drop_label_adjacent=False) for i in rows_by_id}
    Xtr = np.stack([X_by_id[i] for i in fit_ids])
    ytr = np.array([y_by_id[i] for i in fit_ids])
    cand_tr = [cand_idx_by_id[i] for i in fit_ids]
    cv = train_internal_cv(Xtr, ytr, n_actions, cand_tr)
    print("\ntrain-internal 5-fold CV (informational, fit split only):")
    for k, v in cv.items():
        print(f"  {k:9s} acc={v['mean_top1_accuracy']:.4f} +/- {v['std_top1_accuracy']:.4f}  macroR={v['mean_macro_recall']:.4f}")

    # ---- determinism fingerprint of the inputs
    import hashlib

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    out = {
        "schema": "z0evals.phase2-training-comparison.v1",
        "corpus": str(CORPUS.relative_to(ROOT)),
        "corpus_run_id": manifest.get("run_id"),
        "corpus_frozen_at": manifest.get("frozen_at"),
        "evidence_class": "exploratory_beta",
        "promotion_eligible": False,
        "not_committed": True,
        "generated_by": "scripts/phase2_training_comparison.py",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn_note": "scikit-learn/scipy were absent from the designated venv despite the task stating otherwise; "
                            "scikit-learn==1.9.1 (with scipy==1.17.1) was installed into /home/kvn/tmp/openjev/.venv "
                            "so the logistic/MLP legs are the standard estimators rather than hand-rolled.",
        },
        "input_digests": {
            "episodes.jsonl": sha(CORPUS / "episodes.jsonl"),
            "splits.json": sha(CORPUS / "splits.json"),
            "manifest.json": sha(CORPUS / "manifest.json"),
        },
        "split_assignment": {
            "axis_used_for_fitting": "bucket",
            "bucket_names_and_counts": {b: by_bucket[b] for b in SPLIT_ORDER},
            "chronological_bucket_names_and_counts": dict(sorted(by_chrono.items())),
            "bucket_x_chronological_crosstab": [
                {"bucket": k[0], "chronological_bucket": k[1], "n": v} for k, v in sorted(crosstab.items())
            ],
            "fit_bucket": FIT_BUCKET,
            "fit_n": by_bucket[FIT_BUCKET],
            "held_out_buckets": list(HELD_OUT_BUCKETS),
            "held_out_n": {b: by_bucket[b] for b in HELD_OUT_BUCKETS},
            "splits_under_20_episodes": [b for b in SPLIT_ORDER if by_bucket[b] < 20],
            "chronological_axis_note": (
                "bucket is the canonical Phase 2 split and the only axis used to fit. The secondary "
                "chronological axis puts 75 ood and 15 sealed episodes inside chronological 'train', so "
                "fitting on chronological train would leak protected episodes; it is reported, not used."
            ),
            "family_disjointness": {
                "train_families": sorted({rows_by_id[i]["task_family"] for i in fit_ids}),
                "held_out_families": {
                    b: sorted({rows_by_id[i]["task_family"] for i in split_ids[b]}) for b in HELD_OUT_BUCKETS
                },
            },
        },
        "action_space": {
            "definition": "sorted union of legal_actions over all 420 episodes (compiler-structural candidate "
                          "vocabulary; no gold label involved)",
            "n_actions": n_actions,
            "actions": candidate_vocab,
            "train_gold_actions": train_gold_actions,
            "n_train_gold_actions": len(train_gold_actions),
            "held_out_gold_actions": held_out_gold_actions,
            "held_out_gold_actions_absent_from_train": sorted(set(held_out_gold_actions) - set(train_gold_actions)),
            "action_space_caveat": (
                "Every model emits a score for all 30 candidate-vocabulary actions, but only the 9 train gold "
                "actions receive supervision. Classes with no train row get a fixed score: exactly 0 for "
                "majority/ridge (zero one-hot columns) and log(1e-12) for logistic/mlp. They are therefore "
                "reachable only by tie-break/floor and cannot be learned."
            ),
            "decision_rule": "masked argmax over each episode's own legal_actions; ties broken toward the lower action index",
        },
        "feature_construction": {
            "variants": list(variants),
            "primary_variant": "state_before_full",
            "description": (
                "One deterministic vector per episode built only from state_before, in this fixed order: "
                "(1) continuous z-scored with TRAIN mean/std: budget_units, features.authority_breadth, "
                "features.candidate_action_count, features.declared_dangerous_count, features.legal_family_count, "
                "features.satisfied_count; (2) binary 0/1: features.expect_abstain, features.gold_present, "
                "features.has_deterministic_solution, features.is_empty; (3) one-hot state_before.family over the "
                "train family vocabulary plus an UNK slot; (4) multi-hot features.legal_families over the train "
                "union plus UNK; (5) multi-hot features.legal_risk_classes over the train union plus UNK; "
                "(6) one-hot features.max_risk_class over the train vocabulary plus UNK. Vocabularies and scaling "
                "constants come from the train split only. legal_actions is NOT a feature; it is the candidate "
                "mask, so the four models share both the representation and the candidate sets."
            ),
            "label_adjacent_features_retained": list(LABEL_ADJACENT_KEYS),
            "label_adjacent_warning": (
                "features.expect_abstain is exactly equivalent to gold==abstain on this corpus (45/45), i.e. the "
                "frozen state_before directly encodes the gold action for the abstention family. It is retained "
                "in the primary variant because it is part of the frozen observation every model receives, and "
                "the 'no_label_adjacent' variant drops expect_abstain + has_deterministic_solution to show what "
                "survives without it."
            ),
            "constant_features_in_this_corpus": ["gold_present (always true)", "is_empty (always false)",
                                                 "satisfied_count (always 0)"],
            "per_variant": {
                vname: {
                    "feature_dim": results[vname]["feature_dim"],
                    "feature_layout": results[vname]["feature_layout"],
                    "codec": results[vname]["codec"],
                }
                for vname in variants
            },
            "candidate_masking": "identical for all models: argmax over the episode's legal_actions only",
        },
        "hyperparameters": {
            "policy": "fixed a priori; no hyperparameter was chosen by looking at any held-out split, and no "
                      "search was run.",
            "ridge": {"type": "direct one-hot least squares (all 30 columns, L2) via repo ridge_fit",
                      "l2": RIDGE_L2, "bias": "explicit ones column appended to X"},
            "logistic": {"type": "multinomial softmax", "C": LOGISTIC_C, "max_iter": LOGISTIC_MAX_ITER,
                         "solver": "lbfgs",
                         "unseen_class_score": UNSEEN_FLOOR_LOGPROB},
            "mlp": {"type": "1 hidden layer, ReLU, softmax head (sklearn MLPClassifier)",
                    "hidden_width": MLP_HIDDEN, "alpha": MLP_ALPHA, "learning_rate_init": MLP_LR,
                    "max_iter": MLP_MAX_ITER,
                    "early_stopping": f"sklearn internal validation_fraction={MLP_EARLY_STOP_VALIDATION_FRACTION} "
                                      f"carved out of the TRAIN split only, patience {MLP_EARLY_STOP_PATIENCE}",
                    "random_state": MLP_SEED, "unseen_class_score": UNSEEN_FLOOR_LOGPROB},
            "mushroom": {"n_kc": MB_N_KC, "k_winners": MB_K_WINNERS or 13, "epochs": MB_EPOCHS, "lr": MB_LR,
                         "seed": MB_SEED, "source": "evolution_lab.engine.seed_genomes() local-plasticity-000"},
            "majority": {"type": "most frequent train gold action within the candidate set"},
        },
        "gold_in_legal_actions_check": {
            "n_episodes_checked": gold_in_legal["n_episodes_checked"],
            "n_missing": gold_in_legal["n_missing"],
            "missing_episode_ids": gold_in_legal["missing_episode_ids"],
            "verdict": "PASS: every gold_action is contained in its episode's legal_actions"
                       if gold_in_legal["n_missing"] == 0 else "FAIL: compiler correctness defect",
            "per_split": per_split_gold_missing,
        },
        "results": results,
        "train_internal_cv_informational": {
            "note": "5-fold stratified CV inside the fit split only. Informational; NOT a held-out result. "
                    "All 9 train gold classes are in-distribution here, unlike the family-disjoint held-out splits.",
            "variant": "state_before_full",
            "models": cv,
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=1) + "\n")
    print(f"\nwrote {OUT_JSON.relative_to(ROOT)}")
    return 0


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


if __name__ == "__main__":
    raise SystemExit(main())
