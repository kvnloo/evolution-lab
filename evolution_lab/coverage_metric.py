"""Safe offload metric: maximize coverage subject to precision ≥ floor.

Frozen claim language (do not loosen in prose without a new gen lock):

* **Gen-0** (wave-5 hash PN, n_kc=96, k_winners=20, confirm_acc≈0.572):
  DELEGATE *predictions* on chronological confirm cover ≈19.9% of events at
  precision ≈0.980 (n=3063). That is a **prediction-conditional** figure, not
  "the model is 98% accurate" and not population DELEGATE base-rate.
* **Gen-1** (old labels + rich PN): coverage@95 on local EXECUTE|DELEGATE mask
  ≈0.287 > gen-0 0.199; cascade local_prec ≈0.90. Confirm_acc ≈0.630.
* Prefer **coverage@≥0.95 precision**, **lift over base-rate**, and **FPR** when
  reporting DELEGATE gating. Raw precision alone is incomplete under class skew.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Locked gen-0 reference (confirm split, margin rank, DELEGATE-pred mass).
GEN0_COVERAGE_AT_95 = 0.19897362608808628
GEN0_PRECISION_AT_95 = 0.980411361410382
GEN0_N_AT_95 = 3063
GEN0_CONFIRM_ACC = 0.5719760945823048

# Gen-1 locked promote snapshot (production rebench after 2×2 rich PN).
GEN1_COVERAGE_AT_95 = 0.28692997271664283
GEN1_PRECISION_AT_95 = 0.9501924383065429
GEN1_CASCADE_COVERAGE = 0.3687150837988827
GEN1_CASCADE_LOCAL_PRECISION = 0.9001057082452432
GEN1_CONFIRM_ACC = 0.630  # approximate; see champion.json for exact run

CLAIM_SCHEMA = "flyforge.science_claims.v1"


def coverage_at_precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    score: np.ndarray,
    *,
    floor: float = 0.95,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Largest score-ranked prefix with precision ≥ floor.

    score: higher = more confident local absorb (e.g. margin or p_max).
    mask: optional candidate filter (e.g. only DELEGATE|EXECUTE preds).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    score = np.asarray(score, dtype=np.float64)
    n = int(len(y_true))
    if mask is None:
        idx = np.arange(n)
    else:
        idx = np.where(np.asarray(mask, dtype=bool))[0]
    if idx.size == 0:
        return {
            "n": 0,
            "coverage": 0.0,
            "precision": None,
            "floor": floor,
            "score_min": None,
            "beats_gen0": False,
        }
    order = idx[np.argsort(-score[idx])]
    best: dict[str, Any] = {
        "n": 0,
        "coverage": 0.0,
        "precision": None,
        "floor": floor,
        "score_min": None,
        "beats_gen0": False,
    }
    c_ok = 0
    for i, j in enumerate(order, start=1):
        c_ok += int(y_pred[j] == y_true[j])
        prec = c_ok / i
        if prec + 1e-12 >= floor:
            best = {
                "n": i,
                "coverage": i / n,
                "precision": prec,
                "floor": floor,
                "score_min": float(score[j]),
                "beats_gen0": (i / n) > GEN0_COVERAGE_AT_95 + 1e-12,
            }
    return best


def risk_coverage_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    score: np.ndarray,
    *,
    floors: tuple[float, ...] = (0.90, 0.95, 0.99),
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    return {str(f): coverage_at_precision(y_true, y_pred, score, floor=f, mask=mask) for f in floors}


def binary_gate_metrics(
    y_true: np.ndarray,
    y_score_or_pred: np.ndarray,
    *,
    positive: int | bool = 1,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Base-rate, precision, recall/TPR, FPR, lift for a binary gate (e.g. DELEGATE).

    If ``threshold`` is None and ``y_score_or_pred`` is integer/bool labels, treat
    as hard predictions. If threshold is set, predict positive when score >= threshold.
    """
    y = np.asarray(y_true)
    pos = int(positive)
    y_bin = (y == pos).astype(np.int32)
    n = int(y_bin.size)
    base = float(y_bin.mean()) if n else 0.0
    s = np.asarray(y_score_or_pred)
    if threshold is None and s.dtype.kind in "biu":
        pred = (s == pos).astype(np.int32)
    elif threshold is None:
        # scores without threshold → rank-all as positive when score>0 else hardcast
        pred = (s > 0).astype(np.int32) if np.issubdtype(s.dtype, np.floating) else (s == pos).astype(np.int32)
    else:
        pred = (s.astype(np.float64) >= float(threshold)).astype(np.int32)

    tp = int(((pred == 1) & (y_bin == 1)).sum())
    fp = int(((pred == 1) & (y_bin == 0)).sum())
    tn = int(((pred == 0) & (y_bin == 0)).sum())
    fn = int(((pred == 0) & (y_bin == 1)).sum())
    pred_pos = tp + fp
    actual_neg = tn + fp
    precision = (tp / pred_pos) if pred_pos else None
    recall = (tp / (tp + fn)) if (tp + fn) else None
    fpr = (fp / actual_neg) if actual_neg else None
    tnr = (tn / actual_neg) if actual_neg else None
    lift = (precision / base) if (precision is not None and base > 0) else None
    return {
        "schema": "flyforge.binary_gate_metrics.v1",
        "n": n,
        "base_rate": base,
        "n_pred_positive": pred_pos,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "tpr": recall,
        "fpr": fpr,
        "tnr": tnr,
        "lift_over_base_rate": lift,
        "accuracy": float((tp + tn) / n) if n else None,
        "note": "Report lift+FPR with precision; precision alone is incomplete under skew",
    }


def gen0_baseline() -> dict[str, Any]:
    return {
        "schema": "flyforge.gen0_safe_offload.v1",
        "confirm_acc": GEN0_CONFIRM_ACC,
        "coverage_at_95": GEN0_COVERAGE_AT_95,
        "precision_at_95": GEN0_PRECISION_AT_95,
        "n_at_95": GEN0_N_AT_95,
        "note": (
            "DELEGATE-pred mass on frozen chronological confirm. "
            "Precision is prediction-conditional (P(true DELEGATE | pred DELEGATE)), "
            "not overall accuracy and not population base-rate."
        ),
    }


def frozen_claims() -> dict[str, Any]:
    """Machine-readable locked claim language for agents and docs."""
    return {
        "schema": CLAIM_SCHEMA,
        "gen0": {
            **gen0_baseline(),
            "forbidden_phrasing": [
                "98.2% accurate model",
                "model is 98% precise overall",
                "DELEGATE base-rate is 98%",
            ],
            "required_phrasing": (
                "DELEGATE predictions on confirm cover ~19.9% at ~98.0% precision "
                "(n=3063); cite coverage@95 + lift/FPR for gating claims"
            ),
        },
        "gen1": {
            "coverage_at_95": GEN1_COVERAGE_AT_95,
            "precision_at_95": GEN1_PRECISION_AT_95,
            "cascade_coverage": GEN1_CASCADE_COVERAGE,
            "cascade_local_precision": GEN1_CASCADE_LOCAL_PRECISION,
            "confirm_acc_approx": GEN1_CONFIRM_ACC,
            "note": "old_labels + rich PN; beats gen-0 coverage@95 on local mask",
        },
        "promote_metric": "coverage_at_95_precision",
        "precision_floor": 0.95,
        "delegate_reporting": [
            "base_rate",
            "precision",
            "lift_over_base_rate",
            "fpr",
            "coverage_at_95",
        ],
    }
