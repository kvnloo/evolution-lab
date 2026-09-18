"""Safe offload metric: maximize coverage subject to precision ≥ floor.

Gen-0 baseline (wave-5 cascade): DELEGATE predictions alone cover ~19.9% of
confirm at ~98.0% precision. EXECUTE has no ≥0.90 margin slice.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Locked gen-0 reference (confirm split, margin rank, DELEGATE-pred mass).
GEN0_COVERAGE_AT_95 = 0.19897362608808628
GEN0_PRECISION_AT_95 = 0.980411361410382
GEN0_N_AT_95 = 3063
GEN0_CONFIRM_ACC = 0.5719760945823048


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


def gen0_baseline() -> dict[str, Any]:
    return {
        "schema": "flyforge.gen0_safe_offload.v1",
        "confirm_acc": GEN0_CONFIRM_ACC,
        "coverage_at_95": GEN0_COVERAGE_AT_95,
        "precision_at_95": GEN0_PRECISION_AT_95,
        "n_at_95": GEN0_N_AT_95,
        "note": "DELEGATE-pred mass on frozen chronological confirm",
    }
