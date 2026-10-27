"""Synthetic-only, non-actuating P1 comparisons and risk-adjusted event study."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from math import isclose, isfinite
from typing import Any

import numpy as np

from .contracts import SCHEMA_VERSION, TRUTH_TIER, Intervention, Outcome, StateFrame
from .synthetic import EFFECT_WINDOW_S

_FEATURES = ("mem_used_frac", "swap_used_frac", "zram_used_frac")


def _aligned(states: Sequence[StateFrame], outcomes: Sequence[Outcome]) -> None:
    if not states or len(states) != len(outcomes):
        raise ValueError("states and outcomes must be nonempty and aligned")
    if any(
        state.t != outcome.t or outcome.window_s != 1 or state.t != i
        for i, (state, outcome) in enumerate(zip(states, outcomes))
    ):
        raise ValueError("expected consecutive one-second, zero-based windows")


def _matrix(states: Sequence[StateFrame]) -> np.ndarray:
    values = np.asarray(
        [[1.0, *(float(getattr(state, name)) for name in _FEATURES)] for state in states],
        dtype=float,
    )
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("at least one state is required")
    return values


def _prob(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -35.0, 35.0)))


def persistence_baseline(outcomes: Sequence[Outcome]) -> np.ndarray:
    """For each next one-second window, predict the current OOM outcome."""
    if any(outcome.window_s != 1 for outcome in outcomes):
        raise ValueError("persistence expects one-second windows")
    return np.asarray([float(o.oom_kill_in_window) for o in outcomes[:-1]], dtype=float)


@dataclass(frozen=True)
class LogisticStateModel:
    coefficients: tuple[float, float, float, float]

    def predict_proba(self, states: Sequence[StateFrame]) -> np.ndarray:
        """OOM probability for the supplied states, in their input order."""
        return _prob(_matrix(states) @ np.asarray(self.coefficients, dtype=float))


def logistic_state_model(
    train: tuple[Sequence[StateFrame], Sequence[Outcome]],
) -> LogisticStateModel:
    """Fit state-only logistic risk by ridge-stabilized IRLS (numpy only).

    train is a pair of aligned (states, outcomes). The intercept is unpenalized.
    Do not train on treated windows when using this model for causal adjustment.
    """
    states, outcomes = train
    _aligned_pair = len(states) == len(outcomes) and all(
        state.t == outcome.t and outcome.window_s == 1
        for state, outcome in zip(states, outcomes)
    )
    if not states or not _aligned_pair:
        raise ValueError("training states/outcomes must be aligned")
    x = _matrix(states)
    y = np.asarray([float(o.oom_kill_in_window) for o in outcomes])
    beta = np.zeros(x.shape[1], dtype=float)
    beta[0] = float(np.log((y.sum() + 0.5) / (len(y) - y.sum() + 0.5)))
    penalty = np.diag([0.0, 0.10, 0.10, 0.10])
    for _ in range(80):
        probs = _prob(x @ beta)
        weights = np.maximum(probs * (1.0 - probs), 1e-8)
        gradient = x.T @ (y - probs) - penalty @ beta
        curvature = x.T @ (x * weights[:, None]) + penalty
        step = np.linalg.solve(curvature, gradient)
        beta += np.clip(step, -5.0, 5.0)
        if float(np.max(np.abs(step))) < 1e-9:
            break
    return LogisticStateModel(tuple(float(value) for value in beta))


def _exposed(n: int, interventions: Sequence[Intervention]) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for event in interventions:
        if not 0 <= event.t < n:
            raise ValueError("intervention outside observed windows")
        if event.effective_matches_intent:
            mask[event.t:min(n, event.t + EFFECT_WINDOW_S)] = True
    return mask


def intervention_effect(
    states: Sequence[StateFrame],
    outcomes: Sequence[Outcome],
    interventions: Sequence[Intervention],
    pre_s: int,
    post_s: int,
    *,
    seed: int = 0,
    bootstrap_samples: int = 800,
) -> dict[str, Any]:
    """Adjusted post-minus-pre OOM probability effect per isolated knob.

    For each eligible intervention, subtract the state-only model's risk from
    observed OOM, before and after actuation. Fit only on untreated windows.
    Aggregate by group_id and bootstrap groups, not individual seconds.
    Non-actuating attempts and overlapping/missing windows are separately
    listed; effects are probability differences, not logit coefficients.
    """
    _aligned(states, outcomes)
    if type(pre_s) is not int or type(post_s) is not int or pre_s < 1 or post_s < 1:
        raise ValueError("pre_s and post_s must be positive integer seconds")
    if type(seed) is not int or seed < 0 or type(bootstrap_samples) is not int or bootstrap_samples < 2:
        raise ValueError("invalid bootstrap configuration")

    interventions = list(interventions)
    exposed = _exposed(len(states), interventions)
    unaffected = np.flatnonzero(~exposed)
    if len(unaffected) < 5:
        raise ValueError("too few untreated observations to fit baseline")
    model = logistic_state_model((
        [states[int(i)] for i in unaffected],
        [outcomes[int(i)] for i in unaffected],
    ))
    prediction = model.predict_proba(states)
    observed = np.asarray([float(outcome.oom_kill_in_window) for outcome in outcomes])
    residual = observed - prediction

    inactive = [
        {"t": event.t, "knob": event.knob, "group_id": event.group_id,
         "intended_value": float(event.intended_value),
         "effective_value": float(event.effective_value),
         "reason": "actuation_mismatch"}
        for event in interventions if not event.effective_matches_intent
    ]
    active = [event for event in interventions if event.effective_matches_intent]
    scores: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    unestimable: list[dict[str, Any]] = []
    for event in active:
        left = event.t - pre_s
        right = event.t + post_s
        reason = ""
        if left < 0 or right > len(states):
            reason = "incomplete_pre_post_window"
        elif any(
            other is not event
            and other.t < right
            and other.t + EFFECT_WINDOW_S > left
            for other in active
        ):
            reason = "overlapping_active_interventions"
        if reason:
            unestimable.append({
                "t": event.t, "knob": event.knob,
                "group_id": event.group_id, "reason": reason,
            })
            continue
        naive = float(observed[event.t:right].mean() - observed[left:event.t].mean())
        adjusted = float(residual[event.t:right].mean() - residual[left:event.t].mean())
        scores[event.knob][event.group_id].append((naive, adjusted))

    rng = np.random.default_rng(seed)
    effects: dict[str, dict[str, Any]] = {}
    for knob in sorted(scores):
        groups = scores[knob]
        # One independent observation per group, even if several events share it.
        values = np.asarray(
            [np.mean(groups[group], axis=0) for group in sorted(groups)], dtype=float
        )
        naive = float(values[:, 0].mean())
        adjusted = float(values[:, 1].mean())
        if len(groups) >= 2:
            indices = rng.integers(0, len(groups), size=(bootstrap_samples, len(groups)))
            boot = values[indices, 1].mean(axis=1)
            ci95: list[float | None] = [
                float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)),
            ]
        else:
            ci95 = [None, None]
        effects[knob] = {
            "adjusted_effect": adjusted,
            "naive_effect": naive,
            "ci95": ci95,
            "n_events": sum(len(events) for events in groups.values()),
            "n_groups": len(groups),
            "unit": "oom_probability_difference",
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "truth_tier": TRUTH_TIER,
        "effects": effects,
        "ineffective": inactive,
        "unestimable": unestimable,
        "state_model_coefficients": list(model.coefficients),
        "bootstrap_seed": seed,
    }


def grouped_holdout(
    interventions: Sequence[Intervention], frac: float, seed: int
) -> tuple[list[Intervention], list[Intervention]]:
    """Return (train, holdout) without leaking co-timed group IDs."""
    if not isfinite(frac) or not 0.0 <= frac <= 1.0:
        raise ValueError("frac must be finite and in [0, 1]")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    interventions = list(interventions)
    groups = sorted({event.group_id for event in interventions})
    if frac == 1.0:
        return [], interventions
    if len(groups) < 2 or frac == 0.0:
        return interventions, []
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(groups))
    n_holdout = max(1, min(len(groups) - 1, int(round(len(groups) * frac))))
    held = set(shuffled[:n_holdout])
    return (
        [event for event in interventions if event.group_id not in held],
        [event for event in interventions if event.group_id in held],
    )


def p3_gate(p1_score: float, p3_score: float, min_gain: float) -> dict[str, Any]:
    """Require a strictly greater than min_gain reduction in grouped Brier.

    The caller must supply Brier scores computed on the SAME grouped holdout.
    Lower scores are better. In-sample or temporal-only scores do not earn P3.
    """
    if any(not isfinite(float(v)) for v in (p1_score, p3_score, min_gain)):
        raise ValueError("scores and gain must be finite")
    if not 0.0 <= p1_score <= 1.0 or not 0.0 <= p3_score <= 1.0 or min_gain < 0:
        raise ValueError("invalid Brier score or minimum gain")
    improvement = float(p1_score - p3_score)
    earned = bool(improvement > min_gain and not isclose(improvement, min_gain, rel_tol=1e-12, abs_tol=1e-12))
    if earned:
        reason = "P3 grouped-holdout Brier gain exceeds the required minimum"
    elif improvement <= 0:
        reason = "P3 does not beat P1 on grouped-holdout Brier"
    else:
        reason = "P3 grouped-holdout Brier gain is within the minimum threshold"
    return {"earned": earned, "reason": reason}


def summarize_run(
    states: Sequence[StateFrame],
    outcomes: Sequence[Outcome],
    interventions: Sequence[Intervention],
    *,
    pre_s: int = 8,
    post_s: int = EFFECT_WINDOW_S,
    seed: int = 0,
) -> dict[str, Any]:
    """Produce a JSON-able P1 receipt: scores, effects, and split membership.

    Temporal Brier scores do NOT qualify for the P3 gate. The grouped Brier
    score is measured on held-out synthetic intervention windows, which must
    also be used when assessing a competing P3 model.
    """
    _aligned(states, outcomes)
    if len(states) < 12:
        raise ValueError("at least twelve windows required for a run summary")
    interventions = list(interventions)
    split = int(len(states) * 0.75)
    exposed = _exposed(len(states), interventions)
    train_indices = [i for i in range(split) if not exposed[i]]
    model = logistic_state_model((
        [states[i] for i in train_indices],
        [outcomes[i] for i in train_indices],
    ))
    forecast = model.predict_proba(states[split:])
    labels = np.asarray([float(o.oom_kill_in_window) for o in outcomes[split:]])
    persistence = persistence_baseline(outcomes)[split - 1:]
    if len(persistence) != len(labels):
        raise ValueError("persistence alignment error")
    effect_report = intervention_effect(
        states, outcomes, interventions, pre_s, post_s, seed=seed
    )
    train, heldout = grouped_holdout(interventions, 0.25, seed)
    # Held-out groups designate evaluation windows. No affected label was
    # included in state-model training, including labels before the time cut.
    group_windows = sorted({
        i
        for event in heldout if event.effective_matches_intent
        for i in range(event.t, min(len(states), event.t + EFFECT_WINDOW_S))
        if i > 0
    })
    group_probs = model.predict_proba([states[i] for i in group_windows]) if group_windows else []
    group_truth = np.asarray([float(outcomes[i].oom_kill_in_window) for i in group_windows])
    group_persistence = np.asarray([float(outcomes[i - 1].oom_kill_in_window) for i in group_windows])
    return {
        "schema_version": SCHEMA_VERSION,
        "truth_tier": TRUTH_TIER,
        "metrics": {
            "n_steps": len(states),
            "oom_rate": float(np.mean([o.oom_kill_in_window for o in outcomes])),
            "temporal_holdout_start_t": int(split),
            "temporal_holdout_brier_state": float(np.mean((labels - forecast) ** 2)),
            "temporal_holdout_brier_persistence": float(
                np.mean((labels - persistence) ** 2)
            ),
            "grouped_holdout_brier_state": (
                float(np.mean((group_truth - group_probs) ** 2)) if group_windows else None
            ),
            "grouped_holdout_brier_persistence": (
                float(np.mean((group_truth - group_persistence) ** 2)) if group_windows else None
            ),
        },
        "effects": effect_report["effects"],
        "ineffective": effect_report["ineffective"],
        "unestimable": effect_report["unestimable"],
        "grouped_holdout": {
            "train_group_ids": sorted({event.group_id for event in train}),
            "test_group_ids": sorted({event.group_id for event in heldout}),
            "test_window_times": group_windows,
        },
    }
