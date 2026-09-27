"""The promotion gate must refuse to emit a verdict on an uninformative slice.

Reproduces the 2026-09-21 `recovery_action` promotion, which reached canary with
``promote_allowed = true`` on an OOD slice where every label was ``escalate``.
"""

from __future__ import annotations

import pytest

from evolution_lab.slice_informativeness import (
    GATE_FAIL,
    GATE_PASS,
    GATE_UNINFORMATIVE,
    INFORMATIVE,
    UNINFORMATIVE,
    assess_slice,
    gate_status,
)


def _assess(**kw):
    base = dict(
        slice_id="ood",
        gold_labels=[2, 2, 2, 2, 2, 2, 2, 2],
        fit_labels=[0, 0, 1, 1],
        majority_accuracy=1.0,
        metric_unit="accuracy",
        aggregation_unit="per_episode",
        gate_threshold=0.85,
        declared_class_count=5,
    )
    base.update(kw)
    return assess_slice(**base)


def test_recovery_ood_slice_is_uninformative():
    """32/32 `escalate`: a constant predictor clears the gate."""
    s = _assess(gold_labels=[2] * 32, majority_accuracy=1.0)

    assert s.status == UNINFORMATIVE
    assert s.n_gold_classes == 1
    joined = " | ".join(s.reasons)
    assert "single gold class" in joined
    assert "cannot discriminate" in joined


def test_uninformative_outranks_pass_at_the_gate_level():
    """The exact failure: thresholds cleared, slice worthless."""
    s = _assess(gold_labels=[2] * 32, majority_accuracy=1.0)

    # Thresholds all cleared...
    assert gate_status([s], thresholds_passed=True) == GATE_UNINFORMATIVE
    # ...and it still is not a pass.
    assert gate_status([s], thresholds_passed=True) != GATE_PASS
    assert gate_status([s], thresholds_passed=False) == GATE_UNINFORMATIVE


def test_a_healthy_multi_class_slice_is_informative():
    s = _assess(
        gold_labels=[0] * 10 + [1] * 10 + [2] * 10,
        majority_accuracy=1 / 3,
        gate_threshold=0.85,
    )
    assert s.status == INFORMATIVE
    assert s.reasons == ()
    assert gate_status([s], thresholds_passed=True) == GATE_PASS
    assert gate_status([s], thresholds_passed=False) == GATE_FAIL


def test_zero_shot_slice_is_uninformative():
    """Phase 2 `validation`: every held-out label absent from the fit split."""
    s = _assess(
        gold_labels=[7] * 20 + [8] * 20 + [9] * 5,
        fit_labels=[0, 1, 2, 3],
        majority_accuracy=0.0,
        gate_threshold=0.5,
    )
    assert s.status == UNINFORMATIVE
    assert s.zero_shot_ratio == 1.0
    assert any("absent from the fit split" in r for r in s.reasons)


def test_thin_slice_and_thin_support_are_flagged():
    assert "min_n" in " | ".join(_assess(gold_labels=[0] * 15, majority_accuracy=0.0).reasons)

    s = _assess(gold_labels=[0] * 30 + [1], majority_accuracy=0.9)
    assert any("class support" in r for r in s.reasons)


def test_missing_declared_units_are_flagged():
    s = _assess(metric_unit=None, aggregation_unit=None)
    assert any("metric unit" in r for r in s.reasons)
    assert any("aggregation unit" in r for r in s.reasons)


def test_unit_mismatch_is_visible_across_two_assessments():
    """per_episode 1.000 and per_step 0.839 must not look interchangeable."""
    per_episode = _assess(aggregation_unit="per_episode")
    per_step = _assess(aggregation_unit="per_step")
    assert per_episode.aggregation_unit != per_step.aggregation_unit
    assert per_episode.to_dict()["aggregation_unit"] == "per_episode"
    assert per_step.to_dict()["aggregation_unit"] == "per_step"


def test_empty_slice_list_is_not_a_pass():
    """`any()` over an empty iterable is False, so [] used to return GATE_PASS.

    Found by an independent remote reviewer; the gate must not pass when nothing
    was assessed.
    """
    assert gate_status([], thresholds_passed=True) == GATE_UNINFORMATIVE
    assert gate_status([], thresholds_passed=False) == GATE_UNINFORMATIVE
