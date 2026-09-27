"""Slice informativeness — the precondition a promotion gate must satisfy.

A gate compares a candidate against a threshold. That comparison is only
meaningful if the slice can *discriminate*: if a constant predictor already
clears the threshold, if every held-out label is absent from the fit split, or if
the slice has one gold class, then the gate is measuring nothing and its verdict
must not be reported as a pass.

Three real instances, all discovered on 2026-09-22:

1. **`recovery_action` OOD.** All 32/32 gold labels are ``escalate``. The
   majority baseline therefore scores 1.000 and cleared the ``ood_ge = 0.85``
   gate. The specialist was promoted to canary with
   ``promote_allowed = true`` on an OOD slice a constant predictor passes. See
   ``~/.z0int/benchmarks/recovery_action_l2.json``.

2. **Phase 2 ``validation``.** 45/45 held-out gold labels belong to classes that
   never occur in the train split, so every train-only model scores 0.000 —
   *below* the 0.333 uniform-candidate chance. Zero-shot saturation, not a
   ranking.

3. **Phase 2 ``sealed_human_audited``.** n=15 with a single gold class
   (``abstain``). Macro recall is degenerate.

There is no verdict for "the slice is bad" other than refusing to emit a verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

INFORMATIVE = "informative"
UNINFORMATIVE = "uninformative"

#: Gate-level statuses. `UNINFORMATIVE` outranks `FAIL`: a slice that cannot
#: discriminate is not a failed test, it is an absent one.
GATE_PASS = "PASS"
GATE_FAIL = "FAIL"
GATE_UNINFORMATIVE = "UNINFORMATIVE"

DEFAULT_MIN_N = 20
DEFAULT_MIN_SUPPORT = 2
DEFAULT_MIN_CLASSES = 2


@dataclass(frozen=True)
class SliceInformativeness:
    slice_id: str
    n: int
    metric_unit: str | None
    aggregation_unit: str | None
    n_gold_classes: int
    class_support: dict[str, int]
    majority_accuracy: float | None
    uniform_baseline: float | None
    gate_threshold: float | None
    zero_shot_ratio: float | None
    status: str
    reasons: tuple[str, ...]

    @property
    def informative(self) -> bool:
        return self.status == INFORMATIVE

    def to_dict(self) -> dict[str, Any]:
        return {
            "slice_id": self.slice_id,
            "n": self.n,
            "metric_unit": self.metric_unit,
            "aggregation_unit": self.aggregation_unit,
            "n_gold_classes": self.n_gold_classes,
            "class_support": dict(self.class_support),
            "majority_accuracy": self.majority_accuracy,
            "uniform_baseline": self.uniform_baseline,
            "gate_threshold": self.gate_threshold,
            "zero_shot_ratio": self.zero_shot_ratio,
            "status": self.status,
            "informative": self.informative,
            "reasons": list(self.reasons),
        }


def assess_slice(
    *,
    slice_id: str,
    gold_labels: Sequence[Any],
    fit_labels: Sequence[Any] = (),
    majority_accuracy: float | None = None,
    metric_unit: str | None = None,
    aggregation_unit: str | None = None,
    gate_threshold: float | None = None,
    min_n: int = DEFAULT_MIN_N,
    min_support: int = DEFAULT_MIN_SUPPORT,
    min_classes: int = DEFAULT_MIN_CLASSES,
    declared_class_count: int | None = None,
) -> SliceInformativeness:
    """Decide whether a held-out slice is capable of discriminating.

    ``fit_labels`` are the labels available to the model being gated. If no
    held-out label appears among them, every model that only ever saw the fit
    split is guessing and the slice cannot rank anything.
    """
    reasons: list[str] = []

    if not metric_unit:
        reasons.append("no declared metric unit")
    if not aggregation_unit:
        reasons.append("no declared aggregation unit")

    labels = [str(x) for x in gold_labels]
    n = len(labels)
    support: dict[str, int] = {}
    for x in labels:
        support[x] = support.get(x, 0) + 1
    n_classes = len(support)

    if n < min_n:
        reasons.append(f"n={n} < min_n={min_n}")

    # Multiclass means more than one class is *possible*, regardless of what this
    # slice happens to contain.
    multiclass = (declared_class_count or n_classes) >= 3
    if multiclass and n_classes < min_classes:
        reasons.append(
            f"single gold class present ({n_classes} of a possible {declared_class_count}); "
            "a constant predictor is optimal and the slice cannot discriminate"
        )
    if n_classes and min(support.values()) < min_support:
        thin = sorted(k for k, v in support.items() if v < min_support)
        reasons.append(f"class support below min_support={min_support}: {', '.join(thin)}")

    if gate_threshold is not None and majority_accuracy is not None:
        if majority_accuracy + 1e-12 >= float(gate_threshold):
            reasons.append(
                f"trivial majority baseline {majority_accuracy:.4f} already clears the "
                f"gate threshold {gate_threshold:.4f}; the gate cannot discriminate"
            )

    zero_shot: float | None = None
    if fit_labels:
        seen = {str(x) for x in fit_labels}
        absent = [k for k in support if k not in seen]
        zero_shot = len(absent) / n_classes if n_classes else None
        if zero_shot is not None and zero_shot >= 1.0:
            reasons.append(
                "every held-out gold class is absent from the fit split; only zero-shot "
                "generalisation is being measured and no fit-only model can emit these labels"
            )

    return SliceInformativeness(
        slice_id=slice_id,
        n=n,
        metric_unit=metric_unit,
        aggregation_unit=aggregation_unit,
        n_gold_classes=n_classes,
        class_support=support,
        majority_accuracy=majority_accuracy,
        uniform_baseline=(1.0 / n_classes) if n_classes else None,
        gate_threshold=gate_threshold,
        zero_shot_ratio=zero_shot,
        status=INFORMATIVE if not reasons else UNINFORMATIVE,
        reasons=tuple(reasons),
    )


def gate_status(
    slices: Sequence[SliceInformativeness],
    *,
    thresholds_passed: bool,
) -> str:
    """Combine informativeness with the threshold outcome.

    An uninformative slice is reported as ``UNINFORMATIVE`` even when the
    thresholds "passed" — that is exactly the situation that promoted the
    recovery specialist.
    """
    # An empty slice list is the degenerate case of "nothing was assessed".
    # `any()` over an empty iterable is False, so without this guard a caller
    # passing no slices collected a PASS. Reviewer D found this; it is correct.
    if not slices:
        return GATE_UNINFORMATIVE
    if any(not s.informative for s in slices):
        return GATE_UNINFORMATIVE
    return GATE_PASS if thresholds_passed else GATE_FAIL


def summarise(slices: Sequence[SliceInformativeness]) -> dict[str, Any]:
    return {
        s.slice_id: s.to_dict() for s in slices
    }
