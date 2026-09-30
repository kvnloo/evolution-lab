"""Factorized z0 stack experiment contracts.

The experiment treats SoL-Pi, RLM, StatePacket/unified memory, local SLM policy,
and Bend verification as independent mechanisms first. Composition is gated on
measured singleton evidence so a full-stack win cannot hide which mechanism
helped or harmed the task.

This module is evaluation-only. It grants no runtime authority.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any, Iterable, Literal, Mapping, Sequence

SCHEMA = "evolution_lab.z0_factorized_stack.v0"
RESULT_SCHEMA = "evolution_lab.z0_factorized_result.v0"

FactorID = Literal[
    "sol_pi_observation_pack",
    "rlm_evidence_addressing",
    "state_packet_memory",
    "local_slm_policy",
    "bend_structural_verifier",
]

Stage = Literal["observe", "address", "state", "decide", "verify"]

_STAGE_ORDER: dict[Stage, int] = {
    "observe": 0,
    "address": 1,
    "state": 2,
    "decide": 3,
    "verify": 4,
}


@dataclass(frozen=True)
class FactorSpec:
    id: FactorID
    stage: Stage
    role: str
    source_repo: str
    source_ref: str
    evidence_gate: str
    default_mode: str = "shadow"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


FACTORS: tuple[FactorSpec, ...] = (
    FactorSpec(
        id="sol_pi_observation_pack",
        stage="observe",
        role="archive oversized observations and project bounded placeholders",
        source_repo="kvnloo/sol-pi-hermes",
        source_ref="issue/2 + main",
        evidence_gate="retention/equivalence + context reduction on frozen CaptureResult fixtures",
    ),
    FactorSpec(
        id="rlm_evidence_addressing",
        stage="address",
        role="externalize large evidence behind bounded handles and query it selectively",
        source_repo="kvnloo/oh-my-pi",
        source_ref="exp/rlm-evidence-ab-clean",
        evidence_gate="same-task verified outcome with lower evidence/context cost",
    ),
    FactorSpec(
        id="state_packet_memory",
        stage="state",
        role="compile minimum-sufficient current state from temporal/lexical/semantic evidence",
        source_repo="kvnloo/z0intelligence",
        source_ref="issues/22,63",
        evidence_gate="provenance-complete StatePacket with no missed blocking unknown",
    ),
    FactorSpec(
        id="local_slm_policy",
        stage="decide",
        role="choose among already-legal bounded actions with calibrated abstention",
        source_repo="kvnloo/z0intelligence",
        source_ref="issue/20 + evolution-lab/issues/23",
        evidence_gate="safe coverage on frozen grouped tasks; legal-action filter remains authoritative",
    ),
    FactorSpec(
        id="bend_structural_verifier",
        stage="verify",
        role="fast deterministic structural check before/after expensive semantic work",
        source_repo="kvnloo/bend",
        source_ref="exp/hermes-verdict-semantic-attestation-1212",
        evidence_gate="zero unexplained divergence from canonical validator + measurable systems benefit",
    ),
)

_FACTOR_BY_ID = {f.id: f for f in FACTORS}

# These answer specific interaction questions before any full 2^N sweep.
RECOMMENDED_PAIRS: tuple[tuple[FactorID, FactorID], ...] = (
    ("sol_pi_observation_pack", "rlm_evidence_addressing"),
    ("rlm_evidence_addressing", "state_packet_memory"),
    ("state_packet_memory", "local_slm_policy"),
    ("local_slm_policy", "bend_structural_verifier"),
    ("sol_pi_observation_pack", "state_packet_memory"),
)


@dataclass(frozen=True)
class Arm:
    id: str
    factors: tuple[FactorID, ...] = ()
    phase: str = "singleton"

    def __post_init__(self) -> None:
        unknown = [f for f in self.factors if f not in _FACTOR_BY_ID]
        if unknown:
            raise ValueError(f"unknown factors: {unknown}")
        if len(set(self.factors)) != len(self.factors):
            raise ValueError("arm contains duplicate factors")
        ordered = tuple(sorted(self.factors, key=lambda f: _STAGE_ORDER[_FACTOR_BY_ID[f].stage]))
        if ordered != self.factors:
            raise ValueError("arm factors must follow observe->address->state->decide->verify stage order")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "id": self.id,
            "phase": self.phase,
            "factors": list(self.factors),
            "stages": [_FACTOR_BY_ID[f].stage for f in self.factors],
        }


@dataclass(frozen=True)
class FactorReceipt:
    factor: FactorID
    trace_id: str
    schema: str
    source: str
    revision: str
    artifact: str | None = None

    def __post_init__(self) -> None:
        if self.factor not in _FACTOR_BY_ID:
            raise ValueError(f"unknown factor {self.factor!r}")
        for name in ("trace_id", "schema", "source", "revision"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} is required")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArmSummary:
    """Aggregate one arm over the exact same grouped task cohort as control."""

    arm_id: str
    n_work_items: int
    verified_success_rate: float
    hard_policy_violations: int = 0
    provenance_failures: int = 0
    mean_input_tokens: float | None = None
    mean_context_bytes: float | None = None
    mean_raw_source_reads: float | None = None
    mean_frontier_calls: float | None = None
    p95_latency_ms: float | None = None
    mean_retries: float | None = None
    mean_corrections: float | None = None
    peak_vram_gb: float | None = None
    receipts: tuple[FactorReceipt, ...] = ()
    measurements: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_work_items <= 0:
            raise ValueError("n_work_items must be positive")
        if not 0.0 <= self.verified_success_rate <= 1.0:
            raise ValueError("verified_success_rate must be in [0,1]")
        if self.hard_policy_violations < 0 or self.provenance_failures < 0:
            raise ValueError("failure counters cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RESULT_SCHEMA,
            "arm_id": self.arm_id,
            "n_work_items": self.n_work_items,
            "verified_success_rate": self.verified_success_rate,
            "hard_policy_violations": self.hard_policy_violations,
            "provenance_failures": self.provenance_failures,
            "mean_input_tokens": self.mean_input_tokens,
            "mean_context_bytes": self.mean_context_bytes,
            "mean_raw_source_reads": self.mean_raw_source_reads,
            "mean_frontier_calls": self.mean_frontier_calls,
            "p95_latency_ms": self.p95_latency_ms,
            "mean_retries": self.mean_retries,
            "mean_corrections": self.mean_corrections,
            "peak_vram_gb": self.peak_vram_gb,
            "receipts": [r.to_dict() for r in self.receipts],
            "measurements": dict(self.measurements),
        }


@dataclass(frozen=True)
class GatePolicy:
    quality_noninferiority_margin: float = 0.0
    correction_regression_tolerance: float = 0.0
    retry_regression_tolerance: float = 0.0
    require_efficiency_or_quality_gain: bool = True


@dataclass(frozen=True)
class GateDecision:
    qualified: bool
    reasons: tuple[str, ...]
    deltas: Mapping[str, float | None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "qualified": self.qualified,
            "reasons": list(self.reasons),
            "deltas": dict(self.deltas),
        }


def phase0_arms() -> tuple[Arm, ...]:
    """Native control plus one factor at a time; deterministic order."""
    arms = [Arm(id="control", factors=(), phase="control")]
    for factor in FACTORS:
        arms.append(Arm(id=f"{factor.id}__only", factors=(factor.id,), phase="singleton"))
    return tuple(arms)


def pairwise_arms(
    qualified_singletons: Iterable[FactorID],
    *,
    recommended_only: bool = True,
) -> tuple[Arm, ...]:
    """Return legal pairs only when *both* singleton factors already qualified."""
    qualified = set(qualified_singletons)
    candidates: Sequence[tuple[FactorID, FactorID]]
    if recommended_only:
        candidates = RECOMMENDED_PAIRS
    else:
        candidates = tuple(combinations((f.id for f in FACTORS), 2))  # type: ignore[assignment]

    out: list[Arm] = []
    for left, right in candidates:
        if left not in qualified or right not in qualified:
            continue
        ordered = tuple(
            sorted((left, right), key=lambda f: _STAGE_ORDER[_FACTOR_BY_ID[f].stage])
        )
        out.append(Arm(id="__x__".join(ordered), factors=ordered, phase="pairwise"))
    return tuple(out)


def assert_composition_allowed(arm: Arm, qualified_singletons: Iterable[FactorID]) -> None:
    """Fail closed when a composite arm hides an unqualified constituent."""
    if len(arm.factors) <= 1:
        return
    qualified = set(qualified_singletons)
    missing = [f for f in arm.factors if f not in qualified]
    if missing:
        raise ValueError(f"composition blocked; singleton evidence missing for {missing}")


def _delta(candidate: float | None, control: float | None) -> float | None:
    if candidate is None or control is None:
        return None
    return candidate - control


def compare(candidate: ArmSummary, control: ArmSummary) -> dict[str, float | None]:
    """Candidate-minus-control deltas. Negative resource deltas are improvements."""
    if candidate.n_work_items != control.n_work_items:
        raise ValueError("candidate/control must use the same frozen work-item count")
    return {
        "verified_success_rate": _delta(
            candidate.verified_success_rate, control.verified_success_rate
        ),
        "mean_input_tokens": _delta(candidate.mean_input_tokens, control.mean_input_tokens),
        "mean_context_bytes": _delta(candidate.mean_context_bytes, control.mean_context_bytes),
        "mean_raw_source_reads": _delta(
            candidate.mean_raw_source_reads, control.mean_raw_source_reads
        ),
        "mean_frontier_calls": _delta(
            candidate.mean_frontier_calls, control.mean_frontier_calls
        ),
        "p95_latency_ms": _delta(candidate.p95_latency_ms, control.p95_latency_ms),
        "mean_retries": _delta(candidate.mean_retries, control.mean_retries),
        "mean_corrections": _delta(candidate.mean_corrections, control.mean_corrections),
        "peak_vram_gb": _delta(candidate.peak_vram_gb, control.peak_vram_gb),
    }


def qualify_singleton(
    candidate: ArmSummary,
    control: ArmSummary,
    *,
    policy: GatePolicy | None = None,
) -> GateDecision:
    """Mechanism-neutral singleton gate.

    Verified quality and hard invariants are gates. Efficiency is reported on
    independent axes rather than blended into one score.
    """
    policy = policy or GatePolicy()
    deltas = compare(candidate, control)
    reasons: list[str] = []

    if candidate.hard_policy_violations:
        reasons.append("hard_policy_violation")
    if candidate.provenance_failures:
        reasons.append("provenance_failure")

    quality_delta = deltas["verified_success_rate"]
    assert quality_delta is not None
    if quality_delta < -policy.quality_noninferiority_margin:
        reasons.append("verified_quality_regression")

    correction_delta = deltas["mean_corrections"]
    if (
        correction_delta is not None
        and correction_delta > policy.correction_regression_tolerance
    ):
        reasons.append("correction_regression")

    retry_delta = deltas["mean_retries"]
    if retry_delta is not None and retry_delta > policy.retry_regression_tolerance:
        reasons.append("retry_regression")

    if policy.require_efficiency_or_quality_gain and not reasons:
        quality_gain = quality_delta > 0
        efficiency_gain = any(
            value is not None and value < 0
            for key, value in deltas.items()
            if key
            in {
                "mean_input_tokens",
                "mean_context_bytes",
                "mean_raw_source_reads",
                "mean_frontier_calls",
                "p95_latency_ms",
            }
        )
        if not quality_gain and not efficiency_gain:
            reasons.append("no_measured_benefit")

    return GateDecision(qualified=not reasons, reasons=tuple(reasons), deltas=deltas)


def experiment_manifest() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "authority": "evaluation_only",
        "factors": [f.to_dict() for f in FACTORS],
        "phase0_arms": [a.to_dict() for a in phase0_arms()],
        "recommended_pairs": [list(pair) for pair in RECOMMENDED_PAIRS],
        "promotion_sequence": ["control", "singleton", "pairwise", "higher_order"],
        "hard_constraints": [
            "downstream_only",
            "shadow_or_default_off",
            "deterministic_policy_authoritative",
            "protected_judge_outside_evolvable_surface",
            "same_grouped_work_items_per_arm",
            "no_full_stack_before_singleton_evidence",
            "preserve_factor_receipts",
            "no_two_heavy_slms_co_resident_on_12gb_target",
        ],
    }
