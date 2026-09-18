"""Capability discovery objects for z0int.

Population = (TASK × DATA × MODEL × POLICY) experiments.
A CapabilityCard is a micro-process worth specializing; a BenchmarkSpec
defines how to prove it at L0–L3 evidence levels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


EVIDENCE_LEVELS = ("L0_imitation", "L1_teacher", "L2_outcome", "L3_closed_loop")
DEFAULT_CONTROLS = ("majority", "rule", "ridge", "mb", "openjev_06b", "jev")
DEFAULT_METRICS = (
    "verified_success",
    "coverage_at_95_precision",
    "latency_p50_ms",
    "tokens_saved",
    "fallback_rate",
    "transfer_sealed",
)


@dataclass
class CapabilityCard:
    """Bounded micro-process candidate for specialist offload."""

    id: str
    description: str
    stage: str
    trigger: str
    input_contract: list[str]
    output_contract: list[str]
    verifier: str
    label_sources: list[str] = field(default_factory=lambda: ["historical"])
    acceptable_actions: list[str] = field(default_factory=list)
    privacy_projection: str = "hashes_counts_only"
    frequency_events: int = 0
    frequency_sessions: int = 0
    events_per_day: float = 0.0
    current_cost_note: str = "unknown_llm_tokens"
    failure_cost: str = "medium"  # low|medium|high
    temporal_dependency: str = "low"  # low|medium|high
    label_quality: str = "medium"  # low|medium|high
    candidate_models: list[str] = field(default_factory=lambda: list(DEFAULT_CONTROLS))
    evidence_notes: list[str] = field(default_factory=list)
    schema: str = "flyforge.capability_card.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkSpec:
    """How to evaluate one capability with sealed splits."""

    capability_id: str
    split: dict[str, str]
    metrics: list[str] = field(default_factory=lambda: list(DEFAULT_METRICS))
    controls: list[str] = field(default_factory=lambda: list(DEFAULT_CONTROLS))
    evidence_levels: list[str] = field(default_factory=lambda: list(EVIDENCE_LEVELS))
    label_rule: str = ""
    feature_recipe: str = "hash64_prev_families"
    success_criterion: str = "mb_beats_ridge_and_majority_on_sealed_L0"
    failure_criterion: str = "no_lift_vs_simplest_control_on_sealed"
    notes: str = ""
    schema: str = "flyforge.benchmark_spec.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CapabilityResult:
    """One model × capability measurement cell."""

    model: str
    capability_id: str
    evidence_level: str
    n: int
    n_sessions: int = 0
    success: float | None = None
    precision: float | None = None
    coverage_at_95: float | None = None
    latency_ms: float | None = None
    cost_note: str = ""
    transfer: str = "unknown"
    source: str = ""
    inference: bool = False  # True if estimated, False if measured
    schema: str = "flyforge.capability_result.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
