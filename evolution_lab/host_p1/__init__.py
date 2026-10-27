"""Synthetic-only host-failure P1 baseline. No host actuation or telemetry I/O."""

from .contracts import (
    ACTUATION_AUTHORITY,
    SCHEMA_VERSION,
    TRUTH_TIER,
    Intervention,
    Outcome,
    StateFrame,
    schema_version,
)
from .synthetic import (
    EFFECT_WINDOW_S,
    TRUE_LOGIT_EFFECTS,
    make_confounded_interventions,
    planted_probability_effect,
    simulate,
)
from .baseline import (
    LogisticStateModel,
    grouped_holdout,
    intervention_effect,
    logistic_state_model,
    p3_gate,
    persistence_baseline,
    summarize_run,
)

__all__ = [
    "ACTUATION_AUTHORITY",
    "SCHEMA_VERSION",
    "TRUTH_TIER",
    "schema_version",
    "StateFrame",
    "Intervention",
    "Outcome",
    "EFFECT_WINDOW_S",
    "TRUE_LOGIT_EFFECTS",
    "make_confounded_interventions",
    "planted_probability_effect",
    "simulate",
    "LogisticStateModel",
    "persistence_baseline",
    "logistic_state_model",
    "intervention_effect",
    "grouped_holdout",
    "p3_gate",
    "summarize_run",
]
