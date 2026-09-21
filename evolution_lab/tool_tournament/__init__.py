"""Local tool-calling tournament: frozen fixtures, system compositions, scorer.

See issue kvnloo/evolution-lab#23. The package compares *system compositions*
(compiler + orchestrator + local models + specialists), not single checkpoints,
over grouped, non-random splits of a frozen decision battery.
"""

from __future__ import annotations

from .compositions import (
    COMPOSITIONS,
    DEFAULT_BACKEND_IDS,
    ROLES,
    BackendChoice,
    BackendRegistry,
    Composition,
    CompositionDecision,
    DeterministicBaselineBackend,
    LocalSLMBackendProvider,
    LogisticBaselineBackend,
    ScriptedBackend,
    TournamentBackend,
    compile_fixture,
    resolve_composition_names,
    selected_compositions,
)
from .fixtures import (
    HOLDOUT_TASK_FAMILIES,
    REQUIRED_FIXTURE_IDS,
    Fixture,
    assert_required_fixtures,
    load_fixtures,
)

__all__ = [
    "COMPOSITIONS",
    "DEFAULT_BACKEND_IDS",
    "HOLDOUT_TASK_FAMILIES",
    "REQUIRED_FIXTURE_IDS",
    "ROLES",
    "BackendChoice",
    "BackendRegistry",
    "Composition",
    "CompositionDecision",
    "DeterministicBaselineBackend",
    "Fixture",
    "LocalSLMBackendProvider",
    "LogisticBaselineBackend",
    "ScriptedBackend",
    "TournamentBackend",
    "assert_required_fixtures",
    "compile_fixture",
    "load_fixtures",
    "resolve_composition_names",
    "selected_compositions",
]
