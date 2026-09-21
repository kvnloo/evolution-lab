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
from .score import (
    TOURNAMENT_SCHEMA,
    TournamentSplit,
    make_tournament_split,
    percentile,
    percentile_triplet,
    run_tournament,
    write_run,
)

__all__ = [
    "COMPOSITIONS",
    "DEFAULT_BACKEND_IDS",
    "HOLDOUT_TASK_FAMILIES",
    "REQUIRED_FIXTURE_IDS",
    "ROLES",
    "TOURNAMENT_SCHEMA",
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
    "TournamentSplit",
    "assert_required_fixtures",
    "compile_fixture",
    "load_fixtures",
    "make_tournament_split",
    "percentile",
    "percentile_triplet",
    "resolve_composition_names",
    "run_tournament",
    "selected_compositions",
    "write_run",
]
