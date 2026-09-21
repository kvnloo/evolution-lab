"""Local tool-calling tournament: frozen fixtures, system compositions, scorer.

See issue kvnloo/evolution-lab#23. The package compares *system compositions*
(compiler + orchestrator + local models + specialists), not single checkpoints,
over grouped, non-random splits of a frozen decision battery.
"""

from __future__ import annotations

from .fixtures import (
    HOLDOUT_TASK_FAMILIES,
    REQUIRED_FIXTURE_IDS,
    Fixture,
    assert_required_fixtures,
    load_fixtures,
)

__all__ = [
    "HOLDOUT_TASK_FAMILIES",
    "REQUIRED_FIXTURE_IDS",
    "Fixture",
    "assert_required_fixtures",
    "load_fixtures",
]
