"""Q-Route: learned per-region routing over a mechanism ladder.

The router learns *which arm is Pareto-optimal* for a region of tool-calling
state space and lets a cheaper arm keep the region whenever a frozen
non-inferiority gate says it is not meaningfully worse.  Complexity is earned
per region; simplification is automatic.  No teacher model's answers are
imitated -- the supervision is the gate's own per-region owner.

See ``AGENTS.md`` for repo conventions and ``schema.py`` for the versioned
artifact schemas (``qroute.*``).
"""

from __future__ import annotations

from .distill import RouterModel, fit_router, gate_targets, non_inferiority
from .gate import (
    LADDER,
    GateConfig,
    GateOutcome,
    GateReport,
    RegionDecision,
    compile_regions,
    evaluate_gate,
    ladder_rank,
    ladder_rung,
    render_gate_markdown,
)
from .qfunc import FEATURE_NAMES, QModel, StateFeatures, fit_q, pareto_arms, ridge_fit
from .schema import (
    GATE_REPORT_SCHEMA,
    QVALUES_SCHEMA,
    ROUTER_SCHEMA,
    RUN_SCHEMA,
    TEACHER_TABLE_SCHEMA,
    UTILITY_SCHEMA,
)
from .teacher import (
    DEFAULT_SOURCES,
    SOURCE_KINDS,
    SourceSpec,
    TeacherRow,
    TeacherTable,
    build_teacher_table,
    load_teacher_table,
    resolve_sources,
    write_teacher_table,
)
from .analysis import (
    ANALYSIS_SCHEMA,
    CellEvidence,
    analyse,
    bucket_ceiling,
    load_observations,
    realised_utility_gap,
    render_md as render_analysis_markdown,
    write_analysis,
)
from .utility import (
    OUTCOME_LABELS,
    OUTCOME_VALUES,
    UtilityConfig,
    downside_risk,
    expected_value,
    explain_utility,
    tail_risk,
    utility,
    value_distribution,
)

__all__ = [
    "ANALYSIS_SCHEMA",
    "CellEvidence",
    "DEFAULT_SOURCES",
    "FEATURE_NAMES",
    "GATE_REPORT_SCHEMA",
    "LADDER",
    "OUTCOME_LABELS",
    "OUTCOME_VALUES",
    "QVALUES_SCHEMA",
    "ROUTER_SCHEMA",
    "RUN_SCHEMA",
    "SOURCE_KINDS",
    "TEACHER_TABLE_SCHEMA",
    "UTILITY_SCHEMA",
    "GateConfig",
    "GateOutcome",
    "GateReport",
    "QModel",
    "RegionDecision",
    "RouterModel",
    "SourceSpec",
    "StateFeatures",
    "TeacherRow",
    "TeacherTable",
    "UtilityConfig",
    "analyse",
    "bucket_ceiling",
    "build_teacher_table",
    "compile_regions",
    "downside_risk",
    "evaluate_gate",
    "expected_value",
    "explain_utility",
    "fit_q",
    "fit_router",
    "gate_targets",
    "ladder_rank",
    "ladder_rung",
    "load_observations",
    "load_teacher_table",
    "non_inferiority",
    "pareto_arms",
    "realised_utility_gap",
    "render_analysis_markdown",
    "render_gate_markdown",
    "resolve_sources",
    "ridge_fit",
    "tail_risk",
    "utility",
    "value_distribution",
    "write_analysis",
    "write_teacher_table",
]
