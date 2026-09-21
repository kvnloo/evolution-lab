"""Schema-string constants for Q-Route.

Naming follows the repo convention already used by ``z0int.*`` and
``flyforge.*`` schemas: a dotted, versioned product name.  Every artifact
written by ``q-route`` stamps one of these strings so downstream consumers can
branch on the version instead of guessing the shape.
"""

from __future__ import annotations

TEACHER_TABLE_SCHEMA = "qroute.teacher_table.v1"
UTILITY_SCHEMA = "qroute.utility.v1"
QVALUES_SCHEMA = "qroute.qvalues.v1"
ROUTER_SCHEMA = "qroute.router.v1"
GATE_REPORT_SCHEMA = "qroute.gate_report.v1"
RUN_SCHEMA = "qroute.run.v1"

SCHEMAS: tuple[str, ...] = (
    TEACHER_TABLE_SCHEMA,
    UTILITY_SCHEMA,
    QVALUES_SCHEMA,
    ROUTER_SCHEMA,
    GATE_REPORT_SCHEMA,
    RUN_SCHEMA,
)

__all__ = [
    "GATE_REPORT_SCHEMA",
    "QVALUES_SCHEMA",
    "ROUTER_SCHEMA",
    "RUN_SCHEMA",
    "SCHEMAS",
    "TEACHER_TABLE_SCHEMA",
    "UTILITY_SCHEMA",
]
