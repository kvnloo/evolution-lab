"""Frozen decision fixtures for the local tool-calling tournament.

The fixtures are the judge-side input contract for ``tool-tournament``. They are
deliberately data, not code: ``data/tool_tournament/v1/fixtures.jsonl`` is locked
on disk and the loader only reads it. Each row is one decision point in a
synthetic work item; the harness compares *system compositions* over these rows
(see ``compositions.py``) and scores them with grouped splits (``score.py``).

Two levels of grouping exist:

* ``work_item`` — the fold boundary. Retries, branches and continuations of one
  work item must never straddle folds.
* ``task_family`` — the coarser holdout boundary (e.g. ``routing``, ``safety``).

``family`` is the *question* family (``tool_selection``, ``schema_correctness``,
…). It is descriptive metadata; the fold boundaries are ``work_item`` and
``task_family`` so no question family leaks across a fold boundary by accident.

Four fixtures are adversarial: ``permission_denied``,
``credential_required_action``, ``destructive_action`` and
``publish_send_deploy_action`` declare ``dangerous_actions``. The correct
behaviour there is that the deterministic compiler removes the action and the
gold answer is a safe alternative (or abstain); selecting a dangerous action is a
hard failure for any learned policy.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

FIXTURE_SCHEMA = "z0int.tool_tournament.fixtures.v1"
DEFAULT_VERSION = "v1"

# Tools the frozen registry knows. Anything outside this set is an unknown tool
# and must be removed by the deterministic compiler.
KNOWN_TOOLS: frozenset[str] = frozenset(
    {
        "fs.read",
        "fs.write",
        "fs.list",
        "fs.stat",
        "fs.delete_tree",
        "shell.run",
        "shell.exec",
        "code.edit",
        "code.test",
        "code.lint",
        "net.fetch",
        "net.deploy",
        "email.send",
        "email.draft",
        "db.query",
        "db.migrate",
        "search.web",
        "search.docs",
        "report.write",
        "doc.write_draft",
        "model.call",
        "model.ask_specialist",
        "task.delegate",
        "task.escalate",
        "control.stop",
        "control.continue",
        "control.retry",
        "control.escalate",
        "control.replan",
        "credential.use",
    }
)

# The issue (#23) names these fixture_ids explicitly. Keep the list frozen so a
# dropped fixture fails the suite instead of silently shrinking the battery.
REQUIRED_FIXTURE_IDS: tuple[str, ...] = (
    "one_obvious_tool",
    "two_similar_tools",
    "no_relevant_tool",
    "required_argument_missing",
    "wrong_argument_type",
    "unknown_tool",
    "tool_returns_malformed_data",
    "tool_fails",
    "tool_times_out",
    "tool_succeeds_after_retry",
    "two_independent_parallel_actions",
    "hard_dependency_a_then_b",
    "diamond_dependency_graph",
    "cheap_model_sufficient",
    "specialist_required",
    "expensive_model_unnecessary",
    "orchestrator_should_abstain",
    "jev_uncertain",
    "slm_uncertain",
    "stop_now",
    "continue_work",
    "retry_action",
    "escalate_action",
    "replan",
    "permission_denied",
    "credential_required_action",
    "destructive_action",
    "publish_send_deploy_action",
)

#: Task families reserved for the sealed holdout; they must never appear in the
#: train or dev folds.
HOLDOUT_TASK_FAMILIES: tuple[str, ...] = ("routing", "safety")

_DANGEROUS_BY_ID: dict[str, tuple[str, ...]] = {
    "permission_denied": ("fs.write",),
    "credential_required_action": ("net.deploy",),
    "destructive_action": ("fs.delete_tree",),
    "publish_send_deploy_action": ("email.send",),
}

_PY_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "int": int,
    "float": (int, float),
    "bool": bool,
    "list": list,
    "object": dict,
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_fixtures_path(version: str = DEFAULT_VERSION, root: Path | None = None) -> Path:
    return (root or repo_root()) / "data" / "tool_tournament" / version / "fixtures.jsonl"


@dataclass
class ActionCandidate:
    """One declared action in a fixture graph."""

    action_id: str
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    required_args: list[str] = field(default_factory=list)
    arg_types: dict[str, str] = field(default_factory=dict)
    requires_capabilities: list[str] = field(default_factory=list)
    requires_credentials: list[str] = field(default_factory=list)
    provides: list[str] = field(default_factory=list)
    risk_class: str = "read"
    cost_units: int = 1
    dependencies: list[str] = field(default_factory=list)
    parallelizable: bool = True
    destructive: bool = False
    publishes: bool = False
    simulated_outcome: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionCandidate":
        tool = str(data.get("tool") or data.get("action_id") or "")
        return cls(
            action_id=str(data.get("action_id") or tool),
            tool=tool,
            args=dict(data.get("args") or {}),
            required_args=[str(x) for x in (data.get("required_args") or [])],
            arg_types={str(k): str(v) for k, v in (data.get("arg_types") or {}).items()},
            requires_capabilities=[str(x) for x in (data.get("requires_capabilities") or [])],
            requires_credentials=[str(x) for x in (data.get("requires_credentials") or [])],
            provides=[str(x) for x in (data.get("provides") or [])],
            risk_class=str(data.get("risk_class") or "read"),
            cost_units=int(data.get("cost_units") or 1),
            dependencies=[str(x) for x in (data.get("dependencies") or [])],
            parallelizable=bool(data.get("parallelizable", True)),
            destructive=bool(data.get("destructive", False)),
            publishes=bool(data.get("publishes", False)),
            simulated_outcome=(str(data["simulated_outcome"]) if data.get("simulated_outcome") else None),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Fixture:
    """One frozen decision point."""

    fixture_id: str
    family: str
    work_item: str
    task_family: str
    harness: str
    state: str
    actions: list[ActionCandidate]
    rules: list[dict[str, Any]]
    granted_capabilities: list[str]
    authority: list[str]
    gold_action: str | None
    risk_class: str
    budget_units: int = 10
    satisfied: list[str] = field(default_factory=list)
    expect_abstain: bool = False
    expect_escalation: bool = False
    gold_plan: list[str] = field(default_factory=list)
    gold_model: str | None = None
    dangerous_actions: list[str] = field(default_factory=list)
    relevant_actions: list[str] = field(default_factory=list)
    parallelizable: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Fixture":
        graph = dict(data.get("graph") or {})
        actions = [ActionCandidate.from_dict(a) for a in (graph.get("actions") or [])]
        gold = data.get("gold_action")
        gold = str(gold) if gold is not None else None
        plan = [str(x) for x in (data.get("gold_plan") or ([] if gold is None else [gold]))]
        relevant = data.get("relevant_actions")
        if relevant is None:
            relevant = list(plan)
        return cls(
            fixture_id=str(data["fixture_id"]),
            family=str(data.get("family") or ""),
            work_item=str(data.get("work_item") or ""),
            task_family=str(data.get("task_family") or ""),
            harness=str(data.get("harness") or ""),
            state=str(data.get("state") or ""),
            actions=actions,
            rules=[dict(r) for r in (graph.get("rules") or [])],
            granted_capabilities=[str(x) for x in (data.get("granted_capabilities") or [])],
            authority=[str(x) for x in (data.get("authority") or [])],
            gold_action=gold,
            risk_class=str(data.get("risk_class") or "read"),
            budget_units=int(data.get("budget_units") or 10),
            satisfied=[str(x) for x in (data.get("satisfied") or [])],
            expect_abstain=bool(data.get("expect_abstain", False)),
            expect_escalation=bool(data.get("expect_escalation", False)),
            gold_plan=plan,
            gold_model=(str(data["gold_model"]) if data.get("gold_model") else None),
            dangerous_actions=[str(x) for x in (data.get("dangerous_actions") or [])],
            relevant_actions=[str(x) for x in relevant],
            parallelizable=bool(data.get("parallelizable", False)),
        )

    @property
    def action_ids(self) -> list[str]:
        return [a.action_id for a in self.actions]

    @property
    def declared_dangerous(self) -> list[str]:
        """Dangerous entries declared on the row plus the frozen by-id table."""
        declared = set(self.dangerous_actions)
        declared.update(_DANGEROUS_BY_ID.get(self.fixture_id, ()))
        return sorted(declared)

    def action(self, action_id: str) -> ActionCandidate | None:
        for a in self.actions:
            if a.action_id == action_id:
                return a
        return None

    @property
    def expected_calls(self) -> int:
        return max(1, len(self.gold_plan)) if not self.expect_abstain else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "family": self.family,
            "work_item": self.work_item,
            "task_family": self.task_family,
            "harness": self.harness,
            "state": self.state,
            "graph": {"actions": [a.to_dict() for a in self.actions], "rules": list(self.rules)},
            "granted_capabilities": list(self.granted_capabilities),
            "authority": list(self.authority),
            "gold_action": self.gold_action,
            "risk_class": self.risk_class,
            "budget_units": self.budget_units,
            "satisfied": list(self.satisfied),
            "expect_abstain": self.expect_abstain,
            "expect_escalation": self.expect_escalation,
            "gold_plan": list(self.gold_plan),
            "gold_model": self.gold_model,
            "dangerous_actions": list(self.dangerous_actions),
            "relevant_actions": list(self.relevant_actions),
            "parallelizable": self.parallelizable,
        }


def value_matches_type(value: Any, type_name: str) -> bool:
    expected = _PY_TYPES.get(type_name)
    if expected is None:
        return True
    if isinstance(value, bool) and expected is not bool:
        return False
    return isinstance(value, expected)


def load_fixture_rows(path: Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_fixtures(path: Path | str | None = None) -> list[Fixture]:
    """Load the frozen fixture suite, preserving on-disk order."""
    target = Path(path) if path is not None else default_fixtures_path()
    fixtures = [Fixture.from_dict(row) for row in load_fixture_rows(target)]
    seen: set[str] = set()
    for f in fixtures:
        if f.fixture_id in seen:
            raise ValueError(f"duplicate fixture_id: {f.fixture_id}")
        seen.add(f.fixture_id)
        if not f.work_item:
            raise ValueError(f"fixture {f.fixture_id} has no work_item (grouped splits need it)")
        if not f.task_family:
            raise ValueError(f"fixture {f.fixture_id} has no task_family (holdout needs it)")
    return fixtures


def assert_required_fixtures(fixtures: Iterable[Fixture]) -> None:
    present = {f.fixture_id for f in fixtures}
    missing = [fid for fid in REQUIRED_FIXTURE_IDS if fid not in present]
    if missing:
        raise ValueError(f"frozen fixture suite is missing: {', '.join(missing)}")


def fixtures_by_id(fixtures: Iterable[Fixture]) -> dict[str, Fixture]:
    return {f.fixture_id: f for f in fixtures}


def summarize_fixtures(fixtures: list[Fixture]) -> dict[str, Any]:
    task_families: dict[str, int] = {}
    families: dict[str, int] = {}
    work_items: dict[str, set[str]] = {}
    for f in fixtures:
        task_families[f.task_family] = task_families.get(f.task_family, 0) + 1
        families[f.family] = families.get(f.family, 0) + 1
        work_items.setdefault(f.task_family, set()).add(f.work_item)
    return {
        "schema": FIXTURE_SCHEMA,
        "fixture_count": len(fixtures),
        "work_item_count": len({f.work_item for f in fixtures}),
        "task_families": dict(sorted(task_families.items())),
        "work_items_per_task_family": {k: len(v) for k, v in sorted(work_items.items())},
        "families": dict(sorted(families.items())),
        "holdout_task_families": list(HOLDOUT_TASK_FAMILIES),
    }
