"""Teacher table: real measured per-``(state, arm)`` outcomes.

Q-Route does **not** distil a teacher model's answers.  The "teacher" here is the
measured outcome of every arm on every decision state: how often it succeeded,
how long it took, and whether it exposed or selected a dangerous action.  Those
rows are aggregated from real ``raw.jsonl`` runs (compositions, bounded-choice
evals, orchestration battery) into a compact, deterministic table.

The state identity is the fixture/scenario id for now.  The table keeps a
``signals`` block per state so a later feature-based key can replace the id
without touching the aggregation contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..tool_tournament.score import percentile
from .schema import TEACHER_TABLE_SCHEMA

SOURCE_KINDS: tuple[str, ...] = ("composition", "bounded", "orchestration")

#: Documented defaults.  Override with ``--sources kind:path,kind:path``.
DEFAULT_SOURCES: tuple["SourceSpec", ...] = ()  # populated after SourceSpec below

#: Eliminations the compiler performs for *safety* reasons.  These are the
#: actions a compiler-first arm would have exposed had it not stripped them, so
#: they seed the per-state declared-dangerous set for bounded rows (which carry
#: no explicit ``dangerous_actions`` field).
_SAFETY_REASON_MARKERS: tuple[str, ...] = (
    "risk_class",
    "destructive",
    "permission",
    "missing capability",
    "credential",
    "publish",
    "denied",
    "dangerous",
    "forbidden",
)


@dataclass(frozen=True)
class SourceSpec:
    """A ``raw.jsonl`` path plus the row schema it carries."""

    path: Path
    kind: str
    label: str = ""

    def __post_init__(self) -> None:
        if self.kind not in SOURCE_KINDS:
            raise ValueError(
                f"unknown source kind: {self.kind!r} (choices: {', '.join(SOURCE_KINDS)})"
            )

    def to_dict(self) -> dict[str, Any]:
        return {"path": str(self.path), "kind": self.kind, "label": self.label or self.path.name}


def _default_source_specs() -> tuple[SourceSpec, ...]:
    work = Path("/mnt/zer0models/workspace/zer0/oss/.work")
    return (
        SourceSpec(work / "compositions" / "raw.jsonl", "composition", "compositions A-F + SUB"),
        SourceSpec(work / "eval-llamacpp" / "raw.jsonl", "bounded", "bounded-choice (llama.cpp)"),
        SourceSpec(work / "eval-thinking" / "raw.jsonl", "bounded", "bounded-choice (thinking)"),
        SourceSpec(work / "orch-v1" / "raw.jsonl", "orchestration", "multi-turn orchestration"),
    )


DEFAULT_SOURCES = _default_source_specs()


def infer_source_kind(path: Path | str) -> str:
    """Best-effort kind for a path supplied without an explicit ``kind:``."""
    name = str(path).lower()
    if "orch" in name:
        return "orchestration"
    if "composition" in name:
        return "composition"
    return "bounded"


def resolve_sources(value: str | None) -> list[SourceSpec]:
    """Parse ``--sources`` (``kind:path`` comma list) or fall back to defaults."""
    if not value:
        return list(DEFAULT_SOURCES)
    specs: list[SourceSpec] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            kind, _, raw_path = part.partition(":")
            kind = kind.strip().lower()
            raw_path = raw_path.strip()
        else:
            raw_path = part
            kind = infer_source_kind(raw_path)
        specs.append(SourceSpec(Path(raw_path), kind))
    return specs


# --------------------------------------------------------------------------
# Rows
# --------------------------------------------------------------------------


@dataclass
class TeacherRow:
    """Aggregated outcome for one ``(state_key, arm)`` pair."""

    state_key: str
    arm: str
    n: int
    success: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_mean_ms: float
    dangerous_rate: float
    invalid_rate: float
    abstain_rate: float
    compiler: bool
    exposed_dangerous: bool
    kind: str = ""
    schema: str = TEACHER_TABLE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_key": self.state_key,
            "arm": self.arm,
            "n": int(self.n),
            "success": float(self.success),
            "latency_p50_ms": float(self.latency_p50_ms),
            "latency_p95_ms": float(self.latency_p95_ms),
            "latency_mean_ms": float(self.latency_mean_ms),
            "dangerous_rate": float(self.dangerous_rate),
            "invalid_rate": float(self.invalid_rate),
            "abstain_rate": float(self.abstain_rate),
            "compiler": bool(self.compiler),
            "exposed_dangerous": bool(self.exposed_dangerous),
            "kind": self.kind,
            "schema": self.schema,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TeacherRow":
        return cls(
            state_key=str(data["state_key"]),
            arm=str(data["arm"]),
            n=int(data["n"]),
            success=float(data["success"]),
            latency_p50_ms=float(data["latency_p50_ms"]),
            latency_p95_ms=float(data["latency_p95_ms"]),
            latency_mean_ms=float(data["latency_mean_ms"]),
            dangerous_rate=float(data["dangerous_rate"]),
            invalid_rate=float(data["invalid_rate"]),
            abstain_rate=float(data.get("abstain_rate", 0.0)),
            compiler=bool(data.get("compiler", False)),
            exposed_dangerous=bool(data.get("exposed_dangerous", False)),
            kind=str(data.get("kind", "")),
            schema=str(data.get("schema", TEACHER_TABLE_SCHEMA)),
        )


@dataclass
class TeacherTable:
    """Deterministic, sorted table of aggregated arm outcomes."""

    rows: list[TeacherRow] = field(default_factory=list)
    signals: dict[str, dict[str, float]] = field(default_factory=dict)
    sources: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    schema: str = TEACHER_TABLE_SCHEMA

    # -- sorted accessors --------------------------------------------------
    def sorted_rows(self) -> list[TeacherRow]:
        return sorted(self.rows, key=lambda r: (r.state_key, r.arm))

    def to_rows(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.sorted_rows()]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "rows": self.to_rows(),
            "signals": {k: dict(v) for k, v in sorted(self.signals.items())},
            "sources": list(self.sources),
            "skipped": list(self.skipped),
            "n_states": len(self.states()),
            "n_arms": len(self.arms()),
            "n_rows": len(self.rows),
        }

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[dict[str, Any]],
        *,
        signals: dict[str, dict[str, float]] | None = None,
        sources: list[dict[str, Any]] | None = None,
        skipped: list[dict[str, Any]] | None = None,
    ) -> "TeacherTable":
        parsed = [TeacherRow.from_dict(r) for r in rows]
        return cls(
            rows=parsed,
            signals=dict(signals or {}),
            sources=list(sources or []),
            skipped=list(skipped or []),
        )

    # -- lookups -----------------------------------------------------------
    def states(self) -> list[str]:
        return sorted({r.state_key for r in self.rows})

    def arms(self) -> list[str]:
        return sorted({r.arm for r in self.rows})

    def rows_for_state(self, state_key: str) -> list[TeacherRow]:
        return sorted(
            (r for r in self.rows if r.state_key == state_key), key=lambda r: r.arm
        )

    def row_for(self, state_key: str, arm: str) -> TeacherRow | None:
        for r in self.rows:
            if r.state_key == state_key and r.arm == arm:
                return r
        return None

    def row_index(self) -> dict[tuple[str, str], TeacherRow]:
        return {(r.state_key, r.arm): r for r in self.rows}


# --------------------------------------------------------------------------
# Raw-row normalisation
# --------------------------------------------------------------------------


def _as_bool(value: Any) -> bool:
    return bool(value) if value is not None else False


def _action_id_of(entry: Any) -> str | None:
    if isinstance(entry, dict):
        for key in ("action_id", "action", "id"):
            if entry.get(key):
                return str(entry[key])
        return None
    if isinstance(entry, str):
        return entry
    return None


def _dangerous_from_eliminated(eliminated: Any) -> set[str]:
    """Safety-motivated eliminations seed the declared-dangerous set."""
    found: set[str] = set()
    if not isinstance(eliminated, list):
        return found
    for entry in eliminated:
        reason = ""
        if isinstance(entry, dict):
            reason = str(entry.get("reason") or "")
        if not reason or not any(marker in reason.lower() for marker in _SAFETY_REASON_MARKERS):
            continue
        action = _action_id_of(entry)
        if action:
            found.add(action)
    return found


@dataclass
class _Observation:
    state_key: str
    arm: str
    kind: str
    success: bool
    latency_ms: float
    dangerous: bool
    invalid: bool
    abstain: bool
    legal_ids: frozenset[str]
    has_legal: bool
    declared_dangerous: frozenset[str]
    candidate_action_count: int | None
    expect_abstain: bool | None


def _normalise(row: dict[str, Any], kind: str) -> _Observation | None:
    if kind == "composition":
        state_key = row.get("fixture_id")
        arm = row.get("composition")
        success = _as_bool(row.get("trajectory_correct"))
    elif kind == "orchestration":
        state_key = row.get("scenario_id")
        arm = row.get("backend")
        success = _as_bool(row.get("solved"))
    else:  # bounded
        state_key = row.get("fixture_id")
        arm = row.get("backend")
        success = _as_bool(row.get("trajectory_correct"))
    if not state_key or not arm:
        return None

    legal = row.get("legal_ids")
    has_legal = isinstance(legal, list)
    legal_set = frozenset(str(x) for x in legal) if has_legal else frozenset()

    declared: set[str] = set()
    for entry in row.get("dangerous_actions") or []:
        if entry:
            declared.add(str(entry))
    declared |= _dangerous_from_eliminated(row.get("eliminated"))

    if kind == "orchestration":
        dangerous = False
        invalid = bool(int(row.get("invalid_calls") or 0) > 0)
        abstain = False
        candidate: int | None = None
        expect_abstain: bool | None = None
    else:
        dangerous = _as_bool(row.get("dangerous_selected"))
        invalid = _as_bool(row.get("invalid_call"))
        abstain = _as_bool(row.get("abstained"))
        try:
            candidate = int(row["candidate_action_count"]) if row.get("candidate_action_count") is not None else None
        except (TypeError, ValueError):
            candidate = None
        expect_abstain = (
            _as_bool(row.get("expect_abstain")) if "expect_abstain" in row else None
        )

    try:
        latency = float(row.get("latency_ms") or 0.0)
    except (TypeError, ValueError):
        latency = 0.0

    return _Observation(
        state_key=str(state_key),
        arm=str(arm),
        kind=kind,
        success=success,
        latency_ms=latency,
        dangerous=dangerous,
        invalid=invalid,
        abstain=abstain,
        legal_ids=legal_set,
        has_legal=has_legal,
        declared_dangerous=frozenset(declared),
        candidate_action_count=candidate,
        expect_abstain=expect_abstain,
    )


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    bad = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if isinstance(obj, dict):
                rows.append(obj)
            else:
                bad += 1
    return rows, bad


def load_source_rows(spec: SourceSpec) -> tuple[list[dict[str, Any]], int]:
    """Read one ``raw.jsonl`` source.  Caller decides what to do when missing."""
    return _read_jsonl(Path(spec.path))


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _rate(values: list[bool]) -> float:
    return float(sum(1 for v in values if v) / len(values)) if values else 0.0


def _name_says_compiler(arm: str) -> bool:
    """Frozen arm-name convention used when the data carries no safety signal."""
    lowered = arm.lower()
    if "alone" in lowered:
        return False
    return "compiler" in lowered


def build_teacher_table(sources: Iterable[SourceSpec]) -> TeacherTable:
    """Aggregate real ``raw.jsonl`` sources into a :class:`TeacherTable`.

    Missing or unusable sources are recorded in ``skipped`` and never raise:
    a partially-populated table is more useful than a crash mid-pipeline.
    """
    specs = [s if isinstance(s, SourceSpec) else SourceSpec(Path(s[0]), s[1]) for s in sources]

    observations: list[_Observation] = []
    source_log: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for spec in specs:
        path = Path(spec.path)
        if not path.exists():
            entry = {**spec.to_dict(), "status": "missing", "rows": 0}
            source_log.append({**entry, "reason": "file does not exist"})
            skipped.append({**entry, "reason": "file does not exist"})
            continue
        try:
            raw_rows, bad = load_source_rows(spec)
        except OSError as exc:  # unreadable file must not kill the pipeline
            entry = {**spec.to_dict(), "status": "unreadable", "rows": 0}
            source_log.append({**entry, "reason": str(exc)})
            skipped.append({**entry, "reason": str(exc)})
            continue
        normalised = [obs for obs in (_normalise(r, spec.kind) for r in raw_rows) if obs is not None]
        observations.extend(normalised)
        source_log.append(
            {
                **spec.to_dict(),
                "status": "ok" if normalised else "empty",
                "rows": len(normalised),
                "malformed_rows": bad,
            }
        )
        if not normalised:
            skipped.append({**spec.to_dict(), "reason": "no usable rows", "rows": 0})

    # Pass 1: per-state declared-dangerous universe (explicit field or safety removals).
    declared_by_state: dict[str, set[str]] = {}
    for obs in observations:
        declared_by_state.setdefault(obs.state_key, set()).update(obs.declared_dangerous)

    # Pass 2: group by (state, arm).
    grouped: dict[tuple[str, str], list[_Observation]] = {}
    for obs in observations:
        grouped.setdefault((obs.state_key, obs.arm), []).append(obs)

    rows: list[TeacherRow] = []
    for (state_key, arm), obs_list in grouped.items():
        declared = declared_by_state.get(state_key, set())
        exposed = False
        for obs in obs_list:
            if obs.has_legal and obs.legal_ids & declared:
                exposed = True
        exposed = exposed or any(obs.dangerous for obs in obs_list)
        kinds = sorted({obs.kind for obs in obs_list})
        kind = kinds[0]

        has_legal = any(obs.has_legal for obs in obs_list)
        if declared and has_legal:
            compiler = not exposed
        elif kind == "bounded":
            compiler = True
        elif kind == "orchestration":
            compiler = False
        else:
            compiler = _name_says_compiler(arm)

        latencies = [obs.latency_ms for obs in obs_list]
        rows.append(
            TeacherRow(
                state_key=state_key,
                arm=arm,
                n=len(obs_list),
                success=_rate([obs.success for obs in obs_list]),
                latency_p50_ms=float(percentile(latencies, 50.0)),
                latency_p95_ms=float(percentile(latencies, 95.0)),
                latency_mean_ms=_mean(latencies),
                dangerous_rate=_rate([obs.dangerous for obs in obs_list]),
                invalid_rate=_rate([obs.invalid for obs in obs_list]),
                abstain_rate=_rate([obs.abstain for obs in obs_list]),
                compiler=compiler,
                exposed_dangerous=exposed,
                kind=kind,
            )
        )

    # Per-state feature signals (the door to a feature-based state key).
    signals: dict[str, dict[str, float]] = {}
    for state_key in sorted({obs.state_key for obs in observations}):
        state_obs = [obs for obs in observations if obs.state_key == state_key]
        candidate_values = [
            obs.candidate_action_count
            for obs in state_obs
            if obs.candidate_action_count is not None
        ]
        if candidate_values:
            candidate_count = max(candidate_values)
        else:
            candidate_count = max((len(obs.legal_ids) for obs in state_obs if obs.has_legal), default=0)
        declared = declared_by_state.get(state_key, set())
        exposed_any = any(obs.has_legal and obs.legal_ids & declared for obs in state_obs)
        signals[state_key] = {
            "candidate_action_count": float(candidate_count),
            "exposed_dangerous": 1.0 if exposed_any else 0.0,
            "expect_abstain": 1.0 if any(obs.expect_abstain for obs in state_obs) else 0.0,
        }

    return TeacherTable(
        rows=sorted(rows, key=lambda r: (r.state_key, r.arm)),
        signals=signals,
        sources=source_log,
        skipped=skipped,
    )


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def write_teacher_table(table: TeacherTable, out_dir: Path | str) -> Path:
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "teacher.json").write_text(
        json.dumps(table.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (dest / "teacher.jsonl").open("w", encoding="utf-8") as fh:
        for row in table.to_rows():
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    return dest / "teacher.json"


def load_teacher_table(path: Path | str) -> TeacherTable:
    """Load ``teacher.json`` (or a ``teacher.jsonl``) written by ``q-route build``."""
    target = Path(path)
    if target.is_dir():
        target = target / "teacher.json"
    data = json.loads(target.read_text(encoding="utf-8"))
    return TeacherTable.from_rows(
        data.get("rows", []),
        signals=data.get("signals", {}),
        sources=data.get("sources", []),
        skipped=data.get("skipped", []),
    )


__all__ = [
    "DEFAULT_SOURCES",
    "SOURCE_KINDS",
    "SourceSpec",
    "TeacherRow",
    "TeacherTable",
    "build_teacher_table",
    "infer_source_kind",
    "load_source_rows",
    "load_teacher_table",
    "resolve_sources",
    "write_teacher_table",
]
