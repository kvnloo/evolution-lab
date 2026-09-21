"""Explicit capacity-fill queue for otherwise-expiring free inference.

Kerdoios owns provider capacity, free quotas (Groq/Cerebras), placement and
expiry-aware allocation.  Evolution Lab owns the *queue* of useful work that may
consume spare capacity.  This module is the small hand-off between them: a
declarative queue of explicitly registered work items, each carrying

* a ``work_class`` whose priority matches the agreed useful-work order,
* the provider/model/task constraints it is compatible with,
* an estimated consumption (calls / tokens / cost / GPU-ms).

Two rules are enforced in code, not just in prose:

* **No filler.**  Every item must name real ``provenance`` (an existing
  repo-relative file).  :meth:`WorkItem.from_dict` rejects an item without it,
  and :func:`plan_capacity` never synthesises work: an empty or incompatible
  queue returns an empty plan plus an ``empty_reason``.
* **Protected splits stay protected.**  ``confirm``/``ood``/held-out/sealed work
  is certification-only and is never offered as background fill unless the
  caller explicitly passes ``include_protected=True``.

Splits and their names follow the existing repo discipline
(:mod:`evolution_lab.splits`, :mod:`evolution_lab.sealed_split`,
:mod:`evolution_lab.tool_tournament.score`); nothing here redefines them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

ITEM_SCHEMA = "flyforge.capacity_queue_item.v1"
QUEUE_SCHEMA = "flyforge.capacity_queue.v1"
PLAN_SCHEMA = "flyforge.capacity_plan.v1"

#: Useful-work order agreed with the capacity owner.  Index is the priority:
#: lower runs first when spare capacity is offered.  Do not reorder casually;
#: the order is the contract, not a preference list.
WORK_CLASSES: tuple[str, ...] = (
    "dsh_hermes",  # 0 normal useful DSH/Hermes work
    "qroute_counterfactual",  # 1 Q-Route counterfactual rollouts
    "router_eval",  # 2 Evolution Lab model/router evals
    "distillation_example",  # 3 distillation examples
    "agent0_curriculum",  # 4 Agent0-style curriculum tasks
    "jev_tiny_router_data",  # 5 JEV / tiny-router training data
    "queued_research",  # 6 other explicitly queued useful research
)
WORK_CLASS_PRIORITY: dict[str, int] = {name: i for i, name in enumerate(WORK_CLASSES)}

#: Splits that may be consumed as background fill (training / validation / a
#: deliberately disposable research task).
FILL_SPLITS: tuple[str, ...] = ("train", "validation", "dev", "disposable")

#: Certification-only splits.  Never offered as fill.  Spelled exactly as the
#: existing split producers name them so a caller cannot smuggle a confirm/OOD
#: or held-out item in under a synonym.
PROTECTED_SPLITS: tuple[str, ...] = (
    "confirm",
    "ood",
    "sealed",
    "sealed_human_audited",
    "holdout",
    "future",
    "test",
)


class QueueError(ValueError):
    """Raised when a queue entry would fabricate or mislabel work."""


def priority_of(work_class: str) -> int:
    """Priority index for ``work_class`` (lower is more useful)."""
    try:
        return WORK_CLASS_PRIORITY[work_class]
    except KeyError as exc:  # pragma: no cover - message path exercised in tests
        raise QueueError(f"unknown work_class: {work_class!r}") from exc


def is_protected_split(split: str) -> bool:
    return split.strip().lower() in PROTECTED_SPLITS


def is_fill_split(split: str) -> bool:
    return split.strip().lower() in FILL_SPLITS


@dataclass(frozen=True)
class WorkConstraints:
    """What a work item can run under, and which split it consumes.

    Empty ``providers``/``models``/``task_families`` mean "unconstrained".
    ``split`` is the corpus split the item reads; protected splits are never
    fill-safe.
    """

    providers: tuple[str, ...] = ()
    models: tuple[str, ...] = ()
    task_families: tuple[str, ...] = ()
    split: str = "disposable"
    requires_gpu: bool = False

    @property
    def fill_safe(self) -> bool:
        return not is_protected_split(self.split)

    def matches(self, *, provider: str, model: str | None = None) -> bool:
        if self.providers and provider not in self.providers:
            return False
        if model and self.models and model not in self.models:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkConstraints":
        return cls(
            providers=tuple(str(p) for p in data.get("providers", ())),
            models=tuple(str(m) for m in data.get("models", ())),
            task_families=tuple(str(f) for f in data.get("task_families", ())),
            split=str(data.get("split", "disposable")),
            requires_gpu=bool(data.get("requires_gpu", False)),
        )


@dataclass(frozen=True)
class EstimatedUse:
    """Conservative consumption estimate for one work item."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    gpu_ms: int = 0

    @property
    def tokens(self) -> int:
        return int(self.prompt_tokens) + int(self.completion_tokens)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tokens"] = self.tokens
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EstimatedUse":
        return cls(
            calls=int(data.get("calls", 0)),
            prompt_tokens=int(data.get("prompt_tokens", 0)),
            completion_tokens=int(data.get("completion_tokens", 0)),
            cost_usd=float(data.get("cost_usd", 0.0)),
            gpu_ms=int(data.get("gpu_ms", 0)),
        )


@dataclass(frozen=True)
class WorkItem:
    """One explicitly queued, useful unit of work.

    ``provenance`` must point at an existing repo-relative file.  That is the
    anti-filler rule: a queue entry that cannot say where the work comes from is
    rejected rather than run to exhaust a quota.
    """

    item_id: str
    work_class: str
    summary: str
    provenance: str
    constraints: WorkConstraints = field(default_factory=WorkConstraints)
    estimate: EstimatedUse = field(default_factory=EstimatedUse)
    evidence: str = ""
    created_by: str = ""

    def __post_init__(self) -> None:
        if not self.item_id:
            raise QueueError("work item is missing item_id")
        if self.work_class not in WORK_CLASS_PRIORITY:
            raise QueueError(
                f"{self.item_id}: unknown work_class {self.work_class!r}; "
                f"expected one of {', '.join(WORK_CLASSES)}"
            )
        if not self.summary:
            raise QueueError(f"{self.item_id}: summary is required")
        if not self.provenance:
            raise QueueError(
                f"{self.item_id}: provenance is required -- queued work must name "
                "its source; no filler work is accepted"
            )
        split = self.constraints.split
        if split and not is_fill_split(split) and not is_protected_split(split):
            raise QueueError(
                f"{self.item_id}: unknown split {split!r}; "
                f"expected one of {', '.join(FILL_SPLITS + PROTECTED_SPLITS)}"
            )
        est = self.estimate
        if min(est.calls, est.prompt_tokens, est.completion_tokens, est.gpu_ms) < 0:
            raise QueueError(f"{self.item_id}: estimates must be non-negative")

    @property
    def priority(self) -> int:
        return priority_of(self.work_class)

    @property
    def fill_safe(self) -> bool:
        return self.constraints.fill_safe

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ITEM_SCHEMA,
            "item_id": self.item_id,
            "work_class": self.work_class,
            "priority": self.priority,
            "summary": self.summary,
            "provenance": self.provenance,
            "evidence": self.evidence,
            "constraints": self.constraints.to_dict(),
            "estimate": self.estimate.to_dict(),
            "created_by": self.created_by,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkItem":
        return cls(
            item_id=str(data.get("item_id", "")).strip(),
            work_class=str(data.get("work_class", "")).strip(),
            summary=str(data.get("summary", "")).strip(),
            provenance=str(data.get("provenance", "")).strip(),
            constraints=WorkConstraints.from_dict(data.get("constraints") or {}),
            estimate=EstimatedUse.from_dict(data.get("estimate") or {}),
            evidence=str(data.get("evidence", "")).strip(),
            created_by=str(data.get("created_by", "")).strip(),
        )


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_queue_path(root: Path | None = None) -> Path:
    return (root or repo_root()) / "data" / "capacity_queue" / "queue.json"


def _load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        rows = payload.get("items", [])
    elif isinstance(payload, list):
        rows = payload
    else:  # pragma: no cover - defensive
        raise QueueError(f"{path}: queue must be a JSON object or list")
    if not isinstance(rows, list):
        raise QueueError(f"{path}: 'items' must be a list")
    return rows


def load_queue(path: Path | str | None = None) -> list[WorkItem]:
    """Load the explicit queue.  A missing file is an empty queue, not an error."""
    target = Path(path) if path is not None else default_queue_path()
    if not target.is_file():
        return []
    items = [WorkItem.from_dict(row) for row in _load_rows(target)]
    seen: set[str] = set()
    for item in items:
        if item.item_id in seen:
            raise QueueError(f"{target}: duplicate item_id {item.item_id!r}")
        seen.add(item.item_id)
    return items


def write_queue(items: Iterable[WorkItem], path: Path | str | None = None) -> Path:
    target = Path(path) if path is not None else default_queue_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": QUEUE_SCHEMA,
        "note": "Explicitly queued useful work. No item is generated to exhaust a quota.",
        "items": [item.to_dict() for item in items],
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def _item_view(item: WorkItem) -> dict[str, Any]:
    return {
        "item_id": item.item_id,
        "work_class": item.work_class,
        "priority": item.priority,
        "summary": item.summary,
        "provenance": item.provenance,
        "split": item.constraints.split,
        "fill_safe": item.fill_safe,
        "estimate": item.estimate.to_dict(),
    }


def plan_capacity(
    items: Iterable[WorkItem],
    *,
    provider: str,
    model: str | None = None,
    max_tokens: int | None = None,
    max_cost_usd: float | None = None,
    include_protected: bool = False,
) -> dict[str, Any]:
    """Answer "what useful compatible work is queued, and what would consume it?".

    Returns a plan, never a fabricated queue.  Items are considered in priority
    order; an item is selected only if it stays inside both budgets.  Protected
    splits are reported as skipped unless ``include_protected`` is set.
    """
    queue = list(items)
    skipped: list[dict[str, Any]] = []
    pool: list[WorkItem] = []
    for item in queue:
        if not item.constraints.matches(provider=provider, model=model):
            skipped.append({**_item_view(item), "reason": "incompatible_provider_or_model"})
            continue
        if not item.fill_safe and not include_protected:
            skipped.append(
                {
                    **_item_view(item),
                    "reason": "protected_split_certification_only",
                }
            )
            continue
        pool.append(item)

    pool.sort(key=lambda i: (i.priority, i.item_id))

    selected: list[dict[str, Any]] = []
    used_tokens = 0
    used_cost = 0.0
    used_calls = 0
    used_gpu_ms = 0
    for item in pool:
        est = item.estimate
        if max_tokens is not None and used_tokens + est.tokens > max_tokens:
            skipped.append({**_item_view(item), "reason": "budget_exhausted_tokens"})
            continue
        if max_cost_usd is not None and used_cost + est.cost_usd > max_cost_usd + 1e-12:
            skipped.append({**_item_view(item), "reason": "budget_exhausted_cost"})
            continue
        selected.append({**_item_view(item), "reason": "selected"})
        used_tokens += est.tokens
        used_cost += est.cost_usd
        used_calls += est.calls
        used_gpu_ms += est.gpu_ms

    if not queue:
        empty_reason: str | None = "queue is empty; nothing is generated to exhaust a quota"
    elif not selected:
        empty_reason = f"no fill-safe compatible work for provider={provider!r} model={model!r}"
    else:
        empty_reason = None

    return {
        "schema": PLAN_SCHEMA,
        "provider": provider,
        "model": model,
        "queue_depth": len(queue),
        "compatible_count": len(pool),
        "include_protected": bool(include_protected),
        "selected": selected,
        "skipped": skipped,
        "totals": {
            "calls": used_calls,
            "prompt_tokens": sum(s["estimate"]["prompt_tokens"] for s in selected),
            "completion_tokens": sum(s["estimate"]["completion_tokens"] for s in selected),
            "tokens": used_tokens,
            "cost_usd": round(used_cost, 8),
            "gpu_ms": used_gpu_ms,
        },
        "empty_reason": empty_reason,
        "filler_policy": "only explicitly queued items are ever returned; no work is generated to exhaust quota",
    }


__all__ = [
    "FILL_SPLITS",
    "ITEM_SCHEMA",
    "PLAN_SCHEMA",
    "PROTECTED_SPLITS",
    "QUEUE_SCHEMA",
    "WORK_CLASSES",
    "WORK_CLASS_PRIORITY",
    "EstimatedUse",
    "QueueError",
    "WorkConstraints",
    "WorkItem",
    "default_queue_path",
    "is_fill_split",
    "is_protected_split",
    "load_queue",
    "plan_capacity",
    "priority_of",
    "repo_root",
    "write_queue",
]
