"""Phase 2 preparation: canonical episodes, supervision labels, and frozen splits.

Phase 1B ends with *inputs* for Phase 2 (Episode Compiler + Automatic
Supervision).  This module is that preparation and nothing more: it compiles
measured receipts into episodes, assigns a supervision label under an explicit
hierarchy, and produces chronological/family splits.  **It trains nothing.**

Two rules from the brief are enforced in code, not in prose:

1. **The label hierarchy is closed.**  ``gold`` comes only from a deterministic
   verifier; ``silver`` from an observed trajectory that was not immediately
   corrected; ``teacher`` from a strong model or a bounded scorer's soft label.
   :func:`Episode.__post_init__` rejects any episode that claims a verified
   outcome without a deterministic verification, which is what "never turn
   synthetic/model preference into ``verified_success=true``" means concretely.
2. **Teacher inference is selective.**  :func:`selective_teacher_candidates`
   nominates only episodes that are unlabelled, ambiguous, disagreed, uncertain
   or otherwise high-information.  Labelling everything with a bigger model is
   exactly the cost blow-up the architecture exists to avoid.

Splits follow the brief's proportions (oldest ~70% train, next ~15% validation,
newest ~15% confirm), with whole task/domain families held out for OOD and a
small sealed human-audited set that is never trained on.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

EPISODE_SCHEMA = "qroute.episode.v1"
SPLIT_SCHEMA = "qroute.phase2_splits.v1"
SELECTION_SCHEMA = "qroute.teacher_selection.v1"

#: The only label provenances the hierarchy admits.
LABEL_SOURCES: tuple[str, ...] = ("gold", "silver", "teacher")

#: Deterministic verifiers that are allowed to mint a gold label.  A free-form
#: model opinion is not on this list and must never be added to it casually.
GOLD_VERIFIERS: tuple[str, ...] = (
    "fixture_gold",
    "unit_test",
    "compiler_contract",
    "deterministic_check",
    "outcome_success",
)


class EpisodeError(ValueError):
    """An episode violated the supervision contract."""


@dataclass
class Episode:
    """One decision, with everything Phase 2 needs to supervise a router."""

    episode_id: str
    # --- context / state
    state_before: dict[str, Any]
    available_evidence: list[dict[str, Any]]
    legal_actions: list[str]
    candidate_models: list[str]
    # --- the decision
    chosen_action: str | None
    chosen_model: str | None
    teacher_distribution: dict[str, float] | None
    observation_before: Any
    action: dict[str, Any]
    observation_after: Any
    state_after: dict[str, Any] | None
    # --- physical accounting (recorded, never reinterpreted: Tokenomics owns it)
    latencies: dict[str, Any]
    resources: dict[str, Any]
    # --- outcome / verification
    outcome: dict[str, Any]
    verification: dict[str, Any]
    # --- provenance
    source: dict[str, Any]
    timestamp: str
    privacy_class: str
    label_source: str
    label_confidence: float | None
    verified_success: bool | None = None
    task_family: str = ""
    task_id: str = ""
    session_id: str = ""
    schema: str = EPISODE_SCHEMA
    extensions: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.label_source not in LABEL_SOURCES:
            raise EpisodeError(
                f"unknown label_source {self.label_source!r}; "
                f"choices: {', '.join(LABEL_SOURCES)}"
            )
        if self.label_source != "gold" and self.verified_success is not None:
            raise EpisodeError(
                "verified_success may only be set from a deterministic verifier; "
                f"label_source={self.label_source!r} cannot claim it"
            )
        if self.label_source == "gold":
            kind = str((self.verification or {}).get("kind") or "")
            if kind not in GOLD_VERIFIERS:
                raise EpisodeError(
                    f"gold label requires a deterministic verifier, got {kind!r}; "
                    f"allowed: {', '.join(GOLD_VERIFIERS)}"
                )
        if not self.episode_id:
            raise EpisodeError("episode_id must be nonempty")

    def to_dict(self) -> dict[str, Any]:
        out = {
            "schema": self.schema,
            "episode_id": self.episode_id,
            "state_before": self.state_before,
            "available_evidence": self.available_evidence,
            "legal_actions": self.legal_actions,
            "candidate_models": self.candidate_models,
            "chosen_action": self.chosen_action,
            "chosen_model": self.chosen_model,
            "teacher_distribution": self.teacher_distribution,
            "observation_before": self.observation_before,
            "action": self.action,
            "observation_after": self.observation_after,
            "state_after": self.state_after,
            "latencies": self.latencies,
            "resources": self.resources,
            "outcome": self.outcome,
            "verification": self.verification,
            "source": self.source,
            "timestamp": self.timestamp,
            "privacy_class": self.privacy_class,
            "label_source": self.label_source,
            "label_confidence": self.label_confidence,
            "verified_success": self.verified_success,
            "task_family": self.task_family,
            "task_id": self.task_id,
            "session_id": self.session_id,
        }
        if self.extensions:
            out["extensions"] = dict(self.extensions)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Episode":
        known = {f for f in cls.__dataclass_fields__ if f != "schema"}  # type: ignore[attr-defined]
        payload = {k: v for k, v in data.items() if k in known}
        payload.setdefault("schema", str(data.get("schema", EPISODE_SCHEMA)))
        payload.setdefault("extensions", {})
        return cls(**payload)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Compiler: measured receipt -> episode
# ---------------------------------------------------------------------------


def _fixture_text_index(fixtures_path: Path | str | None) -> dict[str, dict[str, Any]]:
    """State text/objective for a fixture id, read from its owning repo file."""
    index: dict[str, dict[str, Any]] = {}
    if not fixtures_path:
        return index
    path = Path(fixtures_path)
    if not path.is_file():
        return index
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            fid = row.get("fixture_id")
            if fid:
                index[str(fid)] = row
    return index


def privacy_class_for(source: dict[str, Any]) -> str:
    """Local-only measurement stays local.

    Every Phase 1B receipt came from the local supervisor; nothing was sent to a
    paid endpoint.  If that ever changes, this must be revisited explicitly.
    """
    if source.get("external_inference"):
        return "external"
    return "local_only"


def compile_episode(
    row: Mapping[str, Any],
    *,
    fixtures: Mapping[str, Mapping[str, Any]] | None = None,
    task_family: str | None = None,
) -> Episode:
    """Turn one Phase 1B observation receipt into a canonical episode."""
    state_id = str(row.get("state_id") or "")
    fixture = (fixtures or {}).get(state_id) or {}
    verification = dict(row.get("verification") or {})
    correct = row.get("correct")
    is_gold = str(verification.get("kind") or "") in GOLD_VERIFIERS and correct is not None

    if is_gold:
        label_source = "gold"
        label_confidence = 1.0
        verified_success: bool | None = bool(correct)
    else:
        # An observed trajectory with no verifier is silver at best.  It must
        # never masquerade as verified.
        label_source = "silver"
        label_confidence = 0.5
        verified_success = None

    distribution = row.get("distribution")
    teacher_distribution = (
        {str(k): float(v) for k, v in distribution.items()}
        if isinstance(distribution, Mapping) and row.get("distribution_source") == "scorer"
        else None
    )
    if teacher_distribution is not None and label_source != "gold":
        label_source = "teacher"
        label_confidence = row.get("confidence") if row.get("confidence") is not None else 0.5

    legal = [str(x) for x in (row.get("legal_actions") or [])]
    chosen = row.get("selected_action")
    return Episode(
        episode_id=f"{state_id}:{row.get('arm')}:{row.get('repetition')}:{row.get('trace_id')}",
        state_before={
            "state_id": state_id,
            "text": fixture.get("state"),
            "features": dict(row.get("state_features") or {}),
            "family": fixture.get("family") or (row.get("state_features") or {}).get("family"),
            "authority": list(fixture.get("authority") or []),
            "granted_capabilities": list(fixture.get("granted_capabilities") or []),
            "budget_units": fixture.get("budget_units"),
        },
        available_evidence=[
            {
                "kind": "elimination",
                "action_id": e.get("action_id"),
                "stage": e.get("stage"),
                "reason": e.get("reason"),
            }
            for e in (row.get("eliminated") or [])
        ],
        legal_actions=legal,
        candidate_models=[str(row.get("model_id"))] if row.get("model_id") else [],
        chosen_action=str(chosen) if chosen is not None else None,
        chosen_model=str(row.get("model_id")) if row.get("model_id") else None,
        teacher_distribution=teacher_distribution,
        observation_before=None,
        action={
            "kind": "bounded_choice",
            "arm": row.get("arm"),
            "arm_alias": row.get("arm_alias"),
            "abstained": bool(row.get("abstained")),
            "invalid": bool(row.get("invalid_call")),
            "deterministic_solution": row.get("deterministic_solution"),
        },
        observation_after=None,
        state_after=None,
        latencies={
            "load_ms": row.get("load_ms"),
            "ttft_ms": row.get("ttft_ms"),
            "decision_ms": row.get("decision_ms"),
            "total_ms": row.get("total_ms"),
            "residency": row.get("cold_or_warm"),
        },
        resources={
            "tokens_in": row.get("tokens_in"),
            "tokens_out": row.get("tokens_out"),
            "tok_s": row.get("tok_s"),
            "peak_vram_mib": row.get("peak_vram_mib"),
            "cost_usd": None,
            "accounting_authority": row.get("accounting_authority", "tokenomics"),
        },
        outcome={
            "correct": bool(correct) if correct is not None else None,
            "dangerous_exposed": bool(row.get("dangerous_exposed")),
            "dangerous_selected": bool(row.get("dangerous_selected")),
            "abstained": bool(row.get("abstained")),
            "invalid_call": bool(row.get("invalid_call")),
        },
        verification=verification or {"kind": "none"},
        source={
            "kind": "z0intelligence.phase1b.observation",
            "run_id": row.get("run_id"),
            "trace_id": row.get("trace_id"),
            "fixture_revision": row.get("source_fixture_revision"),
            "compiler_revision": row.get("compiler_revision"),
            "router_revision": row.get("router_revision"),
            "quant": row.get("quant"),
            "model_revision": row.get("model_revision"),
            "max_tokens": row.get("max_tokens"),
            "external_inference": False,
        },
        timestamp=str(row.get("observed_at") or ""),
        privacy_class=privacy_class_for({"external_inference": False}),
        label_source=label_source,
        label_confidence=label_confidence,
        verified_success=verified_success,
        extensions={
            "margin": row.get("margin"),
            "entropy": row.get("entropy"),
            "confidence": row.get("confidence"),
            "distribution_source": row.get("distribution_source"),
        },
        task_family=str(task_family or fixture.get("family") or "uncategorised"),
        task_id=state_id,
        session_id=str(row.get("run_id") or ""),
    )


def compile_episodes(
    observations_path: Path | str,
    *,
    fixtures_path: Path | str | None = None,
    family_map: Mapping[str, str] | None = None,
    arms: Sequence[str] | None = None,
) -> list[Episode]:
    """Compile observation receipts into episodes.

    ``arms`` restricts the corpus to an explicit arm set. The Phase 2 slice is
    defined over five compiler-first arms (28 states x 5 arms x 3 reps = 420
    gold episodes); compiling every receipt in the file would silently include
    the unfiltered controls and the cold probe, which are different experiments
    and must not enter the same corpus.
    """
    wanted = set(arms) if arms else None
    fixtures = _fixture_text_index(fixtures_path)
    episodes: list[Episode] = []
    path = Path(observations_path)
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if wanted is not None and str(row.get("arm")) not in wanted:
                continue
            episodes.append(
                compile_episode(
                    row,
                    fixtures=fixtures,
                    task_family=(family_map or {}).get(str(row.get("state_id"))),
                )
            )
    return episodes


# ---------------------------------------------------------------------------
# Selective teacher inference
# ---------------------------------------------------------------------------


@dataclass
class SelectionCriteria:
    """Why an episode is worth spending teacher inference on."""

    unlabelled: bool = True
    ambiguity: bool = True
    disagreement: bool = True
    uncertainty: bool = True
    high_information: bool = True
    min_entropy: float = 0.0


def selective_teacher_candidates(
    episodes: Sequence[Episode],
    *,
    criteria: SelectionCriteria | None = None,
    disagreement_window: int = 2,
    max_uncertainty: float = 0.75,
) -> dict[str, Any]:
    """Nominate episodes for teacher labelling -- and say why each was picked.

    Nothing is labelled here.  The output is a work list: which episodes a
    teacher would be asked about, and the observable reason.  That keeps teacher
    spend proportional to genuine ambiguity instead of corpus size.
    """
    cfg = criteria or SelectionCriteria()
    by_cell: dict[tuple[str, str], list[Episode]] = defaultdict(list)
    by_state: dict[str, list[Episode]] = defaultdict(list)
    for ep in episodes:
        by_cell[(ep.task_id, str(ep.action.get("arm")))].append(ep)
        by_state[ep.task_id].append(ep)

    # Ambiguity must mean the *task* is ambiguous, not that the arms disagree
    # about it -- on a discriminating benchmark most states have both winners and
    # losers, so "arms disagree" nominates the whole corpus (measured: 78%).
    # A state is genuinely unresolved when no measured arm got it right and the
    # compiler had no deterministic answer: the correct action is not
    # represented in the evidence, which is exactly what a teacher is for.
    unresolved: dict[str, bool] = {}
    for state, group in by_state.items():
        decided = [e for e in group if e.outcome.get("correct") is not None]
        if not decided:
            unresolved[state] = False
            continue
        any_correct = any(bool(e.outcome.get("correct")) for e in decided)
        deterministic = any(
            e.action.get("deterministic_solution") for e in group
        )
        unresolved[state] = (not any_correct) and (not deterministic)

    selected: list[dict[str, Any]] = []
    for ep in episodes:
        reasons: list[str] = []
        cell = by_cell[(ep.task_id, str(ep.action.get("arm")))]
        if cfg.unlabelled and ep.label_source != "gold":
            reasons.append("unlabelled")
        if cfg.disagreement and len(cell) > 1:
            outcomes = {bool(e.outcome.get("correct")) for e in cell}
            if len(outcomes) > 1:
                reasons.append("disagreement")
        if cfg.ambiguity and unresolved.get(ep.task_id):
            reasons.append("ambiguity")
        if cfg.uncertainty and ep.label_confidence is not None:
            if float(ep.label_confidence) < max_uncertainty:
                reasons.append("uncertainty")
        if cfg.high_information and ep.chosen_model and len(ep.legal_actions) > 1:
            # Only a real model call can be uncertain.  The compiler-only arm
            # "abstains" by returning an all-zero distribution, which is not
            # uncertainty, it is the absence of a model; and a singleton legal
            # set has margin 1.0 by construction.
            margin = (ep.extensions or {}).get("margin")
            entropy = (ep.extensions or {}).get("entropy")
            if margin is not None and float(margin) <= (1.0 / len(ep.legal_actions)):
                if entropy is None or float(entropy) >= cfg.min_entropy:
                    reasons.append("high_information")
        if reasons:
            selected.append(
                {
                    "episode_id": ep.episode_id,
                    "task_id": ep.task_id,
                    "arm": ep.action.get("arm"),
                    "reasons": sorted(set(reasons)),
                    "n_in_cell": len(cell),
                }
            )
    reason_counts: dict[str, int] = {}
    for row in selected:
        for reason in row["reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "schema": SELECTION_SCHEMA,
        "n_episodes": len(episodes),
        "n_selected": len(selected),
        "selection_rate": (len(selected) / len(episodes)) if episodes else None,
        "reason_counts": reason_counts,
        "selected": selected,
    }


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


@dataclass
class SplitAssignment:
    episode_id: str
    bucket: str
    task_family: str
    task_id: str
    timestamp: str


def _parse_ts(value: str) -> float:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return 0.0


def chronological_split(
    episodes: Sequence[Episode],
    *,
    train_frac: float = 0.70,
    validation_frac: float = 0.15,
) -> list[SplitAssignment]:
    """Oldest -> train, next -> validation, newest -> confirm.

    Splitting is **by task**, not by episode: three repetitions of one fixture
    are not three independent tasks, and letting two of them sit in train while
    the third is in confirm is leakage.
    """
    by_task: dict[str, list[Episode]] = defaultdict(list)
    for ep in episodes:
        by_task[ep.task_id].append(ep)
    ordered_tasks = sorted(
        by_task,
        key=lambda t: (min(_parse_ts(e.timestamp) for e in by_task[t]), t),
    )
    n = len(ordered_tasks)
    if n == 0:
        return []
    n_train = max(1, int(math.floor(n * train_frac)))
    n_val = max(1, int(math.floor(n * validation_frac))) if n >= 3 else 0
    while n_train + n_val >= n:
        if n_val > 0:
            n_val -= 1
        else:
            n_train -= 1
            break
    train = set(ordered_tasks[:n_train])
    validation = set(ordered_tasks[n_train : n_train + n_val])
    assignments: list[SplitAssignment] = []
    for task in ordered_tasks:
        bucket = "train" if task in train else ("validation" if task in validation else "confirm")
        for ep in by_task[task]:
            assignments.append(
                SplitAssignment(ep.episode_id, bucket, ep.task_family, ep.task_id, ep.timestamp)
            )
    return assignments


def family_ood_split(
    episodes: Sequence[Episode],
    *,
    holdout_families: Sequence[str] | None = None,
    holdout_frac: float = 0.20,
) -> dict[str, Any]:
    """Hold out whole task families so OOD is a property, not a hope.

    Families are chosen deterministically (largest first, then alphabetical) so
    the split is reproducible from the episode set alone.
    """
    by_family: dict[str, int] = defaultdict(int)
    for ep in episodes:
        by_family[ep.task_family] += 1
    if holdout_families:
        held = sorted(set(holdout_families))
    else:
        # Whole families only, and accumulated smallest-first so the OOD set is
        # a spread of task shapes rather than one large family wearing a label.
        target = max(1, int(round(len(episodes) * holdout_frac))) if episodes else 0
        held = []
        running = 0
        for family in sorted(by_family, key=lambda f: (by_family[f], f)):
            if running >= target:
                break
            held.append(family)
            running += by_family[family]
        held = sorted(held)
    return {
        "holdout_families": held,
        "in_family_count": len(by_family) - len(held),
        "family_sizes": dict(sorted(by_family.items())),
        "episodes_in_holdout": [ep.episode_id for ep in episodes if ep.task_family in held],
    }


def sealed_human_audited(
    episodes: Sequence[Episode],
    *,
    size: int = 12,
    max_task_fraction: float = 0.15,
) -> dict[str, Any]:
    """Nominate a sealed, human-audited set that is never trained on.

    The unit is the **task**, not the episode.  Sealing a single episode of a
    task whose siblings stay in train would leak the task into training while
    calling it audited, which defeats the point.  So the whole task is sealed.

    ``size`` is the requested number of *episodes*; ``max_task_fraction`` caps how
    many tasks the sealed set may consume.  With the Phase 1B corpus (28 tasks)
    the cap binds long before ``size`` does, and the plan records the shortfall
    rather than silently sealing a third of the corpus.

    Deterministic selection: tasks ordered by family then id, so the set is
    reproducible and not hand-picked toward easy cases.
    """
    by_task: dict[str, list[Episode]] = defaultdict(list)
    for ep in episodes:
        by_task[ep.task_id].append(ep)
    ordered = sorted(
        by_task, key=lambda t: (by_task[t][0].task_family, t)
    )
    n_tasks = len(ordered)
    budget = max(1, int(math.floor(n_tasks * max_task_fraction))) if n_tasks else 0
    chosen: list[str] = []
    running = 0
    for task in ordered:
        if len(chosen) >= budget or running >= size:
            break
        chosen.append(task)
        running += len(by_task[task])
    if not chosen and ordered:
        chosen.append(ordered[0])
    selected = [ep for task in chosen for ep in by_task[task]]
    return {
        "requested_episodes": size,
        "task_budget": budget,
        "sealed_task_ids": sorted(chosen),
        "selected_tasks": len(chosen),
        "selected_episodes": len(selected),
        "shortfall_reason": (
            None if len(selected) >= size
            else (
                f"whole-task sealing at {n_tasks} tasks allows at most {budget} "
                f"tasks; sealing more would take the training split below a "
                f"usable size"
            )
        ),
        "overshoot_reason": (
            None if len(selected) <= size
            else (
                f"the smallest sealable unit is one task, and the first task "
                f"carries {len(selected)} episodes; sealing a partial task would "
                f"leak its siblings into training"
            )
        ),
        "selected": [
            {
                "episode_id": ep.episode_id,
                "task_id": ep.task_id,
                "task_family": ep.task_family,
                "human_audited": False,
                "note": "sealed: excluded from every training and search split",
            }
            for ep in sorted(selected, key=lambda e: (e.task_family, e.task_id, e.episode_id))
        ],
    }


def build_split_plan(
    episodes: Sequence[Episode],
    *,
    sealed_size: int = 12,
    holdout_families: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Assemble the frozen Phase 2 split plan (no training, no search).

    The sealed set is drawn from the whole corpus, stratified one episode per
    task, and a sealed episode leaves whatever chronological bucket it would have
    had.  Drawing only from the newest confirm slice was tried and does not work
    at this corpus size: 28 tasks give roughly four confirm tasks, so a 12-episode
    audit set cannot be filled without consuming the whole slice.
    """
    chronological = chronological_split(episodes)
    ood = family_ood_split(episodes, holdout_families=holdout_families)
    sealed = sealed_human_audited(episodes, size=sealed_size)
    sealed_ids = {row["episode_id"] for row in sealed["selected"]}
    sealed_task_ids = set(sealed["sealed_task_ids"])
    ood_families = set(ood["holdout_families"])

    final: list[dict[str, Any]] = []
    counts: dict[str, int] = defaultdict(int)
    for assignment in chronological:
        bucket = assignment.bucket
        if assignment.task_id in sealed_task_ids:
            bucket = "sealed_human_audited"
        elif assignment.task_family in ood_families:
            bucket = "ood"
        counts[bucket] += 1
        final.append(
            {
                "episode_id": assignment.episode_id,
                "task_id": assignment.task_id,
                "task_family": assignment.task_family,
                "timestamp": assignment.timestamp,
                "chronological_bucket": assignment.bucket,
                "bucket": bucket,
            }
        )

    digest = hashlib.sha256(
        json.dumps(sorted(final, key=lambda r: r["episode_id"]),
                   sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    return {
        "schema": SPLIT_SCHEMA,
        "n_episodes": len(episodes),
        "n_tasks": len({e.task_id for e in episodes}),
        "proportions": {"train": 0.70, "validation": 0.15, "confirm": 0.15},
        "bucket_counts": dict(sorted(counts.items())),
        "ood": ood,
        "sealed": sealed,
        "assignments": final,
        "split_digest": digest,
        "frozen_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def write_episodes(episodes: Iterable[Episode], path: Path | str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        for ep in episodes:
            fh.write(json.dumps(ep.to_dict(), sort_keys=True, default=str) + "\n")
    return target


def load_episodes(path: Path | str) -> list[Episode]:
    target = Path(path)
    out: list[Episode] = []
    with target.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(Episode.from_dict(json.loads(line)))
    return out


def label_census(episodes: Sequence[Episode]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ep in episodes:
        counts[ep.label_source] = counts.get(ep.label_source, 0) + 1
    return dict(sorted(counts.items()))


__all__ = [
    "EPISODE_SCHEMA",
    "GOLD_VERIFIERS",
    "LABEL_SOURCES",
    "SELECTION_SCHEMA",
    "SPLIT_SCHEMA",
    "Episode",
    "EpisodeError",
    "SelectionCriteria",
    "build_split_plan",
    "chronological_split",
    "compile_episode",
    "compile_episodes",
    "family_ood_split",
    "label_census",
    "load_episodes",
    "privacy_class_for",
    "sealed_human_audited",
    "selective_teacher_candidates",
    "write_episodes",
]
