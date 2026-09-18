"""Vendored ABAB engine from abab-meta-research skill (MIT).

A improves how to investigate. B gathers discriminating evidence and mutates
the graph. Persist fly world at runs/autoresearch/league/abab.json.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal

Mutation = Literal[
    "ADD",
    "MERGE",
    "SPLIT",
    "UPGRADE",
    "DOWNGRADE",
    "PRUNE",
    "DEFER",
    "CONTRADICT",
    "SPAWN_EXPERIMENT",
    "SPAWN_RESEARCH_BRANCH",
    "NO_UPDATE",
]

MODE = Literal[
    "discovery",
    "mechanism",
    "cross_domain",
    "falsification",
    "critical_path",
    "architecture",
    "implementation",
    "autoresearch",
]

CLAIM_GRADE = Literal[
    "established",
    "strongly_supported",
    "promising_unverified",
    "disconfirmed",
    "unknown",
]


@dataclass
class Hypothesis:
    id: str
    claim: str
    mechanism: str = ""
    predictions: list[str] = field(default_factory=list)
    falsifiers: list[str] = field(default_factory=list)
    eig: float = 0.5
    impact: float = 0.5
    decision_change: float = 0.5
    transferability: float = 0.3
    cost: float = 1.0
    status: str = "open"  # open | rival | pruned | promoted | deferred
    rival_of: str | None = None


@dataclass
class Evidence:
    id: str
    claim: str
    source: str
    source_class: str  # measurement | primary | official | analysis | secondary
    scope: str = ""
    confidence: float = 0.5
    counter: bool = False


@dataclass
class Experiment:
    id: str
    hypothesis_id: str
    intervention: str
    baseline: str
    prediction: str
    metric: str
    control: str = ""
    ablation: str = ""
    holdout: str = ""
    rollback: str = "revert"
    promotion: str = ""
    outcome: str | None = None  # keep | revert | inconclusive
    measured: dict[str, Any] = field(default_factory=dict)


@dataclass
class Strategy:
    target_uncertainty: str
    assumption: str
    reframe: str
    mechanism: str
    neighbors: list[str] = field(default_factory=list)
    evidence_needed: list[str] = field(default_factory=list)
    counterexample: str = ""
    falsifier: str = ""
    source_classes: list[str] = field(default_factory=list)
    stop: str = ""
    mode: str = "discovery"


@dataclass
class World:
    objective: str
    constraints: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    baseline: str = ""
    concepts: dict[str, str] = field(default_factory=dict)
    mechanisms: dict[str, str] = field(default_factory=dict)
    contradictions: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    hypotheses: list[Hypothesis] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    experiments: list[Experiment] = field(default_factory=list)
    strategy_log: list[dict[str, Any]] = field(default_factory=list)
    dead_ends: list[str] = field(default_factory=list)
    no_update_streak: int = 0
    wave: int = 0
    wave_budget: int = 8
    deep: bool = False
    loop: str = "fast"  # fast | slow
    last_strategy: Strategy | None = None
    stopped: bool = False
    stop_reason: str = ""
    analog_transfers: list[dict[str, Any]] = field(default_factory=list)


def priority(h: Hypothesis) -> float:
    cost = max(h.cost, 1e-6)
    return (h.eig * h.impact * h.decision_change * h.transferability) / cost


def rank_open(world: World) -> list[Hypothesis]:
    open_h = [h for h in world.hypotheses if h.status in ("open", "rival")]
    return sorted(open_h, key=priority, reverse=True)


def source_weight(cls: str) -> float:
    return {
        "measurement": 1.0,
        "primary": 0.85,
        "official": 0.8,
        "analysis": 0.55,
        "secondary": 0.25,
    }.get(cls, 0.2)


def should_slow(world: World, event: str | None = None) -> bool:
    if event in ("contradiction", "plateau", "surprise", "graph_change", "repeat_fail"):
        return True
    if world.no_update_streak >= 2:
        return True
    fails = [e for e in world.experiments if e.outcome == "revert"]
    return len(fails) >= 2


def experiment_dominates(world: World) -> bool:
    ranked = rank_open(world)
    if not ranked:
        return False
    top = ranked[0]
    research_cost = top.cost
    if top.falsifiers and research_cost >= 2.0 and top.eig < 0.4:
        return True
    if world.wave >= 4 and top.mechanism and top.cost >= 1.5:
        return True
    return False


def stop_reason(world: World) -> str | None:
    if world.wave >= world.wave_budget:
        return "budget_exhausted"
    if world.no_update_streak >= 3:
        return "repeated_no_update"
    ranked = rank_open(world)
    if not ranked and world.wave > 0:
        return "objective_resolved"
    if experiment_dominates(world) and world.wave >= 3:
        return "experiment_dominates"
    if ranked:
        top = ranked[0]
        if top.decision_change < 0.05 and world.wave >= 2:
            return "cannot_change_decision"
        if priority(top) < 0.02 and world.wave >= 2:
            return "eig_collapsed"
    return None


def stage_c_ready(world: World) -> bool:
    survivors = [
        h
        for h in world.hypotheses
        if h.status in ("open", "rival", "promoted") and h.mechanism
    ]
    if not survivors:
        return False
    return experiment_dominates(world) or any(
        h.status == "promoted" for h in survivors
    )


def analog_ok(source_mechanism: str, target_mechanism: str, mapped: bool) -> bool:
    if not mapped:
        return False
    if not source_mechanism or not target_mechanism:
        return False
    if source_mechanism.strip().lower() == target_mechanism.strip().lower():
        return True
    # vocabulary-only: reject if mechanisms empty or only share a noun token
    return mapped


def apply_mutation(world: World, kind: Mutation, payload: dict[str, Any]) -> World:
    w = deepcopy(world)
    w.wave += 1
    if kind == "NO_UPDATE":
        w.no_update_streak += 1
        w.strategy_log.append({"wave": w.wave, "mutation": kind, **payload})
        reason = stop_reason(w)
        if reason:
            w.stopped = True
            w.stop_reason = reason
        return w

    w.no_update_streak = 0
    if kind == "ADD":
        if "hypothesis" in payload:
            h = payload["hypothesis"]
            w.hypotheses.append(h if isinstance(h, Hypothesis) else Hypothesis(**h))
        if "evidence" in payload:
            e = payload["evidence"]
            w.evidence.append(e if isinstance(e, Evidence) else Evidence(**e))
        if "concept" in payload:
            k, v = payload["concept"]
            w.concepts[k] = v
    elif kind == "MERGE":
        keep, drop = payload["keep"], payload["drop"]
        w.hypotheses = [h for h in w.hypotheses if h.id != drop]
        for h in w.hypotheses:
            if h.id == keep:
                h.status = "open"
    elif kind == "SPLIT":
        parent = payload["parent"]
        for spec in payload["children"]:
            w.hypotheses.append(Hypothesis(**spec) if not isinstance(spec, Hypothesis) else spec)
        for h in w.hypotheses:
            if h.id == parent:
                h.status = "deferred"
    elif kind == "UPGRADE":
        for h in w.hypotheses:
            if h.id == payload["id"]:
                h.status = "promoted"
                h.eig = min(1.0, h.eig + 0.1)
    elif kind == "DOWNGRADE":
        for h in w.hypotheses:
            if h.id == payload["id"]:
                h.eig = max(0.0, h.eig - 0.2)
                h.confidence if False else None
    elif kind == "PRUNE":
        for h in w.hypotheses:
            if h.id == payload["id"]:
                h.status = "pruned"
                w.dead_ends.append(h.claim)
    elif kind == "DEFER":
        for h in w.hypotheses:
            if h.id == payload["id"]:
                h.status = "deferred"
    elif kind == "CONTRADICT":
        w.contradictions.append(payload.get("note", ""))
        w.loop = "slow"
        rid = payload.get("id")
        if rid:
            for h in w.hypotheses:
                if h.id == rid:
                    h.status = "rival"
    elif kind == "SPAWN_EXPERIMENT":
        exp = payload["experiment"]
        w.experiments.append(exp if isinstance(exp, Experiment) else Experiment(**exp))
    elif kind == "SPAWN_RESEARCH_BRANCH":
        h = payload["hypothesis"]
        w.hypotheses.append(h if isinstance(h, Hypothesis) else Hypothesis(**h))
    w.strategy_log.append({"wave": w.wave, "mutation": kind})
    reason = stop_reason(w)
    if reason:
        w.stopped = True
        w.stop_reason = reason
    if should_slow(w, payload.get("event")):
        w.loop = "slow"
    return w


def record_experiment_outcome(world: World, exp_id: str, outcome: str, measured: dict[str, Any]) -> World:
    w = deepcopy(world)
    for e in w.experiments:
        if e.id == exp_id:
            e.outcome = outcome
            e.measured = measured
            hid = e.hypothesis_id
            for h in w.hypotheses:
                if h.id == hid:
                    if outcome == "keep":
                        h.status = "promoted"
                    elif outcome == "revert":
                        h.status = "open"
                        h.eig = min(1.0, h.eig + 0.15)  # failed experiment raises remaining uncertainty
                        w.dead_ends.append(f"experiment:{exp_id}")
                        w.loop = "slow"
    w.wave += 1
    w.no_update_streak = 0
    return w


def next_a(world: World) -> Strategy:
    ranked = rank_open(world)
    top = ranked[0] if ranked else None
    mode = "discovery"
    if world.deep:
        # budget hint, not a command
        if world.wave < 5:
            mode = "discovery"
        elif world.wave < 10:
            mode = "mechanism"
        elif world.wave < 15:
            mode = "falsification"
        elif world.wave < 20:
            mode = "architecture"
        else:
            mode = "critical_path"
    if top and top.rival_of:
        mode = "falsification"
    if experiment_dominates(world):
        mode = "implementation"
    if world.loop == "slow" and world.contradictions:
        mode = "mechanism"
    if not top:
        return Strategy(
            target_uncertainty="none",
            assumption="graph empty or resolved",
            reframe="stop or C",
            mechanism="",
            mode=mode,
            stop="objective_resolved",
        )
    return Strategy(
        target_uncertainty=top.claim,
        assumption=top.mechanism or "unspecified",
        reframe="uncertainty → information-acquisition",
        mechanism=top.mechanism,
        neighbors=[],
        evidence_needed=top.falsifiers or ["discriminating measurement"],
        counterexample=top.falsifiers[0] if top.falsifiers else "",
        falsifier=top.falsifiers[0] if top.falsifiers else "",
        source_classes=["measurement", "primary", "official"],
        stop="",
        mode=mode,
    )


def preserve_rivals(world: World, a_id: str, b_id: str) -> World:
    w = deepcopy(world)
    for h in w.hypotheses:
        if h.id in (a_id, b_id):
            h.status = "rival"
            h.rival_of = b_id if h.id == a_id else a_id
    return w


def discriminate(a: Hypothesis, b: Hypothesis) -> str:
    """Prefer a test that splits A from B, not more A-consistent evidence."""
    af = set(a.falsifiers)
    bf = set(b.falsifiers)
    shared = af & bf
    only_a = af - bf
    if only_a:
        return next(iter(only_a))
    if shared:
        return next(iter(shared))
    return f"measure:{a.id}-vs-{b.id}"


def learn_payload(world: World) -> dict[str, Any]:
    """Trajectory for /learn — strategies that survived measurement, not self-talk."""
    kept = [e for e in world.experiments if e.outcome == "keep"]
    failed = [e for e in world.experiments if e.outcome == "revert"]
    return {
        "objective": world.objective,
        "validated_experiments": [asdict(e) for e in kept],
        "failed_experiments": [asdict(e) for e in failed],
        "dead_ends": world.dead_ends,
        "promoted": [asdict(h) for h in world.hypotheses if h.status == "promoted"],
        "require_task_validation_before_default": True,
        "skip_if_no_measurement": not kept,
    }


def save(world: World, path: Path) -> None:
    def conv(o: Any) -> Any:
        if hasattr(o, "__dataclass_fields__"):
            d = asdict(o)
            d["_type"] = type(o).__name__
            return d
        raise TypeError(type(o))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(world), indent=2, default=str), encoding="utf-8")


def load(path: Path) -> World:
    raw = json.loads(path.read_text(encoding="utf-8"))
    hyps = [Hypothesis(**h) for h in raw.get("hypotheses", [])]
    ev = [Evidence(**e) for e in raw.get("evidence", [])]
    ex = [Experiment(**e) for e in raw.get("experiments", [])]
    raw["hypotheses"] = hyps
    raw["evidence"] = ev
    raw["experiments"] = ex
    raw.pop("last_strategy", None)
    return World(**{k: v for k, v in raw.items() if k in World.__dataclass_fields__})


def wave_heartbeat(world: World, changed: str, why: str, failed: str, next_move: str) -> dict[str, str]:
    ranked = rank_open(world)
    top = ranked[0].claim if ranked else "none"
    return {
        "CURRENT_MODEL": world.mechanisms.get("best") or world.baseline or world.objective,
        "WHAT_CHANGED": changed,
        "WHY": why,
        "WHAT_FAILED": failed,
        "GRAPH_DELTA": world.strategy_log[-1]["mutation"] if world.strategy_log else "none",
        "HIGHEST_VALUE_UNCERTAINTY": top,
        "NEXT_MOVE": next_move,
    }
