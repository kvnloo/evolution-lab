"""The frozen non-inferiority gate and the earned-complexity ladder.

Project rule, expressed in code: **a more complex router may keep a region of
state-space only if the next-cheaper mechanism fails a frozen non-inferiority
gate there.**  :func:`compile_regions` walks each state's available arms from
the cheapest ladder rung upward and lets the cheapest passing arm own the
region.  Complexity is therefore earned per region, and simplification is the
default outcome whenever a cheaper arm is not meaningfully worse.

Thresholds in :class:`GateConfig` are set once and must not be tuned against the
split they are reported on.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .schema import GATE_REPORT_SCHEMA
from .teacher import TeacherRow, TeacherTable
from .utility import UtilityConfig, utility

#: Cheapest -> most expensive *mechanism*.  Not a latency ordering; latency is
#: checked separately by the gate so a nominally cheaper mechanism that is
#: slower still fails.
LADDER: tuple[str, ...] = (
    "deterministic",
    "tiny_specialist",
    "bounded_jev",
    "general_function_caller",
    "orchestrator",
    "general_fallback",
)
LADDER_RANK: dict[str, int] = {name: index for index, name in enumerate(LADDER)}

#: Explicit rung for every arm seen in the real batteries.  Unknown arms fall
#: back to the token heuristic in :func:`ladder_rung`.
ARM_LADDER: dict[str, str] = {
    # composition matrix (real runs)
    "A_qwen9b_alone": "general_fallback",
    "B_nemotron_alone": "orchestrator",
    "C_compiler_nemotron": "orchestrator",
    "D_compiler_jev_nemotron": "orchestrator",
    "E_compiler_jev_nemotron_qwen": "general_fallback",
    "E_compiler_jev_nemotron_qwen9b": "general_fallback",
    "F_compiler_tiny_jev_nemotron_qwen": "general_fallback",
    "F_compiler_jev_nemotron_qwen9b_specialists": "general_fallback",
    "SUB_qwen4b_alone": "general_function_caller",
    "SUB_compiler_qwen4b": "general_function_caller",
    "SUB_compiler_hammer3b": "tiny_specialist",
    "SUB_compiler_hammer7b": "general_function_caller",
    "SUB_compiler_functiongemma": "tiny_specialist",
    # Arm names introduced by the Phase 1B densification.  Adding a rung for a
    # newly measured arm does not change the gate's logic or any threshold: it
    # only tells the ladder where the arm sits.  Both spellings are present
    # because the first densified run predates the alias.
    "SUB_compiler_qwen9b": "general_fallback",
    "compiler+qwen3.5_9b": "general_fallback",
    "unfiltered+hammer2.1_3b": "tiny_specialist",
    # tournament/backend aliases
    "qwen4b_instead_of_9b": "general_function_caller",
    "hammer3b_specialist": "tiny_specialist",
    "hammer7b_specialist": "general_function_caller",
    "functiongemma_specialist": "tiny_specialist",
    "rules_baseline": "deterministic",
    "logistic_baseline": "bounded_jev",
    # bounded-choice / orchestration backends (real runs)
    "optimal_greedy": "deterministic",
    "functiongemma_270m": "tiny_specialist",
    "hammer2.1_3b": "tiny_specialist",
    "hammer2.1_7b": "general_function_caller",
    "qwen3.5_4b": "general_function_caller",
    "nemotron_orchestrator_8b": "orchestrator",
    "qwen3.5_9b": "general_fallback",
}

_EPS = 1e-12
_MIN_RATIO_SENTINEL = 2.0  # finite stand-in when the expensive arm has zero latency


def ladder_rung(arm: str) -> str:
    """Mechanism rung for an arm name (explicit table, then token heuristic)."""
    if arm in ARM_LADDER:
        return ARM_LADDER[arm]
    low = arm.lower()
    if any(token in low for token in ("deterministic", "greedy", "rule", "rule_based")):
        return "deterministic"
    if "logistic" in low or ("jev" in low and "nemotron" not in low and "qwen" not in low):
        return "bounded_jev"
    if any(token in low for token in ("functiongemma", "270m", "tiny", "hammer3b", "hammer2.1_3b")):
        return "tiny_specialist"
    if any(token in low for token in ("nemotron", "orchestrat")):
        return "orchestrator"
    if any(token in low for token in ("4b", "7b", "hammer", "function_caller", "specialist")):
        return "general_function_caller"
    if any(token in low for token in ("9b", "qwen", "fallback")):
        return "general_fallback"
    return "general_fallback"


def ladder_rank(arm: str) -> int:
    return LADDER_RANK[ladder_rung(arm)]


@dataclass(frozen=True)
class GateConfig:
    gate_id: str = "qroute.gate.v1"
    # frozen thresholds: set once, never tuned against the same split
    max_utility_regression: float = 0.02
    max_latency_ratio: float = 1.0
    max_dangerous_rate: float = 0.0
    min_observations: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "max_utility_regression": float(self.max_utility_regression),
            "max_latency_ratio": float(self.max_latency_ratio),
            "max_dangerous_rate": float(self.max_dangerous_rate),
            "min_observations": int(self.min_observations),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GateConfig":
        return cls(
            gate_id=str(data.get("gate_id", "qroute.gate.v1")),
            max_utility_regression=float(data.get("max_utility_regression", 0.02)),
            max_latency_ratio=float(data.get("max_latency_ratio", 1.0)),
            max_dangerous_rate=float(data.get("max_dangerous_rate", 0.0)),
            min_observations=int(data.get("min_observations", 1)),
        )


@dataclass(frozen=True)
class GateOutcome:
    passed: bool
    reasons: tuple[str, ...]
    deltas: dict[str, Any]
    cheap_arm: str = ""
    expensive_arm: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reasons": list(self.reasons),
            "deltas": dict(self.deltas),
            "cheap_arm": self.cheap_arm,
            "expensive_arm": self.expensive_arm,
        }


def _latency_ratio(cheap_ms: float, expensive_ms: float, cfg: GateConfig) -> float:
    if expensive_ms > _EPS:
        return float(cheap_ms) / float(expensive_ms)
    if cheap_ms <= _EPS:
        return 1.0
    # Finite sentinel; must exceed any sane max_latency_ratio and stay JSON-safe.
    return max(float(cfg.max_latency_ratio) + 1.0, _MIN_RATIO_SENTINEL)


def evaluate_gate(
    cheap: TeacherRow,
    expensive: TeacherRow,
    cfg: GateConfig,
    utility_config: UtilityConfig | None = None,
) -> GateOutcome:
    """Frozen non-inferiority check of ``cheap`` against ``expensive``.

    ``cheap`` passes only when it is observed enough, never dangerous, not
    meaningfully worse in utility (> ``max_utility_regression``) and not slower
    (``max_latency_ratio``).  Every failure is named in ``reasons``.
    """
    ucfg = utility_config or UtilityConfig()
    reasons: list[str] = []

    if cheap.n < cfg.min_observations or expensive.n < cfg.min_observations:
        reasons.append("insufficient_observations")

    if float(cheap.dangerous_rate) > float(cfg.max_dangerous_rate) + _EPS:
        reasons.append("dangerous_rate_exceeded")

    util_cheap = utility(cheap, ucfg)
    util_expensive = utility(expensive, ucfg)
    regression = util_expensive - util_cheap
    if regression > float(cfg.max_utility_regression) + _EPS:
        reasons.append("utility_regression")

    ratio = _latency_ratio(cheap.latency_p50_ms, expensive.latency_p50_ms, cfg)
    if ratio > float(cfg.max_latency_ratio) + _EPS:
        reasons.append("latency_ratio_exceeded")

    deltas: dict[str, Any] = {
        "n_cheap": int(cheap.n),
        "n_expensive": int(expensive.n),
        "utility_cheap": util_cheap,
        "utility_expensive": util_expensive,
        "utility_regression": regression,
        "max_utility_regression": float(cfg.max_utility_regression),
        "latency_p50_cheap_ms": float(cheap.latency_p50_ms),
        "latency_p50_expensive_ms": float(expensive.latency_p50_ms),
        "latency_ratio": ratio,
        "max_latency_ratio": float(cfg.max_latency_ratio),
        "dangerous_rate_cheap": float(cheap.dangerous_rate),
        "max_dangerous_rate": float(cfg.max_dangerous_rate),
    }
    return GateOutcome(
        passed=not reasons,
        reasons=tuple(reasons),
        deltas=deltas,
        cheap_arm=cheap.arm,
        expensive_arm=expensive.arm,
    )


@dataclass
class RegionDecision:
    """Which arm owns one state, and why anything more complex was needed."""

    state_key: str
    owner: str
    owner_rung: str
    cheapest_arm: str
    cheapest_rung: str
    baseline_arm: str
    escalated_from: list[str] = field(default_factory=list)
    reasons_for_escalation: list[dict[str, Any]] = field(default_factory=list)
    checks: list[dict[str, Any]] = field(default_factory=list)
    owner_latency_p50_ms: float = 0.0
    baseline_latency_p50_ms: float = 0.0
    latency_saved_ms: float = 0.0
    owns_cheapest: bool = False
    n_arms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_key": self.state_key,
            "owner": self.owner,
            "owner_rung": self.owner_rung,
            "cheapest_arm": self.cheapest_arm,
            "cheapest_rung": self.cheapest_rung,
            "baseline_arm": self.baseline_arm,
            "escalated_from": list(self.escalated_from),
            "reasons_for_escalation": [dict(r) for r in self.reasons_for_escalation],
            "checks": [dict(c) for c in self.checks],
            "owner_latency_p50_ms": self.owner_latency_p50_ms,
            "baseline_latency_p50_ms": self.baseline_latency_p50_ms,
            "latency_saved_ms": self.latency_saved_ms,
            "owns_cheapest": self.owns_cheapest,
            "n_arms": self.n_arms,
        }


@dataclass
class GateReport:
    regions: list[RegionDecision] = field(default_factory=list)
    owner_counts: dict[str, int] = field(default_factory=dict)
    owner_rungs: dict[str, str] = field(default_factory=dict)
    n_states: int = 0
    n_escalated_states: int = 0
    n_simplified_states: int = 0
    routed_total_latency_ms: float = 0.0
    baseline_total_latency_ms: float = 0.0
    total_latency_saved_ms: float = 0.0
    gate_config: dict[str, Any] = field(default_factory=dict)
    utility_config: dict[str, Any] = field(default_factory=dict)
    schema: str = GATE_REPORT_SCHEMA

    def region_for(self, state_key: str) -> RegionDecision | None:
        for region in self.regions:
            if region.state_key == state_key:
                return region
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "gate_config": dict(self.gate_config),
            "utility_config": dict(self.utility_config),
            "n_states": int(self.n_states),
            "n_escalated_states": int(self.n_escalated_states),
            "n_simplified_states": int(self.n_simplified_states),
            "owner_counts": {k: int(v) for k, v in sorted(self.owner_counts.items())},
            "owner_rungs": {k: v for k, v in sorted(self.owner_rungs.items())},
            "routed_total_latency_ms": float(self.routed_total_latency_ms),
            "baseline_total_latency_ms": float(self.baseline_total_latency_ms),
            "total_latency_saved_ms": float(self.total_latency_saved_ms),
            "regions": [r.to_dict() for r in self.regions],
        }


def compile_regions(
    table: TeacherTable,
    cfg: GateConfig,
    utility_config: UtilityConfig | None = None,
) -> GateReport:
    """Compile per-state ownership by walking each ladder cheapest -> dearest.

    Baseline for the latency saving is *always use the most expensive arm
    available for that state* (the highest rung, tie-broken by latency then
    name).
    """
    ucfg = utility_config or UtilityConfig()
    regions: list[RegionDecision] = []
    owner_counts: Counter[str] = Counter()
    owner_rungs: dict[str, str] = {}
    routed_total = 0.0
    baseline_total = 0.0

    for state_key in table.states():
        rows = table.rows_for_state(state_key)
        if not rows:
            continue
        ordered = sorted(rows, key=lambda r: (ladder_rank(r.arm), r.latency_p50_ms, r.arm))
        cheapest = ordered[0]
        baseline = ordered[-1]

        owner = cheapest
        escalated_from: list[str] = []
        reasons_for_escalation: list[dict[str, Any]] = []
        checks: list[dict[str, Any]] = []
        for candidate in ordered[1:]:
            outcome = evaluate_gate(owner, candidate, cfg, ucfg)
            check: dict[str, Any] = {
                "cheap": owner.arm,
                "expensive": candidate.arm,
                "passed": outcome.passed,
                "reasons": list(outcome.reasons),
            }
            if not outcome.passed:
                # Complexity must be *earned*: a more expensive arm may take the
                # region only if it is not itself worse in utility.  Otherwise a
                # faster-but-failing arm could win purely on the latency clause.
                candidate_utility = float(outcome.deltas["utility_expensive"])
                owner_utility = float(outcome.deltas["utility_cheap"])
                if candidate_utility < owner_utility - float(cfg.max_utility_regression) - _EPS:
                    check["escalation_refused"] = "candidate_worse_in_utility"
                    check["utility_cheap"] = owner_utility
                    check["utility_expensive"] = candidate_utility
                    checks.append(check)
                    continue
                escalated_from.append(owner.arm)
                reasons_for_escalation.append(
                    {
                        "from": owner.arm,
                        "to": candidate.arm,
                        "reasons": list(outcome.reasons),
                        "deltas": dict(outcome.deltas),
                    }
                )
                owner = candidate
            checks.append(check)

        saved = max(0.0, float(baseline.latency_p50_ms) - float(owner.latency_p50_ms))
        routed_total += float(owner.latency_p50_ms)
        baseline_total += float(baseline.latency_p50_ms)
        owner_counts[owner.arm] += 1
        owner_rungs[owner.arm] = ladder_rung(owner.arm)

        regions.append(
            RegionDecision(
                state_key=state_key,
                owner=owner.arm,
                owner_rung=ladder_rung(owner.arm),
                cheapest_arm=cheapest.arm,
                cheapest_rung=ladder_rung(cheapest.arm),
                baseline_arm=baseline.arm,
                escalated_from=escalated_from,
                reasons_for_escalation=reasons_for_escalation,
                checks=checks,
                owner_latency_p50_ms=float(owner.latency_p50_ms),
                baseline_latency_p50_ms=float(baseline.latency_p50_ms),
                latency_saved_ms=saved,
                owns_cheapest=(owner.arm == cheapest.arm),
                n_arms=len(ordered),
            )
        )

    return GateReport(
        regions=sorted(regions, key=lambda r: r.state_key),
        owner_counts=dict(owner_counts),
        owner_rungs=owner_rungs,
        n_states=len(regions),
        n_escalated_states=sum(1 for r in regions if r.escalated_from),
        n_simplified_states=sum(1 for r in regions if r.owns_cheapest),
        routed_total_latency_ms=routed_total,
        baseline_total_latency_ms=baseline_total,
        total_latency_saved_ms=max(0.0, baseline_total - routed_total),
        gate_config=cfg.to_dict(),
        utility_config=ucfg.to_dict(),
    )


def render_gate_markdown(report: GateReport, table: TeacherTable) -> str:
    """Readable ``gate.md``: ownership per state plus escalation reasons."""
    lines: list[str] = []
    lines.append("# Q-Route gate — earned complexity per region")
    lines.append("")
    lines.append(f"- schema: `{report.schema}`")
    lines.append(f"- gate: `{report.gate_config.get('gate_id')}`")
    lines.append(
        "- thresholds: max_utility_regression={max_utility_regression}, "
        "max_latency_ratio={max_latency_ratio}, max_dangerous_rate={max_dangerous_rate}, "
        "min_observations={min_observations}".format(**report.gate_config)
    )
    lines.append(
        f"- states: {report.n_states} ({report.n_simplified_states} owned by their cheapest arm, "
        f"{report.n_escalated_states} escalated at least once)"
    )
    lines.append(
        "- latency: routed {:.1f} ms vs always-most-expensive {:.1f} ms "
        "(saved {:.1f} ms)".format(
            report.routed_total_latency_ms,
            report.baseline_total_latency_ms,
            report.total_latency_saved_ms,
        )
    )
    lines.append("")
    lines.append("## Ownership by arm")
    lines.append("")
    lines.append("| arm | rung | states owned |")
    lines.append("|---|---|---|")
    for arm in sorted(report.owner_counts, key=lambda a: (-report.owner_counts[a], a)):
        lines.append(
            f"| {arm} | {report.owner_rungs.get(arm, '')} | {report.owner_counts[arm]} |"
        )
    lines.append("")
    lines.append("## Region decisions")
    lines.append("")
    lines.append("| state | owner | rung | cheapest | baseline | saved ms | escalation reason |")
    lines.append("|---|---|---|---|---|---|---|")
    for region in report.regions:
        if region.reasons_for_escalation:
            reason = "; ".join(
                "{} vs {}: {}".format(
                    esc["from"], esc["to"], ", ".join(esc["reasons"]) or "passed"
                )
                for esc in region.reasons_for_escalation
            )
        else:
            reason = "cheapest arm passed the frozen gate"
        lines.append(
            "| {} | {} | {} | {} | {} | {:.1f} | {} |".format(
                region.state_key,
                region.owner,
                region.owner_rung,
                region.cheapest_arm,
                region.baseline_arm,
                region.latency_saved_ms,
                reason,
            )
        )
    lines.append("")
    lines.append("## Gate checks (cheapest -> dearest)")
    lines.append("")
    for region in report.regions:
        lines.append(f"### {region.state_key}")
        lines.append("")
        for check in region.checks:
            verdict = "pass" if check["passed"] else "FAIL"
            detail = ", ".join(check["reasons"]) if check["reasons"] else "non-inferior"
            refused = check.get("escalation_refused")
            if refused:
                detail = f"{detail}; escalation refused: {refused}"
            lines.append(f"- `{check['cheap']}` vs `{check['expensive']}`: {verdict} ({detail})")
        lines.append("")
    return "\n".join(lines) + "\n"


__all__ = [
    "ARM_LADDER",
    "GateConfig",
    "GateOutcome",
    "GateReport",
    "LADDER",
    "LADDER_RANK",
    "RegionDecision",
    "compile_regions",
    "evaluate_gate",
    "ladder_rank",
    "ladder_rung",
    "render_gate_markdown",
]
