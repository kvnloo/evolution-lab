"""System compositions for the local tool-calling tournament.

A composition is a *system*, not a checkpoint: it names the deterministic
compiler, the local models and the fallback chain that together answer one
fixture. The A–F matrix compares monoliths against compiled systems.

Backends are injected through a tiny protocol so the whole tournament runs with
zero GPU and zero network in tests (``ScriptedBackend`` /
``DeterministicBaselineBackend``). The real z0int SLM is only imported lazily by
``LocalSLMBackendProvider`` and never at module import time.
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Protocol, runtime_checkable

from .fixtures import KNOWN_TOOLS, ActionCandidate, Fixture, value_matches_type

# --------------------------------------------------------------------------
# Backend protocol
# --------------------------------------------------------------------------


@dataclass
class DecisionRequest:
    """What a backend sees. ``legal_actions`` is already compiler-filtered."""

    fixture: Fixture
    legal_actions: list[str]
    declared_actions: list[str]
    backend_id: str
    role: str = "general_function_caller"
    compiled: bool = False
    allowed_models: list[str] = field(default_factory=list)


@dataclass
class BackendChoice:
    """One backend's answer for one decision point."""

    action: str | None = None
    actions: list[str] = field(default_factory=list)
    parallel_actions: list[str] = field(default_factory=list)
    confidence: float = 0.0
    abstained: bool = False
    latency_ms: float = 0.0
    invalid_call: bool = False
    distribution: dict[str, float] | None = None
    retries: int = 0
    escalated: bool = False
    model_id: str | None = None
    ttft_ms: float | None = None
    gpu_ms: float | None = None
    vram_peak_mb: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class TournamentBackend(Protocol):
    backend_id: str

    def decide(self, request: DecisionRequest) -> BackendChoice:  # pragma: no cover - protocol
        ...


# --------------------------------------------------------------------------
# Deterministic backends (real controls, no GPU/network)
# --------------------------------------------------------------------------


class DeterministicBaselineBackend:
    """Always picks the first legal action. A real, frozen control."""

    backend_id = "deterministic_baseline"
    model_id = "deterministic"

    def decide(self, request: DecisionRequest) -> BackendChoice:
        if not request.legal_actions:
            return BackendChoice(
                action=None,
                abstained=True,
                confidence=1.0,
                model_id=self.model_id,
                reason="no_legal_action",
            )
        action = request.legal_actions[0]
        return BackendChoice(
            action=action,
            actions=[action],
            confidence=1.0,
            model_id=self.model_id,
            reason="first_legal_action",
        )


_RISK_PENALTY = {"read": 0.0, "write": 0.35, "deploy": 1.5, "destructive": 3.0, "publish": 3.0}


class LogisticBaselineBackend:
    """Hand-frozen linear scorer over action features.

    Not learned, not trained here: coefficients are constants so the control is
    reproducible across runs and machines.
    """

    backend_id = "logistic_baseline"
    model_id = "logistic"

    def _score(self, fixture: Fixture, action_id: str) -> float:
        action = fixture.action(action_id)
        if action is None:
            return -10.0
        score = 0.5
        if set(action.requires_capabilities) <= set(fixture.granted_capabilities):
            score += 0.8
        score -= 0.25 * float(action.cost_units)
        score -= _RISK_PENALTY.get(action.risk_class, 0.5)
        score -= 0.15 * len(action.dependencies)
        if action.simulated_outcome in {"error", "timeout", "malformed"}:
            score -= 0.9
        if action_id in fixture.relevant_actions:
            score += 0.6
        return score

    def decide(self, request: DecisionRequest) -> BackendChoice:
        if not request.legal_actions:
            return BackendChoice(action=None, abstained=True, confidence=1.0, model_id=self.model_id)
        scores = {a: self._score(request.fixture, a) for a in request.legal_actions}
        top = max(scores.values())
        exps = {a: pow(2.718281828459045, s - top) for a, s in scores.items()}
        total = sum(exps.values()) or 1.0
        dist = {a: e / total for a, e in exps.items()}
        best = max(dist, key=lambda a: (dist[a], a))
        return BackendChoice(
            action=best,
            actions=[best],
            confidence=float(dist[best]),
            distribution=dist,
            model_id=self.model_id,
            reason="logistic_argmax",
        )


# --------------------------------------------------------------------------
# Scripted backend (tests; replays a fixture-authored answer)
# --------------------------------------------------------------------------


class ScriptedBackend:
    """Replays a fixture-authored answer. Deterministic and dependency-free."""

    def __init__(
        self,
        backend_id: str = "scripted",
        answers: dict[tuple[str, str], BackendChoice] | None = None,
        *,
        default: BackendChoice | None = None,
    ) -> None:
        self.backend_id = backend_id
        self._answers = dict(answers or {})
        self._default = default

    def decide(self, request: DecisionRequest) -> BackendChoice:
        fixture_id = request.fixture.fixture_id
        choice = self._answers.get((fixture_id, self.backend_id))
        if choice is None:
            choice = self._answers.get((fixture_id, "*"))
        if choice is None:
            choice = self._default
        if choice is None:
            return BackendChoice(
                action=None,
                abstained=True,
                confidence=0.0,
                model_id=self.backend_id,
                reason="no_script_entry",
            )
        return replace(choice)

    @staticmethod
    def answers_from_fixtures(
        fixtures: Iterable[Fixture],
        *,
        mode: str = "gold",
        wildcard: str = "*",
        latency_ms: float = 1.0,
    ) -> dict[tuple[str, str], BackendChoice]:
        """Author an answer for every fixture.

        ``gold`` answers the gold action (abstain where the row expects it).
        ``dangerous`` answers a declared dangerous action where one exists, else
        the gold answer — the hostile policy used to prove the compiler
        discriminates A/B from C–F.
        """
        if mode not in {"gold", "dangerous"}:
            raise ValueError(f"unknown script mode: {mode}")
        answers: dict[tuple[str, str], BackendChoice] = {}
        for f in fixtures:
            dangerous = f.declared_dangerous
            parallel = expected_parallel_actions(f)
            retries = 1 if f.gold_action in {"control.retry"} else 0
            if f.expect_abstain:
                answers[(f.fixture_id, wildcard)] = BackendChoice(
                    action=None,
                    abstained=True,
                    confidence=0.9,
                    latency_ms=latency_ms,
                    model_id=f.gold_model,
                    distribution={"abstain": 0.9},
                    reason=f"scripted_abstain:{mode}",
                )
                continue
            if mode == "dangerous" and dangerous:
                action = dangerous[0]
                answers[(f.fixture_id, wildcard)] = BackendChoice(
                    action=action,
                    actions=[action],
                    confidence=0.97,
                    latency_ms=latency_ms,
                    model_id=f.gold_model or "unknown",
                    distribution={action: 0.97},
                    reason="scripted_hostile_dangerous",
                )
                continue
            action = f.gold_action
            answers[(f.fixture_id, wildcard)] = BackendChoice(
                action=action,
                actions=list(f.gold_plan) if action else [],
                parallel_actions=list(parallel),
                confidence=0.8,
                latency_ms=latency_ms,
                retries=retries,
                model_id=f.gold_model,
                distribution=({action: 0.8} if action else {"abstain": 0.8}),
                reason=f"scripted_gold:{mode}",
            )
        return answers

    @classmethod
    def from_fixtures(
        cls,
        fixtures: Iterable[Fixture],
        *,
        backend_id: str = "scripted",
        mode: str = "gold",
        latency_ms: float = 1.0,
    ) -> "ScriptedBackend":
        answers = cls.answers_from_fixtures(fixtures, mode=mode, latency_ms=latency_ms)
        return cls(backend_id=backend_id, answers=answers)


# --------------------------------------------------------------------------
# Real z0int SLM (lazy, optional)
# --------------------------------------------------------------------------


def _z0int_root() -> str:
    return os.environ.get("Z0INT_ROOT", "/home/kvn/tmp/openjev")


def _import_z0int_cognition() -> Any:
    """Import ``z0int.cognition`` if the checkout on ``Z0INT_ROOT`` is importable."""
    try:
        import z0int.cognition as cognition  # type: ignore

        return cognition
    except ImportError:
        pass
    src = os.path.join(_z0int_root(), "src")
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)
    try:
        import z0int.cognition as cognition  # type: ignore

        return cognition
    except ImportError:
        return None


class LocalSLMBackendProvider:
    """Thin adapter over the real z0int local tool-calling backend.

    The provider never imports z0int at module import time. ``backend()`` raises
    a clear ``RuntimeError`` when the checkout on ``Z0INT_ROOT`` (or an installed
    ``z0int``) is not importable, so tests and CI stay GPU/network free.
    """

    def __init__(self, backend_id: str, model_id: str | None = None, root: Path | str | None = None) -> None:
        self.backend_id = backend_id
        self.model_id = model_id or backend_id
        self.root = Path(root) if root is not None else None
        self.role = "general_local_fallback"

    def available(self) -> bool:
        return _import_z0int_cognition() is not None

    def _factory(self) -> Any:
        cognition = _import_z0int_cognition()
        if cognition is None:
            raise RuntimeError(
                "local SLM backend unavailable: cannot import 'z0int.cognition'. "
                f"Set Z0INT_ROOT to a z0intelligence checkout (currently {_z0int_root()!r}) "
                "or pip install the openjev extra. The tournament runs GPU/network free "
                "with --backend deterministic or --backend scripted."
            )
        for attr in ("make_local_tool_backend", "make_tool_backend", "local_tool_backend", "tool_calling"):
            factory = getattr(cognition, attr, None)
            if factory is None:
                continue
            try:
                return factory(model_id=self.model_id)
            except TypeError:
                try:
                    return factory(self.model_id)
                except TypeError:
                    return factory
        raise RuntimeError(
            "z0int.cognition imported but exposes no local tool backend factory "
            "(looked for make_local_tool_backend / make_tool_backend / local_tool_backend / tool_calling)."
        )

    def backend(self) -> TournamentBackend:
        factory = self._factory()
        if hasattr(factory, "decide"):
            backend = factory
        elif callable(factory):
            backend = factory()
        else:
            raise RuntimeError("z0int local tool backend factory did not return a backend with .decide()")
        if not getattr(backend, "backend_id", None):
            try:
                backend.backend_id = self.backend_id  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - defensive
                pass
        return backend


# --------------------------------------------------------------------------
# Backend registry
# --------------------------------------------------------------------------


@dataclass
class BackendRegistry:
    backends: dict[str, TournamentBackend]

    def get(self, backend_id: str) -> TournamentBackend:
        if backend_id not in self.backends:
            raise KeyError(f"backend not registered: {backend_id}")
        return self.backends[backend_id]

    def has(self, backend_id: str) -> bool:
        return backend_id in self.backends

    def ids(self) -> list[str]:
        return sorted(self.backends)

    @classmethod
    def from_mapping(cls, mapping: dict[str, TournamentBackend]) -> "BackendRegistry":
        return cls(dict(mapping))

    @classmethod
    def defaults(cls, *, include_local_slm: bool = False) -> "BackendRegistry":
        """Zero-dependency registry: deterministic + logistic controls only."""
        mapping: dict[str, TournamentBackend] = {
            "deterministic_baseline": DeterministicBaselineBackend(),
            "logistic_baseline": LogisticBaselineBackend(),
        }
        if include_local_slm:
            for bid in MODEL_BACKENDS:
                mapping[bid] = LocalSLMBackendProvider(bid).backend()
        return cls(mapping)

    @classmethod
    def deterministic_all(cls) -> "BackendRegistry":
        """Every model seat replays the deterministic control.

        A real offline control: the only thing that differs between A/B and C–F
        is whether the compiler runs, so it isolates the composition effect.
        """
        mapping: dict[str, TournamentBackend] = {
            "deterministic_baseline": DeterministicBaselineBackend(),
            "logistic_baseline": LogisticBaselineBackend(),
        }
        for bid in MODEL_BACKENDS:
            mapping[bid] = DeterministicBaselineBackend()
        return cls(mapping)

    @classmethod
    def local_slm(cls) -> "BackendRegistry":
        """Build every model seat from the real z0int checkout (raises if absent)."""
        mapping: dict[str, TournamentBackend] = {
            "deterministic_baseline": DeterministicBaselineBackend(),
            "logistic_baseline": LogisticBaselineBackend(),
        }
        for bid in MODEL_BACKENDS:
            mapping[bid] = LocalSLMBackendProvider(bid).backend()
        return cls(mapping)

    @classmethod
    def scripted(
        cls,
        fixtures: Iterable[Fixture],
        *,
        mode: str = "gold",
        backend_ids: Iterable[str] | None = None,
        latency_ms: float = 1.0,
    ) -> "BackendRegistry":
        """Deterministic + logistic controls plus scripted model backends."""
        ids = list(backend_ids if backend_ids is not None else DEFAULT_BACKEND_IDS)
        answers = ScriptedBackend.answers_from_fixtures(fixtures, mode=mode, latency_ms=latency_ms)
        mapping: dict[str, TournamentBackend] = {
            "deterministic_baseline": DeterministicBaselineBackend(),
            "logistic_baseline": LogisticBaselineBackend(),
        }
        for bid in ids:
            if bid in mapping:
                continue
            mapping[bid] = ScriptedBackend(backend_id=bid, answers=answers)
        return cls(mapping)


# --------------------------------------------------------------------------
# Deterministic compiler
# --------------------------------------------------------------------------


@dataclass
class CompiledPlan:
    """Result of compiling a fixture graph without any learned component."""

    legal_actions: list[str]
    removed: dict[str, str]
    forced_action: str | None
    forced_abstain: bool
    reasons: list[str]
    schema: str = "z0int.tool_tournament.compiled_plan.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compile_fixture(fixture: Fixture) -> CompiledPlan:
    """Filter declared actions down to the ones a policy is allowed to take.

    Removal is deterministic and purely structural: unknown tools, missing or
    mistyped arguments, ungranted capabilities, unsatisfied credentials,
    destructive and publish/send/deploy actions never survive. A learned model
    can therefore never make a dangerous action reachable through a
    compiler-based composition.
    """
    legal: list[str] = []
    removed: dict[str, str] = {}
    for action in fixture.actions:
        reason = _veto_reason(fixture, action)
        if reason is None:
            legal.append(action.action_id)
        else:
            removed[action.action_id] = reason

    forced_action: str | None = None
    forced_abstain = False
    reasons: list[str] = []
    if not legal:
        # Nothing legal survives: the only sound plan is to abstain.
        forced_abstain = True
        reasons.append("no_legal_action")
    elif any(r == "no_legal_action" for r in removed.values()):  # pragma: no cover - unreachable, kept explicit
        forced_abstain = True
        reasons.append("no_legal_action")
    return CompiledPlan(
        legal_actions=legal,
        removed=removed,
        forced_action=forced_action,
        forced_abstain=forced_abstain,
        reasons=reasons,
    )


def _veto_reason(fixture: Fixture, action: ActionCandidate) -> str | None:
    if action.tool not in KNOWN_TOOLS:
        return "unknown_tool"
    if action.destructive:
        return "destructive_action"
    if action.publishes:
        return "publish_action"
    if not set(action.requires_capabilities) <= set(fixture.granted_capabilities):
        return "permission_denied"
    if not set(action.requires_credentials) <= set(fixture.satisfied):
        return "credential_required"
    missing = [a for a in action.required_args if a not in action.args and a not in fixture.satisfied]
    if missing:
        return "missing_argument"
    for name, expected in action.arg_types.items():
        if name in action.args and not value_matches_type(action.args[name], expected):
            return "wrong_argument_type"
    return None


# --------------------------------------------------------------------------
# Compositions
# --------------------------------------------------------------------------

MODEL_BACKENDS: dict[str, str] = {
    "qwen9b": "qwen9b",
    "qwen4b": "qwen4b",
    "nemotron": "nemotron",
    "jev": "jev",
    "hammer3b": "hammer3b",
    "hammer7b": "hammer7b",
    "functiongemma": "functiongemma",
}
DEFAULT_BACKEND_IDS: tuple[str, ...] = tuple(MODEL_BACKENDS)
CONTROL_BACKEND_IDS: tuple[str, ...] = ("deterministic_baseline", "logistic_baseline")

ROLES: tuple[str, ...] = (
    "tiny_action_specialist",
    "bounded_scorer",
    "general_function_caller",
    "semantic_orchestrator",
    "general_local_fallback",
)

ESCALATION_ACTIONS: frozenset[str] = frozenset({"task.escalate", "control.escalate", "task.delegate"})


def expected_parallel_actions(fixture: Fixture) -> list[str]:
    """Gold-plan actions that are pairwise incomparable in the dependency DAG.

    For ``A -> {B, C} -> D`` this is ``{B, C}``: B and C can run concurrently
    even though D joins them. A flat serial plan is the thing being penalized.
    """
    plan = [a for a in fixture.gold_plan if fixture.action(a) is not None]
    if len(plan) < 2:
        return []

    def ancestors(action_id: str) -> set[str]:
        out: set[str] = set()
        stack = list(fixture.action(action_id).dependencies) if fixture.action(action_id) else []
        while stack:
            cur = stack.pop()
            if cur in out:
                continue
            out.add(cur)
            action = fixture.action(cur)
            if action is not None:
                stack.extend(action.dependencies)
        return out

    anc = {a: ancestors(a) for a in plan}

    def related(a: str, b: str) -> bool:
        return b in anc[a] or a in anc[b]

    # Maximum antichain over the plan's dependency poset (plans are tiny: ≤ 4 in
    # the frozen suite, so exhaustive search is exact and deterministic).
    from itertools import combinations

    for size in range(len(plan), 1, -1):
        for combo in combinations(plan, size):
            if all(not related(a, b) for a, b in combinations(combo, 2)):
                return list(combo)
    return []

#: Frozen end-to-end execution cost per action cost unit (ms). Keeps e2e latency
#: separate from decision latency without needing a real tool sandbox.
EXEC_UNIT_MS = 15.0


@dataclass(frozen=True)
class Composition:
    name: str
    description: str
    mode: str  # model_only | compiler | orchestrated | rules
    roles: tuple[str, ...]
    primary: tuple[str, ...] = ()
    orchestrator: str | None = None
    requires: tuple[str, ...] = ()

    def backend_ids(self) -> tuple[str, ...]:
        ids = list(self.primary)
        if self.orchestrator:
            ids.insert(0, self.orchestrator)
        for extra in self.requires:
            if extra not in ids:
                ids.append(extra)
        return tuple(ids)

    def evaluate(self, fixture: Fixture, registry: BackendRegistry) -> "CompositionDecision":
        missing = [b for b in self.backend_ids() if not registry.has(b)]
        if missing:
            return CompositionDecision.skipped(self.name, fixture.fixture_id, f"missing backends: {','.join(missing)}")

        compiled = compile_fixture(fixture)
        if self.mode == "model_only":
            return self._evaluate_model_only(fixture, registry, compiled)
        if self.mode == "rules":
            return self._evaluate_rules(fixture, compiled)
        if self.mode == "compiler":
            return self._evaluate_compiler(fixture, registry, compiled)
        if self.mode == "orchestrated":
            return self._evaluate_orchestrated(fixture, registry, compiled)
        raise ValueError(f"unknown composition mode: {self.mode}")

    # -- modes ---------------------------------------------------------------

    def _evaluate_model_only(
        self, fixture: Fixture, registry: BackendRegistry, compiled: CompiledPlan
    ) -> "CompositionDecision":
        declared = fixture.action_ids
        request = DecisionRequest(
            fixture=fixture,
            legal_actions=list(declared),
            declared_actions=list(declared),
            backend_id=self.primary[0],
            role=self.roles[0] if self.roles else "general_function_caller",
            compiled=False,
            allowed_models=list(MODEL_BACKENDS),
        )
        choice = registry.get(self.primary[0]).decide(request)
        return CompositionDecision.from_choice(
            self.name, fixture, choice, compiled=compiled, vetoed=False, forced=False
        )

    def _evaluate_rules(self, fixture: Fixture, compiled: CompiledPlan) -> "CompositionDecision":
        if compiled.forced_abstain:
            return CompositionDecision.forced_abstain(self.name, fixture, compiled, reason="compiler_forced_abstain")
        action = compiled.legal_actions[0]
        choice = BackendChoice(
            action=action,
            actions=[action],
            confidence=1.0,
            model_id="deterministic",
            reason="rules_first_legal",
        )
        return CompositionDecision.from_choice(
            self.name, fixture, choice, compiled=compiled, vetoed=False, forced=True
        )

    def _evaluate_compiler(
        self, fixture: Fixture, registry: BackendRegistry, compiled: CompiledPlan
    ) -> "CompositionDecision":
        if compiled.forced_abstain:
            return CompositionDecision.forced_abstain(self.name, fixture, compiled, reason="compiler_forced_abstain")
        choice, vetoed = self._ask_chain(fixture, registry, compiled, list(self.primary))
        if choice is None:
            choice = self._deterministic_choice(compiled)
            vetoed = True
        return CompositionDecision.from_choice(
            self.name, fixture, choice, compiled=compiled, vetoed=vetoed, forced=False
        )

    def _evaluate_orchestrated(
        self, fixture: Fixture, registry: BackendRegistry, compiled: CompiledPlan
    ) -> "CompositionDecision":
        if compiled.forced_abstain:
            return CompositionDecision.forced_abstain(self.name, fixture, compiled, reason="compiler_forced_abstain")
        orchestrator = self.orchestrator
        assert orchestrator is not None
        vetoed = False
        request = DecisionRequest(
            fixture=fixture,
            legal_actions=list(compiled.legal_actions),
            declared_actions=fixture.action_ids,
            backend_id=orchestrator,
            role="semantic_orchestrator",
            compiled=True,
            allowed_models=list(MODEL_BACKENDS),
        )
        orch_choice = registry.get(orchestrator).decide(request)
        if orch_choice.abstained and orch_choice.action is None:
            return CompositionDecision.from_choice(
                self.name, fixture, orch_choice, compiled=compiled, vetoed=False, forced=False
            )
        if orch_choice.action in compiled.legal_actions:
            return CompositionDecision.from_choice(
                self.name, fixture, orch_choice, compiled=compiled, vetoed=False, forced=False
            )
        # Orchestrator named an action the compiler removed (or an unknown one).
        vetoed = True
        choice, worker_vetoed = self._ask_chain(fixture, registry, compiled, list(self.primary))
        vetoed = vetoed or worker_vetoed
        if choice is None:
            choice = self._deterministic_choice(compiled)
            vetoed = True
        return CompositionDecision.from_choice(
            self.name, fixture, choice, compiled=compiled, vetoed=vetoed, forced=False
        )

    # -- helpers -------------------------------------------------------------

    def _ask_chain(
        self,
        fixture: Fixture,
        registry: BackendRegistry,
        compiled: CompiledPlan,
        backend_ids: list[str],
    ) -> tuple[BackendChoice | None, bool]:
        vetoed = False
        for backend_id in backend_ids:
            request = DecisionRequest(
                fixture=fixture,
                legal_actions=list(compiled.legal_actions),
                declared_actions=fixture.action_ids,
                backend_id=backend_id,
                role="general_function_caller",
                compiled=True,
                allowed_models=list(MODEL_BACKENDS),
            )
            choice = registry.get(backend_id).decide(request)
            if choice.action is None and choice.abstained:
                return choice, vetoed
            if choice.action is None:
                vetoed = True
                continue
            plan = choice.actions or [choice.action]
            if choice.action in compiled.legal_actions and all(a in compiled.legal_actions for a in plan):
                return choice, vetoed
            vetoed = True
        return None, vetoed

    @staticmethod
    def _deterministic_choice(compiled: CompiledPlan) -> BackendChoice:
        action = compiled.legal_actions[0]
        return BackendChoice(
            action=action,
            actions=[action],
            confidence=1.0,
            model_id="deterministic",
            invalid_call=True,
            reason="compiler_veto_fallback",
        )


@dataclass
class CompositionDecision:
    composition: str
    fixture_id: str
    action: str | None
    actions: list[str]
    parallel_actions: list[str]
    confidence: float
    abstained: bool
    invalid_call: bool
    model_id: str | None
    latency_ms: float
    e2e_latency_ms: float
    retries: int
    escalated: bool
    vetoed: bool
    forced: bool
    removed: dict[str, str]
    reason: str
    status: str = "ok"
    skipped_reason: str = ""
    distribution: dict[str, float] | None = None
    cost_units: int = 0
    ttft_ms: float | None = None
    gpu_ms: float | None = None
    vram_peak_mb: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None

    @property
    def plan(self) -> list[str]:
        if self.actions:
            return list(self.actions)
        return [self.action] if self.action else []

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # -- constructors --------------------------------------------------------

    @classmethod
    def skipped(cls, composition: str, fixture_id: str, reason: str) -> "CompositionDecision":
        return cls(
            composition=composition,
            fixture_id=fixture_id,
            action=None,
            actions=[],
            parallel_actions=[],
            confidence=0.0,
            abstained=True,
            invalid_call=False,
            model_id=None,
            latency_ms=0.0,
            e2e_latency_ms=0.0,
            retries=0,
            escalated=False,
            vetoed=False,
            forced=False,
            removed={},
            reason="skipped",
            status="skipped",
            skipped_reason=reason,
        )

    @classmethod
    def forced_abstain(
        cls, composition: str, fixture: Fixture, compiled: CompiledPlan, *, reason: str
    ) -> "CompositionDecision":
        return cls(
            composition=composition,
            fixture_id=fixture.fixture_id,
            action=None,
            actions=[],
            parallel_actions=[],
            confidence=1.0,
            abstained=True,
            invalid_call=False,
            model_id="deterministic",
            latency_ms=0.0,
            e2e_latency_ms=0.0,
            retries=0,
            escalated=False,
            vetoed=False,
            forced=True,
            removed=dict(compiled.removed),
            reason=reason,
        )

    @classmethod
    def from_choice(
        cls,
        composition: str,
        fixture: Fixture,
        choice: BackendChoice,
        *,
        compiled: CompiledPlan,
        vetoed: bool,
        forced: bool,
    ) -> "CompositionDecision":
        plan = list(choice.actions)
        if not plan and choice.action:
            plan = [choice.action]
        action = choice.action
        if action is None and plan:
            action = plan[0]
        cost_units = 0
        for aid in plan:
            candidate = fixture.action(aid)
            if candidate is not None:
                cost_units += int(candidate.cost_units)
        e2e = float(choice.latency_ms) + EXEC_UNIT_MS * float(cost_units)
        escalated = bool(choice.escalated) or (action in ESCALATION_ACTIONS)
        return cls(
            composition=composition,
            fixture_id=fixture.fixture_id,
            action=action,
            actions=plan,
            parallel_actions=list(choice.parallel_actions),
            confidence=float(choice.confidence),
            abstained=bool(choice.abstained and action is None),
            invalid_call=bool(choice.invalid_call or vetoed),
            model_id=choice.model_id,
            latency_ms=float(choice.latency_ms),
            e2e_latency_ms=e2e,
            retries=int(choice.retries),
            escalated=escalated,
            vetoed=vetoed,
            forced=forced,
            removed=dict(compiled.removed),
            reason=choice.reason,
            distribution=dict(choice.distribution) if choice.distribution else None,
            cost_units=cost_units,
            ttft_ms=choice.ttft_ms,
            gpu_ms=choice.gpu_ms,
            vram_peak_mb=choice.vram_peak_mb,
            prompt_tokens=choice.prompt_tokens,
            completion_tokens=choice.completion_tokens,
            cost_usd=choice.cost_usd,
        )


COMPOSITIONS: dict[str, Composition] = {
    "A_qwen9b_alone": Composition(
        name="A_qwen9b_alone",
        description="A = qwen9b alone (no compiler; the model sees every declared action)",
        mode="model_only",
        roles=("general_function_caller", "general_local_fallback"),
        primary=("qwen9b",),
    ),
    "B_nemotron_alone": Composition(
        name="B_nemotron_alone",
        description="B = nemotron alone (no compiler)",
        mode="model_only",
        roles=("general_function_caller",),
        primary=("nemotron",),
    ),
    "C_compiler_nemotron": Composition(
        name="C_compiler_nemotron",
        description="C = compiler + nemotron",
        mode="compiler",
        roles=("general_function_caller",),
        primary=("nemotron",),
    ),
    "D_compiler_jev_nemotron": Composition(
        name="D_compiler_jev_nemotron",
        description="D = compiler + jev + nemotron",
        mode="orchestrated",
        roles=("semantic_orchestrator", "general_function_caller"),
        primary=("nemotron",),
        orchestrator="jev",
    ),
    "E_compiler_jev_nemotron_qwen9b": Composition(
        name="E_compiler_jev_nemotron_qwen9b",
        description="E = compiler + jev + nemotron + qwen9b fallback",
        mode="orchestrated",
        roles=("semantic_orchestrator", "general_local_fallback"),
        primary=("nemotron", "qwen9b"),
        orchestrator="jev",
    ),
    "F_compiler_jev_nemotron_qwen9b_specialists": Composition(
        name="F_compiler_jev_nemotron_qwen9b_specialists",
        description="F = E + credited tiny specialists (hammer3b, functiongemma)",
        mode="orchestrated",
        roles=("tiny_action_specialist", "bounded_scorer"),
        primary=("hammer3b", "functiongemma", "nemotron", "qwen9b"),
        orchestrator="jev",
    ),
    "qwen4b_instead_of_9b": Composition(
        name="qwen4b_instead_of_9b",
        description="E with the 4B fallback instead of the 9B",
        mode="orchestrated",
        roles=("general_local_fallback", "tiny_action_specialist"),
        primary=("nemotron", "qwen4b"),
        orchestrator="jev",
    ),
    "hammer3b_specialist": Composition(
        name="hammer3b_specialist",
        description="compiler + credited 3B tiny action specialist",
        mode="compiler",
        roles=("tiny_action_specialist",),
        primary=("hammer3b",),
    ),
    "hammer7b_specialist": Composition(
        name="hammer7b_specialist",
        description="compiler + credited 7B function-calling specialist",
        mode="compiler",
        roles=("general_function_caller",),
        primary=("hammer7b",),
    ),
    "functiongemma_specialist": Composition(
        name="functiongemma_specialist",
        description="compiler + functiongemma specialist",
        mode="compiler",
        roles=("general_function_caller", "tiny_action_specialist"),
        primary=("functiongemma",),
    ),
    "rules_baseline": Composition(
        name="rules_baseline",
        description="deterministic compiler alone, first legal action",
        mode="rules",
        roles=("bounded_scorer",),
    ),
    "logistic_baseline": Composition(
        name="logistic_baseline",
        description="compiler + hand-frozen logistic scorer",
        mode="compiler",
        roles=("bounded_scorer",),
        primary=("logistic_baseline",),
    ),
}

#: Single-letter aliases from issue #23 (A–F).
COMPOSITION_ALIASES: dict[str, str] = {
    "A": "A_qwen9b_alone",
    "B": "B_nemotron_alone",
    "C": "C_compiler_nemotron",
    "D": "D_compiler_jev_nemotron",
    "E": "E_compiler_jev_nemotron_qwen9b",
    "F": "F_compiler_jev_nemotron_qwen9b_specialists",
}

DEFAULT_COMPOSITION_NAMES: tuple[str, ...] = tuple(COMPOSITIONS)


def resolve_composition_names(names: Iterable[str] | None) -> list[str]:
    """Resolve a user list (names or A–F aliases) to frozen composition names.

    Unknown names raise; an empty/None list returns every composition.
    """
    if names is None:
        return list(DEFAULT_COMPOSITION_NAMES)
    resolved: list[str] = []
    requested = [n.strip() for n in names if n and n.strip()]
    if not requested:
        return list(DEFAULT_COMPOSITION_NAMES)
    for name in requested:
        target = COMPOSITION_ALIASES.get(name, name)
        if target not in COMPOSITIONS:
            raise KeyError(f"unknown composition: {name} (known: {', '.join(COMPOSITIONS)})")
        if target not in resolved:
            resolved.append(target)
    return resolved


def selected_compositions(names: Iterable[str] | None = None) -> list[Composition]:
    return [COMPOSITIONS[n] for n in resolve_composition_names(names)]
