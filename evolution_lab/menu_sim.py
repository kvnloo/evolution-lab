"""Seeded, budget-bounded simulation of parallel experiment-arm pruning.

All policies see observed outcomes and in-flight counts, never latent probabilities.
No fleet actions, file writes, or global random state are used.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
from dataclasses import dataclass
from statistics import fmean, median
from typing import Callable, Protocol, Sequence

import numpy as np


@dataclass
class SimulationState:
    """Observable policy state. Pending trials are paid for but not yet observed."""

    counts: list[int]
    successes: list[int]
    pending: list[int]
    live_arms: set[int]
    costs: tuple[float, ...]
    budget: float
    spent: float
    rng: np.random.Generator

    @property
    def total_trials(self) -> int:
        return sum(self.counts)

    def mean(self, arm: int) -> float:
        return self.successes[arm] / self.counts[arm] if self.counts[arm] else 0.5

    def affordable(self, arm: int) -> bool:
        return self.costs[arm] <= self.budget - self.spent


class Policy(Protocol):
    name: str

    def choose(self, state: SimulationState, rng: np.random.Generator) -> int | None:
        """Choose an affordable live arm, or None when no trial is schedulable."""
        ...

    def prune(self, state: SimulationState) -> set[int]:
        """Return live arms to eliminate using observations only."""
        ...


class UniformPolicy:
    name = "uniform"

    def __init__(self) -> None:
        self._cursor = 0

    def choose(self, state: SimulationState, rng: np.random.Generator) -> int | None:
        for offset in range(len(state.counts)):
            arm = (self._cursor + offset) % len(state.counts)
            if arm in state.live_arms and state.affordable(arm):
                self._cursor = (arm + 1) % len(state.counts)
                return arm
        return None

    def prune(self, state: SimulationState) -> set[int]:
        return set()


class SuccessiveHalvingPolicy:
    name = "successive_halving"

    def __init__(self, *, eta: int = 2, rung_trials: int = 8) -> None:
        _positive_int(eta, "eta")
        if eta < 2:
            raise ValueError("eta must be at least 2")
        _positive_int(rung_trials, "rung_trials")
        self.eta = eta
        self.rung_trials = rung_trials
        self._rung = 0
        self._start_counts: dict[int, int] | None = None

    def _targets(self, state: SimulationState) -> dict[int, int]:
        if self._start_counts is None:
            self._start_counts = {arm: state.counts[arm] for arm in state.live_arms}
        quota = self.rung_trials * self.eta ** self._rung
        return {arm: self._start_counts[arm] + quota for arm in state.live_arms}

    def choose(self, state: SimulationState, rng: np.random.Generator) -> int | None:
        targets = self._targets(state)
        candidates = [
            arm for arm in state.live_arms
            if state.affordable(arm) and state.counts[arm] + state.pending[arm] < targets[arm]
        ]
        if not candidates:
            return None
        assert self._start_counts is not None
        return min(
            candidates,
            key=lambda arm: (state.counts[arm] + state.pending[arm] - self._start_counts[arm], arm),
        )

    def prune(self, state: SimulationState) -> set[int]:
        if len(state.live_arms) <= 1:
            return set()
        targets = self._targets(state)
        if any(state.counts[arm] < targets[arm] for arm in state.live_arms):
            return set()
        keep = math.ceil(len(state.live_arms) / self.eta)
        ranked = sorted(state.live_arms, key=lambda arm: (-state.mean(arm), arm))
        survivors = set(ranked[:keep])
        removed = state.live_arms - survivors
        self._rung += 1
        self._start_counts = {arm: state.counts[arm] for arm in survivors}
        return removed


class UCB1Policy:
    name = "ucb1"

    def __init__(self, *, delta: float = 0.001) -> None:
        if not math.isfinite(delta) or not 0 < delta < 1:
            raise ValueError("delta must be finite and between 0 and 1")
        self.delta = delta

    def choose(self, state: SimulationState, rng: np.random.Generator) -> int | None:
        candidates = sorted(arm for arm in state.live_arms if state.affordable(arm))
        if not candidates:
            return None
        unobserved = [arm for arm in candidates if state.counts[arm] == 0]
        if unobserved:
            return min(unobserved, key=lambda arm: (state.pending[arm], arm))
        log_t = math.log(max(2, state.total_trials + 1))
        return max(
            candidates,
            key=lambda arm: (
                state.mean(arm)
                + math.sqrt(2 * log_t / (state.counts[arm] + state.pending[arm])),
                -arm,
            ),
        )

    def prune(self, state: SimulationState) -> set[int]:
        if len(state.live_arms) <= 1:
            return set()
        lower: dict[int, float] = {}
        upper: dict[int, float] = {}
        for arm in state.live_arms:
            n = state.counts[arm]
            if n == 0:
                lower[arm], upper[arm] = 0.0, 1.0
                continue
            radius = math.sqrt(math.log(2 * len(state.counts) / self.delta) / (2 * n))
            lower[arm] = max(0.0, state.mean(arm) - radius)
            upper[arm] = min(1.0, state.mean(arm) + radius)
        best_lower = max(lower.values())
        return {arm for arm in state.live_arms if upper[arm] < best_lower}


class ThompsonPolicy:
    name = "thompson"

    def __init__(
        self, *, epsilon: float = 0.005, mc_samples: int = 512, min_trials: int = 12
    ) -> None:
        if not math.isfinite(epsilon) or not 0 < epsilon < 1:
            raise ValueError("epsilon must be finite and between 0 and 1")
        _positive_int(mc_samples, "mc_samples")
        _positive_int(min_trials, "min_trials")
        self.epsilon = epsilon
        self.mc_samples = mc_samples
        self.min_trials = min_trials
        self._last_check = -1

    def choose(self, state: SimulationState, rng: np.random.Generator) -> int | None:
        candidates = sorted(arm for arm in state.live_arms if state.affordable(arm))
        if not candidates:
            return None
        warmup = [
            arm for arm in candidates
            if state.counts[arm] + state.pending[arm] < self.min_trials
        ]
        if warmup:
            return min(warmup, key=lambda arm: (state.counts[arm] + state.pending[arm], arm))
        draws = rng.beta(
            [1 + state.successes[arm] for arm in candidates],
            [1 + state.counts[arm] - state.successes[arm] for arm in candidates],
        )
        return candidates[int(np.argmax(draws))]

    def prune(self, state: SimulationState) -> set[int]:
        arms = sorted(state.live_arms)
        if len(arms) <= 1 or any(state.counts[arm] < self.min_trials for arm in arms):
            return set()
        # A check per approximately one new observation per live arm bounds MC work.
        if self._last_check >= 0 and state.total_trials - self._last_check < len(arms):
            return set()
        self._last_check = state.total_trials
        draws = state.rng.beta(
            [1 + state.successes[arm] for arm in arms],
            [1 + state.counts[arm] - state.successes[arm] for arm in arms],
            size=(self.mc_samples, len(arms)),
        )
        winner_counts = np.bincount(np.argmax(draws, axis=1), minlength=len(arms))
        probabilities = winner_counts / self.mc_samples
        removed = {arm for index, arm in enumerate(arms) if probabilities[index] < self.epsilon}
        if len(removed) == len(arms):
            removed.remove(arms[int(np.argmax(probabilities))])
        return removed


def uniform() -> UniformPolicy:
    return UniformPolicy()


def successive_halving(*, eta: int = 2, rung_trials: int = 8) -> SuccessiveHalvingPolicy:
    return SuccessiveHalvingPolicy(eta=eta, rung_trials=rung_trials)


def ucb1(*, delta: float = 0.001) -> UCB1Policy:
    return UCB1Policy(delta=delta)


def thompson(
    *, epsilon: float = 0.005, mc_samples: int = 512, min_trials: int = 12
) -> ThompsonPolicy:
    return ThompsonPolicy(epsilon=epsilon, mc_samples=mc_samples, min_trials=min_trials)


POLICIES: dict[str, Callable[[], Policy]] = {
    "uniform": uniform,
    "successive_halving": successive_halving,
    "ucb1": ucb1,
    "thompson": thompson,
}


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _positive_float(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def run_simulation(
    policy: Policy,
    *,
    K: int,
    L: int,
    B: float,
    seed: int = 0,
    probabilities: Sequence[float] | None = None,
    costs: Sequence[float] | None = None,
    a: float = 1.0,
    b: float = 1.0,
) -> dict[str, object]:
    """Run one deterministic discrete-event simulation, with no external effects.

    Lane-minutes are reserved on trial dispatch. A trial completes after cost_i
    simulated minutes on its lane; outcomes become visible only on completion.
    All already-dispatched trials are drained after commitment.
    """
    _positive_int(K, "K")
    _positive_int(L, "L")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    budget = float(B)
    if not math.isfinite(budget) or budget < 0:
        raise ValueError("B must be finite and nonnegative")
    alpha, beta = _positive_float(a, "a"), _positive_float(b, "b")
    prices = tuple(1.0 for _ in range(K)) if costs is None else tuple(
        _positive_float(cost, "cost") for cost in costs
    )
    if len(prices) != K:
        raise ValueError("costs must contain exactly K entries")
    truth_rng, outcome_rng, policy_rng = (
        np.random.default_rng(part) for part in np.random.SeedSequence(seed).spawn(3)
    )
    if probabilities is None:
        latent = [float(x) for x in truth_rng.beta(alpha, beta, size=K)]
    else:
        latent = [float(value) for value in probabilities]
        if len(latent) != K or any(not math.isfinite(p) or not 0 <= p <= 1 for p in latent):
            raise ValueError("probabilities must contain K finite values between 0 and 1")
    best = int(np.argmax(latent))
    state = SimulationState(
        counts=[0] * K,
        successes=[0] * K,
        pending=[0] * K,
        live_arms=set(range(K)),
        costs=prices,
        budget=budget,
        spent=0.0,
        rng=policy_rng,
    )
    # Events are (completion time, lane number, arm, hidden Bernoulli outcome).
    events: list[tuple[float, int, int, int]] = []
    free_lanes = list(range(L))
    now = 0.0
    commit: float | None = 0.0 if K == 1 else None
    wrongly_pruned_best = False

    while True:
        if commit is None:
            removed = set(policy.prune(state))
            if not removed.issubset(state.live_arms) or removed == state.live_arms:
                raise ValueError("policy attempted to prune a nonexistent or every live arm")
            wrongly_pruned_best |= best in removed
            state.live_arms.difference_update(removed)
            if len(state.live_arms) == 1:
                commit = state.spent
        if commit is None:
            still_free: list[int] = []
            for lane in sorted(free_lanes):
                arm = policy.choose(state, state.rng)
                if arm is None:
                    still_free.append(lane)
                    continue
                if arm not in state.live_arms or not state.affordable(arm):
                    raise ValueError("policy chose a pruned or unaffordable arm")
                cost = prices[arm]
                state.spent += cost
                state.pending[arm] += 1
                outcome = int(outcome_rng.random() < latent[arm])
                heapq.heappush(events, (now + cost, lane, arm, outcome))
            free_lanes = still_free
        if not events:
            break
        now = events[0][0]
        while events and events[0][0] == now:
            _, lane, arm, outcome = heapq.heappop(events)
            state.pending[arm] -= 1
            state.counts[arm] += 1
            state.successes[arm] += outcome
            free_lanes.append(lane)

    recommendation = min(state.live_arms, key=lambda arm: (-state.mean(arm), arm))
    return {
        "policy": policy.name,
        "seed": seed,
        "best_arm_found": recommendation == best,
        "simple_regret": float(latent[best] - latent[recommendation]),
        "lane_minutes_to_commit": commit,
        "wrongly_pruned_best": wrongly_pruned_best,
        "trials_per_arm": list(state.counts),
        "successes_per_arm": list(state.successes),
        "spent_lane_minutes": state.spent,
        "elapsed_simulated_minutes": now,
        "recommendation": recommendation,
        "true_best_arm": best,
        "live_arms": sorted(state.live_arms),
        "latent_probabilities": latent,
    }


def run_experiment(
    policy_factory: Callable[[], Policy],
    K: int,
    L: int,
    B: float,
    seeds: int | Sequence[int],
    *,
    probabilities: Sequence[float] | None = None,
    costs: Sequence[float] | None = None,
    a: float = 1.0,
    b: float = 1.0,
) -> dict[str, object]:
    """Aggregate independent seeded runs; an integer seeds means range(seeds)."""
    if isinstance(seeds, int) and not isinstance(seeds, bool):
        if seeds < 1:
            raise ValueError("seeds must be positive")
        values = list(range(seeds))
    else:
        values = list(seeds)
    if not values:
        raise ValueError("seeds must not be empty")
    runs = [
        run_simulation(
            policy_factory(), K=K, L=L, B=B, seed=seed,
            probabilities=probabilities, costs=costs, a=a, b=b,
        )
        for seed in values
    ]
    committed = [run["lane_minutes_to_commit"] for run in runs
                 if run["lane_minutes_to_commit"] is not None]
    return {
        "policy": runs[0]["policy"],
        "K": K,
        "L": L,
        "B": B,
        "n_seeds": len(runs),
        "mean_regret": fmean(float(run["simple_regret"]) for run in runs),
        "p_best_found": fmean(int(run["best_arm_found"]) for run in runs),
        "p_wrongly_pruned_best": fmean(int(run["wrongly_pruned_best"]) for run in runs),
        "p_committed": len(committed) / len(runs),
        "median_lane_minutes_to_commit": median(committed) if committed else None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Simulation-only experiment-arm pruning")
    parser.add_argument("--K", type=int, default=16, help="number of arms")
    parser.add_argument("--L", type=int, default=4, help="parallel lanes")
    parser.add_argument("--B", type=float, default=400, help="total lane-minute budget")
    parser.add_argument("--seeds", type=int, default=200, help="independent seeds, 0..N-1")
    parser.add_argument("--a", type=float, default=1.0, help="Beta prior alpha")
    parser.add_argument("--b", type=float, default=1.0, help="Beta prior beta")
    parser.add_argument("--json", action="store_true", help="one JSON line per policy")
    args = parser.parse_args(argv)
    for factory in POLICIES.values():
        summary = run_experiment(
            factory, args.K, args.L, args.B, args.seeds, a=args.a, b=args.b
        )
        if args.json:
            print(json.dumps(summary, sort_keys=True, allow_nan=False))
        else:
            print(
                f"{summary['policy']}: mean_regret={summary['mean_regret']:.4f} "
                f"p_best={summary['p_best_found']:.3f} "
                f"p_wrong_prune={summary['p_wrongly_pruned_best']:.3f} "
                f"median_commit={summary['median_lane_minutes_to_commit']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
