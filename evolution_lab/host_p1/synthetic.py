"""Reproducible synthetic causal experiment with planted log-odds effects.

A confounded schedule selects upward memory-pressure shocks. The state risk
depends on observed fractions; treatment affects outcome risk, not the states.
The treatment lasts EFFECT_WINDOW_S one-second windows from its timestamp.
No real telemetry, machine identifiers, host access, or file I/O is used.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from .contracts import Intervention, Outcome, StateFrame

EFFECT_WINDOW_S = 8
TRUE_LOGIT_EFFECTS = {"harmful": 2.2, "helpful": -2.4, "neutral": 0.0}


def _check_seed(seed: int) -> None:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")


def _pressure_path(n_steps: int, seed: int) -> np.ndarray:
    _check_seed(seed)
    if type(n_steps) is not int or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    rng = np.random.default_rng([seed, 0])
    pressure = np.empty(n_steps, dtype=float)
    pressure[0] = float(np.clip(0.64 + rng.normal(0, 0.08), 0.12, 0.97))
    for t in range(1, n_steps):
        pressure[t] = float(np.clip(
            0.64 + 0.90 * (pressure[t - 1] - 0.64) + rng.normal(0, 0.085),
            0.12,
            0.97,
        ))
    return pressure


def _generate_states(n_steps: int, seed: int) -> list[StateFrame]:
    rng = np.random.default_rng([seed, 1])
    states = []
    for t, value in enumerate(_pressure_path(n_steps, seed)):
        mem = float(value)
        states.append(StateFrame(
            t=t,
            mem_used_frac=mem,
            swap_used_frac=float(np.clip(0.08 + 0.72 * mem + rng.normal(0, 0.04), 0, 1)),
            zram_used_frac=float(np.clip(0.11 + 0.47 * mem + rng.normal(0, 0.05), 0, 1)),
            psi_mem_some_avg10=float(np.clip(1.25 * (mem - 0.25) + rng.normal(0, 0.05), 0, 1)),
            n_procs=int(30 + 70 * mem + rng.poisson(4)),
            ui_cgroup_mem_frac=float(np.clip(0.12 + 0.72 * mem + rng.normal(0, 0.04), 0, 1)),
        ))
    return states


def _sigmoid(value: float | np.ndarray) -> np.ndarray:
    z = np.clip(value, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def _state_logit(state: StateFrame) -> float:
    return (
        -9.0
        + 10.0 * state.mem_used_frac
        + 4.0 * state.swap_used_frac
        - 3.0 * state.zram_used_frac
    )


def simulate(
    n_steps: int,
    interventions: Sequence[Intervention],
    seed: int,
) -> tuple[list[StateFrame], list[Outcome]]:
    """Sample states and per-second outcomes from known logistic structural equations.

    Only effectively applied interventions affect the OOM logit. Co-timed
    effects add, and unmatched intended/effective settings do nothing.
    Random streams for state and outcomes are distinct and deterministic.
    """
    states = _generate_states(n_steps, seed)
    logit_delta = np.zeros(n_steps, dtype=float)
    for event in interventions:
        if not 0 <= event.t < n_steps:
            raise ValueError("intervention timestamp outside simulation")
        if not event.effective_matches_intent:
            continue
        if event.knob not in TRUE_LOGIT_EFFECTS:
            raise ValueError("unknown synthetic knob: " + event.knob)
        logit_delta[event.t:min(n_steps, event.t + EFFECT_WINDOW_S)] += (
            TRUE_LOGIT_EFFECTS[event.knob] * event.effective_value
        )

    rng = np.random.default_rng([seed, 2])
    outcomes = []
    for state, delta in zip(states, logit_delta):
        oom = bool(rng.random() < float(_sigmoid(_state_logit(state) + delta)))
        stall_logit = -7.0 + 5.0 * state.mem_used_frac + 3.0 * state.psi_mem_some_avg10
        stall = bool(rng.random() < float(_sigmoid(stall_logit + 0.7 * oom)))
        outcomes.append(Outcome(
            t=state.t, oom_kill_in_window=oom, ui_stall_in_window=stall, window_s=1
        ))
    return states, outcomes


def make_confounded_interventions(
    n_steps: int,
    seed: int,
    *,
    knobs: tuple[str, ...] = ("harmful", "helpful"),
    pre_s: int = 8,
    post_s: int = EFFECT_WINDOW_S,
) -> list[Intervention]:
    """Select rising-pressure moments; do not randomize treatment times.

    Peak positive innovations inside separated blocks make treated periods
    systematically riskier than the preceding windows. The same state seed
    reproduces the schedule in simulate; only synthetic knob names are legal.
    """
    if pre_s < 1 or post_s < 1 or post_s > EFFECT_WINDOW_S:
        raise ValueError("invalid pre/post window")
    if not knobs or any(knob not in TRUE_LOGIT_EFFECTS for knob in knobs):
        raise ValueError("knobs must be known synthetic knobs")
    pressure = _pressure_path(n_steps, seed)
    events: list[Intervention] = []
    stride = max(28, pre_s + post_s + 12)
    for start in range(pre_s + 6, n_steps - post_s - 12, stride):
        end = start + 12
        innovations = pressure[start:end] - pressure[start - 1:end - 1]
        t = int(start + np.argmax(innovations))
        if pressure[t] < 0.68 or pressure[t] - pressure[t - 1] < 0.065:
            continue
        knob = knobs[len(events) % len(knobs)]
        events.append(Intervention(
            t=t, knob=knob, intended_value=1.0, effective_value=1.0,
            group_id="synthetic-group-" + str(len(events)),
        ))
    return events


def planted_probability_effect(
    states: Sequence[StateFrame],
    interventions: Sequence[Intervention],
    *,
    post_s: int = EFFECT_WINDOW_S,
) -> dict[str, float]:
    """Oracle average risk difference, NOT a measured estimate.

    The planted effects in TRUE_LOGIT_EFFECTS are log-odds increments;
    this converts them to probability-point changes for comparison with
    intervention_effect. Only single-knob, non-overlapping events are valid.
    """
    if not 1 <= post_s <= EFFECT_WINDOW_S:
        raise ValueError("invalid post window")
    indexed = {state.t: state for state in states}
    by_knob: dict[str, list[float]] = defaultdict(list)
    for event in interventions:
        if not event.effective_matches_intent:
            continue
        if event.knob not in TRUE_LOGIT_EFFECTS:
            raise ValueError("unknown synthetic knob")
        window = [indexed.get(t) for t in range(event.t, event.t + post_s)]
        if any(state is None for state in window):
            continue
        delta = TRUE_LOGIT_EFFECTS[event.knob] * event.effective_value
        individual = [
            float(_sigmoid(_state_logit(state) + delta) - _sigmoid(_state_logit(state)))
            for state in window
            if state is not None
        ]
        by_knob[event.knob].append(float(np.mean(individual)))
    return {knob: float(np.mean(effects)) for knob, effects in by_knob.items()}
