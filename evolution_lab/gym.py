"""Gymnasium-style env API without vendoring Gymnasium, FlyGym, OpenEvolve, or TMNF-C.

P0 gym is Hermes recovery. FlyGym / OpenEnv / OpenEvolve / TMNF-C are named traps:
consume later, do not replace this engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .schema import ACTIONS, GenomeError, options_carry_secrets
from .task import N_ACTIONS, Episode, _frame, n_features, teacher_action


class Discrete:
    def __init__(self, n: int) -> None:
        self.n = int(n)

    def sample(self, rng: np.random.Generator) -> int:
        return int(rng.integers(0, self.n))


class Box:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = tuple(shape)
        self.dtype = np.float64


class Env(Protocol):
    """Subset of Gymnasium Env: reset → (obs, info), step → 5-tuple."""

    observation_space: Box
    action_space: Discrete

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]: ...

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]: ...

    def close(self) -> None: ...


# Declared names that look like a shortcut and are not P0.
_TMNF_C = (
    "TMNF-C is a TrackMania Forever vector env with an optional MaleCNS mushroom-body "
    "dopamine learner (kvnloo/TMNF-C, fork of adonis-singh/TMNF-C). Consume later as a "
    "visuo-motor gym; it is not Hermes recovery. Do not fork it into this engine. "
    "The MB learner projects engineered car state into PNs; frozen weights rarely finish."
)
FORK_TRAPS = {
    "flygym": (
        "FlyGym 2.x is NeuroMechFly (body + MuJoCo/Warp). Consume it later as a "
        "visuo-motor env; it is not Hermes recovery and not this experiment engine."
    ),
    "flygym-gymnasium": (
        "flygym-gymnasium is the legacy 1.x Gymnasium API. FlyGym 2.x left that "
        "contract on purpose (~10×). Do not fork v1 to speed up P0."
    ),
    "neuromechfly": (
        "NeuroMechFly walks without a brain. Digital Sphinx: walking ≠ recovery contract."
    ),
    "openevolve": (
        "OpenEvolve evolves LLM-rewritten programs under combined_score. It is a "
        "later mutation backend (research-strategy), not a gym and not the control table."
    ),
    "openenv": (
        "OpenEnv is a Docker/HTTP agent-env standard. Adopt reset/step over HTTP later "
        "for real Hermes sandboxes; P0 stays in-process."
    ),
    "tmnf": _TMNF_C,
    "tmnf-c": _TMNF_C,
    "tmnf-fly": _TMNF_C,
    "trackmania": _TMNF_C,
}


@dataclass
class HermesRecoveryEnv:
    """Typed recovery events → discrete action. Same feature layout as task.py."""

    history: int = 8
    delayed_cue: bool = True
    ood: bool = False
    include_secrets: bool = False

    def __post_init__(self) -> None:
        if self.include_secrets:
            raise GenomeError("secret-bearing observations fail closed at L0")
        self.observation_space = Box((n_features(),))
        self.action_space = Discrete(N_ACTIONS)
        self._rng = np.random.default_rng()
        self._t = 0
        self._last = ACTIONS.index("noop")
        self._retry = 0
        self._sandbox = 1.0
        self._budget = 1.0
        self._kind = 0
        self._unfamiliar = 0.0
        self._policy_hit = 0.0
        self._cue_on = False
        self._cue_seen = False
        self._closed = False

    def close(self) -> None:
        self._closed = True

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if self._closed:
            raise GenomeError("env is closed")
        options = options or {}
        if options_carry_secrets(options):
            raise GenomeError("secret-bearing observations fail closed at L0")
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if "ood" in options:
            self.ood = bool(options["ood"])
        self._t = 0
        self._last = ACTIONS.index("noop")
        self._retry = 0
        self._sandbox = 1.0
        self._budget = float(self._rng.uniform(0.15, 1.0))
        self._unfamiliar = float(self._rng.random() < 0.2)
        self._policy_hit = 0.0
        self._kind = 5 if self.ood else int(self._rng.integers(0, 5))
        self._cue_on = bool(self.delayed_cue and self._rng.random() < 0.4)
        self._cue_seen = False
        obs = self._obs()
        info = {
            "kind": self._kind,
            "env": "ood" if self.ood else "iid",
            "expert": self.expert_action(obs),
            "cue_on": self._cue_on,
        }
        return obs, info

    def _obs(self) -> np.ndarray:
        T = self.history
        elapsed = self._t / max(T - 1, 1)
        transient = hard = already_ok = 0.0
        sandbox = self._sandbox
        policy_hit = self._policy_hit
        unfamiliar = self._unfamiliar
        if self._kind == 0:
            already_ok = 1.0
        elif self._kind == 1:
            transient = 1.0
        elif self._kind == 2:
            hard = 1.0
        elif self._kind == 3:
            sandbox = 0.0
        elif self._kind == 4:
            policy_hit = 1.0
        else:
            hard = 1.0
            unfamiliar = 1.0
        cue = 0.0
        if self.delayed_cue and self._cue_on and self._t == 0:
            cue = 1.0
            self._cue_seen = True
        return _frame(
            sandbox_alive=sandbox,
            retry=self._retry,
            budget=self._budget,
            elapsed=elapsed,
            transient=transient,
            hard=hard,
            unfamiliar=unfamiliar,
            policy_hit=policy_hit,
            already_ok=already_ok,
            cue=cue,
            last_action=self._last,
        )

    def expert_action(self, obs: np.ndarray) -> int:
        if self.delayed_cue and self._cue_on and self._t == self.history - 1:
            return ACTIONS.index("escalate")
        return teacher_action(obs)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._closed:
            raise GenomeError("env is closed")
        action = int(action)
        if not 0 <= action < N_ACTIONS:
            raise GenomeError(f"action {action} outside Discrete({N_ACTIONS})")
        obs = self._obs()
        expert = self.expert_action(obs)
        reward = 1.0 if action == expert else 0.0
        violation = bool(obs[7] > 0.5 and action != ACTIONS.index("page_human"))
        if action == ACTIONS.index("retry"):
            self._retry = min(self._retry + 1, 4)
        if action == ACTIONS.index("restart_sandbox"):
            self._sandbox = 1.0
        self._last = action
        self._t += 1
        terminated = self._t >= self.history
        truncated = False
        next_obs = self._obs() if not terminated else obs
        info = {
            "expert": expert,
            "violation": violation,
            "env": "ood" if self.ood else "iid",
        }
        return next_obs, reward, terminated, truncated, info


def make_env(name: str, **kwargs: Any) -> HermesRecoveryEnv:
    slug = name.strip().lower().replace("_", "-")
    if slug in FORK_TRAPS:
        raise GenomeError(FORK_TRAPS[slug])
    if slug in {"hermes-recovery", "hermes", "p0"}:
        return HermesRecoveryEnv(**kwargs)
    raise GenomeError(
        f"unknown env {name!r}; P0 env is hermes_recovery. "
        "Do not fork FlyGym, OpenEvolve, or TMNF-C as this engine."
    )


def teacher_policy(env: HermesRecoveryEnv, obs: np.ndarray) -> int:
    return env.expert_action(obs)


def rollout_teacher(env: HermesRecoveryEnv, *, seed: int) -> tuple[Episode, float]:
    """Closed-loop teacher. Mean reward is 1.0 when the expert is consistent."""
    obs, info = env.reset(seed=seed)
    frames = []
    labels = []
    rewards = []
    for _ in range(env.history):
        action = teacher_policy(env, obs)
        frames.append(obs)
        labels.append(action)
        obs, reward, terminated, _trunc, info = env.step(action)
        rewards.append(reward)
        if terminated:
            break
    ep = Episode(
        np.stack(frames),
        np.asarray(labels, dtype=np.int64),
        env=str(info.get("env", "iid")),
    )
    return ep, float(np.mean(rewards)) if rewards else 0.0
