"""Hermes recovery task: typed events, bounded actions, external verifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .schema import ACTIONS, ExperimentGenome

# Same integer as evolution_lab.splits.SPLIT_SEED. Canonical P0 lock seed.
SPLIT_SEED = 20260912

N_ACTIONS = len(ACTIONS)
# sandbox_alive, retry_norm, budget, elapsed, transient, hard, unfamiliar,
# policy_hit, already_ok, cue
N_BASE = 10


def n_features(history_onehot: int = N_ACTIONS) -> int:
    return N_BASE + history_onehot


@dataclass
class Episode:
    frames: np.ndarray  # (T, F)
    labels: np.ndarray  # (T,) int action ids
    env: str
    secret: bool = False


def teacher_action(frame: np.ndarray) -> int:
    """Reference policy. Indices match feature layout."""
    sandbox_alive, retry_norm, budget, _elapsed = frame[:4]
    transient, hard, unfamiliar, policy_hit, already_ok, cue = frame[4:10]
    if policy_hit > 0.5:
        return ACTIONS.index("page_human")
    if sandbox_alive < 0.5:
        return ACTIONS.index("restart_sandbox")
    if cue > 0.5:
        return ACTIONS.index("escalate")
    if unfamiliar > 0.5 or (hard > 0.5 and budget < 0.3):
        return ACTIONS.index("escalate")
    if transient > 0.5 and retry_norm < 0.5:
        return ACTIONS.index("retry")
    if already_ok > 0.5:
        return ACTIONS.index("noop")
    if hard > 0.5:
        return ACTIONS.index("escalate")
    return ACTIONS.index("retry")


def _one_hot(action: int) -> np.ndarray:
    v = np.zeros(N_ACTIONS, dtype=np.float64)
    v[action] = 1.0
    return v


def _frame(
    *,
    sandbox_alive: float,
    retry: int,
    budget: float,
    elapsed: float,
    transient: float,
    hard: float,
    unfamiliar: float,
    policy_hit: float,
    already_ok: float,
    cue: float,
    last_action: int,
) -> np.ndarray:
    base = np.array(
        [
            sandbox_alive,
            retry / 4.0,
            budget,
            elapsed,
            transient,
            hard,
            unfamiliar,
            policy_hit,
            already_ok,
            cue,
        ],
        dtype=np.float64,
    )
    return np.concatenate([base, _one_hot(last_action)])


def make_episode(rng: np.random.Generator, *, T: int, delayed_cue: bool, ood: bool) -> Episode:
    last = ACTIONS.index("noop")
    frames = []
    labels = []
    cue_on = bool(delayed_cue and rng.random() < 0.4)
    sandbox = 1.0
    retry = 0
    budget = float(rng.uniform(0.15, 1.0))
    unfamiliar = float(rng.random() < 0.2)
    policy_hit = 0.0
    kind = rng.integers(0, 5)
    if ood:
        kind = 5  # unseen network-class failure: treated as hard+unfamiliar mix
    for t in range(T):
        elapsed = t / max(T - 1, 1)
        transient = hard = already_ok = 0.0
        if kind == 0:
            already_ok = 1.0
        elif kind == 1:
            transient = 1.0
        elif kind == 2:
            hard = 1.0
        elif kind == 3:
            sandbox = 0.0
        elif kind == 4:
            policy_hit = 1.0
        else:
            hard = 1.0
            unfamiliar = 1.0
        cue = 0.0
        if delayed_cue and cue_on and t == 0:
            cue = 1.0
        fr = _frame(
            sandbox_alive=sandbox,
            retry=retry,
            budget=budget,
            elapsed=elapsed,
            transient=transient,
            hard=hard,
            unfamiliar=unfamiliar,
            policy_hit=policy_hit,
            already_ok=already_ok,
            cue=cue,
            last_action=last,
        )
        if delayed_cue and cue_on and t == T - 1:
            y = ACTIONS.index("escalate")
        else:
            y = teacher_action(fr)
        frames.append(fr)
        labels.append(y)
        last = y
        if y == ACTIONS.index("retry"):
            retry = min(retry + 1, 4)
        if y == ACTIONS.index("restart_sandbox"):
            sandbox = 1.0
    env = "ood" if ood else "iid"
    return Episode(np.stack(frames), np.asarray(labels, dtype=np.int64), env=env)


def make_split(
    rng: np.random.Generator,
    n: int,
    *,
    T: int,
    delayed_cue: bool,
    ood: bool,
) -> list[Episode]:
    return [make_episode(rng, T=T, delayed_cue=delayed_cue, ood=ood) for _ in range(n)]


@dataclass
class TaskData:
    train: list[Episode]
    val: list[Episode]
    confirm: list[Episode]
    ood: list[Episode]


def sample_task(
    genome: ExperimentGenome,
    *,
    n_train: int = 192,
    n_val: int = 48,
    n_confirm: int = 48,
    n_ood: int = 32,
    seed: int = SPLIT_SEED,
) -> TaskData:
    """Draw splits from seed. Prefer build_task so L1 can load locked arrays."""
    T = genome.architecture.history
    delayed = genome.curriculum.delayed_cue
    rng = np.random.default_rng(seed)
    train = make_split(rng, n_train, T=T, delayed_cue=delayed, ood=False)
    val = make_split(rng, n_val, T=T, delayed_cue=delayed, ood=False)
    confirm = make_split(rng, n_confirm, T=T, delayed_cue=delayed, ood=False)
    ood = make_split(rng, n_ood, T=T, delayed_cue=delayed, ood=True)
    return TaskData(train, val, confirm, ood)


def build_task(
    genome: ExperimentGenome,
    *,
    n_train: int = 192,
    n_val: int = 48,
    n_confirm: int = 48,
    n_ood: int = 32,
    splits_dir: Path | None = None,
) -> TaskData:
    """Load data/p0/ when it matches this genome so L1 is not a drifting RNG draw."""
    from .splits import try_load_for_genome

    locked = try_load_for_genome(
        genome,
        n_train=n_train,
        n_val=n_val,
        n_confirm=n_confirm,
        n_ood=n_ood,
        splits_dir=splits_dir,
    )
    if locked is not None:
        return locked
    return sample_task(
        genome,
        n_train=n_train,
        n_val=n_val,
        n_confirm=n_confirm,
        n_ood=n_ood,
        seed=SPLIT_SEED,
    )


def apply_strobe(episode: Episode, drop: float, rng: np.random.Generator) -> Episode:
    if drop <= 0:
        return episode
    frames = episode.frames.copy()
    for t in range(frames.shape[0] - 1):
        if rng.random() < drop:
            frames[t] = 0.0
    return Episode(frames, episode.labels, episode.env, episode.secret)


def stack_last(episodes: list[Episode]) -> tuple[np.ndarray, np.ndarray]:
    X = np.stack([ep.frames[-1] for ep in episodes])
    y = np.stack([ep.labels[-1] for ep in episodes])
    return X, y


def stack_flat(episodes: list[Episode]) -> tuple[np.ndarray, np.ndarray]:
    X = np.stack([ep.frames.reshape(-1) for ep in episodes])
    y = np.stack([ep.labels[-1] for ep in episodes])
    return X, y
