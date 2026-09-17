"""P3 DAgger scaffold: train on states the student visits in closed loop.

Last-step supervised training can hit confirm 1.00 while closed-loop reward stays
~0.56 on hermes_recovery. DAgger aggregates expert labels on student rollouts and
retrains. Bounded RL stays fail-closed until tinker_rl is wired.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .gym import HermesRecoveryEnv, make_env, rollout_teacher
from .models import FittedStudent, fit_student
from .schema import ACTIONS, ExperimentGenome
from .splits import load_splits
from .task import Episode, episode_from_prefix


def predict_step(
    student: FittedStudent,
    frames: list[np.ndarray],
    *,
    history: int,
) -> int:
    """One closed-loop action from frames collected so far (inclusive)."""
    ep = episode_from_prefix(frames, history=history)
    return int(student.predict_fn([ep])[0])


def rollout_student(
    env: HermesRecoveryEnv,
    student: FittedStudent,
    *,
    seed: int,
) -> tuple[Episode, float]:
    """Closed-loop student rollout. Labels are the actions the student took."""
    obs, info = env.reset(seed=seed)
    frames: list[np.ndarray] = []
    labels: list[int] = []
    rewards: list[float] = []
    for _ in range(env.history):
        frames.append(obs)
        action = predict_step(student, frames, history=env.history)
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


def expert_labeled_rollout(
    env: HermesRecoveryEnv,
    student: FittedStudent,
    *,
    seed: int,
) -> Episode:
    """DAgger dataset row: student visits states; labels are expert corrections."""
    obs, info = env.reset(seed=seed)
    frames: list[np.ndarray] = []
    expert_labels: list[int] = []
    for _ in range(env.history):
        frames.append(obs)
        expert_labels.append(env.expert_action(obs))
        action = predict_step(student, frames, history=env.history)
        obs, _reward, terminated, _trunc, _info = env.step(action)
        if terminated:
            break
    return Episode(
        np.stack(frames),
        np.asarray(expert_labels, dtype=np.int64),
        env=str(info.get("env", "iid")),
    )


def aggregate_dagger_episodes(
    env: HermesRecoveryEnv,
    student: FittedStudent,
    seeds: list[int],
) -> list[Episode]:
    return [expert_labeled_rollout(env, student, seed=s) for s in seeds]


def mean_closed_loop_reward(
    env: HermesRecoveryEnv,
    student: FittedStudent,
    seeds: list[int],
) -> float:
    if not seeds:
        return 0.0
    rewards = [rollout_student(env, student, seed=s)[1] for s in seeds]
    return float(np.mean(rewards))


@dataclass
class DaggerRound:
    round_index: int
    n_train: int
    n_dagger: int
    student_mean_reward: float
    teacher_mean_reward: float


def run_dagger(
    genome: ExperimentGenome,
    *,
    base_train: list[Episode],
    dagger_seeds: list[int],
    eval_seeds: list[int],
    rounds: int = 2,
    delayed_cue: bool = True,
) -> dict[str, Any]:
    """Iterative DAgger on locked Hermes recovery. Returns before/after closed-loop."""
    env = make_env("hermes_recovery", delayed_cue=delayed_cue, history=genome.architecture.history)
    teacher_rewards = [rollout_teacher(env, seed=s)[1] for s in eval_seeds]
    teacher_mean = float(np.mean(teacher_rewards)) if teacher_rewards else 0.0

    warm_start = genome.architecture.family == "local_plasticity"
    student = fit_student(genome, base_train)
    before = mean_closed_loop_reward(env, student, eval_seeds)

    history: list[DaggerRound] = []
    train_pool = list(base_train)
    for r in range(rounds):
        dagger_eps = aggregate_dagger_episodes(env, student, dagger_seeds)
        train_pool = train_pool + dagger_eps
        student = fit_student(
            genome,
            train_pool,
            init_extras=student.extras if warm_start else None,
        )
        student_mean = mean_closed_loop_reward(env, student, eval_seeds)
        history.append(
            DaggerRound(
                round_index=r + 1,
                n_train=len(train_pool),
                n_dagger=len(dagger_eps),
                student_mean_reward=student_mean,
                teacher_mean_reward=teacher_mean,
            )
        )

    after = history[-1].student_mean_reward if history else before
    return {
        "family": genome.architecture.family,
        "rounds": rounds,
        "eval_seeds": len(eval_seeds),
        "dagger_seeds": len(dagger_seeds),
        "teacher_mean_reward": teacher_mean,
        "student_mean_reward_before": before,
        "student_mean_reward_after": after,
        "history": [round.__dict__ for round in history],
        "note": "DAgger scaffold: expert labels on student-visited states; RL backend still fail-closed.",
    }


def default_dagger_smoke(genome: ExperimentGenome | None = None, *, rounds: int = 2) -> dict[str, Any]:
    from .engine import seed_genomes

    genome = genome or next(g for g in seed_genomes() if g.architecture.family == "local_plasticity")
    data = load_splits()
    dagger_seeds = list(range(32, 32 + 24))
    eval_seeds = list(range(0, 24))
    return run_dagger(
        genome,
        base_train=data.train,
        dagger_seeds=dagger_seeds,
        eval_seeds=eval_seeds,
        rounds=rounds,
    )
