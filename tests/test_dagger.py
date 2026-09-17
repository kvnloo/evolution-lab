from __future__ import annotations

import unittest

from evolution_lab.dagger import (
    aggregate_dagger_episodes,
    mean_closed_loop_reward,
    rollout_student,
    run_dagger,
)
from evolution_lab.engine import seed_genomes
from evolution_lab.gym import make_env, rollout_teacher
from evolution_lab.models import fit_student
from evolution_lab.schema import Architecture, Curriculum, ExperimentGenome
from evolution_lab.task import build_task


class DaggerTests(unittest.TestCase):
    def test_teacher_closed_loop_still_perfect(self):
        env = make_env("hermes_recovery", delayed_cue=True)
        rewards = [rollout_teacher(env, seed=s)[1] for s in range(12)]
        self.assertTrue(all(r == 1.0 for r in rewards))

    def test_student_closed_loop_is_measurable(self):
        g = ExperimentGenome(
            id="dagger-smoke",
            lineage="mushroom-body",
            hypothesis="P3 closed-loop gap",
            architecture=Architecture(family="local_plasticity", hidden=48, history=8),
            curriculum=Curriculum(delayed_cue=True),
        )
        data = build_task(g, n_train=96, n_val=16, n_confirm=16, n_ood=8)
        student = fit_student(g, data.train)
        env = make_env("hermes_recovery", delayed_cue=True, history=8)
        mean = mean_closed_loop_reward(env, student, list(range(16)))
        self.assertGreaterEqual(mean, 0.0)
        self.assertLessEqual(mean, 1.0)

    def test_dagger_round_improves_or_matches_baseline(self):
        g = next(gen for gen in seed_genomes() if gen.architecture.family == "local_plasticity")
        data = build_task(g, n_train=128, n_val=16, n_confirm=16, n_ood=8)
        report = run_dagger(
            g,
            base_train=data.train,
            dagger_seeds=list(range(32, 48)),
            eval_seeds=list(range(0, 16)),
            rounds=1,
        )
        self.assertEqual(report["family"], "local_plasticity")
        self.assertGreaterEqual(report["student_mean_reward_after"], report["student_mean_reward_before"])
        self.assertGreaterEqual(report["student_mean_reward_after"], 0.9)
        self.assertEqual(report["teacher_mean_reward"], 1.0)
        self.assertEqual(len(report["history"]), 1)

    def test_aggregate_dagger_episodes_have_expert_labels(self):
        g = ExperimentGenome(
            id="dagger-agg",
            lineage="mushroom-body",
            hypothesis="aggregate",
            architecture=Architecture(family="local_plasticity", hidden=32, history=8),
            curriculum=Curriculum(delayed_cue=True),
        )
        data = build_task(g, n_train=64, n_val=8, n_confirm=8, n_ood=4)
        student = fit_student(g, data.train)
        env = make_env("hermes_recovery", delayed_cue=True, history=8)
        eps = aggregate_dagger_episodes(env, student, [0, 1, 2])
        self.assertEqual(len(eps), 3)
        for ep in eps:
            self.assertEqual(ep.frames.shape[0], ep.labels.shape[0])
