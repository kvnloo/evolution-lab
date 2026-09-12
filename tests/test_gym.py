from __future__ import annotations

import unittest
import numpy as np

from evolution_lab.gym import (
    FORK_TRAPS,
    HermesRecoveryEnv,
    make_env,
    n_features,
    rollout_teacher,
)
from evolution_lab.schema import ACTIONS, GenomeError, SECRET_FIELD_NAMES


class GymTests(unittest.TestCase):
    def test_spaces_match_task_layout(self):
        env = make_env("hermes_recovery")
        self.assertEqual(env.action_space.n, len(ACTIONS))
        self.assertEqual(env.observation_space.shape, (n_features(),))
        obs, info = env.reset(seed=0)
        self.assertEqual(obs.shape, (n_features(),))
        self.assertIn("expert", info)

    def test_teacher_closed_loop_is_perfect(self):
        env = make_env("hermes_recovery", delayed_cue=True)
        rewards = []
        for seed in range(24):
            _ep, mean = rollout_teacher(env, seed=seed)
            rewards.append(mean)
        self.assertTrue(all(r == 1.0 for r in rewards))

    def test_gymnasium_five_tuple(self):
        env = HermesRecoveryEnv(history=4, delayed_cue=False)
        obs, _info = env.reset(seed=1)
        terminated = False
        for _ in range(env.history):
            obs, reward, terminated, truncated, info = env.step(env.expert_action(obs))
            self.assertIsInstance(reward, float)
            self.assertIsInstance(terminated, bool)
            self.assertFalse(truncated)
            self.assertIn("expert", info)
        self.assertTrue(terminated)

    def test_secrets_fail_closed(self):
        with self.assertRaises(GenomeError):
            HermesRecoveryEnv(include_secrets=True)
        env = HermesRecoveryEnv()
        with self.assertRaises(GenomeError):
            env.reset(options={"include_secrets": True})

    def test_reset_rejects_every_secret_option_key(self):
        env = HermesRecoveryEnv()
        for name in SECRET_FIELD_NAMES:
            with self.subTest(name=name):
                with self.assertRaises(GenomeError):
                    env.reset(options={name: "x"})

    def test_actions_tuple_locked(self):
        self.assertEqual(
            ACTIONS,
            ("retry", "restart_sandbox", "escalate", "noop", "page_human"),
        )

    def test_fork_traps_are_not_p0(self):
        for name in ("flygym", "flygym_gymnasium", "openevolve", "openenv", "neuromechfly"):
            with self.subTest(name=name):
                with self.assertRaises(GenomeError) as ctx:
                    make_env(name)
                self.assertTrue(str(ctx.exception))
        self.assertIn("flygym", FORK_TRAPS)
        with self.assertRaises(GenomeError):
            make_env("cartpole")
