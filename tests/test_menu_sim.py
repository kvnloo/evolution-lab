"""Acceptance and edge-case tests for the menu-pruning simulator."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest

import numpy as np

from evolution_lab.menu_sim import (
    POLICIES,
    SimulationState,
    SuccessiveHalvingPolicy,
    ThompsonPolicy,
    UCB1Policy,
    run_experiment,
    run_simulation,
    successive_halving,
    thompson,
    ucb1,
    uniform,
)


class MenuSimulatorTests(unittest.TestCase):
    def test_seeded_results_and_summaries(self) -> None:
        for name, factory in POLICIES.items():
            with self.subTest(policy=name):
                args = dict(K=4, L=2, B=0)
                first = run_simulation(factory(), seed=11, **args)
                second = run_simulation(factory(), seed=11, **args)
                different = run_simulation(factory(), seed=12, **args)
                self.assertEqual(first, second)
                self.assertNotEqual(
                    first["latent_probabilities"], different["latent_probabilities"]
                )
                summary = run_experiment(factory, 4, 2, 0, [11, 12, 13])
                self.assertEqual(summary, run_experiment(factory, 4, 2, 0, [11, 12, 13]))
                self.assertNotEqual(
                    summary["mean_regret"],
                    run_experiment(factory, 4, 2, 0, [21, 22, 23])["mean_regret"],
                )

    def test_beta_probabilities_are_seeded_and_explicit_probabilities_win(self) -> None:
        sampled = run_simulation(uniform(), K=5, L=2, B=8, seed=8, a=2, b=5)
        self.assertEqual(
            sampled["latent_probabilities"],
            run_simulation(uniform(), K=5, L=2, B=8, seed=8, a=2, b=5)[
                "latent_probabilities"
            ],
        )
        self.assertTrue(all(0 <= p <= 1 for p in sampled["latent_probabilities"]))
        given = [0.9, 0.1, 0.1, 0.1, 0.1]
        observed = run_simulation(uniform(), K=5, L=2, B=8, seed=8,
                                  probabilities=given)
        self.assertEqual(observed["latent_probabilities"], given)
        self.assertEqual(observed["true_best_arm"], 0)

    def test_uniform_never_prunes_and_balances_trials(self) -> None:
        result = run_simulation(
            uniform(), K=8, L=4, B=400, seed=5, probabilities=[0.9] + [0.1] * 7
        )
        counts = result["trials_per_arm"]
        self.assertEqual(result["live_arms"], list(range(8)))
        self.assertIsNone(result["lane_minutes_to_commit"])
        self.assertFalse(result["wrongly_pruned_best"])
        self.assertLessEqual(max(counts) - min(counts), 1)
        self.assertEqual(sum(counts), 400)
        self.assertEqual(result["spent_lane_minutes"], 400)

    def test_well_separated_best_arm_and_commit_across_200_seeds(self) -> None:
        p = [0.9] + [0.1] * 7
        uniform_summary = run_experiment(uniform, 8, 4, 400, 200, probabilities=p)
        self.assertIsNone(uniform_summary["median_lane_minutes_to_commit"])
        self.assertEqual(uniform_summary["p_committed"], 0)
        self.assertEqual(uniform_summary["p_wrongly_pruned_best"], 0)
        for name, factory in (
            ("successive_halving", successive_halving),
            ("ucb1", ucb1),
            ("thompson", thompson),
        ):
            with self.subTest(policy=name):
                summary = run_experiment(factory, 8, 4, 400, 200, probabilities=p)
                self.assertGreaterEqual(summary["p_best_found"], 0.95)
                self.assertIsNotNone(summary["median_lane_minutes_to_commit"])
                self.assertLess(summary["median_lane_minutes_to_commit"], 400)
                self.assertGreater(summary["p_committed"], 0)
                if name == "ucb1":
                    self.assertEqual(summary["p_wrongly_pruned_best"], 0)

    def test_equal_arms_have_zero_regret_for_every_policy(self) -> None:
        for name, factory in POLICIES.items():
            with self.subTest(policy=name):
                summary = run_experiment(
                    factory, 5, 3, 120, 8, probabilities=[0.42] * 5
                )
                self.assertEqual(summary["mean_regret"], 0)
                result = run_simulation(
                    factory(), K=5, L=3, B=120, seed=3, probabilities=[0.42] * 5
                )
                self.assertEqual(result["simple_regret"], 0)

    def test_heterogeneous_costs_obey_budget_on_all_policies(self) -> None:
        costs = [2.5, 1, 3, 1.5]
        for name, factory in POLICIES.items():
            with self.subTest(policy=name):
                result = run_simulation(
                    factory(), K=4, L=3, B=27.5, seed=7,
                    probabilities=[0.7, 0.5, 0.3, 0.1], costs=costs,
                )
                spent = sum(n * c for n, c in zip(result["trials_per_arm"], costs))
                self.assertAlmostEqual(spent, result["spent_lane_minutes"])
                self.assertLessEqual(spent, 27.5)
                self.assertTrue(all(n >= 0 for n in result["trials_per_arm"]))
                self.assertEqual(sum(result["trials_per_arm"]), result["spent_lane_minutes"]
                                 if all(c == 1 for c in costs) else
                                 sum(result["trials_per_arm"]))

    def test_nonunit_trials_occupy_lanes_for_their_cost(self) -> None:
        result = run_simulation(
            uniform(), K=2, L=1, B=3, seed=1, probabilities=[1, 0], costs=[2, 1]
        )
        self.assertEqual(result["trials_per_arm"], [1, 1])
        self.assertEqual(result["spent_lane_minutes"], 3)

    def test_rung_prunes_only_after_every_live_arm_fulfills_quota(self) -> None:
        p = [1, 0, 0, 0]
        before = run_simulation(
            successive_halving(rung_trials=2), K=4, L=2, B=7, seed=1,
            probabilities=p,
        )
        self.assertEqual(len(before["live_arms"]), 4)
        after = run_simulation(
            successive_halving(rung_trials=2), K=4, L=2, B=8, seed=1,
            probabilities=p,
        )
        self.assertEqual(len(after["live_arms"]), 2)
        self.assertIn(0, after["live_arms"])
        self.assertIsNone(after["lane_minutes_to_commit"])

    def test_ucb1_uses_hoeffding_bounds_not_a_single_lucky_trial(self) -> None:
        state = SimulationState(
            counts=[1, 1], successes=[1, 0], pending=[0, 0],
            live_arms={0, 1}, costs=(1.0, 1.0),
            budget=100, spent=2, rng=np.random.default_rng(1),
        )
        self.assertEqual(UCB1Policy().prune(state), set())
        state.counts = [100, 100]
        state.successes = [90, 10]
        self.assertEqual(UCB1Policy().prune(state), {1})

    def test_thompson_prunes_using_seeded_beta_posterior_mc(self) -> None:
        state = SimulationState(
            counts=[20, 20], successes=[20, 0], pending=[0, 0],
            live_arms={0, 1}, costs=(1.0, 1.0),
            budget=100, spent=40, rng=np.random.default_rng(9),
        )
        self.assertEqual(ThompsonPolicy(mc_samples=1024).prune(state), {1})
        second = SimulationState(
            counts=[20, 20], successes=[20, 0], pending=[0, 0],
            live_arms={0, 1}, costs=(1.0, 1.0),
            budget=100, spent=40, rng=np.random.default_rng(9),
        )
        self.assertEqual(ThompsonPolicy(mc_samples=1024).prune(second), {1})

    def test_zero_budget_and_singleton_commit(self) -> None:
        empty = run_simulation(uniform(), K=3, L=4, B=0, seed=0)
        self.assertEqual(empty["trials_per_arm"], [0, 0, 0])
        self.assertEqual(empty["spent_lane_minutes"], 0)
        self.assertIsNone(empty["lane_minutes_to_commit"])
        singleton = run_simulation(ucb1(), K=1, L=4, B=100, seed=0, probabilities=[0.8])
        self.assertEqual(singleton["lane_minutes_to_commit"], 0)
        self.assertEqual(singleton["trials_per_arm"], [0])
        self.assertTrue(singleton["best_arm_found"])

    def test_invalid_parameters_fail_closed(self) -> None:
        invalid = [
            dict(K=0, L=2, B=10),
            dict(K=2, L=0, B=10),
            dict(K=2, L=2, B=-1),
            dict(K=2, L=2, B=float("nan")),
            dict(K=2, L=2, B=float("inf")),
            dict(K=2, L=2, B=10, probabilities=[0.1]),
            dict(K=2, L=2, B=10, probabilities=[0.1, 1.1]),
            dict(K=2, L=2, B=10, costs=[1]),
            dict(K=2, L=2, B=10, costs=[1, 0]),
            dict(K=2, L=2, B=10, costs=[1, float("nan")]),
            dict(K=2, L=2, B=10, a=0),
            dict(K=2, L=2, B=10, b=-1),
            dict(K=2, L=2, B=10, seed=-3),
        ]
        for kwargs in invalid:
            with self.subTest(arguments=kwargs):
                with self.assertRaises(ValueError):
                    run_simulation(uniform(), **kwargs)
        for constructor, args in [
            (SuccessiveHalvingPolicy, dict(eta=1)),
            (SuccessiveHalvingPolicy, dict(rung_trials=0)),
            (UCB1Policy, dict(delta=0)),
            (ThompsonPolicy, dict(epsilon=1)),
            (ThompsonPolicy, dict(mc_samples=0)),
            (ThompsonPolicy, dict(min_trials=0)),
        ]:
            with self.subTest(constructor=constructor.__name__, args=args):
                with self.assertRaises(ValueError):
                    constructor(**args)
        with self.assertRaises(ValueError):
            run_experiment(uniform, 2, 2, 10, 0)
        with self.assertRaises(ValueError):
            run_experiment(uniform, 2, 2, 10, [])

    def test_json_cli_emits_one_parseable_object_per_policy(self) -> None:
        completed = subprocess.run(
            [
                sys.executable, "-m", "evolution_lab.menu_sim",
                "--K", "4", "--L", "2", "--B", "40", "--seeds", "2", "--json",
            ],
            text=True, capture_output=True, check=True,
        )
        lines = completed.stdout.splitlines()
        self.assertEqual(len(lines), 4)
        objects = [json.loads(line) for line in lines]
        self.assertEqual({obj["policy"] for obj in objects}, set(POLICIES))
        for obj in objects:
            self.assertEqual(obj["n_seeds"], 2)
            self.assertEqual(obj["K"], 4)
            self.assertEqual(obj["L"], 2)
            self.assertIn("p_wrongly_pruned_best", obj)
            self.assertIn("median_lane_minutes_to_commit", obj)


if __name__ == "__main__":
    unittest.main()
