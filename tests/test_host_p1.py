"""Synthetic-only causal baseline acceptance tests (stdlib unittest + numpy)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict
import json
import unittest

import numpy as np

from evolution_lab.host_p1 import (
    ACTUATION_AUTHORITY,
    EFFECT_WINDOW_S,
    SCHEMA_VERSION,
    Intervention,
    Outcome,
    StateFrame,
    grouped_holdout,
    intervention_effect,
    logistic_state_model,
    make_confounded_interventions,
    p3_gate,
    persistence_baseline,
    planted_probability_effect,
    simulate,
    summarize_run,
)


class ContractsTests(unittest.TestCase):
    def test_immutable_bounded_synthetic_contracts_and_no_actuation(self):
        state = StateFrame(0, 0.75, 0.50, 0.20, 0.10, 90, 0.60)
        outcome = Outcome(0, True, False, 1)
        event = Intervention(0, "harmful", 1.0, 1.0, "g0")
        self.assertIs(ACTUATION_AUTHORITY, False)
        self.assertEqual(SCHEMA_VERSION, "host_p1.v1")
        for item in (state, outcome, event):
            self.assertEqual(item.truth_tier, "synthetic")
        self.assertEqual(asdict(state)["truth_tier"], "synthetic")
        with self.assertRaises(FrozenInstanceError):
            state.mem_used_frac = 0.1
        with self.assertRaises(ValueError):
            StateFrame(0, 1.01, 0.5, 0.2, 0.1, 5, 0.5)
        with self.assertRaises(ValueError):
            StateFrame(0, 0.5, -0.1, 0.2, 0.1, 5, 0.5)
        with self.assertRaises(ValueError):
            StateFrame(0, 0.5, 0.1, 0.2, 0.1, -1, 0.5)
        with self.assertRaises(ValueError):
            Outcome(0, 1, False, 1)
        with self.assertRaises(ValueError):
            Intervention(-1, "helpful", 1.0, 1.0, "g")

    def test_actuation_match_exact_or_tolerance(self):
        self.assertTrue(Intervention(0, "helpful", 1.0, 1.0, "a").effective_matches_intent)
        self.assertTrue(Intervention(0, "helpful", 1.0, 1.0 + 1e-10, "a").effective_matches_intent)
        self.assertFalse(Intervention(0, "helpful", 1.0, 0.8, "a").effective_matches_intent)


class SyntheticCausalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seed = 41
        cls.interventions = make_confounded_interventions(5000, cls.seed)
        cls.states, cls.outcomes = simulate(5000, cls.interventions, cls.seed)
        cls.report = intervention_effect(
            cls.states, cls.outcomes, cls.interventions, 8, EFFECT_WINDOW_S,
            seed=97, bootstrap_samples=800,
        )

    def test_same_seed_same_states_outcomes_and_estimates(self):
        events = make_confounded_interventions(5000, self.seed)
        states, outcomes = simulate(5000, events, self.seed)
        self.assertEqual(events, self.interventions)
        self.assertEqual(states, self.states)
        self.assertEqual(outcomes, self.outcomes)
        self.assertEqual(
            intervention_effect(states, outcomes, events, 8, EFFECT_WINDOW_S, seed=97),
            self.report,
        )
        self.assertTrue(all(s.truth_tier == "synthetic" for s in states))
        self.assertTrue(all(o.truth_tier == "synthetic" for o in outcomes))

    def test_confounded_scheduling_prefers_rising_high_pressure(self):
        self.assertGreater(len(self.interventions), 60)
        at_events = [self.states[event.t].mem_used_frac for event in self.interventions]
        overall = [frame.mem_used_frac for frame in self.states]
        self.assertGreater(float(np.mean(at_events)), float(np.mean(overall)) + 0.07)
        self.assertGreater(
            float(np.mean([
                self.states[event.t].mem_used_frac - self.states[event.t - 1].mem_used_frac
                for event in self.interventions
            ])),
            0.065,
        )

    def test_5000_step_effect_signs_and_bootstrap_intervals(self):
        effects = self.report["effects"]
        for knob, direction in (("harmful", 1), ("helpful", -1)):
            result = effects[knob]
            self.assertGreater(result["n_groups"], 20)
            self.assertGreater(direction * result["adjusted_effect"], 0)
            self.assertIsNotNone(result["ci95"][0])
            self.assertGreater(direction * result["ci95"][0], 0) if direction > 0 else self.assertLess(result["ci95"][1], 0)
            self.assertEqual(result["unit"], "oom_probability_difference")
        self.assertGreater(effects["harmful"]["ci95"][0], 0)
        self.assertLess(effects["helpful"]["ci95"][1], 0)

    def test_adjustment_reduces_confounding_bias_vs_oracle(self):
        oracle = planted_probability_effect(self.states, self.interventions)
        adjusted_error = []
        naive_error = []
        for knob in ("harmful", "helpful"):
            estimated = self.report["effects"][knob]
            adjusted_error.append(abs(estimated["adjusted_effect"] - oracle[knob]))
            naive_error.append(abs(estimated["naive_effect"] - oracle[knob]))
        self.assertLess(sum(adjusted_error), sum(naive_error))

    def test_failed_actuation_reported_separately_and_has_no_effect(self):
        failed = [Intervention(100, "harmful", 1.0, 0.0, "failed")]
        states, failed_outcomes = simulate(400, failed, seed=2)
        _, no_treatment_outcomes = simulate(400, [], seed=2)
        self.assertEqual(failed_outcomes, no_treatment_outcomes)
        estimate = intervention_effect(states, failed_outcomes, failed, 8, 8, seed=2)
        self.assertEqual(estimate["effects"], {})
        self.assertEqual(len(estimate["ineffective"]), 1)
        self.assertEqual(estimate["ineffective"][0]["knob"], "harmful")
        self.assertEqual(estimate["ineffective"][0]["reason"], "actuation_mismatch")

    def test_json_receipt_has_metrics_and_per_knob_effects(self):
        receipt = summarize_run(self.states, self.outcomes, self.interventions, seed=97)
        self.assertEqual(receipt["schema_version"], "host_p1.v1")
        self.assertEqual(receipt["truth_tier"], "synthetic")
        self.assertEqual(receipt["metrics"]["n_steps"], 5000)
        self.assertIn("temporal_holdout_brier_state", receipt["metrics"])
        self.assertEqual(set(receipt["effects"]), {"harmful", "helpful"})
        self.assertFalse(
            set(receipt["grouped_holdout"]["train_group_ids"])
            & set(receipt["grouped_holdout"]["test_group_ids"])
        )
        self.assertEqual(json.loads(json.dumps(receipt)), receipt)

    def test_overlapping_actuation_not_falsely_attributed(self):
        shared = [
            Intervention(100, "harmful", 1.0, 1.0, "co"),
            Intervention(100, "helpful", 1.0, 1.0, "co"),
        ]
        states, outcomes = simulate(300, shared, 5)
        report = intervention_effect(states, outcomes, shared, 8, 8)
        self.assertFalse(report["effects"])
        self.assertEqual(len(report["unestimable"]), 2)


class PredictiveBaselineTests(unittest.TestCase):
    def test_state_model_improves_brier_over_persistence(self):
        states, outcomes = simulate(4000, [], 123)
        cut = 2800
        model = logistic_state_model((states[:cut], outcomes[:cut]))
        scores = model.predict_proba(states[cut:])
        actual = np.asarray([float(o.oom_kill_in_window) for o in outcomes[cut:]])
        persistence = persistence_baseline(outcomes)[cut - 1:]
        self.assertEqual(len(scores), len(actual))
        self.assertEqual(len(persistence), len(actual))
        self.assertTrue(np.all((scores >= 0.0) & (scores <= 1.0)))
        state_brier = float(np.mean((scores - actual) ** 2))
        persistence_brier = float(np.mean((persistence - actual) ** 2))
        self.assertLess(state_brier, persistence_brier)
        self.assertEqual(len(model.coefficients), 4)

    def test_grouped_holdout_never_splits_co_timed_group(self):
        events = [
            Intervention(20, "harmful", 1.0, 1.0, "g1"),
            Intervention(20, "helpful", 1.0, 1.0, "g1"),
            Intervention(50, "harmful", 1.0, 1.0, "g2"),
            Intervention(60, "helpful", 1.0, 1.0, "g3"),
            Intervention(90, "harmful", 1.0, 1.0, "g4"),
        ]
        train, test = grouped_holdout(events, 0.5, 13)
        self.assertTrue(train)
        self.assertTrue(test)
        self.assertEqual(len(train) + len(test), len(events))
        self.assertFalse({e.group_id for e in train} & {e.group_id for e in test})
        self.assertEqual((train, test), grouped_holdout(events, 0.5, 13))
        self.assertEqual(
            sum(e.group_id == "g1" for e in train + test), 2
        )

    def test_p3_gate_requires_strict_gain_on_grouped_brier(self):
        self.assertFalse(p3_gate(0.20, 0.24, 0.02)["earned"])
        self.assertFalse(p3_gate(0.20, 0.19, 0.02)["earned"])
        self.assertFalse(p3_gate(0.20, 0.18, 0.02)["earned"])
        self.assertTrue(p3_gate(0.20, 0.14, 0.02)["earned"])
        self.assertTrue(p3_gate(0.20, 0.14, 0.02)["reason"])
        with self.assertRaises(ValueError):
            p3_gate(float("nan"), 0.1, 0.01)


if __name__ == "__main__":
    unittest.main()
