"""Phase 1B section G tests: ceiling, collision classification, candidate features."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evolution_lab.q_route.analysis import (
    ANALYSIS_SCHEMA,
    analyse,
    bucket_ceiling,
    candidate_feature_vectors,
    classify_collision,
    load_observations,
    realised_utility_gap,
    score_candidate_features,
)
from evolution_lab.q_route.teacher import TeacherTable
from evolution_lab.q_route.utility import UtilityConfig


def _row(state, arm, *, n=3, success=1.0, latency=100.0, dangerous=0.0, compiler=True):
    return {
        "state_key": state,
        "arm": arm,
        "n": n,
        "success": success,
        "latency_p50_ms": latency,
        "latency_p95_ms": latency * 1.5,
        "latency_mean_ms": latency,
        "dangerous_rate": dangerous,
        "invalid_rate": 0.0,
        "abstain_rate": 0.0,
        "compiler": compiler,
        "exposed_dangerous": False,
    }


def _observation(state, arm, *, correct, alias=None, **features):
    defaults = {
        "candidate_action_count": 3,
        "family": "tool_selection",
        "legal_family_count": 2,
        "legal_risk_classes": ["read"],
        "authority_breadth": 1,
        "budget_units": 8,
        "declared_dangerous_count": 0,
        "satisfied_count": 0,
    }
    defaults.update(features)
    return {
        "schema": "z0int.phase1b.observation.v1",
        "run_id": "p1b-test",
        "trace_id": f"{state}-{arm}",
        "observed_at": "2026-09-21T10:00:00Z",
        "state_id": state,
        "arm": arm,
        "arm_alias": alias,
        "state_features": defaults,
        "correct": correct,
        "dangerous_selected": False,
        "total_ms": 100.0,
        "failure_retry": {"attempts": 1, "error": None},
    }


class BucketCeilingTests(unittest.TestCase):
    def test_single_owner_per_bucket_reaches_one(self):
        table = TeacherTable.from_rows([_row("s1", "a"), _row("s2", "a")])
        vectors = {"s1": [1.0, 0.0], "s2": [1.0, 0.0]}
        result = bucket_ceiling(table, {"s1": "a", "s2": "a"}, vectors)
        self.assertEqual(result["ceiling"], 1.0)
        self.assertEqual(result["collisions"], [])

    def test_collision_lowers_the_ceiling(self):
        table = TeacherTable.from_rows([
            _row("s1", "a"), _row("s2", "b"), _row("s3", "a"),
        ])
        vectors = {"s1": [1.0, 0.0], "s2": [1.0, 0.0], "s3": [1.0, 0.0]}
        result = bucket_ceiling(table, {"s1": "a", "s2": "b", "s3": "a"}, vectors)
        self.assertAlmostEqual(result["ceiling"], 2 / 3)
        self.assertEqual(len(result["collisions"]), 1)
        self.assertEqual(sorted(result["collisions"][0]["states"]), ["s1", "s2", "s3"])

    def test_separating_feature_removes_the_collision(self):
        table = TeacherTable.from_rows([
            _row("s1", "a"), _row("s2", "b"), _row("s3", "a"),
        ])
        vectors = {s: [1.0, 0.0] for s in ("s1", "s2", "s3")}
        candidates = {"split": {"s1": 0.0, "s2": 1.0, "s3": 0.0}}
        scored = score_candidate_features(table, {"s1": "a", "s2": "b", "s3": "a"},
                                          vectors, candidates)
        self.assertEqual(scored[0]["feature"], "split")
        self.assertAlmostEqual(scored[0]["ceiling_gain"], 1 / 3)
        self.assertEqual(scored[0]["remaining_collisions"], 0)

    def test_non_separating_feature_gains_nothing(self):
        table = TeacherTable.from_rows([_row("s1", "a"), _row("s2", "b")])
        vectors = {s: [1.0, 0.0] for s in ("s1", "s2")}
        scored = score_candidate_features(table, {"s1": "a", "s2": "b"}, vectors,
                                          {"flat": {"s1": 1.0, "s2": 1.0}})
        self.assertEqual(scored[0]["ceiling_gain"], 0.0)


class CollisionClassificationTests(unittest.TestCase):
    def test_disagreeing_repeats_are_label_noise(self):
        cells = {
            ("s1", "a"): type("C", (), {"n": 3, "successes": 2, "disagreement": True})(),
            ("s2", "b"): type("C", (), {"n": 3, "successes": 3, "disagreement": False})(),
        }
        result = classify_collision(["s1", "s2"], {"s1": "a", "s2": "b"}, cells,
                                    feature_gain_available=False)
        self.assertIn("noisy_labels", result["reasons"])

    def test_stable_but_different_owners_is_multimodal(self):
        cells = {
            ("s1", "a"): type("C", (), {"n": 3, "successes": 3, "disagreement": False})(),
            ("s2", "b"): type("C", (), {"n": 3, "successes": 3, "disagreement": False})(),
        }
        result = classify_collision(["s1", "s2"], {"s1": "a", "s2": "b"}, cells,
                                    feature_gain_available=False)
        self.assertIn("multimodal_state_action_value", result["reasons"])

    def test_no_evidence_is_undetermined(self):
        result = classify_collision(["s1", "s2"], {"s1": "a", "s2": "b"}, None,
                                    feature_gain_available=False)
        self.assertEqual(result["classification"], "undetermined")


class ObservationIngestTests(unittest.TestCase):
    def test_alias_lines_receipts_up_with_the_teacher_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "observations.jsonl"
            with path.open("w", encoding="utf-8") as fh:
                fh.write(json.dumps(_observation("s1", "compiler+hammer2.1_3b",
                                                 alias="SUB_compiler_hammer3b",
                                                 correct=True)) + "\n")
            cells = load_observations(path)
            self.assertIn(("s1", "SUB_compiler_hammer3b"), cells)
            self.assertEqual(cells[("s1", "SUB_compiler_hammer3b")].arm_raw,
                             "compiler+hammer2.1_3b")

    def test_candidate_features_are_derived_from_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "observations.jsonl"
            with path.open("w", encoding="utf-8") as fh:
                for state, risks in (("s1", ["read"]), ("s2", ["destructive"])):
                    row = _observation(state, "a", correct=True,
                                       legal_risk_classes=risks)
                    fh.write(json.dumps(row) + "\n")
            cells = load_observations(path)
            features = candidate_feature_vectors(cells, path)
            self.assertEqual(features["legal_has_irreversible"]["s1"], 0.0)
            self.assertEqual(features["legal_has_irreversible"]["s2"], 1.0)
            self.assertIn("family=tool_selection", features)


class RealisedUtilityTests(unittest.TestCase):
    def test_matching_owner_has_zero_regret(self):
        table = TeacherTable.from_rows([_row("s1", "a"), _row("s1", "b", latency=900)])
        result = realised_utility_gap(table, {"s1": "a"}, lambda s: "a", UtilityConfig())
        self.assertEqual(result["mean_utility_delta_vs_owner"], 0.0)
        self.assertEqual(result["equal_states"], 1)

    def test_worse_choice_has_positive_regret(self):
        table = TeacherTable.from_rows([
            _row("s1", "good", success=1.0, latency=100.0),
            _row("s1", "bad", success=0.0, latency=3000.0),
        ])
        result = realised_utility_gap(table, {"s1": "good"}, lambda s: "bad", UtilityConfig())
        self.assertGreater(result["mean_regret"], 0.0)
        self.assertEqual(result["worse_states"], 1)


class AnalyseTests(unittest.TestCase):
    def test_analyse_reports_structure_without_training(self):
        rows = [
            _row("s1", "a", success=1.0), _row("s1", "b", success=0.0, latency=900),
            _row("s2", "a", success=1.0), _row("s2", "b", success=1.0, latency=900),
        ]
        table = TeacherTable.from_rows(rows)
        report = analyse(table)
        self.assertEqual(report["schema"], ANALYSIS_SCHEMA)
        self.assertEqual(report["n_states"], 2)
        self.assertIn("ceiling", report)
        self.assertIn("verdict", report)

    def test_analyse_with_observations_and_no_router(self):
        rows = [_row("s1", "compiler+hammer2.1_3b", success=1.0)]
        table = TeacherTable.from_rows(rows)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "observations.jsonl"
            with path.open("w", encoding="utf-8") as fh:
                fh.write(json.dumps(_observation("s1", "compiler+hammer2.1_3b",
                                                 alias="compiler+hammer2.1_3b",
                                                 correct=True)) + "\n")
            report = analyse(table, observations_path=path)
            self.assertTrue(report["coverage"]["available"])
            self.assertIsNone(report["ceiling"]["router_fidelity"])


if __name__ == "__main__":
    unittest.main()
