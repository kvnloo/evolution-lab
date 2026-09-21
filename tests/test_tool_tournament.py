"""Tests for the local tool-calling tournament (issue kvnloo/evolution-lab#23).

GPU-free and network-free: every backend here is either the real deterministic
control or a ``ScriptedBackend`` replaying a fixture-authored answer.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.2")

from evolution_lab.tool_tournament import (  # noqa: E402
    COMPOSITIONS,
    HOLDOUT_TASK_FAMILIES,
    REQUIRED_FIXTURE_IDS,
    ROLES,
    BackendRegistry,
    ScriptedBackend,
    compile_fixture,
    load_fixtures,
    make_tournament_split,
    percentile,
    run_tournament,
    write_run,
)
from evolution_lab.tool_tournament.compositions import (  # noqa: E402
    LocalSLMBackendProvider,
    expected_parallel_actions,
)
from evolution_lab.tool_tournament.score import (  # noqa: E402
    SPLIT_SCHEMA,
    TOURNAMENT_SCHEMA,
)

DANGEROUS_FIXTURES = (
    "permission_denied",
    "credential_required_action",
    "destructive_action",
    "publish_send_deploy_action",
)
COMPILER_COMPOSITIONS = (
    "C_compiler_nemotron",
    "D_compiler_jev_nemotron",
    "E_compiler_jev_nemotron_qwen9b",
    "F_compiler_jev_nemotron_qwen9b_specialists",
    "qwen4b_instead_of_9b",
    "hammer3b_specialist",
    "hammer7b_specialist",
    "functiongemma_specialist",
    "rules_baseline",
    "logistic_baseline",
)


class FixtureSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = load_fixtures()
        cls.by_id = {f.fixture_id: f for f in cls.fixtures}

    def test_loads_and_covers_required_fixture_ids(self):
        self.assertEqual(len(self.fixtures), 28)
        missing = [fid for fid in REQUIRED_FIXTURE_IDS if fid not in self.by_id]
        self.assertEqual(missing, [])
        self.assertEqual(len(REQUIRED_FIXTURE_IDS), 28)

    def test_grouped_splits_have_enough_work_items(self):
        per_family: dict[str, set[str]] = {}
        for f in self.fixtures:
            per_family.setdefault(f.task_family, set()).add(f.work_item)
        self.assertGreaterEqual(len(per_family), 3)
        for family, items in per_family.items():
            self.assertGreaterEqual(len(items), 3, f"{family} has only {len(items)} work items")
        holdout = [fam for fam in HOLDOUT_TASK_FAMILIES if fam in per_family]
        self.assertGreaterEqual(len(holdout), 2)
        self.assertEqual(set(HOLDOUT_TASK_FAMILIES), {"routing", "safety"})

    def test_dangerous_fixtures_are_compiler_removed(self):
        for fid in DANGEROUS_FIXTURES:
            fixture = self.by_id[fid]
            with self.subTest(fixture=fid):
                self.assertTrue(fixture.dangerous_actions)
                plan = compile_fixture(fixture)
                for action in fixture.dangerous_actions:
                    self.assertIn(action, plan.removed)
                    self.assertNotIn(action, plan.legal_actions)
                self.assertNotIn(fixture.gold_action, fixture.dangerous_actions)
                self.assertTrue(plan.legal_actions)
                self.assertFalse(plan.forced_abstain)

    def test_dangerous_reasons_are_specific(self):
        expected = {
            "permission_denied": "permission_denied",
            "credential_required_action": "credential_required",
            "destructive_action": "destructive_action",
            "publish_send_deploy_action": "publish_action",
        }
        for fid, reason in expected.items():
            fixture = self.by_id[fid]
            plan = compile_fixture(fixture)
            self.assertEqual(plan.removed[fixture.dangerous_actions[0]], reason)

    def test_expected_parallel_actions_reads_the_dag(self):
        parallel = self.by_id["two_independent_parallel_actions"]
        self.assertEqual(set(expected_parallel_actions(parallel)), {"fs.read", "search.docs"})
        diamond = self.by_id["diamond_dependency_graph"]
        self.assertEqual(set(expected_parallel_actions(diamond)), {"search.docs", "net.fetch"})
        serial = self.by_id["hard_dependency_a_then_b"]
        self.assertEqual(expected_parallel_actions(serial), [])


class DangerousDiscrimination(unittest.TestCase):
    """Fixtures must actually punish a policy that ignores the compiler."""

    @classmethod
    def setUpClass(cls):
        cls.fixtures = load_fixtures()
        hostile = BackendRegistry.scripted(cls.fixtures, mode="dangerous")
        cls.hostile_run = run_tournament(fixtures=cls.fixtures, registry=hostile)
        gold = BackendRegistry.scripted(cls.fixtures, mode="gold")
        cls.gold_run = run_tournament(fixtures=cls.fixtures, registry=gold)

    def test_model_only_compositions_can_pick_dangerous(self):
        for name in ("A_qwen9b_alone", "B_nemotron_alone"):
            score = self.hostile_run.scores[name]
            with self.subTest(composition=name):
                self.assertGreater(score.dangerous_picks, 0)
                self.assertFalse(score.sections["safety"]["safety_pass"])

    def test_compiler_compositions_never_pick_dangerous(self):
        for name in COMPILER_COMPOSITIONS:
            score = self.hostile_run.scores[name]
            with self.subTest(composition=name):
                self.assertEqual(score.dangerous_picks, 0)
                self.assertTrue(score.sections["safety"]["safety_pass"])

    def test_dangerous_pick_is_a_hard_failure_flag(self):
        rows = [r for r in self.hostile_run.rows if r.composition == "A_qwen9b_alone"]
        dangerous_rows = [r for r in rows if r.dangerous_pick]
        self.assertEqual(len(dangerous_rows), 4)
        for row in dangerous_rows:
            self.assertTrue(row.hard_failure)
            self.assertFalse(row.correct)
        score = self.hostile_run.scores["A_qwen9b_alone"]
        self.assertEqual(score.hard_failures, len(dangerous_rows))
        self.assertEqual(
            score.sections["trajectory_success"]["verified_success_count"],
            sum(1 for r in rows if r.correct and not r.hard_failure),
        )

    def test_compiler_never_returns_a_dangerous_action_in_raw_rows(self):
        by_id = {f.fixture_id: f for f in self.fixtures}
        for row in self.hostile_run.rows:
            if row.composition in ("A_qwen9b_alone", "B_nemotron_alone"):
                continue
            banned = set(by_id[row.fixture_id].declared_dangerous)
            with self.subTest(composition=row.composition, fixture=row.fixture_id):
                self.assertFalse(banned & set(row.plan))

    def test_gold_script_is_fully_correct(self):
        for name in ("C_compiler_nemotron", "F_compiler_jev_nemotron_qwen9b_specialists"):
            score = self.gold_run.scores[name]
            with self.subTest(composition=name):
                self.assertEqual(score.sections["trajectory_success"]["verified_task_success"], 1.0)
                self.assertEqual(score.dangerous_picks, 0)


class GroupedSplits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = load_fixtures()
        cls.split = make_tournament_split(cls.fixtures)

    def test_no_work_item_appears_in_two_folds(self):
        folds = self.split.fold_of_work_item()
        buckets = list(folds.values())
        self.assertEqual(len(buckets), len(set(folds)))
        self.assertNotIn("unknown", set(buckets))
        groups = {
            "train": set(self.split.session_split.train_sessions),
            "dev": set(self.split.session_split.dev_sessions),
            "sealed": set(self.split.session_split.sealed_sessions),
            "future": set(self.split.session_split.future_sessions),
            "holdout": set(self.split.holdout_work_items),
        }
        seen: set[str] = set()
        for bucket, items in groups.items():
            with self.subTest(bucket=bucket):
                self.assertFalse(seen & items, f"{bucket} overlaps another fold")
                seen |= items
        self.assertEqual(seen, {f.work_item for f in self.fixtures})

    def test_holdout_task_families_absent_from_train_and_val(self):
        holdout = set(self.split.holdout_task_families)
        self.assertEqual(holdout, set(HOLDOUT_TASK_FAMILIES))
        train_dev = set(self.split.session_split.train_sessions) | set(self.split.session_split.dev_sessions)
        sealed_and_future = set(self.split.session_split.sealed_sessions) | set(
            self.split.session_split.future_sessions
        )
        for fixture in self.fixtures:
            if fixture.task_family not in holdout:
                continue
            with self.subTest(fixture=fixture.fixture_id):
                self.assertNotIn(fixture.work_item, train_dev)
                self.assertNotIn(fixture.work_item, sealed_and_future)
                self.assertEqual(self.split.bucket_of(fixture.work_item), "holdout")

    def test_split_reuses_sealed_split_schema(self):
        self.assertEqual(self.split.session_split.schema, "flyforge.session_split.v1")
        payload = self.split.to_dict()
        self.assertEqual(payload["schema"], SPLIT_SCHEMA)
        self.assertEqual(payload["schema"], "z0int.tool_tournament_split.v1")
        self.assertTrue(payload["train_sessions"])
        self.assertTrue(payload["holdout_work_items"])

    def test_branch_of_same_work_item_stays_together(self):
        buckets = {f.fixture_id: self.split.bucket_of(f.work_item) for f in self.fixtures}
        self.assertEqual(buckets["one_obvious_tool"], buckets["two_similar_tools"])
        self.assertEqual(buckets["required_argument_missing"], buckets["wrong_argument_type"])
        self.assertEqual(buckets["tool_fails"], buckets["tool_succeeds_after_retry"])


class PercentileTests(unittest.TestCase):
    def test_p50_le_p95_le_p99(self):
        values = [float(i) for i in range(101)]
        p50 = percentile(values, 50.0)
        p95 = percentile(values, 95.0)
        p99 = percentile(values, 99.0)
        self.assertLessEqual(p50, p95)
        self.assertLessEqual(p95, p99)

    def test_percentiles_are_monotone_on_noisy_latencies(self):
        values = [3.0, 1.0, 99.0, 12.5, 42.0, 7.0, 250.0, 18.0]
        self.assertLessEqual(percentile(values, 50.0), percentile(values, 95.0))
        self.assertLessEqual(percentile(values, 95.0), percentile(values, 99.0))

    def test_empty_is_zero(self):
        self.assertEqual(percentile([], 95.0), 0.0)


class ReportShape(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = load_fixtures()
        cls.tournament = run_tournament(
            fixtures=cls.fixtures,
            registry=BackendRegistry.scripted(cls.fixtures, mode="gold"),
        )

    def test_schema_string_and_role_winners(self):
        report = self.tournament.report
        self.assertEqual(report["schema"], TOURNAMENT_SCHEMA)
        self.assertEqual(report["schema"], "z0int.local_tool_tournament.v1")
        self.assertEqual(set(report["role_winners"]), set(ROLES))
        for winner in report["role_winners"].values():
            self.assertIn("composition", winner)
            self.assertIn("rationale", winner)

    def test_sections_keep_schema_and_trajectory_separate(self):
        for name, score in self.tournament.report["compositions"].items():
            if score["status"] != "ok":
                continue
            with self.subTest(composition=name):
                sections = score["sections"]
                self.assertIn("trajectory_success", sections)
                self.assertIn("schema_correctness", sections)
                self.assertIn("tool_selection", sections)
                self.assertIn("model_selection", sections)
                self.assertIn("latency", sections)
                self.assertIn("calibration", sections)
                self.assertNotIn("per_bucket", sections)
                self.assertIn("decision_ms", sections["latency"])
                self.assertIn("end_to_end_ms", sections["latency"])
                self.assertIn("p95", sections["latency"]["decision_ms"])
                self.assertIn("p95", sections["latency"]["end_to_end_ms"])

    def test_pareto_has_frontier_and_promotes_per_role(self):
        self.assertEqual(self.tournament.pareto["schema"], "z0int.local_tool_tournament.pareto.v1")
        self.assertGreater(self.tournament.pareto["n_frontier"], 0)
        self.assertEqual(
            len(self.tournament.pareto["frontier"]) + len(self.tournament.pareto["dominated"]),
            self.tournament.pareto["n_points"],
        )
        for winner in self.tournament.report["role_winners"].values():
            if winner["composition"] is not None:
                self.assertIn(winner["composition"], COMPOSITIONS)

    def test_write_run_writes_all_artifacts_and_is_create_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = write_run(tmp, self.tournament, timestamp="20260101T000000Z")
            for name in ("report.json", "report.md", "pareto.json", "raw.jsonl"):
                self.assertTrue((dest / name).is_file(), name)
            report = json.loads((dest / "report.json").read_text())
            self.assertEqual(report["schema"], TOURNAMENT_SCHEMA)
            self.assertTrue((dest / "raw.jsonl").read_text().strip())
            with self.assertRaises(FileExistsError):
                write_run(tmp, self.tournament, timestamp="20260101T000000Z")

    def test_run_tournament_writes_create_only_run_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = run_tournament(
                fixtures=self.fixtures,
                registry=BackendRegistry.scripted(self.fixtures, mode="gold"),
                out_dir=tmp,
            )
            self.assertIsNotNone(run.run_dir)
            self.assertTrue((run.run_dir / "report.json").is_file())


class BackendProtocolTests(unittest.TestCase):
    def test_scripted_backend_replays_fixture_answer(self):
        fixture = load_fixtures()[0]
        scripted = ScriptedBackend.from_fixtures([fixture], mode="gold")
        decision = COMPOSITIONS["A_qwen9b_alone"].evaluate(fixture, BackendRegistry.from_mapping({"qwen9b": scripted}))
        self.assertEqual(decision.action, fixture.gold_action)

    def test_local_slm_provider_is_lazy_and_raises_clearly(self):
        provider = LocalSLMBackendProvider("qwen9b")
        with mock.patch(
            "evolution_lab.tool_tournament.compositions._import_z0int_cognition",
            return_value=None,
        ):
            self.assertFalse(provider.available())
            with self.assertRaises(RuntimeError) as ctx:
                provider.backend()
        self.assertIn("z0int.cognition", str(ctx.exception))

    def test_importing_the_package_does_not_require_z0int(self):
        import evolution_lab.tool_tournament as pkg

        self.assertTrue(hasattr(pkg, "run_tournament"))
        self.assertTrue(hasattr(pkg, "LocalSLMBackendProvider"))


if __name__ == "__main__":
    unittest.main()
