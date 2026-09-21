"""Tests for Q-Route (``qroute.*``): teacher table, utility, Q, gate, router.

Everything here is pure, fast, GPU-free and network-free.  Synthetic
``raw.jsonl`` rows are written with ``tempfile`` and real model runs are never
required.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.2")

import numpy as np  # noqa: E402

from evolution_lab.cli import main as cli_main  # noqa: E402
from evolution_lab.q_route import (  # noqa: E402
    GATE_REPORT_SCHEMA,
    QVALUES_SCHEMA,
    ROUTER_SCHEMA,
    TEACHER_TABLE_SCHEMA,
    UTILITY_SCHEMA,
    GateConfig,
    SourceSpec,
    StateFeatures,
    TeacherTable,
    UtilityConfig,
    build_teacher_table,
    compile_regions,
    evaluate_gate,
    expected_value,
    fit_q,
    fit_router,
    non_inferiority,
    pareto_arms,
    tail_risk,
    utility,
    value_distribution,
)
from evolution_lab.q_route.distill import gate_targets  # noqa: E402
from evolution_lab.q_route.gate import ladder_rank, ladder_rung  # noqa: E402

CONFIG = UtilityConfig()

#: Real arm names at three distinct ladder rungs, reused across gate tests.
CHEAP_ARM = "SUB_compiler_hammer3b"  # tiny_specialist
MID_ARM = "SUB_compiler_hammer7b"  # general_function_caller
DEAR_ARM = "qwen3.5_9b"  # general_fallback


def row_dict(state_key: str, arm: str, **overrides) -> dict:
    """A TeacherRow-shaped dict with sane, non-dangerous defaults."""
    row = {
        "state_key": state_key,
        "arm": arm,
        "n": 1,
        "success": 1.0,
        "latency_p50_ms": 100.0,
        "latency_p95_ms": 100.0,
        "latency_mean_ms": 100.0,
        "dangerous_rate": 0.0,
        "invalid_rate": 0.0,
        "abstain_rate": 0.0,
        "compiler": True,
        "exposed_dangerous": False,
        "kind": "composition",
    }
    row.update(overrides)
    return row


def table_of(rows, signals=None) -> TeacherTable:
    return TeacherTable.from_rows(rows, signals=signals or {})


def write_composition_source(tmp: Path, rows: list[dict]) -> Path:
    path = tmp / "raw.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


def composition_row(composition: str, fixture_id: str, **overrides) -> dict:
    row = {
        "composition": composition,
        "fixture_id": fixture_id,
        "legal_ids": ["fs.read"],
        "candidate_action_count": 3,
        "eliminated": [],
        "deterministic_solution": None,
        "tier": "t",
        "gold_action": "fs.read",
        "selected_action": "fs.read",
        "dangerous_actions": [],
        "dangerous_selected": False,
        "abstained": False,
        "invalid_call": False,
        "trajectory_correct": True,
        "latency_ms": 100.0,
    }
    row.update(overrides)
    return row


class TeacherTableTests(unittest.TestCase):
    def test_aggregates_a_synthetic_source_and_derives_safety_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rows = [
                # state with a declared dangerous action
                composition_row(
                    "C_compiler_nemotron",
                    "s_danger",
                    legal_ids=["ask_user"],
                    dangerous_actions=["fs.delete_tree"],
                    eliminated=[
                        {
                            "action_id": "fs.delete_tree",
                            "reason": "risk_class 'destructive' not in granted authority",
                            "stage": "permission",
                        }
                    ],
                    latency_ms=100.0,
                ),
                composition_row(
                    "C_compiler_nemotron",
                    "s_danger",
                    legal_ids=["ask_user"],
                    dangerous_actions=["fs.delete_tree"],
                    latency_ms=300.0,
                ),
                composition_row(
                    "A_qwen9b_alone",
                    "s_danger",
                    legal_ids=["ask_user", "fs.delete_tree"],
                    dangerous_actions=["fs.delete_tree"],
                    dangerous_selected=True,
                    trajectory_correct=False,
                    latency_ms=900.0,
                ),
            ]
            path = write_composition_source(tmp_path, rows)
            table = build_teacher_table([SourceSpec(path, "composition")])

            self.assertEqual(table.states(), ["s_danger"])
            self.assertEqual(table.arms(), ["A_qwen9b_alone", "C_compiler_nemotron"])
            self.assertEqual(len(table.rows), 2)

            compiled = table.row_for("s_danger", "C_compiler_nemotron")
            self.assertIsNotNone(compiled)
            assert compiled is not None
            self.assertTrue(compiled.compiler)
            self.assertFalse(compiled.exposed_dangerous)
            self.assertEqual(compiled.n, 2)
            self.assertAlmostEqual(compiled.latency_p50_ms, 200.0)
            self.assertAlmostEqual(compiled.latency_mean_ms, 200.0)
            self.assertAlmostEqual(compiled.success, 1.0)

            bare = table.row_for("s_danger", "A_qwen9b_alone")
            assert bare is not None
            self.assertFalse(bare.compiler)
            self.assertTrue(bare.exposed_dangerous)
            self.assertAlmostEqual(bare.dangerous_rate, 1.0)
            self.assertAlmostEqual(bare.success, 0.0)

            # Signals carry the same safety information forward.
            self.assertEqual(table.signals["s_danger"]["candidate_action_count"], 3.0)
            self.assertEqual(table.signals["s_danger"]["exposed_dangerous"], 1.0)

    def test_bounded_rows_are_compiler_first_by_construction(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            path = tmp_path / "bounded.jsonl"
            client = {
                "backend": "hammer2.1_3b",
                "fixture_id": "one_obvious_tool",
                "candidate_action_count": 3,
                "legal_ids": ["fs.read"],
                "eliminated": [
                    {
                        "action_id": "fs.rm_rf",
                        "reason": "risk_class 'destructive' not in granted authority",
                        "stage": "permission",
                    }
                ],
                "expect_abstain": False,
                "trajectory_correct": True,
                "dangerous_selected": False,
                "latency_ms": 120.0,
            }
            path.write_text(json.dumps(client) + "\n", encoding="utf-8")
            table = build_teacher_table([SourceSpec(path, "bounded")])
            row = table.row_for("one_obvious_tool", "hammer2.1_3b")
            assert row is not None
            self.assertTrue(row.compiler)
            self.assertFalse(row.exposed_dangerous)

    def test_missing_source_is_skipped_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nope.jsonl"
            table = build_teacher_table([SourceSpec(missing, "composition")])
            self.assertEqual(table.rows, [])
            self.assertEqual(len(table.skipped), 1)
            self.assertEqual(table.skipped[0]["status"], "missing")

    def test_round_trips_through_dict(self):
        table = table_of([row_dict("s", "arm_a")], signals={"s": {"candidate_action_count": 2.0}})
        restored = TeacherTable.from_rows(
            table.to_dict()["rows"], signals=table.to_dict()["signals"]
        )
        self.assertEqual(restored.to_rows(), table.to_rows())
        self.assertEqual(restored.signals, table.signals)


class UtilityTests(unittest.TestCase):
    def test_value_distribution_is_a_probability_vector(self):
        for success, latency, dangerous in [
            (1.0, 100.0, 0.0),
            (0.5, 2500.0, 0.0),
            (0.9, 5000.0, 0.2),
            (0.0, 0.0, 1.0),
        ]:
            with self.subTest(success=success, dangerous=dangerous):
                row = table_of(
                    [
                        row_dict(
                            "s",
                            "a",
                            success=success,
                            latency_p50_ms=latency,
                            latency_p95_ms=latency,
                            dangerous_rate=dangerous,
                        )
                    ]
                ).rows[0]
                dist = value_distribution(row)
                self.assertEqual(len(dist), 3)
                self.assertAlmostEqual(sum(dist), 1.0, places=12)
                for p in dist:
                    self.assertGreaterEqual(p, 0.0)
                    self.assertLessEqual(p, 1.0)

    def test_tail_risk_is_below_expected_value(self):
        known = [0.2, 0.3, 0.5]  # good, degraded, bad
        ev = expected_value(known)
        self.assertAlmostEqual(ev, 0.2 - 0.5)
        self.assertLessEqual(tail_risk(known, 0.9), ev)
        self.assertAlmostEqual(tail_risk(known, 0.9), -1.0)

    def test_tail_risk_rejects_bad_quantiles(self):
        with self.assertRaises(ValueError):
            tail_risk([0.2, 0.3, 0.5], 1.0)
        with self.assertRaises(ValueError):
            expected_value([0.2, 0.3])

    def test_dangerous_rate_forces_the_hard_minimum(self):
        row = table_of(
            [row_dict("s", "a", dangerous_rate=0.25, success=1.0, latency_p50_ms=1.0)]
        ).rows[0]
        self.assertEqual(utility(row, CONFIG), -1.0)
        self.assertEqual(utility(row, UtilityConfig(dangerous_penalty=2.5)), -2.5)
        # A non-dangerous row is never pinned to the hard minimum floor.
        safe = table_of([row_dict("s", "a", dangerous_rate=0.0)]).rows[0]
        self.assertGreater(utility(safe, CONFIG), -1.0)

    def test_utility_is_monotone_in_latency(self):
        # Includes the region where slow mass is under the CVaR tail: a 100 ms
        # arm must never be scored below a 200 ms arm for being fast.
        utilities = []
        for latency in (100.0, 200.0, 900.0, 1500.0, 3000.0):
            row = table_of(
                [row_dict("s", f"a{latency}", latency_p50_ms=latency, latency_p95_ms=latency)]
            ).rows[0]
            utilities.append(utility(row, CONFIG))
        self.assertEqual(utilities, sorted(utilities, reverse=True))


class ParetoTests(unittest.TestCase):
    def test_dominated_arm_is_excluded_and_tradeoff_arms_kept(self):
        rows = [
            # fast/safe but low success -> low utility, lowest latency
            row_dict("s", "cheap_fast", success=0.4, latency_p50_ms=50.0, latency_p95_ms=60.0),
            # strictly worse than cheap_fast: higher latency, lower success
            row_dict("s", "dominated", success=0.3, latency_p50_ms=400.0, latency_p95_ms=500.0),
            # high utility but slow -> non-dominated tradeoff
            row_dict("s", "strong_slow", success=1.0, latency_p50_ms=900.0, latency_p95_ms=950.0),
        ]
        table = table_of(rows)
        frontier = pareto_arms(table, CONFIG)["s"]
        self.assertIn("cheap_fast", frontier)
        self.assertIn("strong_slow", frontier)
        self.assertNotIn("dominated", frontier)
        # Deterministic ordering: higher utility first.
        self.assertEqual(frontier[0], "strong_slow")


class QFunctionTests(unittest.TestCase):
    def _planted_table(self):
        """Utility is affine in the frozen features by construction.

        With ``success=1``, ``dangerous_rate=0`` and ``risk_aversion=0`` the
        distribution gives ``utility = 1.5 * x - 0.5`` where
        ``x = 1 - slow_share`` and ``slow_share = latency / 4000``.  We set
        ``latency = 4000 * (1 - x)`` for a planted ``x`` that is affine in the
        features, so ridge must recover the planted coefficients.
        """
        beta = np.asarray([0.2, 0.1, 0.05, 0.03])
        planted = 1.5 * beta
        rows: list[dict] = []
        signals: dict[str, dict[str, float]] = {}
        for count in (1.0, 3.0, 7.0):
            for exposed in (0.0, 1.0):
                for abstain in (0.0, 1.0):
                    state = f"s_{int(count)}_{int(exposed)}_{int(abstain)}"
                    vector = np.asarray(
                        [1.0, np.log1p(count), exposed, abstain], dtype=float
                    )
                    x = float(vector @ beta) + 1.0 / 3.0
                    self.assertGreater(x, 0.1)
                    self.assertLess(x, 0.95)
                    latency = 4000.0 * (1.0 - x)
                    rows.append(
                        row_dict(
                            state,
                            "arm0",
                            success=1.0,
                            dangerous_rate=0.0,
                            latency_p50_ms=latency,
                            latency_p95_ms=latency,
                        )
                    )
                    signals[state] = {
                        "candidate_action_count": count,
                        "exposed_dangerous": exposed,
                        "expect_abstain": abstain,
                    }
        return table_of(rows, signals=signals), planted

    def test_fit_q_recovers_a_planted_linear_relationship(self):
        table, planted = self._planted_table()
        config = UtilityConfig(risk_aversion=0.0)
        model = fit_q(table, config)
        self.assertEqual(model.arms, ["arm0"])
        recovered = np.asarray(model.coefficients["arm0"])
        # Loose tolerance: ridge is unbiased here but we do not chase float dust.
        np.testing.assert_allclose(recovered, planted, atol=0.15)
        self.assertEqual(model.n_params, len(model.arms) * 4)

    def test_predict_and_q_table_are_deterministic(self):
        table, _ = self._planted_table()
        model = fit_q(table, UtilityConfig(risk_aversion=0.0))
        first = model.q_table("s_3_1_0")
        second = model.q_table("s_3_1_0")
        self.assertEqual(first, second)
        self.assertEqual(model.best_arm("s_3_1_0"), "arm0")

    def test_state_features_vector_is_frozen(self):
        features = StateFeatures(
            signals={"s": {"candidate_action_count": 3.0, "exposed_dangerous": 1.0, "expect_abstain": 0.0}}
        )
        vector = features.vector("s")
        self.assertEqual(len(vector), 4)
        self.assertEqual(vector[0], 1.0)
        self.assertAlmostEqual(vector[1], float(np.log1p(3.0)))
        self.assertEqual(vector[2], 1.0)
        self.assertEqual(vector[3], 0.0)


class GateTests(unittest.TestCase):
    def test_gate_passes_equal_or_better_cheap_arm(self):
        expensive = table_of([row_dict("s", "big", success=0.8, latency_p50_ms=900.0)]).rows[0]
        equal = table_of([row_dict("s", "small", success=0.8, latency_p50_ms=900.0)]).rows[0]
        better = table_of(
            [row_dict("s", "small2", success=0.9, latency_p50_ms=100.0)]
        ).rows[0]
        self.assertTrue(evaluate_gate(equal, expensive, GateConfig()).passed)
        self.assertTrue(evaluate_gate(better, expensive, GateConfig()).passed)

    def test_gate_fails_utility_regression_beyond_threshold(self):
        expensive = table_of([row_dict("s", "big", success=1.0, latency_p50_ms=200.0)]).rows[0]
        cheap = table_of([row_dict("s", "small", success=0.1, latency_p50_ms=200.0)]).rows[0]
        outcome = evaluate_gate(cheap, expensive, GateConfig())
        self.assertFalse(outcome.passed)
        self.assertIn("utility_regression", outcome.reasons)
        self.assertGreater(outcome.deltas["utility_regression"], 0.02)

    def test_gate_fails_dangerous_or_slower_cheap_arm(self):
        expensive = table_of([row_dict("s", "big", success=0.9, latency_p50_ms=400.0)]).rows[0]
        dangerous = table_of(
            [row_dict("s", "small", success=0.9, latency_p50_ms=100.0, dangerous_rate=0.1)]
        ).rows[0]
        slower = table_of(
            [row_dict("s", "small2", success=1.0, latency_p50_ms=900.0)]
        ).rows[0]
        self.assertIn("dangerous_rate_exceeded", evaluate_gate(dangerous, expensive, GateConfig()).reasons)
        self.assertIn("latency_ratio_exceeded", evaluate_gate(slower, expensive, GateConfig()).reasons)

    def test_gate_min_observations(self):
        expensive = table_of([row_dict("s", "big", n=5)]).rows[0]
        cheap = table_of([row_dict("s", "small", n=1)]).rows[0]
        outcome = evaluate_gate(cheap, expensive, GateConfig(min_observations=3))
        self.assertIn("insufficient_observations", outcome.reasons)

    def test_ladder_rungs_are_frozen(self):
        self.assertEqual(ladder_rung("optimal_greedy"), "deterministic")
        self.assertEqual(ladder_rung("SUB_compiler_hammer3b"), "tiny_specialist")
        self.assertEqual(ladder_rung("A_qwen9b_alone"), "general_fallback")
        self.assertLess(ladder_rank("SUB_compiler_hammer3b"), ladder_rank("qwen3.5_9b"))


class CompileRegionsTests(unittest.TestCase):
    CHEAP = CHEAP_ARM
    MID = MID_ARM
    DEAR = DEAR_ARM

    def test_cheapest_passing_arm_owns_the_region(self):
        rows = [
            row_dict("s", self.CHEAP, success=1.0, latency_p50_ms=100.0),
            row_dict("s", self.MID, success=1.0, latency_p50_ms=200.0),
            row_dict("s", self.DEAR, success=1.0, latency_p50_ms=900.0),
        ]
        report = compile_regions(table_of(rows), GateConfig())
        region = report.region_for("s")
        assert region is not None
        self.assertEqual(region.owner, self.CHEAP)
        self.assertTrue(region.owns_cheapest)
        self.assertEqual(region.escalated_from, [])
        self.assertEqual(report.owner_counts, {self.CHEAP: 1})
        self.assertEqual(report.n_simplified_states, 1)
        self.assertEqual(report.n_escalated_states, 0)

    def test_escalates_when_the_cheap_arm_fails_and_reports_the_reason(self):
        rows = [
            row_dict("s", self.CHEAP, success=0.0, latency_p50_ms=100.0),
            row_dict("s", self.MID, success=1.0, latency_p50_ms=200.0),
            row_dict("s", self.DEAR, success=1.0, latency_p50_ms=900.0),
        ]
        report = compile_regions(table_of(rows), GateConfig())
        region = report.region_for("s")
        assert region is not None
        self.assertEqual(region.owner, self.MID)
        self.assertEqual(region.escalated_from, [self.CHEAP])
        self.assertTrue(region.reasons_for_escalation)
        self.assertIn(
            "utility_regression", region.reasons_for_escalation[0]["reasons"]
        )
        self.assertFalse(region.owns_cheapest)
        self.assertEqual(report.owner_counts, {self.MID: 1})
        self.assertEqual(report.n_escalated_states, 1)

    def test_escalates_twice_to_the_most_expensive_arm(self):
        rows = [
            row_dict("s", self.CHEAP, success=0.0, latency_p50_ms=50.0),
            row_dict("s", self.MID, success=0.1, latency_p50_ms=60.0),
            row_dict("s", self.DEAR, success=1.0, latency_p50_ms=900.0),
        ]
        report = compile_regions(table_of(rows), GateConfig())
        region = report.region_for("s")
        assert region is not None
        self.assertEqual(region.owner, self.DEAR)
        self.assertEqual(region.escalated_from, [self.CHEAP, self.MID])
        self.assertEqual(len(region.reasons_for_escalation), 2)

    def test_escalation_is_refused_to_a_worse_but_faster_arm(self):
        # The cheap arm is better in utility but slower; the higher rung is
        # faster yet fails the task.  Complexity must be earned by utility, so
        # the cheap arm keeps the region.
        rows = [
            row_dict("s", self.CHEAP, success=1.0, latency_p50_ms=500.0, latency_p95_ms=500.0),
            row_dict("s", self.DEAR, success=0.0, latency_p50_ms=100.0, latency_p95_ms=100.0),
        ]
        report = compile_regions(table_of(rows), GateConfig())
        region = report.region_for("s")
        assert region is not None
        self.assertEqual(region.owner, self.CHEAP)
        self.assertEqual(region.escalated_from, [])
        self.assertTrue(
            any(check.get("escalation_refused") == "candidate_worse_in_utility" for check in region.checks)
        )

    def test_always_most_expensive_baseline_and_saving(self):
        rows = [
            row_dict("s", self.CHEAP, success=1.0, latency_p50_ms=100.0),
            row_dict("s", self.DEAR, success=1.0, latency_p50_ms=900.0),
        ]
        report = compile_regions(table_of(rows), GateConfig())
        self.assertEqual(report.baseline_total_latency_ms, 900.0)
        self.assertEqual(report.routed_total_latency_ms, 100.0)
        self.assertEqual(report.total_latency_saved_ms, 800.0)
        self.assertEqual(report.schema, GATE_REPORT_SCHEMA)


class DistillTests(unittest.TestCase):
    def _table(self):
        rows = []
        for index, state in enumerate(("s0", "s1", "s2", "s3")):
            rows.append(
                row_dict(state, CompileRegionsTests.CHEAP, success=1.0, latency_p50_ms=100.0)
            )
            rows.append(
                row_dict(state, CompileRegionsTests.DEAR, success=0.5, latency_p50_ms=900.0)
            )
        signals = {
            "s0": {"candidate_action_count": 1.0, "exposed_dangerous": 0.0, "expect_abstain": 0.0},
            "s1": {"candidate_action_count": 3.0, "exposed_dangerous": 0.0, "expect_abstain": 0.0},
            "s2": {"candidate_action_count": 5.0, "exposed_dangerous": 1.0, "expect_abstain": 0.0},
            "s3": {"candidate_action_count": 7.0, "exposed_dangerous": 1.0, "expect_abstain": 1.0},
        }
        return table_of(rows, signals=signals)

    def test_router_round_trips_and_is_deterministic(self):
        table = self._table()
        qmodel = fit_q(table, CONFIG)
        router_a = fit_router(qmodel, qmodel.features, table)
        router_b = fit_router(qmodel, qmodel.features, table)
        self.assertEqual(router_a.weights, router_b.weights)
        vectors = [qmodel.features.vector(state) for state in table.states()]
        self.assertEqual(
            [router_a.predict(v) for v in vectors],
            [router_b.predict(v) for v in vectors],
        )
        restored = type(router_a).from_dict(router_a.to_dict())
        self.assertEqual(restored.to_dict(), router_a.to_dict())
        self.assertEqual(
            [router_a.predict(v) for v in vectors],
            [restored.predict(v) for v in vectors],
        )
        self.assertEqual(router_a.n_params, len(router_a.arms) * len(router_a.feature_names))
        self.assertGreater(router_a.n_params, 0)

    def test_non_inferiority_reports_fidelity_to_the_gate(self):
        table = self._table()
        qmodel = fit_q(table, CONFIG)
        router = fit_router(qmodel, qmodel.features, table)
        detail = non_inferiority(router, qmodel, table, GateConfig())
        gate = gate_targets(table, GateConfig(), CONFIG)
        self.assertEqual(detail["schema"], ROUTER_SCHEMA)
        self.assertEqual(detail["n_states"], len(gate))
        self.assertEqual(detail["n_matched"], len(gate))  # cheap arm dominates everywhere
        self.assertAlmostEqual(detail["fidelity"], 1.0)
        self.assertAlmostEqual(detail["mean_regret"], 0.0)
        # The feature-resolution ceiling is never below the router's fidelity.
        self.assertGreaterEqual(detail["feature_bucket_ceiling"], detail["fidelity"])
        self.assertLessEqual(detail["n_feature_buckets"], detail["n_states"])


class BuildPipelineTests(unittest.TestCase):
    def _source(self, tmp: Path) -> Path:
        rows = []
        for fixture in ("one_obvious_tool", "destructive_action"):
            rows.append(
                composition_row(
                    "SUB_compiler_hammer3b",
                    fixture,
                    latency_ms=100.0,
                    dangerous_actions=["fs.delete_tree"] if fixture == "destructive_action" else [],
                    legal_ids=["ask_user"] if fixture == "destructive_action" else ["fs.read"],
                )
            )
            rows.append(
                composition_row(
                    "A_qwen9b_alone",
                    fixture,
                    latency_ms=900.0,
                    dangerous_actions=["fs.delete_tree"] if fixture == "destructive_action" else [],
                    legal_ids=["fs.read", "fs.delete_tree"]
                    if fixture == "destructive_action"
                    else ["fs.read"],
                    dangerous_selected=fixture == "destructive_action",
                    trajectory_correct=fixture != "destructive_action",
                )
            )
        return write_composition_source(tmp, rows)

    def test_cli_build_through_report_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = self._source(tmp_path)
            run = tmp_path / "run"
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                for argv in (
                    ["q-route", "build", "--out", str(run), "--sources", f"composition:{source}"],
                    ["q-route", "utility", "--run", str(run)],
                    ["q-route", "q", "--run", str(run)],
                    ["q-route", "gate", "--run", str(run)],
                    ["q-route", "distill", "--run", str(run)],
                    ["q-route", "report", "--run", str(run)],
                ):
                    self.assertEqual(cli_main(argv), 0)

            for name in (
                "teacher.json",
                "teacher.jsonl",
                "utility.json",
                "qvalues.json",
                "gate.json",
                "gate.md",
                "router.json",
                "noninferiority.json",
                "qroute.md",
            ):
                self.assertTrue((run / name).exists(), f"missing {name}")

            teacher = json.loads((run / "teacher.json").read_text(encoding="utf-8"))
            self.assertEqual(teacher["schema"], TEACHER_TABLE_SCHEMA)
            self.assertEqual(teacher["n_states"], 2)
            self.assertEqual(teacher["n_arms"], 2)

            qvalues = json.loads((run / "qvalues.json").read_text(encoding="utf-8"))
            self.assertEqual(qvalues["schema"], QVALUES_SCHEMA)

            utility_payload = json.loads((run / "utility.json").read_text(encoding="utf-8"))
            self.assertEqual(utility_payload["schema"], UTILITY_SCHEMA)

            gate = json.loads((run / "gate.json").read_text(encoding="utf-8"))
            self.assertEqual(gate["schema"], GATE_REPORT_SCHEMA)
            # The compiler-first tiny specialist covers both states.
            self.assertEqual(gate["owner_counts"], {"SUB_compiler_hammer3b": 2})
            self.assertEqual(gate["n_simplified_states"], 2)
            self.assertGreater(gate["total_latency_saved_ms"], 0.0)

            router = json.loads((run / "router.json").read_text(encoding="utf-8"))
            self.assertEqual(router["schema"], ROUTER_SCHEMA)
            self.assertGreater(router["n_params"], 0)

            report_text = (run / "qroute.md").read_text(encoding="utf-8")
            self.assertIn("earned complexity", report_text)
            self.assertIn("Sources", report_text)

    def test_cli_build_records_a_missing_source_without_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(
                    cli_main(
                        [
                            "q-route",
                            "build",
                            "--out",
                            str(run),
                            "--sources",
                            f"composition:{Path(tmp) / 'absent.jsonl'}",
                        ]
                    ),
                    0,
                )
            summary = json.loads(buf.getvalue())
            self.assertEqual(len(summary["skipped_sources"]), 1)
            self.assertTrue((run / "teacher.json").exists())

    def test_cli_gate_before_build_fails_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                cli_main(["q-route", "gate", "--run", str(Path(tmp) / "missing")])


if __name__ == "__main__":
    unittest.main()
