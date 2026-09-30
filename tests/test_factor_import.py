from __future__ import annotations

import unittest

from evolution_lab.factor_import import (
    import_rlm_evidence_ab,
    import_sol_pi_observation_ledger,
    rlm_arm_summaries,
)
from evolution_lab.factorized_stack import qualify_singleton


class FactorImportTests(unittest.TestCase):
    def test_sol_pi_ledger_import_preserves_estimate_semantics(self):
        rows = [
            {
                "event": "full",
                "id": "obs_a",
                "request": 1,
                "tool": "computer",
                "originalBytes": 20000,
            },
            {
                "event": "placeholder",
                "id": "obs_a",
                "request": 3,
                "tool": "computer",
                "originalBytes": 20000,
                "placeholderTokens": 500,
            },
            {
                "event": "placeholder",
                "id": "obs_b",
                "request": 4,
                "tool": "computer",
                "originalBytes": 12000,
                "placeholderTokens": 400,
            },
        ]
        result = import_sol_pi_observation_ledger(rows, revision="main@test")
        self.assertEqual(result.observations, 3)
        self.assertEqual(result.measurements["placeholder_events"], 2)
        self.assertEqual(result.measurements["original_placeholder_bytes"], 32000)
        self.assertEqual(result.measurements["estimated_placeholder_bytes"], 3600)
        self.assertEqual(result.receipts[0].factor, "sol_pi_observation_pack")

    def test_rlm_import_matches_source_row_shape(self):
        rows = [
            {
                "run": 1,
                "backend": "mechanism",
                "model": "none",
                "workload": "head-control",
                "arm": "full",
                "pass": True,
                "fullBytes": 30000,
                "grantedBytes": 30000,
                "grantRatio": 1.0,
                "promptBytes": 31000,
                "elapsedMs": 0,
            },
            {
                "run": 1,
                "backend": "mechanism",
                "model": "none",
                "workload": "head-control",
                "arm": "search8k",
                "pass": True,
                "fullBytes": 30000,
                "grantedBytes": 1500,
                "grantRatio": 0.05,
                "promptBytes": 2500,
                "elapsedMs": 0,
            },
        ]
        result = import_rlm_evidence_ab(rows, revision="exp/rlm-evidence-ab-clean")
        self.assertEqual(result["arms"]["full"]["pass_rate"], 1.0)
        self.assertEqual(result["arms"]["search8k"]["pass_rate"], 1.0)
        self.assertEqual(result["receipts"][0]["factor"], "rlm_evidence_addressing")

    def test_rlm_rows_can_enter_common_singleton_gate(self):
        rows = []
        for workload in ("a", "b", "c"):
            rows.extend(
                [
                    {
                        "run": 1,
                        "backend": "mechanism",
                        "model": "none",
                        "workload": workload,
                        "arm": "full",
                        "pass": True,
                        "grantedBytes": 30000,
                        "elapsedMs": 0,
                    },
                    {
                        "run": 1,
                        "backend": "mechanism",
                        "model": "none",
                        "workload": workload,
                        "arm": "search8k",
                        "pass": True,
                        "grantedBytes": 4000,
                        "elapsedMs": 0,
                    },
                ]
            )
        control, candidate = rlm_arm_summaries(rows)
        decision = qualify_singleton(candidate, control)
        self.assertTrue(decision.qualified)
        self.assertLess(decision.deltas["mean_context_bytes"], 0)

    def test_rlm_projection_rejects_different_cohorts(self):
        rows = [
            {
                "run": 1,
                "backend": "mechanism",
                "model": "none",
                "workload": "a",
                "arm": "full",
                "pass": True,
                "grantedBytes": 10,
                "elapsedMs": 0,
            },
            {
                "run": 1,
                "backend": "mechanism",
                "model": "none",
                "workload": "b",
                "arm": "search8k",
                "pass": True,
                "grantedBytes": 5,
                "elapsedMs": 0,
            },
        ]
        with self.assertRaisesRegex(ValueError, "same run/workload cohort"):
            rlm_arm_summaries(rows)


if __name__ == "__main__":
    unittest.main()
