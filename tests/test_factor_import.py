from __future__ import annotations

import unittest

from evolution_lab.factor_import import (
    import_bend_attestation_rows,
    import_rlm_evidence_ab,
    import_rlm_querygen,
    import_sol_pi_observation_ledger,
    import_state_packet,
    import_slm_tournament_rows,
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


    def test_state_packet_import_checks_provenance(self):
        packet = {
            "schema": "z0int.context_resolve.v1",
            "evidence": [
                {
                    "source_id": "git:repo",
                    "source_version": "sha:abc",
                    "locator": "README.md:1",
                    "trust_class": "code",
                    "observed_at": "2026-09-30T00:00:00Z",
                },
                {"source_id": "memory:x"},
            ],
            "contradictions": ["old vs new"],
            "unresolved_gaps": ["current CI state"],
            "measurements": {"source_reads": 2, "packet_bytes": 900, "latency_ms": 4.2},
        }
        result = import_state_packet(packet, revision="z0int@abc")
        self.assertEqual(result.measurements["evidence_count"], 2)
        self.assertEqual(result.measurements["provenance_missing"], 1)
        self.assertEqual(result.measurements["unresolved_gap_count"], 1)
        self.assertEqual(result.receipts[0].factor, "state_packet_memory")

    def test_slm_tournament_import_keeps_hard_failures_separate(self):
        rows = [
            {
                "schema": "z0int.tool_tournament.row.v1",
                "composition": "compiler_qwen4b",
                "correct": True,
                "hard_failure": False,
                "invalid_call": False,
                "dangerous_pick": False,
                "abstained": False,
                "latency_ms": 120,
                "retries": 0,
                "prompt_tokens": 80,
                "completion_tokens": 12,
                "gpu_ms": 100,
                "vram_peak_mb": 4300,
            },
            {
                "schema": "z0int.tool_tournament.row.v1",
                "composition": "compiler_qwen4b",
                "correct": False,
                "hard_failure": True,
                "invalid_call": True,
                "dangerous_pick": False,
                "abstained": True,
                "latency_ms": 180,
                "retries": 1,
                "prompt_tokens": 90,
                "completion_tokens": 10,
                "gpu_ms": 150,
                "vram_peak_mb": 4400,
            },
        ]
        result = import_slm_tournament_rows(
            rows, composition="compiler_qwen4b", revision="evolution-lab@test"
        )
        self.assertEqual(result.observations, 2)
        self.assertEqual(result.measurements["correct_rate"], 0.5)
        self.assertEqual(result.measurements["hard_failures"], 1)
        self.assertEqual(result.measurements["invalid_calls"], 1)
        self.assertEqual(result.measurements["peak_vram_mb"], 4400)
        self.assertEqual(result.receipts[0].factor, "local_slm_policy")



    def test_bend_verdict_pass_cannot_hide_semantic_mismatch(self):
        result = import_bend_attestation_rows(
            [
                {
                    "candidate": "2.0.34",
                    "adapter_pass": True,
                    "source_value": False,
                    "certified_value": True,
                    "translation_mismatch": True,
                }
            ],
            revision="bend@2.0.34",
        )
        self.assertEqual(result.measurements["verifier_pass_rate"], 1.0)
        self.assertEqual(result.measurements["semantic_attestation_rate"], 0.0)
        self.assertEqual(result.measurements["translation_mismatches"], 1)
        self.assertFalse(result.measurements["hard_gate_pass"])

    def test_bend_patched_candidate_clears_semantic_oracle(self):
        result = import_bend_attestation_rows(
            [
                {
                    "candidate": "patched-main",
                    "adapter_pass": True,
                    "source_value": False,
                    "certified_value": False,
                    "translation_mismatch": False,
                }
                for _ in range(25)
            ],
            revision="17db447a8b8c17b51517de42b35d3216e563f40f",
        )
        self.assertEqual(result.measurements["semantic_attestation_rate"], 1.0)
        self.assertEqual(result.measurements["translation_mismatches"], 0)
        self.assertTrue(result.measurements["hard_gate_pass"])



    def test_rlm_querygen_gate_preserves_scope(self):
        rows = []
        for workload in ("a", "b", "c", "d", "e"):
            rows.extend(
                [
                    {
                        "run": 1,
                        "arm": "oracle",
                        "workload": workload,
                        "retrievalPass": True,
                        "patternHits": 1,
                        "grantedBytes": 1000,
                        "generatorElapsedMs": 0,
                        "systemTokenProxy": 250,
                    },
                    {
                        "run": 1,
                        "arm": "lexical",
                        "workload": workload,
                        "retrievalPass": workload != "e",
                        "patternHits": 1,
                        "grantedBytes": 1200,
                        "generatorElapsedMs": 0,
                        "systemTokenProxy": 300,
                    },
                    {
                        "run": 1,
                        "arm": "model",
                        "workload": workload,
                        "retrievalPass": workload != "e",
                        "patternHits": 1,
                        "grantedBytes": 1100,
                        "generatorTotalTokens": 20,
                        "generatorElapsedMs": 8,
                        "systemTokenProxy": 295,
                    },
                ]
            )
        result = import_rlm_querygen(
            rows, revision="exp/rlm-evidence-ab-clean"
        )
        gate = result.measurements["provisional_querygen_gate"]
        self.assertTrue(gate["oracle_is_complete"])
        self.assertEqual(gate["model_oracle_retention"], 0.8)
        self.assertTrue(gate["model_not_below_lexical"])
        self.assertTrue(gate["candidate"])
        self.assertIn(
            "sub-gate only", result.measurements["qualification_note"]
        )

    def test_rlm_querygen_rejects_partial_model_cohort(self):
        rows = [
            {
                "run": 1,
                "arm": "oracle",
                "workload": "a",
                "retrievalPass": True,
            },
            {
                "run": 1,
                "arm": "lexical",
                "workload": "a",
                "retrievalPass": True,
            },
            {
                "run": 2,
                "arm": "model",
                "workload": "a",
                "retrievalPass": True,
            },
        ]
        with self.assertRaisesRegex(ValueError, "cohort differs"):
            import_rlm_querygen(rows, revision="test")



if __name__ == "__main__":
    unittest.main()
