from __future__ import annotations

import unittest

from evolution_lab.factorized_stack import (
    Arm,
    ArmSummary,
    FactorReceipt,
    GatePolicy,
    assert_composition_allowed,
    experiment_manifest,
    pairwise_arms,
    phase0_arms,
    qualify_singleton,
)


class FactorizedStackTests(unittest.TestCase):
    def control(self) -> ArmSummary:
        return ArmSummary(
            arm_id="control",
            n_work_items=100,
            verified_success_rate=0.90,
            mean_input_tokens=1000,
            mean_context_bytes=8000,
            mean_raw_source_reads=8,
            mean_frontier_calls=1.0,
            p95_latency_ms=1000,
            mean_retries=0.20,
            mean_corrections=0.10,
        )

    def test_phase0_is_control_plus_five_singletons(self):
        arms = phase0_arms()
        self.assertEqual(len(arms), 6)
        self.assertEqual(arms[0].id, "control")
        self.assertTrue(all(len(a.factors) == 1 for a in arms[1:]))

    def test_arm_requires_stage_order(self):
        with self.assertRaisesRegex(ValueError, "stage order"):
            Arm(
                id="bad",
                factors=("bend_structural_verifier", "local_slm_policy"),
                phase="pairwise",
            )

    def test_pairwise_requires_both_singletons(self):
        arms = pairwise_arms({"state_packet_memory", "local_slm_policy"})
        self.assertEqual(len(arms), 1)
        self.assertEqual(
            arms[0].factors, ("state_packet_memory", "local_slm_policy")
        )
        self.assertEqual(pairwise_arms({"state_packet_memory"}), ())

    def test_explicit_composition_gate_fails_closed(self):
        arm = Arm(
            id="memory_x_slm",
            factors=("state_packet_memory", "local_slm_policy"),
            phase="pairwise",
        )
        with self.assertRaisesRegex(ValueError, "singleton evidence"):
            assert_composition_allowed(arm, {"state_packet_memory"})

    def test_efficiency_win_qualifies_without_blending_scores(self):
        candidate = ArmSummary(
            arm_id="rlm_evidence_addressing__only",
            n_work_items=100,
            verified_success_rate=0.90,
            mean_input_tokens=700,
            mean_context_bytes=5000,
            mean_raw_source_reads=7,
            mean_frontier_calls=1.0,
            p95_latency_ms=950,
            mean_retries=0.20,
            mean_corrections=0.10,
        )
        decision = qualify_singleton(candidate, self.control())
        self.assertTrue(decision.qualified)
        self.assertEqual(decision.deltas["verified_success_rate"], 0.0)
        self.assertLess(decision.deltas["mean_input_tokens"], 0)

    def test_quality_regression_blocks_large_token_savings(self):
        candidate = ArmSummary(
            arm_id="sol_pi_observation_pack__only",
            n_work_items=100,
            verified_success_rate=0.87,
            mean_input_tokens=300,
            mean_context_bytes=2500,
            mean_raw_source_reads=4,
            mean_frontier_calls=0.8,
            p95_latency_ms=800,
            mean_retries=0.20,
            mean_corrections=0.10,
        )
        decision = qualify_singleton(candidate, self.control())
        self.assertFalse(decision.qualified)
        self.assertIn("verified_quality_regression", decision.reasons)

    def test_policy_or_provenance_failure_blocks(self):
        candidate = ArmSummary(
            arm_id="bend_structural_verifier__only",
            n_work_items=100,
            verified_success_rate=0.92,
            hard_policy_violations=1,
            provenance_failures=1,
            mean_input_tokens=900,
            mean_context_bytes=7500,
            mean_raw_source_reads=7,
            mean_frontier_calls=0.9,
            p95_latency_ms=900,
            mean_retries=0.20,
            mean_corrections=0.10,
        )
        decision = qualify_singleton(candidate, self.control())
        self.assertFalse(decision.qualified)
        self.assertIn("hard_policy_violation", decision.reasons)
        self.assertIn("provenance_failure", decision.reasons)

    def test_receipts_remain_factor_specific(self):
        receipt = FactorReceipt(
            factor="state_packet_memory",
            trace_id="trace-1",
            schema="z0int.context_resolve.v1",
            source="kvnloo/z0intelligence",
            revision="abc123",
        )
        row = receipt.to_dict()
        self.assertEqual(row["factor"], "state_packet_memory")
        self.assertEqual(row["trace_id"], "trace-1")

    def test_manifest_is_evaluation_only(self):
        manifest = experiment_manifest()
        self.assertEqual(manifest["authority"], "evaluation_only")
        self.assertIn("no_full_stack_before_singleton_evidence", manifest["hard_constraints"])

    def test_same_cohort_is_required(self):
        candidate = ArmSummary(
            arm_id="local_slm_policy__only",
            n_work_items=99,
            verified_success_rate=0.90,
        )
        with self.assertRaisesRegex(ValueError, "same frozen work-item count"):
            qualify_singleton(candidate, self.control())

    def test_no_benefit_does_not_qualify_by_default(self):
        candidate = ArmSummary(
            arm_id="state_packet_memory__only",
            n_work_items=100,
            verified_success_rate=0.90,
            mean_input_tokens=1000,
            mean_context_bytes=8000,
            mean_raw_source_reads=8,
            mean_frontier_calls=1.0,
            p95_latency_ms=1000,
            mean_retries=0.20,
            mean_corrections=0.10,
        )
        decision = qualify_singleton(candidate, self.control())
        self.assertFalse(decision.qualified)
        self.assertIn("no_measured_benefit", decision.reasons)

    def test_gate_can_measure_noninferiority_only(self):
        candidate = ArmSummary(
            arm_id="bend_structural_verifier__only",
            n_work_items=100,
            verified_success_rate=0.90,
        )
        decision = qualify_singleton(
            candidate,
            ArmSummary(arm_id="control", n_work_items=100, verified_success_rate=0.90),
            policy=GatePolicy(require_efficiency_or_quality_gain=False),
        )
        self.assertTrue(decision.qualified)


if __name__ == "__main__":
    unittest.main()
