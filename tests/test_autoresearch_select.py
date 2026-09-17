from __future__ import annotations

import unittest

from evolution_lab.select import BenchMetrics, BenchResult, FlyCandidate, check_gates, decide_status, latency_score, load_config


class AutoresearchSelectTests(unittest.TestCase):
    def test_gates_pass_on_perfect_metrics(self):
        cfg = load_config()
        metrics = BenchMetrics(
            closed_loop_reward=1.0,
            confirm_success=1.0,
            val_success=0.99,
            ood_success=0.9,
            violations=0.0,
            advise_warm_p99_ms=100.0,
            closed_loop_ms=500.0,
        )
        ok, reasons = check_gates(metrics, cfg)
        self.assertTrue(ok, reasons)

    def test_gates_fail_closed_loop(self):
        cfg = load_config()
        metrics = BenchMetrics(closed_loop_reward=0.5, confirm_success=1.0, val_success=1.0, ood_success=1.0)
        ok, reasons = check_gates(metrics, cfg)
        self.assertFalse(ok)
        self.assertTrue(any("closed_loop" in r for r in reasons))

    def test_decide_keep_when_faster(self):
        cfg = load_config()
        cand = FlyCandidate(genome_id="x", genome={"id": "x", "lineage": "m", "hypothesis": "h", "architecture": {}, "curriculum": {}, "training": {}, "evaluation": {}})
        champion = BenchResult(
            experiment_id="c0",
            candidate=cand,
            metrics=BenchMetrics(advise_warm_p99_ms=200.0, closed_loop_ms=1000.0, closed_loop_reward=1.0, confirm_success=1.0, val_success=1.0, ood_success=1.0),
            latency_score=latency_score(BenchMetrics(advise_warm_p99_ms=200.0, closed_loop_ms=1000.0), cfg),
            gates_pass=True,
            status="keep",
        )
        faster = BenchResult(
            experiment_id="c1",
            candidate=cand,
            metrics=BenchMetrics(advise_warm_p99_ms=50.0, closed_loop_ms=400.0, closed_loop_reward=1.0, confirm_success=1.0, val_success=1.0, ood_success=1.0),
            latency_score=latency_score(BenchMetrics(advise_warm_p99_ms=50.0, closed_loop_ms=400.0), cfg),
            gates_pass=True,
        )
        self.assertEqual(decide_status(faster, champion, cfg), "keep")


if __name__ == "__main__":
    unittest.main()
