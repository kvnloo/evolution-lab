from __future__ import annotations

import unittest

import numpy as np

from evolution_lab.coverage_metric import (
    GEN0_COVERAGE_AT_95,
    binary_gate_metrics,
    coverage_at_precision,
    frozen_claims,
    gen0_baseline,
)


class CoverageMetric(unittest.TestCase):
    def test_perfect_prefix(self):
        y = np.array([0, 0, 1, 1, 0])
        pred = np.array([0, 0, 1, 0, 1])
        score = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        # first 3 correct → prec 1.0 coverage 3/5
        out = coverage_at_precision(y, pred, score, floor=0.95)
        self.assertEqual(out["n"], 3)
        self.assertAlmostEqual(out["coverage"], 0.6)
        self.assertAlmostEqual(out["precision"], 1.0)

    def test_gen0_baseline_keys(self):
        b = gen0_baseline()
        self.assertIn("coverage_at_95", b)
        self.assertAlmostEqual(b["coverage_at_95"], GEN0_COVERAGE_AT_95)

    def test_binary_gate_lift_fpr(self):
        # base_rate=0.25; predict all positives → prec=0.25 lift=1 fpr=1
        y = np.array([1, 0, 0, 0, 1, 0, 0, 0])
        pred = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        m = binary_gate_metrics(y, pred, positive=1)
        self.assertAlmostEqual(m["base_rate"], 0.25)
        self.assertAlmostEqual(m["precision"], 0.25)
        self.assertAlmostEqual(m["lift_over_base_rate"], 1.0)
        self.assertAlmostEqual(m["fpr"], 1.0)
        # perfect gate
        pred2 = y.copy()
        m2 = binary_gate_metrics(y, pred2, positive=1)
        self.assertAlmostEqual(m2["precision"], 1.0)
        self.assertAlmostEqual(m2["fpr"], 0.0)
        self.assertAlmostEqual(m2["lift_over_base_rate"], 4.0)

    def test_frozen_claims_forbid_raw_98(self):
        c = frozen_claims()
        self.assertEqual(c["schema"], "flyforge.science_claims.v1")
        self.assertIn("lift_over_base_rate", c["delegate_reporting"])
        self.assertTrue(any("98" in x for x in c["gen0"]["forbidden_phrasing"]))



if __name__ == "__main__":
    unittest.main()
