from __future__ import annotations

import unittest

import numpy as np

from evolution_lab.coverage_metric import GEN0_COVERAGE_AT_95, coverage_at_precision, gen0_baseline


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


if __name__ == "__main__":
    unittest.main()
