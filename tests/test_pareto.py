from __future__ import annotations

import unittest

from evolution_lab.pareto import Point, delta_hypervolume, dominates, hypervolume_2d, pareto_front


def P(i, s, c):
    return Point(i, success=s, cost=c, ood=s, params=1, violations=0)


class ParetoTests(unittest.TestCase):
    def test_dominance(self):
        a = P("a", 0.9, 0.2)
        b = P("b", 0.8, 0.2)
        self.assertTrue(dominates(a, b))
        self.assertFalse(dominates(b, a))

    def test_front_keeps_tradeoff(self):
        pts = [P("cheap", 0.5, 0.1), P("good", 0.9, 0.4), P("worse", 0.4, 0.5)]
        ids = {p.experiment_id for p in pareto_front(pts)}
        self.assertIn("cheap", ids)
        self.assertIn("good", ids)
        self.assertNotIn("worse", ids)

    def test_delta_hv_positive_for_new_corner(self):
        before = [P("a", 0.5, 0.5)]
        extra = P("b", 0.9, 0.2)
        self.assertGreater(delta_hypervolume(before, extra), 0.0)

    def test_hv_empty(self):
        self.assertEqual(hypervolume_2d([]), 0.0)
