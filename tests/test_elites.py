"""Unit tests for the MAP-Elites archive."""

import unittest

from evolution_lab.elites import MapElites, niche_key
from evolution_lab.schema import PARAM_BUCKETS


class MapElitesTests(unittest.TestCase):
    def test_niche_key_buckets(self):
        self.assertEqual(niche_key("mlp", 99_999, "sgd"), "mlp|<100K|sgd")
        self.assertEqual(niche_key("mlp", 100_000, "sgd"), "mlp|100K-1M|sgd")
        self.assertEqual(niche_key("x", 50_000_000, "m"), "x|10M-100M|m")

    def test_offer_empty_cell_accepts(self):
        archive = MapElites()

        self.assertTrue(archive.offer("k", "e1", 0.5, {"a": 1}))
        self.assertEqual(archive.cells["k"].experiment_id, "e1")
        self.assertEqual(archive.cells["k"].fitness, 0.5)
        self.assertEqual(archive.cells["k"].record, {"a": 1})

    def test_offer_strictly_better_replaces(self):
        archive = MapElites()
        self.assertTrue(archive.offer("k", "e1", 0.5, {"a": 1}))

        self.assertFalse(archive.offer("k", "e2", 0.5, {}))
        self.assertEqual(archive.cells["k"].experiment_id, "e1")
        self.assertEqual(archive.cells["k"].record, {"a": 1})

        self.assertTrue(archive.offer("k", "e3", 0.6, {}))
        self.assertEqual(archive.cells["k"].experiment_id, "e3")
        self.assertEqual(archive.cells["k"].fitness, 0.6)
        self.assertEqual(archive.cells["k"].record, {})

        self.assertFalse(archive.offer("k", "e4", -1.0, {}))
        self.assertEqual(archive.cells["k"].experiment_id, "e3")
        self.assertEqual(archive.cells["k"].fitness, 0.6)

    def test_occupied_counts_cells(self):
        archive = MapElites()
        self.assertEqual(archive.occupied(), 0)

        self.assertTrue(archive.offer("a", "e1", 0.5, {}))
        self.assertTrue(archive.offer("b", "e2", 0.6, {}))
        self.assertEqual(archive.occupied(), 2)

    def test_coverage_denominator(self):
        archive = MapElites()
        archive.offer("a", "e1", 0.5, {})
        archive.offer("b", "e2", 0.6, {})

        self.assertEqual(
            archive.coverage(["a", "b"], ["m"]),
            2 / (2 * len(PARAM_BUCKETS) * 1),
        )
        self.assertEqual(archive.coverage([], []), archive.occupied() / 1)


if __name__ == "__main__":
    unittest.main()
