from __future__ import annotations

import unittest

import numpy as np

from evolution_lab.next_action_gpu import FAM_I, WEAK_FAMILIES, _boost_train


class WeakBoost(unittest.TestCase):
    def test_boost_one_is_noop(self):
        X = np.zeros((10, 4), dtype=np.float32)
        y = np.zeros(10, dtype=np.int32)
        Xb, yb, added = _boost_train(X, y, boost=1)
        self.assertEqual(len(yb), 10)
        self.assertTrue(all(v == 0 for v in added.values()))

    def test_boost_oversamples_edit_only(self):
        n = 20
        X = np.arange(n * 2, dtype=np.float32).reshape(n, 2)
        y = np.zeros(n, dtype=np.int32)
        y[:5] = FAM_I["EDIT"]
        y[5:] = FAM_I["EXECUTE"]
        Xb, yb, added = _boost_train(X, y, families=("EDIT",), boost=3, rng=np.random.default_rng(0))
        # original 5 EDIT + 2 extra copies = 15 EDIT
        self.assertEqual(int((yb == FAM_I["EDIT"]).sum()), 15)
        self.assertEqual(added["EDIT"], 10)
        self.assertEqual(len(yb), 30)


if __name__ == "__main__":
    unittest.main()
