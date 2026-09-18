from __future__ import annotations

import os
import unittest

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.2")

from evolution_lab.gpu_evolve import gpu_filter, _prepare


class GpuEvolveSmoke(unittest.TestCase):
    def test_filter_pop8(self):
        genome, data, X, y, W_pn, H, Hc, yc, k = _prepare(n_train=24, n_kc=32, k_winners=5, seed=0)
        Ws, acc, elapsed = gpu_filter(H, y, Hc, yc, n_pop=8, epochs=3, lr=0.35, seed=1)
        self.assertEqual(Ws.shape[0], 8)
        self.assertEqual(len(acc), 8)
        self.assertTrue((acc >= 0).all() and (acc <= 1).all())
        self.assertGreater(elapsed, 0.0)


if __name__ == "__main__":
    unittest.main()
