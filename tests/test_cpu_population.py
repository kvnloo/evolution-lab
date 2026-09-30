from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from evolution_lab import cpu_population as cp
from evolution_lab.engine import seed_genomes
from evolution_lab.models import _kc_codes, fit_student
from evolution_lab.splits import load_splits


def _reference(gene: cp.Gene, train):
    base = next(g for g in seed_genomes() if g.architecture.family == "local_plasticity")
    genome = replace(
        base,
        architecture=replace(base.architecture, hidden=gene.hidden, k_winners=gene.k_winners),
        training=replace(base.training, seed=gene.seed, plasticity_lr=gene.lr, plasticity_epochs=gene.epochs),
    )
    return fit_student(genome, list(train))


class CpuPopulationParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_splits()
        cls.train = cls.data.train[:24]
        cls.ts = cp.train_set(cls.train)
        cls.genes = [
            cp.Gene(seed=0, hidden=64, k_winners=0, lr=0.35, epochs=3),
            cp.Gene(seed=5, hidden=40, k_winners=4, lr=0.8, epochs=2),
            cp.Gene(seed=9, hidden=16, k_winners=3, lr=0.1, epochs=1),  # hidden floors to 32
        ]
        cls.st = cp.setup_population(cls.genes, cls.ts)

    def test_sparse_codes_match_dense_kc_codes(self):
        g = self.genes[0]
        H = _kc_codes(self.ts.X, self.st.W_pn[0], g.k)
        idx, val = cp.sparse_kc(self.ts.X, self.st.W_pn[0], g.k)
        dense = np.zeros_like(H)
        np.put_along_axis(dense, idx, val, axis=1)
        np.testing.assert_allclose(dense, H, atol=1e-15)

    def test_numpy_float64_matches_fit_student(self):
        W = cp.train_numpy(self.st, self.ts.y)
        for p, g in enumerate(self.genes):
            ref = _reference(g, self.train)
            np.testing.assert_array_equal(ref.extras["W_pn_kc"], self.st.W_pn[p])
            self.assertEqual(ref.extras["k_winners"], g.k)
            np.testing.assert_allclose(cp.dense_weights(self.st, W, p), ref.extras["W_kc_mbon"], atol=1e-12)
            pred_ref = ref.predict_fn(list(self.data.val))
            pred_pop = cp.predict_population(self.st, W, list(self.data.val), protocol=True)[p]
            np.testing.assert_array_equal(pred_ref, pred_pop)

    def test_float32_decisions_agree(self):
        W64 = cp.train_numpy(self.st, self.ts.y)
        W32 = cp.train_numpy(self.st, self.ts.y, dtype=np.float32)
        eps, _ = cp.prefix_episodes(self.data.val)
        a = cp.predict_population(self.st, W64, eps, protocol=False)
        b = cp.predict_population(self.st, W32, eps, protocol=False)
        self.assertGreaterEqual(float((a == b).mean()), 0.999)

    def test_jax_float32_decisions_agree(self):
        try:
            import jax  # noqa: F401
        except ImportError:
            self.skipTest("jax not installed")
        W64 = cp.train_numpy(self.st, self.ts.y)
        Wj = cp.train_jax(self.st, self.ts.y)
        eps, _ = cp.prefix_episodes(self.data.val)
        a = cp.predict_population(self.st, W64, eps, protocol=False)
        b = cp.predict_population(self.st, Wj, eps, protocol=False)
        self.assertGreaterEqual(float((a == b).mean()), 0.999)
        for p in range(len(self.genes)):
            ref = cp.dense_weights(self.st, W64, p)
            rel = np.abs(cp.dense_weights(self.st, Wj, p) - ref).max() / max(1e-9, np.abs(ref).max())
            self.assertLess(rel, 1e-4)

    def test_pareto_front(self):
        pts = np.array([[1.0, -640], [1.0, -160], [0.9, -100], [0.8, -200]])
        self.assertEqual(cp.pareto_front(pts).tolist(), [False, True, True, False])


if __name__ == "__main__":
    unittest.main()
