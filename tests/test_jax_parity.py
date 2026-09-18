from __future__ import annotations

import time
import unittest

import numpy as np

from evolution_lab.engine import seed_genomes
from evolution_lab.jax_mb import (
    N_ACTIONS,
    fit_from_numpy,
    numpy_perms,
    population_mbon,
    predict_numpy,
)
from evolution_lab.models import _kc_codes, _local_plasticity_step_samples, _pn_features, _sparse_pn_kc
from evolution_lab.splits import load_splits


def _numpy_train(H: np.ndarray, y: np.ndarray, perms: np.ndarray, lr: float) -> np.ndarray:
    W = np.zeros((H.shape[1], N_ACTIONS), dtype=np.float64)
    for perm in perms:
        for i in perm:
            h = H[int(i)]
            scores = h @ W
            s = scores - scores.max()
            pred = np.exp(np.clip(s, -20, 20))
            pred = pred / pred.sum()
            target = np.zeros(N_ACTIONS, dtype=np.float64)
            target[int(y[int(i)])] = 1.0
            W = W + lr * np.outer(h, target - pred)
            W = W - 0.02 * lr * np.outer(h, pred)
        lr *= 0.92
    return W


class JaxMbParity(unittest.TestCase):
    def test_float32_train_matches_numpy_argmax(self):
        genome = next(g for g in seed_genomes() if g.architecture.family == "local_plasticity")
        data = load_splits()
        step_eps, y = _local_plasticity_step_samples(data.train[:48], history=genome.architecture.history)
        X = _pn_features(step_eps)
        rng = np.random.default_rng(0)
        n_kc = 96
        W_pn = _sparse_pn_kc(X.shape[1], n_kc, rng)
        k = 10
        H = _kc_codes(X, W_pn, k)
        perms = numpy_perms(len(y), epochs=8, rng=np.random.default_rng(1))
        Wn = _numpy_train(H, y, perms, 0.35)
        pack = fit_from_numpy(X, y, W_pn, k_winners=k, epochs=8, lr=0.35, perms=perms)
        pn = (H @ Wn).argmax(axis=1)
        pj = predict_numpy(pack["H"], pack["W_kc_mbon"])
        agree = float((pn == pj).mean())
        self.assertGreaterEqual(agree, 0.95, f"argmax agree {agree} device={pack['device']}")

    def test_population_vmap_runs(self):
        genome = next(g for g in seed_genomes() if g.architecture.family == "local_plasticity")
        data = load_splits()
        step_eps, y = _local_plasticity_step_samples(data.train[:32], history=genome.architecture.history)
        X = _pn_features(step_eps)
        rng = np.random.default_rng(0)
        W_pn = _sparse_pn_kc(X.shape[1], 64, rng)
        H = _kc_codes(X, W_pn, 8)
        P = 32
        bank = np.stack([numpy_perms(len(y), 4, np.random.default_rng(10 + p)) for p in range(P)])
        t0 = time.perf_counter()
        Ws = population_mbon(H, y, bank, 0.35)
        elapsed = time.perf_counter() - t0
        self.assertEqual(Ws.shape, (P, 64, 5))
        self.assertGreater(elapsed, 0.0)


if __name__ == "__main__":
    unittest.main()
