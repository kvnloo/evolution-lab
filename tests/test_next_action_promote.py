from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evolution_lab.next_action_gpu import _maybe_promote_champion


class PromoteGate(unittest.TestCase):
    def test_weaker_candidate_does_not_clobber(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = root / "data" / "next_action"
            runs = root / "runs" / "gpu-evolve"
            data.mkdir(parents=True)
            runs.mkdir(parents=True)
            pack = {
                "W_pn_kc": np.zeros((64, 96)),
                "W_kc_mbon": np.zeros((96, 8), dtype=np.float32),
                "k_winners": np.int32(20),
                "n_kc": np.int32(96),
                "acc": np.float64(0.57),
                "ridge": np.float64(0.36),
                "majority": np.float64(0.35),
            }
            np.savez(data / "champion.npz", **pack)
            weaker = dict(pack)
            weaker["acc"] = np.float64(0.39)
            report = {
                "recipe": {"source": "next_action"},
                "n_train": 100,
                "gpu_best_confirm_acc": 0.39,
                "ridge_confirm_acc": 0.37,
                "majority_confirm_acc": 0.35,
                "n_kc": 96,
                "k_winners": 10,
                "pn_dim": 64,
            }
            self.assertFalse(_maybe_promote_champion(root, weaker, report, n_pop=128))
            kept = float(np.load(data / "champion.npz")["acc"])
            self.assertAlmostEqual(kept, 0.57)

    def test_stronger_candidate_promotes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = root / "data" / "next_action"
            (root / "runs" / "gpu-evolve").mkdir(parents=True)
            data.mkdir(parents=True)
            pack = {
                "W_pn_kc": np.zeros((64, 96)),
                "W_kc_mbon": np.zeros((96, 8), dtype=np.float32),
                "k_winners": np.int32(20),
                "n_kc": np.int32(96),
                "acc": np.float64(0.40),
                "ridge": np.float64(0.36),
                "majority": np.float64(0.35),
            }
            np.savez(data / "champion.npz", **pack)
            stronger = dict(pack)
            stronger["acc"] = np.float64(0.58)
            report = {
                "recipe": {"source": "next_action"},
                "n_train": 100,
                "gpu_best_confirm_acc": 0.58,
                "ridge_confirm_acc": 0.36,
                "majority_confirm_acc": 0.35,
                "n_kc": 96,
                "k_winners": 20,
                "pn_dim": 64,
            }
            self.assertTrue(_maybe_promote_champion(root, stronger, report, n_pop=256))
            meta = json.loads((data / "champion.json").read_text())
            self.assertAlmostEqual(meta["gpu_best_confirm_acc"], 0.58)
            self.assertTrue((root / "runs" / "gpu-evolve" / "next_action_champion.npz").exists())


if __name__ == "__main__":
    unittest.main()
