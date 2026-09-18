from __future__ import annotations

import unittest

import numpy as np

from evolution_lab.autoresearch_propose import LOCKED_HISTORY, propose_candidate
from evolution_lab.select import FlyCandidate, load_config


class ProposeTests(unittest.TestCase):
    def test_never_mutates_locked_history(self):
        cfg = load_config()
        genome = {
            "id": "x",
            "lineage": "mushroom-body",
            "hypothesis": "h",
            "architecture": {
                "family": "local_plasticity",
                "hidden": 128,
                "history": 8,
                "sparsity": 0.1,
                "trainable": "kc_mbon",
                "topology": "mushroom_body_analogue",
            },
            "curriculum": {"task": "hermes_recovery", "delayed_cue": True},
            "training": {"algorithm": "local_plasticity", "seed": 1},
            "evaluation": {},
        }
        champ = FlyCandidate(genome_id="x", genome=genome)
        rng = np.random.default_rng(0)
        for _ in range(40):
            cand = propose_candidate(champ, rng, cfg)
            self.assertEqual(cand.genome["architecture"]["history"], LOCKED_HISTORY)
            self.assertNotIn("history=", cand.description)
        cfg["search_space"]["k_winners"] = [5, 10, 15]
        saw = False
        rng = np.random.default_rng(1)
        for _ in range(80):
            cand = propose_candidate(champ, rng, cfg)
            if cand.description.startswith("k_winners="):
                saw = True
                self.assertIn(cand.k_winners, (5, 10, 15))
        self.assertTrue(saw)


if __name__ == "__main__":
    unittest.main()
