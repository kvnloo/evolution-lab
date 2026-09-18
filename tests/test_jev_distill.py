from __future__ import annotations

import unittest

import numpy as np

from evolution_lab.jev_distill import _hash_ngrams, collapse_labels, featurize, majority


class JevDistillUnit(unittest.TestCase):
    def test_hash_stable_and_unitish(self):
        a = _hash_ngrams("install a keyboard shortcut")
        b = _hash_ngrams("install a keyboard shortcut")
        c = _hash_ngrams("research this on the web")
        self.assertTrue(np.allclose(a, b))
        self.assertGreater(float(np.linalg.norm(a)), 0.99)
        self.assertLess(float(a @ c), 0.95)

    def test_collapse_and_majority(self):
        pairs = [{"label": "shell"}] * 5 + [{"label": "rare"}] * 2 + [{"label": "none_of_these"}] * 3
        out = collapse_labels(pairs, min_count=3)
        self.assertEqual(out[5]["label"], "other")
        X, y, labels = featurize([{"prompt": "a", "label": x["label"]} for x in out])
        self.assertGreaterEqual(len(labels), 2)
        maj = majority(y[:5], 3)
        self.assertEqual(len(maj), 3)


if __name__ == "__main__":
    unittest.main()
