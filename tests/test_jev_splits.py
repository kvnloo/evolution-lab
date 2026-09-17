from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evolution_lab.jev_splits import JEV_SPLIT_SEED, lock_splits, load_manifest, splits_exist


class JevSplitTests(unittest.TestCase):
    def test_lock_is_deterministic(self):
        tmp = Path(tempfile.mkdtemp())
        first = lock_splits(tmp / "a", seed=JEV_SPLIT_SEED)
        second = lock_splits(tmp / "b", seed=JEV_SPLIT_SEED)
        self.assertTrue(splits_exist(first))
        self.assertEqual(
            (first / "train.jsonl").read_text(),
            (second / "train.jsonl").read_text(),
        )
        manifest = load_manifest(first)
        self.assertEqual(manifest["task"], "jev_synthetic")
        self.assertEqual(manifest["seed"], JEV_SPLIT_SEED)
