from __future__ import annotations

import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np

from evolution_lab.cli import main
from evolution_lab.schema import ACTIONS, Architecture, Curriculum, ExperimentGenome
from evolution_lab.splits import (
    SPLIT_SEED,
    lock_splits,
    load_manifest,
    load_splits,
    sample_canonical_task,
    splits_exist,
)
from evolution_lab.task import build_task, sample_task


def _p0_genome() -> ExperimentGenome:
    return ExperimentGenome(
        id="lock-test",
        lineage="lock",
        hypothesis="locked delayed-cue",
        architecture=Architecture(history=8),
        curriculum=Curriculum(delayed_cue=True),
    )


class SplitsTests(unittest.TestCase):
    def test_seed_is_documented_20260912(self):
        self.assertEqual(SPLIT_SEED, 20260912)
        from evolution_lab.task import SPLIT_SEED as task_seed

        self.assertEqual(task_seed, 20260912)

    def test_lock_roundtrip_matches_seed_draw(self):
        tmp = Path(tempfile.mkdtemp())
        generated = sample_canonical_task(seed=SPLIT_SEED)
        dest = lock_splits(tmp, seed=SPLIT_SEED)
        self.assertTrue(splits_exist(dest))
        loaded = load_splits(dest)
        manifest = load_manifest(dest)
        self.assertEqual(manifest["seed"], 20260912)
        self.assertTrue(manifest["delayed_cue"])
        self.assertEqual(manifest["history"], 8)
        self.assertEqual(len(loaded.train), 192)
        self.assertEqual(len(loaded.val), 48)
        self.assertEqual(len(loaded.confirm), 48)
        self.assertEqual(len(loaded.ood), 32)
        np.testing.assert_array_equal(loaded.train[0].frames, generated.train[0].frames)
        np.testing.assert_array_equal(loaded.train[-1].labels, generated.train[-1].labels)
        np.testing.assert_array_equal(loaded.ood[-1].labels, generated.ood[-1].labels)
        self.assertTrue(all(ep.env == "ood" for ep in loaded.ood))
        self.assertTrue(all(ep.env == "iid" for ep in loaded.train))
        self.assertFalse(any(ep.secret for ep in loaded.train + loaded.ood))

    def test_build_task_reads_disk_not_rng(self):
        tmp = Path(tempfile.mkdtemp())
        lock_splits(tmp, seed=SPLIT_SEED)
        train_path = tmp / "train.npz"
        with np.load(train_path) as z:
            frames = np.array(z["frames"], copy=True)
            labels = np.array(z["labels"])
            env = np.array(z["env"])
            secret = np.array(z["secret"])
            seed = np.array(z["seed"])
        frames[0, 0, 0] = 0.123456
        np.savez_compressed(
            train_path,
            frames=frames,
            labels=labels,
            env=env,
            secret=secret,
            seed=seed,
        )
        data = build_task(_p0_genome(), splits_dir=tmp)
        self.assertAlmostEqual(float(data.train[0].frames[0, 0]), 0.123456)
        sampled = sample_task(_p0_genome(), seed=SPLIT_SEED)
        self.assertNotAlmostEqual(float(sampled.train[0].frames[0, 0]), 0.123456)

    def test_delayed_cue_contract_on_locked_train(self):
        tmp = Path(tempfile.mkdtemp())
        lock_splits(tmp)
        data = load_splits(tmp)
        cued = [ep for ep in data.train if ep.frames[0, 9] > 0.5]
        self.assertTrue(cued)
        for ep in cued:
            self.assertEqual(ACTIONS[int(ep.labels[-1])], "escalate")
            self.assertLess(ep.frames[-1, 9], 0.5)

    def test_incompatible_history_does_not_load_lock(self):
        tmp = Path(tempfile.mkdtemp())
        lock_splits(tmp)
        g = ExperimentGenome(
            id="t6",
            lineage="t",
            hypothesis="other T",
            architecture=Architecture(history=6),
            curriculum=Curriculum(delayed_cue=True),
        )
        data = build_task(g, n_train=8, n_val=4, n_confirm=4, n_ood=4, splits_dir=tmp)
        self.assertEqual(data.train[0].frames.shape[0], 6)

    def test_committed_p0_binds_to_seed(self):
        from evolution_lab.splits import default_splits_dir

        self.assertTrue(splits_exist(default_splits_dir()), "data/p0/ must be git-tracked")
        loaded = load_splits()
        generated = sample_canonical_task(seed=SPLIT_SEED)
        np.testing.assert_array_equal(loaded.train[0].frames, generated.train[0].frames)
        np.testing.assert_array_equal(loaded.confirm[-1].labels, generated.confirm[-1].labels)
        from_disk = build_task(_p0_genome())
        np.testing.assert_array_equal(from_disk.val[0].frames, generated.val[0].frames)
        manifest = load_manifest()
        self.assertEqual(manifest["seed"], 20260912)

    def test_cli_lock_splits(self):
        tmp = Path(tempfile.mkdtemp())
        buf = StringIO()
        with patch("sys.stdout", buf):
            rc = main(["lock-splits", "--data-dir", str(tmp)])
        self.assertEqual(rc, 0)
        self.assertTrue((tmp / "manifest.json").is_file())
        manifest = json.loads((tmp / "manifest.json").read_text())
        self.assertEqual(manifest["seed"], 20260912)
        self.assertIn("locked splits", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
