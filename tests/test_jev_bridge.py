from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evolution_lab.jev_bridge import episode_to_row, lock_hermes_as_jev
from evolution_lab.schema import ACTIONS
from evolution_lab.splits import sample_canonical_task


class JevBridgeTests(unittest.TestCase):
    def test_episode_maps_to_five_actions(self):
        episode = sample_canonical_task().train[0]
        row = episode_to_row(episode)
        self.assertEqual(row.options, tuple(ACTIONS))
        self.assertIn(row.label, range(len(ACTIONS)))
        self.assertIn("Hermes recovery state", row.context)

    def test_lock_hermes_as_jev_writes_jsonl(self):
        tmp = Path(tempfile.mkdtemp())
        out = lock_hermes_as_jev(tmp, counts={"train": 4, "val": 2, "confirm": 2, "ood": 2})
        payload = json.loads((out / "train.jsonl").read_text().splitlines()[0])
        self.assertIn("context", payload)
        self.assertEqual(len(payload["options"]), len(ACTIONS))
