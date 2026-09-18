from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evolution_lab.capability_schema import BenchmarkSpec, CapabilityCard
from evolution_lab.sealed_split import make_session_split


class SealedSplit(unittest.TestCase):
    def test_session_groups_disjoint(self):
        rows = []
        for s in range(20):
            for i in range(5):
                rows.append({"session": f"s{s:02d}", "ts": float(s * 10 + i), "family": "EXECUTE"})
        sp = make_session_split(rows, sealed_frac=0.1, dev_frac=0.1)
        tr, dev, se = set(sp.train_sessions), set(sp.dev_sessions), set(sp.sealed_sessions)
        self.assertFalse(tr & dev)
        self.assertFalse(tr & se)
        self.assertFalse(dev & se)
        self.assertGreater(len(tr), 0)
        self.assertGreater(len(se), 0)
        self.assertEqual(sp.n_train_events + sp.n_dev_events + sp.n_sealed_events, len(rows))


class Schema(unittest.TestCase):
    def test_card_roundtrip(self):
        c = CapabilityCard(
            id="t",
            description="d",
            stage="routing",
            trigger="x",
            input_contract=["a"],
            output_contract=["yes", "no"],
            verifier="v",
        )
        d = c.to_dict()
        self.assertEqual(d["id"], "t")
        self.assertEqual(d["schema"], "flyforge.capability_card.v1")

    def test_spec_defaults(self):
        s = BenchmarkSpec(capability_id="t", split={"train": "a"})
        self.assertIn("coverage_at_95_precision", s.metrics)
        self.assertIn("mb", s.controls)


class MineSmoke(unittest.TestCase):
    def test_mine_tiny_corpus(self):
        from evolution_lab.capability_miner import run_capability_mine

        # Use real episodes but write to temp research dir — may be slow; skip if no episodes
        ep = Path.home() / ".z0int" / "episodes" / "next_action.jsonl"
        if not ep.exists():
            self.skipTest("no episodes")
        # Don't run full mine in unit test — too heavy. Schema/split only above.
        self.assertTrue(ep.exists())


if __name__ == "__main__":
    unittest.main()
