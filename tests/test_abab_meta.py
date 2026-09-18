from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evolution_lab.abab_meta import choose_b_action, load_or_seed, record_b, seed_fly_world
from evolution_lab.abab_state import save


class AbabMetaTests(unittest.TestCase):
    def test_seed_has_promoted_width_floor(self):
        w = seed_fly_world()
        h2 = next(h for h in w.hypotheses if h.id == "H2")
        self.assertEqual(h2.status, "promoted")

    def test_choose_autoresearch_until_idle(self):
        w = seed_fly_world()
        self.assertEqual(choose_b_action(w, idle=0, idle_limit=2), "autoresearch")
        self.assertEqual(choose_b_action(w, idle=2, idle_limit=2), "sweep")

    def test_no_update_does_not_stop_forever_cycle(self):
        w = seed_fly_world()
        for _ in range(5):
            w = record_b(w, action="autoresearch", keeps=0, note="idle")
        self.assertFalse(w.stopped)
        self.assertEqual(w.loop, "slow")
        with tempfile.TemporaryDirectory() as tmp:
            league = Path(tmp)
            save(w, league / "abab.json")
            loaded = load_or_seed(league)
            self.assertGreaterEqual(loaded.wave, 5)


if __name__ == "__main__":
    unittest.main()
