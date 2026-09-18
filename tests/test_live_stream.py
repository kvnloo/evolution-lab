from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from evolution_lab import live_stream as ls


class LiveStream(unittest.TestCase):
    def test_high_info_on_disagree(self):
        row = {"disagree": True, "decision": {"route": "local", "label": "EXECUTE"}}
        self.assertTrue(ls.is_high_info(row))

    def test_append_and_pools(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw = root / "raw.jsonl"
            hi = root / "high_info.jsonl"
            with mock.patch.object(ls, "STREAM_ROOT", root), mock.patch.object(ls, "RAW", raw), mock.patch.object(
                ls, "HIGH_INFO", hi
            ), mock.patch.object(ls, "OUTCOME_GOLD", root / "gold.jsonl"):
                row = ls.build_live_row(
                    prompt="spawn a scout on the auth module",
                    decision={"route": "escalate", "label": "EDIT", "reason": "family_not_local:EDIT"},
                    fly={"ok": True, "label": "EDIT", "p": 0.4, "probs": {"EDIT": 0.4, "EXECUTE": 0.3}},
                    jev={"ok": True, "label": "shell", "p": 0.5},
                    session_id="s1",
                )
                out = ls.append_decision(row)
                self.assertTrue(out["high_info"])
                self.assertTrue(raw.exists())
                self.assertTrue(hi.exists())
                self.assertEqual(ls.pool_stats()["raw"], 1)
                self.assertEqual(ls.pool_stats()["high_info"], 1)


if __name__ == "__main__":
    unittest.main()
