"""Hermetic contract tests for the thin outer capability loop."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import call, patch

from evolution_lab.outer_loop import run_outer_loop


class OuterLoopTests(unittest.TestCase):
    def setUp(self):
        temporary_home = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_home.cleanup)
        self.home = Path(temporary_home.name)

        patches = ExitStack()
        self.addCleanup(patches.close)
        patches.enter_context(
            patch.dict(
                os.environ,
                {
                    "HOME": str(self.home),
                    "Z0INT_HOME": str(self.home / ".z0int"),
                    "XDG_CACHE_HOME": str(self.home / ".cache"),
                },
            )
        )
        patches.enter_context(
            patch("evolution_lab.outer_loop.Path.home", return_value=self.home)
        )
        self.mine_mock = patches.enter_context(
            patch("evolution_lab.capability_miner.run_capability_mine", return_value={})
        )
        self.l2_mock = patches.enter_context(
            patch("evolution_lab.l2_benchmark.run_l2_recovery_benchmark", return_value={})
        )
        self.canary_mock = patches.enter_context(
            patch("evolution_lab.l2_benchmark.canary_active", return_value=True)
        )

    def test_no_steps_no_write(self):
        out = run_outer_loop(mine=False, l2=False, write=False)

        self.assertEqual(out["schema"], "z0int.outer_loop.v1")
        self.assertEqual(out["steps"], {})
        self.assertNotIn("path", out)
        self.assertGreaterEqual(out["elapsed_s"], 0)
        self.assertIsInstance(out["next"], list)
        self.assertEqual(len(out["next"]), 4)
        self.assertTrue(all(isinstance(item, str) for item in out["next"]))
        self.assertFalse((self.home / ".z0int").exists())
        self.mine_mock.assert_not_called()
        self.l2_mock.assert_not_called()
        self.canary_mock.assert_not_called()

    def test_mine_step_truncates_keys(self):
        keys = [f"k{i}" for i in range(20)]
        self.mine_mock.side_effect = [dict.fromkeys(keys), None]

        populated = run_outer_loop(mine=True, l2=False, write=False)
        empty = run_outer_loop(mine=True, l2=False, write=False)

        self.assertEqual(
            populated["steps"]["capability_mine"],
            {"ok": True, "summary_keys": keys[:12]},
        )
        self.assertEqual(
            empty["steps"]["capability_mine"],
            {"ok": True, "summary_keys": []},
        )
        self.assertEqual(self.mine_mock.call_args_list, [call(), call()])
        self.l2_mock.assert_not_called()
        self.canary_mock.assert_not_called()
        self.assertFalse((self.home / ".z0int").exists())

    def test_l2_step_passes_args_and_gate(self):
        gates = {"pass": True}
        self.l2_mock.return_value = {
            "gates": gates,
            "canary_marker": "m",
            "report_path": "r",
        }

        out = run_outer_loop(mine=False, closed_loop_seeds=5, write=False)

        self.assertEqual(
            out["steps"]["l2_recovery"],
            {
                "ok": True,
                "gates": gates,
                "canary_marker": "m",
                "report_path": "r",
                "canary_active": True,
            },
        )
        self.l2_mock.assert_called_once_with(
            closed_loop_seeds=5, write=False, promote=True
        )
        self.canary_mock.assert_called_once_with()
        self.mine_mock.assert_not_called()
        self.assertFalse((self.home / ".z0int").exists())

    def test_l2_missing_gates_is_not_ok(self):
        self.l2_mock.return_value = {}

        out = run_outer_loop(mine=False, write=False)

        step = out["steps"]["l2_recovery"]
        self.assertIs(step["ok"], False)
        self.assertIsNone(step["gates"])
        self.assertIsNone(step["canary_marker"])
        self.assertIsNone(step["report_path"])
        self.assertIs(step["canary_active"], True)
        self.l2_mock.assert_called_once_with(
            closed_loop_seeds=24, write=False, promote=True
        )
        self.canary_mock.assert_called_once_with()
        self.mine_mock.assert_not_called()
        self.assertFalse((self.home / ".z0int").exists())

    def test_write_creates_report(self):
        out = run_outer_loop(mine=False, l2=False, write=True)

        dest = self.home / ".z0int" / "research" / "outer-loop-last.json"
        self.assertEqual(out["path"], str(dest))
        self.assertTrue(dest.is_file())
        saved = json.loads(dest.read_text(encoding="utf-8"))
        self.assertEqual(saved["schema"], out["schema"])
        self.assertEqual(saved["steps"], out["steps"])
        self.assertNotIn("path", saved)
        self.mine_mock.assert_not_called()
        self.l2_mock.assert_not_called()
        self.canary_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
