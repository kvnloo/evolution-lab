"""Unit coverage for the inexpensive experiment_2x2 helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evolution_lab.experiment_2x2 import _ctx_from_ep, _softmax_rows, load_episodes


class Experiment2x2HelperTests(unittest.TestCase):
    def test_ctx_from_ep_fields(self):
        rec = {
            "tool": "pytest",
            "prev": [1, 2],
            "tier": "fail",
            "session": "abcdefghijklmnop",
        }
        self.assertEqual(
            _ctx_from_ep(rec),
            {
                "last_tool": "pytest",
                "session_depth": 2,
                "phase": "verify",
                "has_retry": True,
                "last_tool_fail": True,
                "has_test_signal": True,
                "harness": "hermes",
                "repo": "abcdefghijkl",
                "cwd": "",
                "candidate_count": 0,
            },
        )

    def test_ctx_phase_mapping(self):
        for tool, phase in (
            ("ApplyPatch", "edit"),
            ("Grep", "search"),
            ("Task", "delegate"),
            ("unknown", ""),
        ):
            with self.subTest(tool=tool):
                ctx = _ctx_from_ep({"tool": tool})
                self.assertEqual(ctx["phase"], phase)
                self.assertEqual(ctx["session_depth"], 0)

        missing = _ctx_from_ep({})
        self.assertEqual(missing["last_tool"], "")
        self.assertEqual(missing["phase"], "")
        self.assertEqual(missing["session_depth"], 0)

    def test_load_episodes_plain_and_rich(self):
        records = [
            {"family": "EDIT", "text": "fix the bug", "session": "s1", "gold": True},
            {"family": "NOPE", "text": "x"},
            {
                "family": "READ_SEARCH",
                "user": "find file",
                "prev": ["a"],
                "tier": "gold",
                "session": "s2",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episodes.jsonl"
            path.write_text(
                "".join(json.dumps(rec) + "\n" for rec in records),
                encoding="utf-8",
            )

            X, y, gold, sessions = load_episodes(path)
            self.assertEqual(X.shape, (2, 64))
            self.assertEqual(X.dtype, np.dtype("float32"))
            self.assertEqual(y.tolist(), [1, 0])
            self.assertEqual(gold.tolist(), [True, True])
            self.assertEqual(sessions, ["s1", "s2"])

            X_rich, y_rich, gold_rich, sessions_rich = load_episodes(path, rich=True)
            self.assertEqual(X_rich.shape, (2, 124))
            self.assertEqual(X_rich.dtype, np.dtype("float32"))
            np.testing.assert_array_equal(y_rich, y)
            np.testing.assert_array_equal(gold_rich, gold)
            self.assertEqual(sessions_rich, sessions)
            np.testing.assert_allclose(
                np.linalg.norm(X_rich, axis=1), [1.0, 1.0], atol=1e-6
            )

    def test_load_episodes_skips_blank_lines(self):
        records = [
            {"family": "EDIT", "text": "fix the bug", "session": "s1", "gold": True},
            {"family": "READ_SEARCH", "text": "find file", "session": "s2"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            clean = Path(tmp) / "clean.jsonl"
            blank = Path(tmp) / "blank.jsonl"
            clean.write_text(
                "".join(json.dumps(rec) + "\n" for rec in records),
                encoding="utf-8",
            )
            blank.write_text(
                json.dumps(records[0]) + "\n\n \t\n" + json.dumps(records[1]) + "\n",
                encoding="utf-8",
            )

            for rich in (False, True):
                with self.subTest(rich=rich):
                    expected = load_episodes(clean, rich=rich)
                    actual = load_episodes(blank, rich=rich)
                    self.assertEqual(actual[0].shape, (2, 124 if rich else 64))
                    np.testing.assert_array_equal(actual[0], expected[0])
                    np.testing.assert_array_equal(actual[1], expected[1])
                    np.testing.assert_array_equal(actual[2], expected[2])
                    self.assertEqual(actual[3], ["s1", "s2"])

    def test_load_episodes_no_known_family_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "unknown.jsonl"
            path.write_text(
                json.dumps({"family": "NOPE", "text": "ignored"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(FileNotFoundError) as raised:
                load_episodes(path)
            self.assertIn(str(path), str(raised.exception))

    def test_softmax_rows(self):
        probs = _softmax_rows(np.array([[1000.0, 0.0], [1.0, 1.0]]))
        self.assertEqual(probs.shape, (2, 2))
        self.assertTrue(np.isfinite(probs).all())
        np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(probs[0], [1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(probs[1], [0.5, 0.5], atol=1e-12)
