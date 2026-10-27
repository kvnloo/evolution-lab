"""Edge-case contract tests for chronological sealed session splits."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from evolution_lab.sealed_split import (
    load_episode_rows,
    make_session_split,
    mask_for_bucket,
)


def rows(n: int) -> list[dict[str, Any]]:
    return [{"session": f"s{i}", "ts": float(i)} for i in range(n)]


class SealedSplitEdgeCases(unittest.TestCase):
    def test_default_split_is_chronological(self):
        split = make_session_split(rows(10))

        self.assertEqual(split.train_sessions, [f"s{i}" for i in range(8)])
        self.assertEqual(split.dev_sessions, ["s8"])
        self.assertEqual(split.sealed_sessions, ["s9"])
        self.assertEqual(split.future_sessions, [])
        self.assertEqual(
            (
                split.n_train_events,
                split.n_dev_events,
                split.n_sealed_events,
                split.n_future_events,
            ),
            (8, 1, 1, 0),
        )

    def test_future_fraction(self):
        split = make_session_split(rows(10), future_frac=0.2)

        self.assertEqual(split.train_sessions, [f"s{i}" for i in range(6)])
        self.assertEqual(split.dev_sessions, ["s6"])
        self.assertEqual(split.sealed_sessions, ["s7"])
        self.assertEqual(split.future_sessions, ["s8", "s9"])
        self.assertEqual(
            (
                split.n_train_events,
                split.n_dev_events,
                split.n_sealed_events,
                split.n_future_events,
            ),
            (6, 1, 1, 2),
        )

    def test_order_uses_max_ts_then_name(self):
        # Deliberately unsorted: b's older row must not replace its max timestamp.
        unordered = [
            {"session": "c", "ts": 5.0},
            {"session": "b", "ts": 5.0},
            {"session": "b", "ts": 1.0},
            {"session": "a", "ts": 3.0},
        ]
        split = make_session_split(unordered, sealed_frac=0, dev_frac=0)

        self.assertEqual(split.train_sessions, ["a", "b", "c"])
        self.assertEqual(split.dev_sessions, [])
        self.assertEqual(split.sealed_sessions, [])
        self.assertEqual(split.future_sessions, [])
        self.assertEqual(split.n_train_events, 4)

    def test_tiny_inputs(self):
        empty = make_session_split(rows(0))
        self.assertEqual(
            (
                empty.train_sessions,
                empty.dev_sessions,
                empty.sealed_sessions,
                empty.future_sessions,
            ),
            ([], [], [], []),
        )
        self.assertEqual(
            (
                empty.n_train_events,
                empty.n_dev_events,
                empty.n_sealed_events,
                empty.n_future_events,
            ),
            (0, 0, 0, 0),
        )

        one = make_session_split([{"session": "a", "ts": 1.0}])
        self.assertEqual(one.train_sessions, [])
        self.assertEqual(one.dev_sessions, [])
        self.assertEqual(one.sealed_sessions, ["a"])
        self.assertEqual(one.future_sessions, [])
        self.assertEqual(one.n_sealed_events, 1)

        two = make_session_split(
            [{"session": "a", "ts": 1.0}, {"session": "b", "ts": 2.0}]
        )
        self.assertEqual(two.train_sessions, ["a"])
        self.assertEqual(two.dev_sessions, [])
        self.assertEqual(two.sealed_sessions, ["b"])
        self.assertEqual(two.future_sessions, [])
        self.assertEqual((two.n_train_events, two.n_sealed_events), (1, 1))

        no_holds = make_session_split(rows(3), sealed_frac=0, dev_frac=0)
        self.assertEqual(no_holds.train_sessions, ["s0", "s1", "s2"])
        self.assertEqual(no_holds.dev_sessions, [])
        self.assertEqual(no_holds.sealed_sessions, [])
        self.assertEqual(no_holds.future_sessions, [])
        self.assertEqual(no_holds.n_train_events, 3)

    def test_rows_without_session_are_ignored_and_counted_by_session(self):
        events = [
            {"session": "b", "ts": 3.0},
            {"session": "", "ts": 90.0},
            {"ts": 99.0},
            {"session": "b", "ts": 2.0},
            {"session": "a", "ts": 1.0},
            {"session": "b", "ts": 4.0},
        ]
        split = make_session_split(events)

        self.assertEqual(split.train_sessions, ["a"])
        self.assertEqual(split.dev_sessions, [])
        self.assertEqual(split.sealed_sessions, ["b"])
        self.assertEqual(split.future_sessions, [])
        self.assertEqual(
            (
                split.n_train_events,
                split.n_dev_events,
                split.n_sealed_events,
                split.n_future_events,
            ),
            (1, 0, 3, 0),
        )

    def test_bucket_of_and_mask(self):
        split = make_session_split(rows(10))
        for session, bucket in {
            "s0": "train",
            "s8": "dev",
            "s9": "sealed",
            "zz": "unknown",
        }.items():
            with self.subTest(session=session):
                self.assertEqual(split.bucket_of(session), bucket)

        future_split = make_session_split(rows(10), future_frac=0.2)
        self.assertEqual(future_split.bucket_of("s9"), "future")
        self.assertEqual(
            mask_for_bucket(["s0", "s9", "zz"], split, "sealed"),
            [False, True, False],
        )
        with self.assertRaises(KeyError):
            mask_for_bucket(["s9"], split, "not-a-bucket")

    def test_load_episode_rows_skips_blank_lines(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "episodes.jsonl"
            path.write_text(
                '{"session": "a", "ts": 1.0}\n'
                "\n"
                "  \n"
                '{"session": "b", "ts": 2.0}\n',
                encoding="utf-8",
            )
            loaded = load_episode_rows(path)

        self.assertEqual(
            loaded,
            [{"session": "a", "ts": 1.0}, {"session": "b", "ts": 2.0}],
        )
        self.assertEqual(
            make_session_split(loaded).to_dict()["schema"],
            "flyforge.session_split.v1",
        )


if __name__ == "__main__":
    unittest.main()
