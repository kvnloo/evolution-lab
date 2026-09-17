from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evolution_lab.cycle import _better, _write_overlay, league_dir
from evolution_lab.select import BenchMetrics, BenchResult, FlyCandidate, load_config


def _cand() -> FlyCandidate:
    return FlyCandidate(genome_id="x", genome={"id": "x"})


def _result(score: float, ok: bool = True) -> BenchResult:
    return BenchResult(
        candidate=_cand(),
        metrics=BenchMetrics(closed_loop_reward=1.0 if ok else 0.0),
        latency_score=score,
        gates_pass=ok,
    )


class CycleUnitTests(unittest.TestCase):
    def test_better_requires_gates_and_lower_score(self):
        self.assertTrue(_better(_result(5.0), _result(6.0)))
        self.assertFalse(_better(_result(6.5), _result(6.0)))
        self.assertFalse(_better(_result(1.0, ok=False), _result(6.0)))
        self.assertTrue(_better(_result(6.0), None))

    def test_overlay_drops_history_and_sets_epsilon(self):
        cfg = load_config()
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_overlay(cfg, 0.02, Path(tmp) / "overlay.json")
            data = json.loads(path.read_text())
            self.assertEqual(data["objective"]["improve_epsilon"], 0.02)
            self.assertNotIn("history", data["search_space"])


if __name__ == "__main__":
    unittest.main()
