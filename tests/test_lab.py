from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evolution_lab.lab import run_lab


class LabTests(unittest.TestCase):
    def test_run_lab_writes_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            summary = run_lab(
                run_dir=run_dir,
                level=1,
                evolve_generations=0,
                dagger_rounds=1,
                fresh=True,
            )
            self.assertEqual(summary["schema"], "evolution-lab.experiment-summary.v1")
            self.assertIn("archive", summary)
            self.assertIn("control_table", summary)
            self.assertIn("dagger", summary)
            summary_path = run_dir / "experiment_summary.json"
            self.assertTrue(summary_path.is_file())
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["dagger_rounds"], 1)
