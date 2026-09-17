from __future__ import annotations

import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from evolution_lab.cli import main
from evolution_lab.table import (
    CONTROL_FAMILIES,
    kill_criterion,
    rows_by_family,
    vs_teacher,
)


def _row(family: str, val: float, confirm: float, cost: float, params: int = 10) -> dict:
    return {
        "status": "ok",
        "experiment_id": f"{family}-x",
        "genome": {"architecture": {"family": family}},
        "metrics": {
            "val_success": val,
            "success_rate": confirm,
            "cost": cost,
            "params": params,
            "violations": 0.0,
            "ood_score": val,
        },
    }


class TableTests(unittest.TestCase):
    def test_kill_when_gru_and_direct_dominate_reservoir(self):
        rows = [
            _row("rule", 1.0, 1.0, 0.01, 0),
            _row("gru", 0.9, 0.9, 0.2),
            _row("direct_input", 0.85, 0.8, 0.05),
            _row("fixed_reservoir", 0.7, 0.7, 0.15),
        ]
        verdict = kill_criterion(rows)
        self.assertTrue(verdict["gru_and_direct_dominate_reservoir"])
        self.assertTrue(verdict["do_not_run_malecns_sgd"])
        self.assertTrue(verdict["pivot_to_motif_students"])

    def test_no_kill_when_reservoir_wins_val(self):
        rows = [
            _row("rule", 1.0, 1.0, 0.01, 0),
            _row("gru", 0.92, 0.92, 0.2),
            _row("direct_input", 0.79, 0.79, 0.05),
            _row("fixed_reservoir", 1.0, 1.0, 0.15),
        ]
        verdict = kill_criterion(rows)
        self.assertFalse(verdict["gru_and_direct_dominate_reservoir"])
        self.assertFalse(verdict["do_not_run_malecns_sgd"])

    def test_vs_teacher_uses_product_target(self):
        rows = [
            _row("rule", 1.0, 1.0, 0.2, 0),
            _row("mlp", 0.96, 0.96, 0.08),
        ]
        report = vs_teacher(rows)
        self.assertAlmostEqual(report["teacher_success"], 1.0)
        self.assertAlmostEqual(report["best_learned_success"], 0.96)
        self.assertGreaterEqual(report["success_vs_teacher"], 0.95)
        self.assertTrue(report["meets_success_target"])

    def test_vs_teacher_prefers_cheaper_perfect_student(self):
        rows = [
            _row("rule", 1.0, 1.0, 0.2, 0),
            _row("mlp", 1.0, 1.0, 0.08, 4000),
            _row("local_plasticity", 1.0, 1.0, 0.02, 640),
        ]
        report = vs_teacher(rows)
        self.assertEqual(report["best_learned_family"], "local_plasticity")
        self.assertLess(report["cost_vs_mlp"], 1.0)

    def test_rows_by_family_skips_failed(self):
        rows = [
            _row("mlp", 1.0, 1.0, 0.1),
            {"status": "failed", "genome": {"architecture": {"family": "gru"}}, "metrics": {}},
        ]
        by_fam = rows_by_family(rows)
        self.assertIn("mlp", by_fam)
        self.assertNotIn("gru", by_fam)

    def test_control_families_include_motif(self):
        for name in (
            "rule",
            "direct_input",
            "mlp",
            "gru",
            "fixed_reservoir",
            "rewired_reservoir",
            "local_plasticity",
        ):
            self.assertIn(name, CONTROL_FAMILIES)

    def test_cli_table_writes_json(self):
        tmp = Path(tempfile.mkdtemp())
        buf = StringIO()
        with patch("sys.stdout", buf):
            rc = main(["table", "--run-dir", str(tmp), "--level", "0"])
        self.assertEqual(rc, 0)
        path = tmp / "control_table.json"
        self.assertTrue(path.is_file())
        data = json.loads(path.read_text())
        self.assertIn("kill_criterion", data)
        self.assertIn("vs_teacher", data)
        families = {r["family"] for r in data["rows"] if r.get("status") == "ok"}
        self.assertIn("local_plasticity", families)
        self.assertIn("direct_input", families)
        self.assertIn("do_not_run_malecns_sgd", data["kill_criterion"])


if __name__ == "__main__":
    unittest.main()
