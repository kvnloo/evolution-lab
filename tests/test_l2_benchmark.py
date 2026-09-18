from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path


class L2RecoveryBenchmark(unittest.TestCase):
    def test_run_gates_and_schema(self):
        from evolution_lab.l2_benchmark import CAPABILITY_ID, run_l2_recovery_benchmark

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            old = os.environ.get("Z0INT_HOME")
            os.environ["Z0INT_HOME"] = str(home)
            try:
                report = run_l2_recovery_benchmark(closed_loop_seeds=4, write=True, promote=True)
                self.assertEqual(report["schema"], "z0int.l2_benchmark.v1")
                self.assertEqual(report["capability_id"], CAPABILITY_ID)
                self.assertEqual(report["evidence_level"], "L2_outcome")
                self.assertIn("student", report["controls"])
                self.assertTrue(report["gates"]["bundle_present"])
                self.assertTrue(report["gates"]["pass"], report["gates"])
                self.assertEqual(report["execution_policy"]["current"], "canary")
                self.assertTrue((home / "benchmarks" / "recovery_action_l2.json").is_file(), report)
                self.assertTrue((home / "specialists" / "recovery_action.canary.json").is_file())
            finally:
                if old is None:
                    os.environ.pop("Z0INT_HOME", None)
                else:
                    os.environ["Z0INT_HOME"] = old

    def test_canary_active(self):
        from evolution_lab.l2_benchmark import canary_active

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            spec = home / "specialists"
            spec.mkdir(parents=True)
            old = os.environ.get("Z0INT_HOME")
            os.environ["Z0INT_HOME"] = str(home)
            try:
                self.assertFalse(canary_active())
                (spec / "recovery_action.canary.json").write_text(
                    '{"execution":"canary","capability_id":"recovery_action"}\n'
                )
                self.assertTrue(canary_active())
            finally:
                if old is None:
                    os.environ.pop("Z0INT_HOME", None)
                else:
                    os.environ["Z0INT_HOME"] = old


if __name__ == "__main__":
    unittest.main()
