from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from evolution_lab.next_action_gpu import (
    DEFAULT_COVERAGE_POLICY,
    decide_next_action,
    load_coverage_policy,
)


class CoverageCascade(unittest.TestCase):
    def test_default_policy_has_exec_del_local(self):
        pol = DEFAULT_COVERAGE_POLICY
        self.assertIn("EXECUTE", pol["local_families"])
        self.assertIn("DELEGATE", pol["local_families"])
        self.assertEqual(pol["escalate_to"], "jev")
        self.assertEqual(pol["fallback"], "openjev")

    def test_load_policy_from_file(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "data" / "next_action" / "coverage_policy.json"
            path.parent.mkdir(parents=True)
            custom = {
                "schema": "flyforge.next_action_coverage.v1",
                "local_families": ["EXECUTE"],
                "thresholds": {"EXECUTE": {"min_p": 0.9, "min_margin": 0.5}},
                "escalate_to": "jev",
                "fallback": "openjev",
            }
            path.write_text(json.dumps(custom))
            loaded = load_coverage_policy(root=root)
            self.assertEqual(loaded["local_families"], ["EXECUTE"])
            self.assertEqual(loaded["thresholds"]["EXECUTE"]["min_p"], 0.9)

    def test_high_conf_execute_routes_local(self):
        fake = {
            "ok": True,
            "label": "EXECUTE",
            "p": 0.72,
            "margin": 0.41,
            "second": "READ_SEARCH",
            "source": "next_action_gpu_champion",
            "n_params": 6912,
        }
        with mock.patch("evolution_lab.next_action_gpu.predict_next_action", return_value=fake):
            out = decide_next_action("run the tests", policy=DEFAULT_COVERAGE_POLICY)
        self.assertEqual(out["route"], "local")
        self.assertIsNone(out["escalate_to"])
        self.assertEqual(out["label"], "EXECUTE")
        self.assertTrue(out["reason"].startswith("high_conf_"))

    def test_high_conf_delegate_routes_local(self):
        fake = {
            "ok": True,
            "label": "DELEGATE",
            "p": 0.55,
            "margin": 0.12,
            "second": "EXECUTE",
            "source": "next_action_gpu_champion",
            "n_params": 6912,
        }
        with mock.patch("evolution_lab.next_action_gpu.predict_next_action", return_value=fake):
            out = decide_next_action("spawn a scout", policy=DEFAULT_COVERAGE_POLICY)
        self.assertEqual(out["route"], "local")
        self.assertEqual(out["label"], "DELEGATE")

    def test_ambiguous_execute_escalates_to_jev(self):
        fake = {
            "ok": True,
            "label": "EXECUTE",
            "p": 0.42,
            "margin": 0.05,
            "second": "DELEGATE",
            "source": "next_action_gpu_champion",
            "n_params": 6912,
        }
        with mock.patch("evolution_lab.next_action_gpu.predict_next_action", return_value=fake):
            out = decide_next_action("maybe run something", policy=DEFAULT_COVERAGE_POLICY)
        self.assertEqual(out["route"], "escalate")
        self.assertEqual(out["escalate_to"], "jev")
        self.assertEqual(out["fallback"], "openjev")
        self.assertTrue(out["reason"].startswith("low_"))

    def test_weak_family_always_escalates(self):
        fake = {
            "ok": True,
            "label": "EDIT",
            "p": 0.95,
            "margin": 0.90,
            "second": "EXECUTE",
            "source": "next_action_gpu_champion",
            "n_params": 6912,
        }
        with mock.patch("evolution_lab.next_action_gpu.predict_next_action", return_value=fake):
            out = decide_next_action("rewrite this file", policy=DEFAULT_COVERAGE_POLICY)
        self.assertEqual(out["route"], "escalate")
        self.assertEqual(out["escalate_to"], "jev")
        self.assertIn("family_not_local", out["reason"])


if __name__ == "__main__":
    unittest.main()
