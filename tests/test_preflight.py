from __future__ import annotations

import unittest
from unittest import mock

from evolution_lab.preflight import (
    capability_for_family,
    counterfactual_tokens,
    preflight,
    receipt_from_preflight,
    residual_work_requirement,
)


class CapabilityMap(unittest.TestCase):
    def test_delegate_maps(self):
        self.assertEqual(capability_for_family("DELEGATE"), "coding.delegate")

    def test_hint_wins(self):
        self.assertEqual(
            capability_for_family("DELEGATE", hint="blender.scene_reasoning"),
            "blender.scene_reasoning",
        )


class Residual(unittest.TestCase):
    def test_blender_sets_vision(self):
        req = residual_work_requirement("blender.scene_reasoning", estimated_input_tokens=100, estimated_output_tokens=20)
        self.assertTrue(req["vision"])
        self.assertEqual(req["capability_id"], "blender.scene_reasoning")


class Preflight(unittest.TestCase):
    def test_local_route_packet(self):
        fake = {
            "ok": True,
            "route": "local",
            "reason": "high_conf_delegate",
            "escalate_to": None,
            "fallback": "openjev",
            "label": "DELEGATE",
            "p": 0.99,
            "margin": 0.5,
            "source": "test",
        }
        with mock.patch("evolution_lab.preflight.decide_next_action", return_value=fake):
            out = preflight("please spawn a scout")
        self.assertEqual(out["route"], "local")
        self.assertEqual(out["capability_id"], "coding.delegate")
        self.assertIsNone(out["work_requirement"])
        self.assertGreater(out["estimated_frontier_tokens_avoided"], 0)
        self.assertEqual(out["schema"], "z0int.preflight.v1")

    def test_model_route_packet(self):
        fake = {
            "ok": True,
            "route": "escalate",
            "reason": "family_not_local:EDIT",
            "escalate_to": "jev",
            "fallback": "openjev",
            "label": "EDIT",
            "p": 0.4,
            "margin": 0.1,
            "source": "test",
        }
        with mock.patch("evolution_lab.preflight.decide_next_action", return_value=fake):
            out = preflight("edit the file")
        self.assertEqual(out["route"], "model")
        self.assertEqual(out["capability_id"], "coding.edit")
        self.assertIsNotNone(out["work_requirement"])
        self.assertEqual(out["work_requirement"]["capability_id"], "coding.edit")
        self.assertEqual(out["kerdoios"]["action"], "plan_residual")

    def test_receipt_local(self):
        row = {
            "route": "local",
            "capability_id": "coding.delegate",
            "latency_ms": 1.2,
            "estimated_frontier_tokens_avoided": 2050,
            "counterfactual": {"estimated_input_tokens": 1800, "estimated_output_tokens": 250},
        }
        r = receipt_from_preflight(row)
        self.assertEqual(r["provider"], "z0int")
        self.assertEqual(r["input_tokens"], 0)
        self.assertEqual(r["capability_id"], "coding.delegate")


class Counterfactual(unittest.TestCase):
    def test_override_baseline(self):
        cf = counterfactual_tokens("coding.delegate", baseline_input=50, baseline_output=10)
        self.assertEqual(cf["estimated_input"], 50)
        self.assertEqual(cf["estimated_output"], 10)


if __name__ == "__main__":
    unittest.main()
