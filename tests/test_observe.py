from __future__ import annotations

import unittest

from evolution_lab.advise import advise, clear_student_cache
from evolution_lab.observe import action_guidance, observe_event
from evolution_lab.recovery_cli import handle_request


class ObserveTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_student_cache()

    def test_rate_limit_is_transient(self) -> None:
        observed = observe_event(
            {"kind": "tool_error", "tool": "bash", "message": "HTTP 429 rate limit"},
            harness="omp",
        )
        self.assertEqual(observed["fields"]["transient"], 1.0)

    def test_hub_dead_restarts(self) -> None:
        observed = observe_event({"kind": "hub_dead", "message": "hub process exited"})
        out = advise(observed["fields"], family="local_plasticity")
        self.assertEqual(out["action"], "restart_sandbox")

    def test_plan_from_event(self) -> None:
        payload = handle_request(
            {
                "action": "plan",
                "harness": "omp",
                "event": {
                    "kind": "tool_error",
                    "tool": "eval",
                    "message": "eval kernel crashed",
                },
            }
        )
        self.assertIn(payload["action"], {"restart_sandbox", "escalate", "retry"})
        self.assertIn("guidance", payload)

    def test_action_guidance_known(self) -> None:
        guide = action_guidance("page_human")
        self.assertEqual(guide["action"], "page_human")
        self.assertIn("human", guide["guidance"].lower())


if __name__ == "__main__":
    unittest.main()
