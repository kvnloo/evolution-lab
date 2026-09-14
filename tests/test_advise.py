from __future__ import annotations

import unittest

from evolution_lab.advise import advise, fields_to_frame
from evolution_lab.schema import ACTIONS, GenomeError


class AdviseTests(unittest.TestCase):
    def test_dead_sandbox_restarts(self):
        out = advise({"sandbox_alive": 0.0, "retry": 0, "budget": 1.0})
        self.assertEqual(out["action"], "restart_sandbox")
        self.assertEqual(out["source"], "teacher_rule")

    def test_policy_hit_pages_human(self):
        out = advise({"sandbox_alive": 1.0, "policy_hit": 1.0})
        self.assertEqual(out["action"], "page_human")

    def test_secrets_fail_closed(self):
        with self.assertRaises(GenomeError):
            advise({"sandbox_alive": 1.0, "token": "nope"})

    def test_local_plasticity_returns_valid_action(self):
        out = advise(
            {"sandbox_alive": 1.0, "transient": 1.0, "retry": 0, "budget": 0.8},
            family="local_plasticity",
        )
        self.assertIn(out["action"], ACTIONS)
        self.assertEqual(out["family"], "local_plasticity")
        self.assertGreater(out["n_params"], 0)

    def test_frame_layout(self):
        fr = fields_to_frame({"sandbox_alive": 0.0, "last_action": "retry"})
        self.assertEqual(fr.shape[0], 15)
        self.assertEqual(fr[0], 0.0)


if __name__ == "__main__":
    unittest.main()
