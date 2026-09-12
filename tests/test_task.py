from __future__ import annotations

import unittest
import numpy as np

from evolution_lab.schema import ACTIONS, ExperimentGenome
from evolution_lab.task import build_task, teacher_action


class TaskTests(unittest.TestCase):
    def test_policy_hit_pages_human(self):
        frame = np.zeros(10 + 5)
        frame[7] = 1.0  # policy_hit
        self.assertEqual(ACTIONS[teacher_action(frame)], "page_human")

    def test_dead_sandbox_restarts(self):
        frame = np.zeros(10 + 5)
        frame[0] = 0.0
        self.assertEqual(ACTIONS[teacher_action(frame)], "restart_sandbox")

    def test_delayed_cue_label_on_last_step(self):
        from evolution_lab.schema import Architecture, Curriculum

        g = ExperimentGenome(
            id="t",
            lineage="t",
            hypothesis="cue",
            architecture=Architecture(history=6),
            curriculum=Curriculum(delayed_cue=True),
        )
        data = build_task(g, n_train=40, n_val=8, n_confirm=8, n_ood=8)
        cued = [ep for ep in data.train if ep.frames[0, 9] > 0.5]
        self.assertTrue(cued)
        for ep in cued:
            self.assertEqual(ACTIONS[int(ep.labels[-1])], "escalate")
            self.assertLess(ep.frames[-1, 9], 0.5)
