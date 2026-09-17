from __future__ import annotations

import unittest

from evolution_lab.schema import ACTIONS, Architecture, Curriculum, ExperimentGenome, FAMILIES
from evolution_lab.models import fit_student
from evolution_lab.task import build_task


class LocalPlasticityTests(unittest.TestCase):
    def test_family_is_registered(self):
        self.assertIn("local_plasticity", FAMILIES)

    def test_predicts_valid_actions_on_delayed_cue(self):
        g = ExperimentGenome(
            id="mb-test",
            lineage="mushroom-body",
            hypothesis="KC→MBON only",
            architecture=Architecture(family="local_plasticity", hidden=48, history=8),
            curriculum=Curriculum(delayed_cue=True),
        )
        data = build_task(g, n_train=64, n_val=16, n_confirm=16, n_ood=8)
        student = fit_student(g, data.train)
        pred = student.predict_fn(data.train)
        self.assertEqual(pred.shape[0], len(data.train))
        self.assertTrue(set(pred.tolist()).issubset(set(range(len(ACTIONS)))))
        acc = float((pred == [ep.labels[-1] for ep in data.train]).mean())
        self.assertGreater(acc, 0.35)
        self.assertEqual(student.family, "local_plasticity")
        self.assertGreater(student.n_params, 0)
        self.assertIn("W_kc_mbon", student.extras)
        self.assertIn("W_pn_kc", student.extras)


if __name__ == "__main__":
    unittest.main()
