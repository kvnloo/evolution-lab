from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.2")

from evolution_lab.abab_meta import choose_b_action, mutate_recipe, seed_fly_world
from evolution_lab.schema import DataRecipe, ExperimentGenome, genome_from_dict


def _g(**recipe):
    return genome_from_dict(
        {
            "id": "g",
            "lineage": "test",
            "hypothesis": "recipe",
            "architecture": {"family": "local_plasticity", "hidden": 96},
            "training": {"algorithm": "local_plasticity"},
            "recipe": recipe,
        }
    )


class DataRecipeTests(unittest.TestCase):
    def test_default_roundtrip(self):
        g = _g()
        self.assertEqual(g.recipe.source, "hermes_recovery")
        self.assertEqual(g.recipe.n_train, 64)
        again = genome_from_dict(g.to_dict())
        self.assertEqual(again.recipe.n_train, 64)

    def test_rejects_tiny_n_train(self):
        with self.assertRaises(Exception):
            _g(n_train=1)

    def test_abab_idle_mutates_recipe(self):
        w = seed_fly_world()
        w.wave = 6
        self.assertEqual(choose_b_action(w, idle=99, idle_limit=2), "mutate_recipe")
        r = mutate_recipe(w)
        self.assertGreaterEqual(r.n_train, 32)
        self.assertIn(r.source, {"hermes_recovery", "next_action"})

    def test_abab_odd_wave_sweeps(self):
        w = seed_fly_world()
        w.wave = 7
        self.assertEqual(choose_b_action(w, idle=99, idle_limit=2), "sweep")


class NextActionGpuSmoke(unittest.TestCase):
    def test_fixture_pop8(self):
        from evolution_lab.next_action_gpu import run_next_action_gpu

        rows = [
            {"text": "read the file please", "family": "READ_SEARCH", "gold": True},
            {"text": "edit this function", "family": "EDIT", "gold": True},
            {"text": "run pytest now", "family": "EXECUTE", "gold": True},
            {"text": "search the web", "family": "WEB", "gold": True},
            {"text": "spawn a subagent", "family": "DELEGATE", "gold": True},
            {"text": "check the tests", "family": "VERIFY", "gold": True},
            {"text": "answer the user", "family": "RESPOND", "gold": True},
            {"text": "not sure skip", "family": "ABSTAIN", "gold": True},
        ]
        # pad so n_train>=8 and confirm split nonempty
        rows = rows * 6
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ep.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            recipe = DataRecipe(n_train=32, gold_only=True, source="next_action", confirm_frac=0.25)
            report = run_next_action_gpu(recipe=recipe, n_pop=8, n_kc=16, k_winners=4, epochs=2, episodes=path)
        self.assertEqual(report["n_pop"], 8)
        self.assertGreaterEqual(report["gpu_best_confirm_acc"], 0.0)
        self.assertIn("ridge_confirm_acc", report)


if __name__ == "__main__":
    unittest.main()
