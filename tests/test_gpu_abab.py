"""Read-only tests for the GPU ABAB recipe grid and keep gate."""

from __future__ import annotations

import unittest

from evolution_lab.gpu_abab import MARGIN, _keep, _plan
from evolution_lab.schema import DataRecipe


class GPUABABTests(unittest.TestCase):
    def test_plan_grid_cycles(self):
        expected = (
            DataRecipe(n_train=2048, gold_only=False, source="next_action"),
            DataRecipe(n_train=2048, gold_only=True, source="next_action"),
            DataRecipe(n_train=8192, gold_only=False, source="next_action"),
            DataRecipe(n_train=0, gold_only=True, source="next_action"),
            DataRecipe(n_train=16384, gold_only=False, source="next_action"),
            DataRecipe(n_train=0, gold_only=False, source="next_action"),
            DataRecipe(n_train=8192, gold_only=True, source="next_action"),
            DataRecipe(n_train=4096, gold_only=False, source="next_action", confirm_frac=0.1),
        )

        grid = tuple(_plan(wave) for wave in range(8))
        self.assertEqual(grid, expected)
        self.assertEqual(_plan(0), expected[0])
        self.assertEqual(_plan(3).n_train, 0)
        self.assertTrue(_plan(3).gold_only)
        self.assertEqual(_plan(7).confirm_frac, 0.1)
        self.assertEqual(_plan(8), _plan(0))
        self.assertTrue(all(recipe.source == "next_action" for recipe in grid))
        knobs = {
            (recipe.n_train, recipe.gold_only, recipe.source, recipe.confirm_frac)
            for recipe in grid
        }
        self.assertEqual(len(knobs), 8)

    def test_keep_requires_margin_over_both(self):
        self.assertEqual(MARGIN, 0.01)
        self.assertTrue(
            _keep(
                {
                    "gpu_best_confirm_acc": 0.70,
                    "ridge_confirm_acc": 0.60,
                    "majority_confirm_acc": 0.50,
                }
            )
        )
        self.assertFalse(
            _keep(
                {
                    "gpu_best_confirm_acc": 0.61,
                    "ridge_confirm_acc": 0.60,
                    "majority_confirm_acc": 0.50,
                }
            )
        )
        self.assertFalse(
            _keep(
                {
                    "gpu_best_confirm_acc": 0.70,
                    "ridge_confirm_acc": 0.60,
                    "majority_confirm_acc": 0.695,
                }
            )
        )
        self.assertTrue(
            _keep(
                {
                    "gpu_best_confirm_acc": "0.7",
                    "ridge_confirm_acc": "0.6",
                    "majority_confirm_acc": "0.5",
                }
            )
        )

    def test_keep_missing_key_raises(self):
        with self.assertRaises(KeyError) as context:
            _keep(
                {
                    "gpu_best_confirm_acc": 0.70,
                    "majority_confirm_acc": 0.50,
                }
            )
        self.assertEqual(context.exception.args, ("ridge_confirm_acc",))
