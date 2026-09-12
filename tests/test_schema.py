from __future__ import annotations

import unittest

from evolution_lab.schema import (
    ACTIONS,
    Curriculum,
    ExperimentGenome,
    GenomeError,
    PRODUCT_TARGET,
    SECRET_FIELD_NAMES,
    genome_from_dict,
    observation_is_sanitized,
    options_carry_secrets,
)


class SchemaTests(unittest.TestCase):
    def test_secrets_fail_closed(self):
        g = ExperimentGenome(
            id="x",
            lineage="x",
            hypothesis="leak",
            curriculum=Curriculum(include_secrets=True),
        )
        with self.assertRaises(GenomeError):
            g.validate()

    def test_observation_sanitized(self):
        self.assertTrue(observation_is_sanitized({"sandbox_alive": 1}))
        for name in SECRET_FIELD_NAMES:
            with self.subTest(name=name):
                self.assertFalse(observation_is_sanitized({name: "x"}))
        self.assertTrue(options_carry_secrets({"include_secrets": True}))
        self.assertFalse(options_carry_secrets({"ood": True}))
        for name in SECRET_FIELD_NAMES:
            with self.subTest(opt=name):
                self.assertTrue(options_carry_secrets({name: "x"}))

    def test_actions_tuple_locked(self):
        self.assertEqual(
            ACTIONS,
            ("retry", "restart_sandbox", "escalate", "noop", "page_human"),
        )

    def test_unknown_backend(self):
        data = ExperimentGenome(id="a", lineage="a", hypothesis="h").to_dict()
        data["backend"] = "not_real"
        with self.assertRaises(GenomeError):
            genome_from_dict(data)


class ProductTargetTests(unittest.TestCase):
    def test_product_target_declared(self):
        self.assertEqual(PRODUCT_TARGET.success_vs_teacher, 0.95)
        self.assertEqual(PRODUCT_TARGET.cost_vs_teacher, 0.50)
        self.assertEqual(PRODUCT_TARGET.extra_violations, 0)
        self.assertIsNone(PRODUCT_TARGET.joules)
        from evolution_lab.targets import PRODUCT_TARGET as from_targets
        from evolution_lab import PRODUCT_TARGET as from_pkg

        self.assertIs(PRODUCT_TARGET, from_targets)
        self.assertIs(PRODUCT_TARGET, from_pkg)


if __name__ == "__main__":
    unittest.main()
