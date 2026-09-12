from __future__ import annotations

import unittest

from evolution_lab.schema import (
    Curriculum,
    ExperimentGenome,
    GenomeError,
    PRODUCT_TARGET,
    genome_from_dict,
    observation_is_sanitized,
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
        self.assertFalse(observation_is_sanitized({"credential": "x"}))

    def test_unknown_backend(self):
        data = ExperimentGenome(id="a", lineage="a", hypothesis="h").to_dict()
        data["backend"] = "not_real"
        with self.assertRaises(GenomeError):
            genome_from_dict(data)


class ProductTargetTests(unittest.TestCase):
    def test_product_target_declared(self):
        self.assertEqual(PRODUCT_TARGET.success_vs_teacher, 0.95)
        self.assertEqual(PRODUCT_TARGET.extra_violations, 0)
        self.assertIsNone(PRODUCT_TARGET.joules)
        from evolution_lab.targets import PRODUCT_TARGET as from_targets
        from evolution_lab import PRODUCT_TARGET as from_pkg

        self.assertIs(PRODUCT_TARGET, from_targets)
        self.assertIs(PRODUCT_TARGET, from_pkg)


if __name__ == "__main__":
    unittest.main()
