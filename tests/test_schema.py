from __future__ import annotations

import unittest

from evolution_lab.schema import Curriculum, ExperimentGenome, GenomeError, genome_from_dict, observation_is_sanitized


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


if __name__ == "__main__":
    unittest.main()
