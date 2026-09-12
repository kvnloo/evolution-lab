from __future__ import annotations

import unittest
import numpy as np

from evolution_lab.engine import seed_genomes
from evolution_lab.mutate import mutate
from evolution_lab.schema import GenomeError


class MutateTests(unittest.TestCase):
    def test_child_validates(self):
        rng = np.random.default_rng(1)
        parent = seed_genomes()[2]
        child = mutate(parent, rng, generation=1)
        child.validate()
        self.assertEqual(child.parents, (parent.id,))
        self.assertNotEqual(child.id, parent.id)
