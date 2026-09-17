from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evolution_lab.engine import run_one, seed_jev_genomes
from evolution_lab.archive import Archive
from evolution_lab.elites import MapElites
from evolution_lab.jev_splits import lock_splits
from evolution_lab.openjev_runner import openjev_available
from evolution_lab.schema import GenomeError


class OpenJevBackendTests(unittest.TestCase):
    def test_jev_genomes_validate(self):
        for genome in seed_jev_genomes():
            genome.validate()

    def test_openjev_refuses_without_package(self):
        if openjev_available():
            self.skipTest("openjev installed in this environment")
        tmp = Path(tempfile.mkdtemp())
        lock_splits(tmp / "jev")
        genome = seed_jev_genomes()[0]
        rec = run_one(
            genome,
            Archive(tmp / "archive.jsonl"),
            MapElites(),
            level=0,
            prior=[],
            run_dir=tmp,
            jev_splits_dir=tmp / "jev",
        )
        self.assertEqual(rec["status"], "failed")
        self.assertIn("openjev-phase1", rec["error"])

    def test_openjev_tiny_runs_when_installed(self):
        if not openjev_available():
            self.skipTest("openjev not installed")
        tmp = Path(tempfile.mkdtemp())
        lock_splits(tmp / "jev")
        genome = seed_jev_genomes()[0]
        rec = run_one(
            genome,
            Archive(tmp / "archive.jsonl"),
            MapElites(),
            level=0,
            prior=[],
            run_dir=tmp,
            jev_splits_dir=tmp / "jev",
        )
        self.assertEqual(rec["status"], "ok")
        self.assertGreater(rec["metrics"]["success_rate"], 0.0)
        self.assertIn("jev_checkpoint", rec["metrics"])
    def test_mlp_cannot_use_openjev_backend(self):
        from evolution_lab.schema import Architecture, Curriculum, ExperimentGenome

        with self.assertRaises(GenomeError):
            ExperimentGenome(
                id="bad",
                lineage="bad",
                hypothesis="bad",
                backend="openjev",
                architecture=Architecture(family="mlp"),
                curriculum=Curriculum(task="jev_synthetic"),
            ).validate()
