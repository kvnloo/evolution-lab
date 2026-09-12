from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evolution_lab.archive import Archive
from evolution_lab.elites import MapElites
from evolution_lab.engine import run_one, seed_genomes, summarize
from evolution_lab.schema import ExperimentGenome, GenomeError, Architecture


class EngineTests(unittest.TestCase):
    def test_p0_table_runs(self):
        tmp = Path(tempfile.mkdtemp())
        archive = Archive(tmp / "archive.jsonl")
        elites = MapElites()
        prior = []
        for g in seed_genomes():
            rec = run_one(g, archive, elites, level=0, prior=prior)
            prior.append(rec)
            self.assertIn(rec["status"], {"ok", "killed"})
            self.assertIsNone((rec.get("metrics") or {}).get("joules_per_success"))
        stats = summarize(archive.read())
        self.assertGreaterEqual(stats["ok"], 4)
        self.assertGreater(stats["hypervolume_2d"], 0.0)
        teacher = next(r for r in archive.read() if r["experiment_id"] == "teacher-rule-000")
        self.assertGreaterEqual(teacher["metrics"]["success_rate"], 0.99)

    def test_direct_input_loses_delayed_cue(self):
        tmp = Path(tempfile.mkdtemp())
        archive = Archive(tmp / "archive.jsonl")
        elites = MapElites()
        prior = []
        genomes = {g.id: g for g in seed_genomes()}
        teacher = run_one(genomes["teacher-rule-000"], archive, elites, level=1, prior=prior)
        prior.append(teacher)
        direct = run_one(genomes["direct-input-000"], archive, elites, level=1, prior=prior)
        self.assertGreaterEqual(teacher["metrics"]["success_rate"], 0.99)
        self.assertLess(direct["metrics"]["success_rate"], teacher["metrics"]["success_rate"])

    def test_tinker_refuses(self):
        tmp = Path(tempfile.mkdtemp())
        g = ExperimentGenome(
            id="tinker-x",
            lineage="tinker",
            hypothesis="must not fake SFT",
            backend="tinker_sft",
            architecture=Architecture(family="mlp"),
        )
        rec = run_one(g, Archive(tmp / "a.jsonl"), MapElites(), level=0, prior=[])
        self.assertEqual(rec["status"], "failed")
        self.assertIn("not wired", rec["error"])

    def test_secrets_never_reach_fit(self):
        from evolution_lab.schema import Curriculum

        g = ExperimentGenome(
            id="leak",
            lineage="leak",
            hypothesis="no",
            curriculum=Curriculum(include_secrets=True),
        )
        with self.assertRaises(GenomeError):
            run_one(g, Archive(Path(tempfile.mkdtemp()) / "a.jsonl"), MapElites(), level=0, prior=[])
