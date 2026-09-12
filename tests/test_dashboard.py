from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evolution_lab.dashboard import render_dashboard, write_dashboard
from evolution_lab.engine import seed_genomes, run_one
from evolution_lab.archive import Archive
from evolution_lab.elites import MapElites


class DashboardTests(unittest.TestCase):
    def test_empty_archive_is_honest(self):
        html = render_dashboard([])
        self.assertIn("no ok experiments yet", html)
        self.assertIn("Evolution Lab", html)

    def test_writes_file(self):
        tmp = Path(tempfile.mkdtemp())
        archive = Archive(tmp / "archive.jsonl")
        elites = MapElites()
        prior = []
        rec = run_one(seed_genomes()[0], archive, elites, level=0, prior=prior)
        dest = write_dashboard([rec], tmp / "dashboard.html")
        self.assertTrue(dest.exists())
        self.assertIn("teacher-rule-000", dest.read_text())
