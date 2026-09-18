from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evolution_lab.export_z0int import export_candidate


class ExportZ0intTests(unittest.TestCase):
    def test_export_always_candidate_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            weights = Path(tmp) / "weights.npz"
            np.savez(weights, w=np.array([1.0, 2.0]))
            out = Path(tmp) / "artifact"
            result = export_candidate(candidate=weights, output=out)
            self.assertTrue(result["ok"])
            man = json.loads((out / "manifest.json").read_text())
            self.assertEqual(man["schema"], "z0int.candidate_artifact.v1")
            self.assertEqual(man["promotion"]["status"], "candidate")
            self.assertTrue(man["artifact_id"].startswith("sha256:"))
            self.assertEqual(man["producer"]["repo"], "kvnloo/evolution-lab")
            digest = hashlib.sha256((out / "weights.npz").read_bytes()).hexdigest()
            self.assertEqual(man["files"][0]["sha256"], digest)

    def test_export_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            weights = Path(tmp) / "w.npz"
            np.savez(weights, w=np.ones(3))
            out = Path(tmp) / "out"
            out.mkdir()
            with self.assertRaises(FileExistsError):
                export_candidate(candidate=weights, output=out)


if __name__ == "__main__":
    unittest.main()
