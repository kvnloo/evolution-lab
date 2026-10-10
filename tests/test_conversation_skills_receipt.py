from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/conversation_skills_receipt.py"


class ConversationSkillsReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.generator = self.root / "generator"
        self.generator.mkdir()
        env = dict(os.environ, GIT_AUTHOR_NAME="Fixture", GIT_AUTHOR_EMAIL="fixture@example.invalid",
                   GIT_COMMITTER_NAME="Fixture", GIT_COMMITTER_EMAIL="fixture@example.invalid")
        for args in (["init", "-q"], ["commit", "-q", "--allow-empty", "-m", "fixture"]):
            subprocess.run(["git", "-C", str(self.generator), *args], env=env, check=True,
                           capture_output=True)
        self.ref = subprocess.check_output(["git", "-C", str(self.generator), "rev-parse", "HEAD"], text=True).strip()
        self.corpus = self.root / "input.jsonl"
        self.corpus.write_text('{"synthetic": true}\n')
        self.candidates = self.root / "candidates"
        self.candidates.mkdir()
        for index in range(2):
            folder = self.candidates / str(index)
            folder.mkdir()
            (folder / "SKILL.md").write_text(f"synthetic draft {index}\n")
        self.receipt = dict(generator_exit=0, expected_skill_count=2, generated_skill_files=2,
                            candidate_sha256=sorted(self.digest(p) for p in self.candidates.glob("*/SKILL.md")),
                            generator_ref=self.ref, corpus_sha256=self.digest(self.corpus),
                            outputs="private_only", installed=False,
                            independent_qualification="not_run", efficacy_claim=False)
        self.packet = self.root / "receipt.json"

    @staticmethod
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def run_cli(self):
        self.packet.write_text(json.dumps(self.receipt))
        return subprocess.run([sys.executable, str(CLI), "--receipt", str(self.packet),
                               "--corpus", str(self.corpus), "--candidates", str(self.candidates),
                               "--generator", str(self.generator)], capture_output=True, text=True, cwd=ROOT)

    def test_freezes_exact_private_artifact_identity(self):
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        packet = json.loads(result.stdout)
        self.assertEqual(packet["candidate_sha256"], self.receipt["candidate_sha256"])
        self.assertEqual(packet["corpus_sha256"], self.receipt["corpus_sha256"])
        self.assertEqual(packet["generator_ref"], self.ref)
        self.assertEqual(packet["status"], "review_only_identity_consistent")
        self.assertFalse(packet["promotion_eligible"])
        self.assertNotIn(str(self.root), result.stdout + result.stderr)
        self.assertNotIn("synthetic draft", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
