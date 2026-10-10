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

    def run_cli(self, *, pin=None, raw=None, lineage=None):
        self.packet.write_text(json.dumps(self.receipt))
        if raw is not None:
            self.packet.write_text(raw)
        pin_args = ["--receipt-sha256", pin or self.digest(self.packet)]
        lineage_args = []
        if lineage is not None:
            path = self.root / "lineage.json"
            path.write_text(json.dumps(lineage))
            lineage_args = ["--lineage", str(path), "--lineage-sha256", self.digest(path)]
        return subprocess.run([sys.executable, str(CLI), "--receipt", str(self.packet), *pin_args,
                               *lineage_args,
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

    def assert_refused(self, result):
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "refused")
        self.assertNotIn(str(self.root), result.stdout + result.stderr)
        self.assertNotIn("synthetic draft", result.stdout + result.stderr)

    def test_failed_generator_cannot_reuse_existing_drafts(self):
        self.receipt["generator_exit"] = 1
        self.assert_refused(self.run_cli())

    def test_success_receipt_cannot_be_rebound_under_original_pin(self):
        self.run_cli()
        pin = self.digest(self.packet)
        self.receipt["observed_at"] = "2099-01-01T00:00:00Z"
        self.assert_refused(self.run_cli(pin=pin))

    def test_duplicate_success_exit_key_is_not_accepted(self):
        raw = json.dumps(self.receipt)
        raw = raw[:-1] + ', "generator_exit": 1, "generator_exit": 0}'
        self.assert_refused(self.run_cli(raw=raw))

    def test_missing_candidate_is_refused(self):
        (self.candidates / "0/SKILL.md").unlink()
        self.assert_refused(self.run_cli())

    def test_changed_candidate_is_refused(self):
        (self.candidates / "0/SKILL.md").write_text("changed synthetic draft")
        self.assert_refused(self.run_cli())

    def test_changed_corpus_is_refused(self):
        self.corpus.write_text('{"changed": true}\n')
        self.assert_refused(self.run_cli())

    def test_changed_generator_ref_is_refused(self):
        self.receipt["generator_ref"] = "a" * 40
        self.assert_refused(self.run_cli())

    def test_dirty_generator_is_refused(self):
        (self.generator / "untracked.py").write_text("synthetic")
        self.assert_refused(self.run_cli())

    def test_partial_or_invalid_declarations_are_refused(self):
        for key, value in (("generator_exit", False), ("expected_skill_count", True),
                           ("generated_skill_files", 1), ("candidate_sha256", []),
                           ("candidate_sha256", ["z" * 64] * 2),
                           ("generator_ref", "a" * 7), ("corpus_sha256", "b" * 63),
                           ("installed", True), ("efficacy_claim", True),
                           ("independent_qualification", "verified_success")):
            with self.subTest(key=key, value=value):
                original = self.receipt[key]
                self.receipt[key] = value
                self.assert_refused(self.run_cli())
                self.receipt[key] = original
        del self.receipt["generator_exit"]
        self.assert_refused(self.run_cli())

    def test_symlink_candidate_is_refused(self):
        path = self.candidates / "0/SKILL.md"
        saved = self.root / "saved.md"
        path.rename(saved)
        path.symlink_to(saved)
        self.assert_refused(self.run_cli())

    def test_unaccounted_output_is_refused(self):
        (self.candidates / "0/extra.txt").write_text("synthetic extra output")
        self.assert_refused(self.run_cli())

    def test_malformed_receipt_is_sanitized(self):
        for raw in ("{private malformed", "null", "[]"):
            with self.subTest(raw=raw):
                self.assert_refused(self.run_cli(raw=raw))

    def test_nonstandard_json_constants_are_refused(self):
        for value in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(value=value):
                raw = json.dumps(self.receipt)[:-1] + ', "unknown": ' + value + '}'
                self.assert_refused(self.run_cli(raw=raw))

    def branch_lineage(self) -> dict:
        records = [dict(session_id="synthetic-session", branch_id=str(index),
                        messages=[{"role": "user", "content": "synthetic prefix"},
                                  {"role": "assistant", "content": "shared synthetic reply"},
                                  {"role": "user", "content": f"synthetic branch {index}"}])
                   for index in range(2)]
        lines = [json.dumps(r) for r in records]
        self.corpus.write_text("\n".join(lines) + "\n")
        self.receipt["corpus_sha256"] = self.digest(self.corpus)
        return dict(corpus_sha256=self.receipt["corpus_sha256"],
                    grouping_basis="source_reported_conservative_work_item",
                    records=[dict(record_sha256=hashlib.sha256(line.encode()).hexdigest(),
                                  session_sha256=hashlib.sha256(b"synthetic-session").hexdigest(),
                                  work_item_sha256="a" * 64) for line in lines])

    def test_all_branch_prefix_is_one_exposure_group(self):
        result = self.run_cli(lineage=self.branch_lineage())
        self.assertEqual(result.returncode, 0, result.stderr)
        packet = json.loads(result.stdout)
        self.assertEqual(packet["source_records"], 2)
        self.assertEqual(packet["source_sessions"], 1)
        self.assertEqual(packet["source_work_items"], 1)
        self.assertEqual(packet["shared_prefix_pairs"], 1)
        self.assertEqual(packet["shared_prefix_messages"], 2)
        self.assertEqual(packet["source_group_status"], "producer_asserted_not_independently_verified")

    def test_one_native_session_cannot_be_split_into_work_items(self):
        lineage = self.branch_lineage()
        lineage["records"][1]["work_item_sha256"] = "b" * 64
        self.assert_refused(self.run_cli(lineage=lineage))

    def test_missing_source_mapping_is_refused(self):
        lineage = self.branch_lineage()
        lineage["records"].pop()
        self.assert_refused(self.run_cli(lineage=lineage))

    def test_duplicate_ambiguous_mapping_is_refused(self):
        lineage = self.branch_lineage()
        lineage["records"].append(dict(lineage["records"][0], work_item_sha256="b" * 64))
        self.assert_refused(self.run_cli(lineage=lineage))

    def test_source_mapping_must_match_native_session_identity(self):
        lineage = self.branch_lineage()
        lineage["records"][0]["session_sha256"] = "b" * 64
        self.assert_refused(self.run_cli(lineage=lineage))

    def test_lineage_corpus_pin_cannot_drift(self):
        lineage = self.branch_lineage()
        lineage["corpus_sha256"] = "b" * 64
        self.assert_refused(self.run_cli(lineage=lineage))

    def test_nonstandard_lineage_json_is_refused(self):
        lineage = self.branch_lineage()
        lineage["unknown"] = float("nan")
        self.assert_refused(self.run_cli(lineage=lineage))

    def test_nonstandard_source_json_is_refused(self):
        lineage = self.branch_lineage()
        lines = self.corpus.read_text().splitlines()
        record = json.loads(lines[0])
        record["messages"][0]["unknown"] = float("nan")
        lines[0] = json.dumps(record)
        self.corpus.write_text("\n".join(lines) + "\n")
        self.receipt["corpus_sha256"] = self.digest(self.corpus)
        lineage["corpus_sha256"] = self.receipt["corpus_sha256"]
        lineage["records"][0]["record_sha256"] = hashlib.sha256(lines[0].encode()).hexdigest()
        self.assert_refused(self.run_cli(lineage=lineage))

    def test_lineage_parser_checks_the_actual_bytes_it_reads(self):
        from scripts.conversation_skills_receipt import check_lineage

        lineage = self.branch_lineage()
        frozen_hash = self.receipt["corpus_sha256"]
        assert isinstance(frozen_hash, str)
        # Simulate drift after the initial artifact hash check, before lineage read.
        with self.corpus.open("ab") as stream:
            stream.write(b"\n")
        with self.assertRaises(ValueError):
            check_lineage(lineage, self.corpus, frozen_hash)


if __name__ == "__main__":
    unittest.main()
