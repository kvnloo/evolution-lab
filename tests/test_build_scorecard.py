"""Contract tests for the committed small-network results scorecard."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_scorecard  # noqa: E402


class ResultsScorecardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.results = ROOT / "results"

    def section(self, document: str, filename: str) -> str:
        heading = "## " + chr(96) + "results/" + filename + chr(96)
        return document.split(heading, 1)[1].split("\n## ", 1)[0]

    def test_real_shadow_traffic_ridge_wins_and_mushroom_loses(self) -> None:
        section = self.section(
            build_scorecard.render_scorecard(self.results), "shadow-real-traffic.json"
        )
        self.assertIn(
            "| confirm | mushroom_champion | 0.143 | 0.514 | -0.371 | LOSES_TO_FLOOR |",
            section,
        )
        self.assertIn(
            "| confirm | ridge | 0.643 | 0.514 | +0.129 | BEATS_FLOOR |",
            section,
        )

    def test_recovery_confirm_rule_and_mushroom_beat_majority(self) -> None:
        section = self.section(
            build_scorecard.render_scorecard(self.results), "recovery-compare.json"
        )
        self.assertIn(
            "| confirm | rule | 0.977 | 0.318 | +0.659 | BEATS_FLOOR |",
            section,
        )
        self.assertIn(
            "| confirm | mushroom | 0.839 | 0.318 | +0.521 | BEATS_FLOOR |",
            section,
        )
        self.assertIn(
            "| confirm | majority | 0.318 | 0.318 | +0.000 | TIES_FLOOR |",
            section,
        )
        self.assertIn(
            "| ood | mushroom | 1.000 | 1.000 | +0.000 | TIES_FLOOR |",
            section,
        )

    def test_all_source_caveats_are_preserved_verbatim_in_own_section(self) -> None:
        document = build_scorecard.render_scorecard(self.results)
        for filename in (
            "recovery-compare.json",
            "recovery-llm-compare.json",
            "shadow-real-traffic.json",
        ):
            source = json.loads((self.results / filename).read_text(encoding="utf-8"))
            section = self.section(document, filename)
            self.assertIn(f"Evidence class: {source['evidence_class']}", section)
            self.assertIn("Promotion eligible: false", section)
            for field in build_scorecard.CAVEAT_FIELDS:
                if field in source:
                    self.assertIn("**" + field + ":** " + source[field], section)

    def test_llm_results_do_not_fabricate_cross_encoding_floor(self) -> None:
        section = self.section(
            build_scorecard.render_scorecard(self.results), "recovery-llm-compare.json"
        )
        self.assertIn("NO_COMPARABLE_FLOOR", section)
        self.assertIn("not provided by this source", section)
        self.assertIn("| confirm | hammer2.1_3b | 0.281 | — | — |", section)
        self.assertIn("| confirm | nanojev | 0.000 | — | — |", section)
        self.assertIn("| 48 | 0 | 46.700 |", section)

    def test_path_redaction_and_unknown_schema_listing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            (results / "synthetic.json").write_text(
                json.dumps({
                    "schema": build_scorecard.SCHEMA_SHADOW,
                    "evidence_class": "synthetic",
                    "promotion_eligible": False,
                    "dataset": "/abs/path/x.jsonl",
                    "n_confirm": 3,
                    "majority_floor": 0.25,
                    "results": {"majority": 0.25, "ridge": 0.75},
                    "confidence_note": "synthetic confidence only",
                }),
                encoding="utf-8",
            )
            (results / "unknown.json").write_text(
                json.dumps({"schema": "unrecognized.schema", "secret": "/private/file"}),
                encoding="utf-8",
            )
            document = build_scorecard.render_scorecard(results)
            self.assertIn("x.jsonl (local path redacted)", document)
            self.assertNotIn("/abs/path/x.jsonl", document)
            self.assertNotIn("/private/file", document)
            self.assertIn("## Not rendered", document)
            self.assertIn("unknown.json", document)
            self.assertIn("unrecognized.schema", document)
            self.assertIn("synthetic confidence only", document)

    def test_tilde_path_and_nested_values_redacted(self) -> None:
        self.assertEqual(
            build_scorecard.safe("~/secrets/log.jsonl"),
            "log.jsonl (local path redacted)",
        )
        self.assertEqual(
            build_scorecard.safe({"files": ["/opt/private/config", "relative/file"]}),
            '{"files": ["config (local path redacted)", "relative/file"]}',
        )

    def test_tie_threshold_uses_unrounded_delta(self) -> None:
        self.assertEqual(build_scorecard.verdict(0.00999), "TIES_FLOOR")
        self.assertEqual(build_scorecard.verdict(-0.00999), "TIES_FLOOR")
        self.assertEqual(build_scorecard.verdict(0.01), "BEATS_FLOOR")
        self.assertEqual(build_scorecard.verdict(-0.01), "LOSES_TO_FLOOR")

    def test_summary_has_one_row_per_file_and_arm(self) -> None:
        doc = build_scorecard.render_scorecard(self.results)
        summary = doc.split("## Summary\n\n", 1)[1].split("\n## ", 1)[0]
        data_rows = [line for line in summary.splitlines() if line.startswith("| ")][2:]
        self.assertEqual(len(data_rows), 8)  # 3 recovery + 2 LLM + 3 shadow
        self.assertEqual(len({(row.split("|")[1].strip(), row.split("|")[2].strip())
                              for row in data_rows}), 8)
        self.assertLess(summary.index("BEATS_FLOOR"), summary.index("LOSES_TO_FLOOR"))

    def test_deterministic_and_optional_git_sha(self) -> None:
        first = build_scorecard.render_scorecard(self.results)
        second = build_scorecard.render_scorecard(self.results)
        self.assertEqual(first.encode("utf-8"), second.encode("utf-8"))
        self.assertNotIn("Source git SHA:", first)
        with_sha = build_scorecard.render_scorecard(self.results, source_sha="abc123")
        self.assertIn("Source git SHA: " + chr(96) + "abc123" + chr(96), with_sha)

    def test_check_on_committed_file_and_stale_file(self) -> None:
        self.assertEqual(build_scorecard.main([
            "--results", str(self.results),
            "--out", str(ROOT / "docs" / "SCORECARD.md"),
            "--check",
        ]), 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scorecard.md"
            args = ["--results", str(self.results), "--out", str(path)]
            self.assertEqual(build_scorecard.main(args), 0)
            self.assertEqual(build_scorecard.main(args + ["--check"]), 0)
            path.write_text("stale", encoding="utf-8")
            self.assertEqual(build_scorecard.main(args + ["--check"]), 1)
            self.assertEqual(build_scorecard.main([
                "--results", str(self.results),
                "--out", str(Path(tmp) / "absent.md"), "--check",
            ]), 1)


if __name__ == "__main__":
    unittest.main()
