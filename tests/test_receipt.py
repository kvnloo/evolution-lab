from __future__ import annotations

import re
import unittest
from io import StringIO
from unittest.mock import patch

from evolution_lab.cli import main
from evolution_lab.receipt import (
    MUTATION_NA,
    evidence_receipt,
    git_head_revision,
    receipt_yaml,
)


class ReceiptTests(unittest.TestCase):
    def test_receipt_has_verified_oss_loop_keys(self):
        rec = evidence_receipt(issue="2")
        for key in ("issue", "base_revision", "head_revision", "tests", "mutation", "limitations", "runtime_evidence", "ai_assistance"):
            self.assertIn(key, rec)
        self.assertEqual(rec["issue"], "2")
        self.assertEqual(rec["tests"]["red"].strip().split()[0], "python")
        self.assertIn("unittest", rec["tests"]["green"])
        self.assertIn("test_*.py", rec["tests"]["green"])
        self.assertIn("flygym", rec["tests"]["sabotage"])
        self.assertEqual(rec["mutation"], MUTATION_NA)
        self.assertEqual(rec["mutation"], "n/a")
        self.assertIn("joules_unknown", rec["limitations"])
        self.assertIn("pareto_point_unasserted", rec["limitations"])
        self.assertIsInstance(rec["limitations"], list)
        self.assertIn("gym-smoke", rec["runtime_evidence"][0])
        self.assertIn("unittest", rec["tests"]["green"])
        self.assertEqual(rec["ai_assistance"], "unattended")

    def test_does_not_invent_mutation_score(self):
        rec = evidence_receipt()
        self.assertFalse(re.match(r"^\s*\d+(\.\d+)?\s*$", str(rec["mutation"])))
        self.assertTrue(str(rec["mutation"]).lower().startswith("n/a"))

    def test_revisions_are_git_shas(self):
        rec = evidence_receipt(issue="2")
        sha = re.compile(r"^[0-9a-f]{7,40}$", re.I)
        self.assertRegex(rec["head_revision"], sha)
        self.assertRegex(rec["base_revision"], sha)
        self.assertEqual(rec["head_revision"], git_head_revision())

    def test_yaml_print(self):
        text = receipt_yaml(issue="2")
        self.assertIn("issue:", text)
        self.assertIn("base_revision:", text)
        self.assertIn("head_revision:", text)
        self.assertIn("red:", text)
        self.assertIn("green:", text)
        self.assertIn("mutation:", text)
        self.assertIn("n/a", text)
        self.assertIn("joules_unknown", text)
        self.assertIn("pareto_point_unasserted", text)

    def test_cli_receipt_prints_yaml(self):
        buf = StringIO()
        with patch("sys.stdout", buf):
            rc = main(["receipt", "--issue", "2"])
        self.assertEqual(rc, 0)
        text = buf.getvalue()
        self.assertIn("issue: 2", text)
        self.assertIn("tests:", text)
        self.assertIn("joules_unknown", text)
        self.assertIn("mutation:", text)


if __name__ == "__main__":
    unittest.main()
