from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from evolution_lab.research import recommend, write_report


INVARIANTS = ["independent_outcome", "bounded_authority", "all_attempts", "frozen_judge"]


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.root.joinpath("receipt.json").write_text('{"outcome": "unknown"}\n')
        digest = hashlib.sha256(self.root.joinpath("receipt.json").read_bytes()).hexdigest()
        self.item = {
            "id": "packet", "hypothesis": "packet-only evidence changes the answer",
            "owner": "z0intelligence", "issue": "https://example.test/1",
            "factor": "state_packet_memory", "goals": ["consumed_tokens", "intent_validation"],
            "intent": {"task_success": "Correct source-supported answer", "oracle": "independent checker",
                       "invariants": INVARIANTS},
            "decision": {"supports": "retain", "refutes": "kill", "ambiguous": "hold"},
            "comparator": "deterministic bundle from identical EvidenceRefs",
            "scope": {"code": {"hermes": "a" * 40}, "model": "fixture-only",
                      "routing": "native", "permissions": "read-only", "fixtures": ["frozen-1"]},
            "evidence": {"level": "I", "reconciled": True, "missing": "causal",
                         "strongest": "Component tests", "counterevidence": "Not consumed in Hermes",
                         "sources": [{"path": "receipt.json", "sha256": digest, "kind": "receipt"}]},
            "next": {"action": "experiment", "summary": "Run one paired comparison",
                     "acceptance": "same quality with less total work", "stop": "baseline ties or wins",
                     "reopen": "new supported task", "controls": ["missing source"],
                     "primary_outcome": "verified task success", "accounting": ["cold", "warm", "retries", "failures"],
                     "resources": ["cpu"], "mode": "offline"},
            "priority": {"impact": "high", "information": "high", "reuse": "medium", "cost": "low",
                         "rationale": "One result can remove an optional component."},
            "status": "open", "dependencies": [],
        }
        self.catalog = {"schema": "evolution_lab.research.v1", "snapshot_at": "2026-10-02T00:00:00Z",
                        "intent": {"objective": "Verified tasks with less time and consumed tokens",
                                   "goals": ["e2e_latency", "consumed_tokens", "intent_validation"],
                                   "invariants": INVARIANTS, "resources": ["cpu"]},
                        "candidates": [self.item]}

    def run_plan(self, catalog=None):
        return recommend(catalog or self.catalog, self.root, now="2026-10-02T01:00:00Z")

    def test_frozen_candidate_produces_bounded_recommendation(self):
        r = self.run_plan()
        self.assertEqual(r["selected"]["id"], "packet")
        self.assertEqual(r["selected"]["action"], "experiment")
        self.assertFalse(r["runtime_authority"])
        self.assertIn("verified task success", r["selected"]["packet"]["primary_outcome"])
        self.assertEqual(r["model_calls"], 0)

    def test_missing_receipt_is_reconciliation_not_never_executed(self):
        self.root.joinpath("receipt.json").unlink()
        r = self.run_plan()["eligible"][0]
        self.assertEqual(r["action"], "reconcile")
        self.assertIn("missing", r["evidence_checks"][0]["status"])
        self.assertNotIn("never executed", r["reason"])

    def test_changed_evidence_cannot_certify_old_plan(self):
        self.root.joinpath("receipt.json").write_text("changed")
        r = self.run_plan()["eligible"][0]
        self.assertEqual(r["action"], "reconcile")
        self.assertEqual(r["evidence_checks"][0]["status"], "digest_mismatch")

    def test_intent_constraints_precede_cost_or_priority(self):
        self.item["intent"]["invariants"] = ["all_attempts"]
        r = self.run_plan()
        self.assertIsNone(r["selected"])
        self.assertIn("intent", r["excluded"][0]["reason"])

    def test_intent_requires_independent_task_oracle(self):
        self.item["intent"]["oracle"] = ""
        self.assertIsNone(self.run_plan()["selected"])

    def test_active_owner_is_not_duplicated_and_expiry_needs_reconciliation(self):
        self.item["lease"] = {"owner": "worker-A", "until": "2026-10-02T02:00:00Z"}
        self.assertIsNone(self.run_plan()["selected"])
        self.item["lease"]["until"] = "2026-10-02T00:30:00Z"
        self.assertEqual(self.run_plan()["eligible"][0]["action"], "check_owner")

    def test_settled_negative_is_not_reopened_by_missing_file(self):
        self.item["status"] = "kill"
        self.root.joinpath("receipt.json").unlink()
        self.assertIsNone(self.run_plan()["selected"])

    def test_an_unavailable_runtime_does_not_block_evidence_reconciliation(self):
        self.item["next"]["resources"] = ["real-gpu"]
        self.assertIsNone(self.run_plan()["selected"])
        self.item["evidence"]["reconciled"] = False
        self.assertEqual(self.run_plan()["eligible"][0]["action"], "reconcile")

    def test_no_disposition_change_means_no_filler_experiment(self):
        self.item["decision"] = {"supports": "hold", "refutes": "hold", "ambiguous": "hold"}
        self.assertIsNone(self.run_plan()["selected"])

    def test_duplicate_candidate_does_not_gain_another_vote(self):
        duplicate = copy.deepcopy(self.item)
        duplicate["id"] = "alias"
        self.catalog["candidates"].append(duplicate)
        r = self.run_plan()
        self.assertEqual(len(r["eligible"]), 1)
        self.assertEqual(len(r["duplicates"]), 1)

    def test_pareto_comparison_does_not_fabricate_numeric_information_gain(self):
        other = copy.deepcopy(self.item)
        other.update(id="costly", hypothesis="a different hypothesis")
        other["priority"]["cost"] = "high"
        self.catalog["candidates"].append(other)
        r = self.run_plan()
        self.assertEqual([x["id"] for x in r["frontier"]], ["packet"])
        self.assertIn("costly", [x["id"] for x in r["dominated"]])
        self.assertNotIn("score", r["selected"])

    def test_missing_preregistration_is_repair_not_execution(self):
        self.item["scope"]["fixtures"] = []
        self.assertEqual(self.run_plan()["eligible"][0]["action"], "preregister")

    def test_moving_branch_is_not_an_exact_revision(self):
        self.item["scope"]["code"]["hermes"] = "main"
        self.assertEqual(self.run_plan()["eligible"][0]["action"], "preregister")

    def test_dependency_needs_bound_evidence_not_just_keep_label(self):
        prior = copy.deepcopy(self.item)
        prior.update(id="prerequisite", status="keep")
        prior["evidence"]["reconciled"] = False
        self.catalog["candidates"].append(prior)
        self.item["dependencies"] = ["prerequisite"]
        self.assertIsNone(self.run_plan()["selected"])
        prior["evidence"]["reconciled"] = True
        self.assertIsNotNone(self.run_plan()["selected"])

    def test_stale_owner_metadata_requests_refresh_not_new_execution(self):
        self.item["lease"] = {"owner": "worker-A"}
        r = recommend(self.catalog, self.root, now="2026-10-04T01:00:00Z")
        self.assertEqual(r["eligible"][0]["action"], "refresh_sources")

    def test_stale_snapshot_refreshes_instead_of_claiming_freshness(self):
        r = recommend(self.catalog, self.root, now="2026-10-04T01:00:00Z")
        self.assertEqual(r["eligible"][0]["action"], "refresh_sources")

    def test_replay_and_output_are_deterministic_and_create_only(self):
        a, b = self.run_plan(), self.run_plan(copy.deepcopy(self.catalog))
        self.assertEqual(a, b)
        out = self.root / "report"
        write_report(a, out)
        self.assertEqual(json.loads((out / "report.json").read_text()), a)
        with self.assertRaises(FileExistsError):
            write_report(b, out)

    def test_empty_catalog_produces_no_filler(self):
        self.catalog["candidates"] = []
        self.assertIsNone(self.run_plan()["selected"])

    def test_reconciliation_advances_once_then_negative_closes_leaf(self):
        self.item["evidence"]["reconciled"] = False
        self.assertEqual(self.run_plan()["eligible"][0]["action"], "reconcile")
        self.item["evidence"]["reconciled"] = True
        self.assertEqual(self.run_plan()["selected"]["action"], "experiment")
        self.item["status"] = "kill"
        self.assertIsNone(self.run_plan()["selected"])

    def test_incomparable_priorities_remain_visible_alternatives(self):
        self.item["priority"].update(impact="medium", reuse="high")
        other = copy.deepcopy(self.item)
        other.update(id="alternative", hypothesis="a different hypothesis")
        other["priority"].update(impact="high", reuse="low")
        self.catalog["candidates"].append(other)
        result = self.run_plan()
        self.assertEqual(len(result["frontier"]), 2)
        self.assertIsNone(result["selected"])
        self.assertEqual(result["selection_status"], "unresolved_comparison")

    def test_unrelated_reconciliation_cannot_preempt_a_ready_experiment(self):
        audit = copy.deepcopy(self.item)
        audit.update(id="audit", hypothesis="Check an unrelated old receipt")
        audit["next"]["action"] = "reconcile"
        audit["priority"].update(impact="low", information="low", reuse="low", cost="high")
        self.catalog["candidates"].append(audit)
        result = self.run_plan()
        self.assertEqual(result["selected"]["id"], "packet")
        self.assertEqual(result["selected"]["action"], "experiment")
        self.assertIn("audit", [x["id"] for x in result["dominated"]])

    def test_a_prerequisite_does_not_inherit_an_experiments_cost_or_value(self):
        self.item["priority"]["cost"] = "high"
        self.item["evidence"]["reconciled"] = False
        result = self.run_plan()
        self.assertIsNone(result["selected"])
        self.assertEqual(result["selection_status"], "unassessed_actions")
        row = result["unassessed"][0]
        self.assertEqual(row["action"], "reconcile")
        self.assertEqual(row["declared_action"], "experiment")
        self.assertEqual(row["priority"]["cost"], "high")
        self.assertFalse(row["priority_applies_to_action"])
        self.assertEqual(result["frontier"], [])

    def test_unassessed_prerequisite_stays_visible_without_blocking_ready_work(self):
        pending = copy.deepcopy(self.item)
        pending.update(id="pending", hypothesis="Another mechanism has incomplete evidence")
        pending["evidence"]["reconciled"] = False
        self.catalog["candidates"].append(pending)
        result = self.run_plan()
        self.assertEqual(result["selected"]["id"], "packet")
        self.assertEqual(result["selection_status"], "provisional_curated")
        self.assertEqual([x["id"] for x in result["unassessed"]], ["pending"])

    def test_unresolved_report_does_not_claim_there_is_no_eligible_work(self):
        self.item["evidence"]["reconciled"] = False
        output = self.root / "unresolved"
        write_report(self.run_plan(), output)
        rendered = (output / "NEXT.txt").read_text()
        self.assertIn("No selected action", rendered)
        self.assertIn("packet", rendered)
        self.assertNotIn("No eligible action", rendered)

    def test_future_snapshot_is_invalid_instead_of_zero_age(self):
        with self.assertRaises(ValueError):
            recommend(self.catalog, self.root, now="2026-10-01T01:00:00Z")

    def test_path_escape_is_not_read_as_evidence(self):
        self.item["evidence"]["sources"][0]["path"] = "../outside.txt"
        r = self.run_plan()["eligible"][0]
        self.assertEqual(r["evidence_checks"][0]["status"], "outside_root")

    def test_foreign_factor_cannot_silently_create_another_integration(self):
        self.item["factor"] = "new-control-plane"
        with self.assertRaises(ValueError):
            self.run_plan()

    def test_cli_runs_and_replays_the_actual_catalog(self):
        catalog = self.root / "catalog.json"
        catalog.write_text(json.dumps(self.catalog))
        reports = []
        for name in ("first", "replay"):
            out = self.root / name
            process = subprocess.run(
                [sys.executable, "-m", "evolution_lab", "research", "--catalog", str(catalog),
                 "--evidence-root", str(self.root), "--out", str(out),
                 "--as-of", "2026-10-02T01:00:00Z"],
                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIn("Next: packet", process.stdout)
            self.assertEqual((out / "catalog.json").read_bytes(), catalog.read_bytes())
            reports.append((out / "report.json").read_bytes())
        self.assertEqual(reports[0], reports[1])


if __name__ == "__main__":
    unittest.main()
