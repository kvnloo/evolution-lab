"""Bounded analytical projection, NOT a transcript parser or outcome grader.

Read the already-produced local-cognition cohort with stdlib JSON; preserve
owning evaluator labels and compare the existing report. No model execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


EXPECTED_RAW_SHA256 = "a5e837f7be65ce9e6114ecbd0d7aeaef704e805232a3f728e3c4afc2bab309b7"
EXPECTED_COMPARISON_SHA256 = "3b64b9b142f7e59e1333c83452402f0248da332f88adfc579dbb5e848cf0cef0"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", type=Path)
    args = parser.parse_args()
    raw = (args.cohort / "raw.jsonl").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_RAW_SHA256, "Wrong frozen cohort"
    comparison_bytes = (args.cohort / "comparison.json").read_bytes()
    assert hashlib.sha256(comparison_bytes).hexdigest() == EXPECTED_COMPARISON_SHA256, "Wrong frozen comparison"
    comparison = json.loads(comparison_bytes)
    grouped = defaultdict(list)
    for line in raw.splitlines():
        row = json.loads(line)
        if row["backend"].startswith(("groq/", "cerebras/")):
            grouped[row["backend"]].append(row)
    reports = {row["label"]: row for row in comparison["models"]}
    models = []
    groups = None
    for backend, rows in sorted(grouped.items()):
        ids = {row["fixture_id"] for row in rows}
        assert len(rows) == len(ids) == 28
        assert groups is None or groups == ids, "Unmatched work-item groups"
        groups = ids
        report = reports[backend]
        counts = {key: sum(bool(row[key]) for row in rows) for key in
                  ("schema_valid", "trajectory_correct", "deterministic_solution")}
        for key in ("schema_valid", "trajectory_correct"):
            assert counts[key] == report[key], "Owning report differs from raw labels"
        models.append({
            "backend_label": backend,
            "fixture_rows": len(rows),
            **counts,
            "raw_error_nonnull": sum(row["error"] is not None for row in rows),
            "reported_transport_failures": report["transport_failures"],
            "reported_retries": report["retries"],
            "native_request_id_coverage": sum(bool(row.get("response_id") or row.get("request_id")) for row in rows),
            "input_token_field_coverage": sum("input_tokens" in row for row in rows),
            "physical_request_count": None,
            "independently_verified_development_outcomes": None,
        })
    assert len(models) == 5
    print(json.dumps({
        "comparison_revision": "78869d977e847f8f87f3b8806cdcd6ce44b6bc22",
        "raw_producer_revision": None,
        "raw_provenance": "Locally retained untracked export; digest-frozen, not present in comparison_revision Git tree. Producer checkout had local modifications.",
        "raw_sha256": EXPECTED_RAW_SHA256,
        "comparison_sha256": hashlib.sha256(comparison_bytes).hexdigest(),
        "selected_fixture_rows": sum(row["fixture_rows"] for row in models),
        "matches_reported_120_call_cohort": False,
        "models": models,
        "independent_review": "NOT_RUN",
        "prior_head_coordinator_replication": {
            "reviewed_head": "2e9b4be4df2f6a7cf6bd867a4bd76b8d5dc0f2a3",
            "reviewer": "hermes-coordinator-review",
            "receipt": "https://github.com/kvnloo/evolution-lab/issues/23#issuecomment-6094788975",
            "scope": "PARTIAL analytical replication with provenance correction; not sealed offload efficacy",
        },
        "performance_verdict": "NOT_COMPARABLE",
        "limitation": "These are evaluator fixture rows, not deduplicated physical provider requests. No latency or savings inference.",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
