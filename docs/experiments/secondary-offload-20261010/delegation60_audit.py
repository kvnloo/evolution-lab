"""Bounded projection of retained delegation receipts, not a grader or runner.

No inference, transcript parsing, routing, or source writes. Pass the retained
cohort and the existing authority ledger; output contains aggregates/digests only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics

EXPECTED = {
    "protocol.json": "e9dbeef22ad91af28f453f3063041ac4de6efb04206a57ada4f675b7f317980f",
    "canonical-receipts.jsonl": "f51af91e8d218685ced484114d6a42962cb87c67532e4a9d3f87a3393ea9c5f9",
    "reconciliation.json": "334e4cd5d73a50fb21224684553c434d2b121553ec5a1bc1536a5a1450c3062b",
    "cache-analysis.json": "8cf4153e648a4d7215c7b129580d225fe1665b10f1e8ef531c47aa36888054c5",
}


def audit(cohort: Path, ledger: Path) -> dict:
    for name, digest in EXPECTED.items():
        assert hashlib.sha256((cohort / name).read_bytes()).hexdigest() == digest, f"Wrong frozen {name}"
    protocol = json.loads((cohort / "protocol.json").read_text())
    reconciliation = json.loads((cohort / "reconciliation.json").read_text())
    cache = json.loads((cohort / "cache-analysis.json").read_text())
    pair_paths = sorted(cohort.glob("*/pair.json"))
    pair_digest = hashlib.sha256(b"".join(p.read_bytes() for p in pair_paths)).hexdigest()
    assert pair_digest == "0765720f30f270804aa07a69cfa2757cdd589bdc0e9938c2fcf199274aad106a", "Wrong frozen pairs"
    pairs = [json.loads(p.read_text()) for p in pair_paths]
    assert len(pairs) == 60
    assert {(r["task"], r["replicate"]) for r in pairs} == {(t, i) for t in ("summary", "gate") for i in range(30)}
    raw = (cohort / "canonical-receipts.jsonl").read_bytes().splitlines(keepends=True)
    # Same byte-membership control as original finalize.py; do not import history.
    ledger_lines = set(ledger.read_bytes().splitlines(keepends=True))
    assert all(line in ledger_lines for line in raw), "Canonical export not in supplied ledger"
    rows = [json.loads(line) for line in raw]
    latest = {r["trace_id"]: r for r in rows}
    physical = [r for r in latest.values() if r.get("execution") == "live" and r.get("extra", {}).get("physical_call_attempted")]
    assert len(physical) == 60
    assert all(r["provider"] == "cerebras" and r["model"] == "qwen-3.8-27b" and r["extra"]["status"] == "completed" for r in physical)
    assert all(r["extra"]["request_parameters"]["reasoning_effort"] == "none" for r in physical)
    assert len({r["extra"]["caller_trace_id"] for r in physical}) == 60
    ids = [r["extra"].get("response_id") for r in physical]
    known_ids = [i for i in ids if i]
    assert len(set(known_ids)) == len(known_ids), "Duplicate provider identity"
    family = Counter("chatcmpl-UUID" if re.fullmatch(r"chatcmpl-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", i) else "other" for i in known_ids)
    parent = {}
    for arm in ("A", "B"):
        parent[arm] = {k: sum(r[arm]["usage"].get(k, 0) for r in pairs) for k in ("input_tokens", "output_tokens", "cached_input_tokens", "reasoning_output_tokens")}
        parent[arm]["uncached_input_tokens"] = parent[arm]["input_tokens"] - parent[arm]["cached_input_tokens"]
        for key in ("uncached_input_tokens", "output_tokens", "cached_input_tokens"):
            assert parent[arm][key] == reconciliation["parent"][arm][key]
    functions = {}
    for task in ("summary", "gate"):
        group = [r for r in pairs if r["task"] == task]
        equal_cache = [r for r in group if r["A"]["usage"]["cached_input_tokens"] == r["B"]["usage"]["cached_input_tokens"]]
        delta = statistics.mean(r["parent_total_delta"] for r in equal_cache)
        assert delta == cache["functions"][task]["equal_cache_uncached_delta"]["mean"]
        functions[task] = {"repeats": len(group), "equal_cache_pairs": len(equal_cache), "equal_cache_mean_parent_token_delta_B_minus_A": delta}
    for r in pairs:
        a, b = r["A"]["usage"], r["B"]["usage"]
        assert r["parent_total_delta"] == b["input_tokens"] - b["cached_input_tokens"] + b["output_tokens"] - a["input_tokens"] + a["cached_input_tokens"] - a["output_tokens"]
        assert r["receipt"] in latest
    worker = {k: sum(r[k] for r in physical) for k in ("input_tokens", "output_tokens")}
    assert worker["input_tokens"] == reconciliation["worker_input_tokens"]
    assert worker["output_tokens"] == reconciliation["worker_output_tokens"]
    parent_delta = sum(r["parent_total_delta"] for r in pairs)
    return {
        "source_digests": EXPECTED,
        "pair_bytes_sha256": pair_digest,
        "producer_revision": None,
        "canonical_rows": len(rows), "authority_physical_attempts": len(physical),
        "provider_response_id_coverage": len(known_ids), "unique_provider_response_ids": len(set(known_ids)),
        "response_id_families": dict(family), "canonical_bytes_in_ledger": True,
        "independent_task_specifications": 2, "matched_pairs": len(pairs),
        "functions": functions, "parent": parent, "worker": worker,
        "parent_uncached_input_plus_output_delta_B_minus_A": parent_delta,
        "observed_uncached_parent_plus_worker_token_delta_B_minus_A": parent_delta + sum(worker.values()),
        "cache_neutral_parent_delta_sensitivity_only": sum(parent["B"][k] - parent["A"][k] for k in ("input_tokens", "output_tokens")),
        "provider_reported_cost_nonnull": sum(r["extra"].get("provider_reported_cost_usd") is not None for r in physical),
        "historical_free_only": protocol["free_only"], "historical_credit_balance_measured": protocol["credit_balance_measured"],
        "billing": protocol["billing_class"],
        "recorded_original_parent_quality_passes": {a: reconciliation["parent"][a]["quality_passes"] for a in ("A", "B")},
        "quality_status": "Historical original grades only; separate corrected report exists, not regraded here",
        "verifier_and_coordinator_full_cost": None,
        "independent_review": "NOT_RUN", "performance_verdict": "NOT_COMPARABLE",
        "limitation": "Historical two-specification repeats, not a current matched arm or natural task-population sample. No savings or admission claim; reasoning counters are reported, not added again to output.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", type=Path)
    parser.add_argument("ledger", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.cohort, args.ledger), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
