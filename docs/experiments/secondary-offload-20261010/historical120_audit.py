"""Offline projection of recovered receipts; not a provider runner or outcome judge.

Requires the original private bundle and unchanged Tokenomics/Kerdoios on
PYTHONPATH. Emits aggregates only; never exports request IDs or account data.
"""
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

from kerdoios.quota.receipts import UsageReceipt, summarize_receipts
from tokenomics.jsonl import iter_jsonl

DIGESTS = {
    "bench2.json": "d2bb5499e1fed0ba5b7c35ce7214fa96de8a8e65c4c090274ad40e7983e32387",
    "tokenomics-groq-cerebras.jsonl": "7f0d6e2ec96fde171cb02a58bde69c1fafa52eb653fa39094b5e6883c7d81e31",
    "bench2.py": "8afb00c4f82367cc655aea0ff5802fbd810fc2827ea13312e11ac9eb3dc63ba4",
    "zer0prov.py": "98189475bd77e06de66b2617b0fbf36b27b819bf9e5d08df9e0cedcb4af105d5",
    "emit_tokenomics.py": "d28db83a01c987ff3684253842c8805781fc9c896853c39749c78f7403b7633d",
    "reconcile.py": "885bcbe7690cc78932563d78d4d2ba52bcaac4559b70ac92c13cbe1a70f41822",
    "derive_rows.py": "d6c94ae3b7b9354b9ffb22e520512769e5c9eab024591df7e0138c9ac8aad62f",
}


def audit(root):
    for name, expected in DIGESTS.items():
        assert sha256((root / name).read_bytes()).hexdigest() == expected, name
    raw = json.loads((root / "bench2.json").read_text())["receipts"]
    events = list(iter_jsonl(root / "tokenomics-groq-cerebras.jsonl", strict=True))
    assert len(raw) == len(events) == 120
    identities = [(r["provider"], r["model"], r["response_id"])
                  for r in raw if r.get("response_id")]
    assert len(identities) == len(set(identities)) == 119
    assert all(re.fullmatch(r"chatcmpl-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", r[2])
               for r in identities)
    # Identity join is primary. The failed row's status/latency correspondence
    # is weaker evidence, never a provider request dedupe identity.
    by_id = {(e.model.provider, e.model.name, (e.extra or {}).get("request_id")): e
             for e in events if (e.extra or {}).get("request_id")}
    assert set(by_id) == set(identities)
    failed = [r for r in raw if not r["response_id"]]
    failed_events = [e for e in events if not (e.extra or {}).get("request_id")]
    assert len(failed) == len(failed_events) == 1
    assert failed[0]["status"] == 429 and failed_events[0].status == "error"
    assert failed[0]["latency_ms"] == failed_events[0].latency.duration_ms
    for r in raw:
        if not r["response_id"]:
            continue
        e = by_id[(r["provider"], r["model"], r["response_id"])]
        assert e.usage.input_tokens == r["prompt_tokens"]
        assert e.usage.output_tokens == r["completion_tokens"]
        assert e.latency.duration_ms == r["latency_ms"]
    assert all(e.outcome is None for e in events)
    groups = []
    for provider, model in sorted({(r["provider"], r["model"]) for r in raw}):
        rs = [r for r in raw if (r["provider"], r["model"]) == (provider, model)]
        es = [e for e in events if (e.model.provider, e.model.name) == (provider, model)]
        # Existing consumer, same flattening as the historical reconcile script.
        # It does NOT promote nested extra.request_id to top-level request_id.
        projected = []
        for e in es:
            d = e.to_dict()
            d.update(provider=provider, model=model,
                     input_tokens=e.usage.input_tokens,
                     output_tokens=e.usage.output_tokens,
                     cached_input_tokens=e.usage.cached_input_tokens,
                     latency_ms=e.latency.duration_ms,
                     success=e.status == "ok", verified_outcome=None)
            projected.append(UsageReceipt.from_dict(d))
        usage = summarize_receipts(projected)
        assert usage.requests == len(rs) == 24
        assert all(r.request_id is None and r.retry_state is None for r in projected)
        duplicate_control = summarize_receipts(projected + projected)
        assert duplicate_control.requests == 48
        assert usage.prompt_tokens == sum(r["prompt_tokens"] or 0 for r in rs)
        assert usage.completion_tokens == sum(r["completion_tokens"] or 0 for r in rs)
        groups.append({
            "provider": provider, "model": model, "logical_rows": len(rs),
            "status_counts": dict(Counter(str(r["status"]) for r in rs)),
            "text_http_ok": sum(r["success"] for r in rs if r["kind"] == "text"),
            "tool_presence_success": sum(r["success"] for r in rs if r["kind"] == "tool_call"),
            "raw_verified_outcome_true_not_independent": sum(r["verified_outcome"] for r in rs),
            "unique_response_ids": sum(bool(r["response_id"]) for r in rs),
            "response_id_family": "chatcmpl-uuid; backend revision unknown",
            "input_tokens_observed": usage.prompt_tokens,
            "output_tokens_observed": usage.completion_tokens,
            "input_output_coverage_rows": sum(r["prompt_tokens"] is not None and r["completion_tokens"] is not None for r in rs),
            "cache_field_coverage_rows": sum(r["cache_tokens"] is not None for r in rs),
            "historical_reconcile_request_id_coverage": sum(bool(r.request_id) for r in projected),
            "historical_reconcile_retry_field_coverage": sum(r.retry_state is not None for r in projected),
            "historical_reconcile_verified": usage.verified,
            "historical_reconcile_retried": usage.retried,
            "deliberate_duplicate_input_control_requests": duplicate_control.requests,
            "independent_outcomes": "NOT_RUN",
        })
    return {
        "classification": "PARTIAL", "performance": "NOT_COMPARABLE",
        "source_git_sha": None, "source_sha_reason": "retained bundle has no Git repository",
        "source_sha256": DIGESTS, "tokenomics_revision": "3fa33ee83d02e43a37d7890656a4151e200f2275",
        "tokenomics_current_default": "65e8f2dd9d6c5784a2a1c67f5236d69a5ae94021",
        "kerdoios_revision": "9a74ed073c9a4721ae55e9cd7b058dabfe444ce4",
        "groups": groups, "logical_rows": len(raw), "unique_provider_response_ids": len(identities),
        "unidentified_failure_rows": 1, "physical_requests_total": None,
        "event_emission_time_utc": datetime.fromtimestamp(min(e.ts for e in events), timezone.utc).isoformat(),
        "provider_request_timestamps_coverage": sum("created" in r for r in raw),
        "serialized_zero_cost_rows": sum(e.economics.cost_usd == 0 for e in events),
        "invoice_cost_usd": None, "full_parent_worker_verifier_cost": None,
        "reasoning_token_coverage": 0, "ttft_coverage": 0,
        "new_live_provider_calls": 0, "independent_review": "NOT_RUN",
    }


if __name__ == "__main__":
    print(json.dumps(audit(Path(sys.argv[1])), indent=2, sort_keys=True))
