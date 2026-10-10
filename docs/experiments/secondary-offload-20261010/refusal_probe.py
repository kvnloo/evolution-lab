"""Read-only, offline replay of existing Kerdoios cache/refusal functions.

Set PYTHONPATH to an unchanged Kerdoios checkout and KERDOIOS_CACHE to the
existing cache. This is an experiment, not an admission service or quota store.
No provider client, credential access, cache refresh, or network call occurs.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from kerdoios.cache import CACHE_TTL, cache_path, load_inventory_cache
from kerdoios.quota import PlanningPolicy, free_only_rejection


def main() -> None:
    now = datetime.now(timezone.utc)
    source = cache_path()
    raw = source.read_bytes()
    payload = json.loads(raw)
    fresh = load_inventory_cache(now=now)
    # Diagnostic historical replay ONLY: never use ignore_ttl for admission.
    historical = load_inventory_cache(now=now, ignore_ttl=True) or []
    policy = PlanningPolicy(free_only=True, no_paid_spill=True)
    rows = []
    for offer in historical:
        if offer.provider not in {"groq", "cerebras"}:
            continue
        quota = offer.economics.quota
        rows.append({
            "provider": offer.provider,
            "observed_model": offer.model,
            "quota_observed_at": quota.observed_at if quota else None,
            "known_dimensions": sorted(quota.known_dimensions()) if quota else [],
            "existing_rejection": free_only_rejection(offer, policy),
            "entitlement": "unknown",
            "live_admitted": False,
        })
    assert fresh is None, "Snapshot changed: reevaluate freshness; do not infer entitlement"
    assert len(rows) == 5, "Snapshot changed: reevaluate cohort"
    assert all(r["existing_rejection"] is not None for r in rows)
    assert all(not r["known_dimensions"] for r in rows)
    print(json.dumps({
        "observed_at": now.isoformat(),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_saved_at": payload["saved_at"],
        "cache_ttl_seconds": CACHE_TTL.total_seconds(),
        "fresh_cache_result": None,
        "historical_diagnostic_only": True,
        "rows": rows,
        "physical_provider_calls": 0,
        "independent_review": "NOT_RUN",
        "performance_verdict": "NOT_COMPARABLE",
        "warning": "Existing rejection says exhausted even for wholly unknown quota; this does not prove zero balance.",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
