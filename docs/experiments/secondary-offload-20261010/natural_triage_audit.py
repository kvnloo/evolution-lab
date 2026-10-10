"""Source-bound metadata projection, not a CI judge, parser or model benchmark.

Input is the frozen public GitHub metadata projection described in natural-triage.md.
Only this experiment's three groups are accepted. No network or provider calls.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

from evolution_lab import sealed_split

assert hashlib.sha256(Path(sealed_split.__file__).read_bytes()).hexdigest() == (
    "74334f8477619daa7387e44036715911342dc76ad15bc658598b42e08f1ff9ad"
), "Owning grouping source drift"
make_session_split = sealed_split.make_session_split

SOURCE_SHA256 = "090eea0264c3d8e59bff992343bd8298536f642265752bc62dc87f3935f8d6bf"
HEADS = {
    "kvnloo/evolution-lab#28": "26d92191440449eac7cd6cdd783d0d6c646e46a6",
    "kvnloo/evolution-lab#36": "fea47981ade0fcdd7cc1555b226bc3ec0c165873",
    "kvnloo/z0evals#81": "6652713e25fb159a047457acfe348bd5ec98e146",
}


def project(data: dict) -> dict:
    groups = data["groups"]
    assert len(groups) == len(HEADS)
    assert {g["group"] for g in groups} == set(HEADS)
    rows, result = [], []
    for group in groups:
        name = group["group"]
        assert group["head_sha"] == HEADS[name], "Head drift"
        ids = [r["id"] for r in group["runs"]]
        assert len(ids) == len(set(ids)), "Duplicate run"
        assert ids, "Missing run evidence"
        failures, skips = [], []
        for run in group["runs"]:
            assert run["head_sha"] == group["head_sha"], "Wrong source head"
            rows.append({"session": name, "ts": 0.0})
            for job in run["jobs"]:
                failures.extend({"run": run["html_url"], "job": job["name"], **step}
                                for step in job["failed_steps"])
                skips.extend({"run": run["html_url"], **step}
                             for step in job["skipped_steps"])
        # Deliberately inadequate rule, shown only as a negative control.
        any_green = any(r["conclusion"] == "success" for r in group["runs"])
        result.append({"group": name, "head_sha": group["head_sha"],
                       "sampled_runs": len(ids), "any_green_baseline": any_green,
                       "contradicted_by_failed_steps": bool(any_green and failures),
                       "failed_steps": failures, "skipped_steps": skips,
                       "branch_head_is_checkout_attestation": False})
    # All inspected evidence remains development-only. No protected split made.
    split = make_session_split(rows, sealed_frac=0, dev_frac=0, future_frac=0)
    assert set(split.train_sessions) == set(HEADS)
    assert split.n_train_events == len(rows)
    assert not split.sealed_sessions and not split.future_sessions
    return {"source_sha256": SOURCE_SHA256, "observed_at": data["observed_at"],
            "coverage": data["sampling"], "groups": result,
            "group_count": len(groups), "sampled_runs": len(rows),
            "existing_grouping": split.to_dict(),
            "performance": "NOT_COMPARABLE", "independent_review": "NOT_RUN",
            "new_provider_calls": 0, "full_workflow_cost_usd": None}


def main() -> None:
    raw = Path(sys.argv[1]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA256, "Wrong frozen source"
    print(json.dumps(project(json.loads(raw)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
