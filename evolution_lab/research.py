"""Candidate-local evidence checks over an explicitly curated research catalog.

This is a planning surface in Evolution Lab, not an executor or an outcome judge.
Source hashes establish identity, never truth. Qualitative priorities are declared
research judgments; they are not measured information gain or predicted savings.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .factorized_stack import FACTORS

SCHEMA = "evolution_lab.research.v1"
POLICY = "candidate-local-pareto-v2"
GOALS = {"e2e_latency", "consumed_tokens", "intent_validation"}
INVARIANTS = {"independent_outcome", "bounded_authority", "all_attempts", "frozen_judge"}
RATINGS = {"low": 0, "medium": 1, "high": 2}
TERMINAL = {"keep", "kill", "complete", "hold"}


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _time(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("timestamps must include a timezone")
    return result.astimezone(timezone.utc)


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _strings(value: Any) -> bool:
    return isinstance(value, list) and all(_text(x) for x in value)


def _sources(evidence: dict, root: Path) -> list[dict]:
    checks = []
    for source in evidence.get("sources", []):
        if not isinstance(source, dict) or not _text(source.get("path")):
            raise ValueError("evidence sources require a local path")
        path = (root / source["path"]).resolve()
        expected = source.get("sha256")
        row = {"path": source["path"], "kind": source.get("kind"),
               "expected_sha256": expected, "actual_sha256": None}
        if source.get("kind") not in {"issue", "receipt", "source", "review"}:
            raise ValueError("unknown evidence kind")
        if not path.is_relative_to(root):
            row["status"] = "outside_root"
        elif not path.is_file():
            row["status"] = "missing"
        else:
            row["actual_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            row["status"] = "match" if row["actual_sha256"] == expected else "digest_mismatch"
        checks.append(row)
    return checks


def _preregistration_gaps(item: dict) -> list[str]:
    scope, step = item.get("scope", {}), item.get("next", {})
    gaps = []
    code = scope.get("code")
    if not isinstance(code, dict) or not code or not all(
            isinstance(v, str) and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", v) for v in code.values()):
        gaps.append("code revisions")
    for key in ("model", "routing", "permissions"):
        if not _text(scope.get(key)):
            gaps.append(key)
    if not _strings(scope.get("fixtures")) or not scope["fixtures"]:
        gaps.append("frozen fixture IDs")
    for key in ("acceptance", "stop", "reopen", "primary_outcome"):
        if not _text(step.get(key)):
            gaps.append(key)
    if not _strings(step.get("controls")) or not step["controls"]:
        gaps.append("negative controls")
    if not {"cold", "warm", "retries", "failures"}.issubset(step.get("accounting", [])):
        gaps.append("complete cold/warm/retry/failure accounting")
    if step.get("mode") not in {"offline", "shadow", "bounded"}:
        gaps.append("bounded experiment mode")
    return gaps


def _dominates(a: dict, b: dict) -> bool:
    # Ordinal Pareto comparison, never multiplication of made-up cardinal scores.
    if not set(a["goals"]).issuperset(b["goals"]):
        return False
    av = tuple(RATINGS[a["priority"][k]] for k in ("impact", "information", "reuse"))
    bv = tuple(RATINGS[b["priority"][k]] for k in ("impact", "information", "reuse"))
    av += (-RATINGS[a["priority"]["cost"]],)
    bv += (-RATINGS[b["priority"]["cost"]],)
    return all(x >= y for x, y in zip(av, bv)) and (
        any(x > y for x, y in zip(av, bv)) or set(a["goals"]) != set(b["goals"]))


def recommend(catalog: dict, root: Path, *, now: str | None = None,
              max_age_hours: float = 24) -> dict:
    """Return a replayable recommendation; never execute commands or fetch sources."""
    if catalog.get("schema") != SCHEMA:
        raise ValueError(f"expected schema {SCHEMA}")
    intent = catalog.get("intent", {})
    if not _text(intent.get("objective")) or not _strings(intent.get("goals")):
        raise ValueError("an explicit objective and goal list are required")
    if not intent["goals"] or not set(intent["goals"]).issubset(GOALS):
        raise ValueError("unsupported or empty goals")
    if not INVARIANTS.issubset(intent.get("invariants", [])):
        raise ValueError("independent outcome, authority, denominator and judge constraints are required")
    if not _strings(intent.get("resources")):
        raise ValueError("available resources must be explicit")
    now = now or datetime.now(timezone.utc).isoformat()
    current, snapshot = _time(now), _time(catalog["snapshot_at"])
    age = (current - snapshot).total_seconds() / 3600
    if age < 0 or not 0 < max_age_hours <= 24 * 365:
        raise ValueError("future snapshot or invalid freshness bound")
    root = Path(root).resolve()
    items = catalog.get("candidates")
    if not isinstance(items, list) or not all(isinstance(x, dict) for x in items):
        raise ValueError("candidates must be a list of objects")
    ids = [x.get("id") for x in items]
    if not all(_text(x) for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("candidate IDs must be nonempty and unique")
    by_id = {x["id"]: x for x in items}
    eligible, excluded, duplicates, seen = [], [], [], {}
    factors = {f.id for f in FACTORS}

    for item in sorted(items, key=lambda x: x["id"]):
        ident = item["id"]
        if item.get("factor") is not None and item["factor"] not in factors:
            raise ValueError(f"{ident}: use an existing factor ID")
        status = item.get("status")
        if status not in TERMINAL | {"open"}:
            raise ValueError(f"{ident}: unknown disposition")
        if status in TERMINAL:
            excluded.append({"id": ident, "reason": f"settled disposition: {status}; explicit new evidence/reopen required"})
            continue
        goal_list = item.get("goals", [])
        if not _strings(goal_list) or not set(goal_list).issubset(GOALS):
            raise ValueError(f"{ident}: invalid goals")
        goals = sorted(set(goal_list) & set(intent["goals"]))
        contract = item.get("intent", {})
        if not goals or not all(_text(contract.get(k)) for k in ("task_success", "oracle")) or not set(
                intent["invariants"]).issubset(contract.get("invariants", [])):
            excluded.append({"id": ident, "reason": "intent contract incomplete or incompatible"})
            continue
        decision = item.get("decision", {})
        if not all(_text(decision.get(k)) for k in ("supports", "refutes", "ambiguous")) or len(set(decision.values())) < 2:
            excluded.append({"id": ident, "reason": "no named disposition-changing result"})
            continue
        if not all(_text(item.get(k)) for k in ("hypothesis", "owner", "issue", "comparator")):
            raise ValueError(f"{ident}: hypothesis, existing owner, source and simpler comparator required")
        evidence, step = item.get("evidence", {}), item.get("next", {})
        if evidence.get("level") not in {"D", "I", "R", "C", "X"} or type(evidence.get("reconciled")) is not bool:
            raise ValueError(f"{ident}: explicit evidence level and reconciliation state required")
        if not all(_text(evidence.get(k)) for k in ("missing", "strongest", "counterevidence")):
            raise ValueError(f"{ident}: evidence and counterevidence must remain explicit")
        priority = item.get("priority", {})
        if not all(priority.get(k) in RATINGS for k in ("impact", "information", "reuse", "cost")) or not _text(priority.get("rationale")):
            raise ValueError(f"{ident}: justified ordinal judgments are required, not inferred numerical scores")
        if step.get("action") not in {"experiment", "analysis", "review", "reconcile"} or not _text(step.get("summary")):
            raise ValueError(f"{ident}: unsupported next action")
        fingerprint = _hash({"hypothesis": item["hypothesis"].strip(), "scope": item.get("scope"),
                             "comparator": item["comparator"], "action": step["action"]})
        if fingerprint in seen:
            duplicates.append({"id": ident, "same_as": seen[fingerprint]})
            continue
        seen[fingerprint] = ident
        checks = _sources(evidence, root)
        action, reason = step["action"], step["summary"]
        lease = item.get("lease")
        if age > max_age_hours:
            action, reason = "refresh_sources", "Cached snapshot is stale; refresh ownership and evidence metadata."
        elif lease:
            if not _text(lease.get("owner")):
                raise ValueError(f"{ident}: lease owner required")
            if not lease.get("until") or _time(lease["until"]) > current:
                excluded.append({"id": ident, "reason": "existing owner is active; do not duplicate", "owner": lease["owner"]})
                continue
            action, reason = "check_owner", "Lease expired: locate owner outcome before rescheduling."
        elif not evidence["reconciled"] or not checks or any(c["status"] != "match" for c in checks):
            action, reason = "reconcile", "Locate/bind existing evidence before scheduling execution; missing data is not proof of nonexecution."
        elif step["action"] == "experiment" and (gaps := _preregistration_gaps(item)):
            action, reason = "preregister", "Complete the existing experiment contract: " + ", ".join(gaps)
        if action in {"experiment", "analysis", "review"}:
            resources = step.get("resources")
            if not _strings(resources):
                raise ValueError(f"{ident}: resources must be explicit")
            unavailable = sorted(set(resources) - set(intent["resources"]))
            dependencies = item.get("dependencies", [])
            if not _strings(dependencies) or any(d not in by_id for d in dependencies):
                raise ValueError(f"{ident}: unknown dependency")
            waiting = []
            for dependency in dependencies:
                other = by_id[dependency]
                proof = other.get("evidence", {})
                bound = _sources(proof, root)
                if (other.get("status") not in {"keep", "complete"} or proof.get("reconciled") is not True
                        or not bound or any(c["status"] != "match" for c in bound)):
                    waiting.append(dependency)
            if unavailable or waiting:
                excluded.append({"id": ident, "reason": "unavailable resource or unresolved dependency",
                                 "resources": unavailable, "dependencies": waiting})
                continue
        eligible.append({"id": ident, "action": action, "reason": reason, "goals": goals,
                         "owner": item["owner"], "issue": item["issue"], "hypothesis": item["hypothesis"],
                         "priority": dict(priority), "declared_action": step["action"],
                         "priority_applies_to_action": action == step["action"],
                         "evidence_level_reported": evidence["level"],
                         "missing_evidence": evidence["missing"], "strongest": evidence["strongest"],
                         "counterevidence": evidence["counterevidence"], "evidence_checks": checks,
                         "decision": decision, "intent": contract, "comparator": item["comparator"],
                         "scope": item.get("scope", {}), "packet": step, "fingerprint": fingerprint})

    # Evidence gates constrain the affected candidate, not the entire portfolio.
    # A prerequisite is a different action; do not invent its cost or inherit the
    # deferred experiment's value. Keep it visible until explicitly assessed.
    unassessed = [x for x in eligible if not x["priority_applies_to_action"]]
    peers = [x for x in eligible if x["priority_applies_to_action"]]
    frontier = [x for x in peers if not any(_dominates(y, x) for y in peers)]
    dominated = [x for x in peers if x not in frontier]
    selected = frontier[0] if len(frontier) == 1 else None
    selection_status = ("provisional_curated" if selected else
                        "unresolved_comparison" if frontier else
                        "unassessed_actions" if unassessed else "no_eligible_action")
    report = {"schema": "evolution_lab.research_report.v2", "policy": POLICY,
              "engine_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "factor_registry_sha256": _hash([f.to_dict() for f in FACTORS]),
              "as_of": current.isoformat(), "snapshot_at": snapshot.isoformat(),
              "max_age_hours": max_age_hours, "catalog_sha256": _hash(catalog),
              "intent": intent, "selected": selected, "selection_status": selection_status,
              "frontier": frontier, "dominated": dominated, "eligible": eligible,
              "unassessed": unassessed,
              "excluded": excluded, "duplicates": duplicates, "runtime_authority": False,
              "model_calls": 0, "network_calls": 0,
              "selection_note": "Candidate-local gates, then ordinal Pareto comparison of explicitly assessed current actions. A sole frontier item is a provisional curated suggestion. Unassessed actions remain visible; unresolved comparisons have no selected winner.",
              "limits": "Curated claims and priorities remain judgments, not measured workload value or an optimal policy. Matching hashes bind bytes, not truth. Declared intent fields do not prove alignment with the user's actual goal. No scientific certification, promotion, or runtime authority is granted."}
    report["receipt_sha256"] = _hash(report)
    return report


def write_report(report: dict, output: Path) -> None:
    """Create a new immutable-by-convention result directory; never overwrite a run."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    chosen = report["selected"]
    lines = ["RESEARCH CANDIDATE REVIEW", report["intent"]["objective"], "", f"As of: {report['as_of']}"]
    if chosen:
        lines += [f"Next: {chosen['id']} — {chosen['action']} (provisional curated suggestion)", chosen["reason"],
                  f"Work: {chosen['packet']['summary']}",
                  f"Decision: {json.dumps(chosen['decision'], sort_keys=True)}",
                  f"Simpler comparator: {chosen['comparator']}", f"Existing result: {chosen['strongest']}",
                  f"Counterevidence: {chosen['counterevidence']}",
                  f"Missing: {chosen['missing_evidence']}", f"Owner: {chosen['owner']} — {chosen['issue']}",
                  f"Acceptance: {chosen['packet'].get('acceptance', 'Complete the named evidence binding')}",
                  f"Stop: {chosen['packet'].get('stop', 'No disposition-changing gap remains')}",
                  f"Reopen: {chosen['packet'].get('reopen', 'New relevant evidence')}"]
    elif report["frontier"]:
        lines.append("No selected action: declared priorities leave unresolved alternatives.")
    elif report["unassessed"]:
        lines.append("No selected action: current prerequisites have no action-specific priority assessment.")
    else:
        lines.append("No eligible action; do not synthesize filler work.")
    lines += ["", "Nondominated alternatives: " + ", ".join(x["id"] for x in report["frontier"]),
              "", "Unassessed prerequisites (no inferred cost or priority):"]
    lines += [f"- {x['id']}: {x['action']} — {x['reason']}" for x in report["unassessed"]]
    lines += ["", "Excluded:"]
    lines += [f"- {x['id']}: {x['reason']}" for x in report["excluded"]]
    lines += ["", report["selection_note"], report["limits"],
              f"Receipt: {report['receipt_sha256']}"]
    (output / "NEXT.txt").write_text("\n".join(lines) + "\n")
