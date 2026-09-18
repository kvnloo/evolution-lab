"""Private live learning stream for z0int next-action.

Collect first; learn asynchronously. Never writes public git paths.
Pools under ~/.z0int/stream/:
  raw.jsonl           — every decision row
  high_info.jsonl     — disagree / high-entropy / weak-family / escalate
  outcome_gold.jsonl  — rows upgraded with deterministic outcomes
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

STREAM_ROOT = Path.home() / ".z0int" / "stream"
RAW = STREAM_ROOT / "raw.jsonl"
HIGH_INFO = STREAM_ROOT / "high_info.jsonl"
OUTCOME_GOLD = STREAM_ROOT / "outcome_gold.jsonl"
SCHEMA = "flyforge.live_decision.v1"
WEAK = frozenset({"EDIT", "WEB", "VERIFY", "ABSTAIN"})


def _ensure() -> None:
    STREAM_ROOT.mkdir(parents=True, exist_ok=True)


def make_trace_id() -> str:
    return uuid.uuid4().hex


def prompt_hash(text: str) -> str:
    return hashlib.blake2b((text or "").encode("utf-8"), digest_size=16).hexdigest()


def _entropy(probs: dict[str, float] | None) -> float | None:
    if not probs:
        return None
    ps = [float(p) for p in probs.values() if float(p) > 0]
    if not ps:
        return None
    s = sum(ps)
    if s <= 0:
        return None
    import math

    return float(-sum((p / s) * math.log(p / s) for p in ps))


def is_high_info(row: dict[str, Any]) -> bool:
    """Prioritize disagreement, entropy, weak families, escalate route."""
    if row.get("disagree"):
        return True
    route = (row.get("decision") or {}).get("route") or row.get("route")
    if route == "escalate":
        return True
    label = (row.get("decision") or {}).get("label") or (row.get("fly") or {}).get("label")
    if label in WEAK:
        return True
    ent = row.get("fly_entropy")
    if ent is None:
        ent = _entropy((row.get("fly") or {}).get("probs"))
    if ent is not None and ent >= 1.5:  # nats; ~soft max over 8 classes
        return True
    teachers = row.get("teachers") or {}
    labels = [t.get("label") for t in teachers.values() if isinstance(t, dict) and t.get("label")]
    if len(set(labels)) >= 2:
        return True
    return False


def append_decision(row: dict[str, Any], *, path: Path | None = None) -> dict[str, Any]:
    """Append one decision row to raw (+ high_info if prioritized)."""
    _ensure()
    out = dict(row)
    out.setdefault("schema", SCHEMA)
    out.setdefault("ts", time.time())
    out.setdefault("trace_id", make_trace_id())
    if "prompt" in out and "prompt_hash" not in out:
        out["prompt_hash"] = prompt_hash(str(out.get("prompt") or ""))
        # keep short prompt preview only
        out["prompt"] = str(out.get("prompt") or "")[:400]
    fly = out.get("fly") or out.get("next_action") or {}
    if isinstance(fly, dict) and "fly_entropy" not in out:
        out["fly_entropy"] = _entropy(fly.get("probs") if isinstance(fly.get("probs"), dict) else None)
    out["high_info"] = bool(is_high_info(out))
    dest = path or RAW
    with dest.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(out, default=str) + "\n")
    if out["high_info"] and dest == RAW:
        with HIGH_INFO.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(out, default=str) + "\n")
    _maybe_emit_z0int_receipt(out)
    return out


def _import_z0int_receipt():
    """Import z0int.receipt even when not pip-installed (EL 3.14 vs z0int <3.14)."""
    try:
        from z0int.receipt import append_receipt, build_receipt, join_outcome as z0_join  # type: ignore

        return append_receipt, build_receipt, z0_join
    except ImportError:
        pass
    import os
    import sys

    root = os.environ.get("Z0INT_ROOT", "/home/kvn/tmp/openjev")
    src = os.path.join(root, "src")
    if src not in sys.path and os.path.isdir(src):
        sys.path.insert(0, src)
    try:
        from z0int.receipt import append_receipt, build_receipt, join_outcome as z0_join  # type: ignore

        return append_receipt, build_receipt, z0_join
    except ImportError:
        return None


def _maybe_emit_z0int_receipt(row: dict[str, Any]) -> None:
    """Best-effort dual-write to z0int.decision_receipt.v1 under ~/.z0int/receipts."""
    mods = _import_z0int_receipt()
    if mods is None:
        return
    append_receipt, build_receipt, _z0_join = mods
    decision = row.get("decision") or {}
    fly = row.get("fly") or {}
    try:
        append_receipt(
            build_receipt(
                trace_id=str(row.get("trace_id") or ""),
                session_id=row.get("session_id"),
                capability_id=row.get("capability_id") or "coding.next_action",
                provider="local_mb",
                model=str(fly.get("source") or "mb"),
                prediction=decision.get("label") or fly.get("label"),
                confidence=(
                    float(decision["p"])
                    if decision.get("p") is not None
                    else (float(fly["p"]) if fly.get("p") is not None else None)
                ),
                action_taken=decision.get("route") or row.get("route"),
                route=decision.get("route") or row.get("route"),
                execution="log_only",
                latency_ms=row.get("latency_ms"),
                extra={"live_stream": True, "disagree": row.get("disagree")},
            )
        )
    except Exception:
        return




def join_outcome(
    trace_id: str,
    outcome: dict[str, Any],
    *,
    gold_path: Path | None = None,
) -> dict[str, Any] | None:
    """Attach an outcome to a prior trace and append outcome_gold when strong."""
    _ensure()
    if not RAW.exists():
        return None
    matched: dict[str, Any] | None = None
    for line in RAW.open(encoding="utf-8"):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("trace_id") == trace_id:
            matched = rec
    if matched is None:
        return None
    row = dict(matched)
    row["outcome"] = outcome
    row["outcome_ts"] = time.time()
    strong = bool(
        outcome.get("test_pass") is True
        or outcome.get("tool_ok") is True
        or outcome.get("task_done") is True
        or outcome.get("verifier_ok") is True
    )
    negative = bool(
        outcome.get("test_pass") is False
        or outcome.get("tool_ok") is False
        or outcome.get("user_correction") is True
        or outcome.get("reverted") is True
    )
    row["outcome_tier"] = "gold" if strong else ("negative" if negative else "soft")
    dest = gold_path or OUTCOME_GOLD
    if strong or negative:
        with dest.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
    try:
        mods = _import_z0int_receipt()
        if mods is not None:
            _append, _build, z0_join = mods
            z0_join(trace_id, outcome)
    except Exception:
        pass
    return row



def build_live_row(
    *,
    prompt: str,
    decision: dict[str, Any],
    fly: dict[str, Any],
    jev: dict[str, Any] | None = None,
    openjev: dict[str, Any] | None = None,
    session_id: str | None = None,
    latency_ms: float | None = None,
) -> dict[str, Any]:
    teachers: dict[str, Any] = {}
    if isinstance(jev, dict) and jev.get("ok"):
        teachers["jev"] = {
            "label": jev.get("label"),
            "p": jev.get("p"),
            "probs": jev.get("probs"),
            "source": jev.get("source"),
        }
    if isinstance(openjev, dict) and openjev.get("ok"):
        teachers["openjev"] = {
            "label": openjev.get("label"),
            "p": openjev.get("p"),
            "probs": openjev.get("probs"),
            "source": openjev.get("source"),
        }
    fly_label = fly.get("label") if isinstance(fly, dict) else None
    disagree = None
    if teachers and fly_label:
        disagree = any(t.get("label") and t.get("label") != fly_label for t in teachers.values())
    return {
        "schema": SCHEMA,
        "trace_id": make_trace_id(),
        "session_id": session_id or os.environ.get("OMP_SESSION_ID") or os.environ.get("HERMES_SESSION_ID"),
        "prompt": (prompt or "")[:400],
        "prompt_hash": prompt_hash(prompt or ""),
        "decision": {
            "route": decision.get("route"),
            "reason": decision.get("reason"),
            "escalate_to": decision.get("escalate_to"),
            "fallback": decision.get("fallback"),
            "label": decision.get("label"),
            "p": decision.get("p"),
            "margin": decision.get("margin"),
        },
        "fly": {
            "label": fly.get("label"),
            "p": fly.get("p"),
            "margin": fly.get("margin"),
            "probs": fly.get("probs"),
            "source": fly.get("source"),
            "n_params": fly.get("n_params"),
        },
        "teachers": teachers,
        "disagree": disagree,
        "latency_ms": latency_ms,
        "generation": 0,
    }


def pool_stats() -> dict[str, Any]:
    def _n(p: Path) -> int:
        if not p.exists():
            return 0
        with p.open(encoding="utf-8") as fh:
            return sum(1 for _ in fh)

    return {
        "schema": "flyforge.live_stream_stats.v1",
        "root": str(STREAM_ROOT),
        "raw": _n(RAW),
        "high_info": _n(HIGH_INFO),
        "outcome_gold": _n(OUTCOME_GOLD),
    }

