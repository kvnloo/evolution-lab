"""Outer capability loop (thin): mine → L2 recovery battery → canary marker.

Does not expand onboard adapters. Does not free-discover cards beyond SEED_CARDS
until savings proof is durable (receipts + kerdoios economics).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def run_outer_loop(
    *,
    mine: bool = True,
    l2: bool = True,
    closed_loop_seeds: int = 24,
    write: bool = True,
) -> dict[str, Any]:
    t0 = time.time()
    out: dict[str, Any] = {
        "schema": "z0int.outer_loop.v1",
        "ts": t0,
        "steps": {},
    }
    if mine:
        from .capability_miner import run_capability_mine

        out["steps"]["capability_mine"] = {
            "ok": True,
            "summary_keys": list((run_capability_mine() or {}).keys())[:12],
        }
    if l2:
        from .l2_benchmark import canary_active, run_l2_recovery_benchmark

        rep = run_l2_recovery_benchmark(
            closed_loop_seeds=closed_loop_seeds,
            write=write,
            promote=True,
        )
        out["steps"]["l2_recovery"] = {
            "ok": bool(rep.get("gates", {}).get("pass")),
            "gates": rep.get("gates"),
            "canary_marker": rep.get("canary_marker"),
            "report_path": rep.get("report_path"),
            "canary_active": canary_active(),
        }
    out["elapsed_s"] = time.time() - t0
    out["next"] = [
        "Join live outcomes via `z0int receipt join` for non-gym world events",
        "Record Kerdoios premium tokens: `python -m kerdoios economics --capability-id recovery_action`",
        "Unlock context_file_relevance after path/outcome instrumentation",
        "Do not expand onboard/data adapters until savings proof is durable",
    ]
    if write:
        dest = Path.home() / ".z0int" / "research" / "outer-loop-last.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
        out["path"] = str(dest)
    return out
