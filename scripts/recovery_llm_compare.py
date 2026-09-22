#!/usr/bin/env python3
"""Do a cheap *language* backend and a 3B tool-caller earn a rung on recovery?

Completes item 5's comparison: rule / MLP / GRU are already measured (P0 control table),
the mushroom is in `recovery_backend_compare.py`, and this adds NanoJev and Hammer3B.

**The encoding caveat, stated up front.** The mushroom was fitted on the raw 8x15 frame
tensor. NanoJev and Hammer3B consume a typed state, so they are handed a *rendering* of
that tensor, introduced here. This is therefore not a like-for-like comparison against a
policy fitted on the original features, and the artifact says so. It is still the right
question — "could we route recovery to something cheaper than the current path?" — but
the numbers must not be read as a controlled bake-off.

One decision per episode, expanded across the 8 steps, exactly as the mushroom path does,
so the two are at least scored the same way.

Usage:
  python scripts/recovery_llm_compare.py --backend nanojev
  python scripts/recovery_llm_compare.py --backend nanojev --backend hammer2.1_3b
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# z0int lives in its own repo; this script drives its DecisionBackends.
Z0INT = Path("/home/kvn/tmp/openjev/src")
if Z0INT.is_dir():
    sys.path.insert(0, str(Z0INT))

P0 = ROOT / "data" / "p0"
ACTIONS = ("retry", "restart_sandbox", "escalate", "noop", "page_human")
FEATURE_NAMES = (
    "sandbox_alive", "retry_norm", "budget", "elapsed", "transient",
    "hard", "unfamiliar", "policy_hit", "already_ok", "cue",
)
SERVE = "http://127.0.0.1:11500"

INSTRUCTIONS = (
    "You are the recovery policy for a tool-using agent. Given the recent step history, "
    "choose the single next action. Reply with ONLY one of: " + ", ".join(ACTIONS) + "."
)


def render(frames_i: np.ndarray) -> dict:
    """Render an 8x15 frame tensor into a compact typed state.

    Introduced by this script; not the encoding the mushroom was fitted on. Only the 10
    named features are surfaced — the trailing 5 columns are a one-hot of the action
    taken, which is not available at decision time.

    Rendered as one short line per feature rather than 8 JSON objects: NanoJev's
    encoder caps at 512 tokens and refuses to truncate, so the verbose form was
    rejected outright ("encoded candidate path exceeds max_length=512").
    """
    # Values are quantised to 0-9 and unpadded. NanoJev's encoder is capped at 512
    # tokens and refuses to truncate, and "1.00 1.00 1.00 ..." tokenises at roughly
    # four tokens per number, so 80 floats overflowed the budget on their own. The
    # quantisation is part of this rendering and is declared in the artifact.
    lines = [f"recovery history, {frames_i.shape[0]} steps, oldest first, each value 0-9"]
    for i, name in enumerate(FEATURE_NAMES):
        vals = "".join(str(min(9, max(0, round(float(frames_i[t, i]) * 9)))) for t in range(frames_i.shape[0]))
        lines.append(f"{name} {vals}")
    return {"task": "hermes_recovery", "features": "\n".join(lines)}


def ask_serve(model: str, state: dict, timeout: float = 120.0) -> tuple[str | None, float]:
    """Ask a model on the local z0int supervisor (OpenAI-compatible)."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": json.dumps(state, separators=(",", ":"))},
        ],
        "temperature": 0,
        "max_tokens": 16,
    }
    t0 = time.perf_counter()
    proc = subprocess.run(
        ["curl", "-sS", "--max-time", str(int(timeout)), f"{SERVE}/v1/chat/completions",
         "-H", "Content-Type: application/json", "-d", json.dumps(payload)],
        capture_output=True, text=True,
    )
    dt = (time.perf_counter() - t0) * 1000.0
    try:
        body = json.loads(proc.stdout)
        text = ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    except Exception:
        return None, dt
    low = text.strip().lower()
    for a in sorted(ACTIONS, key=len, reverse=True):
        if a in low:
            return a, dt
    return None, dt


def ask_nanojev(state: dict) -> tuple[str | None, float]:
    from z0int.backends.base import request_from_mapping
    from z0int.backends.registry import create_backend

    be = create_backend("nanojev")
    req = request_from_mapping({
        "state": state,
        "questions": [{
            "id": "action", "type": "choice", "instructions": INSTRUCTIONS,
            "options": [{"id": a, "description": a.replace("_", " ")} for a in ACTIONS],
        }],
    })
    t0 = time.perf_counter()
    res = be.evaluate(req)
    dt = (time.perf_counter() - t0) * 1000.0
    return (res.answers[0].value if res.answers else None), float(res.latency_ms or dt)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", action="append", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    out: dict = {
        "schema": "z0evals.recovery-llm-compare.v1",
        "evidence_class": "exploratory_beta",
        "promotion_eligible": False,
        "task": "hermes_recovery",
        "encoding_caveat": (
            "NanoJev and Hammer3B receive a text rendering of the frame tensor introduced "
            "by this script, not the 8x15 encoding the mushroom was fitted on. Values are "
            "quantised to 0-9 because NanoJev's encoder is capped at 512 tokens and refuses "
            "to truncate. Not a controlled bake-off; do not merge with "
            "recovery-compare.json."
        ),
        "backends": {},
    }

    for name in a.backend:
        rows = {}
        for split in ("confirm", "ood"):
            d = np.load(P0 / f"{split}.npz", allow_pickle=True)
            frames, labels = d["frames"], d["labels"]
            n = frames.shape[0] if not a.limit else min(a.limit, frames.shape[0])
            pred_step = np.zeros_like(labels[:n])
            lat, misses = [], 0
            for i in range(n):
                st = render(frames[i])
                if name == "nanojev":
                    pick, dt = ask_nanojev(st)
                else:
                    pick, dt = ask_serve(name, st)
                lat.append(dt)
                if pick is None:
                    misses += 1
                idx = ACTIONS.index(pick) if pick in ACTIONS else -1
                pred_step[i, :] = idx
            acc = float((pred_step == labels[:n]).mean()) if n else 0.0
            rows[split] = {
                "n": n, "accuracy": round(acc, 4), "unparsed": misses,
                "latency_p50_ms": round(float(np.median(lat)), 1) if lat else None,
            }
            print(f"  {name:16} {split:8} n={n:3} acc={acc:.4f} unparsed={misses} "
                  f"p50={rows[split]['latency_p50_ms']}ms")
        out["backends"][name] = rows
        print()

    print("cross-reference (different metric / different split, do not merge):")
    print("  rule            confirm 0.9766  ood 1.0000   (per-step, this repo)")
    print("  mushroom        confirm 0.8385  ood 1.0000   (per-step, this repo)")
    print("  P0 closed-loop  teacher-rule 1.000 | mushroom 1.000 | mlp 1.000 | gru 0.958")

    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
