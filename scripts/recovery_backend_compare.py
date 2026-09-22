#!/usr/bin/env python3
"""Recovery-domain comparison on the frozen P0 splits: can a cheap policy take these?

Item 5 of the Phase 2 block, and the integration question behind item 2: `hermes_recovery`
is the task the mushroom-body specialist was actually fitted for, so this is the only
place its checkpoint can be pointed at honestly.

Everything runs on the frozen splits (`data/p0/*.npz`, manifest seed 20260912) — nothing
is re-sampled. The mushroom path is imported from `evolution_lab.models` rather than
re-implemented, so it is byte-for-byte the P0 inference path (PN features -> k-winner
KC codes -> MBON argmax -> delayed-cue protocol) and not a lookalike.

Metric: **per-step action accuracy** over all 8 steps of every episode. This is a proxy,
not the P0 closed-loop reward — the P0 control table already carries closed-loop numbers
and they are quoted in the output for cross-reference.

Risk here is wrong actions, and the recovery action set includes `escalate` and
`page_human`, so a wrong answer is not automatically unsafe. Nothing here measures danger.

Usage:
  python scripts/recovery_backend_compare.py
  python scripts/recovery_backend_compare.py --json results/recovery-compare.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evolution_lab.models import (  # noqa: E402
    ACTIONS,
    _apply_delayed_cue_protocol,
    _kc_codes,
    _pn_features,
)
from evolution_lab.task import teacher_action  # noqa: E402

P0 = ROOT / "data" / "p0"
BUNDLE = P0 / "recovery_student.npz"
META = P0 / "recovery_student.json"
SPLITS = ("train", "val", "confirm", "ood")


class Frames:
    """Minimal stand-in for evolution_lab.Episode: the P0 helpers only need `.frames`."""

    __slots__ = ("frames",)

    def __init__(self, frames: np.ndarray) -> None:
        self.frames = frames


def load_split(name: str) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(P0 / f"{name}.npz", allow_pickle=True)
    return d["frames"], d["labels"]


# ------------------------------------------------------------------ predictors


def predict_majority(tr_labels: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """Most frequent per-step label in train, broadcast to every step."""
    flat = tr_labels.reshape(-1)
    top = int(np.bincount(flat, minlength=len(ACTIONS)).argmax())
    return np.full(frames.shape[:2], top, dtype=np.int64)


def predict_rule(frames: np.ndarray) -> np.ndarray:
    """The reference policy, applied per step. This is the thing to beat."""
    return np.stack(
        [[teacher_action(frames[i, t]) for t in range(frames.shape[1])]
         for i in range(frames.shape[0])]
    ).astype(np.int64)


def predict_mushroom(frames: np.ndarray) -> tuple[np.ndarray, dict]:
    """P0's own inference path for the local_plasticity (mushroom-body) student.

    `apply_strobe` is a training augmentation and is deliberately omitted here; the
    delayed-cue protocol is part of the policy contract and is kept.
    """
    b = np.load(BUNDLE, allow_pickle=True)
    meta = json.loads(META.read_text())
    W_kc_mbon = b["W_kc_mbon"]
    W_pn_kc = b["W_pn_kc"]
    k_winners = int(np.asarray(b["k_winners"]).reshape(-1)[0])

    eps = [Frames(frames[i]) for i in range(frames.shape[0])]
    Xp = _pn_features(eps)                     # (N, 10*n_features + 1) = (N, 151)
    Ht = _kc_codes(Xp, W_pn_kc, k_winners)     # (N, hidden)
    raw = (Ht @ W_kc_mbon).argmax(axis=1)      # one action per episode
    per_ep = _apply_delayed_cue_protocol(eps, raw)
    # Expand the episode-level decision across the 8 steps so the comparison is
    # step-for-step with the per-step predictors above.
    per_step = np.repeat(per_ep[:, None], frames.shape[1], axis=1)
    info = {
        "bundle": str(BUNDLE.relative_to(ROOT)),
        "bundle_version": meta.get("version"),
        "genome_id": meta.get("genome_id"),
        "encoder": meta.get("encoder"),
        "hidden": meta.get("hidden"),
        "history": meta.get("history"),
        "k_winners": k_winners,
        "n_params": meta.get("n_params"),
        "pn_features_dim": int(Xp.shape[1]),
        "p0_closed_loop_metric": (meta.get("metrics") or {}).get("closed_loop"),
    }
    return per_step, info


def accuracy(pred: np.ndarray, labels: np.ndarray) -> float:
    return float((pred == labels).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    tr_f, tr_l = load_split("train")
    print(f"frozen splits from {P0.relative_to(ROOT)} (manifest seed 20260912)")
    for s in SPLITS:
        f, l = load_split(s)
        print(f"  {s:8} frames {f.shape} labels {l.shape}")
    print(f"actions: {ACTIONS}")
    print()

    preds: dict[str, dict[str, np.ndarray]] = {}
    for s in ("confirm", "ood"):
        f, _ = load_split(s)
        preds.setdefault("majority", {})[s] = predict_majority(tr_l, f)
        preds.setdefault("rule", {})[s] = predict_rule(f)
        m, info = predict_mushroom(f)
        preds.setdefault("mushroom", {})[s] = m

    print("mushroom bundle:", json.dumps(info, indent=1))
    print()

    header = f"{'policy':10} {'confirm':>10} {'ood':>10}"
    print(header)
    print("-" * len(header))
    table: dict[str, dict] = {}
    for name in ("rule", "mushroom", "majority"):
        row = {}
        for s in ("confirm", "ood"):
            _, l = load_split(s)
            row[s] = round(accuracy(preds[name][s], l), 4)
        table[name] = row
        print(f"{name:10} {row['confirm']:>10.4f} {row['ood']:>10.4f}")

    print()
    print("P0 control table (closed-loop val_success, from the P0 run — a different")
    print("metric on a different split, quoted only for cross-reference):")
    print("  teacher-rule 1.000 | mushroom-body 1.000 | mlp 1.000 | gru 0.958")
    print("  rewired-reservoir 1.000 | direct-input 0.625")

    out = {
        "schema": "z0evals.recovery-backend-compare.v1",
        "evidence_class": "exploratory_beta",
        "promotion_eligible": False,
        "task": "hermes_recovery",
        "splits": str(P0.relative_to(ROOT)),
        "manifest_seed": 20260912,
        "actions": list(ACTIONS),
        "metric": "per-step action accuracy over all 8 steps",
        "metric_caveat": (
            "Per-step accuracy is a proxy for the P0 closed-loop reward, not the same "
            "number. The P0 control table is quoted separately and must not be merged "
            "with these values."
        ),
        "mushroom": info,
        "results": table,
        "p0_control_table_crossref": {
            "teacher-rule": 1.0, "mushroom-body-g1-8447": 1.0, "mlp": 1.0,
            "gru": 0.958, "rewired-reservoir": 1.0, "direct-input": 0.625,
        },
        "not_measured": (
            "NanoJev and Hammer3B are not in this table: they consume a typed state, not "
            "an 8x15 Hermes frame tensor, so including them needs a frame->state rendering "
            "that does not exist yet. Rendering it would be a new encoding, and comparing "
            "a fresh encoding against a policy fitted on the original features would not "
            "be a like-for-like comparison."
        ),
    }
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
