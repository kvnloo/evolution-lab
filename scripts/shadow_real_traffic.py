#!/usr/bin/env python3
"""Shadow the cheap paths over REAL Hermes traffic, not the synthetic corpus.

Item 6. Items 3-5 measured steal rates on a 28-task corpus and on 38k frozen recovery
frames. Neither is the traffic the agent actually produces. This uses the harness's own
next-action log — `~/.z0int/episodes/next_action.jsonl`, 76,970 rows recorded from live
sessions — where the tool the agent actually chose is known.

Task: given the user turn and the previous actions, predict the action family
(READ_SEARCH / EXECUTE / WEB / EDIT / DELEGATE / VERIFY / ABSTAIN).

**Split discipline.** Chronological, never random: rows are ordered by session time, and a
random split would let the same session appear in train and confirm. The candidate policy
(mushroom champion) is evaluated only on the held-out tail, exactly as the GPU trainer does.

**Why this dataset and not the corpus.** The corpus is 28 hand-written tasks x 3 reps. This
is tens of thousands of real decisions. If a cheap path is going to earn a runtime rung, it
has to survive this, not just the corpus.

Note on a number already in the repo: the champion bundle records `acc: 1.0` and
`ridge: 1.0` on `next_action`. A ridge baseline at 1.0 means the split it was scored on was
separable by construction. This script reports its own split, its own baselines, and the
row counts, so the two are not confusable.

Usage:
  python scripts/shadow_real_traffic.py --json results/shadow-real-traffic.json
  python scripts/shadow_real_traffic.py --backend nanojev
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
Z0INT = Path("/home/kvn/tmp/openjev/src")
if Z0INT.is_dir():
    sys.path.insert(0, str(Z0INT))

# `evolution_lab.next_action_gpu` imports the jax population trainer at module scope,
# and jax is not installed. Only the four pure helpers are needed here, so they are
# mirrored verbatim rather than importing a GPU trainer to hash some text.
import hashlib  # noqa: E402

EPISODES = Path.home() / ".z0int" / "episodes" / "next_action.jsonl"
FAMILIES = (
    "READ_SEARCH", "EDIT", "EXECUTE", "WEB",
    "DELEGATE", "VERIFY", "RESPOND", "ABSTAIN",
)
FAM_I = {n: i for i, n in enumerate(FAMILIES)}


def _hash(text: str, dim: int = 64) -> np.ndarray:
    out = np.zeros(dim, dtype=np.float32)
    t = (text or "").lower()
    if len(t) < 3:
        return out
    for i in range(len(t) - 2):
        h = hashlib.blake2b(t[i : i + 3].encode(), digest_size=8).digest()
        out[int.from_bytes(h[:4], "little") % dim] += 1.0
    n = float(np.linalg.norm(out))
    return out / n if n > 0 else out


def load_xy(path: Path, *, gold_only: bool, dim: int = 64) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            fam = rec.get("family")
            if fam not in FAM_I:
                continue
            if gold_only and not (rec.get("gold") or rec.get("tier") == "gold"):
                continue
            text = rec.get("text") or rec.get("user") or ""
            prev = rec.get("prev") or []
            if prev:
                text = text + " " + " ".join(str(x) for x in prev)
            xs.append(_hash(text, dim))
            ys.append(FAM_I[fam])
    if not xs:
        raise FileNotFoundError(f"no next-action rows in {path}")
    return np.stack(xs), np.asarray(ys, dtype=np.int32)


def ridge_acc(Xtr, ytr, Xte, yte) -> float:
    k = int(max(ytr.max(), yte.max())) + 1
    Y = np.eye(k, dtype=np.float64)[ytr]
    w = np.linalg.pinv(Xtr.T @ Xtr + 1.0 * np.eye(Xtr.shape[1])) @ Xtr.T @ Y
    return float(((Xte @ w).argmax(axis=1) == yte).mean())

CHAMPION = ROOT / "runs" / "gpu-evolve" / "next_action_champion.npz"


def mushroom_logits(X: np.ndarray, bundle: Path) -> np.ndarray:
    """Champion local_plasticity readout: relu(X @ W_pn_kc) -> k-winner -> MBON logits."""
    b = np.load(bundle, allow_pickle=True)
    drive = np.maximum(X @ b["W_pn_kc"], 0.0)
    k = int(np.asarray(b["k_winners"]).reshape(-1)[0])
    k = max(1, min(k, drive.shape[1]))
    idx = np.argpartition(drive, -k, axis=1)[:, -k:]
    mask = np.zeros_like(drive)
    np.put_along_axis(mask, idx, 1.0, axis=1)
    return drive * mask @ b["W_kc_mbon"]


def mushroom_predict(X: np.ndarray, bundle: Path) -> np.ndarray:
    return mushroom_logits(X, bundle).argmax(axis=1)


def margin_confidence(logits: np.ndarray) -> np.ndarray:
    """Softmax top-1 probability from the raw logits.

    The bundle has no calibrated head, so this is a *derived* confidence: softmax over
    the MBON readout. It is a monotone function of the logit margin and is good enough
    to sweep coverage, but it is not a calibrated probability and the artifact says so.
    """
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True)).max(axis=1)


def coverage_stats(pred: np.ndarray, y: np.ndarray, conf: np.ndarray, thrs) -> list[dict]:
    out = []
    for t in thrs:
        m = conf >= t
        cov = int(m.sum())
        correct = int((pred[m] == y[m]).sum()) if cov else 0
        out.append({
            "threshold": float(t), "covered": cov,
            "coverage": round(cov / len(y), 4) if len(y) else 0.0,
            "success_given_covered": round(correct / cov, 4) if cov else None,
            "escalated": int(len(y) - cov),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=str(EPISODES))
    ap.add_argument("--confirm-frac", type=float, default=0.30)
    ap.add_argument("--backend", action="append", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    path = Path(a.path)
    X, y = load_xy(path, gold_only=True)
    n = len(y)
    cut = int(n * (1 - a.confirm_frac))
    Xtr, ytr, Xte, yte = X[:cut], y[:cut], X[cut:], y[cut:]

    counts = np.bincount(yte, minlength=len(FAMILIES))
    majority = float(counts.max() / len(yte))

    k = int(max(ytr.max(), yte.max())) + 1
    Y = np.eye(k, dtype=np.float64)[ytr]
    W = np.linalg.pinv(Xtr.T @ Xtr + 1.0 * np.eye(Xtr.shape[1])) @ Xtr.T @ Y
    ridge_logits = Xte @ W
    ridge_pred = ridge_logits.argmax(axis=1)
    ridge = float((ridge_pred == yte).mean())

    champ = None
    if CHAMPION.is_file():
        champ_logits = mushroom_logits(Xte, CHAMPION)
        champ = float((champ_logits.argmax(axis=1) == yte).mean())

    print(f"dataset    {path}")
    print(f"gold rows  {n}")
    print(f"split      chronological  train {len(ytr)}  confirm {len(yte)}")
    print(f"families   {dict(zip(FAMILIES, counts.tolist()))}")
    print()
    print(f"  {'policy':22} {'confirm acc':>12}")
    print(f"  {'majority':22} {majority:>12.4f}")
    print(f"  {'ridge (64-dim hash)':22} {ridge:>12.4f}")
    if champ is not None:
        print(f"  {'mushroom champion':22} {champ:>12.4f}")

    out = {
        "schema": "z0evals.shadow-real-traffic.v1",
        "evidence_class": "exploratory_beta",
        "promotion_eligible": False,
        "dataset": str(path),
        "dataset_kind": "live Hermes next-action log",
        "n_gold_rows": n,
        "split": "chronological",
        "n_train": len(ytr),
        "n_confirm": len(yte),
        "families": dict(zip(FAMILIES, counts.tolist())),
        "results": {"majority": majority, "ridge": ridge},
    }
    if champ is not None:
        out["results"]["mushroom_champion"] = champ
    out["confidence_note"] = (
        "Mushroom confidence is derived as softmax over the MBON logits. The bundle has "
        "no calibrated head, so this is not a calibrated probability; it is monotone in "
        "the logit margin and is adequate to sweep coverage."
    )
    out["majority_floor"] = majority

    # An 8-way softmax rarely exceeds 0.4, so the usual 0.5+ thresholds put every row
    # below the bar and the curve came back empty. Sweep where the mass actually is.
    thrs = (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6)

    # Item 2's risk/coverage sweep, run on the bounded-choice mushroom with a
    # *derived* (softmax) confidence since the bundle has no calibrated head.
    print()
    print("  risk/coverage sweep (confidence derived from logits; not calibrated)")
    print(f"  {'policy':12} {'thr':>5} {'coverage':>9} {'success':>8} {'vs majority':>12}")
    sweeps: dict[str, list[dict]] = {}
    for name, lg, pr in (("ridge", ridge_logits, ridge_pred),
                         ("mushroom", champ_logits if champ is not None else None,
                          None)):
        if lg is None:
            continue
        conf = margin_confidence(lg)
        pr = pr if pr is not None else lg.argmax(axis=1)
        sweeps[name] = coverage_stats(pr, yte, conf, thrs)
        for row in sweeps[name]:
            d = (row["success_given_covered"] or 0) - majority
            print(f"  {name:12} {row['threshold']:>5} {row['coverage']:>9.4f} "
                  f"{str(row['success_given_covered']):>8} {d:>+12.4f}")
    out_sweeps = sweeps
    for name in a.backend or []:
        if name == "nanojev":
            from z0int.backends.base import request_from_mapping
            from z0int.backends.registry import create_backend

            be = create_backend("nanojev")
            preds, confs, lat, unparsed = [], [], [], 0
            # Confirm tail only — the split must be respected for every policy.
            with path.open(encoding="utf-8") as fh:
                rows = [json.loads(l) for l in fh if l.strip()]
            gold = [r for r in rows if r.get("family") in FAM_I and (r.get("gold") or r.get("tier") == "gold")]
            for rec in gold[cut:]:
                text = (rec.get("text") or rec.get("user") or "")
                prev = rec.get("prev") or []
                if prev:
                    text = text + " " + " ".join(str(x) for x in prev)
                req = request_from_mapping({
                    "state": {"task": "next_action", "context": text[:1200]},
                    "questions": [{
                        "id": "family", "type": "choice",
                        "instructions": "Which action family comes next?",
                        "options": [{"id": f, "description": f.replace("_", " ")} for f in FAMILIES],
                    }],
                })
                t0 = time.perf_counter()
                try:
                    res = be.evaluate(req)
                    v = res.answers[0].value if res.answers else None
                    cf = res.answers[0].confidence if res.answers else None
                except Exception:
                    v, cf = None, None
                lat.append((time.perf_counter() - t0) * 1000.0)
                if v is None:
                    unparsed += 1
                preds.append(FAM_I.get(v, -1))
                confs.append(cf if cf is not None else 0.0)
            preds = np.asarray(preds)
            acc = float((preds == yte[: len(preds)]).mean()) if len(preds) else 0.0
            print(f"  {'nanojev':22} {acc:>12.4f}   (unparsed {unparsed}, p50 {np.median(lat):.1f}ms)")
            out["results"]["nanojev"] = {
                "confirm_acc": round(acc, 4), "n": int(len(preds)),
                "unparsed": unparsed, "latency_p50_ms": round(float(np.median(lat)), 1),
            }
            out["nanojev_coverage"] = coverage_stats(
                preds, yte[: len(preds)], np.asarray(confs), thrs
            )

    out["coverage_sweeps"] = out_sweeps

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
