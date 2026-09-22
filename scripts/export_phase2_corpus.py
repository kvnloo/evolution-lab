#!/usr/bin/env python3
"""Export the frozen Phase 2 corpus from `runs/` (gitignored) into a tracked path.

`runs/` is gitignored on purpose — it holds run outputs. But the Phase 2 corpus is
not a scratch output: it is the artifact that the next block trains against, and it
was sitting only on one disk with no upstream branch. This copies it somewhere
durable and writes a manifest that pins every byte.

Nothing is recomputed here. The episodes and splits are copied verbatim and hashed,
so the manifest certifies the artifacts that were actually produced rather than a
re-derivation of them. The digest that matters — `split_digest` — is carried through
unchanged.

Usage:
  python scripts/export_phase2_corpus.py --run p1b-20260921T1430Z
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FILES = ("episodes.jsonl", "splits.json", "teacher_selection.json")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="p1b-20260921T1430Z")
    ap.add_argument("--src", default=None, help="default: runs/phase2/<run>")
    ap.add_argument("--dest", default="corpus/phase2")
    a = ap.parse_args()

    src = Path(a.src) if a.src else ROOT / "runs" / "phase2" / a.run
    if not src.is_dir():
        print(f"no such corpus dir: {src}", file=sys.stderr)
        return 2

    dest = ROOT / a.dest / a.run
    dest.mkdir(parents=True, exist_ok=True)

    digests: dict[str, str] = {}
    for name in FILES:
        s = src / name
        if not s.is_file():
            print(f"missing {s}", file=sys.stderr)
            return 2
        shutil.copy2(s, dest / name)
        digests[name] = sha256_file(dest / name)

    splits = json.loads((dest / "splits.json").read_text(encoding="utf-8"))
    episodes = [
        json.loads(line)
        for line in (dest / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    import collections

    arms = collections.Counter(e["episode_id"].split(":")[1] for e in episodes)
    label_sources = collections.Counter(e.get("label_source") for e in episodes)
    families = collections.Counter(e.get("task_family") for e in episodes)
    budget_units_present = sum(1 for e in episodes if "budget_units" in json.dumps(e))

    manifest = {
        "schema": "evolution-lab.phase2-corpus.v1",
        "run_id": a.run,
        "evidence_class": "corpus",
        "promotion_eligible": False,
        "trained": False,
        "trained_note": (
            "This block writes and seals the corpus and stops. No model was trained on it, "
            "which is deliberate: the splits must be frozen before anything is fitted."
        ),
        "episodes": len(episodes),
        "tasks": splits.get("n_tasks"),
        "arms": dict(sorted(arms.items())),
        "n_arms": len(arms),
        "label_sources": dict(sorted(label_sources.items())),
        "families": dict(sorted(families.items())),
        "new_feature": "budget_units",
        "budget_units_present_in_episodes": budget_units_present,
        # Carried through unchanged from the producer. This is the frozen identity
        # of the split; recomputing it here would defeat the point.
        "split_digest": splits.get("split_digest"),
        "bucket_counts": splits.get("bucket_counts"),
        "proportions": splits.get("proportions"),
        "ood_families": (splits.get("ood") or {}).get("families")
        or sorted({e["task_family"] for e in episodes if ":" in e["episode_id"]}),
        "sealed": {
            "task_ids": (splits.get("sealed") or {}).get("sealed_task_ids"),
            "requested_episodes": (splits.get("sealed") or {}).get("requested_episodes"),
            "overshoot_reason": (splits.get("sealed") or {}).get("overshoot_reason"),
        },
        "frozen_at": splits.get("frozen_at"),
        "sha256": digests,
        "provenance": {
            "compiler_revision": next(
                (e.get("source", {}).get("compiler_revision") for e in episodes), None
            ),
            "fixture_revision": next(
                (e.get("source", {}).get("fixture_revision") for e in episodes), None
            ),
            "external_inference": sorted(
                {bool(e.get("source", {}).get("external_inference")) for e in episodes}
            ),
            "source": "z0intelligence Phase 1B observations p1b-20260921T1430Z",
        },
    }

    (dest / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"exported {len(episodes)} episodes -> {dest}")
    print(f"  arms            {len(arms)}: {', '.join(sorted(arms))}")
    print(f"  label_sources   {dict(label_sources)}")
    print(f"  budget_units    present in {budget_units_present}/{len(episodes)} episodes")
    print(f"  split_digest    {manifest['split_digest']}")
    print(f"  bucket_counts   {manifest['bucket_counts']}")
    print(f"  sealed task     {manifest['sealed']['task_ids']}")
    for n, d in digests.items():
        print(f"  sha256 {n:24} {d[:16]}…")
    return 0


if __name__ == "__main__":
    sys.exit(main())
