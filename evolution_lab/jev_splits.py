"""Locked Jev choice splits for Track B (OpenJev Route A)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .jev_data import write_synthetic_splits
from .schema import ExperimentGenome

JEV_SPLIT_SEED = 20260917
JEV_SPLIT_NAMES = ("train", "val", "confirm", "ood")
LOCKED_JEV_TASK = "jev_synthetic"
LOCKED_JEV_COUNTS = {"train": 512, "val": 128, "confirm": 128, "ood": 128}
MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class JevSplitPaths:
    train: Path
    val: Path
    confirm: Path
    ood: Path
    root: Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_jev_dir(root: Path | None = None) -> Path:
    return (root or repo_root()) / "data" / "jev" / "synthetic"


def splits_exist(directory: Path | None = None) -> bool:
    directory = directory or default_jev_dir()
    if not (directory / MANIFEST_NAME).is_file():
        return False
    return all((directory / f"{name}.jsonl").is_file() for name in JEV_SPLIT_NAMES)


def lock_splits(directory: Path | None = None, *, seed: int = JEV_SPLIT_SEED) -> Path:
    directory = directory or default_jev_dir()
    counts = write_synthetic_splits(directory, counts=LOCKED_JEV_COUNTS, seed=seed)
    manifest: dict[str, Any] = {
        "seed": seed,
        "seed_note": "Track B lock. Subsample for promotion levels; do not re-sample L1.",
        "task": LOCKED_JEV_TASK,
        "counts": counts,
        "files": {name: f"{name}.jsonl" for name in JEV_SPLIT_NAMES},
        "format": "openjev_choice_jsonl",
    }
    (directory / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return directory


def load_manifest(directory: Path | None = None) -> dict[str, Any]:
    directory = directory or default_jev_dir()
    return json.loads((directory / MANIFEST_NAME).read_text())


def _materialize_subset(source: Path, dest: Path, n: int) -> Path:
    from .jev_data import subset_jsonl

    if dest.exists() and dest.stat().st_size > 0:
        return dest
    return subset_jsonl(source, dest, n)


def resolve_jev_splits(
    genome: ExperimentGenome,
    *,
    n_train: int,
    n_val: int,
    n_confirm: int,
    n_ood: int,
    work_dir: Path,
    splits_dir: Path | None = None,
) -> JevSplitPaths:
    """Return JSONL paths sized for the promotion level."""
    task = genome.curriculum.task
    root = work_dir / "jev_splits" / task
    root.mkdir(parents=True, exist_ok=True)

    if task == "jev_synthetic":
        locked = splits_dir or default_jev_dir()
        if not splits_exist(locked):
            raise FileNotFoundError(
                f"locked Jev splits missing under {locked}; run `python -m evolution_lab lock-jev-splits`"
            )
        return JevSplitPaths(
            train=_materialize_subset(locked / "train.jsonl", root / "train.jsonl", n_train),
            val=_materialize_subset(locked / "val.jsonl", root / "val.jsonl", n_val),
            confirm=_materialize_subset(locked / "confirm.jsonl", root / "confirm.jsonl", n_confirm),
            ood=_materialize_subset(locked / "ood.jsonl", root / "ood.jsonl", n_ood),
            root=root,
        )

    if task == "hermes_as_jev":
        from .jev_bridge import lock_hermes_as_jev

        level_counts = {
            "train": n_train,
            "val": n_val,
            "confirm": n_confirm,
            "ood": n_ood,
        }
        bridge_root = lock_hermes_as_jev(root, counts=level_counts)
        return JevSplitPaths(
            train=bridge_root / "train.jsonl",
            val=bridge_root / "val.jsonl",
            confirm=bridge_root / "confirm.jsonl",
            ood=bridge_root / "ood.jsonl",
            root=bridge_root,
        )

    raise ValueError(f"unsupported Jev curriculum task {task!r}")


def genome_matches_locked_jev(genome: ExperimentGenome) -> bool:
    return genome.curriculum.task in {"jev_synthetic", "hermes_as_jev"} and not genome.curriculum.include_secrets
