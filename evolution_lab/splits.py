"""Locked delayed-cue Hermes recovery splits (train / val / confirm / ood).

P0 arrays are bound to SPLIT_SEED = 20260912 and written under data/p0/ so L1
is not re-sampled from a drifting numpy RNG. load_splits returns TaskData of
Episode objects, the same types evolution_lab.task uses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .task import Episode, TaskData, make_split, n_features

SPLIT_SEED = 20260912
SPLIT_NAMES = ("train", "val", "confirm", "ood")
LOCKED_TASK = "hermes_recovery"
LOCKED_HISTORY = 8
LOCKED_DELAYED_CUE = True
LOCKED_COUNTS = {"train": 192, "val": 48, "confirm": 48, "ood": 32}
MANIFEST_NAME = "manifest.json"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_splits_dir(root: Path | None = None) -> Path:
    return (root or repo_root()) / "data" / "p0"


def splits_exist(directory: Path | None = None) -> bool:
    directory = directory or default_splits_dir()
    if not (directory / MANIFEST_NAME).is_file():
        return False
    return all((directory / f"{name}.npz").is_file() for name in SPLIT_NAMES)


def sample_canonical_task(seed: int = SPLIT_SEED) -> TaskData:
    """Draw the delayed-cue P0 task from SPLIT_SEED. Used only to lock arrays."""
    rng = np.random.default_rng(seed)
    T = LOCKED_HISTORY
    delayed = LOCKED_DELAYED_CUE
    train = make_split(rng, LOCKED_COUNTS["train"], T=T, delayed_cue=delayed, ood=False)
    val = make_split(rng, LOCKED_COUNTS["val"], T=T, delayed_cue=delayed, ood=False)
    confirm = make_split(rng, LOCKED_COUNTS["confirm"], T=T, delayed_cue=delayed, ood=False)
    ood = make_split(rng, LOCKED_COUNTS["ood"], T=T, delayed_cue=delayed, ood=True)
    return TaskData(train, val, confirm, ood)


def _episodes_to_arrays(episodes: list[Episode]) -> dict[str, np.ndarray]:
    return {
        "frames": np.stack([ep.frames for ep in episodes]),
        "labels": np.stack([ep.labels for ep in episodes]),
        "env": np.array([ep.env for ep in episodes], dtype="U8"),
        "secret": np.array([ep.secret for ep in episodes], dtype=bool),
        "seed": np.int64(SPLIT_SEED),
    }


def _arrays_to_episodes(payload: dict[str, np.ndarray]) -> list[Episode]:
    frames = np.asarray(payload["frames"])
    labels = np.asarray(payload["labels"])
    env = np.asarray(payload["env"])
    secret = np.asarray(payload["secret"])
    out: list[Episode] = []
    for i in range(frames.shape[0]):
        out.append(
            Episode(
                frames=np.asarray(frames[i], dtype=np.float64),
                labels=np.asarray(labels[i], dtype=np.int64),
                env=str(env[i]),
                secret=bool(secret[i]),
            )
        )
    return out


def write_splits(data: TaskData, directory: Path, *, seed: int = SPLIT_SEED) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    counts = {
        "train": len(data.train),
        "val": len(data.val),
        "confirm": len(data.confirm),
        "ood": len(data.ood),
    }
    for name in SPLIT_NAMES:
        episodes = getattr(data, name)
        np.savez_compressed(directory / f"{name}.npz", **_episodes_to_arrays(episodes))
    T = int(data.train[0].frames.shape[0]) if data.train else LOCKED_HISTORY
    F = int(data.train[0].frames.shape[1]) if data.train else n_features()
    manifest: dict[str, Any] = {
        "seed": int(seed),
        "seed_note": "P0 lock seed 20260912. Load these arrays; do not re-sample L1.",
        "task": LOCKED_TASK,
        "delayed_cue": LOCKED_DELAYED_CUE,
        "history": T,
        "n_features": F,
        "counts": counts,
        "format": "npz-v1",
        "files": {name: f"{name}.npz" for name in SPLIT_NAMES},
    }
    (directory / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    return directory


def load_manifest(directory: Path | None = None) -> dict[str, Any]:
    directory = directory or default_splits_dir()
    path = directory / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"locked splits manifest missing: {path}")
    return json.loads(path.read_text())


def load_splits(directory: Path | None = None) -> TaskData:
    """Load locked train/val/confirm/ood as TaskData / Episode."""
    directory = directory or default_splits_dir()
    if not splits_exist(directory):
        raise FileNotFoundError(f"locked splits missing under {directory}")
    loaded: dict[str, list[Episode]] = {}
    for name in SPLIT_NAMES:
        with np.load(directory / f"{name}.npz", allow_pickle=False) as z:
            payload = {k: z[k] for k in z.files}
        loaded[name] = _arrays_to_episodes(payload)
    return TaskData(loaded["train"], loaded["val"], loaded["confirm"], loaded["ood"])


def slice_task(
    data: TaskData,
    *,
    n_train: int,
    n_val: int,
    n_confirm: int,
    n_ood: int,
) -> TaskData:
    if n_train > len(data.train) or n_val > len(data.val) or n_confirm > len(data.confirm) or n_ood > len(data.ood):
        raise ValueError("requested split larger than locked P0 arrays")
    return TaskData(
        data.train[:n_train],
        data.val[:n_val],
        data.confirm[:n_confirm],
        data.ood[:n_ood],
    )


def compatible_with_lock(genome: Any, manifest: dict[str, Any] | None = None) -> bool:
    """Locked arrays are the delayed-cue T=8 contract, not every genome."""
    meta = manifest or {}
    history = int(meta.get("history", LOCKED_HISTORY))
    delayed = bool(meta.get("delayed_cue", LOCKED_DELAYED_CUE))
    return genome.architecture.history == history and bool(genome.curriculum.delayed_cue) == delayed


def try_load_for_genome(
    genome: Any,
    *,
    n_train: int,
    n_val: int,
    n_confirm: int,
    n_ood: int,
    splits_dir: Path | None = None,
) -> TaskData | None:
    directory = splits_dir or default_splits_dir()
    if not splits_exist(directory):
        return None
    manifest = load_manifest(directory)
    if not compatible_with_lock(genome, manifest):
        return None
    counts = manifest.get("counts") or LOCKED_COUNTS
    if n_train > int(counts["train"]) or n_val > int(counts["val"]) or n_confirm > int(counts["confirm"]) or n_ood > int(counts["ood"]):
        return None
    return slice_task(
        load_splits(directory),
        n_train=n_train,
        n_val=n_val,
        n_confirm=n_confirm,
        n_ood=n_ood,
    )


def lock_splits(directory: Path | None = None, *, seed: int = SPLIT_SEED) -> Path:
    """Generate the delayed-cue task from seed and persist under data/p0/."""
    dest = directory or default_splits_dir()
    data = sample_canonical_task(seed=seed)
    return write_splits(data, dest, seed=seed)
