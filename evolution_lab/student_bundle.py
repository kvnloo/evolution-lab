"""Persist trained local_plasticity recovery students for fast OMP load."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .models import FittedStudent, _local_plasticity_predict_fn
from .schema import ExperimentGenome
from .splits import default_splits_dir

BUNDLE_VERSION = "locked_per_step_v2"
BUNDLE_NAME = "recovery_student.npz"
META_NAME = "recovery_student.json"


def default_bundle_path(root: Path | None = None) -> Path:
    return default_splits_dir(root) / BUNDLE_NAME


def default_meta_path(root: Path | None = None) -> Path:
    return default_splits_dir(root) / META_NAME


def save_bundle(
    genome: ExperimentGenome,
    student: FittedStudent,
    path: Path | None = None,
    *,
    metrics: dict[str, Any] | None = None,
) -> Path:
    """Write KC→MBON weights and metadata for advise() fast-path."""
    if student.family != "local_plasticity":
        raise ValueError(f"bundle supports local_plasticity only, not {student.family!r}")
    dest = path or default_bundle_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dest,
        W_kc_mbon=np.asarray(student.extras["W_kc_mbon"], dtype=np.float64),
        W_pn_kc=np.asarray(student.extras["W_pn_kc"], dtype=np.float64),
        k_winners=np.asarray([int(student.extras["k_winners"])], dtype=np.int64),
    )
    meta = {
        "version": BUNDLE_VERSION,
        "genome_id": genome.id,
        "seed": genome.training.seed,
        "history": genome.architecture.history,
        "hidden": genome.architecture.hidden,
        "n_params": student.n_params,
        "encoder": student.extras.get("encoder"),
        "plastic": student.extras.get("plastic"),
        "supervision": student.extras.get("supervision"),
        "metrics": metrics or {},
    }
    (dest.parent / META_NAME).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return dest


def load_bundle(
    genome: ExperimentGenome,
    path: Path | None = None,
) -> FittedStudent:
    """Rehydrate a FittedStudent from a bundle written by save_bundle."""
    src = path or default_bundle_path()
    if not src.is_file():
        raise FileNotFoundError(f"recovery student bundle missing at {src}")
    data = np.load(src)
    W = np.asarray(data["W_kc_mbon"], dtype=np.float64)
    W_pn_kc = np.asarray(data["W_pn_kc"], dtype=np.float64)
    k_winners = int(data["k_winners"][0])
    predict_fn = _local_plasticity_predict_fn(
        genome,
        W_pn_kc=W_pn_kc,
        k_winners=k_winners,
        W=W,
    )
    extras = {
        "W_kc_mbon": W,
        "W_pn_kc": W_pn_kc,
        "k_winners": k_winners,
        "encoder": "flatten_last_and_maxpool_hermes_history_not_compound_eye",
        "plastic": "kc_to_mbon_only",
        "supervision": "per_step_prefix",
        "bundle": str(src),
    }
    return FittedStudent("local_plasticity", n_params=int(W.size), predict_fn=predict_fn, extras=extras)


def bundle_exists(root: Path | None = None) -> bool:
    return default_bundle_path(root).is_file()


def bundle_meta(root: Path | None = None) -> dict[str, Any] | None:
    path = default_meta_path(root)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
