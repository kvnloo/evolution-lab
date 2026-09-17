"""Hermes recovery episodes → OpenJev choice rows (distill / Track B bridge)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .jev_data import ChoiceRow, write_jsonl
from .schema import ACTIONS
from .splits import SPLIT_SEED, sample_canonical_task
from .task import Episode


def frame_to_context(frame: np.ndarray) -> str:
    """Render the P0 observation vector as a compact recovery brief."""
    names = (
        "sandbox_alive",
        "retry_norm",
        "budget",
        "elapsed",
        "transient",
        "hard",
        "unfamiliar",
        "policy_hit",
        "already_ok",
        "cue",
    )
    base = frame[: len(names)]
    parts = [f"{name}={float(value):.3f}" for name, value in zip(names, base)]
    if frame.shape[0] > len(names):
        hist = frame[len(names) :]
        last = int(np.argmax(hist)) if hist.size else 0
        parts.append(f"last_action={ACTIONS[last]}")
    return "Hermes recovery state: " + ", ".join(parts) + "."


def episode_to_row(episode: Episode) -> ChoiceRow:
    context = frame_to_context(episode.frames[-1])
    options = tuple(ACTIONS)
    label = int(episode.labels[-1])
    if label < 0 or label >= len(options):
        raise ValueError(f"invalid action label {label}")
    return ChoiceRow(context, options, label)


def episodes_to_rows(episodes: list[Episode]) -> list[ChoiceRow]:
    return [episode_to_row(ep) for ep in episodes]


def write_episode_split(episodes: list[Episode], path: Path) -> None:
    write_jsonl(episodes_to_rows(episodes), path)


def lock_hermes_as_jev(work_root: Path, *, counts: dict[str, int]) -> Path:
    """Materialize Hermes P0 episodes as Jev JSONL for the requested counts."""
    data = sample_canonical_task(SPLIT_SEED)
    out = work_root / "hermes_bridge"
    out.mkdir(parents=True, exist_ok=True)
    mapping = {
        "train": data.train[: counts["train"]],
        "val": data.val[: counts["val"]],
        "confirm": data.confirm[: counts["confirm"]],
        "ood": data.ood[: counts["ood"]],
    }
    manifest = {"source": "hermes_recovery_p0_lock", "seed": SPLIT_SEED, "counts": counts}
    for name, episodes in mapping.items():
        write_episode_split(episodes, out / f"{name}.jsonl")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return out
