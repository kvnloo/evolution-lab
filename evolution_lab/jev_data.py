"""Minimal Jev JSONL builders (stdlib). Mirrors openjev synthetic rows without torch."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

COLOURS = ("amber", "azure", "crimson", "gold", "ivory", "jade", "obsidian", "silver")
ANIMALS = ("badger", "crane", "dragon", "falcon", "heron", "lynx", "otter", "sparrow")


@dataclass(frozen=True)
class ChoiceRow:
    context: str
    options: tuple[str, ...]
    label: int

    def to_json(self) -> dict:
        return {"context": self.context, "options": list(self.options), "label": self.label}


def synthetic_row(seed: int) -> ChoiceRow:
    rng = random.Random(seed)
    target = f"{rng.choice(COLOURS)} {rng.choice(ANIMALS)}"
    options = {target}
    while len(options) < 3:
        options.add(f"{rng.choice(COLOURS)} {rng.choice(ANIMALS)}")
    options_list = list(options)
    rng.shuffle(options_list)
    notes = " ".join(rng.choice(("north", "south", "east", "west")) for _ in range(8))
    context = f"Choose the exact badge {target}. Notes: {notes}. Badge: {target}."
    return ChoiceRow(context, tuple(options_list), options_list.index(target))


def write_jsonl(rows: list[ChoiceRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_json()) + "\n")


def write_synthetic_splits(
    output: Path,
    *,
    counts: dict[str, int],
    seed: int,
) -> dict[str, int]:
    """Write train/val/confirm/ood JSONL splits with deterministic synthetic rows."""
    output.mkdir(parents=True, exist_ok=True)
    written: dict[str, int] = {}
    offset = 0
    for split, size in counts.items():
        rows = [synthetic_row(seed + offset + index * 104729) for index in range(size)]
        write_jsonl(rows, output / f"{split}.jsonl")
        written[split] = size
        offset += size * 104729
    return written


def subset_jsonl(source: Path, dest: Path, n: int) -> Path:
    """Copy the first n non-empty lines (deterministic promotion subsample)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with source.open(encoding="utf-8") as inp, dest.open("w", encoding="utf-8") as out:
        for line in inp:
            if not line.strip():
                continue
            out.write(line if line.endswith("\n") else line + "\n")
            kept += 1
            if kept >= n:
                break
    if kept < n:
        raise ValueError(f"{source} has only {kept} rows; need {n}")
    return dest
