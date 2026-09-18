"""Session-group train/dev/sealed/future splits.

Per Frontier KB harness-evolver: credit only on sealed battery the search
cannot see. Adjacent tool rows are NOT independent tasks — split by session.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class SessionSplit:
    train_sessions: list[str]
    dev_sessions: list[str]
    sealed_sessions: list[str]
    future_sessions: list[str]
    n_train_events: int
    n_dev_events: int
    n_sealed_events: int
    n_future_events: int
    schema: str = "flyforge.session_split.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def bucket_of(self, session: str) -> str:
        if session in self.train_sessions:
            return "train"
        if session in self.dev_sessions:
            return "dev"
        if session in self.sealed_sessions:
            return "sealed"
        if session in self.future_sessions:
            return "future"
        return "unknown"


def _session_order(rows: Iterable[dict[str, Any]]) -> tuple[list[str], dict[str, int], dict[str, float]]:
    counts: dict[str, int] = defaultdict(int)
    tmax: dict[str, float] = defaultdict(float)
    for r in rows:
        s = str(r.get("session") or "")
        if not s:
            continue
        counts[s] += 1
        t = float(r.get("ts") or 0.0)
        if t >= tmax[s]:
            tmax[s] = t
    ordered = sorted(tmax.keys(), key=lambda s: (tmax[s], s))
    return ordered, dict(counts), dict(tmax)


def make_session_split(
    rows: list[dict[str, Any]],
    *,
    sealed_frac: float = 0.10,
    dev_frac: float = 0.10,
    future_frac: float = 0.0,
) -> SessionSplit:
    """Chronological session split. Oldest → train; newest tails → sealed/future."""
    ordered, counts, _tmax = _session_order(rows)
    n = len(ordered)
    if n == 0:
        return SessionSplit([], [], [], [], 0, 0, 0, 0)
    n_seal = max(1, int(round(n * sealed_frac))) if sealed_frac > 0 else 0
    n_fut = max(0, int(round(n * future_frac))) if future_frac > 0 else 0
    n_dev = max(1, int(round(n * dev_frac))) if dev_frac > 0 else 0
    # ensure non-overlap and leave train non-empty when possible
    while n_seal + n_fut + n_dev >= n and n_dev > 0:
        n_dev -= 1
    while n_seal + n_fut + n_dev >= n and n_fut > 0:
        n_fut -= 1
    while n_seal + n_fut + n_dev >= n and n_seal > 1:
        n_seal -= 1
    future = ordered[n - n_fut :] if n_fut else []
    sealed = ordered[n - n_fut - n_seal : n - n_fut] if n_seal else []
    dev = ordered[n - n_fut - n_seal - n_dev : n - n_fut - n_seal] if n_dev else []
    train = ordered[: n - n_fut - n_seal - n_dev]

    def _n(ss: list[str]) -> int:
        return sum(counts.get(s, 0) for s in ss)

    return SessionSplit(
        train_sessions=train,
        dev_sessions=dev,
        sealed_sessions=sealed,
        future_sessions=future,
        n_train_events=_n(train),
        n_dev_events=_n(dev),
        n_sealed_events=_n(sealed),
        n_future_events=_n(future),
    )


def load_episode_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def mask_for_bucket(
    sessions: list[str],
    split: SessionSplit,
    bucket: str,
) -> list[bool]:
    want = {
        "train": set(split.train_sessions),
        "dev": set(split.dev_sessions),
        "sealed": set(split.sealed_sessions),
        "future": set(split.future_sessions),
    }[bucket]
    return [s in want for s in sessions]
