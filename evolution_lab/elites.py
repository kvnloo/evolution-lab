"""MAP-Elites archive: best individual per niche."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .schema import PARAM_BUCKETS, param_bucket


def niche_key(family: str, n_params: int, learning_mode: str) -> str:
    return f"{family}|{param_bucket(n_params)}|{learning_mode}"


@dataclass
class Elite:
    experiment_id: str
    fitness: float
    record: dict[str, Any]


@dataclass
class MapElites:
    cells: dict[str, Elite] = field(default_factory=dict)

    def offer(self, key: str, experiment_id: str, fitness: float, record: dict[str, Any]) -> bool:
        cur = self.cells.get(key)
        if cur is None or fitness > cur.fitness:
            self.cells[key] = Elite(experiment_id, fitness, record)
            return True
        return False

    def occupied(self) -> int:
        return len(self.cells)

    def coverage(self, families: list[str], modes: list[str]) -> float:
        total = max(len(families) * len(PARAM_BUCKETS) * len(modes), 1)
        return self.occupied() / total
