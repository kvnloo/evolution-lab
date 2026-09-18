"""ABAB around GPU next-action: A mutates DataRecipe, B trains a population.

Frozen judge = ridge + majority on the confirm split. GPU acc alone never keeps.
World: runs/gpu-evolve/abab.json (does not touch the recovery league champion).
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .abab_meta import heartbeat, record_b, seed_fly_world
from .abab_state import save as save_abab
from .next_action_gpu import run_next_action_gpu
from .schema import DataRecipe

ROOT = Path(__file__).resolve().parents[1]
LEAGUE = ROOT / "runs" / "gpu-evolve"
MARGIN = 0.01


def _plan(wave: int) -> DataRecipe:
    """One knob per wave. n_train=0 means the full time-split train set."""
    grid = (
        DataRecipe(n_train=2048, gold_only=False, source="next_action"),
        DataRecipe(n_train=2048, gold_only=True, source="next_action"),
        DataRecipe(n_train=8192, gold_only=False, source="next_action"),
        DataRecipe(n_train=0, gold_only=True, source="next_action"),
        DataRecipe(n_train=16384, gold_only=False, source="next_action"),
        DataRecipe(n_train=0, gold_only=False, source="next_action"),
        DataRecipe(n_train=8192, gold_only=True, source="next_action"),
        DataRecipe(n_train=4096, gold_only=False, source="next_action", confirm_frac=0.1),
    )
    return grid[wave % len(grid)]


def _keep(report: dict[str, Any]) -> bool:
    gpu = float(report["gpu_best_confirm_acc"])
    ridge = float(report["ridge_confirm_acc"])
    maj = float(report["majority_confirm_acc"])
    return gpu > ridge + MARGIN and gpu > maj + MARGIN


def run_gpu_abab(*, max_waves: int = 8, n_pop: int = 256, epochs: int = 6, idle_limit: int = 3) -> dict[str, Any]:
    LEAGUE.mkdir(parents=True, exist_ok=True)
    world = seed_fly_world()
    world.objective = "GPU next-action: mutate DataRecipe; ridge+majority frozen."
    idle = 0
    waves: list[dict[str, Any]] = []
    for wave in range(max_waves):
        world.wave = wave
        recipe = _plan(wave)
        report = run_next_action_gpu(recipe=recipe, n_pop=n_pop, epochs=epochs)
        kept = _keep(report)
        nkeep = 1 if kept else 0
        world = record_b(
            world,
            action="mutate_recipe",
            keeps=nkeep,
            note=json.dumps(
                {
                    "recipe": asdict(recipe),
                    "gpu": report["gpu_best_confirm_acc"],
                    "ridge": report["ridge_confirm_acc"],
                    "majority": report["majority_confirm_acc"],
                    "n_train": report["n_train"],
                    "n_confirm": report["n_confirm"],
                }
            ),
        )
        save_abab(world, LEAGUE / "abab.json")
        row = {
            "wave": wave,
            "keep": kept,
            "recipe": asdict(recipe),
            "gpu": report["gpu_best_confirm_acc"],
            "ridge": report["ridge_confirm_acc"],
            "majority": report["majority_confirm_acc"],
            "gpu_beats_ridge": report["gpu_beats_ridge"],
            "n_train": report["n_train"],
            "n_confirm": report["n_confirm"],
            "gpu_filter_s": report["gpu_filter_s"],
            "heartbeat": heartbeat(world, "mutate_recipe", nkeep),
        }
        waves.append(row)
        print(json.dumps(row, indent=2), flush=True)
        if kept:
            idle = 0
        else:
            idle += 1
            if idle >= idle_limit:
                break
    summary = {
        "schema": "flyforge.gpu_abab.v1",
        "waves": waves,
        "keeps": sum(1 for w in waves if w["keep"]),
        "stopped": "idle_plateau" if idle >= idle_limit else "max_waves",
        "world": str(LEAGUE / "abab.json"),
    }
    (LEAGUE / "abab_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
