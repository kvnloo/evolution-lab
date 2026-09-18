"""ABAB two-timescale wrapper around autoresearch (A) vs bench (B).

A chooses the investigation (1-knob session, parallel sweep, or C-test).
B executes via existing public APIs. The frozen judge stays bench.py/select.py.

World JSON: runs/autoresearch/league/abab.json
Engine: vendored evolution_lab.abab_state (abab-meta-research skill).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .abab_state import (
    Evidence,
    Hypothesis,
    World,
    apply_mutation,
    load,
    next_a,
    save,
    should_slow,
    stop_reason,
    wave_heartbeat,
)


def world_path(league: Path) -> Path:
    return league / "abab.json"


def seed_fly_world() -> World:
    """Start from measured 2026-09-17 cycle + ABAB-5, do not redo P0/P1."""
    return World(
        objective="Evolve production local_plasticity recovery kernel under PRODUCT_TARGET gates without editing the frozen judge.",
        constraints=[
            "Do not edit bench.py, select.py gates, data/p0 splits, observe.py.",
            "history locked at 8.",
            "Hosted joules unknown.",
        ],
        metrics=["closed_loop_reward", "latency_score"],
        baseline="fly-4080a10a hidden=96 dagger=3 latency_score=5.923 closed_loop=1.0",
        mechanisms={
            "best": "KC→MBON; width floor hidden>=96 for CL=1.0; 1-knob plateaued.",
        },
        hypotheses=[
            Hypothesis(
                id="H1",
                claim="1-knob search is exhausted; remaining keeps need sweep or joint knobs",
                mechanism="single-knob walk cannot move seed×lr×k_winners jointly",
                predictions=["idle sessions accumulate", "sweep may still lose to serial champion"],
                falsifiers=["1-knob keep at eps=0.01"],
                eig=0.55,
                impact=0.6,
                decision_change=0.4,
                transferability=0.5,
                cost=0.7,
                status="open",
            ),
            Hypothesis(
                id="H2",
                claim="Kenyon-cell width floor is hidden>=96 for closed_loop=1.0",
                mechanism="hidden<=64 never CL=1.0; DAgger does not lift it",
                predictions=["hidden=64 C-tests fail CL"],
                falsifiers=["hidden=64 dagger=3 CL=1.0"],
                eig=0.2,
                impact=0.9,
                decision_change=0.2,
                transferability=0.4,
                cost=0.4,
                status="promoted",
            ),
            Hypothesis(
                id="H4",
                claim="k_winners as SearchDriver knob is the next capability",
                mechanism="k_winners was hardcoded 10% n_kc; sparse values fail CL",
                predictions=["k_winners in {5,8,12} fail CL", "some setting beats 5.923 with gates"],
                falsifiers=["k_winners grid none beat serial champion with gates"],
                eig=0.65,
                impact=0.7,
                decision_change=0.5,
                transferability=0.6,
                cost=0.8,
                status="open",
                rival_of="H1",
            ),
        ],
        wave_budget=10_000,
        deep=False,
        loop="fast",
    )


def load_or_seed(league: Path) -> World:
    path = world_path(league)
    if path.is_file():
        return load(path)
    w = seed_fly_world()
    save(w, path)
    return w


def mutate_recipe(world: World):
    """A mutates the dataset recipe, never the frozen judge."""
    from .schema import DataRecipe

    sizes = (32, 64, 128, 256, 512)
    return DataRecipe(
        n_train=sizes[world.wave % len(sizes)],
        gold_only=bool(world.wave % 2),
        source="hermes_recovery" if world.wave % 3 else "next_action",
        confirm_frac=0.2,
    )


def choose_b_action(world: World, *, idle: int, idle_limit: int) -> str:
    """Wave 0 = frontier ideas pack. Idle plateau mutates recipe then sweeps."""
    if world.wave == 0:
        return "ideas"
    if idle >= idle_limit or world.loop == "slow":
        return "mutate_recipe" if world.wave % 2 == 0 else "sweep"
    return "autoresearch"


def run_ideas_pack() -> dict[str, Any]:
    """First B: Jev skill distill + parallel k_winners/2-knob sweep. Frozen judge."""
    from .jev_distill import run_jev_distill
    from .sweep import run_sweep

    jev = run_jev_distill()
    sweep = run_sweep()
    return {
        "ideas": ["jev_distill", "sweep"],
        "jev_distill": jev,
        "sweep": sweep,
        "keeps": list(sweep.get("keeps") or []),
    }


def record_b(

    world: World,
    *,
    action: str,
    keeps: int,
    note: str,
) -> World:
    if keeps > 0:
        return apply_mutation(
            world,
            "ADD",
            {
                "evidence": Evidence(
                    id=f"E{world.wave + 1}",
                    claim=f"{action} produced {keeps} keep(s). {note}",
                    source="cycle B executor",
                    source_class="measurement",
                    scope="production kernel",
                    confidence=0.7,
                ),
                "event": "graph_change",
            },
        )
    w = apply_mutation(world, "NO_UPDATE", {"event": "plateau", "note": note})
    # Forever cycle must not die on 3 NO_UPDATEs — that is the signal to sweep.
    if w.stopped and w.stop_reason == "repeated_no_update":
        w.stopped = False
        w.stop_reason = ""
        w.no_update_streak = 0
        w.loop = "slow"
    return w


def heartbeat(world: World, action: str, keeps: int) -> dict[str, str]:
    return wave_heartbeat(
        world,
        changed=f"B={action} keeps={keeps}",
        why="ABAB A chose executor; B used frozen judge",
        failed="editing bench/select",
        next_move=action,
    )
