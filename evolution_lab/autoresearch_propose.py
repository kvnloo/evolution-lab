"""Propose next local_plasticity candidate (SearchDriver side)."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any

import numpy as np

from .engine import seed_genomes
from .schema import ExperimentGenome, genome_from_dict
from .select import FlyCandidate


def champion_from_genome(genome: ExperimentGenome, **knobs: Any) -> FlyCandidate:
    return FlyCandidate(
        genome_id=genome.id,
        genome=genome.to_dict(),
        dagger_rounds=int(knobs.get("dagger_rounds") or 0),
        plasticity_lr=float(knobs.get("plasticity_lr") or genome.training.plasticity_lr),
        plasticity_epochs=int(knobs.get("plasticity_epochs") or genome.training.plasticity_epochs),
        description=str(knobs.get("description") or "champion"),
    )


def default_champion_candidate() -> FlyCandidate:
    genome = next(g for g in seed_genomes() if g.architecture.family == "local_plasticity")
    return champion_from_genome(genome, description="default champion")


def propose_candidate(
    champion: FlyCandidate,
    rng: np.random.Generator,
    config: dict[str, Any],
) -> FlyCandidate:
    """Mutate exactly one search-space knob from the current champion."""
    space = config.get("search_space") or {}
    genome = genome_from_dict(deepcopy(champion.genome))
    dagger_rounds = int(champion.dagger_rounds)
    lr = float(champion.plasticity_lr)
    epochs = int(champion.plasticity_epochs)
    knob = str(rng.choice(["hidden", "history", "dagger_rounds", "plasticity_lr", "plasticity_epochs", "seed"]))
    if knob == "hidden":
        choices = [int(x) for x in space.get("hidden") or [128]]
        hidden = int(rng.choice([c for c in choices if c != genome.architecture.hidden] or choices))
        arch = replace(genome.architecture, hidden=hidden, family="local_plasticity")
        genome = replace(genome, architecture=arch, id=f"{genome.lineage}-ar-{int(rng.integers(1000,9999))}")
        desc = f"hidden={hidden}"
    elif knob == "history":
        choices = [int(x) for x in space.get("history") or [8]]
        history = int(rng.choice([c for c in choices if c != genome.architecture.history] or choices))
        arch = replace(genome.architecture, history=history)
        genome = replace(genome, architecture=arch, id=f"{genome.lineage}-ar-{int(rng.integers(1000,9999))}")
        desc = f"history={history}"
    elif knob == "dagger_rounds":
        choices = [int(x) for x in space.get("dagger_rounds") or [0]]
        dagger_rounds = int(rng.choice([c for c in choices if c != champion.dagger_rounds] or choices))
        desc = f"dagger_rounds={dagger_rounds}"
    elif knob == "plasticity_lr":
        choices = [float(x) for x in space.get("plasticity_lr") or [0.35]]
        lr = float(rng.choice([c for c in choices if abs(c - champion.plasticity_lr) > 1e-9] or choices))
        desc = f"plasticity_lr={lr}"
    elif knob == "plasticity_epochs":
        choices = [int(x) for x in space.get("plasticity_epochs") or [20]]
        epochs = int(rng.choice([c for c in choices if c != champion.plasticity_epochs] or choices))
        desc = f"plasticity_epochs={epochs}"
    else:
        deltas = [int(x) for x in space.get("seed_delta") or [0]]
        seed = int(genome.training.seed + rng.choice(deltas))
        genome = replace(genome, training=replace(genome.training, seed=max(0, seed)))
        desc = f"seed={genome.training.seed}"
    return FlyCandidate(
        genome_id=genome.id,
        genome=genome.to_dict(),
        dagger_rounds=dagger_rounds,
        plasticity_lr=lr,
        plasticity_epochs=epochs,
        description=desc,
    )
