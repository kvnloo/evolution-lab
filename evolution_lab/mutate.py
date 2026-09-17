"""Genome mutations (architecture, curriculum, experiment — not Tinker weights)."""

from __future__ import annotations

from dataclasses import replace
import numpy as np

from .schema import FAMILIES, ExperimentGenome, ROLES


def mutate(genome: ExperimentGenome, rng: np.random.Generator, *, generation: int) -> ExperimentGenome:
    arch = genome.architecture
    cur = genome.curriculum
    tr = genome.training
    hidden = arch.hidden
    if rng.random() < 0.5:
        hidden = max(4, int(hidden * rng.choice([0.5, 1.0, 2.0])))
        hidden = min(hidden, 256)
    history = arch.history
    if rng.random() < 0.3:
        history = int(rng.integers(4, 13))
    family = arch.family
    backend = genome.backend
    task = cur.task
    if family != "rule" and rng.random() < 0.15:
        choices = [f for f in FAMILIES if f not in {"rule", "hybrid"}]
        if genome.backend == "openjev":
            choices = [f for f in choices if f.startswith("jev_")]
        family = str(rng.choice(choices))
        if family.startswith("jev_"):
            backend = "openjev"
            task = "jev_synthetic"
        elif backend == "openjev":
            backend = "local_numpy"
            task = "hermes_recovery"
    drop = cur.strobe_drop
    if rng.random() < 0.4:
        drop = float(np.clip(drop + rng.normal(0, 0.08), 0.0, 0.6))
    delayed = cur.delayed_cue
    if rng.random() < 0.2 and backend != "openjev":
        delayed = bool(rng.integers(0, 2))
    l2 = float(np.clip(tr.l2 * rng.choice([0.3, 1.0, 3.0]), 1e-5, 1.0))
    jev_rank = tr.jev_rank
    jev_epochs = tr.jev_epochs
    jev_lr = tr.jev_lr
    if backend == "openjev":
        if rng.random() < 0.5:
            jev_rank = int(np.clip(jev_rank * rng.choice([0.5, 1.0, 2.0]), 16, 256))
        if rng.random() < 0.4:
            jev_epochs = int(np.clip(jev_epochs + rng.integers(-2, 3), 2, 16))
        if rng.random() < 0.3:
            jev_lr = float(np.clip(jev_lr * rng.choice([0.5, 1.0, 2.0]), 1e-4, 1e-2))
    role = genome.role
    if rng.random() < 0.2:
        role = str(rng.choice(ROLES))
    new_id = f"{genome.lineage}-g{generation}-{int(rng.integers(1000, 9999))}"
    return ExperimentGenome(
        id=new_id,
        lineage=genome.lineage,
        hypothesis=f"Mutated from {genome.id}: hidden={hidden}, strobe={drop:.2f}, family={family}",
        parents=(genome.id,),
        role=role,
        backend=backend,
        architecture=replace(arch, family=family, hidden=hidden, history=history),
        curriculum=replace(cur, task=task, strobe_drop=drop, delayed_cue=delayed),
        training=replace(
            tr,
            l2=l2,
            seed=int(rng.integers(0, 10_000)),
            jev_rank=jev_rank,
            jev_epochs=jev_epochs,
            jev_lr=jev_lr,
        ),
        evaluation=genome.evaluation,
    )


def crossover(a: ExperimentGenome, b: ExperimentGenome, rng: np.random.Generator, *, generation: int) -> ExperimentGenome:
    hidden = int(rng.choice([a.architecture.hidden, b.architecture.hidden]))
    family = a.architecture.family if rng.random() < 0.5 else b.architecture.family
    if family == "rule":
        family = a.architecture.family if a.architecture.family != "rule" else "mlp"
    drop = float(rng.choice([a.curriculum.strobe_drop, b.curriculum.strobe_drop]))
    new_id = f"x-{a.lineage}-{b.lineage}-g{generation}-{int(rng.integers(1000, 9999))}"
    return ExperimentGenome(
        id=new_id,
        lineage=f"{a.lineage}+{b.lineage}",
        hypothesis=f"Crossover {a.id} x {b.id}",
        parents=(a.id, b.id),
        role="explorer",
        backend="local_numpy",
        architecture=replace(a.architecture, family=family, hidden=hidden),
        curriculum=replace(a.curriculum, strobe_drop=drop),
        training=replace(a.training, seed=int(rng.integers(0, 10_000))),
        evaluation=a.evaluation,
    )
