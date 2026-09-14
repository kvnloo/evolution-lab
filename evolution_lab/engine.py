"""Run genomes, score the archive, evolve a generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import time
import numpy as np

from .archive import Archive
from .elites import MapElites, niche_key
from .models import fit_student
from .pareto import Point, delta_hypervolume, hypervolume_2d, pareto_front
from .promote import LEVELS, epistemic, promote
from .schema import ExperimentGenome, GenomeError
from .task import TaskData, build_task


REF_COST = 1.0


@dataclass
class RunConfig:
    level: int = 1
    run_dir: Path = Path("runs/default")


def _success(pred: np.ndarray, episodes) -> float:
    y = np.stack([ep.labels[-1] for ep in episodes])
    return float((pred == y).mean()) if len(y) else 0.0


def _cost(n_params: int, latency_s: float) -> float:
    """Unitless cost in [0, 1) for HV. Not joules. Hosted energy stays unknown."""
    param_term = np.tanh(n_params / 50_000.0)
    time_term = np.tanh(latency_s / 2.0)
    return float(0.7 * param_term + 0.3 * time_term)


def evaluate_genome(genome: ExperimentGenome, data: TaskData) -> dict[str, Any]:
    if genome.backend in {"tinker_sft", "tinker_rl"}:
        raise GenomeError(f"{genome.backend} is declared, not wired — refusing to fake SFT")
    if genome.backend == "fly_sim":
        raise GenomeError("fly_sim backend is not wired; fly-wirehead stays a pinout demo")
    t0 = time.perf_counter()
    student = fit_student(genome, data.train)
    fit_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    pred_c = student.predict_fn(data.confirm)
    pred_o = student.predict_fn(data.ood)
    pred_v = student.predict_fn(data.val)
    latency = time.perf_counter() - t1
    success = _success(pred_c, data.confirm)
    ood = _success(pred_o, data.ood)
    val = _success(pred_v, data.val)
    # policy violations: none on this sanitized task by construction
    violations = 0.0
    n_params = student.n_params
    cost = _cost(n_params, fit_s + latency)
    joules_per_success = None  # unknown on this CPU; proxy only
    return {
        "success_rate": success,
        "val_success": val,
        "ood_score": ood,
        "params": n_params,
        "latency_s": latency,
        "fit_s": fit_s,
        "cost": cost,
        "violations": violations,
        "joules_per_success": joules_per_success,
        "joules_unknown": True,
    }


def points_from_archive(rows: list[dict[str, Any]]) -> list[Point]:
    pts = []
    for r in rows:
        m = r.get("metrics") or {}
        if r.get("status") != "ok":
            continue
        pts.append(
            Point(
                experiment_id=r["experiment_id"],
                success=float(m.get("success_rate", 0)),
                cost=float(m.get("cost", REF_COST)),
                ood=float(m.get("ood_score", 0)),
                params=float(m.get("params", 0)),
                violations=float(m.get("violations", 0)),
            )
        )
    return pts


def run_one(
    genome: ExperimentGenome,
    archive: Archive,
    elites: MapElites,
    *,
    level: int,
    prior: list[dict[str, Any]],
) -> dict[str, Any]:
    genome.validate()
    spec = LEVELS[min(level, 2)]
    data = build_task(
        genome,
        n_train=spec["n_train"],
        n_val=spec["n_val"],
        n_confirm=spec["n_confirm"],
        n_ood=spec["n_ood"],
    )
    metrics_seeds = []
    try:
        for s in range(spec["seeds"]):
            g = genome.with_seed(genome.training.seed + s)
            metrics_seeds.append(evaluate_genome(g, data))
    except GenomeError as e:
        rec = {
            "experiment_id": genome.id,
            "status": "failed",
            "level": level,
            "error": str(e),
            "genome": genome.to_dict(),
            "metrics": {},
        }
        archive.append(rec)
        return rec

    keys = ("success_rate", "val_success", "ood_score", "params", "latency_s", "fit_s", "cost", "violations")
    metrics = {k: float(np.mean([m[k] for m in metrics_seeds])) for k in keys}
    metrics["joules_per_success"] = None
    metrics["joules_unknown"] = True
    metrics["seeds"] = spec["seeds"]

    pts_before = points_from_archive(prior)
    pt = Point(
        genome.id,
        metrics["success_rate"],
        metrics["cost"],
        metrics["ood_score"],
        metrics["params"],
        metrics["violations"],
    )
    dhv = delta_hypervolume(pts_before, pt, ref_cost=REF_COST)
    key = niche_key(genome.architecture.family, int(metrics["params"]), genome.training.algorithm)
    fitness = metrics["success_rate"] - 0.15 * metrics["cost"] + 0.1 * metrics["ood_score"]
    new_niche = elites.offer(key, genome.id, fitness, {"metrics": metrics})
    killed = not promote(level, metrics["success_rate"], metrics["violations"])
    rec = {
        "experiment_id": genome.id,
        "status": "killed" if killed else "ok",
        "level": level,
        "genome": genome.to_dict(),
        "metrics": metrics,
        "delta_hv": dhv,
        "niche": key,
        "new_niche": bool(new_niche),
        "epistemic": epistemic(dhv, replicated=False, new_niche=bool(new_niche)),
        "role": genome.role,
    }
    archive.append(rec)
    return rec


def seed_genomes() -> list[ExperimentGenome]:
    from .schema import Architecture, Curriculum, Training

    delayed = Curriculum(delayed_cue=True)

    return [
        ExperimentGenome(
            id="teacher-rule-000",
            lineage="teacher",
            hypothesis="Reference recovery policy; not a learned student.",
            role="replicator",
            architecture=Architecture(family="rule", hidden=1, history=8),
            training=Training(algorithm="fixed", seed=0),
            curriculum=delayed,
        ),
        ExperimentGenome(
            id="direct-input-000",
            lineage="direct",
            hypothesis="Last-step linear readout matches FLM's mandatory control.",
            role="skeptic",
            architecture=Architecture(family="direct_input", hidden=1, history=8),
            curriculum=delayed,
        ),
        ExperimentGenome(
            id="mlp-000",
            lineage="mlp",
            hypothesis="Flattened history without recurrence is enough.",
            architecture=Architecture(family="mlp", hidden=32, history=8),
            curriculum=delayed,
        ),
        ExperimentGenome(
            id="gru-000",
            lineage="gru",
            hypothesis="Recurrent compression of the event stream, including delayed cue.",
            architecture=Architecture(family="gru", hidden=24, history=8),
            curriculum=delayed,
        ),
        ExperimentGenome(
            id="fixed-reservoir-000",
            lineage="reservoir",
            hypothesis="Frozen sparse recurrence + readout is a connectome-like prior.",
            architecture=Architecture(family="fixed_reservoir", hidden=48, history=8, sparsity=0.08),
            curriculum=delayed,
        ),
        ExperimentGenome(
            id="rewired-reservoir-000",
            lineage="rewired",
            hypothesis="Same sparsity, new wiring, refit readout — topology control.",
            role="skeptic",
            architecture=Architecture(family="rewired_reservoir", hidden=48, history=8, sparsity=0.08),
            training=Training(seed=7),
            curriculum=delayed,
        ),
        ExperimentGenome(
            id="local-plasticity-000",
            lineage="mushroom-body",
            hypothesis=(
                "Frozen sparse PN→KC on flattened Hermes history (not the compound eye) "
                "plus local KC→MBON plasticity. Motif student; not MaleCNS SGD."
            ),
            role="neuroscience",
            architecture=Architecture(
                family="local_plasticity",
                hidden=128,
                history=8,
                sparsity=0.10,
                trainable="kc_mbon",
                topology="mushroom_body_analogue",
            ),
            training=Training(algorithm="local_plasticity", seed=0),
            curriculum=delayed,
        ),
    ]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pts = points_from_archive(rows)
    front = pareto_front(pts)
    learned_rows = [
        r
        for r in rows
        if r.get("status") == "ok"
        and (r.get("genome") or {}).get("architecture", {}).get("family") != "rule"
    ]
    learned_pts = points_from_archive(learned_rows)
    learned_front = pareto_front(learned_pts)
    return {
        "n": len(rows),
        "ok": sum(1 for r in rows if r.get("status") == "ok"),
        "hypervolume_2d": hypervolume_2d(pts, ref_cost=REF_COST),
        "hypervolume_2d_learned": hypervolume_2d(learned_pts, ref_cost=REF_COST),
        "front": [p.experiment_id for p in front],
        "front_learned": [p.experiment_id for p in learned_front],
        "best_success": max((p.success for p in pts), default=0.0),
        "direct_input_success": next(
            (
                float((r.get("metrics") or {}).get("success_rate") or 0)
                for r in rows
                if (r.get("genome") or {}).get("architecture", {}).get("family") == "direct_input"
            ),
            None,
        ),
    }
