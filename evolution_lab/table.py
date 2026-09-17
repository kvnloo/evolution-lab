"""P1 control table: shared-feature Track A on locked Hermes recovery.

Kill criterion (GitHub #3 / PER-1525): if GRU *and* direct-input dominate a
fly reservoir on val, do not spend a cycle on MaleCNS SGD. Pivot to motif
students. Scientific policy still refuses MaleCNS SGD even when the kill
flag is false — `local_plasticity` is that motif pivot.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from .archive import Archive
from .elites import MapElites
from .engine import run_one, seed_genomes
from .schema import ExperimentGenome
from .targets import PRODUCT_TARGET

CONTROL_FAMILIES = (
    "rule",
    "direct_input",
    "mlp",
    "gru",
    "fixed_reservoir",
    "rewired_reservoir",
    "local_plasticity",
)


def _family(row: dict[str, Any]) -> str | None:
    return (row.get("genome") or {}).get("architecture", {}).get("family")


def _metric(row: dict[str, Any] | None, key: str) -> float:
    if not row:
        return 0.0
    return float((row.get("metrics") or {}).get(key) or 0.0)


def rows_by_family(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Best ok row per architecture family (highest val_success)."""
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        fam = _family(row)
        if not fam:
            continue
        prev = out.get(fam)
        if prev is None or _metric(row, "val_success") > _metric(prev, "val_success"):
            out[fam] = row
    return out


def kill_criterion(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """GRU+direct-input dominating the reservoir on val → no MaleCNS SGD."""
    by_fam = rows_by_family(rows)
    gru = by_fam.get("gru")
    direct = by_fam.get("direct_input")
    reservoir = by_fam.get("fixed_reservoir")
    missing = gru is None or direct is None or reservoir is None
    if missing:
        dominate = False
    else:
        dominate = _metric(gru, "val_success") >= _metric(reservoir, "val_success") and _metric(
            direct, "val_success"
        ) >= _metric(reservoir, "val_success")
    return {
        "gru_and_direct_dominate_reservoir": dominate,
        "do_not_run_malecns_sgd": dominate,
        "pivot_to_motif_students": dominate,
        "always_refuse_malecns_sgd": True,
        "missing_controls": missing,
        "val": {
            "gru": None if gru is None else _metric(gru, "val_success"),
            "direct_input": None if direct is None else _metric(direct, "val_success"),
            "fixed_reservoir": None if reservoir is None else _metric(reservoir, "val_success"),
            "local_plasticity": (
                None
                if by_fam.get("local_plasticity") is None
                else _metric(by_fam["local_plasticity"], "val_success")
            ),
        },
    }


def vs_teacher(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Learned success / teacher success vs PRODUCT_TARGET (confirm split)."""
    by_fam = rows_by_family(rows)
    teacher = by_fam.get("rule")
    teacher_success = _metric(teacher, "success_rate") if teacher else 0.0
    teacher_cost = _metric(teacher, "cost") if teacher else 0.0
    learned = [row for fam, row in by_fam.items() if fam != "rule"]
    if learned:
        # Highest confirm success; cheapest among ties (capability vs resources).
        best = max(learned, key=lambda row: (_metric(row, "success_rate"), -_metric(row, "cost")))
        best_success = _metric(best, "success_rate")
        best_cost = _metric(best, "cost")
        best_family = _family(best)
    else:
        best_success = 0.0
        best_cost = 0.0
        best_family = None
    mlp = by_fam.get("mlp")
    mlp_cost = _metric(mlp, "cost") if mlp else None
    success_ratio = (best_success / teacher_success) if teacher_success else 0.0
    cost_ratio = (best_cost / teacher_cost) if teacher_cost else None
    cost_vs_mlp = (best_cost / mlp_cost) if mlp_cost else None
    return {
        "teacher_success": teacher_success,
        "best_learned_success": best_success,
        "best_learned_family": best_family,
        "success_vs_teacher": success_ratio,
        "meets_success_target": success_ratio >= PRODUCT_TARGET.success_vs_teacher,
        "target_success": PRODUCT_TARGET.success_vs_teacher,
        "teacher_cost": teacher_cost,
        "best_learned_cost": best_cost,
        "cost_vs_teacher": cost_ratio,
        "meets_cost_target": bool(
            cost_ratio is not None and cost_ratio <= PRODUCT_TARGET.cost_vs_teacher
        ),
        "target_cost": PRODUCT_TARGET.cost_vs_teacher,
        "mlp_cost": mlp_cost,
        "cost_vs_mlp": cost_vs_mlp,
        "joules": PRODUCT_TARGET.joules,
        "cost_note": (
            "Teacher has ~0 params, so cost_vs_teacher is not hosted energy. "
            "cost_vs_mlp compares the cheapest perfect student to the ridge MLP."
        ),
    }


def _row_public(rec: dict[str, Any]) -> dict[str, Any]:
    metrics = rec.get("metrics") or {}
    return {
        "family": _family(rec),
        "experiment_id": rec.get("experiment_id"),
        "status": rec.get("status"),
        "val_success": metrics.get("val_success"),
        "success_rate": metrics.get("success_rate"),
        "ood_score": metrics.get("ood_score"),
        "cost": metrics.get("cost"),
        "params": metrics.get("params"),
        "violations": metrics.get("violations"),
        "joules_per_success": metrics.get("joules_per_success"),
    }


def ensure_control_genomes(genomes: list[ExperimentGenome] | None = None) -> list[ExperimentGenome]:
    seeds = seed_genomes()
    if not genomes:
        return list(seeds)
    have = {g.architecture.family for g in genomes}
    extra = [g for g in seeds if g.architecture.family not in have]
    return list(genomes) + extra


def write_control_table(
    *,
    level: int,
    run_dir: Path,
    genomes: list[ExperimentGenome] | None = None,
) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    archive = Archive(run_dir / "archive.jsonl")
    elites = MapElites()
    prior = archive.read()
    recs: list[dict[str, Any]] = []
    for genome in ensure_control_genomes(genomes):
        rec = run_one(genome, archive, elites, level=level, prior=prior)
        prior.append(rec)
        recs.append(rec)
    payload = {
        "level": level,
        "product_target": {
            "success_vs_teacher": PRODUCT_TARGET.success_vs_teacher,
            "cost_vs_teacher": PRODUCT_TARGET.cost_vs_teacher,
            "extra_violations": PRODUCT_TARGET.extra_violations,
            "joules": PRODUCT_TARGET.joules,
        },
        "rows": [_row_public(r) for r in recs],
        "kill_criterion": kill_criterion(recs),
        "vs_teacher": vs_teacher(recs),
        "note": (
            "PN encoder is flatten + last-step + max-over-time Hermes history, "
            "not the compound eye. fly_connectome remains an alias of fixed_reservoir. "
            "Do not SGD MaleCNS. Hosted joules stay unknown. Teacher has ~0 params so "
            "cost_vs_teacher is not a hosted-energy comparison."
        ),
        "useful_object": (
            "local_plasticity is a cheap Hermes recovery specialist "
            "({retry, restart_sandbox, escalate, noop, page_human}), "
            "not a DEX trader and not a Qwen replacement."
        ),
    }
    path = run_dir / "control_table.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path
