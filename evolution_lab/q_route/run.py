"""``q-route`` pipeline: build -> utility -> q -> gate -> distill -> report.

Each step reads the artifacts the previous step wrote into the run directory,
so the CLI stays a thin dispatcher and every stage is independently rerunnable.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .distill import RouterModel, fit_router, non_inferiority
from .gate import GateConfig, compile_regions, render_gate_markdown
from .qfunc import QModel, fit_q, pareto_arms
from .schema import (
    GATE_REPORT_SCHEMA,
    QVALUES_SCHEMA,
    ROUTER_SCHEMA,
    RUN_SCHEMA,
    TEACHER_TABLE_SCHEMA,
    UTILITY_SCHEMA,
)
from .teacher import (
    TeacherTable,
    build_teacher_table,
    load_teacher_table,
    resolve_sources,
    write_teacher_table,
)
from .utility import UtilityConfig, explain_utility, utility

DEFAULT_RUN_DIRNAME = "q_route"


def _dump(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _require(run_dir: Path, name: str, hint: str) -> Path:
    path = Path(run_dir) / name
    if not path.exists():
        raise RuntimeError(f"missing {path}; run `{hint}` first")
    return path


def load_utility_config(run_dir: Path | str) -> UtilityConfig:
    path = Path(run_dir) / "utility.json"
    if not path.exists():
        return UtilityConfig()
    data = json.loads(path.read_text(encoding="utf-8"))
    return UtilityConfig.from_dict(data.get("config", {}))


def load_gate_config(run_dir: Path | str) -> GateConfig:
    path = Path(run_dir) / "gate.json"
    if not path.exists():
        return GateConfig()
    data = json.loads(path.read_text(encoding="utf-8"))
    return GateConfig.from_dict(data.get("gate_config", {}))


def require_teacher(run_dir: Path | str) -> TeacherTable:
    return load_teacher_table(_require(Path(run_dir), "teacher.json", "q-route build --out <dir>"))


def require_qmodel(run_dir: Path | str) -> QModel:
    path = _require(Path(run_dir), "qvalues.json", "q-route q --run <dir>")
    return QModel.from_dict(json.loads(path.read_text(encoding="utf-8")))


def require_router(run_dir: Path | str) -> RouterModel:
    path = _require(Path(run_dir), "router.json", "q-route distill --run <dir>")
    return RouterModel.from_dict(json.loads(path.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------


def build_run(out_dir: Path | str, sources_value: str | None = None) -> dict[str, Any]:
    """Build the teacher table from the real (or overridden) sources."""
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    specs = resolve_sources(sources_value)
    table = build_teacher_table(specs)
    write_teacher_table(table, dest)

    run_payload = {
        "schema": RUN_SCHEMA,
        "teacher_schema": TEACHER_TABLE_SCHEMA,
        "utility_schema": UTILITY_SCHEMA,
        "qvalues_schema": QVALUES_SCHEMA,
        "gate_schema": GATE_REPORT_SCHEMA,
        "router_schema": ROUTER_SCHEMA,
        "sources": table.sources,
        "skipped_sources": table.skipped,
        "utility_config": UtilityConfig().to_dict(),
        "gate_config": GateConfig().to_dict(),
    }
    _dump(dest / "run.json", run_payload)

    arm_counts = Counter(r.arm for r in table.rows)
    return {
        "schema": RUN_SCHEMA,
        "run_dir": str(dest),
        "n_states": len(table.states()),
        "n_arms": len(table.arms()),
        "n_rows": len(table.rows),
        "sources": table.sources,
        "skipped_sources": table.skipped,
        "arms": sorted(arm_counts),
    }


def build_utility(run_dir: Path | str) -> dict[str, Any]:
    dest = Path(run_dir)
    table = require_teacher(dest)
    config = UtilityConfig()
    rows = [explain_utility(row, config) for row in table.sorted_rows()]
    by_arm: dict[str, list[float]] = {}
    for row in table.sorted_rows():
        by_arm.setdefault(row.arm, []).append(utility(row, config))
    payload = {
        "schema": UTILITY_SCHEMA,
        "config": config.to_dict(),
        "outcome_labels": ["good", "degraded", "bad"],
        "outcome_values": [1.0, 0.0, -1.0],
        "n_rows": len(rows),
        "mean_utility_by_arm": {
            arm: float(sum(vals) / len(vals)) for arm, vals in sorted(by_arm.items())
        },
        "rows": rows,
    }
    _dump(dest / "utility.json", payload)
    return {
        "schema": UTILITY_SCHEMA,
        "run_dir": str(dest),
        "n_rows": len(rows),
        "outcome_values": payload["outcome_values"],
        "config": config.to_dict(),
    }


def build_q(run_dir: Path | str) -> dict[str, Any]:
    dest = Path(run_dir)
    table = require_teacher(dest)
    config = load_utility_config(dest)
    model = fit_q(table, config)
    payload = model.to_dict()
    payload["pareto"] = pareto_arms(table, config)
    payload["best_arm"] = {state: model.best_arm(state) for state in table.states()}
    _dump(dest / "qvalues.json", payload)
    return {
        "schema": QVALUES_SCHEMA,
        "run_dir": str(dest),
        "n_arms": len(model.arms),
        "n_params": model.n_params,
        "feature_names": list(model.feature_names),
    }


def build_gate(run_dir: Path | str) -> dict[str, Any]:
    dest = Path(run_dir)
    table = require_teacher(dest)
    utility_config = load_utility_config(dest)
    report = compile_regions(table, GateConfig(), utility_config)
    _dump(dest / "gate.json", report.to_dict())
    (dest / "gate.md").write_text(render_gate_markdown(report, table), encoding="utf-8")
    return {
        "schema": GATE_REPORT_SCHEMA,
        "run_dir": str(dest),
        "n_states": report.n_states,
        "n_simplified_states": report.n_simplified_states,
        "n_escalated_states": report.n_escalated_states,
        "owner_counts": report.owner_counts,
        "total_latency_saved_ms": report.total_latency_saved_ms,
    }


def build_distill(run_dir: Path | str, seed: int = 42) -> dict[str, Any]:
    dest = Path(run_dir)
    table = require_teacher(dest)
    qmodel = require_qmodel(dest)
    gate_config = load_gate_config(dest)
    report = compile_regions(table, gate_config, qmodel.utility_config)
    router = fit_router(
        qmodel,
        qmodel.features,
        table,
        gate_report=report,
        gate_config=gate_config,
        utility_config=qmodel.utility_config,
        seed=seed,
    )
    _dump(dest / "router.json", router.to_dict())
    detail = non_inferiority(
        router,
        qmodel,
        table,
        gate_config,
        utility_config=qmodel.utility_config,
        gate_report=report,
    )
    _dump(dest / "noninferiority.json", detail)
    return {
        "schema": ROUTER_SCHEMA,
        "run_dir": str(dest),
        "n_arms": len(router.arms),
        "n_params": router.n_params,
        "fidelity": detail["fidelity"],
        "n_states": detail["n_states"],
        "mean_utility_delta": detail["mean_utility_delta"],
        "mean_regret": detail["mean_regret"],
    }


def build_analyze(
    run_dir: Path | str,
    *,
    observations_path: Path | str | None = None,
) -> dict[str, Any]:
    """Section G: diagnose the router's representation limit from measurements.

    Trains nothing.  It re-derives the feature-bucket ceiling, classifies every
    collision, scores candidate missing features by ceiling gain, and reports
    counterfactual realised utility of the *already fitted* router.
    """
    from .analysis import analyse, write_analysis
    from .distill import RouterModel
    from .qfunc import StateFeatures

    dest = Path(run_dir)
    table = require_teacher(dest)
    utility_config = load_utility_config(dest)
    gate_config = load_gate_config(dest)
    qmodel: QModel | None
    try:
        qmodel = require_qmodel(dest)
    except SystemExit:
        qmodel = None
    router: RouterModel | None
    try:
        router = require_router(dest)
    except SystemExit:
        router = None
    features = StateFeatures.from_teacher_table(table)
    report = analyse(
        table,
        observations_path=observations_path,
        router=router,
        router_features=features,
        qmodel=qmodel,
        gate_config=gate_config,
        utility_config=utility_config,
    )
    path = write_analysis(report, dest)
    return {
        "schema": report["schema"],
        "run_dir": str(dest),
        "analysis": str(path),
        "n_collisions": report["ceiling"]["n_collisions"],
        "ceiling": report["ceiling"]["ceiling"],
        "router_fidelity": report["ceiling"]["router_fidelity"],
        "dominant_limitation": report["verdict"]["dominant_limitation"],
        "best_candidate_feature": report["verdict"]["best_candidate_feature"],
    }


def build_report(run_dir: Path | str) -> Path:
    dest = Path(run_dir)
    table = require_teacher(dest)
    utility_payload = (
        json.loads((dest / "utility.json").read_text(encoding="utf-8"))
        if (dest / "utility.json").exists()
        else {}
    )
    q_payload = (
        json.loads((dest / "qvalues.json").read_text(encoding="utf-8"))
        if (dest / "qvalues.json").exists()
        else {}
    )
    gate_payload = (
        json.loads((dest / "gate.json").read_text(encoding="utf-8"))
        if (dest / "gate.json").exists()
        else {}
    )
    router_payload = (
        json.loads((dest / "router.json").read_text(encoding="utf-8"))
        if (dest / "router.json").exists()
        else {}
    )
    ni_payload = (
        json.loads((dest / "noninferiority.json").read_text(encoding="utf-8"))
        if (dest / "noninferiority.json").exists()
        else {}
    )
    run_payload = (
        json.loads((dest / "run.json").read_text(encoding="utf-8"))
        if (dest / "run.json").exists()
        else {}
    )

    lines: list[str] = []
    lines.append("# Q-Route report")
    lines.append("")
    lines.append(f"- schema: `{RUN_SCHEMA}`")
    lines.append(f"- teacher table: `{TEACHER_TABLE_SCHEMA}`")
    lines.append(
        f"- states: {len(table.states())} · arms: {len(table.arms())} · rows: {len(table.rows)}"
    )
    lines.append(
        f"- features: `{', '.join(q_payload.get('feature_names', ['bias']))}`"
    )
    lines.append("")

    lines.append("## Sources")
    lines.append("")
    lines.append("| source | kind | status | rows | note |")
    lines.append("|---|---|---|---|---|")
    seen_paths: set[str] = set()
    for source in run_payload.get("sources", []):
        seen_paths.add(str(source.get("path")))
        note = str(source.get("reason", ""))
        if source.get("malformed_rows"):
            note = f"{note} malformed={source['malformed_rows']}".strip()
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                source.get("label") or source.get("path"),
                source.get("kind"),
                source.get("status"),
                source.get("rows", 0),
                note,
            )
        )
    for skipped in run_payload.get("skipped_sources", []):
        if str(skipped.get("path")) in seen_paths:
            continue
        lines.append(
            "| {} | {} | skipped | 0 | {} |".format(
                skipped.get("label") or skipped.get("path"),
                skipped.get("kind"),
                skipped.get("reason", ""),
            )
        )
    lines.append("")

    if utility_payload:
        lines.append("## Utility")
        lines.append("")
        lines.append(f"- config: `{json.dumps(utility_payload.get('config', {}), sort_keys=True)}`")
        lines.append(
            "- outcome values (good/degraded/bad): "
            f"`{utility_payload.get('outcome_values')}`"
        )
        lines.append("")
        lines.append("| arm | mean utility |")
        lines.append("|---|---|")
        for arm, value in sorted(
            utility_payload.get("mean_utility_by_arm", {}).items(),
            key=lambda kv: (-kv[1], kv[0]),
        ):
            lines.append(f"| {arm} | {value:.4f} |")
        lines.append("")

    if gate_payload:
        lines.append("## Gate — earned complexity")
        lines.append("")
        lines.append(
            "- states: {n_states} · cheapest arm owns {n_simplified_states} · "
            "escalated {n_escalated_states}".format(**gate_payload)
        )
        lines.append(
            "- latency: routed {:.1f} ms vs always-most-expensive {:.1f} ms "
            "(saved {:.1f} ms)".format(
                gate_payload.get("routed_total_latency_ms", 0.0),
                gate_payload.get("baseline_total_latency_ms", 0.0),
                gate_payload.get("total_latency_saved_ms", 0.0),
            )
        )
        lines.append("")
        lines.append("| arm | rung | states owned |")
        lines.append("|---|---|---|")
        owner_rungs = gate_payload.get("owner_rungs", {})
        for arm, count in sorted(
            gate_payload.get("owner_counts", {}).items(), key=lambda kv: (-kv[1], kv[0])
        ):
            lines.append(f"| {arm} | {owner_rungs.get(arm, '')} | {count} |")
        lines.append("")
        escalated = [r for r in gate_payload.get("regions", []) if r.get("escalated_from")]
        lines.append(f"### Escalations ({len(escalated)})")
        lines.append("")
        lines.append("| state | owner | escalated from | reason |")
        lines.append("|---|---|---|---|")
        for region in escalated:
            reason = "; ".join(
                "{} vs {}: {}".format(
                    esc["from"], esc["to"], ", ".join(esc["reasons"]) or "passed"
                )
                for esc in region.get("reasons_for_escalation", [])
            )
            lines.append(
                "| {} | {} | {} | {} |".format(
                    region["state_key"],
                    region["owner"],
                    ", ".join(region["escalated_from"]),
                    reason,
                )
            )
        lines.append("")

    if ni_payload:
        lines.append("## Router — distilling the gate, not a teacher")
        lines.append("")
        lines.append(
            "- params: {} · states: {} · fidelity: {:.3f} ({} matched) · "
            "feature-map ceiling: {:.3f} over {} buckets · "
            "mean utility delta: {:+.4f} · mean regret: {:.4f} · max regret: {:.4f}".format(
                ni_payload.get("n_router_params", router_payload.get("n_params", 0)),
                ni_payload.get("n_states", 0),
                ni_payload.get("fidelity", 0.0),
                ni_payload.get("n_matched", 0),
                ni_payload.get("feature_bucket_ceiling", 0.0),
                ni_payload.get("n_feature_buckets", 0),
                ni_payload.get("mean_utility_delta", 0.0),
                ni_payload.get("mean_regret", 0.0),
                ni_payload.get("max_regret", 0.0),
            )
        )
        lines.append("")
        if ni_payload.get("fidelity", 0.0) < 1.0:
            mismatches = [r for r in ni_payload.get("per_state", []) if not r.get("match")]
            lines.append("| state | gate arm | router arm | regret |")
            lines.append("|---|---|---|---|")
            for row in mismatches:
                regret = row.get("regret")
                lines.append(
                    "| {} | {} | {} | {} |".format(
                        row["state_key"],
                        row["gate_arm"],
                        row["router_arm"],
                        "n/a" if regret is None else f"{regret:.4f}",
                    )
                )
            lines.append("")

    if q_payload.get("pareto"):
        lines.append("## Cost/quality Pareto frontier")
        lines.append("")
        lines.append(f"- states with a frontier: {len(q_payload['pareto'])}")
        lines.append("")
        lines.append("| state | frontier arms (utility ↓) |")
        lines.append("|---|---|")
        for state in sorted(q_payload["pareto"]):
            lines.append(f"| {state} | {', '.join(q_payload['pareto'][state])} |")
        lines.append("")

    path = dest / "qroute.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


__all__ = [
    "DEFAULT_RUN_DIRNAME",
    "build_distill",
    "build_gate",
    "build_q",
    "build_report",
    "build_run",
    "build_utility",
    "load_gate_config",
    "load_utility_config",
    "require_qmodel",
    "require_router",
    "require_teacher",
]
