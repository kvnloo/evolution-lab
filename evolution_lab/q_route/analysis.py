"""Phase 1B section G -- validate Q-Route as a router, and diagnose its limit.

Phase 1 fitted a 32-parameter router to 38 states and reported fidelity 0.579
against a feature-bucket ceiling of 0.658.  Those two numbers do not say *why*
the router falls short, and a fidelity target computed over noisy ``n=1`` cells
is not a target worth optimising.  This module answers the questions the brief
asks, using measurements only -- it never trains anything:

* recompute the feature-bucket ceiling on the densified evidence;
* enumerate **collisions**: feature buckets in which the frozen gate's owner is
  not constant, i.e. states the current representation cannot tell apart;
* for every collision, decide between the four possible causes --
  ``noisy_labels``, ``multimodal_state_action_value``, ``insufficient_features``
  and ``insufficient_router_capacity`` -- using repeated observations rather
  than assertion;
* score candidate **missing state variables** by how much ceiling they recover,
  so the next feature is chosen from evidence instead of invented.

The counterfactual-realised-utility view matters more than imitation accuracy:
:func:`realised_utility_gap` reports what the router's choices actually cost,
which is the number a promotion decision should read.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .gate import GateConfig, compile_regions
from .qfunc import FEATURE_NAMES, QModel, StateFeatures
from .schema import TEACHER_TABLE_SCHEMA
from .teacher import TeacherTable
from .utility import UtilityConfig, utility

ANALYSIS_SCHEMA = "qroute.analysis.v1"


# ---------------------------------------------------------------------------
# Observation-level evidence (densified receipts from z0intelligence)
# ---------------------------------------------------------------------------


@dataclass
class CellEvidence:
    """Repeated observations for one ``(state, arm)`` cell."""

    state_key: str
    arm: str
    arm_raw: str = ""
    n: int = 0
    successes: int = 0
    danger: int = 0
    attempts: int = 0
    errors: int = 0
    latencies: list[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return (self.successes / self.n) if self.n else 0.0

    @property
    def disagreement(self) -> bool:
        """Repeated draws of the same cell did not all agree on success."""
        return 0 < self.successes < self.n

    @property
    def error_rate(self) -> float:
        return (self.errors / self.n) if self.n else 0.0


def load_observations(path: Path | str) -> dict[tuple[str, str], CellEvidence]:
    """Aggregate raw Phase 1B receipts into per-cell disagreement evidence.

    Only the *raw* receipts are read.  Aggregating here (rather than trusting a
    pre-aggregated table) is the whole point: disagreement is invisible once the
    individual draws have been averaged away.
    """
    target = Path(path)
    cells: dict[tuple[str, str], CellEvidence] = {}
    if not target.is_file():
        return cells
    with target.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            state = row.get("state_id")
            arm_raw = row.get("arm")
            if not state or not arm_raw:
                continue
            # Phase 1B receipts carry ``arm_alias`` so a densified arm name lines
            # up with the arm name the teacher table already uses.  Without this
            # the collision classifier sees zero matching cells and reports
            # "undetermined" for every bucket.
            arm = str(row.get("arm_alias") or arm_raw)
            cell = cells.setdefault(
                (str(state), arm), CellEvidence(str(state), arm, str(arm_raw))
            )
            cell.n += 1
            cell.successes += int(bool(row.get("correct")))
            cell.danger += int(bool(row.get("dangerous_selected")))
            cell.attempts += int((row.get("failure_retry") or {}).get("attempts") or 0)
            cell.errors += int(bool((row.get("failure_retry") or {}).get("error")))
            cell.latencies.append(float(row.get("total_ms") or 0.0))
    return cells


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def coverage(cells: Mapping[tuple[str, str], CellEvidence] | None) -> dict[str, Any]:
    if not cells:
        return {"available": False}
    counts: dict[int, int] = {}
    for cell in cells.values():
        counts[cell.n] = counts.get(cell.n, 0) + 1
    return {
        "available": True,
        "cells": len(cells),
        "observations": sum(c.n for c in cells.values()),
        "cells_ge_3": sum(1 for c in cells.values() if c.n >= 3),
        "cells_ge_10": sum(1 for c in cells.values() if c.n >= 10),
        "cell_size_histogram": {str(k): v for k, v in sorted(counts.items())},
    }


def label_noise_report(cells: Mapping[tuple[str, str], CellEvidence] | None) -> dict[str, Any]:
    if not cells:
        return {"available": False}
    repeated = {k: c for k, c in cells.items() if c.n >= 2}
    noisy = {k: c for k, c in repeated.items() if c.disagreement}
    by_arm: dict[str, dict[str, int]] = {}
    for (_, arm), cell in repeated.items():
        bucket = by_arm.setdefault(arm, {"repeated_cells": 0, "noisy_cells": 0})
        bucket["repeated_cells"] += 1
        bucket["noisy_cells"] += int(cell.disagreement)
    return {
        "available": True,
        "repeated_cells": len(repeated),
        "noisy_cells": len(noisy),
        "noise_rate": (len(noisy) / len(repeated)) if repeated else None,
        "noisy_examples": sorted(
            f"{state}/{arm}" for state, arm in noisy
        )[:50],
        "by_arm": by_arm,
    }


# ---------------------------------------------------------------------------
# Ceiling and collisions
# ---------------------------------------------------------------------------


def _bucket_key(vector: Sequence[float]) -> tuple[float, ...]:
    return tuple(round(float(v), 6) for v in vector)


def bucket_ceiling(
    table: TeacherTable,
    owners: Mapping[str, str],
    vectors: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    """The best achievable fidelity for a *fixed* representation.

    For each feature bucket, the best a classifier over these features can do is
    predict the bucket's majority owner.  That is the ceiling; the router can
    only ever be as good as the representation allows.
    """
    buckets: dict[tuple[float, ...], dict[str, int]] = {}
    members: dict[tuple[float, ...], list[str]] = {}
    for state in table.states():
        owner = owners.get(state)
        if owner is None:
            continue
        key = _bucket_key(vectors.get(state, ()))
        buckets.setdefault(key, {})
        buckets[key][owner] = buckets[key].get(owner, 0) + 1
        members.setdefault(key, []).append(state)
    hits = sum(max(counts.values()) for counts in buckets.values())
    n = sum(sum(counts.values()) for counts in buckets.values())
    collisions = [
        {"bucket": list(key), "states": sorted(members[key]), "owners": counts}
        for key, counts in buckets.items()
        if len(counts) > 1
    ]
    collisions.sort(key=lambda c: (-len(c["states"]), c["states"][0]))
    return {
        "n_states": n,
        "n_buckets": len(buckets),
        "ceiling_hits": hits,
        "ceiling": (hits / n) if n else 0.0,
        "collisions": collisions,
        "buckets": {",".join(f"{v:g}" for v in key): sorted(members[key]) for key in buckets},
    }


def candidate_feature_vectors(
    cells: Mapping[tuple[str, str], CellEvidence] | None,
    observations_path: Path | str | None,
) -> dict[str, dict[str, float]]:
    """Per-state values for candidate *missing* state variables.

    These are the variables the densified receipts made cheap to record.  Each
    becomes a candidate extra feature; the ceiling gain decides whether it is
    worth adding, which is a measurement rather than a preference.
    """
    if not observations_path:
        return {}
    target = Path(observations_path)
    if not target.is_file():
        return {}

    scalar_keys = (
        "legal_family_count",
        "authority_breadth",
        "budget_units",
        "declared_dangerous_count",
        "satisfied_count",
        "candidate_action_count",
    )
    numeric: dict[str, dict[str, list[float]]] = {k: {} for k in scalar_keys}
    irreversible: dict[str, list[float]] = {}
    families: dict[str, set[str]] = {}

    with target.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            state = str(row.get("state_id") or "")
            if not state:
                continue
            features = row.get("state_features") or {}
            for key in scalar_keys:
                value = features.get(key)
                if isinstance(value, (int, float)):
                    numeric[key].setdefault(state, []).append(float(value))
            risks = features.get("legal_risk_classes") or []
            irreversible.setdefault(state, []).append(
                1.0 if any(r in ("destructive", "publish", "credential", "payment")
                           for r in risks) else 0.0
            )
            family = features.get("family")
            if isinstance(family, str):
                families.setdefault(state, set()).add(family)

    out: dict[str, dict[str, float]] = {}
    for key, per_state in numeric.items():
        if per_state:
            out[key] = {s: sum(v) / len(v) for s, v in per_state.items()}
    if irreversible:
        out["legal_has_irreversible"] = {
            s: sum(v) / len(v) for s, v in irreversible.items()
        }
    # One-hot the state family: the strongest available categorical separator.
    for state, names in families.items():
        for name in sorted(names):
            out.setdefault(f"family={name}", {})[state] = 1.0
    return out


def score_candidate_features(
    table: TeacherTable,
    owners: Mapping[str, str],
    base_vectors: Mapping[str, Sequence[float]],
    candidates: Mapping[str, Mapping[str, float]],
) -> list[dict[str, Any]]:
    baseline = bucket_ceiling(table, owners, base_vectors)
    scored: list[dict[str, Any]] = []
    for name, values in sorted(candidates.items()):
        if not any(state in values for state in table.states()):
            continue
        extended: dict[str, list[float]] = {}
        for state in table.states():
            base = list(base_vectors.get(state, (0.0,) * len(FEATURE_NAMES)))
            extended[state] = base + [float(values.get(state, 0.0))]
        result = bucket_ceiling(table, owners, extended)
        scored.append(
            {
                "feature": name,
                # ``family=*`` one-hots are fixture metadata, not something a
                # router could read at decision time.  They are still scored
                # (the ceiling they unlock is informative) but they must not be
                # mistaken for an implementable state variable.
                "runtime_observable": not name.startswith("family="),
                "ceiling": result["ceiling"],
                "ceiling_gain": result["ceiling"] - baseline["ceiling"],
                "n_buckets": result["n_buckets"],
                "remaining_collisions": len(result["collisions"]),
            }
        )
    scored.sort(key=lambda row: (-row["ceiling_gain"], row["feature"]))
    return scored


def best_observable_feature(scored: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """The highest-gain candidate a runtime router could actually read."""
    for row in scored:
        if row.get("runtime_observable"):
            return row
    return None


# ---------------------------------------------------------------------------
# Cause classification
# ---------------------------------------------------------------------------


def classify_collision(
    states: Sequence[str],
    owners: Mapping[str, str],
    cells: Mapping[tuple[str, str], CellEvidence] | None,
    *,
    feature_gain_available: bool,
) -> dict[str, Any]:
    """Decide *why* a bucket collides, from repeated observations."""
    distinct = sorted({owners[s] for s in states if s in owners})
    reasons: list[str] = []
    evidence: dict[str, Any] = {"distinct_owners": distinct}

    if cells:
        noisy_states = []
        stable_states = []
        for state in states:
            state_cells = [c for (s, _), c in cells.items() if s == state and c.n >= 2]
            if not state_cells:
                continue
            if any(c.disagreement for c in state_cells):
                noisy_states.append(state)
            else:
                stable_states.append(state)
        evidence["noisy_states"] = sorted(noisy_states)
        evidence["stable_states"] = sorted(stable_states)
        if noisy_states:
            reasons.append("noisy_labels")
        if stable_states and len(distinct) > 1:
            reasons.append("multimodal_state_action_value")
    if feature_gain_available:
        reasons.append("insufficient_features")
    if not reasons:
        reasons.append("undetermined")

    classification = "undetermined"
    if "insufficient_features" in reasons and (
        "multimodal_state_action_value" in reasons or "noisy_labels" in reasons
    ):
        classification = "insufficient_features"
    elif reasons:
        classification = reasons[0]
    return {
        "states": list(states),
        "distinct_owners": distinct,
        "classification": classification,
        "reasons": reasons,
        "evidence": evidence,
    }


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def realised_utility_gap(
    table: TeacherTable,
    owners: Mapping[str, str],
    choose: Callable[[str], str | None] | None,
    utility_config: UtilityConfig,
) -> dict[str, Any]:
    """Counterfactual realised utility: router choice versus the gate's owner.

    This is the number a promotion decision should read.  Imitation accuracy can
    look poor while realised utility is nearly identical (the router picks an
    equally good arm), and it can look good while utility is worse.
    """
    if choose is None:
        return {"available": False}
    deltas: list[float] = []
    worse = equal = better = 0
    details: list[dict[str, Any]] = []
    for state in table.states():
        owner = owners.get(state)
        if owner is None:
            continue
        picked = choose(state)
        if picked is None:
            continue
        owner_row = table.row_for(state, owner)
        picked_row = table.row_for(state, picked)
        if owner_row is None or picked_row is None:
            continue
        delta = utility(picked_row, utility_config) - utility(owner_row, utility_config)
        deltas.append(delta)
        if delta < -1e-9:
            worse += 1
        elif delta > 1e-9:
            better += 1
        else:
            equal += 1
        details.append(
            {"state_key": state, "owner": owner, "picked": picked, "utility_delta": delta}
        )
    mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
    return {
        "available": True,
        "states": len(deltas),
        "mean_utility_delta_vs_owner": mean_delta,
        "mean_regret": max(0.0, -mean_delta),
        "worse_states": worse,
        "equal_states": equal,
        "better_states": better,
        "details": details,
    }


def analyse(
    table: TeacherTable,
    *,
    observations_path: Path | str | None = None,
    router: Any | None = None,
    router_features: StateFeatures | None = None,
    qmodel: QModel | None = None,
    gate_config: GateConfig | None = None,
    utility_config: UtilityConfig | None = None,
) -> dict[str, Any]:
    """Assemble the section G report.  Reads measurements; trains nothing.

    ``router`` is the *distilled gate-targeted router* (the artifact whose
    fidelity Phase 1 reported); ``qmodel`` is the per-arm Q function, which is a
    different object and, used as a chooser, scores far worse.  Both are
    reported so the distinction cannot be lost again.
    """
    gate_config = gate_config or GateConfig()
    utility_config = utility_config or UtilityConfig()
    report = compile_regions(table, gate_config, utility_config)
    owners = {region.state_key: region.owner for region in report.regions}

    features = StateFeatures.from_teacher_table(table)
    vectors = {state: features.vector(state) for state in table.states()}
    baseline = bucket_ceiling(table, owners, vectors)

    cells = load_observations(observations_path) if observations_path else {}
    candidates = candidate_feature_vectors(cells, observations_path)
    scored = score_candidate_features(table, owners, vectors, candidates)

    collisions = []
    for collision in baseline["collisions"]:
        classified = classify_collision(
            collision["states"], owners, cells or None,
            feature_gain_available=bool(scored),
        )
        classified["bucket"] = collision["bucket"]
        classified["owners"] = collision["owners"]
        collisions.append(classified)

    fidelity = None
    if router is not None:
        matches = 0
        total = 0
        for state, owner in owners.items():
            picked = _router_choice(router, router_features, state)
            if picked is None:
                continue
            total += 1
            matches += int(picked == owner)
        fidelity = (matches / total) if total else None

    qfidelity = None
    if qmodel is not None:
        matches = 0
        total = 0
        for state, owner in owners.items():
            try:
                picked = qmodel.best_arm(state)
            except KeyError:
                continue
            total += 1
            matches += int(picked == owner)
        qfidelity = (matches / total) if total else None

    dominant = "insufficient_router_capacity"
    if collisions:
        counts: dict[str, int] = {}
        for c in collisions:
            counts[c["classification"]] = counts.get(c["classification"], 0) + 1
        dominant = max(sorted(counts), key=lambda k: counts[k])
    elif fidelity is not None and baseline["ceiling"] - fidelity > 0.05:
        dominant = "insufficient_router_capacity"

    return {
        "schema": ANALYSIS_SCHEMA,
        "teacher_table_schema": TEACHER_TABLE_SCHEMA,
        "n_states": baseline["n_states"],
        "n_arms": len(table.arms()),
        "n_rows": len(table.rows),
        "frozen_features": list(FEATURE_NAMES),
        "coverage": coverage(cells or None),
        "label_noise": label_noise_report(cells or None),
        "ceiling": {
            "n_buckets": baseline["n_buckets"],
            "ceiling": baseline["ceiling"],
            "router_fidelity": fidelity,
            "qmodel_as_chooser_fidelity": qfidelity,
            "gap_to_ceiling": (baseline["ceiling"] - fidelity)
            if fidelity is not None else None,
            "n_collisions": len(collisions),
        },
        "collisions": collisions,
        "candidate_features": scored,
        "realised_utility": realised_utility_gap(
            table, owners,
            (lambda state: _router_choice(router, router_features, state))
            if router is not None else (
                (lambda state: qmodel.best_arm(state)) if qmodel is not None else None
            ),
            utility_config,
        ),
        "realised_utility_qmodel": realised_utility_gap(
            table, owners,
            (lambda state: qmodel.best_arm(state)) if qmodel is not None else None,
            utility_config,
        ),
        "verdict": {
            "dominant_limitation": dominant,
            "collision_class_counts": _counts(c["classification"] for c in collisions),
            "best_candidate_feature": (scored[0]["feature"] if scored else None),
            "best_candidate_gain": (scored[0]["ceiling_gain"] if scored else None),
            "best_runtime_observable_feature": (
                best_observable_feature(scored) or {}
            ).get("feature"),
            "best_runtime_observable_gain": (
                best_observable_feature(scored) or {}
            ).get("ceiling_gain"),
        },
    }


def _router_choice(router: Any, features: StateFeatures | None, state: str) -> str | None:
    """The router's arm for a state, or None when it cannot be evaluated."""
    if router is None or features is None:
        return None
    if not hasattr(router, "predict"):
        return None
    try:
        return str(router.predict(features.vector(state)))
    except Exception:  # noqa: BLE001 - an unevaluable state is not a crash
        return None


def _counts(values: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return out


def render_md(report: Mapping[str, Any]) -> str:
    ceiling = report["ceiling"]
    noise = report.get("label_noise") or {}
    cov = report.get("coverage") or {}
    fidelity = ceiling["router_fidelity"]
    gap = ceiling["gap_to_ceiling"]
    fidelity_text = "n/a" if fidelity is None else f"{fidelity:.3f}"
    gap_text = "n/a" if gap is None else f"{gap:.3f}"
    qfid = ceiling.get("qmodel_as_chooser_fidelity")
    qfid_text = "n/a" if qfid is None else f"{qfid:.3f}"
    lines = [
        "# Q-Route router validation (Phase 1B section G)",
        "",
        f"- schema: `{ANALYSIS_SCHEMA}`",
        f"- states: {report['n_states']}  arms: {report['n_arms']}  rows: {report['n_rows']}",
        f"- frozen feature vector: `{', '.join(report['frozen_features'])}`",
        f"- feature buckets: {ceiling['n_buckets']}  ceiling: **{ceiling['ceiling']:.3f}**",
        f"- router fidelity: {fidelity_text}  gap to ceiling: {gap_text}",
        "- per-arm Q-model used directly as a chooser: " + qfid_text,
        "  (the Q function scores each arm; it is not the gate-targeted router)",
        f"- collisions (buckets with disagreeing owners): {ceiling['n_collisions']}",
        "",
        "## Evidence coverage",
        "",
    ]
    if cov.get("available"):
        lines += [
            f"- cells: {cov['cells']}  observations: {cov['observations']}",
            f"- cells with >=3 repetitions: {cov['cells_ge_3']}",
            f"- cells with >=10 repetitions: {cov['cells_ge_10']}",
        ]
    else:
        lines.append("- no densified observations supplied")
    lines += ["", "## Label noise (from repeated observations)", ""]
    if noise.get("available"):
        rate = noise.get("noise_rate")
        lines += [
            f"- repeated cells: {noise['repeated_cells']}",
            f"- cells whose repeated draws disagreed on success: {noise['noisy_cells']}",
            "- disagreement rate: " + (f"{rate:.3f}" if rate is not None else "n/a"),
        ]
    else:
        lines.append("- n/a")
    lines += ["", "## Collision classification", "",
              "| bucket | states | distinct owners | classification | reasons |",
              "|---|---|---|---|---|"]
    for c in report["collisions"]:
        lines.append(
            f"| {','.join(f'{v:g}' for v in c['bucket'])} | {', '.join(c['states'])} "
            f"| {', '.join(c['distinct_owners'])} | {c['classification']} "
            f"| {', '.join(c['reasons'])} |"
        )
    if report.get("candidate_features"):
        lines += ["", "## Candidate missing state variables (ceiling gain)", "",
                  "| feature | ceiling with it | gain | buckets | remaining collisions |",
                  "|---|---|---|---|---|"]
        for row in report["candidate_features"]:
            marker = "" if row.get("runtime_observable", True) else " *(fixture metadata)*"
            lines.append(
                f"| {row['feature']}{marker} | {row['ceiling']:.3f} | {row['ceiling_gain']:+.3f} "
                f"| {row['n_buckets']} | {row['remaining_collisions']} |"
            )
    util = report.get("realised_utility") or {}
    if util.get("available"):
        lines += [
            "", "## Counterfactual realised utility (router vs gate owner)", "",
            f"- states compared: {util['states']}",
            f"- mean utility delta: {util['mean_utility_delta_vs_owner']:+.4f}",
            f"- mean regret: {util['mean_regret']:.4f}",
            f"- worse / equal / better: {util['worse_states']} / {util['equal_states']} "
            f"/ {util['better_states']}",
        ]
    verdict = report["verdict"]
    lines += ["", "## Verdict", "",
              f"- dominant limitation: **{verdict['dominant_limitation']}**",
              f"- collision classes: {json.dumps(verdict['collision_class_counts'], sort_keys=True)}",
              f"- best candidate feature: {verdict['best_candidate_feature']} "
              f"(gain {verdict['best_candidate_gain']})",
              f"- best *runtime-observable* candidate: "
              f"{verdict.get('best_runtime_observable_feature')} "
              f"(gain {verdict.get('best_runtime_observable_gain')})"]
    return "\n".join(lines) + "\n"


def write_analysis(report: Mapping[str, Any], out_dir: Path | str) -> Path:
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    (dest / "analysis.md").write_text(render_md(report), encoding="utf-8")
    return dest / "analysis.json"


__all__ = [
    "ANALYSIS_SCHEMA",
    "CellEvidence",
    "analyse",
    "best_observable_feature",
    "bucket_ceiling",
    "candidate_feature_vectors",
    "classify_collision",
    "coverage",
    "label_noise_report",
    "load_observations",
    "realised_utility_gap",
    "render_md",
    "score_candidate_features",
    "write_analysis",
]
