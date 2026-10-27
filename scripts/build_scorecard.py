#!/usr/bin/env python3
"""Build a deterministic, privacy-safe Markdown scorecard from committed results.

Only in-file baselines are comparable. The LLM artifact has no declared trivial
floor and explicitly disclaims cross-encoding comparisons, so its verdicts are
NO_COMPARABLE_FLOOR rather than fabricated wins or losses.
"""

from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Sequence

SCHEMA_RECOVERY = "z0evals.recovery-backend-compare.v1"
SCHEMA_LLM = "z0evals.recovery-llm-compare.v1"
SCHEMA_SHADOW = "z0evals.shadow-real-traffic.v1"
KNOWN_SCHEMAS = {SCHEMA_RECOVERY, SCHEMA_LLM, SCHEMA_SHADOW}
CAVEAT_FIELDS = ("metric_caveat", "encoding_caveat", "confidence_note", "not_measured")
VERDICT_ORDER = {
    "BEATS_FLOOR": 0,
    "TIES_FLOOR": 1,
    "LOSES_TO_FLOOR": 2,
    "NO_COMPARABLE_FLOOR": 3,
}
CODE = chr(96)
TIE_THRESHOLD = Decimal("0.01")


def safe(value: Any) -> str:
    """Format source metadata without exposing leading-slash or tilde paths."""
    if isinstance(value, str):
        if value.startswith(("/", "~")):
            return f"{Path(value).name or '[path]'} (local path redacted)"
        return value
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "null"
    if isinstance(value, (list, dict)):
        def sanitize(item: Any) -> Any:
            if isinstance(item, str) and item.startswith(("/", "~")):
                return f"{Path(item).name or '[path]'} (local path redacted)"
            if isinstance(item, list):
                return [sanitize(element) for element in item]
            if isinstance(item, dict):
                return {str(key): sanitize(element) for key, element in item.items()}
            if isinstance(item, Decimal):
                return float(item)
            return item
        return json.dumps(sanitize(value), sort_keys=True, ensure_ascii=False)
    return str(value)


def cell(value: Any) -> str:
    """Keep untrusted metadata from breaking Markdown table columns."""
    return safe(value).replace("|", r"\|").replace("\r", " ").replace("\n", " ")


def verdict(delta: Decimal | float | None) -> str:
    if delta is None:
        return "NO_COMPARABLE_FLOOR"
    exact_delta = Decimal(str(delta))
    if abs(exact_delta) < TIE_THRESHOLD:
        return "TIES_FLOOR"
    return "BEATS_FLOOR" if exact_delta > 0 else "LOSES_TO_FLOOR"


def make_row(
    arm: str, split: str, value: Decimal | float, floor: Decimal | float | None,
    *, n: int | None = None, unparsed: int | None = None,
    latency_p50_ms: Decimal | float | None = None,
) -> dict[str, Any]:
    # Evaluate accuracy differences in decimal, not binary float: 0.29 - 0.28
    # must reach the exact 0.01 boundary instead of rounding just below it.
    exact_value = Decimal(str(value))
    exact_floor = Decimal(str(floor)) if floor is not None else None
    delta = exact_value - exact_floor if exact_floor is not None else None
    return {
        "arm": arm, "split": split, "value": exact_value, "floor": exact_floor,
        "delta": delta, "verdict": verdict(delta), "n": n,
        "unparsed": unparsed, "latency_p50_ms": latency_p50_ms,
    }


def rows_for(data: dict[str, Any]) -> list[dict[str, Any]]:
    schema = data["schema"]
    rows: list[dict[str, Any]] = []
    if schema == SCHEMA_RECOVERY:
        floor_by_split = data["results"]["majority"]
        for arm, splits in sorted(data["results"].items()):
            for split, value in sorted(splits.items()):
                rows.append(make_row(arm, split, value, floor_by_split[split]))
    elif schema == SCHEMA_LLM:
        # This source declares no majority/trivial floor. Do not combine with
        # recovery-compare.json: its encoding_caveat forbids that comparison.
        for arm, splits in sorted(data["backends"].items()):
            for split, stats in sorted(splits.items()):
                rows.append(make_row(
                    arm, split, stats["accuracy"], None,
                    n=int(stats["n"]), unparsed=int(stats["unparsed"]),
                    latency_p50_ms=stats["latency_p50_ms"],
                ))
    elif schema == SCHEMA_SHADOW:
        floor = data["majority_floor"]
        for arm, value in sorted(data["results"].items()):
            rows.append(make_row(arm, "confirm", value, floor))
    else:
        raise ValueError(f"unsupported result schema: {safe(schema)}")
    return sorted(rows, key=lambda row: (row["split"], row["arm"]))


def number(value: Decimal | float | None, *, signed: bool = False) -> str:
    if value is None:
        return "—"
    # Preserve the committed scorecard's three-decimal float display only.
    # Verdict and sorting use exact decimal differences, not this rounding.
    displayed = float(value)
    return f"{displayed:+.3f}" if signed else f"{displayed:.3f}"


def render_scorecard(results_dir: Path, *, source_sha: str | None = None) -> str:
    """Render known schemas in filename order; report unknown schemas separately."""
    docs: list[tuple[str, dict[str, Any], list[dict[str, Any]]]] = []
    unrendered: list[tuple[str, str]] = []
    for path in sorted(results_dir.glob("*.json"), key=lambda item: item.name):
        data = json.loads(path.read_text(encoding="utf-8"), parse_float=Decimal)
        if not isinstance(data, dict):
            unrendered.append((path.name, "(non-object JSON)"))
            continue
        schema = data.get("schema")
        if schema not in KNOWN_SCHEMAS:
            unrendered.append((path.name, safe(schema)))
            continue
        docs.append((path.name, data, rows_for(data)))

    lines = [
        "# Results scorecard",
        "",
        "Generated from committed result JSON. These are exploratory observations, not promotion approvals.",
        "Baselines are taken from the same file and split. No across-encoding comparisons are inferred.",
        "The summary uses the confirm split when present; other splits are shown in the details.",
        "Verdicts use full-precision deltas: |delta| < 0.01 is TIES_FLOOR.",
    ]
    if source_sha is not None:
        lines.append(f"Source git SHA: {CODE}{cell(source_sha)}{CODE}")
    lines += [
        "",
        "## Summary",
        "",
        "| File | Arm | Split | Accuracy | Floor | Δ vs floor | Verdict | Caveats |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    summary: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for filename, data, rows in docs:
        for arm in sorted({row["arm"] for row in rows}):
            candidates = [row for row in rows if row["arm"] == arm]
            chosen = next((row for row in candidates if row["split"] == "confirm"), candidates[0])
            summary.append((filename, data, chosen))
    summary.sort(key=lambda item: (
        VERDICT_ORDER[item[2]["verdict"]],
        -(item[2]["delta"] if item[2]["delta"] is not None else Decimal(0)),
        item[0], item[2]["arm"],
    ))
    for filename, data, row in summary:
        caveats = ", ".join(key for key in CAVEAT_FIELDS if key in data) or "none supplied"
        lines.append(
            f"| {cell(filename)} | {cell(row['arm'])} | {cell(row['split'])} | "
            f"{number(row['value'])} | {number(row['floor'])} | "
            f"{number(row['delta'], signed=True)} | {row['verdict']} | "
            f"{cell(caveats)} (see section) |"
        )
    if not summary:
        lines.append("| — | — | — | — | — | — | — | — |")

    for filename, data, rows in docs:
        schema = data["schema"]
        metric = (
            data.get("metric", "action accuracy")
            if schema == SCHEMA_RECOVERY
            else "action accuracy (text-rendered inputs)"
            if schema == SCHEMA_LLM
            else "held-out action-family accuracy"
        )
        task = data.get("task", "Hermes next-action shadow traffic")
        lines += [
            "",
            f"## {CODE}results/{cell(filename)}{CODE}",
            "",
            f"- Task: {cell(task)}",
            f"- Metric: {cell(metric)}",
            f"- Splits: {', '.join(sorted({cell(row['split']) for row in rows}))}",
            f"- Evidence class: {cell(data.get('evidence_class', 'not supplied'))}",
            f"- Promotion eligible: {cell(data.get('promotion_eligible', 'not supplied'))}",
        ]
        for key in ("split", "splits", "dataset", "dataset_kind", "n_confirm", "n_train", "n_gold_rows"):
            if key in data:
                lines.append(f"- {key}: {cell(data[key])}")
        if schema == SCHEMA_LLM:
            lines.append(
                "- Floor: not provided by this source; verdicts are NO_COMPARABLE_FLOOR. "
                "The different frame encoding prevents a controlled cross-file comparison."
            )
        lines += [
            "",
            "| Split | Arm | Accuracy | Floor (majority) | Δ vs floor | Verdict |"
            + (" N | Unparsed | Latency p50 (ms) |" if schema == SCHEMA_LLM else ""),
            "| --- | --- | ---: | ---: | ---: | --- |"
            + (" ---: | ---: | ---: |" if schema == SCHEMA_LLM else ""),
        ]
        for row in rows:
            detail = ""
            if schema == SCHEMA_LLM:
                detail = (
                    f" {row['n']} | {row['unparsed']} | "
                    f"{number(row['latency_p50_ms'])} |"
                )
            lines.append(
                f"| {cell(row['split'])} | {cell(row['arm'])} | "
                f"{number(row['value'])} | {number(row['floor'])} | "
                f"{number(row['delta'], signed=True)} | {row['verdict']} |" + detail
            )
        lines += ["", "### Caveats", ""]
        any_caveat = False
        for key in CAVEAT_FIELDS:
            if key in data:
                # Preserve the source wording; continuation lines stay in the bullet.
                value = safe(data[key]).replace("\n", "\n  ")
                lines.append(f"- **{key}:** {value}")
                any_caveat = True
        if not any_caveat:
            lines.append("- No caveat fields supplied in this source.")

    lines += ["", "## Not rendered", ""]
    if unrendered:
        for filename, schema in unrendered:
            lines.append(
                f"- {CODE}{cell(filename)}{CODE} — unknown schema: {CODE}{cell(schema)}{CODE}"
            )
    else:
        lines.append("- None.")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("docs/SCORECARD.md"))
    parser.add_argument("--source-sha", default=None)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    output = render_scorecard(args.results, source_sha=args.source_sha).encode("utf-8")
    if args.check:
        return 0 if args.out.is_file() and args.out.read_bytes() == output else 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
