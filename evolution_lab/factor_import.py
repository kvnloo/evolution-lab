"""Import existing z0 mechanism receipts into the factorized-stack experiment.

These adapters consume the source mechanisms' existing output shapes. They do
not modify source runtimes or reinterpret model agreement as verification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Iterable, Mapping

from .factorized_stack import ArmSummary, FactorReceipt


@dataclass(frozen=True)
class ImportResult:
    factor: str
    observations: int
    measurements: Mapping[str, Any] = field(default_factory=dict)
    receipts: tuple[FactorReceipt, ...] = ()


def _rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def import_sol_pi_observation_ledger(
    rows: Iterable[Mapping[str, Any]],
    *,
    revision: str,
    trace_id: str = "sol-pi-observation-ledger",
) -> ImportResult:
    """Summarize the real sol-pi-hermes ObservationPack ledger shape."""
    data = _rows(rows)
    relevant = [r for r in data if r.get("event") in {"full", "placeholder"}]
    placeholders = [r for r in relevant if r.get("event") == "placeholder"]
    full = [r for r in relevant if r.get("event") == "full"]

    original_placeholder_bytes = sum(
        int(r.get("originalBytes") or 0) for r in placeholders
    )
    estimated_placeholder_bytes = sum(
        int(r.get("placeholderTokens") or 0) * 4 for r in placeholders
    )
    estimated_avoided = max(0, original_placeholder_bytes - estimated_placeholder_bytes)

    receipt = FactorReceipt(
        factor="sol_pi_observation_pack",
        trace_id=trace_id,
        schema="sol_pi.observation_pack.ledger.v1",
        source="kvnloo/sol-pi-hermes",
        revision=revision,
    )
    return ImportResult(
        factor="sol_pi_observation_pack",
        observations=len(relevant),
        measurements={
            "full_events": len(full),
            "placeholder_events": len(placeholders),
            "original_placeholder_bytes": original_placeholder_bytes,
            "estimated_placeholder_bytes": estimated_placeholder_bytes,
            "estimated_context_bytes_avoided": estimated_avoided,
            "unique_observations": len(
                {str(r.get("id")) for r in relevant if r.get("id")}
            ),
            "measurement_note": "placeholder byte estimate uses source placeholderTokens*4",
        },
        receipts=(receipt,),
    )


def import_rlm_evidence_ab(
    rows: Iterable[Mapping[str, Any]],
    *,
    revision: str,
    trace_id: str = "rlm-evidence-ab",
) -> dict[str, Any]:
    """Summarize the real OMP evidence-ab.jsonl row shape."""
    data = _rows(rows)
    valid = [
        r
        for r in data
        if r.get("arm") in {"full", "fixed8k", "search8k"}
        and r.get("backend") in {"mechanism", "live"}
    ]
    if not valid:
        raise ValueError("no RLM evidence A/B rows")

    def summarize(arm: str) -> dict[str, Any]:
        group = [r for r in valid if r.get("arm") == arm]
        if not group:
            return {
                "rows": 0,
                "pass_rate": None,
                "mean_granted_bytes": None,
                "mean_prompt_bytes": None,
                "mean_total_tokens": None,
                "mean_elapsed_ms": None,
            }
        tokens = [float(r["totalTokens"]) for r in group if r.get("totalTokens") is not None]
        return {
            "rows": len(group),
            "pass_rate": sum(bool(r.get("pass")) for r in group) / len(group),
            "mean_granted_bytes": mean(float(r.get("grantedBytes") or 0) for r in group),
            "mean_prompt_bytes": mean(float(r.get("promptBytes") or 0) for r in group),
            "mean_total_tokens": mean(tokens) if tokens else None,
            "mean_elapsed_ms": mean(float(r.get("elapsedMs") or 0) for r in group),
        }

    summary = {arm: summarize(arm) for arm in ("full", "fixed8k", "search8k")}
    backends = sorted({str(r.get("backend")) for r in valid})
    models = sorted({str(r.get("model")) for r in valid})

    receipt = FactorReceipt(
        factor="rlm_evidence_addressing",
        trace_id=trace_id,
        schema="omp.rlm.evidence_ab.v1",
        source="kvnloo/oh-my-pi",
        revision=revision,
    )
    return {
        "factor": "rlm_evidence_addressing",
        "observations": len(valid),
        "backends": backends,
        "models": models,
        "arms": summary,
        "receipts": [receipt.to_dict()],
        "scope_note": (
            "search8k tests addressing with a supplied lexical query; "
            "live mode is descriptive, not native OMP end-to-end"
        ),
    }


def rlm_arm_summaries(
    rows: Iterable[Mapping[str, Any]],
    *,
    control_id: str = "control",
    candidate_id: str = "rlm_evidence_addressing__only",
) -> tuple[ArmSummary, ArmSummary]:
    """Project RLM full/search rows into the common quality+cost gate."""
    data = _rows(rows)
    full = [r for r in data if r.get("arm") == "full"]
    search = [r for r in data if r.get("arm") == "search8k"]
    if not full or not search:
        raise ValueError("full and search8k rows are required")

    def keys(group: list[dict[str, Any]]) -> set[tuple[Any, Any]]:
        return {(r.get("run"), r.get("workload")) for r in group}

    if keys(full) != keys(search):
        raise ValueError("RLM arms do not cover the same run/workload cohort")

    def make(arm_id: str, group: list[dict[str, Any]]) -> ArmSummary:
        tokens = [float(r["totalTokens"]) for r in group if r.get("totalTokens") is not None]
        elapsed = sorted(float(r.get("elapsedMs") or 0) for r in group)
        p95_i = max(0, min(len(elapsed) - 1, int(round(0.95 * (len(elapsed) - 1)))))
        return ArmSummary(
            arm_id=arm_id,
            n_work_items=len(group),
            verified_success_rate=sum(bool(r.get("pass")) for r in group) / len(group),
            mean_input_tokens=mean(tokens) if tokens else None,
            mean_context_bytes=mean(float(r.get("grantedBytes") or 0) for r in group),
            p95_latency_ms=elapsed[p95_i],
            measurements={
                "source_success_semantics": "RLM evidence A/B pass field",
                "backend": sorted({str(r.get("backend")) for r in group}),
            },
        )

    return make(control_id, full), make(candidate_id, search)


def import_state_packet(
    packet: Mapping[str, Any],
    *,
    revision: str,
    trace_id: str = "state-packet",
) -> ImportResult:
    """Normalize the current z0int ContextPacket/StatePacket-compatible shape.

    This importer only measures state construction/provenance. A packet by
    itself is not a verified task outcome, so it cannot independently qualify
    the memory factor for promotion.
    """
    row = dict(packet)
    schema = str(row.get("schema") or "")
    if not schema.startswith("z0int.context_resolve") and not schema.startswith("z0int.state"):
        raise ValueError("unsupported StatePacket/context schema")
    evidence = row.get("evidence") or []
    gaps = row.get("unresolved_gaps") or row.get("blocking_unknowns") or []
    contradictions = row.get("contradictions") or []
    if not isinstance(evidence, list) or not isinstance(gaps, list) or not isinstance(contradictions, list):
        raise ValueError("invalid packet collections")

    provenance_missing = 0
    for item in evidence:
        if not isinstance(item, Mapping):
            provenance_missing += 1
            continue
        if not item.get("source_id") or not item.get("source_version") or not item.get("locator"):
            provenance_missing += 1

    measurements = dict(row.get("measurements") or {})
    receipt = FactorReceipt(
        factor="state_packet_memory",
        trace_id=trace_id,
        schema=schema,
        source="kvnloo/z0intelligence",
        revision=revision,
    )
    return ImportResult(
        factor="state_packet_memory",
        observations=1,
        measurements={
            "evidence_count": len(evidence),
            "unresolved_gap_count": len(gaps),
            "contradiction_count": len(contradictions),
            "provenance_missing": provenance_missing,
            "source_reads": measurements.get("source_reads"),
            "latency_ms": measurements.get("latency_ms"),
            "packet_bytes": measurements.get("packet_bytes"),
            "cache_hit": measurements.get("cache_hit"),
            "qualification_note": "state construction alone is not verified task success",
        },
        receipts=(receipt,),
    )


def import_slm_tournament_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    composition: str,
    revision: str,
    trace_id: str = "local-slm-tournament",
) -> ImportResult:
    """Normalize Evolution Lab local-tool-tournament row receipts.

    The tournament already separates compiler legality from learned choice and
    exposes hard_failure, correct, latency, token, GPU, VRAM and retry fields.
    """
    data = [
        dict(r)
        for r in rows
        if r.get("schema") == "z0int.tool_tournament.row.v1"
        and r.get("composition") == composition
    ]
    if not data:
        raise ValueError("no matching local SLM tournament rows")

    prompt_tokens = [float(r["prompt_tokens"]) for r in data if r.get("prompt_tokens") is not None]
    completion_tokens = [float(r["completion_tokens"]) for r in data if r.get("completion_tokens") is not None]
    gpu_ms = [float(r["gpu_ms"]) for r in data if r.get("gpu_ms") is not None]
    vram = [float(r["vram_peak_mb"]) for r in data if r.get("vram_peak_mb") is not None]

    receipt = FactorReceipt(
        factor="local_slm_policy",
        trace_id=trace_id,
        schema="z0int.local_tool_tournament.v1",
        source="kvnloo/evolution-lab",
        revision=revision,
    )
    return ImportResult(
        factor="local_slm_policy",
        observations=len(data),
        measurements={
            "composition": composition,
            "correct_rate": sum(bool(r.get("correct")) for r in data) / len(data),
            "hard_failures": sum(bool(r.get("hard_failure")) for r in data),
            "invalid_calls": sum(bool(r.get("invalid_call")) for r in data),
            "dangerous_picks": sum(bool(r.get("dangerous_pick")) for r in data),
            "abstention_rate": sum(bool(r.get("abstained")) for r in data) / len(data),
            "mean_latency_ms": mean(float(r.get("latency_ms") or 0) for r in data),
            "mean_retries": mean(float(r.get("retries") or 0) for r in data),
            "mean_prompt_tokens": mean(prompt_tokens) if prompt_tokens else None,
            "mean_completion_tokens": mean(completion_tokens) if completion_tokens else None,
            "mean_gpu_ms": mean(gpu_ms) if gpu_ms else None,
            "peak_vram_mb": max(vram) if vram else None,
        },
        receipts=(receipt,),
    )


def import_bend_attestation_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    revision: str,
    trace_id: str = "bend-semantic-attestation",
) -> ImportResult:
    """Normalize Bend source->BendTT semantic-attestation results.

    A successful verifier exit is never enough. The candidate only has a
    semantically clean row when source_value == certified_value and the
    translation_mismatch flag is false.
    """
    data = _rows(rows)
    if not data:
        raise ValueError("no Bend attestation rows")

    mismatches = 0
    verifier_passes = 0
    semantic_passes = 0
    latencies: list[float] = []
    for row in data:
        adapter_pass = bool(row.get("adapter_pass") or row.get("verdict_pass"))
        if adapter_pass:
            verifier_passes += 1
        source = row.get("source_value")
        certified = row.get("certified_value")
        mismatch = bool(row.get("translation_mismatch")) or (
            source is not None and certified is not None and source != certified
        )
        if mismatch:
            mismatches += 1
        if adapter_pass and not mismatch and source is not None and certified is not None:
            semantic_passes += 1
        if row.get("latency_ms") is not None:
            latencies.append(float(row["latency_ms"]))

    receipt = FactorReceipt(
        factor="bend_structural_verifier",
        trace_id=trace_id,
        schema="bend.semantic_attestation.v1",
        source="kvnloo/bend + kvnloo/hermes-agent",
        revision=revision,
    )
    return ImportResult(
        factor="bend_structural_verifier",
        observations=len(data),
        measurements={
            "verifier_pass_rate": verifier_passes / len(data),
            "semantic_attestation_rate": semantic_passes / len(data),
            "translation_mismatches": mismatches,
            "mean_latency_ms": mean(latencies) if latencies else None,
            "hard_gate_pass": mismatches == 0 and semantic_passes == len(data),
            "measurement_note": (
                "source/certified agreement is authoritative for this factor; "
                "--verdict success alone is insufficient"
            ),
        },
        receipts=(receipt,),
    )


def import_rlm_querygen(
    rows: Iterable[Mapping[str, Any]],
    *,
    revision: str,
    trace_id: str = "rlm-querygen",
) -> ImportResult:
    """Normalize OMP RLM querygen.jsonl into the RLM factor.

    This is a retrieval-formation sub-gate only. It does not count as verified
    end-to-end task success and therefore cannot promote RLM by itself.
    """
    data = [
        dict(r)
        for r in rows
        if r.get("arm") in {"oracle", "lexical", "model"}
        and r.get("retrievalPass") is not None
    ]
    if not data:
        raise ValueError("no RLM query-generation rows")

    cohorts = {
        arm: {(r.get("run"), r.get("workload")) for r in data if r.get("arm") == arm}
        for arm in ("oracle", "lexical", "model")
    }
    if not cohorts["oracle"] or not cohorts["lexical"]:
        raise ValueError("oracle and lexical query-generation arms are required")
    if cohorts["oracle"] != cohorts["lexical"]:
        raise ValueError("oracle/lexical query-generation cohorts differ")
    if cohorts["model"] and cohorts["model"] != cohorts["oracle"]:
        raise ValueError("model query-generation cohort differs from oracle")

    def summary(arm: str) -> dict[str, Any]:
        group = [r for r in data if r.get("arm") == arm]
        if not group:
            return {"rows": 0, "retrieval_rate": None}
        generator_tokens = [
            float(r["generatorTotalTokens"])
            for r in group
            if r.get("generatorTotalTokens") is not None
        ]
        return {
            "rows": len(group),
            "retrieval_rate": sum(bool(r.get("retrievalPass")) for r in group) / len(group),
            "mean_pattern_hits": mean(float(r.get("patternHits") or 0) for r in group),
            "mean_granted_bytes": mean(float(r.get("grantedBytes") or 0) for r in group),
            "mean_generator_tokens": mean(generator_tokens) if generator_tokens else None,
            "mean_generator_ms": mean(float(r.get("generatorElapsedMs") or 0) for r in group),
            "mean_system_token_proxy": mean(float(r.get("systemTokenProxy") or 0) for r in group),
        }

    arms = {arm: summary(arm) for arm in ("oracle", "lexical", "model")}
    oracle_rate = float(arms["oracle"]["retrieval_rate"])
    lexical_rate = float(arms["lexical"]["retrieval_rate"])
    model_rate = arms["model"]["retrieval_rate"]
    provisional = None
    if model_rate is not None:
        retention = float(model_rate) / oracle_rate if oracle_rate else 0.0
        provisional = {
            "oracle_is_complete": oracle_rate == 1.0,
            "model_oracle_retention": retention,
            "model_not_below_lexical": float(model_rate) >= lexical_rate,
            "candidate": oracle_rate == 1.0 and retention >= 0.8 and float(model_rate) >= lexical_rate,
        }

    receipt = FactorReceipt(
        factor="rlm_evidence_addressing",
        trace_id=trace_id,
        schema="omp.rlm.querygen.v1",
        source="kvnloo/oh-my-pi",
        revision=revision,
    )
    return ImportResult(
        factor="rlm_evidence_addressing",
        observations=len(data),
        measurements={
            "arms": arms,
            "provisional_querygen_gate": provisional,
            "qualification_note": (
                "query-generation retrieval is a sub-gate only; "
                "native end-to-end verified task evidence is still required"
            ),
        },
        receipts=(receipt,),
    )
