"""z0int preflight — local cognition filter before residual model allocation.

Returns a JSON packet:
  route=local  → stop; frontier tokens avoided (counterfactual)
  route=model  → residual WorkRequirement for Kerdoios (capability_id set)

Does not execute flies inside Kerdoios. Does not call providers.
"""

from __future__ import annotations

import time
from typing import Any

from .next_action_gpu import decide_next_action

# Map coarse next-action families → capability_id taxonomy shared with Kerdoios.
FAMILY_TO_CAPABILITY: dict[str, str] = {
    "DELEGATE": "coding.delegate",
    "EXECUTE": "coding.execute",
    "EDIT": "coding.edit",
    "READ_SEARCH": "coding.file_relevance",
    "VERIFY": "coding.verify_needed",
    "WEB": "coding.web",
    "ABSTAIN": "coding.abstain",
    "RESPOND": "coding.respond",
}

# Counterfactual frontier burn if the host would have called a premium model.
# Coarse defaults; host may override via baseline_* args. Not measured L2.
DEFAULT_COUNTERFACTUAL: dict[str, tuple[int, int]] = {
    "coding.delegate": (1800, 250),
    "coding.execute": (1200, 200),
    "coding.edit": (2500, 600),
    "coding.file_relevance": (2000, 300),
    "coding.verify_needed": (1500, 200),
    "coding.web": (1600, 400),
    "coding.abstain": (800, 80),
    "coding.respond": (1500, 400),
    "coding.next_action": (1800, 250),
    "blender.scene_reasoning": (4200, 800),
    "blender.need_render": (900, 120),
    "coding.failure_recovery": (1400, 250),
}


def capability_for_family(family: str | None, *, hint: str | None = None) -> str:
    if hint:
        return hint
    if family and family in FAMILY_TO_CAPABILITY:
        return FAMILY_TO_CAPABILITY[family]
    return "coding.next_action"


def counterfactual_tokens(
    capability_id: str,
    *,
    baseline_input: int | None = None,
    baseline_output: int | None = None,
) -> dict[str, int]:
    default_in, default_out = DEFAULT_COUNTERFACTUAL.get(capability_id, (1800, 250))
    return {
        "estimated_input": int(baseline_input if baseline_input is not None else default_in),
        "estimated_output": int(baseline_output if baseline_output is not None else default_out),
    }


def residual_work_requirement(
    capability_id: str,
    *,
    estimated_input_tokens: int,
    estimated_output_tokens: int,
    label: str | None = None,
) -> dict[str, Any]:
    """Minimal WorkRequirement-shaped dict for Kerdoios (capability_id + coarse axes)."""
    # Coarse capability axes still required by current Kerdoios hard filters.
    coding = 0.7
    reasoning = 0.6
    vision = False
    if capability_id.startswith("blender."):
        coding = 0.5
        reasoning = 0.85
        vision = True
    if capability_id in ("coding.edit", "coding.patch"):
        coding = 0.85
    if capability_id in ("coding.verify_needed", "coding.failure_recovery"):
        reasoning = 0.7
    return {
        "capability_id": capability_id,
        "coding": coding,
        "reasoning": reasoning,
        "vision": vision,
        "tool_use": True,
        "estimated_input_tokens": estimated_input_tokens,
        "estimated_output_tokens": estimated_output_tokens,
        "label_hint": label,
        "mode": "balanced",
    }


def preflight(
    text: str,
    *,
    capability_hint: str | None = None,
    baseline_input_tokens: int | None = None,
    baseline_output_tokens: int | None = None,
    pack=None,
) -> dict[str, Any]:
    """Run local cascade; return route + capability packet.

    Evidence level: L0 specialist decision (imitation + coverage policy).
    Counterfactual tokens are estimates until receipt joins exist.
    """
    t0 = time.perf_counter()
    decision = decide_next_action(text, pack=pack)
    label = str(decision.get("label") or "")
    capability_id = capability_for_family(label, hint=capability_hint)
    cf = counterfactual_tokens(
        capability_id,
        baseline_input=baseline_input_tokens,
        baseline_output=baseline_output_tokens,
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0
    base = {
        "schema": "z0int.preflight.v1",
        "ok": bool(decision.get("ok", True)),
        "capability_id": capability_id,
        "label": label,
        "confidence": float(decision.get("p") or 0.0),
        "margin": decision.get("margin"),
        "reason": decision.get("reason"),
        "latency_ms": round(latency_ms, 3),
        "decision": {
            "route": decision.get("route"),
            "escalate_to": decision.get("escalate_to"),
            "fallback": decision.get("fallback"),
            "source": decision.get("source"),
        },
        "context_refs": [],
        "evidence_level": "L0_imitation",
    }
    if decision.get("route") == "local":
        avoided = cf["estimated_input"] + cf["estimated_output"]
        return {
            **base,
            "route": "local",
            "work_requirement": None,
            "estimated_frontier_tokens_avoided": avoided,
            "counterfactual": {
                "would_call": "frontier",
                "estimated_input_tokens": cf["estimated_input"],
                "estimated_output_tokens": cf["estimated_output"],
            },
            "kerdoios": None,
        }
    # Residual model path — Kerdoios owns allocation.
    req = residual_work_requirement(
        capability_id,
        estimated_input_tokens=cf["estimated_input"],
        estimated_output_tokens=cf["estimated_output"],
        label=label,
    )
    return {
        **base,
        "route": "model",
        "work_requirement": req,
        "estimated_context_tokens": cf["estimated_input"],
        "estimated_frontier_tokens_avoided": 0,
        "counterfactual": None,
        "kerdoios": {
            "action": "plan_residual",
            "capability_id": capability_id,
            "note": "Feed work_requirement into kerdoios plan; harness executes.",
        },
    }


def receipt_from_preflight(
    preflight_row: dict[str, Any],
    *,
    provider: str | None = None,
    model: str | None = None,
    completed: bool | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    latency_ms: float | None = None,
    actual_cost: float = 0.0,
) -> dict[str, Any]:
    """Build a Kerdoios Observation-shaped receipt (dict) from a turn.

    Local route → provider=z0int model=mb (or specialist id), tokens=0.
    """
    route = preflight_row.get("route")
    cap = preflight_row.get("capability_id")
    if route == "local":
        return {
            "provider": provider or "z0int",
            "model": model or "mb_local",
            "task_type": "capability",
            "capability_id": cap,
            "completed": True if completed is None else bool(completed),
            "actual_cost": 0.0,
            "retried": False,
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": latency_ms if latency_ms is not None else preflight_row.get("latency_ms"),
            "counterfactual": preflight_row.get("counterfactual"),
            "estimated_frontier_tokens_avoided": preflight_row.get(
                "estimated_frontier_tokens_avoided"
            ),
        }
    return {
        "provider": provider or "unknown",
        "model": model or "unknown",
        "task_type": "capability",
        "capability_id": cap,
        "completed": bool(completed) if completed is not None else False,
        "actual_cost": float(actual_cost),
        "retried": False,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": latency_ms,
        "counterfactual": None,
        "estimated_frontier_tokens_avoided": 0,
    }
