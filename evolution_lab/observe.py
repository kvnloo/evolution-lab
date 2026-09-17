"""Map harness operational events to sanitized Hermes recovery fields."""

from __future__ import annotations

import re
from typing import Any

from .schema import SECRET_FIELD_NAMES, observation_is_sanitized

_HARNESS_ALIASES = {"omp", "pi", "hermes", "cursor", "evolution_lab"}

_TRANSIENT = re.compile(
    r"(rate.?limit|429|503|502|504|timeout|timed out|temporar|"
    r"connection reset|econnreset|eagain|unavailable|overloaded|"
    r"try again|retry later|socket hang up)",
    re.I,
)
_SANDBOX_DEAD = re.compile(
    r"(sandbox.?dead|process.?exit|hub .*(not ready|failed|exited)|"
    r"eval kernel|session crashed|worker died|pty closed|"
    r"container (stopped|exited)|no such process)",
    re.I,
)
_POLICY = re.compile(
    r"(policy|approval denied|blocked by|fail.?closed|"
    r"permission denied|not allowed|forbidden|dangerous|"
    r"secret.?bearing|vault|credential|api[_-]?key|token leak)",
    re.I,
)
_HARD = re.compile(
    r"(unknown error|fatal|panic|assertion|segmentation|"
    r"unhandled|internal error|not implemented|"
    r"cannot recover|persistent)",
    re.I,
)
_ALREADY_OK = re.compile(r"(already (ok|done|complete|succeeded)|no.?op needed)", re.I)


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _classify_text(text: str) -> dict[str, float]:
    flags = {
        "transient": 0.0,
        "hard": 0.0,
        "policy_hit": 0.0,
        "already_ok": 0.0,
        "unfamiliar": 0.0,
    }
    if not text.strip():
        return flags
    if _ALREADY_OK.search(text):
        flags["already_ok"] = 1.0
        return flags
    if _POLICY.search(text):
        flags["policy_hit"] = 1.0
        return flags
    if _SANDBOX_DEAD.search(text):
        return flags
    if _TRANSIENT.search(text):
        flags["transient"] = 1.0
        return flags
    if _HARD.search(text):
        flags["hard"] = 1.0
        return flags
    flags["unfamiliar"] = 1.0
    return flags


def observe_event(
    event: dict[str, Any],
    *,
    harness: str = "omp",
    retry: int = 0,
    budget: float = 1.0,
    last_action: str | None = None,
) -> dict[str, Any]:
    """Convert a harness event into sanitized recovery observation fields."""
    harness_key = (harness or "omp").strip().lower()
    if harness_key not in _HARNESS_ALIASES:
        harness_key = "omp"

    kind = str(event.get("kind") or event.get("type") or "tool_error").strip().lower()
    message = str(
        event.get("message")
        or event.get("error")
        or event.get("detail")
        or event.get("stderr")
        or ""
    )
    tool = str(event.get("tool") or event.get("tool_name") or "")
    is_error = bool(event.get("is_error", event.get("isError", True)))

    fields: dict[str, Any] = {
        "sandbox_alive": 1.0,
        "retry": _coerce_int(event.get("retry"), retry),
        "budget": _coerce_float(event.get("budget"), budget),
        "elapsed": _coerce_float(event.get("elapsed"), 1.0),
        "transient": 0.0,
        "hard": 0.0,
        "unfamiliar": 0.0,
        "policy_hit": 0.0,
        "already_ok": 0.0,
        "cue": _coerce_float(event.get("cue"), 0.0),
        "cue_seen": _coerce_float(event.get("cue_seen"), 0.0),
    }
    if last_action is not None:
        fields["last_action"] = last_action
    elif event.get("last_action") is not None:
        fields["last_action"] = event["last_action"]

    text = " ".join(x for x in (kind, tool, message) if x)
    flags = _classify_text(text)

    if kind in {"sandbox_dead", "process_dead", "hub_dead", "eval_crash"}:
        fields["sandbox_alive"] = 0.0
    elif _SANDBOX_DEAD.search(text):
        fields["sandbox_alive"] = 0.0
    elif kind in {"policy_hit", "approval_denied", "secret_detected"}:
        fields["policy_hit"] = 1.0
    elif kind in {"already_ok", "noop"}:
        fields["already_ok"] = 1.0
    elif kind in {"transient", "rate_limit", "timeout"}:
        fields["transient"] = 1.0
    elif kind in {"hard", "unknown"}:
        fields["hard"] = 1.0
    elif kind in {"unfamiliar"}:
        fields["unfamiliar"] = 1.0
    else:
        for key, value in flags.items():
            if value > fields[key]:
                fields[key] = value

    if not is_error and kind == "tool_error":
        fields["already_ok"] = 1.0
        fields["transient"] = 0.0
        fields["hard"] = 0.0
        fields["unfamiliar"] = 0.0

    for key in (
        "sandbox_alive",
        "transient",
        "hard",
        "unfamiliar",
        "policy_hit",
        "already_ok",
        "cue",
        "cue_seen",
    ):
        if key in event:
            fields[key] = _coerce_float(event[key], fields[key])

    if not observation_is_sanitized(fields):
        bad = sorted(SECRET_FIELD_NAMES.intersection(fields))
        raise ValueError(f"secret-bearing observation fields: {bad}")

    return {
        "harness": harness_key,
        "kind": kind,
        "tool": tool or None,
        "fields": fields,
    }


def action_guidance(action: str) -> dict[str, str]:
    guides = {
        "retry": (
            "Retry the same bounded operation once after a short backoff. "
            "Do not widen scope or repeat more than the retry budget allows."
        ),
        "restart_sandbox": (
            "Restart the isolated execution surface (eval kernel, hub process, or "
            "fresh shell session) before retrying. Preserve user desktop focus."
        ),
        "escalate": (
            "Escalate to a stronger planner/model or delegate_task with typed "
            "context. Do not paste secrets into the handoff."
        ),
        "noop": (
            "No recovery action required. Continue the current plan without "
            "restarting or re-prompting the user."
        ),
        "page_human": (
            "Stop autonomous recovery and page the human with a concise, typed "
            "summary. This is for policy hits or irreversible risk."
        ),
    }
    if action not in guides:
        raise ValueError(f"unknown recovery action {action!r}")
    return {"action": action, "guidance": guides[action]}
