"""Real Hermes recovery episodes -> hermes_recovery action space (T3b).

Schema-level code only. Raw message/tool text is read locally, reduced to
typed numeric features and category labels, and never written anywhere by this
module. Callers write episodes under ~/.z0int (private), never into git.

Episode = one failed tool result in a Hermes session + the agent's next
decision turn + the observable downstream outcome.

Mapping of observed next-decision categories (``raw_action``) onto
ACTIONS = (retry, restart_sandbox, escalate, noop, page_human):

  raw_action        strict            loose        why ambiguous
  retry_exact       retry             retry        -
  retry_modified    (unknown)         retry        same tool, edited args = "fix then retry"
  restart           restart_sandbox   restart_sandbox  kill/restart/process-kill command
  delegate          escalate          escalate     delegate_task / a2a_call to another agent
  clarify           page_human        page_human   explicit ask-the-human tool
  stop_interactive  page_human        page_human   final text to a present human after failure
  stop_headless     (unknown)         noop         cron/subagent run ends after failure = give up
  human_interrupt   (unknown)         page_human   human stepped in before the agent acted
  switch_tool       (unknown)         noop         moved on with a different tool (diagnose / route around)
  none              (unknown)         (unknown)    session ended with no further turn

Label = mapped action only when the downstream outcome is verified good;
otherwise the episode keeps label -1 (unknown) with its outcome recorded.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .schema import ACTIONS
from .task import _frame, n_features

RAW_ACTIONS = (
    "retry_exact",
    "retry_modified",
    "restart",
    "delegate",
    "clarify",
    "stop_interactive",
    "stop_headless",
    "human_interrupt",
    "switch_tool",
    "none",
)

STRICT_MAP: dict[str, str | None] = {
    "retry_exact": "retry",
    "retry_modified": None,
    "restart": "restart_sandbox",
    "delegate": "escalate",
    "clarify": "page_human",
    "stop_interactive": "page_human",
    "stop_headless": None,
    "human_interrupt": None,
    "switch_tool": None,
    "none": None,
}
LOOSE_MAP: dict[str, str | None] = {
    **STRICT_MAP,
    "retry_modified": "retry",
    "stop_headless": "noop",
    "human_interrupt": "page_human",
    "switch_tool": "noop",
}

INTERACTIVE_SOURCES = frozenset({"cli", "desktop", "tui"})
SOURCES = ("cron", "interactive", "a2a", "subagent", "kanban", "other")
TOOL_FAMILIES = ("terminal", "mcp", "file", "search", "web", "code", "agent", "process", "skill", "other")

_TRANSIENT = re.compile(
    r"timed?[ _-]?out|timeout|rate[ _-]?limit|\b429\b|\b50[234]\b|temporar|try again|"
    r"econnreset|connection reset|overloaded|unavailable|eai_again|network is unreachable|"
    r"read operation timed out|too many requests",
    re.I,
)
_SANDBOX = re.compile(
    r"connection refused|econnrefused|no such process|not running|daemon|socket|"
    r"container .{0,40}(exited|not found)|session .{0,20}not found|process .{0,30}(exited|died)|"
    r"\bkilled\b|broken pipe|server disconnected|not connected|connection closed|no such session",
    re.I,
)
_POLICY = re.compile(
    r"approval|denied by|blocked by|requires? approval|user (declined|denied|rejected)|"
    r"not allowed by|by policy|dangerous command",
    re.I,
)
_KNOWN_HARD = re.compile(
    r"no such file|not found|syntax ?error|invalid|permission denied|traceback|exception|"
    r"cannot|could not|unable to|failed|error:|typeerror|valueerror|keyerror|modulenotfound|"
    r"command not found|unknown|unexpected|missing|conflict|already exists|exit code",
    re.I,
)
_RESTART_CMD = re.compile(
    r"\b(restart|pkill|killall|kill\s+-?\d*|reboot|systemctl\s+(--user\s+)?(re)?start|"
    r"docker(\s+compose)?\s+(re)?start|docker(\s+compose)?\s+up|service\s+\S+\s+restart|"
    r"tmux\s+kill|launchctl\s+kickstart|respawn)\b",
    re.I,
)


def tool_family(name: str | None) -> str:
    n = (name or "").lower()
    if n == "terminal":
        return "terminal"
    if n.startswith("mcp__"):
        return "mcp"
    if n in {"read_file", "write_file", "patch"}:
        return "file"
    if n in {"search_files", "session_search", "tool_search", "tool_describe"}:
        return "search"
    if n.startswith("web_") or n.startswith("browser") or n == "vision_analyze":
        return "web"
    if n in {"execute_code", "tool_call"}:
        return "code"
    if n in {"delegate_task", "a2a_call", "a2a_history", "a2a_list"}:
        return "agent"
    if n.startswith("process"):
        return "process"
    if n.startswith("skill"):
        return "skill"
    return "other"


def source_bucket(source: str | None) -> str:
    s = (source or "").lower()
    if s in INTERACTIVE_SOURCES:
        return "interactive"
    if s in SOURCES:
        return s
    return "other"


def args_hash(arguments: Any) -> str:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (json.JSONDecodeError, ValueError):
            pass
    blob = json.dumps(arguments, sort_keys=True, default=str) if not isinstance(arguments, str) else arguments
    return hashlib.sha256(blob.encode("utf-8", errors="replace")).hexdigest()[:16]


def parse_args(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            v = json.loads(arguments)
            return v if isinstance(v, dict) else {}
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


def is_restart_call(name: str | None, arguments: Any) -> bool:
    n = (name or "").lower()
    a = parse_args(arguments)
    if n.startswith("process"):
        return str(a.get("action", "")).lower() in {"kill", "restart", "stop", "start"}
    if n == "terminal":
        return bool(_RESTART_CMD.search(str(a.get("command", ""))[:2000]))
    return False


@dataclass
class ToolResult:
    """Typed view of one tool result. Text fields are transient (never stored)."""

    failed: bool
    policy: bool = False
    benign_nonzero: bool = False
    exit_code: int | None = None
    text: str = ""  # local only: error/output tail used for regex classes


def classify_result(
    tool_name: str | None,
    content_head: str | None,
    j_error: Any,
    j_exit: Any,
    j_status: Any,
    j_success: Any,
    j_exit_meaning: Any,
    j_approval_pending: Any,
    j_smart_denied: Any,
    output_tail: str | None,
) -> ToolResult:
    """Decide failed/policy/benign from structured JSON fields first, text second."""
    head = content_head or ""
    tail = output_tail or ""
    policy = bool(j_approval_pending) or bool(j_smart_denied)
    exit_code: int | None
    try:
        exit_code = int(j_exit) if j_exit is not None else None
    except (TypeError, ValueError):
        exit_code = None
    err = j_error if j_error not in (None, "", 0, False, "null") else None
    failed = False
    if err is not None:
        failed = True
    if exit_code not in (None, 0):
        failed = True
    if str(j_success).lower() in {"0", "false"}:
        failed = True
    if str(j_status or "").lower() in {"error", "failed", "failure", "timeout", "denied", "blocked"}:
        failed = True
    stripped = head.lstrip()
    if not stripped.startswith("{"):
        low = stripped[:200].lower()
        if low.startswith(("error", "failed", "exception", "traceback", "mcp error", "tool error")) or (
            "error" in low[:60] and ("mcp" in low[:60] or "status" in low[:60])
        ):
            failed = True
    if policy:
        failed = True
    benign = bool(j_exit_meaning) and exit_code not in (None, 0) and err is None
    text = f"{err or ''}\n{tail}" if failed else ""
    if failed and not stripped.startswith("{"):
        text = stripped[:1500]
    return ToolResult(failed=failed, policy=policy, benign_nonzero=benign, exit_code=exit_code, text=text)


@dataclass
class ErrorClass:
    transient: bool
    sandbox_dead: bool
    policy: bool
    hard: bool
    unfamiliar: bool
    benign: bool

    def name(self) -> str:
        if self.policy:
            return "policy"
        if self.sandbox_dead:
            return "sandbox"
        if self.benign:
            return "benign"
        if self.transient:
            return "transient"
        if self.unfamiliar:
            return "unfamiliar"
        return "hard"


def error_class(res: ToolResult) -> ErrorClass:
    t = res.text[-3000:] if res.text else ""
    policy = res.policy or bool(_POLICY.search(t))
    sandbox = bool(_SANDBOX.search(t)) or res.exit_code in (137, 143)
    transient = bool(_TRANSIENT.search(t)) or res.exit_code == 124
    known = bool(_KNOWN_HARD.search(t)) or res.exit_code in (1, 2, 126, 127)
    unfamiliar = not (policy or sandbox or transient or known or res.benign_nonzero)
    hard = not (transient or res.benign_nonzero)
    return ErrorClass(
        transient=transient,
        sandbox_dead=sandbox,
        policy=policy,
        hard=hard,
        unfamiliar=unfamiliar,
        benign=res.benign_nonzero,
    )


EXIT_BUCKETS = ("none", "1", "2", "126_127", "124_137_143", "other")


def exit_bucket(code: int | None) -> str:
    if code is None:
        return "none"
    if code in (1, 2):
        return str(code)
    if code in (126, 127):
        return "126_127"
    if code in (124, 137, 143):
        return "124_137_143"
    return "other"


@dataclass
class RealEpisode:
    """One mined decision point. Only numeric/categorical fields; safe to store privately."""

    episode_id: str
    lineage: str
    source: str
    tool_family: str
    err: dict[str, bool]
    exit_bucket: str
    consec_fail_same_tool: int
    session_fail_count: int
    turn_idx: int
    n_parallel: int
    n_failed_in_turn: int
    last_action: str  # previous mapped action in session, or noop
    raw_action: str
    local_recovered: bool | None
    session_complete: bool | None
    ts: float
    extra: dict[str, Any] = field(default_factory=dict)

    def label(self, mapping: str = "strict") -> int:
        m = STRICT_MAP if mapping == "strict" else LOOSE_MAP
        a = m.get(self.raw_action)
        if a is None or not self.good_outcome():
            return -1
        return ACTIONS.index(a)

    def good_outcome(self) -> bool:
        return bool(self.local_recovered) and self.session_complete is not False

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "lineage": self.lineage,
            "source": self.source,
            "tool_family": self.tool_family,
            "err": self.err,
            "exit_bucket": self.exit_bucket,
            "consec_fail_same_tool": self.consec_fail_same_tool,
            "session_fail_count": self.session_fail_count,
            "turn_idx": self.turn_idx,
            "n_parallel": self.n_parallel,
            "n_failed_in_turn": self.n_failed_in_turn,
            "last_action": self.last_action,
            "raw_action": self.raw_action,
            "local_recovered": self.local_recovered,
            "session_complete": self.session_complete,
            "ts": self.ts,
            "extra": self.extra,
        }


BUDGET_TURNS = 90.0  # Hermes default iteration budget scale used for elapsed/budget


def gym_frame(ep: dict[str, Any]) -> np.ndarray:
    """Project a real episode onto the exact gym feature layout (task._frame)."""
    e = ep["err"]
    elapsed = min(ep["turn_idx"] / BUDGET_TURNS, 1.0)
    last = ep.get("last_action") or "noop"
    return _frame(
        sandbox_alive=0.0 if e["sandbox_dead"] else 1.0,
        retry=min(int(ep["consec_fail_same_tool"]), 4),
        budget=max(0.0, 1.0 - elapsed),
        elapsed=elapsed,
        transient=float(e["transient"]),
        hard=float(e["hard"]),
        unfamiliar=float(e["unfamiliar"]),
        policy_hit=float(e["policy"]),
        already_ok=float(e["benign"]),
        cue=0.0,
        last_action=ACTIONS.index(last) if last in ACTIONS else ACTIONS.index("noop"),
    )


def rich_features(ep: dict[str, Any]) -> np.ndarray:
    """Gym frame + real-only context (source, tool family, exit bucket, counts)."""
    base = gym_frame(ep)
    src = np.array([ep["source"] == s for s in SOURCES], dtype=np.float64)
    fam = np.array([ep["tool_family"] == f for f in TOOL_FAMILIES], dtype=np.float64)
    exb = np.array([ep["exit_bucket"] == b for b in EXIT_BUCKETS], dtype=np.float64)
    cnt = np.array(
        [
            min(ep["session_fail_count"], 20) / 20.0,
            min(ep["n_parallel"], 10) / 10.0,
            min(ep["n_failed_in_turn"], 10) / 10.0,
            float(ep["n_failed_in_turn"] == ep["n_parallel"]),
        ],
        dtype=np.float64,
    )
    return np.concatenate([base, src, fam, exb, cnt])


def n_rich_features() -> int:
    return n_features() + len(SOURCES) + len(TOOL_FAMILIES) + len(EXIT_BUCKETS) + 4
