"""T3b action-space v1: {noop, retry, edit_retry, switch_tool, ask, abort}.

Pre-registered in docs/t3b-action-space-v1-prereg.md. Everything here is pure and
operates on *structural skeletons* of the turns after a failure (tool families,
same-tool / same-args flags, argument-token overlap scores, success flags, one
regex bit over a final message). Raw message or tool text never enters this
module; the miner reduces text to these fields in-process and discards it.

Skeleton of one follow-up assistant turn::

    {"users_before": int, "fail_report": bool | None,
     "calls": [{"fam": str, "same_tool": bool, "exact": bool, "jac": float,
                "ovl": float, "path_shared": int, "restart": bool,
                "clarify": bool, "delegate": bool, "ok": bool | None}, ...]}
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from .real_recovery import EXIT_BUCKETS, SOURCES, TOOL_FAMILIES, gym_frame

ACTIONS_V1 = ("noop", "retry", "edit_retry", "switch_tool", "ask", "abort")
A1 = len(ACTIONS_V1)
IDX = {a: i for i, a in enumerate(ACTIONS_V1)}

# ---------------------------------------------------------------- token overlap (local only)

RETRY_SIM = 0.5  # same threshold as the T3b miner's retry_modified
RELATED_OVL = 0.25  # containment of failed-call salient value tokens
_STOP = frozenset({"true", "false", "null", "none", "json", "text", "utf-8", "http", "https"})
_TOK = re.compile(r"[A-Za-z0-9_./~-]+")


def _values(obj: Any, out: list[str], depth: int = 0) -> None:
    if depth > 6:
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _values(v, out, depth + 1)
    elif isinstance(obj, list):
        for v in obj[:50]:
            _values(v, out, depth + 1)
    elif obj is not None:
        out.append(str(obj)[:4000])


def salient_tokens(args: Any) -> frozenset:
    """Argument VALUE tokens (keys dropped), len>=4, not all-digit, not stop words."""
    vals: list[str] = []
    _values(args, vals)
    toks = set()
    for v in vals:
        for t in _TOK.findall(v):
            tl = t.lower().strip("./-")
            if len(tl) >= 4 and not tl.isdigit() and tl not in _STOP:
                toks.add(tl)
    return frozenset(toks)


def is_pathlike(t: str) -> bool:
    return len(t) >= 6 and ("/" in t or ("." in t and not t.replace(".", "").isdigit()))


def overlap(failed: frozenset, other: frozenset) -> tuple[float, int]:
    """(containment of failed tokens in other, number of shared path-like tokens)."""
    if not failed:
        return 0.0, 0
    sh = failed & other
    return len(sh) / len(failed), sum(1 for t in sh if is_pathlike(t))


def related(c: dict) -> bool:
    return c["ovl"] >= RELATED_OVL or c["path_shared"] >= 1


FAIL_REPORT = re.compile(
    r"\b(unable to|could ?n[o']t|can ?n[o']t|cannot|failed|not possible|blocked|denied|"
    r"did not succeed|was ?n[o']t able|not able to|unsuccessful|aborting|giving up|gave up)\b",
    re.I,
)

# ---------------------------------------------------------------- labelling rule (pre-registered)


def taken_action(next_turns: list[dict], *, interactive: bool) -> tuple[str | None, str]:
    """Action the agent took at the first decision turn after the failure.

    Returns (action or None if excluded, rule id that fired). Precedence R0..R9.
    """
    if not next_turns:
        return None, "R0_none"
    n = next_turns[0]
    if n["users_before"] > 0:
        return None, "R0_human_interrupt"
    calls = n["calls"]
    if not calls:
        if interactive:
            return "ask", "R1_stop_interactive"
        return ("abort", "R1_stop_headless_failreport") if n.get("fail_report") else ("noop", "R1_stop_headless_clean")
    if any(c["clarify"] for c in calls):
        return "ask", "R2_clarify"
    if any(c["delegate"] for c in calls):
        return "switch_tool", "R3_delegate"
    if any(c["restart"] for c in calls):
        return "edit_retry", "R4_restart"
    same = [c for c in calls if c["same_tool"]]
    if any(c["exact"] for c in same):
        return "retry", "R5_exact"
    if any(c["jac"] >= RETRY_SIM for c in same):
        return "edit_retry", "R6_similar_args"
    if any(related(c) for c in same):
        return "edit_retry", "R7_same_tool_same_target"
    if any(related(c) for c in calls if not c["same_tool"]):
        return "switch_tool", "R8_other_tool_same_target"
    return "noop", "R9_moved_on"


def _matching(action: str, calls: list[dict]) -> list[dict]:
    if action == "retry":
        return [c for c in calls if c["same_tool"] and c["exact"]]
    if action == "edit_retry":
        return [c for c in calls if c["restart"] or (c["same_tool"] and (c["jac"] >= RETRY_SIM or related(c)))]
    if action == "switch_tool":
        return [c for c in calls if c["delegate"] or (not c["same_tool"] and related(c))]
    return []


def recovered(action: str, rule: str, next_turns: list[dict], session_complete: bool | None) -> bool | None:
    """Outcome verifier for the taken action (pre-registered). None = unverifiable."""
    if session_complete is False:
        return False
    n = next_turns[0]
    if action in ("retry", "edit_retry"):
        # a same-tool call on the same target (exact / similar args / shared target
        # tokens) succeeds within the next 3 turns
        for t in next_turns[:3]:
            for c in t["calls"]:
                if c["same_tool"] and not c["restart"] and (c["exact"] or c["jac"] >= RETRY_SIM or related(c)) and c["ok"]:
                    return True
        return False
    if action == "switch_tool":
        m = [c for c in _matching("switch_tool", n["calls"]) if c["ok"] is not None]
        return bool(m) and any(c["ok"] for c in m)
    if action == "ask":
        if rule == "R2_clarify":
            return any(c["ok"] for c in n["calls"] if c["clarify"])
        # interactive stop: verified iff the human replied
        return True if (len(next_turns) > 1 and next_turns[1]["users_before"] > 0) else None
    if action == "abort":
        return bool(session_complete)
    if action == "noop":
        # moved on and the session still completed; a failed re-attempt of the same
        # target in the next 3 turns means the failure mattered -> not a good noop
        for t in next_turns[:3]:
            for c in t["calls"]:
                if c["same_tool"] and related(c) and c["ok"] is False:
                    return False
        return bool(session_complete)
    return None


def label_v1(ep: dict, *, filtered: bool = True) -> int:
    """Primary: taken action iff verified good (else -1). filtered=False: behavioral label."""
    a = ep.get("v1_action")
    if a is None:
        return -1
    if filtered and ep.get("v1_recovered") is not True:
        return -1
    return IDX[a]


# ---------------------------------------------------------------- cost (pre-registered)


def cost_matrix() -> np.ndarray:
    """C[true, pred]. Wrong abort 5, missed ask 4, needless ask 1.5, redundant retry 0.5."""
    C = np.ones((A1, A1))
    np.fill_diagonal(C, 0.0)
    r, e, s, q, ab = IDX["retry"], IDX["edit_retry"], IDX["switch_tool"], IDX["ask"], IDX["abort"]
    C[e, r] = C[s, r] = 0.5  # redundant retry: one wasted call
    C[r, e] = 0.5  # editing when a plain retry would do
    for t in range(A1):
        if t != q:
            C[t, q] = 1.5  # needless ask: human interrupted
    for p in range(A1):
        if p != q:
            C[q, p] = 4.0  # missed ask
    for t in range(A1):
        if t != ab:
            C[t, ab] = 5.0  # wrong abort (dominates missed ask for true=ask, pred=abort)
    return C


COST = cost_matrix()


def mean_cost(y: np.ndarray, p: np.ndarray) -> float:
    return float(COST[y, p].mean()) if len(y) else float("nan")


def min_cost_decode(P: np.ndarray) -> np.ndarray:
    """Bayes decision under COST given class probabilities P[n, A1]."""
    return (P @ COST).argmin(1)


# ---------------------------------------------------------------- a-priori rule baseline


def rule_v1(ep: dict) -> int:
    """Hand-written before scoring; uses failure-time fields only. Never aborts."""
    e = ep["err"]
    inter = ep["source"] == "interactive"
    if e["benign"]:
        return IDX["noop"]
    if e["policy"]:
        return IDX["ask"] if inter else IDX["switch_tool"]
    if e["sandbox_dead"]:
        return IDX["edit_retry"]
    if ep["consec_fail_same_tool"] >= 3:
        return IDX["ask"] if inter else IDX["switch_tool"]
    if e["transient"] and ep["consec_fail_same_tool"] < 2:
        return IDX["retry"]
    if e["unfamiliar"]:
        return IDX["switch_tool"]
    if e["hard"]:
        return IDX["edit_retry"]
    return IDX["retry"]


# ---------------------------------------------------------------- features (failure-time only)


def v1_frame(ep: dict) -> np.ndarray:
    """10 gym base fields + prev-action one-hot (v1 space) + source/tool/exit one-hots + counts."""
    base = gym_frame(ep)[:10]
    prev = np.zeros(A1)
    prev[IDX.get(ep.get("v1_prev_action") or "noop", 0)] = 1.0
    src = np.array([ep["source"] == s for s in SOURCES], dtype=np.float64)
    fam = np.array([ep["tool_family"] == f for f in TOOL_FAMILIES], dtype=np.float64)
    exb = np.array([ep["exit_bucket"] == b for b in EXIT_BUCKETS], dtype=np.float64)
    cnt = np.array(
        [
            min(ep["session_fail_count"], 20) / 20.0,
            min(ep["n_parallel"], 10) / 10.0,
            min(ep["n_failed_in_turn"], 10) / 10.0,
            float(ep["n_failed_in_turn"] == ep["n_parallel"]),
        ]
    )
    return np.concatenate([base, prev, src, fam, exb, cnt])


def v1_history(episodes: list[dict], history: int = 8):
    """Per episode, prefix-padded Episode of earlier failure frames in the same lineage."""
    from collections import defaultdict

    from .task import episode_from_prefix

    by_lin: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(episodes):
        by_lin[e["lineage"]].append(i)
    out: list = [None] * len(episodes)
    for idxs in by_lin.values():
        idxs.sort(key=lambda i: (episodes[i]["ts"], episodes[i]["turn_idx"]))
        frames = [v1_frame(episodes[i]) for i in idxs]
        for pos, i in enumerate(idxs):
            out[i] = episode_from_prefix(frames[max(0, pos - history + 1) : pos + 1], history=history, env="real")
    return out


# ---------------------------------------------------------------- MB trainer, n-action generalisation


def plasticity_train_k(
    H: np.ndarray, y: np.ndarray, *, W: np.ndarray, rng: np.random.Generator, epochs: int, lr: float
) -> np.ndarray:
    """models._plasticity_train on precomputed KC codes H, for any number of actions.

    Identical update (local delta + 0.02 LTD, lr decay 0.92); with W.shape[1]==5 it
    reproduces the reference trainer bit-for-bit (tested).
    """
    n_act = W.shape[1]
    n = H.shape[0]
    for _epoch in range(epochs):
        for i in rng.permutation(n):
            h = H[i]
            s = h @ W
            s = s - s.max()
            pred = np.exp(np.clip(s, -20, 20))
            pred = pred / pred.sum()
            target = np.zeros(n_act)
            target[int(y[i])] = 1.0
            W += lr * np.outer(h, target - pred)
            W -= 0.02 * lr * np.outer(h, pred)
        lr *= 0.92
    return W
