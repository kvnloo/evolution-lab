"""T3b: mine real recovery episodes from the local Hermes state.db (read-only).

Private output (never committed):  ~/.z0int/research/t3b/episodes.jsonl
Committed output (aggregates only): results/t3b-mine-stats.json

Raw text is read in-process for regex classification only; nothing but
numeric/categorical features and hashed ids is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path

from evolution_lab.real_recovery import (
    _POLICY,
    _SANDBOX,
    _TRANSIENT,
    LOOSE_MAP,
    RAW_ACTIONS,
    STRICT_MAP,
    RealEpisode,
    args_hash,
    classify_result,
    error_class,
    exit_bucket,
    is_restart_call,
    parse_args,
    source_bucket,
    tool_family,
)

DB = Path.home() / ".hermes" / "state.db"
PRIV = Path.home() / ".z0int" / "research" / "t3b"
REPO = Path(__file__).resolve().parents[1]

Q_SESSIONS = """
SELECT id, source, parent_session_id, end_reason, started_at, session_key
FROM sessions
"""

Q_MESSAGES = """
SELECT session_id, id, role, tool_call_id, tool_name,
       CASE WHEN role='assistant' THEN tool_calls END,
       finish_reason, timestamp,
       CASE WHEN role='assistant' THEN length(coalesce(content,'')) END,
       CASE WHEN role='tool' THEN substr(coalesce(content,''),1,1500) END,
       CASE WHEN role='tool' AND json_valid(content) AND json_type(content)='object' THEN json_extract(content,'$.error') END,
       CASE WHEN role='tool' AND json_valid(content) AND json_type(content)='object' THEN json_extract(content,'$.exit_code') END,
       CASE WHEN role='tool' AND json_valid(content) AND json_type(content)='object' THEN json_extract(content,'$.status') END,
       CASE WHEN role='tool' AND json_valid(content) AND json_type(content)='object' THEN json_extract(content,'$.success') END,
       CASE WHEN role='tool' AND json_valid(content) AND json_type(content)='object' THEN json_extract(content,'$.exit_code_meaning') END,
       CASE WHEN role='tool' AND json_valid(content) AND json_type(content)='object' THEN json_extract(content,'$.approval_pending') END,
       CASE WHEN role='tool' AND json_valid(content) AND json_type(content)='object' THEN json_extract(content,'$.smart_denied') END,
       CASE WHEN role='tool' AND json_valid(content) AND json_type(content)='object' THEN substr(coalesce(json_extract(content,'$.output'),''),-1500) END
FROM messages
WHERE (active=1 OR compacted=1)
ORDER BY session_id, id
"""


def h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def lineage_root(sid: str, parent: dict[str, str | None]) -> str:
    seen = set()
    cur = sid
    while parent.get(cur) and cur not in seen:
        seen.add(cur)
        cur = parent[cur]  # type: ignore[assignment]
    return cur


def family_key(root: str, source: str, session_key: str | None) -> str:
    """Coarser group: cron job id / a2a session key; used for a sensitivity split."""
    if root.startswith("cron_"):
        parts = root.split("_")
        if len(parts) >= 2:
            return "cronjob:" + parts[1]
    if session_key:
        return "key:" + session_key
    return "root:" + root


def build_turns(rows):
    """Group a session's rows into assistant turns with their tool results."""
    turns = []  # each: {"calls":[...], "results":{call_id: ToolResult}, "text_len", "users_before"}
    pending_users = 0
    by_call: dict[str, tuple[int, int]] = {}
    for r in rows:
        (_sid, mid, role, tcid, tname, tcalls, finish, ts, alen, head, jerr, jexit, jstat, jsucc, jmean, jappr, jdeny, otail) = r
        if role == "user":
            pending_users += 1
            continue
        if role == "assistant":
            calls = []
            if tcalls:
                try:
                    arr = json.loads(tcalls)
                except (json.JSONDecodeError, ValueError):
                    arr = []
                for c in arr if isinstance(arr, list) else []:
                    fn = (c or {}).get("function") or {}
                    cid = c.get("id") or c.get("call_id") or ""
                    calls.append(
                        {
                            "id": cid,
                            "name": fn.get("name"),
                            "ah": args_hash(fn.get("arguments")),
                            "tok": arg_tokens(fn.get("arguments")),
                            "restart": is_restart_call(fn.get("name"), fn.get("arguments")),
                            "res": None,
                        }
                    )
            turns.append({"calls": calls, "text_len": alen or 0, "users_before": pending_users, "ts": ts, "finish": finish})
            pending_users = 0
            for k, c in enumerate(calls):
                if c["id"]:
                    by_call[c["id"]] = (len(turns) - 1, k)
            continue
        if role == "tool":
            res = classify_result(tname, head, jerr, jexit, jstat, jsucc, jmean, jappr, jdeny, otail)
            loc = by_call.get(tcid or "")
            if loc is None and turns:
                # unmatched id: attach to first call of last turn with same name and no result
                ti = len(turns) - 1
                for k, c in enumerate(turns[ti]["calls"]):
                    if c["res"] is None and c["name"] == tname:
                        loc = (ti, k)
                        break
            if loc is not None:
                ti, k = loc
                turns[ti]["calls"][k]["res"] = res
    trailing_users = pending_users
    return turns, trailing_users


RETRY_SIM = 0.5


def arg_tokens(arguments) -> frozenset:
    """Local-only token set of call arguments (never stored) for retry similarity."""
    import re as _re

    a = parse_args(arguments)
    blob = json.dumps(a, sort_keys=True, default=str) if a else str(arguments or "")
    return frozenset(_re.findall(r"[A-Za-z0-9_./-]+", blob[:4000]))


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def ok(c) -> bool:
    return c["res"] is not None and not c["res"].failed


def classify_next(turns, ti, failed_call, interactive: bool):
    """Raw action taken at the next decision turn + local recovery verdict."""
    if ti + 1 >= len(turns):
        return "none", None, {}
    nxt = turns[ti + 1]
    name, ah = failed_call["name"], failed_call["ah"]
    calls = nxt["calls"]
    extra = {}
    if nxt["users_before"] > 0:
        # human spoke before the agent acted; recovered if tool later succeeds
        later = any(ok(c) and c["name"] == name for t in turns[ti + 1 : ti + 4] for c in t["calls"])
        return "human_interrupt", later, extra
    if not calls:
        if interactive:
            # page_human verified if the human replied (next turn preceded by a user message)
            replied = ti + 2 < len(turns) and turns[ti + 2]["users_before"] > 0
            return "stop_interactive", (True if replied else None), extra
        extra["weak_verifier"] = True
        return "stop_headless", "SESSION", extra
    restart = [c for c in calls if c["restart"]]
    if restart:
        later = any(ok(c) and c["name"] == name for t in turns[ti + 1 : ti + 4] for c in t["calls"] if not c["restart"])
        return "restart", later, extra
    deleg = [c for c in calls if (c["name"] or "") in {"delegate_task", "a2a_call"}]
    if deleg:
        return "delegate", all(ok(c) for c in deleg), extra
    if any((c["name"] or "") == "clarify" for c in calls):
        return "clarify", any(ok(c) for c in calls if c["name"] == "clarify"), extra
    same = [c for c in calls if c["name"] == name]
    exact = [c for c in same if c["ah"] == ah]
    if exact:
        return "retry_exact", any(ok(c) for c in exact), extra
    near = [c for c in same if jaccard(c["tok"], failed_call["tok"]) >= RETRY_SIM]
    if near:
        return "retry_modified", any(ok(c) for c in near), extra
    if same:
        extra["same_tool_new_args"] = True
    extra["weak_verifier"] = True
    return "switch_tool", all(ok(c) for c in calls if c["res"] is not None) and any(c["res"] is not None for c in calls), extra


def mine(db: Path, limit_sessions: int | None = None):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.execute("PRAGMA query_only=1")
    sess = {}
    parent = {}
    for sid, source, par, end_reason, started, skey in con.execute(Q_SESSIONS):
        sess[sid] = (source, end_reason, started, skey)
        parent[sid] = par
    episodes: list[RealEpisode] = []
    stats = Counter()
    cur_sid = None
    buf: list = []

    def flush(sid, rows):
        if sid is None or sid not in sess:
            return
        source, end_reason, started, skey = sess[sid]
        sb = source_bucket(source)
        interactive = sb == "interactive"
        turns, trailing_users = build_turns(rows)
        stats["sessions_scanned"] += 1
        last = turns[-1] if turns else None
        complete = bool(last and not last["calls"] and last["text_len"] > 0 and trailing_users == 0)
        root = lineage_root(sid, parent)
        fam = family_key(root, source, skey)
        consec: dict[str, int] = defaultdict(int)
        fail_count = 0
        last_action = "noop"
        any_fail = False
        for ti, t in enumerate(turns):
            n_par = len(t["calls"])
            failed_calls = [c for c in t["calls"] if c["res"] is not None and c["res"].failed]
            stats["tool_results"] += sum(1 for c in t["calls"] if c["res"] is not None)
            stats["tool_calls"] += n_par
            for c in t["calls"]:
                if c["res"] is None:
                    stats["tool_calls_no_result"] += 1
            for c in failed_calls:
                any_fail = True
                res = c["res"]
                ec = error_class(res)
                for nm, rx in (("policy", _POLICY), ("sandbox", _SANDBOX), ("transient", _TRANSIENT)):
                    m = rx.search(res.text[-3000:]) if res.text else None
                    if m:
                        # regex vocabulary only (audit of what drove the class)
                        kw = m.group(0).lower()
                        kw = kw if kw in rx.pattern.lower() else "<wildcard-branch>"
                        stats[f"kw_{nm}:{kw}"] += 1
                if res.policy:
                    stats["kw_policy:<structured_key>"] += 1
                raw, local, extra = classify_next(turns, ti, c, interactive)
                if local == "SESSION":
                    local = complete
                ep = RealEpisode(
                    episode_id=h(f"{sid}|{ti}|{c['id']}"),
                    lineage=h(root),
                    source=sb,
                    tool_family=tool_family(c["name"]),
                    err={
                        "transient": ec.transient,
                        "sandbox_dead": ec.sandbox_dead,
                        "policy": ec.policy,
                        "hard": ec.hard,
                        "unfamiliar": ec.unfamiliar,
                        "benign": ec.benign,
                        "class": ec.name(),
                    },
                    exit_bucket=exit_bucket(res.exit_code),
                    consec_fail_same_tool=consec[c["name"] or ""],
                    session_fail_count=fail_count,
                    turn_idx=ti,
                    n_parallel=n_par,
                    n_failed_in_turn=len(failed_calls),
                    last_action=last_action,
                    raw_action=raw,
                    local_recovered=local,
                    session_complete=complete,
                    ts=float(t["ts"] or 0.0),
                    extra={**extra, "family": h(fam), "end_reason": end_reason or ""},
                )
                episodes.append(ep)
                fail_count += 1
            for c in t["calls"]:
                if c["res"] is None:
                    continue
                if c["res"].failed:
                    consec[c["name"] or ""] += 1
                else:
                    consec[c["name"] or ""] = 0
            if failed_calls:
                # previous recovery action feature for the next episode (loose map, else noop)
                last_raw = episodes[-1].raw_action
                last_action = LOOSE_MAP.get(last_raw) or "noop"
        stats["sessions_with_failure"] += int(any_fail)
        stats["sessions_complete"] += int(complete)

    for row in con.execute(Q_MESSAGES):
        sid = row[0]
        if sid != cur_sid:
            flush(cur_sid, buf)
            buf = []
            cur_sid = sid
            if limit_sessions and stats["sessions_scanned"] >= limit_sessions:
                break
        buf.append(row)
    else:
        flush(cur_sid, buf)
    con.close()
    return episodes, stats


def aggregate(episodes: list[RealEpisode], stats: Counter) -> dict:
    out: dict = {"stats": dict(stats), "n_episodes": len(episodes)}
    out["n_lineages"] = len({e.lineage for e in episodes})
    out["n_families"] = len({e.extra["family"] for e in episodes})
    out["by_source"] = dict(Counter(e.source for e in episodes))
    out["by_tool_family"] = dict(Counter(e.tool_family for e in episodes))
    out["by_error_class"] = dict(Counter(e.err["class"] for e in episodes))
    out["by_raw_action"] = dict(Counter(e.raw_action for e in episodes))
    tab = defaultdict(Counter)
    for e in episodes:
        tab[e.raw_action][str(e.local_recovered)] += 1
    out["raw_action_x_local_recovered"] = {k: dict(v) for k, v in tab.items()}
    tab2 = defaultdict(Counter)
    for e in episodes:
        tab2[e.raw_action][str(e.session_complete)] += 1
    out["raw_action_x_session_complete"] = {k: dict(v) for k, v in tab2.items()}
    for m in ("strict", "loose"):
        labs = Counter(e.label(m) for e in episodes)
        out[f"labels_{m}"] = {("unknown" if k < 0 else ["retry", "restart_sandbox", "escalate", "noop", "page_human"][k]): v for k, v in sorted(labs.items())}
    tab3 = defaultdict(Counter)
    for e in episodes:
        tab3[e.err["class"]][e.raw_action] += 1
    out["error_class_x_raw_action"] = {k: dict(v) for k, v in tab3.items()}
    out["mapping"] = {"strict": STRICT_MAP, "loose": LOOSE_MAP, "raw_actions": list(RAW_ACTIONS)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-sessions", type=int, default=None)
    ap.add_argument("--stats-out", default=str(REPO / "results" / "t3b-mine-stats.json"))
    args = ap.parse_args()
    t0 = time.time()
    episodes, stats = mine(DB, args.limit_sessions)
    PRIV.mkdir(parents=True, exist_ok=True)
    PRIV.chmod(0o700)
    priv = PRIV / "episodes.jsonl"
    with priv.open("w") as f:
        for e in episodes:
            f.write(json.dumps(e.to_dict()) + "\n")
    priv.chmod(0o600)
    agg = aggregate(episodes, stats)
    agg["mine_seconds"] = round(time.time() - t0, 1)
    agg["private_store"] = "~/.z0int/research/t3b/episodes.jsonl (not committed)"
    Path(args.stats_out).write_text(json.dumps(agg, indent=1, sort_keys=True) + "\n")
    print(json.dumps(agg, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
