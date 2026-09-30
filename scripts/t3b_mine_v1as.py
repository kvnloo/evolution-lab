"""T3b action-space v1: re-derive labels from what the agent did next (read-only).

Re-walks ~/.hermes/state.db with the same episode definition and ids as
scripts/t3b_mine_hermes.py, and adds, per episode, a structural skeleton of the
next 3 assistant turns (no text) plus the pre-registered v1 action / recovery.

Private output (never committed): ~/.z0int/research/t3b/episodes_v1as.jsonl
Committed output (aggregates only): results/t3b-v1as-label-stats.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evolution_lab.action_space_v1 import (  # noqa: E402
    ACTIONS_V1,
    FAIL_REPORT,
    label_v1,
    overlap,
    recovered,
    salient_tokens,
    taken_action,
)
from evolution_lab.real_recovery import (  # noqa: E402
    args_hash,
    classify_result,
    error_class,
    exit_bucket,
    is_restart_call,
    parse_args,
    source_bucket,
    tool_family,
)
from t3b_mine_hermes import DB, PRIV, Q_SESSIONS, arg_tokens, family_key, h, jaccard, lineage_root  # noqa: E402

Q_MESSAGES = """
SELECT session_id, id, role, tool_call_id, tool_name,
       CASE WHEN role='assistant' THEN tool_calls END,
       finish_reason, timestamp,
       CASE WHEN role='assistant' THEN length(coalesce(content,'')) END,
       CASE WHEN role='assistant' THEN substr(coalesce(content,''),1,4000) END,
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


def build_turns(rows):
    """As t3b_mine_hermes.build_turns, plus per-call salient tokens and a fail-report bit.

    Assistant text is reduced to one regex bit in-process and dropped.
    """
    turns = []
    pending_users = 0
    by_call: dict[str, tuple[int, int]] = {}
    for r in rows:
        (_sid, _mid, role, tcid, tname, tcalls, finish, ts, alen, atext, head, jerr, jexit, jstat, jsucc, jmean, jappr, jdeny, otail) = r
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
                    a = fn.get("arguments")
                    calls.append(
                        {
                            "id": cid,
                            "name": fn.get("name"),
                            "ah": args_hash(a),
                            "tok": arg_tokens(a),
                            "sal": salient_tokens(parse_args(a) or a),
                            "restart": is_restart_call(fn.get("name"), a),
                            "res": None,
                        }
                    )
            fail_report = bool(FAIL_REPORT.search(atext or "")) if not calls else None
            turns.append({"calls": calls, "text_len": alen or 0, "users_before": pending_users, "ts": ts,
                          "finish": finish, "fail_report": fail_report})
            pending_users = 0
            for k, c in enumerate(calls):
                if c["id"]:
                    by_call[c["id"]] = (len(turns) - 1, k)
            continue
        if role == "tool":
            res = classify_result(tname, head, jerr, jexit, jstat, jsucc, jmean, jappr, jdeny, otail)
            loc = by_call.get(tcid or "")
            if loc is None and turns:
                ti = len(turns) - 1
                for k, c in enumerate(turns[ti]["calls"]):
                    if c["res"] is None and c["name"] == tname:
                        loc = (ti, k)
                        break
            if loc is not None:
                ti, k = loc
                turns[ti]["calls"][k]["res"] = res
    return turns, pending_users


def skeleton(turns, ti, failed) -> list[dict]:
    """Structural view (no text) of the next 3 assistant turns relative to the failed call."""
    out = []
    for t in turns[ti + 1 : ti + 4]:
        calls = []
        for c in t["calls"]:
            ovl, ps = overlap(failed["sal"], c["sal"])
            nm = c["name"] or ""
            calls.append(
                {
                    "fam": tool_family(nm),
                    "same_tool": nm == (failed["name"] or ""),
                    "exact": nm == (failed["name"] or "") and c["ah"] == failed["ah"],
                    "jac": round(jaccard(c["tok"], failed["tok"]), 3),
                    "ovl": round(ovl, 3),
                    "path_shared": ps,
                    "restart": bool(c["restart"]),
                    "clarify": nm == "clarify",
                    "delegate": nm in {"delegate_task", "a2a_call"},
                    "ok": None if c["res"] is None else (not c["res"].failed),
                }
            )
        out.append({"users_before": t["users_before"], "fail_report": t["fail_report"], "calls": calls})
    return out


def mine(db: Path):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.execute("PRAGMA query_only=1")
    sess, parent = {}, {}
    for sid, source, par, end_reason, started, skey in con.execute(Q_SESSIONS):
        sess[sid] = (source, end_reason, started, skey)
        parent[sid] = par
    episodes: list[dict] = []

    def flush(sid, rows):
        if sid is None or sid not in sess:
            return
        source, end_reason, _started, skey = sess[sid]
        sb = source_bucket(source)
        interactive = sb == "interactive"
        turns, trailing_users = build_turns(rows)
        last = turns[-1] if turns else None
        complete = bool(last and not last["calls"] and last["text_len"] > 0 and trailing_users == 0)
        root = lineage_root(sid, parent)
        fam = family_key(root, source, skey)
        consec: dict[str, int] = defaultdict(int)
        fail_count = 0
        prev_by_turn: list[tuple[int, str]] = []  # (turn index, v1 behavioral action)
        for ti, t in enumerate(turns):
            failed_calls = [c for c in t["calls"] if c["res"] is not None and c["res"].failed]
            prev_action = next((a for (pti, a) in reversed(prev_by_turn) if pti < ti), "noop")
            for c in failed_calls:
                res = c["res"]
                ec = error_class(res)
                sk = skeleton(turns, ti, c)
                act, rule = taken_action(sk, interactive=interactive)
                rec = recovered(act, rule, sk, complete) if act else None
                episodes.append(
                    {
                        "episode_id": h(f"{sid}|{ti}|{c['id']}"),
                        "lineage": h(root),
                        "source": sb,
                        "tool_family": tool_family(c["name"]),
                        "err": {"transient": ec.transient, "sandbox_dead": ec.sandbox_dead, "policy": ec.policy,
                                "hard": ec.hard, "unfamiliar": ec.unfamiliar, "benign": ec.benign, "class": ec.name()},
                        "exit_bucket": exit_bucket(res.exit_code),
                        "consec_fail_same_tool": consec[c["name"] or ""],
                        "session_fail_count": fail_count,
                        "turn_idx": ti,
                        "n_parallel": len(t["calls"]),
                        "n_failed_in_turn": len(failed_calls),
                        "last_action": "noop",
                        "session_complete": complete,
                        "ts": float(t["ts"] or 0.0),
                        "v1_prev_action": prev_action,
                        "v1_action": act,
                        "v1_rule": rule,
                        "v1_recovered": rec,
                        "v1_skeleton": sk,
                        "extra": {"family": h(fam), "end_reason": end_reason or ""},
                    }
                )
                fail_count += 1
            for c in t["calls"]:
                if c["res"] is None:
                    continue
                consec[c["name"] or ""] = consec[c["name"] or ""] + 1 if c["res"].failed else 0
            if failed_calls:
                acts = [e["v1_action"] for e in episodes[-len(failed_calls):] if e["v1_action"]]
                if acts:
                    prev_by_turn.append((ti + 1, Counter(acts).most_common(1)[0][0]))

    cur, buf = None, []
    for row in con.execute(Q_MESSAGES):
        if row[0] != cur:
            flush(cur, buf)
            cur, buf = row[0], []
        buf.append(row)
    flush(cur, buf)
    con.close()
    return episodes


def aggregate(eps: list[dict], split: dict | None) -> dict:
    out: dict = {"n_episodes": len(eps), "actions": list(ACTIONS_V1)}
    out["by_rule"] = dict(Counter(e["v1_rule"] for e in eps))
    out["behavioral"] = dict(Counter(e["v1_action"] or "excluded" for e in eps))
    tab = defaultdict(Counter)
    for e in eps:
        tab[e["v1_action"] or "excluded"][str(e["v1_recovered"])] += 1
    out["action_x_recovered"] = {k: dict(v) for k, v in tab.items()}
    out["primary_labels"] = {(ACTIONS_V1[k] if k >= 0 else "unknown"): v for k, v in
                             sorted(Counter(label_v1(e) for e in eps).items())}
    rs = defaultdict(Counter)
    for e in eps:
        rs[e["source"]][e["v1_action"] or "excluded"] += 1
    out["source_x_action"] = {k: dict(v) for k, v in rs.items()}
    ec = defaultdict(Counter)
    for e in eps:
        ec[e["err"]["class"]][ACTIONS_V1[label_v1(e)] if label_v1(e) >= 0 else "unknown"] += 1
    out["error_class_x_primary_label"] = {k: dict(v) for k, v in ec.items()}
    if split:
        ps = defaultdict(Counter)
        for e in eps:
            k = label_v1(e)
            ps[split[e["lineage"]]][ACTIONS_V1[k] if k >= 0 else "unknown"] += 1
        out["per_split_primary"] = {k: dict(v) for k, v in ps.items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats-out", default=str(ROOT / "results" / "t3b-v1as-label-stats.json"))
    args = ap.parse_args()
    t0 = time.time()
    eps = mine(DB)
    old = {json.loads(l)["episode_id"] for l in (PRIV / "episodes.jsonl").read_text().splitlines() if l.strip()}
    priv = PRIV / "episodes_v1as.jsonl"
    with priv.open("w") as f:
        for e in eps:
            f.write(json.dumps(e) + "\n")
    priv.chmod(0o600)
    split = json.loads((PRIV / "split_v1.json").read_text())["lineage_split"]
    new_lin = {e["lineage"] for e in eps} - set(split)
    agg = aggregate([e for e in eps if e["lineage"] in split], split)
    agg["id_overlap_with_t3b_episodes"] = {"t3b": len(old), "v1as": len(eps),
                                            "shared": len(old & {e["episode_id"] for e in eps})}
    agg["new_lineages_not_in_frozen_split"] = len(new_lin)
    agg["episodes_in_new_lineages_dropped"] = sum(1 for e in eps if e["lineage"] in new_lin)
    agg["mine_seconds"] = round(time.time() - t0, 1)
    agg["private_store"] = "~/.z0int/research/t3b/episodes_v1as.jsonl (not committed)"
    Path(args.stats_out).write_text(json.dumps(agg, indent=1, sort_keys=True) + "\n")
    print(json.dumps(agg, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
