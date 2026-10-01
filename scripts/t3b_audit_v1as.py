"""Audit the v1 labelling rule on a pre-registered stratified sample.

  skeleton        print the blind structural sample (no text, no rule output) for a
                  structural audit; the key (audit id -> episode) stays private.
  score-skeleton  compare auditor labels (private TSV: audit_id<TAB>action<TAB>recovered)
                  with the rule; prints/writes aggregates only.
  content         owner-only: write a private worksheet with the raw failed call / result
                  and next turns under ~/.z0int (0600). Never printed.
  score           as score-skeleton, for the owner's content labels.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evolution_lab.action_space_v1 import ACTIONS_V1  # noqa: E402

PRIV = Path.home() / ".z0int" / "research" / "t3b"
SEED = 20260930
PER_ACTION = 12


def sample(eps):
    rng = random.Random(SEED)
    by = {}
    for e in eps:
        if e["v1_action"]:
            by.setdefault(e["v1_action"], []).append(e)
    out = []
    for a in ACTIONS_V1:
        pool = sorted(by.get(a, []), key=lambda e: e["episode_id"])
        out += rng.sample(pool, min(PER_ACTION, len(pool)))
    rng.shuffle(out)
    return out


def fmt_call(c):
    flags = [k for k in ("same_tool", "exact", "restart", "clarify", "delegate") if c[k]]
    return f"{c['fam']}[{','.join(flags) or '-'}] jac={c['jac']:.2f} ovl={c['ovl']:.2f} path={c['path_shared']} ok={c['ok']}"


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def score(labels_path: Path, eps_by_id, key, tag: str):
    rows = [l.split("\t") for l in labels_path.read_text().splitlines() if l.strip() and not l.startswith("#")]
    ra, aa, rr, ar = [], [], [], []
    conf = Counter()
    for aid, act, rec in (r[:3] for r in rows):
        e = eps_by_id[key[aid]]
        ra.append(e["v1_action"])
        aa.append(act.strip())
        conf[(e["v1_action"], act.strip())] += 1
        if rec.strip() in ("True", "False", "None"):
            rr.append(str(e["v1_recovered"]))
            ar.append(rec.strip())
    out = {
        "audit": tag, "n": len(ra),
        "action_agreement": round(sum(x == y for x, y in zip(ra, aa)) / len(ra), 4),
        "action_kappa": round(kappa(ra, aa), 4),
        "recovered_agreement": round(sum(x == y for x, y in zip(rr, ar)) / max(1, len(rr)), 4),
        "recovered_kappa": round(kappa(rr, ar), 4) if rr else None,
        "confusion_rule_vs_auditor": {f"{k[0]}->{k[1]}": v for k, v in sorted(conf.items()) if k[0] != k[1]},
        "per_rule_action_agreement": {
            a: round(sum(1 for x, y in zip(ra, aa) if x == a and y == a) / max(1, sum(1 for x in ra if x == a)), 3)
            for a in ACTIONS_V1
        },
        "n_per_rule_action": dict(Counter(ra)),
    }
    return out


def write_content_worksheet(key: dict[str, str]) -> None:
    """Owner-only. Raw text goes to a 0600 file under ~/.z0int and is never printed."""
    import hashlib
    import sqlite3

    want = {eid: aid for aid, eid in key.items()}
    h = lambda s: hashlib.sha256(s.encode()).hexdigest()[:16]  # noqa: E731
    con = sqlite3.connect(f"file:{Path.home() / '.hermes' / 'state.db'}?mode=ro", uri=True)
    q = ("SELECT session_id, role, tool_call_id, tool_name, tool_calls, substr(coalesce(content,''),1,1200) "
         "FROM messages WHERE (active=1 OR compacted=1) ORDER BY session_id, id")
    sessions: dict[str, list] = {}
    for row in con.execute(q):
        sessions.setdefault(row[0], []).append(row)
    con.close()
    blocks: dict[str, str] = {}
    for sid, rows in sessions.items():
        turns, res = [], {}
        users = 0
        for _s, role, tcid, _tn, tcalls, content in rows:
            if role == "user":
                users += 1
            elif role == "assistant":
                try:
                    calls = [(c.get("id") or c.get("call_id") or "", (c.get("function") or {}).get("name"),
                              str((c.get("function") or {}).get("arguments"))[:600]) for c in json.loads(tcalls or "[]")]
                except (json.JSONDecodeError, ValueError, AttributeError):
                    calls = []
                turns.append({"calls": calls, "text": content or "", "users": users})
                users = 0
            elif role == "tool":
                res[tcid or ""] = content or ""
        for ti, t in enumerate(turns):
            for cid, name, a in t["calls"]:
                eid = h(f"{sid}|{ti}|{cid}")
                if eid not in want:
                    continue
                lines = [f"## {want[eid]}", f"FAILED {name} args={a}", f"RESULT {res.get(cid, '')[:800]}"]
                for j, nt in enumerate(turns[ti + 1 : ti + 4]):
                    lines.append(f"-- N+{j} users_before={nt['users']} text={nt['text'][:400]!r}")
                    for ncid, nn, na in nt["calls"][:6]:
                        lines.append(f"   CALL {nn} args={na[:300]} -> {res.get(ncid, '')[:300]!r}")
                blocks[want[eid]] = "\n".join(lines)
    out = PRIV / "audit_v1as_content_worksheet.md"
    hdr = ("# v1 content audit (PRIVATE). Label each id in audit_v1as_content_labels.tsv as\n"
           "# audit_id<TAB>action<TAB>recovered(True/False/None); actions: " + ", ".join(ACTIONS_V1) + "\n\n")
    out.write_text(hdr + "\n\n".join(blocks[k] for k in sorted(blocks)))
    out.chmod(0o600)
    print(f"wrote {len(blocks)} blocks to {out} (private; not printed)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["skeleton", "score-skeleton", "content", "score"])
    ap.add_argument("--labels", default=None)
    ap.add_argument("--out", default=str(ROOT / "results" / "t3b-v1as-audit.json"))
    args = ap.parse_args()
    eps = [json.loads(l) for l in (PRIV / "episodes_v1as.jsonl").read_text().splitlines() if l.strip()]
    s = sample(eps)
    key = {f"a{i:03d}": e["episode_id"] for i, e in enumerate(s)}
    (PRIV / "audit_v1as_key.json").write_text(json.dumps(key))
    (PRIV / "audit_v1as_key.json").chmod(0o600)
    by_id = {e["episode_id"]: e for e in eps}
    if args.mode == "skeleton":
        for aid, eid in key.items():
            e = by_id[eid]
            print(f"## {aid} source={e['source']} failed={e['tool_family']}/{e['err']['class']} "
                  f"session_complete={e['session_complete']}")
            for j, t in enumerate(e["v1_skeleton"]):
                head = f"  N+{j}: users_before={t['users_before']}"
                if not t["calls"]:
                    print(f"{head} no-calls fail_report={t['fail_report']}")
                    continue
                print(f"{head} calls={len(t['calls'])}")
                for c in t["calls"][:6]:
                    print(f"     {fmt_call(c)}")
                if len(t["calls"]) > 6:
                    print(f"     (+{len(t['calls']) - 6} more)")
        return
    if args.mode == "content":
        write_content_worksheet(key)
        return
    lab = Path(args.labels or PRIV / ("audit_v1as_skeleton_labels.tsv" if args.mode == "score-skeleton"
                                      else "audit_v1as_content_labels.tsv"))
    res = score(lab, by_id, key, "structural-blind" if args.mode == "score-skeleton" else "content-owner")
    print(json.dumps(res, indent=1))
    outp = Path(args.out)
    prev = json.loads(outp.read_text()) if outp.exists() else {}
    prev[res["audit"]] = res
    outp.write_text(json.dumps(prev, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
