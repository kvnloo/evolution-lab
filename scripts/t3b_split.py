"""T3b: frozen split grouped by session lineage (no lineage crosses splits).

Private: ~/.z0int/research/t3b/split_v1.json  (lineage-hash -> split)
Committed: results/t3b-split-stats.json      (counts + sha256 of the private file)
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from evolution_lab.real_recovery import ACTIONS, MAPPINGS, label_of

PRIV = Path.home() / ".z0int" / "research" / "t3b"
REPO = Path(__file__).resolve().parents[1]
SALT = "t3b-split-v1-20260930"
FRACS = (("train", 0.6), ("val", 0.2), ("test", 0.2))


def bucket(lineage: str) -> str:
    u = int(hashlib.sha256(f"{SALT}|{lineage}".encode()).hexdigest()[:12], 16) / float(16**12)
    acc = 0.0
    for name, f in FRACS:
        acc += f
        if u < acc:
            return name
    return FRACS[-1][0]


def main() -> None:
    eps = [json.loads(l) for l in (PRIV / "episodes.jsonl").read_text().splitlines() if l.strip()]
    out = PRIV / "split_v1.json"
    if out.exists():
        split = json.loads(out.read_text())["lineage_split"]
        missing = {e["lineage"] for e in eps} - set(split)
        if missing:
            raise SystemExit(f"frozen split exists but {len(missing)} lineages are new; mint split_v2 instead")
    else:
        split = {lin: bucket(lin) for lin in sorted({e["lineage"] for e in eps})}
        out.write_text(json.dumps({"salt": SALT, "fracs": FRACS, "lineage_split": split}, sort_keys=True))
        out.chmod(0o600)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    stats: dict = {"split_file_sha256": sha, "salt": SALT, "fracs": dict(FRACS), "grouping": "lineage root (parent_session_id chain)"}
    lin_by = Counter(split.values())
    stats["lineages"] = dict(lin_by)
    per = defaultdict(Counter)
    fam_by = defaultdict(set)
    for e in eps:
        s = split[e["lineage"]]
        per[s]["episodes"] += 1
        per[s][f"source:{e['source']}"] += 1
        fam_by[s].add(e["extra"]["family"])
        for m in MAPPINGS:
            y = label_of(e, m)
            per[s][f"{m}:{'unknown' if y < 0 else ACTIONS[y]}"] += 1
    stats["per_split"] = {k: dict(sorted(v.items())) for k, v in per.items()}
    stats["families_per_split"] = {k: len(v) for k, v in fam_by.items()}
    stats["families_shared_train_test"] = len(fam_by["train"] & fam_by["test"])
    (REPO / "results" / "t3b-split-stats.json").write_text(json.dumps(stats, indent=1, sort_keys=True) + "\n")
    print(json.dumps(stats, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
