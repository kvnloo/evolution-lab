"""Distill TypeSafe Jev skill-routing into a mushroom-body student.

Teacher = Jev winners time-joined to OMP history.db prompts (operator text).
Not RLHF: Jev already emits typed choices + probabilities. RLHF would discard
that teacher. Direct-input ridge is the mandatory skeptic (P1).

Data stays under runs/ (gitignored). Do not commit operator prompts.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .models import _kc_codes, _sparse_pn_kc, ridge_fit, softmax_predict

JEV_LOGS = (
    Path.home() / ".omp/agent/extensions/typesafe-jev.log",
    Path.home() / ".omp/agent/extensions/typesafe-jev/route.log",
)
HISTORY_DB = Path.home() / ".omp/agent/history.db"
JOIN_MAX_S = 4.0
NGRAM = 3
PN_DIM = 96
N_KC = 96
MIN_CLASS = 3
SECRET_RE = re.compile(r"(api[_-]?key|sk-|password|secret|token=|bearer\s)", re.I)


def _hash_ngrams(text: str, dim: int = PN_DIM, n: int = NGRAM) -> np.ndarray:
    x = np.zeros(dim, dtype=np.float64)
    t = (text or "").lower()
    if len(t) < n:
        t = t.ljust(n)
    for i in range(len(t) - n + 1):
        digest = hashlib.blake2b(t[i : i + n].encode(), digest_size=8).digest()
        idx = int.from_bytes(digest, "little") % dim
        x[idx] += 1.0
    nrm = np.linalg.norm(x)
    if nrm > 0:
        x /= nrm
    return x


def parse_jev_logs(paths: tuple[Path, ...] = JEV_LOGS) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[float, str]] = set()
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"(\S+)\tsuggest\t(\S+)", line)
            if not m:
                continue
            ts = datetime.fromisoformat(m.group(1).replace("Z", "+00:00")).timestamp()
            rest = line.split("\tsuggest\t", 1)[1]
            winner = rest.split()[0]
            pm = re.search(r"\bp=([0-9.]+)", rest)
            key = (round(ts, 3), winner)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "ts": ts,
                    "label": winner,
                    "p": float(pm.group(1)) if pm else None,
                    "raw": rest,
                }
            )
    rows.sort(key=lambda r: r["ts"])
    return rows


def load_history(db: Path = HISTORY_DB) -> list[dict[str, Any]]:
    if not db.is_file():
        return []
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    out = []
    for prompt, created_at, cwd in con.execute(
        "select prompt, created_at, cwd from history order by created_at"
    ):
        text = prompt or ""
        if SECRET_RE.search(text):
            continue
        out.append({"prompt": text, "ts": float(created_at), "cwd": cwd})
    return out


def join_pairs(
    jev: list[dict[str, Any]],
    history: list[dict[str, Any]],
    *,
    max_dt: float = JOIN_MAX_S,
) -> list[dict[str, Any]]:
    used: set[int] = set()
    pairs: list[dict[str, Any]] = []
    for event in jev:
        best_i = None
        best_dt = max_dt + 1
        for i, h in enumerate(history):
            if i in used:
                continue
            dt = abs(h["ts"] - event["ts"])
            if dt < best_dt:
                best_dt = dt
                best_i = i
        if best_i is None or best_dt > max_dt:
            continue
        used.add(best_i)
        h = history[best_i]
        pairs.append(
            {
                "prompt": h["prompt"],
                "label": event["label"],
                "p": event["p"],
                "dt": best_dt,
                "ts": event["ts"],
            }
        )
    return pairs


def collapse_labels(pairs: list[dict[str, Any]], min_count: int = MIN_CLASS) -> list[dict[str, Any]]:
    counts = Counter(p["label"] for p in pairs)
    out = []
    for p in pairs:
        q = dict(p)
        if counts[p["label"]] < min_count:
            q["label_raw"] = p["label"]
            q["label"] = "other"
        out.append(q)
    return out


def classes_from(pairs: list[dict[str, Any]]) -> list[str]:
    return sorted({p["label"] for p in pairs})


def featurize(pairs: list[dict[str, Any]], dim: int = PN_DIM) -> tuple[np.ndarray, np.ndarray, list[str]]:
    labels = classes_from(pairs)
    index = {n: i for i, n in enumerate(labels)}
    X = np.stack([_hash_ngrams(p["prompt"], dim=dim) for p in pairs])
    y = np.asarray([index[p["label"]] for p in pairs], dtype=np.int64)
    return X, y, labels


def time_split(n: int, confirm_frac: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    n_c = max(8, int(round(n * confirm_frac)))
    n_tr = n - n_c
    if n_tr < 8:
        n_tr = n // 2
        n_c = n - n_tr
    return np.arange(n_tr), np.arange(n_tr, n)


def fit_ridge(X: np.ndarray, y: np.ndarray, n_out: int, l2: float = 1e-2) -> np.ndarray:
    return ridge_fit(X, y, n_out, l2)


def fit_fly(
    X: np.ndarray,
    y: np.ndarray,
    n_out: int,
    *,
    n_kc: int = N_KC,
    seed: int = 1,
    epochs: int = 20,
    lr: float = 0.35,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    W_pn = _sparse_pn_kc(X.shape[1], n_kc, rng)
    k_winners = max(5, int(round(0.10 * n_kc)))
    W = np.zeros((n_kc, n_out), dtype=np.float64)
    H = _kc_codes(X, W_pn, k_winners)
    n = X.shape[0]
    eta = lr
    for _ in range(epochs):
        for i in rng.permutation(n):
            h = H[i]
            scores = h @ W
            s = scores - scores.max()
            pred = np.exp(np.clip(s, -20, 20))
            pred = pred / pred.sum()
            target = np.zeros(n_out, dtype=np.float64)
            target[int(y[i])] = 1.0
            W += eta * np.outer(h, target - pred)
            W -= 0.02 * eta * np.outer(h, pred)
        eta *= 0.92
    return {"W_pn_kc": W_pn, "W_kc_mbon": W, "k_winners": k_winners, "n_params": int(W.size)}


def predict_ridge(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    return softmax_predict(X, W).argmax(axis=1)


def predict_fly(X: np.ndarray, pack: dict[str, Any]) -> np.ndarray:
    H = _kc_codes(X, pack["W_pn_kc"], pack["k_winners"])
    return (H @ pack["W_kc_mbon"]).argmax(axis=1)


def accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    return float((pred == y).mean()) if len(y) else 0.0


def majority(y_train: np.ndarray, n: int) -> np.ndarray:
    if len(y_train) == 0:
        return np.zeros(n, dtype=np.int64)
    mode = int(Counter(y_train.tolist()).most_common(1)[0][0])
    return np.full(n, mode, dtype=np.int64)


def run_jev_distill(*, confirm_frac: float = 0.25, seed: int = 1) -> dict[str, Any]:
    t0 = __import__("time").perf_counter()
    jev = parse_jev_logs()
    hist = load_history()
    pairs = collapse_labels(join_pairs(jev, hist))
    if len(pairs) < 16:
        raise SystemExit(f"need >=16 joined pairs, got {len(pairs)}")
    X, y, labels = featurize(pairs)
    tr, cf = time_split(len(pairs), confirm_frac)
    Xtr, ytr, Xcf, ycf = X[tr], y[tr], X[cf], y[cf]
    maj = majority(ytr, len(ycf))
    W = fit_ridge(Xtr, ytr, len(labels))
    fly = fit_fly(Xtr, ytr, len(labels), seed=seed)
    pred_r = predict_ridge(Xcf, W)
    pred_f = predict_fly(Xcf, fly)
    t_fly = __import__("time").perf_counter()
    for _ in range(200):
        predict_fly(Xcf[:1], fly)
    fly_ms = (__import__("time").perf_counter() - t_fly) / 200 * 1000
    # Binary cascade gate: none_of_these vs call-Jev (any skill).
    none_i = labels.index("none_of_these") if "none_of_these" in labels else None
    if none_i is not None:
        ytr_b = (ytr != none_i).astype(np.int64)
        ycf_b = (ycf != none_i).astype(np.int64)
        Wb = fit_ridge(Xtr, ytr_b, 2)
        flyb = fit_fly(Xtr, ytr_b, 2, seed=seed)
        bin_maj = accuracy(majority(ytr_b, len(ycf_b)), ycf_b)
        bin_ridge = accuracy(predict_ridge(Xcf, Wb), ycf_b)
        bin_fly = accuracy(predict_fly(Xcf, flyb), ycf_b)
    else:
        bin_maj = bin_ridge = bin_fly = None

    report = {
        "task": "jev_skill_route_distill",
        "n_pairs": len(pairs),
        "n_jev_events": len(jev),
        "n_history": len(hist),
        "join_max_s": JOIN_MAX_S,
        "labels": labels,
        "label_counts": dict(Counter(p["label"] for p in pairs)),
        "n_train": int(len(tr)),
        "n_confirm": int(len(cf)),
        "majority_confirm": accuracy(maj, ycf),
        "ridge_confirm": accuracy(pred_r, ycf),
        "fly_confirm": accuracy(pred_f, ycf),
        "fly_n_params": fly["n_params"],
        "fly_predict_ms": fly_ms,
        "binary_none_vs_skill_majority": bin_maj,
        "binary_none_vs_skill_ridge": bin_ridge,
        "binary_none_vs_skill_fly": bin_fly,
        "jev_mean_s": float(np.mean([0.25])),  # placeholder overwritten below
        "note": "Teacher is Jev. Fly can match Jev; beating Jev needs an outcome verifier (did the skill help), which these logs do not contain. Next: live cascade + DAgger on disagreements.",
        "wall_s": __import__("time").perf_counter() - t0,
    }
    # Jev latency from raw if present
    secs = []
    for p in pairs:
        m = re.search(r"([0-9.]+)s model=", str(p.get("p") and ""))
    for event in jev:
        m = re.search(r"([0-9.]+)s model=", event.get("raw") or "")
        if m:
            secs.append(float(m.group(1)))
    if secs:
        report["jev_mean_s"] = float(np.mean(secs))
        report["jev_p50_s"] = float(np.median(secs))
    root = Path(__file__).resolve().parents[1]
    dest = root / "runs" / "jev-distill"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    # do not write prompts
    return report
