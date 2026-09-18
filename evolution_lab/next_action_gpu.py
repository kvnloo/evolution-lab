"""GPU mushroom-body population on compiled Hermes next-action episodes.

Frozen judge is still CPU last-step / ridge comparison. Recipe (n_train,
gold_only, confirm_frac) is the evolvable object. Champion packs promote
only when confirm acc beats the on-disk pack (never clobber on weaker runs).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from .jax_mb import numpy_perms, population_mbon
from .models import _kc_codes, _sparse_pn_kc
from .schema import DataRecipe

EPISODES = Path.home() / ".z0int" / "episodes" / "next_action.jsonl"
FAMILIES = (
    "READ_SEARCH",
    "EDIT",
    "EXECUTE",
    "WEB",
    "DELEGATE",
    "VERIFY",
    "RESPOND",
    "ABSTAIN",
)
FAM_I = {n: i for i, n in enumerate(FAMILIES)}
_XY_CACHE: dict[tuple, tuple] = {}


def _hash(text: str, dim: int) -> np.ndarray:
    out = np.zeros(dim, dtype=np.float32)
    t = (text or "").lower()
    if len(t) < 3:
        return out
    for i in range(len(t) - 2):
        h = hashlib.blake2b(t[i : i + 3].encode(), digest_size=8).digest()
        out[int.from_bytes(h[:4], "little") % dim] += 1.0
    n = float(np.linalg.norm(out))
    return out / n if n > 0 else out


def load_xy(path: Path, *, gold_only: bool, dim: int = 64) -> tuple[np.ndarray, np.ndarray]:
    key = (str(path), bool(gold_only), int(dim))
    hit = _XY_CACHE.get(key)
    if hit is not None:
        return hit
    xs, ys = [], []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            fam = rec.get("family")
            if fam not in FAM_I:
                continue
            if gold_only and not (rec.get("gold") or rec.get("tier") == "gold"):
                continue
            text = rec.get("text") or rec.get("user") or ""
            prev = rec.get("prev") or []
            if prev:
                text = text + " " + " ".join(str(x) for x in prev)
            xs.append(_hash(text, dim))
            ys.append(FAM_I[fam])
    if not xs:
        raise FileNotFoundError(f"no next-action rows in {path}")
    out = (np.stack(xs), np.asarray(ys, dtype=np.int32))
    _XY_CACHE[key] = out
    return out


def ridge_acc(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray) -> float:
    k = int(max(ytr.max(), yte.max())) + 1
    Y = np.eye(k, dtype=np.float64)[ytr]
    w = np.linalg.pinv(Xtr.T @ Xtr + 1.0 * np.eye(Xtr.shape[1])) @ Xtr.T @ Y
    return float(((Xte @ w).argmax(1) == yte).mean())


def run_next_action_gpu(
    *,
    recipe: DataRecipe | None = None,
    n_pop: int = 256,
    n_kc: int = 96,
    k_winners: int = 20,
    epochs: int = 6,
    lr: float = 0.35,
    seed: int = 0,
    episodes: Path | None = None,
    dim: int = 64,
) -> dict[str, Any]:
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.25")
    recipe = recipe or DataRecipe(source="next_action", n_train=2048)
    path = episodes or EPISODES
    X, y = load_xy(path, gold_only=recipe.gold_only, dim=dim)
    n = len(y)
    cut = max(8, int(n * (1.0 - recipe.confirm_frac)))
    Xtr, ytr = X[:cut], y[:cut]
    Xte, yte = X[cut:], y[cut:]
    if recipe.n_train > 0 and len(Xtr) > recipe.n_train:
        Xtr, ytr = Xtr[-recipe.n_train :], ytr[-recipe.n_train :]
    rng = np.random.default_rng(seed)
    W_pn = _sparse_pn_kc(Xtr.shape[1], n_kc, rng)
    H = _kc_codes(Xtr, W_pn, k_winners)
    Ht = _kc_codes(Xte, W_pn, k_winners)
    bank = np.stack([numpy_perms(len(ytr), epochs, np.random.default_rng(seed + i)) for i in range(n_pop)])
    t0 = time.perf_counter()
    Ws = population_mbon(H, ytr, bank, lr)
    gpu_s = time.perf_counter() - t0
    acc = np.stack([(Ht @ Ws[p]).argmax(1) == yte for p in range(n_pop)]).mean(axis=1)
    ridge = ridge_acc(Xtr, ytr, Xte, yte)
    best = float(acc.max())
    counts = np.bincount(yte, minlength=int(yte.max()) + 1)
    majority = float(counts.max() / len(yte)) if len(yte) else 0.0
    report = {
        "schema": "flyforge.next_action_gpu.v1",
        "recipe": recipe.__dict__ if hasattr(recipe, "__dict__") else dict(recipe),
        "n_train": int(len(ytr)),
        "n_confirm": int(len(yte)),
        "n_pop": n_pop,
        "n_kc": int(n_kc),
        "k_winners": int(k_winners),
        "pn_dim": int(Xtr.shape[1]),
        "gpu_filter_s": gpu_s,
        "gpu_best_confirm_acc": best,
        "ridge_confirm_acc": ridge,
        "majority_confirm_acc": majority,
        "gpu_beats_ridge": best > ridge + 1e-12,
        "note": "GPU filters candidates; ridge is the CPU baseline. Do not keep from GPU acc alone.",
    }
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "runs" / "gpu-evolve"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "next_action_report.json").write_text(json.dumps(report, indent=2) + "\n")
    best_i = int(np.argmax(acc))
    pack = {
        "W_pn_kc": np.asarray(W_pn),
        "W_kc_mbon": np.asarray(Ws[best_i]),
        "k_winners": np.int32(k_winners),
        "n_kc": np.int32(n_kc),
        "acc": np.float64(best),
        "ridge": np.float64(ridge),
        "majority": np.float64(majority),
    }
    # Always write run artifact under runs/; only promote data/ champion if better.
    np.savez(out_dir / "next_action_candidate.npz", **pack)
    report["promoted"] = _maybe_promote_champion(root, pack, report, n_pop=n_pop)
    return report


def _maybe_promote_champion(
    root: Path,
    pack: dict[str, Any],
    report: dict[str, Any],
    *,
    n_pop: int,
) -> bool:
    """Write data/next_action/champion.* only if confirm acc improves."""
    data_dir = root / "data" / "next_action"
    data_dir.mkdir(parents=True, exist_ok=True)
    champ_npz = data_dir / "champion.npz"
    best = float(report["gpu_best_confirm_acc"])
    prev = -1.0
    if champ_npz.exists():
        try:
            prev = float(np.load(champ_npz)["acc"])
        except Exception:
            prev = -1.0
    if best <= prev + 1e-12:
        return False
    np.savez(champ_npz, **pack)
    out_dir = root / "runs" / "gpu-evolve"
    np.savez(out_dir / "next_action_champion.npz", **pack)
    meta = {
        "recipe": report["recipe"],
        "n_train": report["n_train"],
        "gpu_best_confirm_acc": best,
        "ridge_confirm_acc": report["ridge_confirm_acc"],
        "majority_confirm_acc": report["majority_confirm_acc"],
        "n_pop": n_pop,
        "n_kc": report.get("n_kc"),
        "k_winners": report.get("k_winners"),
        "pn_dim": report.get("pn_dim"),
        "promoted": True,
    }
    (data_dir / "champion.json").write_text(json.dumps(meta, indent=2) + "\n")
    return True


def predict_next_action(text: str, *, pack: Path | None = None) -> dict[str, Any]:
    """CPU readout from the frozen wave-5 champion. Not a keep path."""
    root = Path(__file__).resolve().parents[1]
    pack = pack or root / "data" / "next_action" / "champion.npz"
    data = np.load(pack)
    W_pn = data["W_pn_kc"]
    W = data["W_kc_mbon"]
    k = int(data["k_winners"])
    x = _hash(text, int(W_pn.shape[0]))
    H = _kc_codes(x[None, :], W_pn, k)
    scores = H[0] @ W
    i = int(np.argmax(scores))
    exps = np.exp(scores - scores.max())
    return {
        "ok": True,
        "label": FAMILIES[i],
        "p": float(exps[i] / exps.sum()),
        "source": "next_action_gpu_champion",
        "n_params": int(W_pn.size + W.size),
    }


def shadow_confirm(
    *,
    pack: Path | None = None,
    episodes: Path | None = None,
    confirm_frac: float = 0.2,
    gold_only: bool = False,
) -> dict[str, Any]:
    """Offline confirm-split readout for the frozen champion pack."""
    root = Path(__file__).resolve().parents[1]
    pack = pack or root / "data" / "next_action" / "champion.npz"
    path = episodes or EPISODES
    data = np.load(pack)
    W_pn = data["W_pn_kc"]
    W = data["W_kc_mbon"]
    k = int(data["k_winners"])
    dim = int(W_pn.shape[0])
    X, y = load_xy(path, gold_only=gold_only, dim=dim)
    cut = max(8, int(len(y) * (1.0 - confirm_frac)))
    Xte, yte = X[cut:], y[cut:]
    Ht = _kc_codes(Xte, W_pn, k)
    pred = (Ht @ W).argmax(1)
    acc = float((pred == yte).mean()) if len(yte) else 0.0
    counts = np.bincount(yte, minlength=len(FAMILIES))
    majority = float(counts.max() / len(yte)) if len(yte) else 0.0
    by_fam = {
        FAMILIES[i]: {
            "n": int(counts[i]) if i < len(counts) else 0,
            "acc": float((pred[yte == i] == i).mean()) if i < len(counts) and counts[i] else None,
        }
        for i in range(len(FAMILIES))
    }
    report = {
        "schema": "flyforge.next_action_shadow_confirm.v1",
        "n_confirm": int(len(yte)),
        "champion_confirm_acc": acc,
        "majority": majority,
        "pack_reported_gpu": float(data["acc"]) if "acc" in data.files else None,
        "n_kc": int(data["n_kc"]) if "n_kc" in data.files else int(W.shape[0]),
        "k_winners": k,
        "pn_dim": dim,
        "by_family": by_fam,
    }
    shadow_dir = Path.home() / ".z0int" / "shadow"
    shadow_dir.mkdir(parents=True, exist_ok=True)
    (shadow_dir / "champion_confirm.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def live_shadow(prompt: str, *, log: Path | None = None) -> dict[str, Any]:
    """One shadow row: next-action champion (+ jev-distill fly when available)."""
    t0 = time.perf_counter()
    next_a = predict_next_action(prompt)
    next_a["ms"] = (time.perf_counter() - t0) * 1000.0
    try:
        from .jev_distill import predict_prompt as _jev_predict

        jev: dict[str, Any] = dict(_jev_predict(prompt))
    except Exception as exc:  # pragma: no cover - optional peer pack
        jev = {"ok": False, "error": type(exc).__name__}
    row = {
        "ts": time.time(),
        "prompt": (prompt or "")[:400],
        "next_action": next_a,
        "jev": jev,
        "disagree": bool(next_a.get("ok") and jev.get("ok")) and next_a.get("label") != jev.get("label"),
    }
    dest = log or (Path.home() / ".z0int" / "shadow" / "jev-fly.jsonl")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    return row
