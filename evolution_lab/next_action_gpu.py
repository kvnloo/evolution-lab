"""GPU mushroom-body population on compiled Hermes next-action episodes.

Frozen judge is still CPU last-step / ridge comparison. Recipe (n_train,
gold_only, confirm_frac) is the evolvable object.
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
    xs, ys = [], []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            fam = rec.get("family")
            if fam not in FAM_I:
                continue
            if gold_only and not rec.get("gold"):
                continue
            xs.append(_hash(rec.get("text") or "", dim))
            ys.append(FAM_I[fam])
    if not xs:
        raise FileNotFoundError(f"no next-action rows in {path}")
    return np.stack(xs), np.asarray(ys, dtype=np.int32)


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
    k_winners: int = 10,
    epochs: int = 6,
    lr: float = 0.35,
    seed: int = 0,
    episodes: Path | None = None,
) -> dict[str, Any]:
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.25")
    recipe = recipe or DataRecipe(source="next_action", n_train=2048)
    path = episodes or EPISODES
    X, y = load_xy(path, gold_only=recipe.gold_only)
    n = len(y)
    cut = max(8, int(n * (1.0 - recipe.confirm_frac)))
    Xtr, ytr = X[:cut], y[:cut]
    Xte, yte = X[cut:], y[cut:]
    if len(Xtr) > recipe.n_train:
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
    report = {
        "schema": "flyforge.next_action_gpu.v1",
        "recipe": recipe.__dict__ if hasattr(recipe, "__dict__") else dict(recipe),
        "n_train": int(len(ytr)),
        "n_confirm": int(len(yte)),
        "n_pop": n_pop,
        "gpu_filter_s": gpu_s,
        "gpu_best_confirm_acc": best,
        "ridge_confirm_acc": ridge,
        "gpu_beats_ridge": best > ridge,
        "note": "GPU filters candidates; ridge is the CPU baseline. Do not keep from GPU acc alone.",
    }
    out = Path(__file__).resolve().parents[1] / "runs" / "gpu-evolve" / "next_action_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    return report
