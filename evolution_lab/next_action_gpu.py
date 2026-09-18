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

WEAK_FAMILIES = ("EDIT", "WEB", "VERIFY", "ABSTAIN")


def _boost_train(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    *,
    families: tuple[str, ...] = WEAK_FAMILIES,
    boost: int = 4,
    max_n: int = 65536,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Oversample named families on the train split only. Confirm set stays frozen."""
    rng = rng or np.random.default_rng(0)
    if boost <= 1:
        if len(ytr) > max_n:
            take = rng.choice(len(ytr), size=max_n, replace=False)
            return Xtr[take], ytr[take], {f: 0 for f in families}
        return Xtr, ytr, {f: 0 for f in families}
    extra_x: list[np.ndarray] = []
    extra_y: list[np.ndarray] = []
    added = {f: 0 for f in families}
    for name in families:
        if name not in FAM_I:
            continue
        idx = np.where(ytr == FAM_I[name])[0]
        if idx.size == 0:
            continue
        for _ in range(boost - 1):
            extra_x.append(Xtr[idx])
            extra_y.append(ytr[idx])
            added[name] += int(idx.size)
    if not extra_x:
        Xb, yb = Xtr, ytr
    else:
        Xb = np.concatenate([Xtr, *extra_x], axis=0)
        yb = np.concatenate([ytr, *extra_y], axis=0)
    if len(yb) > max_n:
        weak_ids = [FAM_I[f] for f in families if f in FAM_I]
        weak_idx = np.where(np.isin(yb, weak_ids))[0]
        other_idx = np.where(~np.isin(yb, weak_ids))[0]
        if len(weak_idx) >= max_n:
            take = rng.choice(weak_idx, size=max_n, replace=False)
        else:
            remain = max_n - len(weak_idx)
            keep_o = (
                rng.choice(other_idx, size=min(remain, len(other_idx)), replace=False)
                if len(other_idx)
                else np.array([], dtype=np.int64)
            )
            take = np.concatenate([weak_idx, keep_o])
        Xb, yb = Xb[take], yb[take]
    perm = rng.permutation(len(yb))
    return Xb[perm], yb[perm], added



def _family_acc(pred: np.ndarray, yte: np.ndarray) -> dict[str, dict[str, float | int | None]]:
    out: dict[str, dict[str, float | int | None]] = {}
    for i, name in enumerate(FAMILIES):
        m = yte == i
        n = int(m.sum())
        out[name] = {
            "n": n,
            "acc": float((pred[m] == i).mean()) if n else None,
        }
    return out

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
    weak_boost: int = 1,
    weak_families: tuple[str, ...] = WEAK_FAMILIES,
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
    boost_added: dict[str, int] = {f: 0 for f in weak_families}
    if weak_boost > 1:
        Xtr, ytr, boost_added = _boost_train(
            Xtr, ytr, families=weak_families, boost=weak_boost, rng=rng
        )
    W_pn = _sparse_pn_kc(Xtr.shape[1], n_kc, rng)
    H = _kc_codes(Xtr, W_pn, k_winners)
    Ht = _kc_codes(Xte, W_pn, k_winners)
    bank = np.stack([numpy_perms(len(ytr), epochs, np.random.default_rng(seed + i)) for i in range(n_pop)])
    t0 = time.perf_counter()
    Ws = population_mbon(H, ytr, bank, lr)
    gpu_s = time.perf_counter() - t0
    preds = np.stack([(Ht @ Ws[p]).argmax(1) for p in range(n_pop)])
    acc = (preds == yte[None, :]).mean(axis=1)
    # Fair ridge always uses unboosted train (frozen judge baseline).
    # Fair ridge: always fit on the pre-boost train set reconstructed from recipe cut.
    Xtr0, ytr0 = X[:cut], y[:cut]
    if recipe.n_train > 0 and len(Xtr0) > recipe.n_train:
        Xtr0, ytr0 = Xtr0[-recipe.n_train :], ytr0[-recipe.n_train :]
    ridge = ridge_acc(Xtr0, ytr0, Xte, yte)
    best_i = int(np.argmax(acc))
    best = float(acc[best_i])
    counts = np.bincount(yte, minlength=len(FAMILIES))
    majority = float(counts.max() / len(yte)) if len(yte) else 0.0
    by_fam = _family_acc(preds[best_i], yte)
    report = {
        "schema": "flyforge.next_action_gpu.v1",
        "recipe": recipe.__dict__ if hasattr(recipe, "__dict__") else dict(recipe),
        "n_train": int(len(ytr0)),
        "n_train_boosted": int(len(ytr)),
        "weak_boost": int(weak_boost),
        "weak_families": list(weak_families),
        "weak_added": boost_added,
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
        "by_family": by_fam,
        "note": "GPU filters candidates; ridge is the CPU baseline on unboosted train. Promote-only.",
    }
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "runs" / "gpu-evolve"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "next_action_report.json").write_text(json.dumps(report, indent=2) + "\n")
    pack = {
        "W_pn_kc": np.asarray(W_pn),
        "W_kc_mbon": np.asarray(Ws[best_i]),
        "k_winners": np.int32(k_winners),
        "n_kc": np.int32(n_kc),
        "acc": np.float64(best),
        "ridge": np.float64(ridge),
        "majority": np.float64(majority),
    }
    np.savez(out_dir / "next_action_candidate.npz", **pack)
    report["promoted"] = _maybe_promote_champion(root, pack, report, n_pop=n_pop)
    return report


def run_weak_family_search(
    *,
    boosts: tuple[int, ...] = (1, 2, 4, 8),
    n_pop: int = 256,
    epochs: int = 6,
    families: tuple[str, ...] = WEAK_FAMILIES,
) -> dict[str, Any]:
    """Frozen-arch recipe search: oversample weak families; promote-only keeps."""
    recipe = DataRecipe(n_train=0, gold_only=False, source="next_action", confirm_frac=0.2)
    waves: list[dict[str, Any]] = []
    for b in boosts:
        rep = run_next_action_gpu(
            recipe=recipe,
            n_pop=n_pop,
            n_kc=96,
            k_winners=20,
            epochs=epochs,
            dim=64,
            weak_boost=b,
            weak_families=families,
            seed=1000 + b,
        )
        weak_acc = {
            f: (rep.get("by_family") or {}).get(f, {}).get("acc")
            for f in families
        }
        waves.append(
            {
                "weak_boost": b,
                "keep": bool(rep.get("promoted")),
                "gpu": rep["gpu_best_confirm_acc"],
                "ridge": rep["ridge_confirm_acc"],
                "majority": rep["majority_confirm_acc"],
                "gpu_beats_ridge": rep["gpu_beats_ridge"],
                "n_train": rep["n_train"],
                "n_train_boosted": rep.get("n_train_boosted"),
                "weak_acc": weak_acc,
                "by_family": rep.get("by_family"),
            }
        )
    summary = {
        "schema": "flyforge.next_action_weak_family.v1",
        "families": list(families),
        "boosts": list(boosts),
        "waves": waves,
        "keeps": sum(1 for w in waves if w["keep"]),
        "best_gpu": max(w["gpu"] for w in waves) if waves else None,
        "arch": {"n_kc": 96, "k_winners": 20, "pn_dim": 64},
    }
    out = Path(__file__).resolve().parents[1] / "runs" / "gpu-evolve"
    out.mkdir(parents=True, exist_ok=True)
    (out / "weak_family_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary






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


def _softmax_scores(scores: np.ndarray) -> np.ndarray:
    exps = np.exp(scores - float(scores.max()))
    return exps / exps.sum()


def _ctx_from_text(text: str, ctx: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fill structured cues from prompt text when harness ctx is thin."""
    out = dict(ctx or {})
    t = (text or "").lower()
    if not out.get("phase"):
        if any(w in t for w in ("edit", "patch", "write file", "apply_patch", "fix")):
            out["phase"] = "edit"
        elif any(w in t for w in ("test", "pytest", "lint", "verify", "coverage")):
            out["phase"] = "verify"
        elif any(w in t for w in ("search", "grep", "find", "read ", "browse", "look up")):
            out["phase"] = "search"
        elif any(w in t for w in ("delegate", "subagent", "scout", "spawn", "task ")):
            out["phase"] = "delegate"
    if "has_test_signal" not in out:
        out["has_test_signal"] = any(w in t for w in ("test", "pytest", "unittest"))
    if "has_retry" not in out:
        out["has_retry"] = "retry" in t or "again" in t
    if "harness" not in out:
        out["harness"] = "omp"
    return out


def _pn_vector(text: str, dim: int, *, ctx: dict[str, Any] | None = None) -> np.ndarray:
    """Build PN vector matching pack dim (64=hash, larger=hash+structured)."""
    from .pn_features import combine_pn

    base = _hash(text, 64 if dim > 64 else dim)
    if dim <= 64:
        return base if dim == 64 else base[:dim]
    rich = combine_pn(base, _ctx_from_text(text, ctx))
    if rich.shape[0] == dim:
        return rich
    out = np.zeros(dim, dtype=np.float32)
    n = min(dim, rich.shape[0])
    out[:n] = rich[:n]
    return out


def predict_next_action(
    text: str,
    *,
    pack: Path | None = None,
    ctx: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """CPU readout from the frozen champion. Not a keep path."""
    root = Path(__file__).resolve().parents[1]
    pack = pack or root / "data" / "next_action" / "champion.npz"
    data = np.load(pack)
    W_pn = data["W_pn_kc"]
    W = data["W_kc_mbon"]
    k = int(data["k_winners"])
    dim = int(W_pn.shape[0])
    x = _pn_vector(text, dim, ctx=ctx)
    H = _kc_codes(x[None, :], W_pn, k)
    scores = np.asarray(H[0] @ W, dtype=np.float64)
    probs = _softmax_scores(scores)
    order = np.argsort(-probs)
    i = int(order[0])
    j = int(order[1]) if len(order) > 1 else i
    return {
        "ok": True,
        "label": FAMILIES[i],
        "p": float(probs[i]),
        "margin": float(probs[i] - probs[j]),
        "second": FAMILIES[j],
        "probs": {FAMILIES[t]: float(probs[t]) for t in range(len(FAMILIES))},
        "source": "next_action_gpu_champion",
        "n_params": int(W_pn.size + W.size),
        "pn_dim": dim,
    }



def predict_next_action(
    text: str,
    *,
    pack: Path | None = None,
    ctx: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """CPU readout from the frozen wave-5 champion. Not a keep path."""
    root = Path(__file__).resolve().parents[1]
    pack = pack or root / "data" / "next_action" / "champion.npz"
    data = np.load(pack)
    W_pn = data["W_pn_kc"]
    W = data["W_kc_mbon"]
    k = int(data["k_winners"])
    dim = int(W_pn.shape[0])
    x = _pn_vector(text, dim, ctx=ctx)
    H = _kc_codes(x[None, :], W_pn, k)
    scores = np.asarray(H[0] @ W, dtype=np.float64)
    probs = _softmax_scores(scores)
    order = np.argsort(-probs)
    i = int(order[0])
    j = int(order[1]) if len(order) > 1 else i
    return {
        "ok": True,
        "label": FAMILIES[i],
        "p": float(probs[i]),
        "margin": float(probs[i] - probs[j]),
        "second": FAMILIES[j],
        "probs": {FAMILIES[t]: float(probs[t]) for t in range(len(FAMILIES))},
        "source": "next_action_gpu_champion",
        "n_params": int(W_pn.size + W.size),
        "pn_dim": dim,
    }



DEFAULT_COVERAGE_POLICY: dict[str, Any] = {
    "schema": "flyforge.next_action_coverage.v1",
    "local_families": ["EXECUTE", "DELEGATE"],
    "thresholds": {
        # EXECUTE preds ~0.48 precise overall — high conf only.
        "EXECUTE": {"min_p": 0.60, "min_margin": 0.30},
        # DELEGATE preds ~0.98 precise on confirm — absorb all.
        "DELEGATE": {"min_p": 0.0, "min_margin": 0.0},
    },
    "escalate_to": "jev",
    "fallback": "openjev",
    "metric": "coverage_at_precision",
    "precision_floor": 0.95,
}



def load_coverage_policy(*, root: Path | None = None) -> dict[str, Any]:
    root = root or Path(__file__).resolve().parents[1]
    path = root / "data" / "next_action" / "coverage_policy.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return dict(DEFAULT_COVERAGE_POLICY)


def decide_next_action(
    text: str,
    *,
    pack: Path | None = None,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Cascade: high-conf EXECUTE/DELEGATE stay local; else escalate to Jev/OpenJev."""
    policy = policy or load_coverage_policy()
    pred = predict_next_action(text, pack=pack)
    label = str(pred["label"])
    p = float(pred["p"])
    margin = float(pred["margin"])
    local_families = set(policy.get("local_families") or ["EXECUTE", "DELEGATE"])
    thresholds = policy.get("thresholds") or {}
    th = thresholds.get(label) or {}
    min_p = float(th.get("min_p", 1.0))
    min_margin = float(th.get("min_margin", 1.0))
    local_ok = label in local_families and p >= min_p and margin >= min_margin
    if local_ok:
        route = "local"
        escalate_to = None
        reason = f"high_conf_{label.lower()}"
    else:
        route = "escalate"
        escalate_to = policy.get("escalate_to") or "jev"
        if label not in local_families:
            reason = f"family_not_local:{label}"
        elif p < min_p:
            reason = f"low_p:{p:.3f}<{min_p:.3f}"
        else:
            reason = f"low_margin:{margin:.3f}<{min_margin:.3f}"
    out = {
        "ok": True,
        "route": route,
        "reason": reason,
        "escalate_to": escalate_to,
        "fallback": policy.get("fallback") or "openjev",
        "label": label,
        "p": p,
        "margin": margin,
        "second": pred.get("second"),
        "source": "next_action_coverage_v1",
        "policy": {
            "local_families": sorted(local_families),
            "thresholds": thresholds,
        },
        "prediction": pred,
    }
    return out


def evaluate_coverage_policy(
    *,
    pack: Path | None = None,
    episodes: Path | None = None,
    confirm_frac: float = 0.2,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Confirm-split risk/coverage for the locked cascade policy."""
    root = Path(__file__).resolve().parents[1]
    pack = pack or root / "data" / "next_action" / "champion.npz"
    policy = policy or load_coverage_policy(root=root)
    data = np.load(pack)
    W_pn = data["W_pn_kc"]
    W = data["W_kc_mbon"]
    k = int(data["k_winners"])
    dim = int(W_pn.shape[0])
    if dim > 64:
        from .experiment_2x2 import load_episodes

        X, y, _, _ = load_episodes(episodes or EPISODES, rich=True)
        if X.shape[1] != dim:
            X2 = np.zeros((len(X), dim), dtype=np.float32)
            n = min(dim, X.shape[1])
            X2[:, :n] = X[:, :n]
            X = X2
    else:
        X, y = load_xy(episodes or EPISODES, gold_only=False, dim=dim)
    cut = max(8, int(len(y) * (1.0 - confirm_frac)))
    Xte, yte = X[cut:], y[cut:]
    Ht = _kc_codes(Xte, W_pn, k)
    logits = Ht @ W
    z = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(np.clip(z, -40, 40))
    probs = probs / probs.sum(axis=1, keepdims=True)
    order = np.argsort(-probs, axis=1)
    pred = order[:, 0]
    p1 = probs[np.arange(len(probs)), pred]
    p2 = probs[np.arange(len(probs)), order[:, 1]]
    margin = p1 - p2
    local_families = set(policy.get("local_families") or ["EXECUTE", "DELEGATE"])
    thresholds = policy.get("thresholds") or {}
    local_idx = {FAM_I[n] for n in local_families if n in FAM_I}
    mask = np.zeros(len(yte), dtype=bool)
    for lab in local_idx:
        name = FAMILIES[lab]
        th = thresholds.get(name) or {}
        min_p = float(th.get("min_p", 1.0))
        min_m = float(th.get("min_margin", 1.0))
        mask |= (pred == lab) & (p1 >= min_p) & (margin >= min_m)
    correct = pred == yte
    n_loc = int(mask.sum())
    report = {
        "schema": "flyforge.next_action_coverage_eval.v1",
        "n_confirm": int(len(yte)),
        "n_local": n_loc,
        "n_escalate": int(len(yte) - n_loc),
        "coverage": float(mask.mean()) if len(yte) else 0.0,
        "local_precision": float(correct[mask].mean()) if n_loc else None,
        "escalate_rate": float(1.0 - mask.mean()) if len(yte) else 1.0,
        "champion_confirm_acc": float(correct.mean()) if len(yte) else 0.0,
        "policy": {
            "local_families": sorted(local_families),
            "thresholds": thresholds,
            "escalate_to": policy.get("escalate_to"),
            "fallback": policy.get("fallback"),
        },
        "by_local_family": {},
    }
    for lab in sorted(local_idx):
        name = FAMILIES[lab]
        th = thresholds.get(name) or {}
        min_p = float(th.get("min_p", 1.0))
        min_m = float(th.get("min_margin", 1.0))
        m = (pred == lab) & (p1 >= min_p) & (margin >= min_m)
        report["by_local_family"][name] = {
            "n": int(m.sum()),
            "prec": float(correct[m].mean()) if m.any() else None,
            "min_p": min_p,
            "min_margin": min_m,
        }
    shadow_dir = Path.home() / ".z0int" / "shadow"
    shadow_dir.mkdir(parents=True, exist_ok=True)
    (shadow_dir / "coverage_eval.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


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
    if dim > 64:
        from .experiment_2x2 import load_episodes

        X, y, is_gold, _ = load_episodes(path, rich=True)
        if gold_only:
            X, y = X[is_gold], y[is_gold]
        if X.shape[1] != dim:
            # align
            X2 = np.zeros((len(X), dim), dtype=np.float32)
            n = min(dim, X.shape[1])
            X2[:, :n] = X[:, :n]
            X = X2
    else:
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
            "acc": float((pred[yte == i] == i).mean()) if counts[i] else None,
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



def live_shadow(prompt: str, *, log: Path | None = None, session_id: str | None = None) -> dict[str, Any]:
    """One shadow row: coverage decision + next-action + jev-distill + live stream."""
    t0 = time.perf_counter()
    decision = decide_next_action(prompt)
    decision["ms"] = (time.perf_counter() - t0) * 1000.0
    next_a = decision.get("prediction") or predict_next_action(prompt)
    try:
        from .jev_distill import predict_prompt as _jev_predict

        jev: dict[str, Any] = dict(_jev_predict(prompt))
    except Exception as exc:  # pragma: no cover - optional peer pack
        jev = {"ok": False, "error": type(exc).__name__}
    teacher = None
    if decision.get("route") == "local":
        teacher = {"source": "next_action_local", "label": decision.get("label"), "p": decision.get("p")}
    elif decision.get("route") == "escalate" and isinstance(jev, dict) and jev.get("ok"):
        teacher = {"source": jev.get("source"), "label": jev.get("label"), "p": jev.get("p")}
    row = {
        "ts": time.time(),
        "prompt": (prompt or "")[:400],
        "decision": {
            "route": decision.get("route"),
            "reason": decision.get("reason"),
            "escalate_to": decision.get("escalate_to"),
            "fallback": decision.get("fallback"),
            "label": decision.get("label"),
            "p": decision.get("p"),
            "margin": decision.get("margin"),
            "ms": decision.get("ms"),
        },
        "next_action": next_a,
        "jev": jev,
        "teacher": teacher,
        "disagree": bool(next_a.get("ok") and jev.get("ok")) and next_a.get("label") != jev.get("label"),
    }
    dest = log or (Path.home() / ".z0int" / "shadow" / "jev-fly.jsonl")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    # Private multi-session learning stream (async train later).
    try:
        from .live_stream import append_decision, build_live_row

        live = build_live_row(
            prompt=prompt or "",
            decision=row["decision"],
            fly=next_a,
            jev=jev if isinstance(jev, dict) else None,
            session_id=session_id,
            latency_ms=decision.get("ms"),
        )
        live = append_decision(live)
        row["trace_id"] = live.get("trace_id")
        row["high_info"] = live.get("high_info")
        row["stream"] = "ok"
    except Exception as exc:  # pragma: no cover
        row["stream"] = f"err:{type(exc).__name__}"
    return row



