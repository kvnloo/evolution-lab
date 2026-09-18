"""2×2: {old, refined labels} × {old, rich PNs} → coverage@≥95% precision.

Frozen arch: n_kc=96, k_winners=20. Frozen chronological confirm (last 20% of
all episodes). Promote-only if coverage@0.95 beats gen-0 DELEGATE baseline.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from .coverage_metric import (
    GEN0_CONFIRM_ACC,
    GEN0_COVERAGE_AT_95,
    coverage_at_precision,
    gen0_baseline,
    risk_coverage_curve,
)
from .jax_mb import numpy_perms, population_mbon
from .models import _kc_codes, _sparse_pn_kc
from .next_action_gpu import EPISODES, FAMILIES, FAM_I, _hash, ridge_acc
from .pn_features import combine_pn, structured_cues

N_KC = 96
K_WIN = 20
TEXT_DIM = 64


def _ctx_from_ep(rec: dict[str, Any]) -> dict[str, Any]:
    tool = str(rec.get("tool") or "")
    prev = rec.get("prev") or []
    tier = str(rec.get("tier") or "")
    fam = str(rec.get("family") or "")
    phase = ""
    tl = tool.lower()
    if any(x in tl for x in ("edit", "write", "patch", "apply")):
        phase = "edit"
    elif any(x in tl for x in ("test", "lint", "verify", "pytest")):
        phase = "verify"
    elif any(x in tl for x in ("search", "read", "grep", "glob", "browse")):
        phase = "search"
    elif any(x in tl for x in ("task", "agent", "delegate", "spawn")):
        phase = "delegate"
    return {
        "last_tool": tool,
        "session_depth": len(prev),
        "phase": phase,
        "has_retry": "retry" in tl or tier == "fail",
        "last_tool_fail": tier == "fail",
        "has_test_signal": "test" in tl or "pytest" in tl,
        "harness": "hermes",  # compiled episodes are Hermes/OMP histories
        "repo": str(rec.get("session") or "")[:12],
        "cwd": "",
        "candidate_count": 0,
    }


def load_episodes(
    path: Path | None = None,
    *,
    rich: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return X, y, is_gold, sessions. X always covers all families."""
    path = path or EPISODES
    xs: list[np.ndarray] = []
    ys: list[int] = []
    gold: list[bool] = []
    sessions: list[str] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            fam = rec.get("family")
            if fam not in FAM_I:
                continue
            text = rec.get("text") or rec.get("user") or ""
            prev = rec.get("prev") or []
            if prev:
                text = text + " " + " ".join(str(x) for x in prev)
            base = _hash(text, TEXT_DIM)
            if rich:
                x = combine_pn(base, _ctx_from_ep(rec))
            else:
                x = base
            xs.append(x)
            ys.append(FAM_I[fam])
            gold.append(bool(rec.get("gold") or rec.get("tier") == "gold"))
            sessions.append(str(rec.get("session") or ""))
    if not xs:
        raise FileNotFoundError(f"no episodes in {path}")
    return (
        np.stack(xs).astype(np.float32),
        np.asarray(ys, dtype=np.int32),
        np.asarray(gold, dtype=bool),
        sessions,
    )


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(np.clip(z, -40, 40))
    return e / e.sum(axis=1, keepdims=True)


def _train_eval(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    *,
    n_pop: int,
    epochs: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    W_pn = _sparse_pn_kc(Xtr.shape[1], N_KC, rng)
    H = _kc_codes(Xtr, W_pn, K_WIN)
    Ht = _kc_codes(Xte, W_pn, K_WIN)
    bank = np.stack(
        [numpy_perms(len(ytr), epochs, np.random.default_rng(seed + i)) for i in range(n_pop)]
    )
    t0 = time.perf_counter()
    Ws = population_mbon(H, ytr, bank, 0.35)
    gpu_s = time.perf_counter() - t0
    logits = np.stack([Ht @ Ws[p] for p in range(n_pop)])
    preds = logits.argmax(axis=2)
    acc = (preds == yte[None, :]).mean(axis=1)
    best_i = int(np.argmax(acc))
    best_logits = logits[best_i]
    best_pred = preds[best_i]
    probs = _softmax_rows(best_logits)
    order = np.argsort(-probs, axis=1)
    p1 = probs[np.arange(len(probs)), order[:, 0]]
    p2 = probs[np.arange(len(probs)), order[:, 1]]
    margin = p1 - p2
    ridge = ridge_acc(Xtr, ytr, Xte, yte)
    majority = float(np.bincount(yte, minlength=len(FAMILIES)).max() / max(len(yte), 1))

    # Local-candidate mask: EXECUTE | DELEGATE preds (cascade family set)
    del_i = FAMILIES.index("DELEGATE")
    exec_i = FAMILIES.index("EXECUTE")
    local_mask = np.isin(best_pred, [del_i, exec_i])
    del_mask = best_pred == del_i

    cov95_all = coverage_at_precision(yte, best_pred, margin, floor=0.95)
    cov95_local = coverage_at_precision(yte, best_pred, margin, floor=0.95, mask=local_mask)
    cov95_del = coverage_at_precision(yte, best_pred, margin, floor=0.95, mask=del_mask)
    # Also p-score ranking
    cov95_p = coverage_at_precision(yte, best_pred, p1, floor=0.95, mask=local_mask)

    curves = {
        "margin_all": risk_coverage_curve(yte, best_pred, margin),
        "margin_local_families": risk_coverage_curve(yte, best_pred, margin, mask=local_mask),
        "margin_delegate_pred": risk_coverage_curve(yte, best_pred, margin, mask=del_mask),
        "p_local_families": risk_coverage_curve(yte, best_pred, p1, mask=local_mask),
    }

    pack = {
        "W_pn_kc": np.asarray(W_pn),
        "W_kc_mbon": np.asarray(Ws[best_i]),
        "k_winners": np.int32(K_WIN),
        "n_kc": np.int32(N_KC),
        "acc": np.float64(float(acc[best_i])),
        "ridge": np.float64(ridge),
        "majority": np.float64(majority),
        "pn_dim": np.int32(Xtr.shape[1]),
        "coverage_at_95": np.float64(cov95_local["coverage"]),
        "precision_at_95": np.float64(cov95_local["precision"] or 0.0),
    }
    return {
        "gpu_best_confirm_acc": float(acc[best_i]),
        "ridge_confirm_acc": ridge,
        "majority_confirm_acc": majority,
        "gpu_s": gpu_s,
        "n_train": int(len(ytr)),
        "n_confirm": int(len(yte)),
        "pn_dim": int(Xtr.shape[1]),
        "coverage_at_95": cov95_local,
        "coverage_at_95_all": cov95_all,
        "coverage_at_95_delegate": cov95_del,
        "coverage_at_95_p": cov95_p,
        "curves": curves,
        "beats_gen0_cov95": bool(cov95_local.get("beats_gen0")),
        "pack": pack,
        "best_i": best_i,
    }


def run_2x2(
    *,
    n_pop: int = 256,
    epochs: int = 6,
    seed: int = 0,
    confirm_frac: float = 0.2,
    episodes: Path | None = None,
    promote: bool = False,
) -> dict[str, Any]:
    """Run four cells; optionally promote if cov@95% beats gen-0 after rebench."""
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.25")
    cells: dict[str, dict[str, Any]] = {}
    best_name = None
    best_cov = -1.0
    best_pack = None

    configs = (
        ("old_labels_old_pn", False, False),
        ("old_labels_rich_pn", False, True),
        ("refined_labels_old_pn", True, False),
        ("refined_labels_rich_pn", True, True),
    )
    cache: dict[bool, tuple] = {}

    for name, refined, rich in configs:
        if rich not in cache:
            cache[rich] = load_episodes(episodes, rich=rich)
        X, y, is_gold, _sessions = cache[rich]
        n = len(y)
        cut = max(8, int(n * (1.0 - confirm_frac)))
        Xte, yte = X[cut:], y[cut:]
        if refined:
            tr = np.where(is_gold[:cut])[0]
            if tr.size < 64:
                # fall back: gold ∪ fail-tier already in gold flag; use all if too thin
                tr = np.arange(cut)
            Xtr, ytr = X[tr], y[tr]
        else:
            Xtr, ytr = X[:cut], y[:cut]
        cell = _train_eval(Xtr, ytr, Xte, yte, n_pop=n_pop, epochs=epochs, seed=seed + hash(name) % 1000)
        cell["labels"] = "refined_gold" if refined else "old_all"
        cell["features"] = "rich_structured" if rich else "hash_trigram"
        cell["name"] = name
        cells[name] = {k: v for k, v in cell.items() if k != "pack"}
        cov = float((cell["coverage_at_95"] or {}).get("coverage") or 0.0)
        if cov > best_cov + 1e-12:
            best_cov = cov
            best_name = name
            best_pack = cell["pack"]

    report = {
        "schema": "flyforge.next_action_2x2.v1",
        "gen0": gen0_baseline(),
        "metric": "coverage_at_95_local_EXEC_DEL_margin",
        "arch": {"n_kc": N_KC, "k_winners": K_WIN, "text_pn_dim": TEXT_DIM},
        "cells": cells,
        "best_cell": best_name,
        "best_coverage_at_95": best_cov,
        "beats_gen0": best_cov > GEN0_COVERAGE_AT_95 + 1e-12,
        "promoted": False,
    }

    root = Path(__file__).resolve().parents[1]
    out_dir = root / "runs" / "gpu-evolve"
    out_dir.mkdir(parents=True, exist_ok=True)
    if best_pack is not None:
        np.savez(out_dir / "next_action_2x2_candidate.npz", **best_pack)

    if promote and report["beats_gen0"] and best_pack is not None:
        from .coverage_metric import coverage_at_precision
        from .next_action_gpu import evaluate_coverage_policy, shadow_confirm

        cand_path = out_dir / "next_action_2x2_candidate.npz"
        # Production-path rebench with matching feature dim.
        conf = shadow_confirm(pack=cand_path)
        cov_pol = evaluate_coverage_policy(pack=cand_path)
        report["rebench"] = {
            "confirm_acc": conf.get("champion_confirm_acc"),
            "cascade_coverage": cov_pol.get("coverage"),
            "cascade_local_precision": cov_pol.get("local_precision"),
            "by_local_family": cov_pol.get("by_local_family"),
            "pn_dim": conf.get("pn_dim"),
        }
        new_acc = float(conf.get("champion_confirm_acc") or 0.0)
        new_cov = float(best_pack["coverage_at_95"])
        del_prec = ((cov_pol.get("by_local_family") or {}).get("DELEGATE") or {}).get("prec")
        ok_acc = new_acc + 1e-12 >= GEN0_CONFIRM_ACC - 0.01
        ok_del = del_prec is None or float(del_prec) >= 0.90
        ok_cov = new_cov > GEN0_COVERAGE_AT_95 + 1e-12
        if ok_acc and ok_del and ok_cov:
            champ = root / "data" / "next_action"
            champ.mkdir(parents=True, exist_ok=True)
            pack_write = dict(best_pack)
            pack_write["acc"] = np.float64(new_acc)
            np.savez(champ / "champion.npz", **pack_write)
            np.savez(out_dir / "next_action_champion.npz", **pack_write)
            (champ / "champion.json").write_text(
                json.dumps(
                    {
                        "recipe": {"source": "2x2", "cell": best_name},
                        "n_train": cells[best_name]["n_train"],
                        "gpu_best_confirm_acc": new_acc,
                        "coverage_at_95": new_cov,
                        "n_kc": N_KC,
                        "k_winners": K_WIN,
                        "pn_dim": int(best_pack["pn_dim"]),
                        "generation": 1,
                        "promoted": True,
                        "promote_metric": "coverage_at_95",
                    },
                    indent=2,
                )
                + "\n"
            )
            report["promoted"] = True
            report["promote_note"] = "gen-1 rich PN after production rebench"
        else:
            report["promoted"] = False
            report["promote_note"] = (
                f"blocked rebench acc={new_acc:.4f} cov={new_cov:.4f} "
                f"del_prec={del_prec} ok_acc={ok_acc} ok_del={ok_del} ok_cov={ok_cov}"
            )
    elif promote:
        report["promote_note"] = "no promote: beats_gen0 false or no pack"


    shadow = Path.home() / ".z0int" / "shadow"
    shadow.mkdir(parents=True, exist_ok=True)
    # strip non-json from nested
    serializable = json.loads(json.dumps(report, default=str))
    (shadow / "experiment_2x2.json").write_text(json.dumps(serializable, indent=2) + "\n")
    (out_dir / "experiment_2x2.json").write_text(json.dumps(serializable, indent=2) + "\n")
    return report
