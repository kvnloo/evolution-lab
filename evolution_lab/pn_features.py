"""Structured PN cues for next-action (gen-1 representation path).

Keep wave-5 hash PNs; concatenate a small structured bank. Architecture
(n_kc/k_winners) stays frozen while ABAB mutates which cues are on.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

# Fixed slots — order is part of the feature recipe contract.
CUE_NAMES = (
    "has_retry",
    "last_tool_fail",
    "in_git_repo",
    "has_test_signal",
    "phase_edit",
    "phase_verify",
    "phase_search",
    "phase_delegate",
    "harness_omp",
    "harness_hermes",
    "candidate_count_norm",
    "session_depth_norm",
)


def _bucket_hash(s: str, dim: int, salt: bytes) -> np.ndarray:
    out = np.zeros(dim, dtype=np.float32)
    if not s:
        return out
    h = hashlib.blake2b(salt + s.encode("utf-8"), digest_size=8).digest()
    out[int.from_bytes(h[:4], "little") % dim] = 1.0
    return out


def structured_cues(ctx: dict[str, Any] | None, *, dim: int = 16) -> np.ndarray:
    """Return length-len(CUE_NAMES)+extra hash dims structured vector."""
    ctx = ctx or {}
    flags = np.zeros(len(CUE_NAMES), dtype=np.float32)
    mapping = {
        "has_retry": bool(ctx.get("retry") or ctx.get("has_retry")),
        "last_tool_fail": bool(ctx.get("last_tool_fail") or ctx.get("tool_error")),
        "in_git_repo": bool(ctx.get("in_git_repo") or ctx.get("git")),
        "has_test_signal": bool(ctx.get("test_state") or ctx.get("has_test_signal")),
        "phase_edit": str(ctx.get("phase") or "").lower() in {"edit", "write", "patch"},
        "phase_verify": str(ctx.get("phase") or "").lower() in {"verify", "test", "lint"},
        "phase_search": str(ctx.get("phase") or "").lower() in {"search", "read", "browse"},
        "phase_delegate": str(ctx.get("phase") or "").lower() in {"delegate", "task", "subagent"},
        "harness_omp": str(ctx.get("harness") or "").lower() in {"omp", "oh-my-pi", "pi"},
        "harness_hermes": str(ctx.get("harness") or "").lower() == "hermes",
    }
    for i, name in enumerate(CUE_NAMES):
        if name.endswith("_norm"):
            continue
        flags[i] = 1.0 if mapping.get(name) else 0.0
    # norms
    cand = float(ctx.get("candidate_count") or 0.0)
    depth = float(ctx.get("session_depth") or 0.0)
    flags[CUE_NAMES.index("candidate_count_norm")] = float(np.tanh(cand / 8.0))
    flags[CUE_NAMES.index("session_depth_norm")] = float(np.tanh(depth / 20.0))
    # hashed categorical crumbs (cwd/repo/last_tool) — private, not logged raw
    extra = []
    for key, salt in (("cwd", b"cwd:"), ("repo", b"repo:"), ("last_tool", b"tool:")):
        extra.append(_bucket_hash(str(ctx.get(key) or ""), dim, salt))
    return np.concatenate([flags, *extra]).astype(np.float32)


def combine_pn(text_pn: np.ndarray, ctx: dict[str, Any] | None = None) -> np.ndarray:
    """Concat text hash PN with structured cues; L2-normalize."""
    s = structured_cues(ctx)
    v = np.concatenate([np.asarray(text_pn, dtype=np.float32).reshape(-1), s])
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def feature_recipe_default() -> dict[str, Any]:
    return {
        "schema": "flyforge.pn_feature_recipe.v1",
        "text_pn_dim": 64,
        "cue_names": list(CUE_NAMES),
        "hash_buckets_per_key": 16,
        "hash_keys": ["cwd", "repo", "last_tool"],
        "note": "Gen-1 scaffold. Train heads on concat only after 2x2 labels×features cell.",
    }
