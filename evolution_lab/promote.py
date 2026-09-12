"""Promotion ladder: most experiments die young (ASHA-style)."""

from __future__ import annotations

LEVELS = {
    0: {"name": "sanity", "seeds": 1, "n_train": 16, "n_val": 8, "n_confirm": 8, "n_ood": 8},
    1: {"name": "tiny", "seeds": 1, "n_train": 192, "n_val": 48, "n_confirm": 48, "n_ood": 32},
    2: {"name": "medium", "seeds": 3, "n_train": 192, "n_val": 48, "n_confirm": 48, "n_ood": 32},
    3: {"name": "full", "seeds": 5, "n_train": 256, "n_val": 64, "n_confirm": 64, "n_ood": 48},
}

# L4 replication / L5 adversarial are declared, not implemented as extra compute here.
DECLARED_ONLY = {4: "independent_replication", 5: "ood_adversarial"}


def promote(level: int, success: float, violations: float, *, min_success: float = 0.45) -> bool:
    if violations > 0:
        return False
    if level <= 0:
        return True
    return success >= min_success


def epistemic(delta_hv: float, replicated: bool, new_niche: bool) -> str:
    if replicated and delta_hv > 1e-4:
        return "important"
    if delta_hv > 1e-4 or new_niche:
        return "real"
    return "cool"
