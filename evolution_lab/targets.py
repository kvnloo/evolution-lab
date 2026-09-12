"""Registered product target for FlyForge P0.

Hosted joules stay unknown until measured. This is a declaration, not a
measured Pareto point.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProductTarget:
    success_vs_teacher: float = 0.95
    extra_violations: int = 0
    joules: float | None = None


PRODUCT_TARGET = ProductTarget(success_vs_teacher=0.95, extra_violations=0, joules=None)
