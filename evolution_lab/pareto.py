"""Pareto dominance and 2-D hypervolume (success vs cost)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Point:
    experiment_id: str
    success: float
    cost: float
    ood: float
    params: float
    violations: float

    def maximize_vector(self) -> tuple[float, ...]:
        # maximize success, ood; minimize cost/params/violations
        return (self.success, self.ood, -self.cost, -self.params, -self.violations)


def dominates(a: Point, b: Point) -> bool:
    av, bv = a.maximize_vector(), b.maximize_vector()
    return all(x >= y for x, y in zip(av, bv)) and any(x > y for x, y in zip(av, bv))


def pareto_front(points: Sequence[Point]) -> list[Point]:
    front = []
    for p in points:
        if any(dominates(q, p) for q in points if q is not p):
            continue
        front.append(p)
    return front


def hypervolume_2d(points: Sequence[Point], *, ref_success: float = 0.0, ref_cost: float = 1.0) -> float:
    """Axis-aligned HV in (success ↑, cost ↓) with reference (ref_success, ref_cost).

    Cost is normalized so larger cost is worse. Points with cost >= ref_cost
    contribute nothing on that axis.
    """
    usable = [p for p in points if p.success >= ref_success and p.cost <= ref_cost]
    if not usable:
        return 0.0
    front = pareto_front(usable)
    ordered = sorted(front, key=lambda p: p.cost)
    hv = 0.0
    prev_cost = ref_cost
    # walk cheap → expensive, adding rectangles of height (success - ref)
    # using 2d front projected onto (cost, success) only
    # Recompute 2d-non-dominated on those two axes
    cand = []
    for p in sorted(usable, key=lambda p: (p.cost, -p.success)):
        if cand and p.success <= cand[-1].success:
            continue
        cand.append(p)
    prev_cost = ref_cost
    last_s = ref_success
    # sort by increasing cost
    for p in sorted(cand, key=lambda p: p.cost):
        width = prev_cost - p.cost
        height = p.success - ref_success
        if width > 0 and height > 0:
            hv += width * height
        prev_cost = p.cost
        last_s = p.success
        _ = last_s
    return float(hv)


def delta_hypervolume(before: Sequence[Point], extra: Point, **kw) -> float:
    return hypervolume_2d(list(before) + [extra], **kw) - hypervolume_2d(before, **kw)
