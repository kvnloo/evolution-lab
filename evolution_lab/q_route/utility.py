"""Trajectory utility: success, latency and risk as one scalar plus a tail term.

The utility of one ``(state, arm)`` row is intentionally explicit and tiny:

* the outcome space is frozen to three ordered outcomes --
  ``good`` (+1.0), ``degraded`` (0.0), ``bad`` (-1.0);
* :func:`value_distribution` turns ``success``, ``latency_p50_ms`` /
  ``latency_p95_ms`` and ``dangerous_rate`` into a probability vector over that
  frozen outcome space (it sums to 1 by construction);
* :func:`expected_value` is the probability-weighted mean outcome;
* :func:`tail_risk` is the CVaR of the worst ``1 - q`` probability mass -- the
  downside the mean hides;
* :func:`utility` is ``EV - risk_aversion * downside`` where
  ``downside = max(0, -tail_risk)``, and any non-zero ``dangerous_rate``
  collapses the score to a hard minimum.

Why the downside magnitude rather than ``abs(tail_risk)``: when an arm is so
fast and safe that its worst ``1 - q`` mass still sits on the *good* outcome,
CVaR is positive -- it describes an upside, not a risk.  Penalising that would
make a 100 ms arm score worse than a 200 ms arm purely for being fast, so the
risk term only charges for genuine downside.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Ordered from best to worst; the index order is the frozen contract.
OUTCOME_LABELS: tuple[str, ...] = ("good", "degraded", "bad")
OUTCOME_VALUES: tuple[float, ...] = (1.0, 0.0, -1.0)

#: Latency beyond this saturates: p50/p95 at the budget are treated as fully slow.
DEFAULT_LATENCY_BUDGET_MS = 4000.0

#: Share of the non-dangerous failure mass routed to the ``bad`` outcome; the
#: remainder is ``degraded`` so a plain failure is not scored like a disaster.
FAILURE_TO_BAD_SHARE = 0.5

#: Share of the latency-slow mass routed to ``bad``; the rest is ``degraded``.
#: Without a bad share a slow-but-successful arm would only lose good mass to the
#: neutral outcome, which can make it out-score a much faster arm.
SLOW_TO_BAD_SHARE = 0.5

_EPS = 1e-12


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


@dataclass(frozen=True)
class UtilityConfig:
    latency_budget_ms: float = 4000.0
    risk_aversion: float = 0.5
    dangerous_penalty: float = 1.0
    tail_quantile: float = 0.9

    def to_dict(self) -> dict[str, float]:
        return {
            "latency_budget_ms": float(self.latency_budget_ms),
            "risk_aversion": float(self.risk_aversion),
            "dangerous_penalty": float(self.dangerous_penalty),
            "tail_quantile": float(self.tail_quantile),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UtilityConfig":
        return cls(
            latency_budget_ms=float(data.get("latency_budget_ms", 4000.0)),
            risk_aversion=float(data.get("risk_aversion", 0.5)),
            dangerous_penalty=float(data.get("dangerous_penalty", 1.0)),
            tail_quantile=float(data.get("tail_quantile", 0.9)),
        )


def _distribution(row: Any, latency_budget_ms: float) -> list[float]:
    """Probability vector ``[p_good, p_degraded, p_bad]`` (documented above).

    Construction:

    * ``p_good = (1 - dangerous_rate) * success * (1 - slow_share)`` --
      a safe success whose latency is not past the budget.
    * ``slow_share = clamp(0.5 * p50/budget + 0.5 * p95/budget, 0, 1)`` --
      latency converts some good mass into the tail: half of the slow mass
      becomes ``degraded`` and half becomes ``bad``, so a slow-but-successful
      arm cannot out-score a much faster one.
    * ``p_bad = dangerous_rate + (1 - dangerous_rate) * (1 - success) * 0.5
      + (1 - dangerous_rate) * success * slow_share * 0.5`` --
      dangerous selections are always bad; half of the remaining failure mass is
      bad, half is degraded.
    * ``p_degraded`` is whatever mass is left, so the vector always sums to 1.
    """
    budget = float(latency_budget_ms) if latency_budget_ms and latency_budget_ms > 0 else DEFAULT_LATENCY_BUDGET_MS
    success = _clamp(row.success)
    dangerous = _clamp(row.dangerous_rate)
    p50 = max(0.0, float(row.latency_p50_ms))
    p95 = max(p50, float(row.latency_p95_ms))

    lat_pen = _clamp(p50 / budget)
    tail_pen = _clamp(p95 / budget)
    slow_share = _clamp(0.5 * lat_pen + 0.5 * tail_pen)

    p_good = (1.0 - dangerous) * success * (1.0 - slow_share)
    p_bad = (
        dangerous
        + (1.0 - dangerous) * (1.0 - success) * FAILURE_TO_BAD_SHARE
        + (1.0 - dangerous) * success * slow_share * SLOW_TO_BAD_SHARE
    )
    p_degraded = max(0.0, 1.0 - p_good - p_bad)
    # Renormalise away float dust so callers can rely on an exact sum of 1.
    total = p_good + p_degraded + p_bad
    if total <= _EPS:
        return [0.0, 0.0, 1.0]
    return [p_good / total, p_degraded / total, p_bad / total]


def value_distribution(row: Any) -> list[float]:
    """Probability mass over ``(good, degraded, bad)`` for one teacher row."""
    return _distribution(row, DEFAULT_LATENCY_BUDGET_MS)


def _check_distribution(dist: list[float]) -> list[float]:
    if len(dist) != len(OUTCOME_VALUES):
        raise ValueError(
            f"expected {len(OUTCOME_VALUES)} outcome probabilities, got {len(dist)}"
        )
    values = [float(p) for p in dist]
    for p in values:
        if not (0.0 - _EPS <= p <= 1.0 + _EPS):
            raise ValueError(f"probability out of [0, 1]: {p}")
    if abs(sum(values) - 1.0) > 1e-9:
        raise ValueError(f"distribution does not sum to 1: {sum(values)}")
    return values


def expected_value(dist: list[float]) -> float:
    """Probability-weighted mean over the frozen outcome values."""
    probs = _check_distribution(dist)
    return float(sum(p * v for p, v in zip(probs, OUTCOME_VALUES)))


def tail_risk(dist: list[float], q: float) -> float:
    """CVaR: mean outcome over the worst ``1 - q`` probability mass.

    Always ``<= expected_value`` for a distribution over :data:`OUTCOME_VALUES`,
    because it averages a subset of the lower tail.
    """
    if not 0.0 < float(q) < 1.0:
        raise ValueError(f"tail quantile must be in (0, 1), got {q}")
    probs = _check_distribution(dist)
    target = 1.0 - float(q)
    pairs = sorted(zip(OUTCOME_VALUES, probs), key=lambda pair: pair[0])
    remaining = target
    accumulated = 0.0
    for value, prob in pairs:
        if remaining <= _EPS:
            break
        take = min(prob, remaining)
        accumulated += take * value
        remaining -= take
    return float(accumulated / target)


def downside_risk(dist: list[float], q: float) -> float:
    """Magnitude of the tail loss: ``max(0, -tail_risk(dist, q))``.

    Zero when the worst ``1 - q`` mass is still an upside, so a nearly risk-free
    arm is never charged a risk penalty.
    """
    return max(0.0, -tail_risk(dist, q))


def utility(row: Any, config: UtilityConfig) -> float:
    """Scalar utility: ``EV - risk_aversion * downside``, hard-min on danger.

    A row with ``dangerous_rate > 0`` short-circuits to
    ``-abs(config.dangerous_penalty)``; complexity is never allowed to buy back
    a dangerous selection.  Deterministic and side-effect free.
    """
    if float(row.dangerous_rate) > 0.0:
        return -abs(float(config.dangerous_penalty))
    dist = _distribution(row, config.latency_budget_ms)
    ev = expected_value(dist)
    risk = downside_risk(dist, config.tail_quantile)
    return float(ev - float(config.risk_aversion) * risk)


def explain_utility(row: Any, config: UtilityConfig) -> dict[str, Any]:
    """Human-readable breakdown used by ``utility.json`` and the report."""
    dist = _distribution(row, config.latency_budget_ms)
    ev = expected_value(dist)
    cvar = tail_risk(dist, config.tail_quantile)
    risk = max(0.0, -cvar)
    hard = float(row.dangerous_rate) > 0.0
    return {
        "state_key": getattr(row, "state_key", ""),
        "arm": getattr(row, "arm", ""),
        "probabilities": {label: p for label, p in zip(OUTCOME_LABELS, dist)},
        "outcome_values": list(OUTCOME_VALUES),
        "expected_value": ev,
        "tail_risk": cvar,
        "downside_risk": risk,
        "risk_aversion": float(config.risk_aversion),
        "hard_minimum": hard,
        "utility": -abs(float(config.dangerous_penalty))
        if hard
        else ev - float(config.risk_aversion) * risk,
    }


__all__ = [
    "DEFAULT_LATENCY_BUDGET_MS",
    "FAILURE_TO_BAD_SHARE",
    "OUTCOME_LABELS",
    "OUTCOME_VALUES",
    "SLOW_TO_BAD_SHARE",
    "UtilityConfig",
    "downside_risk",
    "expected_value",
    "explain_utility",
    "tail_risk",
    "utility",
    "value_distribution",
]
