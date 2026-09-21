"""State features, per-arm Q(s, a) and the cost/quality Pareto frontier.

Q is fit **per arm** as a ridge linear model over a small, explicit feature
vector.  We only use signals that exist on disk:

* ``bias`` -- the intercept, so no separate offset term is needed;
* ``log1p(candidate_action_count)`` -- how wide the decision was;
* ``exposed_dangerous`` -- a dangerous action survived into the legal set;
* ``expect_abstain`` -- the state is supposed to refuse.

Family and state-text length live in the fixture file and are deliberately *not*
used in v0: keeping the vector frozen makes the ridge coefficients comparable
across runs and keeps the router small.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .schema import QVALUES_SCHEMA
from .teacher import TeacherTable
from .utility import UtilityConfig, utility

#: Frozen v0 feature vector.  Order is a contract; append, never reorder.
FEATURE_NAMES: tuple[str, ...] = (
    "bias",
    "log1p_candidate_action_count",
    "exposed_dangerous",
    "expect_abstain",
)


@dataclass(frozen=True)
class StateFeatures:
    """Maps a state key to its numeric feature vector."""

    signals: dict[str, dict[str, float]] = field(default_factory=dict)
    feature_names: tuple[str, ...] = FEATURE_NAMES

    @classmethod
    def from_teacher_table(cls, table: TeacherTable) -> "StateFeatures":
        full: dict[str, dict[str, float]] = {key: dict(value) for key, value in table.signals.items()}
        for state_key in table.states():
            full.setdefault(
                state_key,
                {"candidate_action_count": 0.0, "exposed_dangerous": 0.0, "expect_abstain": 0.0},
            )
        return cls(signals=full)

    def vector(self, state_key: str) -> list[float]:
        signal = self.signals.get(state_key, {})
        count = float(signal.get("candidate_action_count", 0.0) or 0.0)
        return [
            1.0,
            float(np.log1p(max(0.0, count))),
            1.0 if signal.get("exposed_dangerous", 0.0) else 0.0,
            1.0 if signal.get("expect_abstain", 0.0) else 0.0,
        ]

    def matrix(self, state_keys: Sequence[str]) -> np.ndarray:
        if not state_keys:
            return np.zeros((0, len(self.feature_names)), dtype=float)
        return np.asarray([self.vector(k) for k in state_keys], dtype=float)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "vectors": {k: self.vector(k) for k in sorted(self.signals)},
        }


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    """Closed-form ridge: ``(X^T X + λI)^-1 X^T y`` (pseudo-inverse fallback)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.size == 0:
        return np.zeros(X.shape[1] if X.ndim == 2 else 0, dtype=float)
    n_features = X.shape[1]
    gram = X.T @ X + float(lam) * np.eye(n_features)
    rhs = X.T @ y
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:  # pragma: no cover - λ>0 keeps this rare
        return np.linalg.pinv(gram) @ rhs


#: Small enough to be nearly ordinary least squares, large enough to keep the
#: closed-form solve well conditioned on the handful of real states.
DEFAULT_RIDGE_LAMBDA = 1e-3


@dataclass
class QModel:
    """One ridge model per arm over :data:`FEATURE_NAMES`."""

    coefficients: dict[str, list[float]]
    arms: list[str]
    features: StateFeatures = field(default_factory=StateFeatures)
    utility_config: UtilityConfig = field(default_factory=UtilityConfig)
    ridge_lambda: float = 1e-3
    schema: str = QVALUES_SCHEMA

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.features.feature_names

    @property
    def n_params(self) -> int:
        return len(self.arms) * len(self.feature_names)

    def _vector(self, state: str | Sequence[float]) -> np.ndarray:
        if isinstance(state, str):
            return np.asarray(self.features.vector(state), dtype=float)
        return np.asarray(state, dtype=float)

    def predict(self, state: str | Sequence[float], arm: str) -> float:
        coef = self.coefficients.get(arm)
        if coef is None:
            raise KeyError(f"unknown arm: {arm}")
        return float(np.dot(np.asarray(coef, dtype=float), self._vector(state)))

    def q_table(self, state: str | Sequence[float]) -> dict[str, float]:
        return {arm: self.predict(state, arm) for arm in self.arms}

    def best_arm(self, state: str | Sequence[float]) -> str:
        table = self.q_table(state)
        return min(sorted(table), key=lambda arm: (-table[arm], arm))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "feature_names": list(self.feature_names),
            "arms": list(self.arms),
            "coefficients": {arm: list(self.coefficients[arm]) for arm in sorted(self.coefficients)},
            "vectors": {k: self.features.vector(k) for k in sorted(self.features.signals)},
            "utility_config": self.utility_config.to_dict(),
            "ridge_lambda": float(self.ridge_lambda),
            "n_params": self.n_params,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QModel":
        feature_names = tuple(data.get("feature_names", FEATURE_NAMES))
        vectors = data.get("vectors", {})
        # Rebuild signals from stored vectors so predict-by-key round-trips.
        signals: dict[str, dict[str, float]] = {}
        for key, vector in vectors.items():
            values = [float(v) for v in vector]
            count = float(np.expm1(max(0.0, values[1]))) if len(values) > 1 else 0.0
            signals[key] = {
                "candidate_action_count": count,
                "exposed_dangerous": values[2] if len(values) > 2 else 0.0,
                "expect_abstain": values[3] if len(values) > 3 else 0.0,
            }
        return cls(
            coefficients={
                arm: [float(v) for v in coef]
                for arm, coef in data.get("coefficients", {}).items()
            },
            arms=[str(a) for a in data.get("arms", [])],
            features=StateFeatures(signals=signals, feature_names=feature_names),
            utility_config=UtilityConfig.from_dict(data.get("utility_config", {})),
            ridge_lambda=float(data.get("ridge_lambda", 1e-3)),
            schema=str(data.get("schema", QVALUES_SCHEMA)),
        )


def fit_q(table: TeacherTable, config: UtilityConfig) -> QModel:
    """Fit one ridge regression per arm with target ``utility(row, config)``.

    States with no observation for an arm simply do not contribute to that
    arm's fit.  An arm with no observations gets zero coefficients.
    """
    features = StateFeatures.from_teacher_table(table)
    arms = table.arms()
    coefficients: dict[str, list[float]] = {}
    for arm in arms:
        rows = [r for r in table.sorted_rows() if r.arm == arm]
        if not rows:
            coefficients[arm] = [0.0] * len(features.feature_names)
            continue
        X = features.matrix([r.state_key for r in rows])
        y = np.asarray([utility(r, config) for r in rows], dtype=float)
        coefficients[arm] = [float(v) for v in ridge_fit(X, y, lam=DEFAULT_RIDGE_LAMBDA)]
    return QModel(
        coefficients=coefficients,
        arms=arms,
        features=features,
        utility_config=config,
        ridge_lambda=DEFAULT_RIDGE_LAMBDA,
    )


def pareto_arms(table: TeacherTable, config: UtilityConfig) -> dict[str, list[str]]:
    """Per state, the arms on the cost/quality Pareto frontier.

    Cost is ``latency_p50_ms`` (minimise); quality is ``utility(row, config)``
    (maximise).  Arm *a* is dominated when some other arm *b* has
    ``utility_b >= utility_a`` and ``latency_b <= latency_a`` with at least one
    strict.  Returns a deterministic, sorted mapping.
    """
    out: dict[str, list[str]] = {}
    for state_key in table.states():
        points = [
            (row.arm, utility(row, config), float(row.latency_p50_ms))
            for row in table.rows_for_state(state_key)
        ]
        frontier: list[str] = []
        for arm, util, latency in points:
            dominated = False
            for other_arm, other_util, other_latency in points:
                if other_arm == arm:
                    continue
                ge = other_util >= util - 1e-12 and other_latency <= latency + 1e-12
                gt = other_util > util + 1e-12 or other_latency < latency - 1e-12
                if ge and gt:
                    dominated = True
                    break
            if not dominated:
                frontier.append(arm)
        scored = {arm: utility for arm, utility, _ in points}
        latency_of = {arm: latency for arm, _, latency in points}
        frontier.sort(key=lambda arm: (-scored[arm], latency_of[arm], arm))
        out[state_key] = frontier
    return out


__all__ = [
    "DEFAULT_RIDGE_LAMBDA",
    "FEATURE_NAMES",
    "QModel",
    "StateFeatures",
    "fit_q",
    "pareto_arms",
    "ridge_fit",
]
