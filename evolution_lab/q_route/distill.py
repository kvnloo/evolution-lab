"""Distil the gate's per-region decision into a small interpretable router.

The router's target is **the Pareto-optimal / gate-winning arm**, never a
teacher model's answer.  There is no teacher answer to copy: the supervision is
the frozen gate's own owner per state (:func:`evolution_lab.q_route.gate.compile_regions`),
so the router learns "which mechanism earned this region", not "what would a
bigger model have said".

The model is a multinomial logistic regression over the frozen feature vector,
trained with deterministic full-batch gradient descent.  It is small
(``n_arms * n_features`` parameters), serialises to plain JSON and reproduces
the same predictions for the same weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .gate import GateConfig, compile_regions
from .qfunc import QModel, StateFeatures
from .schema import ROUTER_SCHEMA
from .teacher import TeacherTable
from .utility import UtilityConfig, utility

#: Frozen training schedule; deterministic and small on purpose.
DEFAULT_SEED = 42
DEFAULT_EPOCHS = 400
DEFAULT_LR = 0.5
DEFAULT_L2 = 1e-3


def gate_targets(
    table: TeacherTable,
    gate_config: GateConfig | None = None,
    utility_config: UtilityConfig | None = None,
) -> dict[str, str]:
    """Gate-winning arm per state (the router's supervision)."""
    report = compile_regions(table, gate_config or GateConfig(), utility_config)
    return {region.state_key: region.owner for region in report.regions}


@dataclass
class RouterModel:
    """Small multinomial-logistic router over the frozen feature vector."""

    arms: list[str]
    feature_names: tuple[str, ...]
    weights: list[list[float]]
    seed: int = DEFAULT_SEED
    epochs: int = DEFAULT_EPOCHS
    lr: float = DEFAULT_LR
    l2: float = DEFAULT_L2
    schema: str = ROUTER_SCHEMA

    @property
    def n_params(self) -> int:
        return int(len(self.arms) * len(self.feature_names))

    def _logits(self, features: Sequence[float]) -> np.ndarray:
        x = np.asarray(features, dtype=float)
        w = np.asarray(self.weights, dtype=float)
        if w.size == 0:
            return np.zeros(len(self.arms), dtype=float)
        return w @ x

    def predict(self, features: Sequence[float]) -> str:
        if not self.arms:
            raise ValueError("router has no arms")
        logits = self._logits(features)
        best = min(range(len(self.arms)), key=lambda i: (-float(logits[i]), self.arms[i]))
        return self.arms[best]

    def predict_proba(self, features: Sequence[float]) -> dict[str, float]:
        logits = self._logits(features)
        shifted = logits - float(np.max(logits)) if logits.size else logits
        exp = np.exp(shifted)
        total = float(exp.sum()) or 1.0
        return {arm: float(exp[i] / total) for i, arm in enumerate(self.arms)}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "arms": list(self.arms),
            "feature_names": list(self.feature_names),
            "weights": [[float(v) for v in row] for row in self.weights],
            "seed": int(self.seed),
            "epochs": int(self.epochs),
            "lr": float(self.lr),
            "l2": float(self.l2),
            "n_params": self.n_params,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RouterModel":
        return cls(
            arms=[str(a) for a in data.get("arms", [])],
            feature_names=tuple(data.get("feature_names", ())),
            weights=[[float(v) for v in row] for row in data.get("weights", [])],
            seed=int(data.get("seed", DEFAULT_SEED)),
            epochs=int(data.get("epochs", DEFAULT_EPOCHS)),
            lr=float(data.get("lr", DEFAULT_LR)),
            l2=float(data.get("l2", DEFAULT_L2)),
            schema=str(data.get("schema", ROUTER_SCHEMA)),
        )


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    denom = np.sum(exp, axis=1, keepdims=True)
    denom[denom == 0.0] = 1.0
    return exp / denom


def fit_router(
    qmodel: QModel,
    features: StateFeatures,
    table: TeacherTable,
    *,
    gate_report: Any | None = None,
    gate_config: GateConfig | None = None,
    utility_config: UtilityConfig | None = None,
    seed: int = DEFAULT_SEED,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    l2: float = DEFAULT_L2,
) -> RouterModel:
    """Fit the router to the gate-winning arm per state.

    Deterministic given ``seed`` and the inputs: initial weights are zero, the
    update is full-batch and there is no sampling.
    """
    ucfg = utility_config or qmodel.utility_config
    if gate_report is not None:
        targets = {region.state_key: region.owner for region in gate_report.regions}
    else:
        targets = gate_targets(table, gate_config or GateConfig(), ucfg)

    states = sorted(state for state in targets if table.row_for(state, targets[state]) is not None)
    arms = sorted({targets[state] for state in states})
    feature_names = features.feature_names
    if not states or not arms:
        return RouterModel(
            arms=arms,
            feature_names=feature_names,
            weights=[[0.0] * len(feature_names) for _ in arms],
            seed=seed,
            epochs=epochs,
            lr=lr,
            l2=l2,
        )

    X = features.matrix(states)
    y = np.asarray([arms.index(targets[state]) for state in states], dtype=int)
    onehot = np.zeros((len(states), len(arms)), dtype=float)
    onehot[np.arange(len(states)), y] = 1.0

    W = np.zeros((len(arms), len(feature_names)), dtype=float)
    n = float(len(states))
    for _ in range(int(epochs)):
        probs = _softmax(X @ W.T)
        grad = (probs - onehot).T @ X / n
        grad += float(l2) * W
        W -= float(lr) * grad

    return RouterModel(
        arms=arms,
        feature_names=feature_names,
        weights=[[float(v) for v in row] for row in W],
        seed=seed,
        epochs=epochs,
        lr=lr,
        l2=l2,
    )


def non_inferiority(
    router: RouterModel,
    qmodel: QModel,
    table: TeacherTable,
    cfg: GateConfig | None = None,
    *,
    utility_config: UtilityConfig | None = None,
    gate_report: Any | None = None,
) -> dict[str, Any]:
    """Router fidelity to the gate owner plus utility delta versus that owner.

    ``utility_delta = utility(router_arm) - utility(gate_arm)``; negative means
    the router is worse.  ``regret`` is the same number flipped, clipped at 0.
    A router prediction with no measured row for that state is counted as
    infeasible and excluded from the utility averages (but still counts against
    fidelity).
    """
    ucfg = utility_config or qmodel.utility_config
    gate_config = cfg or GateConfig()
    if gate_report is not None:
        targets = {region.state_key: region.owner for region in gate_report.regions}
    else:
        targets = gate_targets(table, gate_config, ucfg)

    per_state: list[dict[str, Any]] = []
    matches = 0
    deltas: list[float] = []
    regrets: list[float] = []
    infeasible = 0
    for state_key in sorted(targets):
        target = targets[state_key]
        predicted = router.predict(qmodel.features.vector(state_key))
        match = predicted == target
        matches += int(match)
        target_row = table.row_for(state_key, target)
        pred_row = table.row_for(state_key, predicted)
        if target_row is None:
            delta = None
            regret = None
        elif pred_row is None:
            infeasible += 1
            delta = None
            regret = None
        else:
            delta = utility(pred_row, ucfg) - utility(target_row, ucfg)
            regret = max(0.0, -delta)
            deltas.append(delta)
            regrets.append(regret)
        per_state.append(
            {
                "state_key": state_key,
                "gate_arm": target,
                "router_arm": predicted,
                "match": match,
                "utility_delta": delta,
                "regret": regret,
                "feasible": pred_row is not None,
            }
        )

    n = len(targets)
    # Any deterministic function of the frozen features can only assign one arm
    # per distinct feature vector, so the majority owner per vector is the
    # feature-resolution ceiling.  Reporting it separates "router is bad" from
    # "v0 features cannot tell these states apart".
    buckets: dict[tuple[float, ...], dict[str, int]] = {}
    for state_key in sorted(targets):
        key = tuple(qmodel.features.vector(state_key))
        counts = buckets.setdefault(key, {})
        counts[targets[state_key]] = counts.get(targets[state_key], 0) + 1
    ceiling_hits = sum(max(counts.values()) for counts in buckets.values())
    return {
        "schema": ROUTER_SCHEMA,
        "n_states": n,
        "n_matched": matches,
        "fidelity": float(matches / n) if n else 0.0,
        "n_feature_buckets": len(buckets),
        "feature_bucket_ceiling": float(ceiling_hits / n) if n else 0.0,
        "n_infeasible_predictions": infeasible,
        "mean_utility_delta": float(np.mean(deltas)) if deltas else 0.0,
        "mean_regret": float(np.mean(regrets)) if regrets else 0.0,
        "max_regret": float(max(regrets)) if regrets else 0.0,
        "n_scored": len(deltas),
        "gate_config": gate_config.to_dict(),
        "utility_config": ucfg.to_dict(),
        "n_router_params": router.n_params,
        "per_state": per_state,
    }


__all__ = [
    "DEFAULT_EPOCHS",
    "DEFAULT_L2",
    "DEFAULT_LR",
    "DEFAULT_SEED",
    "RouterModel",
    "fit_router",
    "gate_targets",
    "non_inferiority",
]
