"""Numpy students: ridge readouts on frozen random features (P0, no torch)."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .schema import ExperimentGenome
from .task import ACTIONS, Episode, N_ACTIONS, apply_strobe, teacher_action


def ridge_fit(X: np.ndarray, y: np.ndarray, n_out: int, l2: float) -> np.ndarray:
    Y = np.eye(n_out, dtype=np.float64)[y]
    A = X.T @ X + l2 * np.eye(X.shape[1])
    B = X.T @ Y
    return np.linalg.solve(A, B)


def softmax_predict(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    logits = X @ W
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def _spectral_radius_scale(W: np.ndarray, radius: float) -> np.ndarray:
    eig = np.linalg.eigvals(W)
    r = float(np.max(np.abs(eig)))
    if r < 1e-9:
        return W
    return W * (radius / r)


def _sparse_recurrent(hidden: int, sparsity: float, radius: float, rng: np.random.Generator) -> np.ndarray:
    W = rng.normal(0.0, 1.0, (hidden, hidden))
    mask = rng.random((hidden, hidden)) < sparsity
    np.fill_diagonal(mask, False)
    W = W * mask
    return _spectral_radius_scale(W, radius)


@dataclass
class FittedStudent:
    family: str
    n_params: int
    predict_fn: object
    extras: dict


def _gru_last_hidden(frames: np.ndarray, Wz, Uz, bz, Wr, Ur, br, Wh, Uh, bh) -> np.ndarray:
    """frames: (T, F) -> hidden vector."""
    hidden = bz.shape[0]
    h = np.zeros(hidden, dtype=np.float64)

    def sig(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    for x in frames:
        z = sig(Wz @ x + Uz @ h + bz)
        r = sig(Wr @ x + Ur @ h + br)
        h_hat = np.tanh(Wh @ x + Uh @ (r * h) + bh)
        h = (1.0 - z) * h + z * h_hat
    return h


def extract_features(genome: ExperimentGenome, episodes: list[Episode], rng: np.random.Generator, *, weights=None) -> tuple[np.ndarray, np.ndarray, dict]:
    drop = genome.curriculum.strobe_drop
    eps = [apply_strobe(ep, drop, rng) for ep in episodes]
    y = np.stack([ep.labels[-1] for ep in eps])
    family = genome.architecture.family
    hidden = genome.architecture.hidden
    F = eps[0].frames.shape[-1]
    T = eps[0].frames.shape[0]

    if family == "rule":
        return np.zeros((len(eps), 1)), y, {"rule": True}

    if family == "direct_input":
        X = np.stack([ep.frames[-1] for ep in eps])
        return X, y, {"n_in": F}

    if family == "mlp":
        W1 = weights["W1"] if weights else rng.normal(0, 1 / np.sqrt(T * F), (T * F, hidden))
        b1 = weights["b1"] if weights else rng.normal(0, 0.01, hidden)
        Xflat = np.stack([ep.frames.reshape(-1) for ep in eps])
        H = np.maximum(0.0, Xflat @ W1 + b1)
        return H, y, {"W1": W1, "b1": b1, "n_in": T * F}

    if family == "gru":
        scale = 1 / np.sqrt(F)
        hs = 1 / np.sqrt(hidden)
        pack = weights or {
            "Wz": rng.normal(0, scale, (hidden, F)),
            "Uz": rng.normal(0, hs, (hidden, hidden)),
            "bz": np.zeros(hidden),
            "Wr": rng.normal(0, scale, (hidden, F)),
            "Ur": rng.normal(0, hs, (hidden, hidden)),
            "br": np.zeros(hidden),
            "Wh": rng.normal(0, scale, (hidden, F)),
            "Uh": rng.normal(0, hs, (hidden, hidden)),
            "bh": np.zeros(hidden),
        }
        H = np.stack(
            [
                _gru_last_hidden(
                    ep.frames,
                    pack["Wz"],
                    pack["Uz"],
                    pack["bz"],
                    pack["Wr"],
                    pack["Ur"],
                    pack["br"],
                    pack["Wh"],
                    pack["Uh"],
                    pack["bh"],
                )
                for ep in eps
            ]
        )
        return H, y, pack

    if family in {"fixed_reservoir", "rewired_reservoir", "fly_connectome"}:
        Win = weights["Win"] if weights else rng.normal(0, 1 / np.sqrt(F), (hidden, F))
        Wrec = weights["Wrec"] if weights else _sparse_recurrent(
            hidden, genome.architecture.sparsity, genome.architecture.spectral_radius, rng
        )
        H = []
        for ep in eps:
            h = np.zeros(hidden)
            for x in ep.frames:
                h = np.tanh(Wrec @ h + Win @ x)
            H.append(h)
        return np.stack(H), y, {"Win": Win, "Wrec": Wrec}

    raise ValueError(f"no feature map for {family}")


def fit_student(genome: ExperimentGenome, train: list[Episode]) -> FittedStudent:
    rng = np.random.default_rng(genome.training.seed)
    if genome.architecture.family == "rule":
        def predict(eps: list[Episode]) -> np.ndarray:
            out = []
            for ep in eps:
                ep2 = apply_strobe(ep, genome.curriculum.strobe_drop, rng)
                cue_seen = bool(ep2.frames[:, 9].max() > 0.5)
                y = teacher_action(ep2.frames[-1])
                if cue_seen:
                    y = ACTIONS.index("escalate")
                out.append(y)
            return np.asarray(out, dtype=np.int64)

        return FittedStudent("rule", n_params=0, predict_fn=predict, extras={})

    X, y, extras = extract_features(genome, train, rng)
    W = ridge_fit(X, y, N_ACTIONS, genome.training.l2)
    n_params = int(W.size) + sum(int(np.asarray(v).size) for k, v in extras.items() if k != "n_in" and k != "rule" and isinstance(v, np.ndarray))

    def predict(eps: list[Episode]) -> np.ndarray:
        rng2 = np.random.default_rng(genome.training.seed + 999)
        Xt, _, _ = extract_features(genome, eps, rng2, weights=extras)
        probs = softmax_predict(Xt, W)
        return probs.argmax(axis=1)

    extras = dict(extras)
    extras["W"] = W
    return FittedStudent(genome.architecture.family, n_params=n_params, predict_fn=predict, extras=extras)
