"""Numpy students: ridge readouts on frozen random features (P0, no torch)."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .schema import ExperimentGenome
from .task import ACTIONS, Episode, N_ACTIONS, apply_strobe, episode_from_prefix, teacher_action


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


def _pn_features(episodes: list[Episode]) -> np.ndarray:
    """Engineered PN drive from Hermes events.

    Flattened history + last-step fields + max-over-time (so a delayed cue is
    a PN, not a hidden recurrence). This is not the compound eye and not MaleCNS.
    """
    rows = []
    for ep in episodes:
        flat = ep.frames.reshape(-1)
        last = ep.frames[-1]
        pooled = ep.frames.max(axis=0)
        cue_seen = np.array([ep.frames[:, 9].max()], dtype=np.float64)
        rows.append(np.concatenate([flat, last, pooled, cue_seen]))
    return np.stack(rows)


def _kc_codes(X: np.ndarray, W_pn_kc: np.ndarray, k_winners: int) -> np.ndarray:
    """k-winner Kenyon-cell codes. PN→KC is frozen; this is not a learned encoder."""
    drive = np.maximum(X @ W_pn_kc, 0.0)
    n, n_kc = drive.shape
    k = max(1, min(int(k_winners), n_kc))
    idx = np.argpartition(drive, -k, axis=1)[:, -k:]
    mask = np.zeros_like(drive)
    mask[np.arange(n)[:, None], idx] = 1.0
    kc = drive * mask
    norms = np.linalg.norm(kc, axis=1, keepdims=True)
    return kc / np.maximum(norms, 1e-8)


def _sparse_pn_kc(n_pn: int, n_kc: int, rng: np.random.Generator) -> np.ndarray:
    """Frozen random PN→KC. Fan-in is sparse; weights are not updated."""
    fan_in = max(8, min(n_pn, max(12, n_pn // 6)))
    W = np.zeros((n_pn, n_kc), dtype=np.float64)
    scale = 1.0 / np.sqrt(fan_in)
    for j in range(n_kc):
        idx = rng.choice(n_pn, size=fan_in, replace=False)
        W[idx, j] = rng.normal(0.0, scale, size=fan_in)
    return W


def _local_plasticity_step_samples(
    episodes: list[Episode],
    *,
    history: int,
) -> tuple[list[Episode], np.ndarray]:
    """Expand teacher episodes into per-step rows aligned with closed-loop predict."""
    step_eps: list[Episode] = []
    labels: list[int] = []
    for ep in episodes:
        T = ep.frames.shape[0]
        for t in range(T):
            step_eps.append(
                episode_from_prefix(ep.frames[: t + 1], history=history, env=ep.env)
            )
            labels.append(int(ep.labels[t]))
    return step_eps, np.asarray(labels, dtype=np.int64)


def _plasticity_train(
    step_eps: list[Episode],
    y: np.ndarray,
    *,
    W_pn_kc: np.ndarray,
    k_winners: int,
    W: np.ndarray,
    rng: np.random.Generator,
    epochs: int = 20,
    lr: float = 0.35,
) -> np.ndarray:
    X = _pn_features(step_eps)
    H = _kc_codes(X, W_pn_kc, k_winners)
    n = X.shape[0]
    for _epoch in range(epochs):
        for i in rng.permutation(n):
            h = H[i]
            scores = h @ W
            s = scores - scores.max()
            pred = np.exp(np.clip(s, -20, 20))
            pred = pred / pred.sum()
            target = np.zeros(N_ACTIONS, dtype=np.float64)
            target[int(y[i])] = 1.0
            err = target - pred
            W += lr * np.outer(h, err)
            W -= 0.02 * lr * np.outer(h, pred)
        lr *= 0.92
    return W


def _fit_local_plasticity(
    genome: ExperimentGenome,
    train: list[Episode],
    rng: np.random.Generator,
    *,
    init_extras: dict | None = None,
) -> FittedStudent:
    """Mushroom-body analogue for Hermes recovery.

    PN drive is an engineered encoder of Hermes history (flatten + last-step +
    max-over-time), not the compound eye and not MaleCNS. PN→KC stays frozen.
    Only KC→MBON is plastic (local Hebbian / class-conditional LTD + local delta).
    Trains on **per-step** prefixes so closed-loop matches confirm accuracy.
    """
    drop = genome.curriculum.strobe_drop
    eps = [apply_strobe(ep, drop, rng) for ep in train]
    history = int(genome.architecture.history)
    step_eps, y = _local_plasticity_step_samples(eps, history=history)
    n_kc = max(32, int(genome.architecture.hidden))
    if init_extras and "W_pn_kc" in init_extras and "W_kc_mbon" in init_extras:
        W_pn_kc = np.asarray(init_extras["W_pn_kc"], dtype=np.float64)
        k_winners = int(init_extras["k_winners"])
        W = np.asarray(init_extras["W_kc_mbon"], dtype=np.float64).copy()
        n_kc = W.shape[0]
    else:
        X_probe = _pn_features(step_eps[:1])
        W_pn_kc = _sparse_pn_kc(X_probe.shape[1], n_kc, rng)
        k_winners = max(5, int(round(0.10 * n_kc)))
        W = np.zeros((n_kc, N_ACTIONS), dtype=np.float64)
    W = _plasticity_train(step_eps, y, W_pn_kc=W_pn_kc, k_winners=k_winners, W=W, rng=rng)

    def predict(episodes: list[Episode]) -> np.ndarray:
        rng2 = np.random.default_rng(genome.training.seed + 999)
        e2 = [apply_strobe(ep, genome.curriculum.strobe_drop, rng2) for ep in episodes]
        Xp = _pn_features(e2)
        Ht = _kc_codes(Xp, W_pn_kc, k_winners)
        return (Ht @ W).argmax(axis=1)

    extras = {
        "W_kc_mbon": W,
        "W_pn_kc": W_pn_kc,
        "k_winners": k_winners,
        "encoder": "flatten_last_and_maxpool_hermes_history_not_compound_eye",
        "plastic": "kc_to_mbon_only",
        "supervision": "per_step_prefix",
    }
    return FittedStudent("local_plasticity", n_params=int(W.size), predict_fn=predict, extras=extras)


def fit_student(
    genome: ExperimentGenome,
    train: list[Episode],
    *,
    init_extras: dict | None = None,
) -> FittedStudent:
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

    if genome.architecture.family == "local_plasticity":
        return _fit_local_plasticity(genome, train, rng, init_extras=init_extras)

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
