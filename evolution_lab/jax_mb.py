"""local_jax mushroom body: same algorithm as numpy local_plasticity, jnp.float32.

Stage 1: parity on locked splits (identical update, given the same permutation).
Stage 2: vmap over population. Frozen judge stays CPU.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import lax
except ImportError as exc:  # pragma: no cover
    raise ImportError("local_jax backend requires jax") from exc

N_ACTIONS = 5


def kc_codes(X: jnp.ndarray, W_pn_kc: jnp.ndarray, k_winners: int) -> jnp.ndarray:
    drive = jnp.maximum(X @ W_pn_kc, 0.0)
    k = int(k_winners)
    idx = jnp.argpartition(drive, -k, axis=1)[:, -k:]
    rows = jnp.arange(drive.shape[0])[:, None]
    mask = jnp.zeros_like(drive).at[rows, idx].set(1.0)
    kc = drive * mask
    norms = jnp.linalg.norm(kc, axis=1, keepdims=True)
    return kc / jnp.maximum(norms, 1e-8)


def _sample_update(W: jnp.ndarray, h: jnp.ndarray, yi: jnp.ndarray, lr: jnp.ndarray, n_actions: int) -> jnp.ndarray:
    scores = h @ W
    s = scores - jnp.max(scores)
    pred = jnp.exp(jnp.clip(s, -20.0, 20.0))
    pred = pred / jnp.sum(pred)
    target = jax.nn.one_hot(yi, n_actions, dtype=W.dtype)
    W = W + lr * jnp.outer(h, target - pred)
    W = W - 0.02 * lr * jnp.outer(h, pred)
    return W


def train_mbon_with_perms(
    H: jnp.ndarray,
    y: jnp.ndarray,
    perms: jnp.ndarray,
    lr0: float,
    n_actions: int = N_ACTIONS,
) -> jnp.ndarray:
    """perms: [epochs, n] int permutation rows — same order as numpy."""
    n_kc = H.shape[1]
    W = jnp.zeros((n_kc, n_actions), dtype=H.dtype)
    lr = jnp.asarray(lr0, dtype=H.dtype)

    def epoch(carry, perm):
        W, lr = carry

        def step(W, idx):
            W = _sample_update(W, H[idx], y[idx], lr, n_actions)
            return W, None

        W, _ = lax.scan(step, W, perm)
        return (W, lr * 0.92), None

    (W, _), _ = lax.scan(epoch, (W, lr), perms)
    return W


def numpy_perms(n: int, epochs: int, rng: np.random.Generator) -> np.ndarray:
    return np.stack([rng.permutation(n) for _ in range(epochs)]).astype(np.int32)


def fit_from_numpy(
    X: np.ndarray,
    y: np.ndarray,
    W_pn_kc: np.ndarray,
    *,
    k_winners: int,
    epochs: int = 20,
    lr: float = 0.35,
    perms: np.ndarray | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    if perms is None:
        perms = numpy_perms(X.shape[0], epochs, np.random.default_rng(seed))
    Xj = jnp.asarray(X, dtype=jnp.float32)
    yj = jnp.asarray(y, dtype=jnp.int32)
    Wp = jnp.asarray(W_pn_kc, dtype=jnp.float32)
    H = kc_codes(Xj, Wp, k_winners)
    W = train_mbon_with_perms(H, yj, jnp.asarray(perms), float(lr))
    W.block_until_ready()
    return {
        "W_pn_kc": np.asarray(jax.device_get(Wp)),
        "W_kc_mbon": np.asarray(jax.device_get(W)),
        "H": np.asarray(jax.device_get(H)),
        "k_winners": int(k_winners),
        "device": str(jax.devices()[0]),
    }


def predict_numpy(H: np.ndarray, W: np.ndarray) -> np.ndarray:
    return np.asarray(np.argmax(H @ W, axis=1))


def population_mbon(
    H: np.ndarray,
    y: np.ndarray,
    perm_bank: np.ndarray,
    lr: float,
    n_actions: int | None = None,
) -> np.ndarray:
    """perm_bank: [P, epochs, n]. Returns W [P, KC, A]."""
    if n_actions is None:
        n_actions = int(np.max(y)) + 1
    H = jnp.asarray(H, dtype=jnp.float32)
    y = jnp.asarray(y, dtype=jnp.int32)
    perms = jnp.asarray(perm_bank)
    n_a = int(n_actions)
    vm = jax.vmap(lambda p: train_mbon_with_perms(H, y, p, lr, n_a))
    Ws = vm(perms)
    Ws.block_until_ready()
    return np.asarray(jax.device_get(Ws))
