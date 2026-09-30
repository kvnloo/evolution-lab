"""CPU population evolution for mushroom-body (local_plasticity) candidates.

P0 "population evolution" on a host with no CUDA: train many local_plasticity flies at
once on the locked P0 train split, filter on **val only**, then hand the survivors to the
unchanged serial CPU judge (`bench.run_fly_bench`, which retrains through `fit_student`)
before anything is kept. Confirm, OOD and the held-out gym battery are never used for
selection.

Why it is fast: k-winner KC codes are k-sparse, so every KC->MBON update touches only the
k winning rows. The reference learner (`models._plasticity_train`) does a dense 128x5 outer
product per sample; the population trainer gathers k rows per candidate, so one step is an
O(P*k*A) array op instead of P separate O(KC*A) Python iterations.

Genes are exactly the knobs the frozen judge understands (`FlyCandidate`):
  seed (PN->KC draw + sample order), hidden (n_kc), k_winners, plasticity_lr, plasticity_epochs.
Given the same genes, the population trainer reproduces `fit_student` for local_plasticity:
same PN->KC matrix, same permutation stream, same update rule. The only float64 difference
is the summation order of the k-sparse score (sparse gather vs dense BLAS dot).

Backends:
  numpy  float64 (parity) or float32; vectorized over the population
  jax    jax-cpu, lax.scan over steps + vmap over the population; float32 (roadmap) or float64
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace
from typing import Any, Sequence

import numpy as np

from .models import (
    _apply_delayed_cue_protocol,
    _local_plasticity_step_samples,
    _pn_features,
    _sparse_pn_kc,
)
from .task import N_ACTIONS, Episode, episode_from_prefix

LTD = 0.02
DECAY = 0.92
HISTORY = 8


@dataclass(frozen=True)
class Gene:
    seed: int = 0
    hidden: int = 128
    k_winners: int = 0  # 0 -> repo default max(5, round(0.1 * n_kc))
    lr: float = 0.35
    epochs: int = 20

    @property
    def n_kc(self) -> int:
        return max(32, int(self.hidden))  # same floor as models._fit_local_plasticity

    @property
    def k(self) -> int:
        kw = int(self.k_winners)
        k = kw if kw > 0 else max(5, int(round(0.10 * self.n_kc)))
        return max(1, min(k, self.n_kc))

    @property
    def n_params(self) -> int:
        return self.n_kc * N_ACTIONS

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------------- features


def sparse_kc(X: np.ndarray, W_pn_kc: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Same codes as models._kc_codes, returned as (indices [n,k], values [n,k])."""
    drive = np.maximum(X @ W_pn_kc, 0.0)
    n, n_kc = drive.shape
    k = max(1, min(int(k), n_kc))
    idx = np.argpartition(drive, -k, axis=1)[:, -k:]
    vals = np.take_along_axis(drive, idx, axis=1)
    norms = np.linalg.norm(vals, axis=1, keepdims=True)
    return idx.astype(np.int64), vals / np.maximum(norms, 1e-8)


def prefix_episodes(episodes: Sequence[Episode]) -> tuple[list[Episode], np.ndarray]:
    eps, ys = [], []
    for ep in episodes:
        for t in range(ep.frames.shape[0]):
            eps.append(episode_from_prefix(ep.frames[: t + 1], history=HISTORY, env=ep.env))
            ys.append(int(ep.labels[t]))
    return eps, np.asarray(ys, dtype=np.int64)


@dataclass
class TrainSet:
    X: np.ndarray  # PN features of per-step prefixes [n, n_pn]
    y: np.ndarray  # [n]


def train_set(train: Sequence[Episode]) -> TrainSet:
    step_eps, y = _local_plasticity_step_samples(list(train), history=HISTORY)
    return TrainSet(_pn_features(step_eps), y)


@dataclass
class Setup:
    """Per-candidate frozen state, padded to a common shape."""

    genes: list[Gene]
    W_pn: list[np.ndarray]
    idx: np.ndarray  # [P, n, kmax] (pad slots point at private dump rows)
    val: np.ndarray  # [P, n, kmax] (pad slots are 0)
    perms: np.ndarray  # [P, Emax, n]
    lr: np.ndarray  # [P]
    epochs: np.ndarray  # [P]
    rows: int  # KCmax + kmax


def setup_population(genes: Sequence[Gene], ts: TrainSet) -> Setup:
    n, n_pn = ts.X.shape
    P = len(genes)
    kmax = max(g.k for g in genes)
    kcmax = max(g.n_kc for g in genes)
    emax = max(int(g.epochs) for g in genes)
    idx = np.empty((P, n, kmax), dtype=np.int64)
    val = np.zeros((P, n, kmax), dtype=np.float64)
    perms = np.empty((P, emax, n), dtype=np.int64)
    W_pns = []
    for p, g in enumerate(genes):
        # identical rng stream to fit_student(local_plasticity): strobe(0) draws nothing,
        # then _sparse_pn_kc, then one permutation per epoch.
        rng = np.random.default_rng(int(g.seed))
        W_pn = _sparse_pn_kc(n_pn, g.n_kc, rng)
        for e in range(int(g.epochs)):
            perms[p, e] = rng.permutation(n)
        for e in range(int(g.epochs), emax):
            perms[p, e] = perms[p, 0]  # inert: lr is masked to 0 on these epochs
        i, v = sparse_kc(ts.X, W_pn, g.k)
        idx[p, :, : g.k] = i
        val[p, :, : g.k] = v
        if g.k < kmax:
            idx[p, :, g.k :] = kcmax + np.arange(kmax - g.k)[None, :]
        W_pns.append(W_pn)
    return Setup(
        genes=list(genes),
        W_pn=W_pns,
        idx=idx,
        val=val,
        perms=perms,
        lr=np.asarray([float(g.lr) for g in genes], dtype=np.float64),
        epochs=np.asarray([int(g.epochs) for g in genes], dtype=np.int64),
        rows=kcmax + kmax,
    )


# ----------------------------------------------------------------------------- trainers


def train_numpy(st: Setup, y: np.ndarray, *, dtype=np.float64) -> np.ndarray:
    """Vectorized over the population; sequential over samples (the rule is online)."""
    P, emax, n = st.perms.shape
    kmax = st.idx.shape[2]
    W = np.zeros((P, st.rows, N_ACTIONS), dtype=dtype)
    pr = np.arange(P)
    prk = pr[:, None]
    eye = np.eye(N_ACTIONS, dtype=dtype)
    idx_all, val_all = st.idx, st.val.astype(dtype)
    lr = st.lr.astype(dtype).copy()
    for e in range(emax):
        lr_e = np.where(e < st.epochs, lr, 0.0).astype(dtype)[:, None, None]
        ltd_e = (dtype(LTD) * lr_e) if dtype is not np.float64 else LTD * lr_e
        perm_e = st.perms[:, e, :]
        for s in range(n):
            i = perm_e[:, s]
            ix = idx_all[pr, i]  # [P, k]
            v = val_all[pr, i]  # [P, k]
            Wg = W[prk, ix]  # [P, k, A]
            scores = np.einsum("pk,pka->pa", v, Wg)
            sc = scores - scores.max(axis=1, keepdims=True)
            pred = np.exp(np.clip(sc, -20, 20))
            pred = pred / pred.sum(axis=1, keepdims=True)
            err = eye[y[i]] - pred
            Wg = Wg + lr_e * (v[:, :, None] * err[:, None, :])
            Wg = Wg - ltd_e * (v[:, :, None] * pred[:, None, :])
            W[prk, ix] = Wg
        lr = lr * dtype(DECAY) if dtype is not np.float64 else lr * DECAY
    del kmax
    return W


def train_jax(st: Setup, y: np.ndarray, *, x64: bool = False, chunk: int | None = None) -> np.ndarray:
    """jax-cpu: lax.scan over (epoch, sample) and vmap over the population."""
    import jax

    if x64:
        jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax import lax

    dt = jnp.float64 if x64 else jnp.float32
    P, emax, n = st.perms.shape
    rows = st.rows

    @jax.jit
    def run(idx, val, perms, lr0, epochs, yy):
        def one(idx, val, perms, lr0, epochs):
            W0 = jnp.zeros((rows, N_ACTIONS), dtype=dt)

            def epoch(carry, inp):
                W, lr = carry
                perm, e = inp
                lr_e = jnp.where(e < epochs, lr, 0.0).astype(dt)

                def step(W, i):
                    ix = idx[i]
                    v = val[i]
                    Wg = W[ix]
                    scores = v @ Wg
                    sc = scores - jnp.max(scores)
                    pred = jnp.exp(jnp.clip(sc, -20.0, 20.0))
                    pred = pred / jnp.sum(pred)
                    err = jax.nn.one_hot(yy[i], N_ACTIONS, dtype=dt) - pred
                    Wg = Wg + lr_e * (v[:, None] * err[None, :])
                    Wg = Wg - (LTD * lr_e) * (v[:, None] * pred[None, :])
                    return W.at[ix].set(Wg), None

                W, _ = lax.scan(step, W, perm)
                return (W, lr * DECAY), None

            (W, _), _ = lax.scan(epoch, (W0, lr0.astype(dt)), (perms, jnp.arange(emax)))
            return W

        return jax.vmap(one)(idx, val, perms, lr0, epochs)

    out = []
    step = chunk or P
    yy = jnp.asarray(y, dtype=jnp.int32)
    for a in range(0, P, step):
        b = min(P, a + step)
        W = run(
            jnp.asarray(st.idx[a:b], dtype=jnp.int32),
            jnp.asarray(st.val[a:b], dtype=dt),
            jnp.asarray(st.perms[a:b], dtype=jnp.int32),
            jnp.asarray(st.lr[a:b], dtype=dt),
            jnp.asarray(st.epochs[a:b], dtype=jnp.int32),
            yy,
        )
        W.block_until_ready()
        out.append(np.asarray(W))
    return np.concatenate(out, axis=0)


def dense_weights(st: Setup, W: np.ndarray, p: int) -> np.ndarray:
    """Candidate p's KC->MBON matrix in the reference layout [n_kc, A] (float64)."""
    return np.asarray(W[p, : st.genes[p].n_kc], dtype=np.float64)


# ----------------------------------------------------------------------------- scoring


def predict_population(
    st: Setup, W: np.ndarray, eps: Sequence[Episode], *, protocol: bool, X: np.ndarray | None = None
) -> np.ndarray:
    """[P, n] actions. protocol=True applies the repo's hand-written delayed-cue rule."""
    X = _pn_features(list(eps)) if X is None else X
    out = np.empty((len(st.genes), X.shape[0]), dtype=np.int64)
    for p, g in enumerate(st.genes):
        i, v = sparse_kc(X, st.W_pn[p], g.k)
        scores = np.einsum("nk,nka->na", v, W[p][i].astype(np.float64))
        raw = scores.argmax(axis=1)
        out[p] = _apply_delayed_cue_protocol(list(eps), raw) if protocol else raw
    return out


# ----------------------------------------------------------------------------- evolution


def random_gene(rng: np.random.Generator) -> Gene:
    hidden = int(rng.choice([32, 48, 64, 96, 128, 192, 256]))
    return Gene(
        seed=int(rng.integers(0, 2**31 - 1)),
        hidden=hidden,
        k_winners=int(rng.integers(2, max(3, hidden // 4))),
        lr=float(np.exp(rng.uniform(np.log(0.05), np.log(1.5)))),
        epochs=int(rng.integers(2, 21)),
    )


def mutate(g: Gene, rng: np.random.Generator) -> Gene:
    hidden = int(np.clip(round(g.n_kc * float(np.exp(rng.normal(0, 0.25)))), 32, 512))
    k = int(np.clip(round(g.k * float(np.exp(rng.normal(0, 0.3)))), 1, hidden))
    return replace(
        g,
        seed=int(rng.integers(0, 2**31 - 1)) if rng.random() < 0.5 else g.seed,
        hidden=hidden,
        k_winners=k,
        lr=float(np.clip(g.lr * np.exp(rng.normal(0, 0.3)), 0.01, 3.0)),
        epochs=int(np.clip(g.epochs + rng.integers(-3, 4), 1, 30)),
    )


def pareto_front(points: np.ndarray) -> np.ndarray:
    """points [n, m], all objectives to MAXIMIZE. Returns boolean mask of the front."""
    n = points.shape[0]
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        dom = np.all(points >= points[i], axis=1) & np.any(points > points[i], axis=1)
        if dom.any():
            mask[i] = False
    return mask


def nondominated_rank(points: np.ndarray) -> np.ndarray:
    rank = np.full(points.shape[0], -1)
    remaining = np.arange(points.shape[0])
    r = 0
    while remaining.size:
        m = pareto_front(points[remaining])
        rank[remaining[m]] = r
        remaining = remaining[~m]
        r += 1
    return rank


def evolve(
    ts: TrainSet,
    val_eps: Sequence[Episode],
    *,
    pop: int = 64,
    generations: int = 5,
    protocol: bool = True,
    backend: str = "numpy",
    seed: int = 0,
    seed_genes: Sequence[Gene] = (Gene(),),
    log=print,
) -> dict[str, Any]:
    """(mu + lambda) on val per-step accuracy (max) and params (min). Val only."""
    rng = np.random.default_rng(seed)
    v_eps, v_y = prefix_episodes(val_eps)
    Xv = _pn_features(v_eps)
    genes = list(seed_genes) + [random_gene(rng) for _ in range(pop - len(seed_genes))]
    history = []
    scored: dict[Gene, dict[str, float]] = {}
    for gen in range(generations):
        todo = [g for g in dict.fromkeys(genes) if g not in scored]
        t0 = time.perf_counter()
        if todo:
            st = setup_population(todo, ts)
            t1 = time.perf_counter()
            W = train_jax(st, ts.y) if backend == "jax" else train_numpy(st, ts.y)
            t2 = time.perf_counter()
            pv = predict_population(st, W, v_eps, protocol=protocol, X=Xv)
            accs = (pv == v_y[None, :]).mean(axis=1)
            for g, a in zip(todo, accs):
                scored[g] = {"val_step": float(a), "params": g.n_params}
        else:
            t1 = t2 = time.perf_counter()
        uniq = list(dict.fromkeys(genes))
        pts = np.asarray([[scored[g]["val_step"], -scored[g]["params"]] for g in uniq])
        rank = nondominated_rank(pts)
        order = np.lexsort((pts[:, 1] * -1, -pts[:, 0], rank))
        parents = [uniq[i] for i in order[: max(2, pop // 2)]]
        best = uniq[int(np.lexsort((-pts[:, 1], -pts[:, 0]))[0])]
        history.append(
            {
                "generation": gen,
                "trained": len(todo),
                "setup_s": t1 - t0,
                "train_s": t2 - t1,
                "best_val_step": scored[best]["val_step"],
                "best_gene": best.to_dict(),
                "front": [uniq[i].to_dict() | scored[uniq[i]] for i in np.where(rank == 0)[0]],
            }
        )
        log(
            f"gen {gen}: trained {len(todo)} in {t2 - t1:.2f}s (setup {t1 - t0:.2f}s) "
            f"best val_step={scored[best]['val_step']:.4f} params={best.n_params} front={int((rank == 0).sum())}"
        )
        children = [mutate(parents[int(rng.integers(0, len(parents)))], rng) for _ in range(pop - len(parents))]
        genes = parents + children
    uniq = list(scored)
    pts = np.asarray([[scored[g]["val_step"], -scored[g]["params"]] for g in uniq])
    front = [uniq[i] for i in np.where(pareto_front(pts))[0]]
    return {"history": history, "scored": scored, "front": front}
