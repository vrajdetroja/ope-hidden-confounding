"""Episode generation. Recorded columns (what estimators see) and hidden columns (u or diabetes,
the true behavior probabilities, the exact nominal propensities) live in separate objects
and separate files; estimators only ever receive `Episodes`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .sim.confounding import World


@dataclass(frozen=True)
class Episodes:
    """Recorded data as (n, H) arrays, padded after termination (alive = False, r = 0)."""
    s: np.ndarray
    a: np.ndarray
    r: np.ndarray
    s_next: np.ndarray
    done: np.ndarray
    alive: np.ndarray
    n_states: int

    @property
    def n(self) -> int:
        return self.s.shape[0]

    @property
    def H(self) -> int:
        return self.s.shape[1]

    def returns(self) -> np.ndarray:
        return self.r.sum(axis=1)

    def subset(self, idx) -> "Episodes":
        return Episodes(self.s[idx], self.a[idx], self.r[idx], self.s_next[idx], self.done[idx],
                        self.alive[idx], self.n_states)

    def to_frame(self) -> pd.DataFrame:
        ep, t = np.nonzero(self.alive)
        return pd.DataFrame({"episode": ep, "t": t, "s": self.s[ep, t], "a": self.a[ep, t],
                             "r": self.r[ep, t], "s_next": self.s_next[ep, t], "done": self.done[ep, t]})


@dataclass(frozen=True)
class Hidden:
    """Diagnostics only: never passed to an estimator."""
    h: np.ndarray        # (n, H) hidden value at each step (u_t, or diabetes in R3)
    pb: np.ndarray       # (n, H) true pi_b(a_t | s_t, h_t)
    e_true: np.ndarray   # (n, H) exact nominal propensity e(a_t | s_t) under the analyst's phases

    def to_frame(self, alive: np.ndarray) -> pd.DataFrame:
        ep, t = np.nonzero(alive)
        return pd.DataFrame({"episode": ep, "t": t, "hidden": self.h[ep, t],
                             "pi_b": self.pb[ep, t], "e": self.e_true[ep, t]})


class RowSampler:
    """Exact categorical sampling from the rows of a sparse row-stochastic matrix."""

    def __init__(self, P: sp.csr_matrix):
        P = sp.csr_matrix(P)
        P.eliminate_zeros()
        self.indptr, self.indices = P.indptr, P.indices
        counts = np.diff(P.indptr)
        row_of = np.repeat(np.arange(P.shape[0]), counts)
        cum = np.cumsum(P.data)
        within = cum - (cum[P.indptr[:-1]] - P.data[P.indptr[:-1]])[row_of]
        within /= within[P.indptr[1:] - 1][row_of]            # last entry of each row is exactly 1
        self.keys = row_of + within                            # increasing across the whole matrix

    def sample(self, rows: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        pos = np.searchsorted(self.keys, rows + rng.random(rows.size), side="right")
        pos = np.minimum(pos, self.indptr[rows + 1] - 1)
        return self.indices[pos]


def sample_categorical(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    cum = np.cumsum(probs, axis=1)
    a = (rng.random(probs.shape[0])[:, None] > cum).sum(axis=1)
    return np.minimum(a, probs.shape[1] - 1)


def _samplers(world: World) -> list:
    if "samplers" not in world.extra:
        world.extra["samplers"] = [RowSampler(K) for K in world.kernels]
    return world.extra["samplers"]


def generate(world: World, n: int, rng: np.random.Generator) -> tuple[Episodes, Hidden]:
    """Simulate n episodes of the clinician. Episodes stop on entering discharge or death and
    are truncated at H steps (unresolved episodes score 0), as in Oberst & Sontag."""
    from .truth import nominal_propensity

    H, A, n_rec = world.H, world.A, world.n_rec
    if "e_true" not in world.extra:
        world.extra["e_true"] = nominal_propensity(world)
    e_true = world.extra["e_true"]
    samplers = _samplers(world)
    S, Aa, S2, Hh = (np.zeros((n, H), dtype=np.int64) for _ in range(4))
    R, PB, E = np.zeros((n, H)), np.ones((n, H)), np.ones((n, H))
    D, alive = np.zeros((n, H), bool), np.zeros((n, H), bool)
    x = rng.choice(world.n_aug, size=n, p=world.mu)
    active = np.arange(n)
    for t in range(H):
        if active.size == 0:
            break
        xa = x[active]
        probs = world.behavior[world.bphase_of_t[t]][xa]
        a = sample_categorical(probs, rng)
        x2 = samplers[world.kernel_of_t[t]].sample(xa * A + a, rng)
        rec, rec2 = xa % n_rec, x2 % n_rec
        S[active, t], Aa[active, t], S2[active, t], Hh[active, t] = rec, a, rec2, xa // n_rec
        R[active, t] = world.rec_reward[rec2]
        D[active, t] = world.rec_terminal[rec2]
        alive[active, t] = True
        PB[active, t] = probs[np.arange(a.size), a]
        E[active, t] = e_true[world.est_phase_of_t[t], rec, a]
        x[active] = x2
        active = active[~world.rec_terminal[rec2]]
    return Episodes(S, Aa, R, S2, D, alive, n_rec), Hidden(Hh, PB, E)


def save(episodes: Episodes, hidden: Hidden, directory: str | Path) -> None:
    """Two files: estimators read only recorded.parquet."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    episodes.to_frame().to_parquet(directory / "recorded.parquet", index=False)
    hidden.to_frame(episodes.alive).to_parquet(directory / "hidden.parquet", index=False)


def load_recorded(directory: str | Path, n_states: int, H: int) -> Episodes:
    df = pd.read_parquet(Path(directory) / "recorded.parquet")
    n = int(df["episode"].max()) + 1
    arrays = {k: np.zeros((n, H), dtype=dt) for k, dt in
              (("s", np.int64), ("a", np.int64), ("r", float), ("s_next", np.int64), ("done", bool))}
    alive = np.zeros((n, H), bool)
    ep, t = df["episode"].to_numpy(), df["t"].to_numpy()
    for k in arrays:
        arrays[k][ep, t] = df[k].to_numpy()
    alive[ep, t] = True
    return Episodes(arrays["s"], arrays["a"], arrays["r"], arrays["s_next"], arrays["done"], alive, n_states)
