"""Nominal estimators: PDIS, WIS, CWPDIS, model-based (MB) and weighted doubly robust (WDR).

All of them assume sequential ignorability. Inputs are recorded episodes and episode weights
(1 for the point estimate, bootstrap multiplicities otherwise). `phase_of_t` is the analyst's
time structure: steps in the same phase share one propensity table and one transition model
(R1: step 0 apart from steps 1+; R2 and R3: all steps pooled).

The empirical model lives on the states the data visit (`Fit.idx`); every other state is
unreachable under it. Arrays indexed by compact state carry a `_c` suffix.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .data import Episodes

N_ACTIONS = 8
NOMINAL = ("pdis", "wis", "cwpdis", "mb", "wdr", "mb_death")


def complete_q(q, seen, pi_t, unseen="renormalize"):
    """Fill the Q of never-observed (s, a) pairs.

    'death' scores them -1 (Oberst & Sontag). 'renormalize' gives them the value of the
    policy's observed actions in s, i.e. evaluates pi restricted to the data's support; a state
    where pi puts no mass on any observed action still scores -1.
    """
    if unseen == "death":
        return np.where(seen, q, -1.0)
    ws = np.where(seen, pi_t, 0.0)
    den = ws.sum(axis=1)
    v = np.divide((ws * np.where(seen, q, 0.0)).sum(axis=1), den,
                  out=np.full(den.shape, -1.0), where=den > 0)
    return np.where(seen, q, v[:, None])


class Fit:
    """Count propensities e_hat(a | s) with add-`smoothing` pseudo-counts, and the empirical
    transition model, for one set of episode weights."""

    def __init__(self, ep: Episodes, phase_of_t: np.ndarray, w: np.ndarray | None = None,
                 smoothing: float = 0.0625):
        n, H, S, A = ep.n, ep.H, ep.n_states, N_ACTIONS
        self.w = np.ones(n) if w is None else np.asarray(w, dtype=float)
        self.phase_of_t = np.asarray(phase_of_t)
        self.n_phase = int(self.phase_of_t.max()) + 1
        self.S, self.A = S, A
        sel = ep.alive & (self.w[:, None] > 0)
        # compact states: everything the recorded data visit, whatever the bootstrap weights
        self.idx = np.unique(np.concatenate([ep.s[ep.alive], ep.s_next[ep.alive]]))
        self.pos = np.full(S, -1)
        self.pos[self.idx] = np.arange(self.idx.size)
        n_c = self.idx.size
        ph = np.broadcast_to(self.phase_of_t[None, :], (n, H))[sel]
        ww = np.broadcast_to(self.w[:, None], (n, H))[sel]
        s_c, s2_c, a = self.pos[ep.s[sel]], self.pos[ep.s_next[sel]], ep.a[sel]
        self.counts_c = np.bincount((ph * n_c + s_c) * A + a, weights=ww,
                                    minlength=self.n_phase * n_c * A).reshape(self.n_phase, n_c, A)
        total = self.counts_c.sum(axis=-1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.e_hat_c = (self.counts_c + smoothing) / (total + smoothing * A)
        self.seen_c = self.counts_c > 0
        self.P, self.rows = [], []
        for p in range(self.n_phase):
            m = ph == p
            M = sp.csr_matrix((ww[m], (s_c[m] * A + a[m], s2_c[m])), shape=(n_c * A, n_c))
            mass = self.counts_c[p].ravel()
            inv = np.divide(1.0, mass, out=np.zeros_like(mass), where=mass > 0)
            P = sp.csr_matrix(sp.diags(inv) @ M)
            self.P.append(P)
            self.rows.append(np.repeat(np.arange(n_c * A), np.diff(P.indptr)))
        self.start_c = np.bincount(self.pos[ep.s[:, 0]], weights=self.w, minlength=n_c) / self.w.sum()

    def e_hat(self, phase, s, a):
        """Propensity lookup by full state index (every state in the data is compact)."""
        return self.e_hat_c[phase, self.pos[s], a]

    def compact(self, pi):
        return pi[:, self.idx]

    def expand(self, values_c, fill=0.0):
        out = np.full((self.S,) + values_c.shape[1:], fill, dtype=float)
        out[self.idx] = values_c
        return out


def mb_values(fit: Fit, pi: np.ndarray, rec_reward, rec_terminal, gamma: float = 1.0,
              unseen: str = "renormalize"):
    """Backward induction on the empirical model; see complete_q for never-observed pairs.
    Returns compact Q (H, n_c, A) and V (H + 1, n_c)."""
    H, n_c, A = pi.shape[0], fit.idx.size, fit.A
    pi_c, rr, alive = fit.compact(pi), rec_reward[fit.idx], ~rec_terminal[fit.idx]
    Q = np.zeros((H, n_c, A))
    V = np.zeros((H + 1, n_c))
    for t in reversed(range(H)):
        p = fit.phase_of_t[t]
        y = rr + gamma * alive * V[t + 1]
        Q[t] = complete_q((fit.P[p] @ y).reshape(n_c, A), fit.seen_c[p], pi_c[t], unseen)
        V[t] = np.einsum("sa,sa->s", pi_c[t], Q[t])
    return Q, V


def unseen_mass(fit: Fit, pi: np.ndarray, rec_terminal) -> float:
    """Probability, under pi and the empirical model, of choosing a never-observed (s, a) pair:
    the mass the unseen-pair convention decides (logged as a diagnostic)."""
    pi_c, alive = fit.compact(pi), ~rec_terminal[fit.idx]
    d, total = fit.start_c.copy(), 0.0
    for t in range(pi.shape[0]):
        p = fit.phase_of_t[t]
        flow = d[:, None] * pi_c[t]
        total += float(flow[~fit.seen_c[p]].sum())
        flow = np.where(fit.seen_c[p], flow, 0.0).ravel()
        d = (fit.P[p].T @ flow) * alive
    return total


def step_ratios(ep: Episodes, fit: Fit, pi: np.ndarray) -> np.ndarray:
    """pi_e(a_t | s_t) / e_hat(a_t | s_t), and 1 after termination."""
    t = np.arange(ep.H)[None, :]
    pe = pi[t, ep.s, ep.a]
    eh = np.where(ep.alive, fit.e_hat(fit.phase_of_t[t], ep.s, ep.a), 1.0)
    return np.where(ep.alive, pe / eh, 1.0)


def nominal_estimates(ep: Episodes, fit: Fit, pi: np.ndarray, rec_reward, rec_terminal,
                      gamma: float = 1.0) -> dict:
    """The nominal estimators for one policy, plus the effective sample size of WIS.
    MB and WDR use the renormalised convention for unseen pairs; mb_death is the
    Oberst & Sontag convention, reported for comparison only."""
    w, W = fit.w, fit.w.sum()
    rho = np.cumprod(step_ratios(ep, fit, pi), axis=1)
    disc = gamma ** np.arange(ep.H)[None, :]
    wr = w[:, None] * rho
    pdis = float((wr * ep.r * disc).sum() / W)
    G = (ep.r * disc).sum(axis=1)
    den_H = wr[:, -1].sum()
    wis = float((wr[:, -1] * G).sum() / den_H) if den_H > 0 else np.nan
    den_t = wr.sum(axis=0)
    norm = np.divide(wr, den_t, out=np.zeros_like(wr), where=den_t > 0)
    cwpdis = float((norm * ep.r * disc).sum())
    Q, V = mb_values(fit, pi, rec_reward, rec_terminal, gamma)
    tt = np.arange(ep.H)[None, :]
    s_c = np.where(ep.alive, fit.pos[ep.s], 0)
    q_sa = np.where(ep.alive, Q[tt, s_c, ep.a], 0.0)
    v_s = np.where(ep.alive, V[tt, s_c], 0.0)
    prev = np.concatenate([w[:, None] / W, norm[:, :-1]], axis=1)
    wdr = float((disc * (norm * ep.r - (norm * q_sa - prev * v_s))).sum())
    mb = float(fit.start_c @ V[0])
    mb_death = float(fit.start_c @ mb_values(fit, pi, rec_reward, rec_terminal, gamma, "death")[1][0])
    ess = float(den_H ** 2 / (w * rho[:, -1] ** 2).sum()) if den_H > 0 else 0.0
    return {"pdis": pdis, "wis": wis, "cwpdis": cwpdis, "mb": mb, "wdr": wdr, "mb_death": mb_death,
            "_ess": ess, "_V": V}


def clinician_estimate(ep: Episodes, w: np.ndarray | None = None) -> float:
    """The clinician's value is identified: the (weighted) mean observed return."""
    w = np.ones(ep.n) if w is None else w
    return float(w @ ep.returns() / w.sum())


def bootstrap_weights(n: int, B: int, rng: np.random.Generator) -> np.ndarray:
    """Episode-level bootstrap: (B, n) multiplicities, each row a multinomial draw of n episodes."""
    return rng.multinomial(n, np.full(n, 1.0 / n), size=B).astype(float)
