"""Sensitivity bounds on V(pi_e) under the marginal sensitivity model, loosest to most structured.

box         Namkoong et al.'s Lemma 2 in MSM form: at each confounded step the inverse propensity
            is replaced by beta when the return is negative and alpha when it is positive (no
            normalisation). Valid, expected to be nearly vacuous.
sort        R1: the continuation value V_1 comes from steps 1+ only (unconfounded in R1); in each
            start cell (s_0, a_0) the outcomes Z = r_0 + V_1(s_1) are sorted and tilted.
            Cells with fewer than `min_cell` visits borrow the cell pooled over glucose.
robust_dp   R2: the same tilt on every (s, a) row of the empirical model at every step,
            backward from the horizon (Bruns-Smith & Zhou's robust Bellman operator).
structured  R1, stretch: only the antibiotic factor is tilted. Outcomes are first reweighted for
            the identified ventilation / vasopressor choice, then sorted in (s_0, abx) cells
            under the binary-factor MSM.

Every bound returns its value at Gamma = 1 when called with gamma = 1, which is the nominal
estimate it is built on (PDIS, MB with pooled start cells, MB, and a factor-reweighted MB).
"""

from __future__ import annotations

import numpy as np

from .data import Episodes
from .estimators import Fit, complete_q, mb_values, step_ratios
from .msm import grouped_bound, inverse_propensity_box, robust_rows
from .sim.spaces import decode, encode, factors_for

BOUND_METHODS = ("box", "sort", "robust_dp", "structured")


def glucose_pool_map(n_states: int) -> tuple[np.ndarray, int]:
    """Map each recorded state to its cell with glucose dropped (dense ids)."""
    factors, sizes = factors_for(n_states)
    f = decode(np.arange(n_states), factors, sizes)
    f["glu"] = np.zeros_like(f["glu"])
    _, pool = np.unique(encode(f, factors, sizes), return_inverse=True)
    return pool, int(pool.max()) + 1


def box_bound(ep: Episodes, fit: Fit, pi, gamma, confounded_t, lower=True) -> float:
    t = np.arange(ep.H)[None, :]
    pe = pi[t, ep.s, ep.a]
    eh = np.where(ep.alive, fit.e_hat(fit.phase_of_t[t], ep.s, ep.a), 1.0)
    alpha, beta = inverse_propensity_box(eh, gamma)
    G = ep.returns()
    big = (G < 0) if lower else (G > 0)                 # where the large weight hurts
    inv = np.where(big[:, None], beta, alpha)
    factor = np.where(ep.alive & confounded_t[None, :], pe * inv, step_ratios(ep, fit, pi))
    return float(fit.w @ (np.prod(factor, axis=1) * G) / fit.w.sum())


class StartCells:
    """Start-cell bookkeeping shared by the sort and structured bounds (compact states)."""

    def __init__(self, ep: Episodes, fit: Fit, pool: np.ndarray, n_pool: int, smoothing: float):
        self.s0, self.a0 = fit.pos[ep.s[:, 0]], ep.a[:, 0]
        self.s1 = fit.pos[ep.s_next[:, 0]]
        self.pool_c, self.n_pool, self.sm = pool[fit.idx], n_pool, smoothing
        self.w = fit.w
        self.counts_t0 = fit.counts_c[fit.phase_of_t[0]]      # (n_c, A) counts behind e_hat at t = 0
        self.e0 = fit.e_hat_c[fit.phase_of_t[0]]

    def pooled_counts(self, counts):
        pooled = np.zeros((self.n_pool,) + counts.shape[1:])
        np.add.at(pooled, self.pool_c, counts)
        return pooled


def _smooth(counts, sm):
    k = counts.shape[-1]
    return (counts + sm) / (counts.sum(axis=-1, keepdims=True) + sm * k)


def _select(raw, pooled_for_cell, n_raw, min_cell, pi_t, n_cols):
    """Use the cell if it has min_cell visits, else its glucose pool; a start cell with no data
    at all gets the renormalised value of the policy's other start actions (as MB does)."""
    use_raw = n_raw >= min_cell
    value = np.where(use_raw, raw, pooled_for_cell)
    has = ~np.isnan(value)
    filled = complete_q(np.nan_to_num(value).reshape(-1, n_cols), has.reshape(-1, n_cols), pi_t)
    return filled.ravel(), (~use_raw & has), ~has


def sort_bound(ep: Episodes, fit: Fit, pi, gamma, V1, cells: StartCells, min_cell,
               lower=True, return_diag=False):
    """Sharp bound with the hidden variable acting at the first decision only (R1).
    V1 is the compact continuation value from mb_values(...)[1][1]."""
    n_c, A = fit.idx.size, fit.A
    z = ep.r[:, 0] + (~ep.done[:, 0]) * V1[cells.s1]
    cell = cells.s0 * A + cells.a0
    raw = grouped_bound(cell, z, cells.w, cells.e0.ravel(), gamma, n_c * A, lower)
    n_raw = np.bincount(cell, weights=cells.w, minlength=n_c * A)
    e_pool = _smooth(cells.pooled_counts(cells.counts_t0), cells.sm)
    pcell = cells.pool_c[cells.s0] * A + cells.a0
    pooled = grouped_bound(pcell, z, cells.w, e_pool.ravel(), gamma, cells.n_pool * A, lower)
    pooled_for_cell = pooled[(cells.pool_c[:, None] * A + np.arange(A)).ravel()]
    pi0 = fit.compact(pi)[0]
    lb, used_pool, empty = _select(raw, pooled_for_cell, n_raw, min_cell, pi0, A)
    weight = (fit.start_c[:, None] * pi0).ravel()
    value = float(weight @ lb)
    if return_diag:
        return value, {"pooled_share": float(weight @ used_pool), "empty_share": float(weight @ empty)}
    return value


def robust_dp(fit: Fit, pi, gamma, rec_reward, rec_terminal, lower=True, tilt_t=None) -> float:
    """Worst-case value with every (s, a) row of the empirical model tilted inside the MSM box
    at every step (or at the steps in tilt_t). Unseen pairs follow complete_q."""
    H, n_c, A = pi.shape[0], fit.idx.size, fit.A
    tilt_t = np.ones(H, bool) if tilt_t is None else tilt_t
    pi_c, rr, alive = fit.compact(pi), rec_reward[fit.idx], ~rec_terminal[fit.idx]
    V = np.zeros(n_c)
    for t in reversed(range(H)):
        p = fit.phase_of_t[t]
        q = robust_rows(fit.P[p], rr + alive * V, fit.e_hat_c[p].ravel(),
                        gamma if tilt_t[t] else 1.0, lower, fit.rows[p])
        Q = complete_q(np.nan_to_num(q).reshape(n_c, A), fit.seen_c[p], pi_c[t])
        V = np.einsum("sa,sa->s", pi_c[t], Q)
    return float(fit.start_c @ V)


def structured_bound(ep: Episodes, fit: Fit, pi, gamma_factor, V1, cells: StartCells, min_cell,
                     lower=True) -> float:
    """Tilt only the antibiotic factor at the first decision (R1).

    pi_b(a | s, u) = pi_b(abx | s, u) * protocol(vent, vaso | s, abx): the second factor does not
    depend on u, so reweighting outcomes by pi_e(rest | s, abx) / e_hat(rest | s, abx) leaves the
    hidden variable acting only through the binary antibiotic propensity e(abx | s).
    """
    n_c, sm = fit.idx.size, cells.sm
    z = ep.r[:, 0] + (~ep.done[:, 0]) * V1[cells.s1]
    abx, rest = cells.a0 // 4, cells.a0 % 4
    c = cells.counts_t0.reshape(n_c, 2, 4)
    e_abx, e_rest = _smooth(c.sum(axis=2), sm), _smooth(c, sm)
    pi0 = fit.compact(pi)[0].reshape(n_c, 2, 4)
    pi_abx = pi0.sum(axis=2)
    pi_rest = np.divide(pi0, pi_abx[..., None], out=np.zeros_like(pi0), where=pi_abx[..., None] > 0)
    ratio = pi_rest[cells.s0, abx, rest]
    cell = cells.s0 * 2 + abx
    raw = grouped_bound(cell, z * ratio / e_rest[cells.s0, abx, rest], cells.w, e_abx.ravel(),
                        gamma_factor, n_c * 2, lower)
    n_raw = np.bincount(cell, weights=cells.w, minlength=n_c * 2)
    pc = cells.pooled_counts(c)
    pe_abx, pe_rest = _smooth(pc.sum(axis=2), sm), _smooth(pc, sm)
    ps0 = cells.pool_c[cells.s0]
    pooled = grouped_bound(ps0 * 2 + abx, z * ratio / pe_rest[ps0, abx, rest], cells.w,
                           pe_abx.ravel(), gamma_factor, cells.n_pool * 2, lower)
    pooled_for_cell = pooled[(cells.pool_c[:, None] * 2 + np.arange(2)).ravel()]
    lb, _, _ = _select(raw, pooled_for_cell, n_raw, min_cell, pi_abx, 2)
    return float((fit.start_c[:, None] * pi_abx).ravel() @ lb)


def bound_value(method, ep, fit, pi, gamma, ctx, lower=True, V1=None):
    """Dispatch one bound construction. ctx carries the world-level pieces."""
    if method == "box":
        return box_bound(ep, fit, pi, gamma, ctx["confounded_t"], lower)
    if method == "robust_dp":
        return robust_dp(fit, pi, gamma, ctx["rec_reward"], ctx["rec_terminal"], lower)
    if V1 is None:
        V1 = mb_values(fit, pi, ctx["rec_reward"], ctx["rec_terminal"])[1][1]
    if method == "sort":
        return sort_bound(ep, fit, pi, gamma, V1, ctx["cells"], ctx["min_cell"], lower)
    if method == "structured":
        return structured_bound(ep, fit, pi, gamma, V1, ctx["cells"], ctx["min_cell"], lower)
    raise ValueError(method)
