"""Sharp worst-case expectations under Tan's marginal sensitivity model (MSM).

Under the MSM with parameter Gamma, the true inverse propensity of the taken action,
W = 1 / pi_b(a | s, u), lies in [alpha, beta] and averages to 1 / e(a | s) within the
(s, a) cell. Rescaled as omega = e * W it lies in [w_lo, w_hi] and has mean 1, so the
worst-case mean of an outcome puts w_hi on the lowest 1 / (1 + Gamma) share of the cell's
mass and w_lo on the rest: a sort. The first three functions are the reference versions;
`robust_rows` and `grouped_bound` are the vectorised forms the estimators use, tested
against the references and against a generic LP.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def inverse_propensity_box(e, gamma):
    """[alpha, beta] for W = 1 / pi_b(a | s, u) given the nominal propensity e."""
    e = np.asarray(e, dtype=float)
    return 1 + (1 / e - 1) / gamma, 1 + gamma * (1 / e - 1)


def omega_box(e, gamma):
    """[w_lo, w_hi] for omega = e * W, which has mean 1 within each cell."""
    e = np.asarray(e, dtype=float)
    return e + (1 - e) / gamma, e + gamma * (1 - e)


def cell_lower_bound(z, e_hat, gamma):
    """Worst-case mean of outcomes z in one (state, action) cell under the MSM."""
    z = np.sort(np.asarray(z, dtype=float))   # ascending: worst outcomes first
    n = z.size
    w_lo = e_hat + (1 - e_hat) / gamma
    w_hi = e_hat + gamma * (1 - e_hat)
    w = np.full(n, w_lo)
    extra = n * (1 - w_lo)                    # mass needed so weights average 1
    cap = w_hi - w_lo
    k = int(extra // cap) if cap > 0 else 0   # about n / (1 + gamma) units
    w[:k] += cap
    if k < n:
        w[k] += extra - k * cap               # fractional unit at the quantile
    return float(w @ z / n)


def cell_upper_bound(z, e_hat, gamma):
    return -cell_lower_bound(-np.asarray(z, dtype=float), e_hat, gamma)


def robust_backup(p_next, y, e_hat, gamma):
    """Worst-case expectation of y over next states with probabilities p_next."""
    order = np.argsort(y)                     # worst next states first
    p, y = p_next[order], y[order]
    w_lo = e_hat + (1 - e_hat) / gamma
    w_hi = e_hat + gamma * (1 - e_hat)
    cap = p * (w_hi - w_lo)                   # extra mass each next state can absorb
    room = (1 - w_lo) - np.concatenate(([0.0], np.cumsum(cap)[:-1]))
    q = p * w_lo + np.clip(np.minimum(cap, room), 0.0, None)
    return float(q @ y)


def robust_rows(P, y, e_row, gamma, lower=True, rows=None):
    """Worst-case expectation of y under every row of a row-stochastic CSR matrix.

    Row r is tilted inside the MSM box of its propensity e_row[r]. `gamma` is a scalar or
    one value per row. Empty rows return NaN. At gamma = 1 this is exactly P @ y.
    `rows` (the row of every stored entry) may be passed to skip recomputing it.
    """
    P = P if sp.isspmatrix_csr(P) else sp.csr_matrix(P)
    n_rows = P.shape[0]
    counts = np.diff(P.indptr)
    y = np.asarray(y, dtype=float)
    if np.all(np.asarray(gamma) == 1.0):
        out = P @ y
        out[counts == 0] = np.nan
        return out
    if rows is None:
        rows = np.repeat(np.arange(n_rows), counts)
    sign = 1.0 if lower else -1.0
    yy = sign * y[P.indices]
    span = 2.0 * (np.abs(yy).max() + 1.0) if yy.size else 1.0
    order = np.argsort(rows * span + yy)      # rows stay contiguous, worst outcome first
    yy, p = yy[order], P.data[order]
    gam = np.asarray(gamma, dtype=float)
    gam = gam if gam.ndim == 0 else gam[rows]
    w_lo, w_hi = omega_box(np.asarray(e_row, dtype=float)[rows], gam)
    cap = p * (w_hi - w_lo)
    before = np.cumsum(cap) - cap             # extra mass already placed, global running sum...
    before -= before[P.indptr[:-1][rows]]     # ...restarted at the first entry of each row
    room = (1 - w_lo) - before
    q = p * w_lo + np.clip(np.minimum(cap, room), 0.0, None)
    out = sign * np.bincount(rows, weights=q * yy, minlength=n_rows)
    out[counts == 0] = np.nan
    return out


def grouped_bound(cell, z, mass, e_cell, gamma, n_cells, lower=True):
    """The sort applied to every cell at once: worst-case weighted mean of samples z.

    cell[i] is the cell of sample i and mass[i] >= 0 its weight (bootstrap multiplicity).
    Returns one value per cell, NaN where the cell has no mass.
    """
    cell, z, mass = (np.asarray(v) for v in (cell, z, mass))
    keep = mass > 0
    cell, z, mass = cell[keep], z[keep].astype(float), mass[keep].astype(float)
    total = np.bincount(cell, weights=mass, minlength=n_cells)
    P = sp.csr_matrix((mass / total[cell], (cell, np.arange(cell.size))),
                      shape=(n_cells, cell.size))
    return robust_rows(P, z, e_cell, gamma, lower)


def msm_odds_ratio(p, e):
    """max(OR, 1/OR) between a true propensity p and a nominal propensity e."""
    p, e = np.asarray(p, dtype=float), np.asarray(e, dtype=float)
    ratio = (p / (1 - p)) / (e / (1 - e))
    return np.maximum(ratio, 1 / ratio)
