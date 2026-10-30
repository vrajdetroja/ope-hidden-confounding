"""Breakdown Gamma and the benchmark Gamma-hat calibrated from recorded variables.

Decision: deploy pi_e if its value beats the clinician's, which is identified (mean return).
Breakdown Gamma: the largest Gamma at which the lower bound on V(pi_e) still exceeds V_hat(pi_b).
Benchmark Gamma-hat: refit the propensity without each recorded variable j; the 95th percentile,
over well-visited (s, a), of the MSM odds ratio between e_hat(a | s) and e_hat_{-j}(a | s_{-j});
the largest over j ("a hidden factor as strong as the strongest recorded one"). This adapts
Hsu & Small's (2013) calibration for matched studies to OPE.
Alarm: fires when breakdown Gamma < Gamma-hat. The oracle alarm uses Gamma* instead.
"""

from __future__ import annotations

import numpy as np

from .data import Episodes
from .msm import msm_odds_ratio
from .sim.spaces import decode, encode, factors_for

N_ACTIONS = 8


def breakdown_gamma(lower_bound_at, v_clinician, lo=1.0, hi=20.0, tol=1e-3):
    """Largest Gamma at which the decision 'deploy pi_e' survives."""
    if lower_bound_at(lo) <= v_clinician:
        return None                           # nominal analysis already says no
    if lower_bound_at(hi) > v_clinician:
        return hi                             # robust across the whole grid
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if lower_bound_at(mid) > v_clinician:
            lo = mid
        else:
            hi = mid
    return lo


def benchmark_gamma(ep: Episodes, smoothing=0.5, min_count=20, quantile=0.95) -> tuple[float, dict]:
    """Gamma-hat from the recorded variables, using all steps pooled."""
    S, A = ep.n_states, N_ACTIONS
    factors, sizes = factors_for(S)
    counts = np.bincount((ep.s * A + ep.a)[ep.alive], minlength=S * A).reshape(S, A).astype(float)
    e_full = (counts + smoothing) / (counts.sum(axis=1, keepdims=True) + smoothing * A)
    well = counts >= min_count
    f = decode(np.arange(S), factors, sizes)
    per_var = {}
    for j in factors:
        key = encode({**f, j: np.zeros_like(f[j])}, factors, sizes)
        reduced = np.zeros((S, A))
        np.add.at(reduced, key, counts)
        e_red = (reduced + smoothing) / (reduced.sum(axis=1, keepdims=True) + smoothing * A)
        g = msm_odds_ratio(e_full[well], e_red[key][well])
        per_var[j] = float(np.quantile(g, quantile)) if g.size else np.nan
    values = [v for v in per_var.values() if np.isfinite(v)]
    return (max(values) if values else np.nan), per_var
