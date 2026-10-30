"""Bounds equal the nominal estimate at Gamma = 1, widen monotonically, and cover at Gamma*."""

import numpy as np

from ope_conf.msm import cell_lower_bound, cell_upper_bound
from ope_conf.toy import Bandit

GAMMAS = [1.0, 1.01, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0, 50.0]


def test_cell_bounds_collapse_and_widen():
    rng = np.random.default_rng(10)
    for _ in range(300):
        z = rng.choice([-1.0, 0.0, 1.0], size=int(rng.integers(1, 50)))
        e = float(rng.uniform(0.02, 0.98))
        lo = [cell_lower_bound(z, e, g) for g in GAMMAS]
        hi = [cell_upper_bound(z, e, g) for g in GAMMAS]
        assert abs(lo[0] - z.mean()) < 1e-12 and abs(hi[0] - z.mean()) < 1e-12
        assert np.all(np.diff(lo) <= 1e-12) and np.all(np.diff(hi) >= -1e-12)
        assert all(l <= z.mean() + 1e-12 <= h + 2e-12 for l, h in zip(lo, hi))
        # Gamma -> infinity: e * mean + (1 - e) * worst outcome
        assert abs(cell_lower_bound(z, e, 1e9) - (e * z.mean() + (1 - e) * z.min())) < 1e-6


def test_population_bound_brackets_truth_at_gamma_star():
    rng = np.random.default_rng(11)
    for _ in range(200):
        bandit = Bandit.random(rng, n_states=int(rng.integers(1, 6)), lam=float(rng.uniform(0, 3)))
        pi = rng.dirichlet(np.ones(2), size=len(bandit.q))
        g_star = bandit.gamma_star()
        lo, hi = (bandit.population_bound(pi, g_star, lower=b) for b in (True, False))
        assert lo - 1e-12 <= bandit.truth(pi) <= hi + 1e-12
        nominal = bandit.nominal_limit(pi)
        assert abs(bandit.population_bound(pi, 1.0) - nominal) < 1e-12
        assert lo <= nominal + 1e-12 <= hi + 2e-12


def test_lambda_zero_has_no_bias_and_gamma_star_one():
    rng = np.random.default_rng(12)
    bandit = Bandit.random(rng, n_states=5, lam=0.0)
    pi = rng.dirichlet(np.ones(2), size=5)
    assert abs(bandit.gamma_star() - 1.0) < 1e-12
    assert abs(bandit.truth(pi) - bandit.nominal_limit(pi)) < 1e-12
