"""The sort is the exact solution of the MSM linear program (to 1e-9 on 1,000 random cells)."""

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from ope_conf.msm import (cell_lower_bound, cell_upper_bound, grouped_bound, omega_box,
                          robust_backup, robust_rows)

TOL = 1e-9
LP_OPTIONS = {"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10}


def lp_cell(z, e, gamma, lower=True):
    """min (or max) of mean(omega * z) s.t. w_lo <= omega_i <= w_hi, mean(omega) = 1."""
    n = len(z)
    w_lo, w_hi = omega_box(e, gamma)
    sign = 1.0 if lower else -1.0
    res = linprog(sign * np.asarray(z) / n, A_eq=np.ones((1, n)) / n, b_eq=[1.0],
                  bounds=[(w_lo, w_hi)] * n, method="highs-ds", options=LP_OPTIONS)
    assert res.status == 0, res.message
    return sign * res.fun


def lp_backup(p, y, e, gamma):
    """min sum_j q_j y_j s.t. p_j w_lo <= q_j <= p_j w_hi, sum_j q_j = 1."""
    w_lo, w_hi = omega_box(e, gamma)
    res = linprog(y, A_eq=np.ones((1, len(p))), b_eq=[1.0],
                  bounds=list(zip(p * w_lo, p * w_hi)), method="highs-ds", options=LP_OPTIONS)
    assert res.status == 0, res.message
    return res.fun


def random_cell(rng):
    n = int(rng.integers(1, 60))
    if rng.random() < 0.5:   # returns like the simulator's: ties at -1, 0, 1
        z = rng.choice([-1.0, 0.0, 1.0], size=n)
    else:
        z = rng.normal(size=n)
    e = float(rng.uniform(0.01, 0.99))
    gamma = float(np.exp(rng.uniform(0, np.log(20))))
    return z, e, gamma


def test_sort_matches_lp_on_1000_cells():
    rng = np.random.default_rng(0)
    for _ in range(1000):
        z, e, gamma = random_cell(rng)
        assert abs(cell_lower_bound(z, e, gamma) - lp_cell(z, e, gamma, lower=True)) < TOL
        assert abs(cell_upper_bound(z, e, gamma) - lp_cell(z, e, gamma, lower=False)) < TOL


def test_robust_backup_matches_lp():
    rng = np.random.default_rng(1)
    for _ in range(1000):
        k = int(rng.integers(1, 30))
        p = rng.dirichlet(np.ones(k) * 0.5)
        y = rng.choice([-1.0, 0.0, 0.3, 1.0], size=k) if rng.random() < 0.5 else rng.normal(size=k)
        e = float(rng.uniform(0.01, 0.99))
        gamma = float(np.exp(rng.uniform(0, np.log(20))))
        assert abs(robust_backup(p, y, e, gamma) - lp_backup(p, y, e, gamma)) < TOL


def test_vectorised_forms_match_reference():
    rng = np.random.default_rng(2)
    rows, cols, vals, ys, es, gams, refs_lo, refs_hi = [], [], [], [], [], [], [], []
    col = 0
    for r in range(1000):
        k = int(rng.integers(1, 25))
        p = rng.dirichlet(np.ones(k))
        y = rng.normal(size=k)
        e, gamma = float(rng.uniform(0.01, 0.99)), float(np.exp(rng.uniform(0, np.log(20))))
        rows += [r] * k
        cols += list(range(col, col + k))
        vals += list(p)
        ys += list(y)
        es.append(e)
        gams.append(gamma)
        refs_lo.append(robust_backup(p, y, e, gamma))
        refs_hi.append(-robust_backup(p, -y, e, gamma))
        col += k
    P = sp.csr_matrix((vals, (rows, cols)), shape=(1000, col))
    lo = robust_rows(P, np.array(ys), np.array(es), np.array(gams), lower=True)
    hi = robust_rows(P, np.array(ys), np.array(es), np.array(gams), lower=False)
    np.testing.assert_allclose(lo, refs_lo, atol=TOL)
    np.testing.assert_allclose(hi, refs_hi, atol=TOL)


def test_grouped_bound_matches_cellwise_sort():
    rng = np.random.default_rng(3)
    cells, zs, es, gamma = [], [], [], 3.0
    expect = []
    for c in range(300):
        z, e, _ = random_cell(rng)
        cells += [c] * len(z)
        zs += list(z)
        es.append(e)
        expect.append(cell_lower_bound(z, e, gamma))
    got = grouped_bound(np.array(cells), np.array(zs), np.ones(len(zs)), np.array(es), gamma, 300)
    np.testing.assert_allclose(got, expect, atol=TOL)


def test_grouped_bound_weights_equal_replication():
    """A sample with bootstrap multiplicity m counts like m copies of itself."""
    rng = np.random.default_rng(4)
    z = rng.normal(size=40)
    m = rng.integers(0, 4, size=40)
    replicated = np.repeat(z, m)
    got = grouped_bound(np.zeros(40, int), z, m, np.array([0.3]), 2.5, 1)[0]
    assert abs(got - cell_lower_bound(replicated, 0.3, 2.5)) < TOL
