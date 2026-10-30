"""The ported transition tensor matches the original sim-v2 sampling code."""

import numpy as np
import pytest
from scipy import stats

from ope_conf.sim.dynamics import start_distribution, transition_matrix
from ope_conf.sim.spaces import N_ACTIONS, N_FULL, decode, outcome_tables

from sepsis_original import sample_next_states, sample_start_states


def chi_square_pvalue(samples, probs, min_expected=5.0):
    """Goodness of fit, pooling cells with small expected counts into one bin."""
    n = samples.size
    observed = np.bincount(samples, minlength=probs.size).astype(float)
    assert np.all(probs[observed > 0] > 0), "original simulator reached a state the port forbids"
    expected = n * probs
    big = expected >= min_expected
    obs = np.append(observed[big], observed[~big].sum())
    exp = np.append(expected[big], expected[~big].sum())
    keep = exp > 0
    obs, exp = obs[keep], exp[keep]
    if obs.size < 2:
        return 1.0
    return stats.chisquare(obs, exp).pvalue


@pytest.fixture(scope="module")
def P():
    return transition_matrix()


def test_rows_are_distributions_and_treatments_follow_action(P):
    np.testing.assert_allclose(np.asarray(P.sum(axis=1)).ravel(), 1.0, atol=1e-12)
    rows, cols = P.nonzero()
    s, a = rows // N_ACTIONS, rows % N_ACTIONS
    fs, fn = decode(s), decode(cols)
    assert np.all(fs["diab"] == fn["diab"])
    assert np.all(fn["abx"] == a // 4) and np.all(fn["vent"] == (a // 2) % 2)
    assert np.all(fn["vaso"] == a % 2)


def test_chi_square_200_pairs_1000_draws(P):
    rng = np.random.default_rng(40)
    _, terminal = outcome_tables()
    states = rng.choice(np.flatnonzero(~terminal), size=200)
    actions = rng.integers(0, N_ACTIONS, size=200)
    pvals = []
    for k, (s, a) in enumerate(zip(states, actions)):
        draws = sample_next_states(int(s), int(a), 1000, seed=1000 + k)
        pvals.append(chi_square_pvalue(draws, P[s * N_ACTIONS + a].toarray().ravel()))
    pvals = np.array(pvals)
    assert pvals.min() > 1e-4, f"min p-value {pvals.min():.2e} (Bonferroni-scale failure)"
    assert stats.kstest(pvals, "uniform").pvalue > 1e-3   # p-values look uniform overall


def test_start_distribution_matches_get_new_state():
    draws = sample_start_states(40_000, seed=41)
    assert chi_square_pvalue(draws, start_distribution()) > 1e-3


@pytest.mark.slow
def test_chi_square_high_power_pairs(P):
    """Fewer pairs, 100,000 draws each: detects probability errors of about 0.005."""
    rng = np.random.default_rng(42)
    _, terminal = outcome_tables()
    for k, s in enumerate(rng.choice(np.flatnonzero(~terminal), size=16)):
        for a in (0, 7, int(rng.integers(1, 7))):
            draws = sample_next_states(int(s), a, 100_000, seed=5000 + 10 * k + a)
            assert chi_square_pvalue(draws, P[s * N_ACTIONS + a].toarray().ravel()) > 1e-4
