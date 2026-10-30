"""In a one-step problem, weighted IS with unsmoothed count propensities equals the model-based
estimate (when the policy only uses observed actions)."""

import numpy as np

from ope_conf.data import Episodes
from ope_conf.estimators import Fit, nominal_estimates
from ope_conf.toy import Bandit


def one_step_episodes(s, a, y, n_states):
    col = lambda v: np.asarray(v)[:, None]
    # one decision; the next state is a terminal placeholder (reward already paid)
    return Episodes(col(s), col(a), col(y), col(np.zeros_like(s)), col(np.ones_like(s, bool)),
                    col(np.ones_like(s, bool)), n_states)


def test_wis_equals_mb_one_step():
    rng = np.random.default_rng(50)
    for _ in range(20):
        bandit = Bandit.random(rng, n_states=5, lam=float(rng.uniform(0, 2)))
        s, a, y, _u = bandit.sample(3000, rng)
        S = 8 * 5                                  # pad the action space to the simulator's 8
        ep = one_step_episodes(s, a, y, S)
        pi = np.zeros((1, S, 8))
        pi[0, :5, :2] = rng.dirichlet(np.ones(2), size=5)
        fit = Fit(ep, np.zeros(1, int), smoothing=0.0)
        reward, terminal = np.zeros(S), np.ones(S, bool)
        est = nominal_estimates(ep, fit, pi, reward, terminal)
        # MB here averages observed outcomes within cells, so it needs reward on the transition:
        # recompute MB directly as sum_s q(s) sum_a pi(a|s) mean(y | s, a)
        mb = sum(np.mean(s == k) * sum(pi[0, k, j] * y[(s == k) & (a == j)].mean() for j in range(2))
                 for k in range(5))
        assert abs(est["wis"] - mb) < 1e-12
        assert abs(est["pdis"] - mb) < 1e-12   # with count propensities, PDIS = WIS here too
