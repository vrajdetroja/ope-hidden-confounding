"""Exact dynamic-programming values match Monte Carlo within 3 standard errors.

Slow tests (make validate): 10^6 episodes of the vectorised sampler per regime, and the
protocol's exact value against the untouched original simulator.
"""

import numpy as np
import pytest
from joblib import Parallel, delayed

from ope_conf.config import Config
from ope_conf.data import generate
from ope_conf.sim.confounding import build_world
from ope_conf.sim.policies import discounted_value_iteration, eps_greedy, greedy
from ope_conf.sim.dynamics import transition_matrix
from ope_conf.sim.spaces import N_FULL, outcome_tables
from ope_conf.truth import true_value_clinician, true_value

from sepsis_original import simulate_returns


@pytest.mark.slow
@pytest.mark.parametrize("cfg", [Config(regime="R1", lam=2.0, kappa_p=0.5), Config(regime="R2", lam=1.0),
                                 Config(regime="R3", w=0.75)], ids=["R1", "R2", "R3"])
def test_clinician_value_matches_1e6_monte_carlo(cfg):
    world = build_world(cfg)
    returns = np.concatenate([generate(world, 100_000, np.random.default_rng(70 + k))[0].returns()
                              for k in range(10)])
    se = returns.std() / np.sqrt(returns.size)
    assert abs(returns.mean() - true_value_clinician(world)) < 3 * se


@pytest.mark.slow
def test_protocol_value_matches_original_simulator():
    """The port, the start distribution and the DP agree with MDP.transition: 10^6 episodes of the
    original code under the u-free protocol (1440 x 8 policy array), split over workers."""
    P = transition_matrix()
    reward, terminal = outcome_tables(N_FULL)
    _, Q = discounted_value_iteration(P, reward, terminal, 0.99)
    policy = eps_greedy(greedy(Q), 0.1)
    world = build_world(Config(regime="R1", lam=0.0, kappa=0.0))   # u has no effect: plain simulator
    exact = true_value(world, np.repeat(policy[None], world.H, axis=0))
    chunks = Parallel(n_jobs=-1)(delayed(simulate_returns)(policy, 62_500, 20, 900 + k) for k in range(16))
    returns = np.concatenate(chunks)
    se = returns.std() / np.sqrt(returns.size)
    assert abs(returns.mean() - exact) < 3 * se, (returns.mean(), exact, se)
