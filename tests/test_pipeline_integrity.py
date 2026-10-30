"""Guards against silent failures: hidden columns reaching estimators, invalid returns, reused seeds."""

import inspect

import numpy as np

from ope_conf import bounds, estimators, learn
from ope_conf.config import Config, expand_sweep
from ope_conf.data import Episodes, Hidden, generate
from ope_conf.estimators import Fit, mb_values, nominal_estimates
from ope_conf.run import fixed_policies
from ope_conf.sim.confounding import build_world

HIDDEN_FIELDS = {"h", "pb", "e_true", "u", "hidden", "pi_b"}


def test_hidden_columns_never_reach_estimators():
    """Estimators take Episodes, which has no hidden field; nothing accepts a Hidden object."""
    assert not HIDDEN_FIELDS & set(Episodes.__dataclass_fields__)
    for module in (estimators, bounds, learn):
        for name, fn in inspect.getmembers(module, inspect.isfunction):
            params = set(inspect.signature(fn).parameters)
            assert not params & HIDDEN_FIELDS, f"{module.__name__}.{name} takes a hidden column"
            assert "Hidden" not in str(inspect.signature(fn)), f"{module.__name__}.{name}"


def test_returns_and_values_are_valid():
    world = build_world(Config(regime="R2", lam=2.0, kappa_p=0.3))
    ep, hid = generate(world, 3000, np.random.default_rng(60))
    G = ep.returns()
    assert set(np.unique(G)) <= {-1.0, 0.0, 1.0}
    assert np.all(ep.done.sum(axis=1) <= 1)                       # at most one terminal step
    assert np.all((ep.alive.sum(axis=1) == world.H) | ep.done.any(axis=1))
    assert np.all(hid.pb[ep.alive] > 0) and np.all(hid.e_true[ep.alive] > 0)
    fit = Fit(ep, world.est_phase_of_t)
    for pi in fixed_policies(world).values():
        _, V = mb_values(fit, pi, world.rec_reward, world.rec_terminal)
        assert np.all((V >= -1 - 1e-12) & (V <= 1 + 1e-12))


def test_no_two_datasets_share_a_seed():
    _, reps, cells = expand_sweep("configs/sweep_main.yaml")
    _, reps_h, cells_h = expand_sweep("configs/sweep_horizon.yaml")
    seeds = [int(c.seed_sequence(r).generate_state(1)[0]) for c in cells for r in range(reps)]
    seeds += [int(c.seed_sequence(r).generate_state(1)[0]) for c in cells_h for r in range(reps_h)]
    assert len(seeds) == len(set(seeds))


def test_true_propensity_pdis_is_unbiased_without_confounding():
    """With the exact propensities, PDIS of the protocol is unbiased at lambda = 0: the data
    generator, the truth and the estimator agree (an oracle check; estimators never see e_true)."""
    from ope_conf.truth import true_value
    world = build_world(Config(regime="R1", lam=0.0))
    pi = world.protocol
    rng = np.random.default_rng(61)
    errs = []
    for _ in range(40):
        ep, hid = generate(world, 2000, rng)
        t = np.arange(world.H)[None, :]
        ratio = np.where(ep.alive, pi[t, ep.s, ep.a] / hid.e_true, 1.0)
        errs.append(float(np.mean((np.cumprod(ratio, axis=1) * ep.r).sum(axis=1))))
    errs = np.array(errs) - true_value(world, pi)
    assert abs(errs.mean()) < 3 * errs.std() / np.sqrt(len(errs))


def test_bounds_collapse_at_gamma_one_and_order():
    """Every estimator-level bound equals its own nominal value at Gamma = 1 and widens with Gamma."""
    world = build_world(Config(regime="R1", lam=1.0, kappa_p=0.5))
    ep, _ = generate(world, 1500, np.random.default_rng(62))
    fit = Fit(ep, world.est_phase_of_t)
    pool, n_pool = bounds.glucose_pool_map(world.n_rec)
    ctx = {"confounded_t": world.confounded_t, "rec_reward": world.rec_reward,
           "rec_terminal": world.rec_terminal, "min_cell": 20,
           "cells": bounds.StartCells(ep, fit, pool, n_pool, 0.0625)}
    for pi in fixed_policies(world).values():
        est = nominal_estimates(ep, fit, pi, world.rec_reward, world.rec_terminal)
        V1 = est["_V"][1]
        assert abs(bounds.bound_value("robust_dp", ep, fit, pi, 1.0, ctx, True, V1) - est["mb"]) < 1e-10
        assert abs(bounds.bound_value("box", ep, fit, pi, 1.0, ctx, True, V1) - est["pdis"]) < 1e-10
        for m in bounds.BOUND_METHODS:
            lo = [bounds.bound_value(m, ep, fit, pi, g, ctx, True, V1) for g in (1.0, 1.5, 3.0, 10.0)]
            hi = [bounds.bound_value(m, ep, fit, pi, g, ctx, False, V1) for g in (1.0, 1.5, 3.0, 10.0)]
            assert abs(lo[0] - hi[0]) < 1e-10, m
            assert np.all(np.diff(lo) <= 1e-10) and np.all(np.diff(hi) >= -1e-10), m
