"""Gamma* is computed exactly from the clinician we wrote: lambda = 0 gives 1, and the binary
Rosenbaum Gamma of the antibiotic decision is exactly e^(2 lambda)."""

import numpy as np
import pytest

from ope_conf.config import Config
from ope_conf.sim.confounding import build_world
from ope_conf.truth import gamma_star, nominal_limit, true_value


@pytest.mark.parametrize("regime", ["R1", "R2"])
def test_gamma_star_lambda_zero_and_rosenbaum(regime):
    g0 = gamma_star(build_world(Config(regime=regime, lam=0.0)))
    assert all(abs(v - 1.0) < 1e-12 for v in g0.values())
    for lam in (0.5, 1.5, 3.0):
        g = gamma_star(build_world(Config(regime=regime, lam=lam)))
        assert abs(g["rosenbaum_factor"] / np.exp(2 * lam) - 1) < 1e-9
        # MSM never exceeds Rosenbaum for the binary factor; action-level MSM is smaller still
        assert 1.0 < g["msm_action"] <= g["msm_factor"] + 1e-9 <= g["rosenbaum_factor"] + 2e-9


def test_r3_gamma_star_grows_with_w():
    vals = [gamma_star(build_world(Config(regime="R3", w=w)))["msm_action"] for w in (0.0, 0.5, 1.0)]
    assert abs(vals[0] - 1.0) < 1e-12 and vals[0] < vals[1] < vals[2]


@pytest.mark.parametrize("regime", ["R1", "R2"])
def test_no_confounding_no_asymptotic_bias(regime):
    """At lambda = 0 the observational kernels identify the truth for every policy."""
    world = build_world(Config(regime=regime, lam=0.0, kappa_p=0.3))
    rng = np.random.default_rng(5)
    for _ in range(3):
        pi = rng.dirichlet(np.ones(8), size=(world.H, world.n_rec))
        assert abs(nominal_limit(world, pi) - true_value(world, pi)) < 1e-10


def test_values_in_range_and_effect_size():
    """Values lie in [-1, 1]; the admission antibiotic gap clears the 0.05 effect-size check."""
    from ope_conf.run import fixed_policies
    world = build_world(Config(regime="R1"))
    pols = fixed_policies(world)
    vals = {k: true_value(world, pi) for k, pi in pols.items()}
    assert all(-1 <= v <= 1 for v in vals.values())
    assert vals["always"] - vals["withhold"] > 0.05
