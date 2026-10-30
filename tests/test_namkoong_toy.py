"""Namkoong et al. (2020, Section 4): always-treat and never-treat are truly equal, but nominal OPE
rates always-treat higher even with infinite data, and the gap grows with the number of decisions."""

import numpy as np

from ope_conf.toy import namkoong_toy, namkoong_toy_sample


def test_infinite_data_ranking_is_wrong_and_gap_grows():
    for gamma in (1.5, 2.0, 4.0):
        prev_is, prev_wis = 0.0, 0.0
        for T in range(1, 11):
            out = namkoong_toy(gamma, T)
            assert out["true_treat"] == out["true_never"] == 0.5
            gap_is = out["is_treat"] - out["is_never"]
            gap_wis = out["wis_treat"] - out["wis_never"]
            assert gap_is > 0 and gap_wis > 0
            assert gap_is > prev_is and gap_wis > prev_wis
            prev_is, prev_wis = gap_is, gap_wis
        # importance-sampling bias is exponential: ratio per extra decision = 2 sqrt(G) / (1 + sqrt(G))
        r = np.sqrt(gamma)
        a, b = namkoong_toy(gamma, 6)["is_treat"], namkoong_toy(gamma, 5)["is_treat"]
        assert abs(a / b - 2 * r / (1 + r)) < 1e-12


def test_no_confounding_no_gap():
    out = namkoong_toy(1.0, 8)
    assert abs(out["is_treat"] - 0.5) < 1e-12 and abs(out["wis_never"] - 0.5) < 1e-12


def test_gamma_conventions():
    out = namkoong_toy(4.0, 3)
    assert abs(out["gamma_rosenbaum_per_step"] - 4.0) < 1e-12   # their Assumption C level
    assert abs(out["gamma_msm_per_step"] - 2.0) < 1e-12         # the MSM level is sqrt(Gamma)


def test_simulation_matches_closed_form():
    rng = np.random.default_rng(30)
    for T in (1, 3, 5):
        sim = namkoong_toy_sample(2.0, T, 400_000, rng)
        exact = namkoong_toy(2.0, T)
        for key in ("is_treat", "is_never", "wis_treat", "wis_never"):
            assert abs(sim[key] - exact[key]) < 0.02, (T, key, sim[key], exact[key])
