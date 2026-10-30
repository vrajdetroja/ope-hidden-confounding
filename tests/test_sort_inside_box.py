"""The sharp sort bound always lies inside the naive box bound (it adds the mean-1 constraint)."""

import numpy as np

from ope_conf.msm import cell_lower_bound, cell_upper_bound, omega_box


def box_cell(z, e, gamma, lower=True):
    """Namkoong et al.'s Lemma 2 in MSM form: w_hi on negative outcomes, w_lo on positive ones."""
    w_lo, w_hi = omega_box(e, gamma)
    if lower:
        w = np.where(z < 0, w_hi, w_lo)
    else:
        w = np.where(z > 0, w_hi, w_lo)
    return float(np.mean(w * z))


def test_sort_inside_box():
    rng = np.random.default_rng(20)
    for _ in range(2000):
        z = rng.choice([-1.0, 0.0, 1.0], size=int(rng.integers(1, 60)))
        e = float(rng.uniform(0.01, 0.99))
        gamma = float(np.exp(rng.uniform(0, np.log(20))))
        assert box_cell(z, e, gamma, True) <= cell_lower_bound(z, e, gamma) + 1e-12
        assert cell_upper_bound(z, e, gamma) <= box_cell(z, e, gamma, False) + 1e-12
