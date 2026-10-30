"""Where does WIS's bias at lam = 0 come from: the estimated propensities or the formula?

Same datasets, same WIS formula; only the propensity differs: the count-based e_hat the
estimators use vs the exact nominal propensity e_true, which no analyst could know. R1,
lam = 0 (no confounding), n = 2,000, 30 datasets per kappa_p.

    python scripts/oracle_wis.py --out figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ope_conf.config import Config
from ope_conf.data import generate
from ope_conf.estimators import Fit, step_ratios
from ope_conf.run import fixed_policies
from ope_conf.sim.confounding import build_world
from ope_conf.truth import true_value


def wis(ep, ratio):
    rho = np.cumprod(ratio, axis=1)[:, -1]
    return float((rho * ep.r.sum(axis=1)).sum() / rho.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    ap.add_argument("--datasets", type=int, default=30)
    args = ap.parse_args()
    rows = []
    for kp in (0.0, 1.0):
        cfg = Config(regime="R1", lam=0.0, kappa_p=kp)
        world = build_world(cfg)
        pols = fixed_policies(world)
        rng = np.random.default_rng(7)
        t = np.arange(world.H)[None, :]
        for k in range(args.datasets):
            ep, hid = generate(world, 2000, rng)
            fit = Fit(ep, world.est_phase_of_t, None, cfg.smoothing)
            for p in ("protocol", "withhold"):
                pi = pols[p]
                exact = np.where(ep.alive, pi[t, ep.s, ep.a] / np.where(ep.alive, hid.e_true, 1.0), 1.0)
                rows.append({"kappa_p": kp, "policy": p, "dataset": k, "v_true": true_value(world, pi),
                             "wis_estimated_e": wis(ep, step_ratios(ep, fit, pi)), "wis_exact_e": wis(ep, exact)})
    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    g = df.assign(est=df.wis_estimated_e - df.v_true, exact=df.wis_exact_e - df.v_true).groupby(["kappa_p", "policy"])
    tab = pd.DataFrame({"bias_estimated_e": g.est.mean(), "se_estimated_e": g.est.std() / np.sqrt(g.est.count()),
                        "bias_exact_e": g.exact.mean(), "se_exact_e": g.exact.std() / np.sqrt(g.exact.count())})
    tab.round(4).to_csv(out / "oracle_wis.csv")
    print(tab.round(3).to_string())


if __name__ == "__main__":
    main()
