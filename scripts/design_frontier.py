"""Exact, data-free 'phase diagram' of the breaking point (Figure 0).

For a grid of treatment-side strength lam (-> Gamma*) and outcome-side strength kappa_p in R1,
compute exactly the infinite-data nominal advantage of 'withhold antibiotics' over the
clinicians and its true advantage. Where the nominal advantage is positive while the true one is
negative, every nominal estimator converges to the wrong decision.

    python scripts/design_frontier.py --out figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from ope_conf.config import Config  # noqa: E402
from ope_conf.run import fixed_policies  # noqa: E402
from ope_conf.sim.confounding import build_world  # noqa: E402
from ope_conf.truth import gamma_star, nominal_limit, observational_kernels, true_value, true_value_clinician  # noqa: E402

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
# diverging: blue (negative: OPE says keep the clinicians) <-> gray midpoint <-> red (says deploy)
DIVERGING = LinearSegmentedColormap.from_list(
    "bluered", ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f1a6a5", "#e34948", "#9c2423"])


def compute(lams, kps):
    rows = []
    for kp in kps:
        for lam in lams:
            world = build_world(Config(regime="R1", lam=float(lam), kappa_p=float(kp)))
            pol = fixed_policies(world)["withhold"]
            v_clin = true_value_clinician(world)
            rows.append({"lam": lam, "kappa_p": kp, "gamma_star": gamma_star(world)["msm_action"],
                         "true_adv": true_value(world, pol) - v_clin,
                         "nominal_adv": nominal_limit(world, pol, observational_kernels(world)) - v_clin})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lams, kps = np.round(np.linspace(0, 3, 13), 3), np.round(np.linspace(0, 1, 11), 3)
    df = compute(lams, kps)
    df.to_csv(out / "design_frontier.csv", index=False)
    Z = df.pivot(index="kappa_p", columns="lam", values="nominal_adv").to_numpy()
    T = df.pivot(index="kappa_p", columns="lam", values="true_adv").to_numpy()
    g = df.groupby("lam").gamma_star.first().to_numpy()

    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "font.size": 9.5, "axes.edgecolor": "#8a8984", "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    lim = float(np.nanmax(np.abs(Z)))
    mesh = ax.pcolormesh(lams, kps, Z, cmap=DIVERGING, norm=TwoSlopeNorm(0.0, -lim, lim), shading="gouraud")
    cs = ax.contour(lams, kps, Z, levels=[0.0], colors=INK, linewidths=2)
    ax.clabel(cs, fmt={0.0: "breaking point"}, fontsize=8.5)
    ax.set_xticks(lams[::2], [f"{lm:g}\n({gs:.3g})" for lm, gs in zip(lams[::2], g[::2])])
    ax.set_xlabel("treatment side: λ  (Γ* in brackets)")
    ax.set_ylabel("outcome side: prognostic channel κp")
    cb = fig.colorbar(mesh, ax=ax, pad=0.02)
    cb.set_label("infinite-data nominal advantage\nof 'withhold antibiotics' over clinicians")
    ax.set_title(f"R1, exact: right of the line, every nominal estimator recommends withholding.\n"
                 f"True advantage is negative everywhere ({T.max():+.2f} to {T.min():+.2f}).",
                 loc="left", fontsize=9.5)
    fig.savefig(out / "fig0_frontier.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / "fig0_frontier.pdf", bbox_inches="tight")
    print(f"wrote {out / 'fig0_frontier.png'}; true advantage range {T.min():+.3f} .. {T.max():+.3f}")


if __name__ == "__main__":
    main()
