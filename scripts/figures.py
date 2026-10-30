"""Every figure and table is a query on results/: one function per figure.

    python scripts/figures.py --results results/main [--horizon results/horizon] --out figures

A figure whose data are not in the results yet is skipped with a note. summary.json collects
the numbers quoted in README.md and REPORT.md, so the text never drifts from the data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import NullLocator  # noqa: E402

from ope_conf.config import load_config  # noqa: E402

# Reference palette (dataviz skill, light mode), categorical slots in fixed order.
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CRITICAL, GOOD = "#d03b3b", "#0ca30c"
EST_ORDER = ["wis", "mb", "wdr", "pdis", "cwpdis", "mb_death"]
EST_LABEL = {"wis": "WIS", "mb": "MB", "wdr": "WDR", "pdis": "PDIS", "cwpdis": "CWPDIS",
             "mb_death": "MB (unseen = death)"}
BOUND_ORDER = ["sort", "robust_dp", "structured", "box"]
BOUND_LABEL = {"sort": "sort (R1)", "robust_dp": "robust DP (R2)", "structured": "structured (antibiotics only)",
               "box": "naive box"}
MARKERS = ["o", "s", "^", "D", "v", "P"]
FACTOR_KP = (0.0, 0.5, 1.0)  # kappa_p levels drawn in the figures; the 0.3 ablation is in the tables
FIXED = ("protocol", "withhold", "always")
ALARM_BOUND = {"R1": "sort", "R2": "robust_dp", "R3": "robust_dp"}  # primary bound per regime
GAMMA_HI = load_config(Path(__file__).resolve().parents[1] / "configs" / "base.yaml").gamma_hi
GAMMA_LABEL = "true confounding strength  Γ*  (MSM, log scale)"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5,
    "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 2,
    "lines.markersize": 5, "figure.titlesize": 11, "figure.titleweight": "bold",
})

NOMINAL = set(EST_ORDER)


def _save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out / name}.png")


def _gamma_axis(ax, gammas=None):
    """Log Gamma* axis with a tick at every grid value; the shared label goes on the figure."""
    ax.set_xscale("log")
    if gammas is not None:
        gammas = sorted(gammas)
        ax.set_xticks(gammas, [f"{g:.2g}" if g < 10 else f"{g:.0f}" for g in gammas])
        ax.xaxis.set_minor_locator(NullLocator())


def _pick_n(df, preferred=2000):
    ns = sorted(df["n"].unique())
    return preferred if preferred in ns else ns[-1]


def deploy_wrong(d: pd.DataFrame) -> pd.Series:
    """OPE says deploy (estimate beats the observed clinician mean) but the policy is truly worse."""
    return (d["estimate"] > d["v_hat_clinician"]) & (d["v_true"] < d["v_true_clinician"])


def wilson(k, m, z=1.96):
    """95% Wilson interval for a share k / m."""
    p = k / m
    den = 1 + z * z / m
    mid = (p + z * z / (2 * m)) / den
    hw = z * np.sqrt(p * (1 - p) / m + z * z / (4 * m * m)) / den
    return mid - hw, mid + hw


def wrong_rates(e: pd.DataFrame) -> pd.DataFrame:
    """Per lam: Gamma*, share of datasets that deploy a truly worse policy, and its Wilson interval."""
    r = e.assign(w=deploy_wrong(e)).groupby("lam").agg(g=("gamma_star", "first"), k=("w", "sum"), m=("w", "size"))
    r["rate"] = r.k / r.m
    r["lo"], r["hi"] = wilson(r.k, r.m)
    return r


# ---------------------------------------------------------------- Figure 1 headline
def fig_headline(df, out, summary, policy="withhold"):
    d_all = df[(df.regime == "R1") & (df.policy == policy) & (df.kappa == 0.5)]
    if d_all.empty:
        return print("  skip headline: no R1 rows")
    d = d_all[d_all.kappa_p.isin(FACTOR_KP)]
    n, n_big = _pick_n(d), d.n.max()
    kps = sorted(d.kappa_p.unique())
    fig, axes = plt.subplots(2, len(kps), figsize=(3.6 * len(kps), 6.0), sharex=True, sharey="row", squeeze=False,
                             layout="constrained")
    for j, kp in enumerate(kps):
        ax = axes[0][j]
        e = d[(d.kappa_p == kp) & (d.n == n)]
        g = e.groupby("lam")
        x = g.gamma_star.first()
        ax.axhline(0, color=INK2, lw=1)
        true_adv = (g.v_true.first() - g.v_true_clinician.first())
        lim_adv = (g.v_limit.first() - g.v_true_clinician.first())
        for est, col, mk in (("wis", SLOT[0], "o"), ("mb", SLOT[1], "s")):
            s = e[e.method == est].assign(adv=lambda t: t.estimate - t.v_hat_clinician).groupby("lam").adv
            ax.fill_between(x, s.quantile(0.025), s.quantile(0.975), color=col, alpha=0.13, lw=0)
            ax.plot(x, s.mean(), color=col, marker=mk, label=f"nominal {EST_LABEL[est]} (mean, 95% of datasets)")
        b = e[e.method == "sort"].assign(lo=lambda t: t.bound_lo - t.v_hat_clinician,
                                         hi=lambda t: t.bound_hi - t.v_hat_clinician).groupby("lam")
        ax.vlines(x * 1.04, b.lo.median(), b.hi.median(), color=INK2, lw=3, alpha=0.55,
                  label="sort bound at Γ = Γ* (median)")
        ax.plot(x, lim_adv, color=MUTED, ls=":", lw=1.6, label="nominal, infinite data (exact)")
        ax.plot(x, true_adv, color=INK, lw=2.2, label="true advantage")
        ax.set_title(f"prognostic channel κp = {kp:g}")
        # bottom row: the decision itself, at n = 2,000 (with 95% CI) and at the largest n
        ax = axes[1][j]
        ax.axhline(0.5, color=INK2, lw=1, ls="--", label="half the datasets")
        for est, col, mk in (("wis", SLOT[0], "o"), ("mb", SLOT[1], "s")):
            for nn, ls in [(n, "-")] + ([(n_big, "--")] if n_big != n else []):
                f = d[(d.kappa_p == kp) & (d.n == nn) & (d.method == est)]
                if f.empty:
                    continue
                r = wrong_rates(f)
                if nn == n:
                    ax.fill_between(r.g, r.lo, r.hi, color=col, alpha=0.13, lw=0)
                ax.plot(r.g, r.rate, color=col, marker=mk, ls=ls, mfc=col if nn == n else SURFACE,
                        label=f"{EST_LABEL[est]}, n = {nn:,}" + (" (95% CI)" if nn == n else ""))
        ax.set_ylim(-0.03, 1.03)
        _gamma_axis(ax, x)
    axes[0][0].set_ylabel(f"advantage of '{policy} antibiotics'\nover the clinicians (value)")
    axes[1][0].set_ylabel("share of datasets that deploy\nthe truly worse policy")
    axes[1][len(kps) // 2].set_xlabel(GAMMA_LABEL)
    h0, l0 = axes[0][0].get_legend_handles_labels()
    h1, l1 = axes[1][0].get_legend_handles_labels()
    fig.legend(h0 + h1, l0 + l1, loc="outside lower center", ncol=4, fontsize=8.5)
    fig.suptitle(f"R1: nominal OPE vs the truth as hidden confounding grows (top: n = {n:,})", x=0.01, ha="left")
    _save(fig, out, "fig1_headline")
    # summary: per kappa_p and n, the wrong-deploy rate with its interval, and the breaking point
    # (smallest Gamma* at which more than half the datasets deploy a truly worse policy)
    for est in ("wis", "mb"):
        for (kp, nn), e in d_all[d_all.method == est].groupby(["kappa_p", "n"]):
            r = wrong_rates(e)
            flip = r[r.rate > 0.5]
            summary[f"headline_{policy}_{est}_kp{kp:g}_n{nn}"] = {
                "first_gamma_majority_wrong": float(flip.g.min()) if len(flip) else None,
                "wrong_rate_by_gamma": {f"{t.g:.3g}": [round(float(t.rate), 3), round(float(t.lo), 3),
                                                       round(float(t.hi), 3)] for t in r.itertuples()}}
    for kp, e in d_all.groupby("kappa_p"):
        g = e.groupby("lam").agg(g=("gamma_star", "first"), lim=("v_limit", "first"), true=("v_true", "first"),
                                 clin=("v_true_clinician", "first"))
        pos = g[g.lim > g.clin]
        summary[f"headline_{policy}_exact_kp{kp:g}"] = {
            "first_gamma_nominal_limit_says_deploy": float(pos.g.min()) if len(pos) else None,
            "true_advantage_by_gamma": {f"{t.g:.3g}": round(float(t.true - t.clin), 3) for t in g.itertuples()},
            "nominal_limit_advantage_by_gamma": {f"{t.g:.3g}": round(float(t.lim - t.clin), 3) for t in g.itertuples()}}


# ---------------------------------------------------------------- Figure 2 bias curves
def fig_bias(df, out, summary, policy="withhold", kappa_p=None):
    d = df[(df.regime == "R1") & (df.policy == policy) & (df.method.isin(EST_ORDER)) & (df.kappa == 0.5)]
    if d.empty:
        return print("  skip bias: no rows")
    kappa_p = max(d.kappa_p.unique()) if kappa_p is None else kappa_p
    d = d[d.kappa_p == kappa_p].assign(err=lambda t: t.estimate - t.v_true)
    ns = sorted(d.n.unique())
    fig, axes = plt.subplots(1, len(ns), figsize=(3.5 * len(ns), 3.5), sharey=True, squeeze=False,
                             layout="constrained")
    for ax, n in zip(axes[0], ns):
        e = d[d.n == n]
        x = e.groupby("lam").gamma_star.first()
        sd0 = e[(e.lam == 0) & (e.method == "wis")].err.std()
        if np.isfinite(sd0):
            ax.axhspan(-2 * sd0, 2 * sd0, color=GRID, alpha=0.45, lw=0, label="±2 SD of WIS at Γ* = 1")
            ax.text(0.03, 0.03, f"band: ±{2 * sd0:.2f}", transform=ax.transAxes, color=INK2, fontsize=8)
        ax.axhline(0, color=INK2, lw=1)
        lim = e.groupby("lam").apply(lambda t: (t.v_limit - t.v_true).iloc[0], include_groups=False)
        ax.plot(x, lim, color=INK, ls=":", lw=1.8, label="asymptotic bias (exact, n → ∞)")
        for i, est in enumerate(EST_ORDER):
            s = e[e.method == est].groupby("lam").err
            ax.plot(x, s.mean(), color=SLOT[i], marker=MARKERS[i], lw=1.6, label=EST_LABEL[est],
                    ls="--" if est == "mb_death" else "-")
        _gamma_axis(ax, x)
        ax.set_title(f"n = {n:,}")
    axes[0][0].set_ylabel(f"bias of $\\hat{{V}}$('{policy}')  (mean over datasets)")
    axes[0][-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8.5)
    fig.supxlabel(GAMMA_LABEL, color=INK2, fontsize=9.5)
    fig.suptitle(f"R1, κp = {kappa_p:g}: estimator bias as confounding grows", x=0.01, ha="left")
    _save(fig, out, "fig2_bias")


# ---------------------------------------------------------------- Figure 3 bounds
def bounds_frame(df):
    d = df[df.method.isin(BOUND_ORDER)]
    return d.assign(cover=(d.bound_lo <= d.v_true) & (d.v_true <= d.bound_hi), width=d.bound_hi - d.bound_lo)


def fig_bounds(df, out, summary, policies=FIXED):
    d = df[(df.regime == "R1") & (df.method.isin(BOUND_ORDER)) & (df.policy.isin(policies)) & (df.kappa == 0.5)
           & df.kappa_p.isin(FACTOR_KP)]
    if d.empty:
        return print("  skip bounds: no rows")
    n = _pick_n(d)
    d = bounds_frame(d[d.n == n])
    kps = sorted(d.kappa_p.unique())
    fig, axes = plt.subplots(2, len(kps), figsize=(3.5 * len(kps), 5.4), sharex=True, sharey="row", squeeze=False,
                             layout="constrained")
    for j, kp in enumerate(kps):
        e = d[d.kappa_p == kp]
        x = e.groupby("lam").gamma_star.first()
        for i, m in enumerate(BOUND_ORDER):
            s = e[e.method == m].groupby("lam")
            if s.ngroups == 0:
                continue
            axes[0][j].plot(x, s.cover.mean(), color=SLOT[i], marker=MARKERS[i], label=BOUND_LABEL[m])
            axes[1][j].plot(x, s.width.median(), color=SLOT[i], marker=MARKERS[i], label=BOUND_LABEL[m])
        axes[0][j].axhline(0.95, color=INK2, lw=1, ls="--")
        axes[0][j].set_ylim(-0.03, 1.03)
        axes[0][j].set_title(f"κp = {kp:g}")
        axes[1][j].set_yscale("log")
        _gamma_axis(axes[1][j], x)
    axes[0][0].set_ylabel("coverage of the truth\n(share of datasets)")
    axes[1][0].set_ylabel("width (median, log)")
    axes[0][-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8.5)
    fig.supxlabel(GAMMA_LABEL, color=INK2, fontsize=9.5)
    fig.suptitle(f"R1, n = {n:,}: sensitivity intervals at Γ = Γ* ({', '.join(policies)})", x=0.01, ha="left")
    _save(fig, out, "fig3_bounds")


def bounds_summary(df, summary):
    """Bound numbers for the text: every regime and n, the fixed policies, the main grid (H = 20)."""
    d = bounds_frame(df[(df.kappa == 0.5) & (df.H == 20) & df.kappa_p.isin(FACTOR_KP) & df.policy.isin(FIXED)])
    for (reg, m, n), s in d.groupby(["regime", "method", "n"]):
        unconf = np.isclose(s.gamma_star, 1.0)
        summary[f"bound_{reg}_{m}_n{n}"] = {
            "coverage": round(float(s.cover.mean()), 3), "median_width": round(float(s.width.median()), 3),
            "coverage_at_gamma1": round(float(s[unconf].cover.mean()), 3),
            "coverage_above_gamma1": round(float(s[~unconf].cover.mean()), 3),
            "coverage_by_policy": {p: round(float(c), 3) for p, c in s.groupby("policy").cover.mean().items()}}


# ---------------------------------------------------------------- Figure 4 horizon
def fig_horizon(df, out, summary):
    # the horizon sweep's cells (n = 2,000, base kappa and kappa_p) and their H = 20 twins in the main grid
    d = df[df.method.isin(["robust_dp", "sort"]) & df.regime.isin(["R2", "R3"]) & (df.policy == "protocol")
           & (df.n == 2000) & (df.kappa == 0.5) & (df.kappa_p == 0.0)]
    if d.empty or d.H.nunique() < 2:
        return print("  skip horizon: needs the horizon sweep (R2, R3 at several H)")
    d = bounds_frame(d)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3), layout="constrained")
    for i, (reg, m) in enumerate((("R2", "robust_dp"), ("R3", "robust_dp"), ("R3", "sort"))):
        s = d[(d.regime == reg) & (d.method == m)].groupby("H")
        if s.ngroups == 0:
            continue
        lab = f"{BOUND_LABEL[m].split(' (')[0]} in {reg}"
        axes[0].plot(s.cover.mean().index, s.cover.mean(), color=SLOT[i], marker=MARKERS[i], label=lab)
        axes[1].plot(s.width.median().index, s.width.median(), color=SLOT[i], marker=MARKERS[i], label=lab)
        summary[f"horizon_{reg}_{m}"] = {
            f"H{h}": {"coverage": round(float(t.cover.mean()), 3),
                      "coverage_above_gamma1": round(float(t[~np.isclose(t.gamma_star, 1.0)].cover.mean()), 3),
                      "median_width": round(float(t.width.median()), 3)} for h, t in s}
    axes[0].axhline(0.95, color=INK2, lw=1, ls="--")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("coverage of the truth")
    axes[1].set_ylabel("median width")
    for ax in axes:
        ax.set_xlabel("horizon H")
        ax.set_xticks(sorted(d.H.unique()))
    axes[1].legend(fontsize=8.5)
    fig.suptitle("R2 (fresh confounder) vs R3 (persistent): protocol, n = 2,000", x=0.01, ha="left")
    _save(fig, out, "fig4_horizon")


# ---------------------------------------------------------------- Figure 5 alarm
def alarm_table(df, regime="R1", method="sort"):
    """One row per (dataset, policy) where the bound's own nominal value says deploy."""
    d = df[(df.regime == regime) & (df.method == method) & df.breakdown_gamma.notna()].copy()
    d["wrong"] = d.v_true < d.v_true_clinician
    d["alarm"] = d.breakdown_gamma < d.benchmark_gamma
    d["oracle_alarm"] = d.breakdown_gamma < d.gamma_star
    return d


def _pr(d, col):
    tp = int((d[col] & d.wrong).sum())
    return {"precision": round(tp / max(int(d[col].sum()), 1), 3), "recall": round(tp / max(int(d.wrong.sum()), 1), 3),
            "alarms": int(d[col].sum()), "wrong": int(d.wrong.sum()), "decisions": int(len(d)),
            "base_rate": round(float(d.wrong.mean()), 3)}


def fig_alarm(df, out, summary):
    tabs = {r: alarm_table(df, r, m) for r, m in ALARM_BOUND.items()}
    tabs = {r: d for r, d in tabs.items() if not d.empty}
    if not tabs:
        return print("  skip alarm: no deploy decisions")
    fig, axes = plt.subplots(1, len(tabs), figsize=(3.9 * len(tabs), 4.0), sharey=True, squeeze=False,
                             layout="constrained")
    for ax, (reg, d) in zip(axes[0], tabs.items()):
        m = ALARM_BOUND[reg]
        for wrong, col, mk, lab in ((False, SLOT[0], "o", "deploy was right"), (True, CRITICAL, "X", "deploy was wrong")):
            e = d[d.wrong == wrong]
            ax.scatter(e.benchmark_gamma, e.breakdown_gamma, s=10, c=col, marker=mk, alpha=0.3, lw=0, label=lab)
        lo, hi = 1.0, float(d.benchmark_gamma.max()) * 1.2
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=1)
        ax.axhline(GAMMA_HI, color=MUTED, lw=1, ls="--")
        ax.text(lo * 1.1, GAMMA_HI * 1.08, f"search cap Γ = {GAMMA_HI:g}", color=INK2, fontsize=8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(0.9, 2 * GAMMA_HI)
        ax.set_xlabel("benchmark $\\hat{\\Gamma}$ (strongest recorded variable)")
        pr = _pr(d, "alarm")
        ax.set_title(f"{reg}, {BOUND_LABEL[m].split(' (')[0]} bound\n{pr['wrong']:,} of {pr['decisions']:,} deploys "
                     f"wrong\nalarm precision {pr['precision']:.2f} (base rate {pr['base_rate']:.2f}), "
                     f"recall {pr['recall']:.2f}", loc="left", fontsize=9.5)
        summary[f"alarm_{reg}_{m}"] = pr
        summary[f"oracle_alarm_{reg}_{m}"] = _pr(d, "oracle_alarm")
        summary[f"alarm_{reg}_{m}_by_policy"] = {
            p: {"deploys": int(len(s)), "wrong_share": round(float(s.wrong.mean()), 3),
                "median_breakdown": round(float(s.breakdown_gamma.median()), 3),
                "breakdown_at_cap": round(float((s.breakdown_gamma >= GAMMA_HI - 1e-6).mean()), 3),
                "median_benchmark": round(float(s.benchmark_gamma.median()), 3)}
            for p, s in d.groupby("policy")}
    axes[0][0].set_ylabel("breakdown Γ of the deploy decision")
    leg = fig.legend(*axes[0][0].get_legend_handles_labels(), loc="outside lower center", ncol=2, fontsize=8.5,
                     markerscale=1.5)
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    fig.suptitle("Does the alarm flag the wrong deploy decisions? One point per decision; the alarm fires below "
                 "the diagonal (breakdown Γ < $\\hat{\\Gamma}$)", x=0.01, ha="left")
    _save(fig, out, "fig5_alarm")


# ---------------------------------------------------------------- Figure 6 disentangling
def fig_disentangle(df, out, summary):
    d = df[(df.regime == "R1") & df.policy.isin(["rl", "rl_split"]) & df.method.isin(["wis", "mb"]) & (df.kappa == 0.5)]
    if d.empty:
        return print("  skip disentangle: no RL rows")
    n = _pick_n(d)
    lam_hi = d.lam.max()
    d = d[(d.n == n) & d.lam.isin([0.0, lam_hi])]
    kp = d.kappa_p.max()
    d = d[d.kappa_p == kp]
    deploy = d.estimate > d.v_hat_clinician
    best = np.maximum(d.v_true, d.v_true_clinician)
    chosen = np.where(deploy, d.v_true, d.v_true_clinician)
    d = d.assign(regret=best - chosen)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=True, layout="constrained")
    for ax, est in zip(axes, ("wis", "mb")):
        e = d[d.method == est].groupby(["policy", "lam"]).regret
        m, se = e.mean(), e.std() / np.sqrt(e.count())
        xs = np.arange(2)
        for i, pol in enumerate(("rl", "rl_split")):
            vals = [m.get((pol, lam), np.nan) for lam in (0.0, lam_hi)]
            errs = [se.get((pol, lam), np.nan) for lam in (0.0, lam_hi)]
            ax.bar(xs + (i - 0.5) * 0.36, vals, width=0.34, yerr=1.96 * np.array(errs), color=SLOT[i],
                   label="RL learned on the same data" if pol == "rl" else "RL learned on a split",
                   error_kw={"ecolor": INK2, "lw": 1})
        ax.set_xticks(xs, ["no confounding\n(λ = 0)", f"confounding\n(λ = {lam_hi:g})"])
        ax.set_title(f"decision rule: {EST_LABEL[est]}", loc="left")
        summary[f"regret_R1_n{n}_kp{kp:g}_{est}"] = {f"{pol}_lam{lam:g}": round(float(v), 3)
                                                     for (pol, lam), v in m.items()}
    axes[0].set_ylabel("regret of the deploy decision\n(true value lost, mean ± 95% CI)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=2, fontsize=8.5)
    fig.suptitle(f"R1, n = {n:,}, κp = {kp:g}: overfitting vs confounding", x=0.01, ha="left")
    _save(fig, out, "fig6_disentangle")


# ---------------------------------------------------------------- tables
def coverage_table(df):
    """Bias, RMSE and CI coverage by estimator and cell, for every policy."""
    d = df[df.method.isin(EST_ORDER)].assign(err=lambda t: t.estimate - t.v_true,
                                             cover=lambda t: (t.ci_lo <= t.v_true) & (t.v_true <= t.ci_hi),
                                             wrong=lambda t: deploy_wrong(t))
    keys = ["regime", "kappa", "kappa_p", "lam", "w", "n", "H", "policy", "method"]
    return d.groupby(keys).agg(gamma_star=("gamma_star", "first"), bias=("err", "mean"),
                               rmse=("err", lambda e: float(np.sqrt(np.mean(e ** 2)))),
                               ci_coverage=("cover", "mean"), wrong_rate=("wrong", "mean"),
                               datasets=("replicate", "nunique")).reset_index()


def bounds_table(df):
    """Coverage and width of every bound by cell, for every policy."""
    keys = ["regime", "kappa", "kappa_p", "lam", "w", "n", "H", "policy", "method"]
    return bounds_frame(df).groupby(keys).agg(gamma_star=("gamma_star", "first"), coverage=("cover", "mean"),
                                              median_width=("width", "median"),
                                              datasets=("replicate", "nunique")).reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--horizon", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    df = pd.read_parquet(Path(args.results) / "results.parquet")
    if args.horizon and (Path(args.horizon) / "results.parquet").exists():
        df = pd.concat([df, pd.read_parquet(Path(args.horizon) / "results.parquet")], ignore_index=True)
    out = Path(args.out)
    summary = {"rows": int(len(df)), "cells": int(df.groupby(["regime", "kappa", "kappa_p", "lam", "w", "n", "H"]).ngroups),
               "replicates_per_cell": int(df.replicate.nunique()), "git_sha": sorted(df.git_sha.unique().tolist()),
               "rows_by_git_sha": {k: int(v) for k, v in df.git_sha.value_counts().items()}}
    for f in (fig_headline, fig_bias, fig_bounds, fig_horizon, fig_alarm, fig_disentangle):
        f(df, out, summary)
    bounds_summary(df, summary)
    coverage_table(df).to_csv(out / "table_estimators.csv", index=False)
    bounds_table(df).to_csv(out / "table_bounds.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"  wrote {out / 'summary.json'}, table_estimators.csv and table_bounds.csv")


if __name__ == "__main__":
    main()
