"""One cell in, one block of result rows out.

A row is one (replicate, policy, method). Nominal methods fill estimate / ci_lo / ci_hi;
bound methods fill estimate (their value at Gamma = 1), bound_lo / bound_hi (bootstrap 5th
percentile of the lower bound and 95th of the upper bound at Gamma = Gamma*), the plug-in
bounds at Gamma*, and the breakdown Gamma. Every row carries the truth, the exact
infinite-data limit of the nominal estimators, Gamma*, the benchmark Gamma-hat and provenance.
"""

from __future__ import annotations

import functools

import numpy as np

from .alarm import benchmark_gamma, breakdown_gamma
from .bounds import BOUND_METHODS, StartCells, bound_value, glucose_pool_map, sort_bound
from .config import Config, git_sha
from .data import Episodes, generate
from .estimators import NOMINAL, Fit, bootstrap_weights, clinician_estimate, nominal_estimates, unseen_mass
from .learn import learn_policy
from .sim.confounding import World, build_world
from .sim.policies import force_factor, uniform
from .truth import gamma_star, nominal_limit, observational_kernels, true_value, true_value_clinician

FIXED_POLICIES = ("protocol", "withhold", "always", "uniform")
LEARNED_POLICIES = ("rl", "rl_pess", "rl_split")


def bound_methods_for(regime: str) -> tuple:
    """structured is an R1 construction (antibiotic factor at the first decision only)."""
    return BOUND_METHODS if regime == "R1" else ("box", "sort", "robust_dp")


def fixed_policies(world: World) -> dict:
    proto, conf, f = world.protocol, world.confounded_t, world.forced_factor
    withhold, always = proto.copy(), proto.copy()
    withhold[conf] = force_factor(proto[conf], f, 0)
    always[conf] = force_factor(proto[conf], f, 1)
    return {"protocol": proto, "withhold": withhold, "always": always,
            "uniform": uniform(world.H, world.n_rec)}


@functools.lru_cache(maxsize=2)
def cell_context(cfg: Config) -> dict:
    world = build_world(cfg)
    policies = fixed_policies(world)
    k_obs = observational_kernels(world)
    pool, n_pool = glucose_pool_map(world.n_rec)
    gs = gamma_star(world)
    return {
        "cfg": cfg, "world": world, "policies": policies, "k_obs": k_obs, "gamma_star": gs,
        "v_true": {k: true_value(world, pi) for k, pi in policies.items()},
        "v_limit": {k: nominal_limit(world, pi, k_obs) for k, pi in policies.items()},
        "v_true_clinician": true_value_clinician(world),
        "pool": pool, "n_pool": n_pool,
        "gamma_used": {"box": gs["msm_action"], "sort": gs["msm_action"],
                       "robust_dp": gs["msm_action"], "structured": gs["msm_factor"]},
        "methods": bound_methods_for(cfg.regime),
        "git_sha": git_sha(), "config_hash": cfg.hash(),
    }


def _bound_ctx(ctx, ep, fit):
    w = ctx["world"]
    return {"confounded_t": w.confounded_t, "rec_reward": w.rec_reward, "rec_terminal": w.rec_terminal,
            "min_cell": ctx["cfg"].min_cell,
            "cells": StartCells(ep, fit, ctx["pool"], ctx["n_pool"], ctx["cfg"].smoothing)}


def evaluate(ep: Episodes, policies: dict, ctx: dict, w=None, with_bounds=True) -> dict:
    """Every nominal estimate and every bound at Gamma* for one set of episode weights."""
    world, cfg = ctx["world"], ctx["cfg"]
    fit = Fit(ep, world.est_phase_of_t, w, cfg.smoothing)
    bctx = _bound_ctx(ctx, ep, fit)
    out = {("clinician", "mean"): clinician_estimate(ep, fit.w)}
    for name, pi in policies.items():
        est = nominal_estimates(ep, fit, pi, world.rec_reward, world.rec_terminal, cfg.gamma)
        for k in NOMINAL:
            out[(name, k)] = est[k]
        if with_bounds:
            V1 = est["_V"][1]
            for m in ctx["methods"]:
                g = ctx["gamma_used"][m]
                out[(name, m, "lo")] = bound_value(m, ep, fit, pi, g, bctx, True, V1)
                out[(name, m, "hi")] = bound_value(m, ep, fit, pi, g, bctx, False, V1)
    return out


def _point_extras(ep: Episodes, policies: dict, ctx: dict, v_clin: float) -> dict:
    """Unbootstrapped extras on the full data: bound values at Gamma = 1, breakdown Gamma,
    and diagnostics."""
    world, cfg = ctx["world"], ctx["cfg"]
    fit = Fit(ep, world.est_phase_of_t, None, cfg.smoothing)
    bctx = _bound_ctx(ctx, ep, fit)
    out = {}
    for name, pi in policies.items():
        est = nominal_estimates(ep, fit, pi, world.rec_reward, world.rec_terminal, cfg.gamma)
        V1 = est["_V"][1]
        out[(name, "ess")] = est["_ess"]
        out[(name, "unseen_mass")] = unseen_mass(fit, pi, world.rec_terminal)
        out[(name, "pooled_share")] = sort_bound(ep, fit, pi, 1.0, V1, bctx["cells"], cfg.min_cell,
                                                 return_diag=True)[1]["pooled_share"]
        for m in ctx["methods"]:
            lower_at = functools.partial(bound_value, m, ep, fit, pi, ctx=bctx, lower=True, V1=V1)
            out[(name, m, "at1")] = lower_at(1.0)
            out[(name, m, "breakdown")] = breakdown_gamma(lower_at, v_clin, 1.0, cfg.gamma_hi, cfg.gamma_tol)
    return out


def _bootstrap(ep, policies, ctx, rng):
    cfg = ctx["cfg"]
    draws = [evaluate(ep, policies, ctx, w) for w in bootstrap_weights(ep.n, cfg.n_boot, rng)]
    keys = draws[0].keys()
    return {k: np.array([d[k] for d in draws]) for k in keys}


def run_replicate(cfg: Config, replicate: int) -> list[dict]:
    ctx = cell_context(cfg)
    world = ctx["world"]
    ss = cfg.seed_sequence(replicate)
    seed = int(ss.generate_state(1)[0])
    rng_data, rng_split, rng_boot, rng_boot_b = (np.random.default_rng(s) for s in ss.spawn(4))
    ep, _hidden = generate(world, cfg.n, rng_data)          # hidden columns never go further

    # learned policies: same data, pessimistic, and split (learn on half A, evaluate on half B)
    fit_all = Fit(ep, world.est_phase_of_t, None, cfg.smoothing)
    learned = {
        "rl": learn_policy(fit_all, cfg.H, world.rec_reward, world.rec_terminal, 0.0, cfg.soften, cfg.gamma),
        "rl_pess": learn_policy(fit_all, cfg.H, world.rec_reward, world.rec_terminal, cfg.rl_pessimism_c,
                                cfg.soften, cfg.gamma),
    }
    perm = rng_split.permutation(cfg.n)
    half_a, half_b = np.sort(perm[: cfg.n // 2]), np.sort(perm[cfg.n // 2:])
    fit_a = Fit(ep.subset(half_a), world.est_phase_of_t, None, cfg.smoothing)
    learned["rl_split"] = learn_policy(fit_a, cfg.H, world.rec_reward, world.rec_terminal, 0.0, cfg.soften,
                                       cfg.gamma)
    v_true = dict(ctx["v_true"])
    v_limit = dict(ctx["v_limit"])
    for k, pi in learned.items():
        v_true[k] = true_value(world, pi)
        v_limit[k] = nominal_limit(world, pi, ctx["k_obs"])

    # evaluation sets: everything on the full data except rl_split, which uses half B
    sets = {"full": (ep, {**ctx["policies"], "rl": learned["rl"], "rl_pess": learned["rl_pess"]}, rng_boot),
            "half_b": (ep.subset(half_b), {"rl_split": learned["rl_split"]}, rng_boot_b)}
    g_hat, g_hat_per_var = benchmark_gamma(ep, cfg.smoothing, cfg.benchmark_min_count, cfg.benchmark_quantile)
    gs = ctx["gamma_star"]
    base = {"regime": cfg.regime, "lam": cfg.lam, "w": cfg.w, "kappa": cfg.kappa, "kappa_p": cfg.kappa_p,
            "n": cfg.n, "H": cfg.H, "replicate": replicate, "seed": seed,
            "gamma_star": gs["msm_action"], "gamma_star_factor": gs["msm_factor"],
            "gamma_rosenbaum": gs["rosenbaum_factor"], "benchmark_gamma": g_hat,
            "benchmark_var": max(g_hat_per_var, key=lambda k: np.nan_to_num(g_hat_per_var[k], nan=-1)),
            "v_true_clinician": ctx["v_true_clinician"], "git_sha": ctx["git_sha"],
            "config_hash": ctx["config_hash"]}
    rows = []
    for set_name, (data, pols, rng_b) in sets.items():
        point = evaluate(data, pols, ctx)
        boot = _bootstrap(data, pols, ctx, rng_b)
        v_clin = point[("clinician", "mean")]
        extras = _point_extras(data, pols, ctx, v_clin)
        common = {**base, "eval_set": set_name, "n_eval": data.n, "v_hat_clinician": v_clin}
        if set_name == "full":
            lo, hi = np.percentile(boot[("clinician", "mean")], [2.5, 97.5])
            rows.append({**common, "policy": "clinician", "method": "mean", "estimate": v_clin,
                         "ci_lo": lo, "ci_hi": hi, "v_true": ctx["v_true_clinician"],
                         "v_limit": ctx["v_true_clinician"]})
        for name in pols:
            pc = {**common, "policy": name, "v_true": v_true[name], "v_limit": v_limit[name],
                  "ess": extras[(name, "ess")], "unseen_mass": extras[(name, "unseen_mass")],
                  "pooled_share": extras[(name, "pooled_share")]}
            for k in NOMINAL:
                lo, hi = np.nanpercentile(boot[(name, k)], [2.5, 97.5])
                rows.append({**pc, "method": k, "estimate": point[(name, k)], "ci_lo": lo, "ci_hi": hi})
            for m in ctx["methods"]:
                rows.append({**pc, "method": m, "estimate": extras[(name, m, "at1")],
                             "gamma_used": ctx["gamma_used"][m],
                             "bound_point_lo": point[(name, m, "lo")], "bound_point_hi": point[(name, m, "hi")],
                             "bound_lo": float(np.nanpercentile(boot[(name, m, "lo")], 5)),
                             "bound_hi": float(np.nanpercentile(boot[(name, m, "hi")], 95)),
                             "breakdown_gamma": extras[(name, m, "breakdown")]})
    return rows
