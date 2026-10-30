"""Exact ground truth by backward induction on the augmented (hidden, recorded) state, and the
exact confounding strength Gamma* by enumeration over every reachable (state, hidden, action)."""

from __future__ import annotations

import numpy as np

from .msm import msm_odds_ratio
from .sim.confounding import World
from .sim.spaces import action_bits


def value_of_aug_policy(world: World, pi_aug: np.ndarray, gamma: float | None = None) -> float:
    """V = E[sum_t gamma^t r_t] for a policy over augmented states, pi_aug (H, n_aug, A)."""
    gamma = world.cfg.gamma if gamma is None else gamma
    reward, alive_next = world.reward, ~world.terminal
    V = np.zeros(world.n_aug)
    for t in reversed(range(world.H)):
        Q = (world.kernels[world.kernel_of_t[t]] @ (reward + gamma * alive_next * V))
        V = np.einsum("xa,xa->x", pi_aug[t], Q.reshape(world.n_aug, world.A))
    return float(world.mu @ V)


def true_value(world: World, pi_rec: np.ndarray) -> float:
    """True value of an evaluation policy (H, n_rec, A) that sees only the recorded state."""
    return value_of_aug_policy(world, world.lift(pi_rec))


def true_value_clinician(world: World) -> float:
    return value_of_aug_policy(world, world.behavior_by_t())


def occupancy(world: World, pi_aug: np.ndarray) -> np.ndarray:
    """(H, n_aug): probability of being alive in state x when the decision at step t is made."""
    d = np.zeros((world.H, world.n_aug))
    d[0] = world.mu
    alive = ~world.terminal
    for t in range(world.H - 1):
        flow = (d[t][:, None] * pi_aug[t]).ravel()
        d[t + 1] = (world.kernels[world.kernel_of_t[t]].T @ flow) * alive
    return d


def nominal_propensity(world: World, occ: np.ndarray | None = None) -> np.ndarray:
    """(n_est_phases, n_rec, A): the propensity the data identify, e(a | s) = P_b(A_t = a | S_t = s),
    averaged over the steps the analyst pools (exact, via the clinician's occupancy)."""
    pb = world.behavior_by_t()
    occ = occupancy(world, pb) if occ is None else occ
    n_phase = world.n_est_phases
    num = np.zeros((n_phase, world.n_rec, world.A))
    den = np.zeros((n_phase, world.n_rec))
    for t in range(world.H):
        p = world.est_phase_of_t[t]
        mass = occ[t].reshape(world.n_hid, world.n_rec)
        num[p] += np.einsum("hs,hsa->sa", mass, pb[t].reshape(world.n_hid, world.n_rec, world.A))
        den[p] += mass.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        e = num / den[..., None]
    return np.where(den[..., None] > 0, e, 1.0 / world.A)


def observational_kernels(world: World, occ: np.ndarray | None = None) -> list:
    """What the data identify about the dynamics, per analyst phase: P_obs(s' | s, a) =
    P_b(S_{t+1} = s' | S_t = s, A_t = a), pooled over the phase's steps (exact)."""
    import scipy.sparse as sp

    pb = world.behavior_by_t()
    occ = occupancy(world, pb) if occ is None else occ
    n_rec, n_hid, A = world.n_rec, world.n_hid, world.A
    x = np.arange(world.n_aug)
    rows_agg = sp.csr_matrix((np.ones(world.n_aug * A),
                              (((x % n_rec)[:, None] * A + np.arange(A)).ravel(),
                               np.arange(world.n_aug * A))), shape=(n_rec * A, world.n_aug * A))
    cols_agg = sp.csr_matrix((np.ones(world.n_aug), (x, x % n_rec)), shape=(world.n_aug, n_rec))
    joint = [sp.csr_matrix((n_rec * A, n_rec)) for _ in range(world.n_est_phases)]
    for t in range(world.H):
        flow = sp.diags((occ[t][:, None] * pb[t]).ravel())
        joint[world.est_phase_of_t[t]] += rows_agg @ flow @ world.kernels[world.kernel_of_t[t]] @ cols_agg
    out = []
    for J in joint:
        J = sp.csr_matrix(J)
        mass = np.asarray(J.sum(axis=1)).ravel()
        out.append(sp.diags(np.divide(1.0, mass, out=np.zeros_like(mass), where=mass > 0)) @ J)
    return out


def nominal_limit(world: World, pi_rec: np.ndarray, kernels_obs: list | None = None) -> float:
    """The value every nominal estimator converges to as n -> infinity: backward induction on the
    observational kernels, as if sequential ignorability held. In R1 and R2 the observed
    process is an MDP with these kernels, so model-based and importance-sampling estimators
    share this limit; the gap to the truth is the asymptotic confounding bias."""
    kernels_obs = observational_kernels(world) if kernels_obs is None else kernels_obs
    start = world.mu.reshape(world.n_hid, world.n_rec).sum(axis=0)
    y_next, V = world.rec_reward, np.zeros(world.n_rec)
    for t in reversed(range(world.H)):
        P = kernels_obs[world.est_phase_of_t[t]]
        Q = (P @ (y_next + world.cfg.gamma * (~world.rec_terminal) * V)).reshape(world.n_rec, world.A)
        V = np.einsum("sa,sa->s", pi_rec[t], Q)
    return float(start @ V)


def gamma_star(world: World, tol: float = 1e-15) -> dict:
    """Smallest Gamma for which the MSM holds, over every reachable (s, h, a) at confounded steps.

    msm_action:      MSM over the 8 actions (what the action-level bounds need)
    msm_factor:      MSM for the binary factor the hidden variable tilts (antibiotics in R1/R2;
                     the forced factor in R3), what the structured bound needs
    rosenbaum_factor: odds ratio of that factor between the two hidden values (= e^(2 lam) in R1/R2)
    """
    pb = world.behavior_by_t()
    occ = occupancy(world, pb)
    e = nominal_propensity(world, occ)
    factor = "abx" if world.cfg.regime in ("R1", "R2") else world.forced_factor
    on = action_bits(np.arange(world.A))[factor] == 1
    g_act = g_fac = g_ros = 1.0
    for t in np.flatnonzero(world.confounded_t):
        reach = occ[t] > tol
        if not reach.any():
            continue
        p_t = pb[t]
        e_t = np.tile(e[world.est_phase_of_t[t]], (world.n_hid, 1))
        g_act = max(g_act, float(msm_odds_ratio(p_t[reach], e_t[reach]).max()))
        p_f, e_f = p_t[:, on].sum(axis=1), e_t[:, on].sum(axis=1)
        g_fac = max(g_fac, float(msm_odds_ratio(p_f[reach], e_f[reach]).max()))
        # Rosenbaum: compare the hidden values at the same recorded state (both reachable)
        both = reach.reshape(world.n_hid, world.n_rec).all(axis=0)
        if both.any():
            pf = p_f.reshape(world.n_hid, world.n_rec)[:, both]
            g_ros = max(g_ros, float(msm_odds_ratio(pf[1], pf[0]).max()))
    return {"msm_action": g_act, "msm_factor": g_fac, "rosenbaum_factor": g_ros}
