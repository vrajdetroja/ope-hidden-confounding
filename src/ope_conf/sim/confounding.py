"""The three confounding regimes as one augmented MDP over (hidden, recorded) states.

An augmented state is x = h * n_rec + s: s is what the data records and h is what it hides.

R1 single decision  h = u, drawn once at admission (P(u = 1) = p_u). It tilts only the first
                    antibiotic decision and acts on outcomes only through the first transition.
                    Recorded state: the full simulator state (1,440, diabetes included).
R2 memoryless       h = u_t, drawn fresh at every step; both channels act at every step.
R3 persistent       no u. h = diabetes, hidden from the data (recorded state: 720 observed
                    states). The clinician mixes a diabetes-blind protocol with a
                    diabetes-aware one: pi_b = (1 - w) pi_blind + w pi_full.

The protocol is the best u-blind (in R3, diabetes-blind) policy: value iteration on the true
dynamics with the hidden variable averaged over its prior, mixed with uniform at epsilon. It is
stationary wherever the dynamics are, so pooled-over-time propensities are correctly specified
when there is no confounding. In R1 the admission step differs (the u channel acts only there),
so the protocol's admission action comes from a one-step lookahead on the u-averaged first
transition, and the analyst models step 0 separately from steps 1+ (as Namkoong et al. do).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from ..config import Config
from .dynamics import start_distribution, transition_matrix
from .policies import discounted_value_iteration, eps_greedy, greedy, tilt_antibiotics
from .spaces import N_ACTIONS, N_FULL, N_OBS, outcome_tables


@dataclass
class World:
    cfg: Config
    n_rec: int
    n_hid: int
    mu: np.ndarray               # (n_aug,) start distribution
    kernels: list                # CSR (n_aug * A, n_aug), one per dynamics phase
    kernel_of_t: np.ndarray      # (H,) dynamics phase used at step t
    behavior: np.ndarray         # (n_bphase, n_aug, A) clinician policy per phase
    bphase_of_t: np.ndarray      # (H,)
    confounded_t: np.ndarray     # (H,) bool: the hidden variable tilts the decision at step t
    est_phase_of_t: np.ndarray   # (H,) the analyst's time structure for propensities and models
    protocol: np.ndarray         # (H, n_rec, A)
    forced_factor: str           # treatment that the withhold / always policies force
    rec_reward: np.ndarray       # (n_rec,) reward for entering a recorded state
    rec_terminal: np.ndarray     # (n_rec,)
    extra: dict = field(default_factory=dict)

    @property
    def A(self) -> int:
        return N_ACTIONS

    @property
    def H(self) -> int:
        return self.cfg.H

    @property
    def n_aug(self) -> int:
        return self.n_rec * self.n_hid

    @property
    def reward(self) -> np.ndarray:
        return np.tile(self.rec_reward, self.n_hid)

    @property
    def terminal(self) -> np.ndarray:
        return np.tile(self.rec_terminal, self.n_hid)

    def lift(self, pi_rec: np.ndarray) -> np.ndarray:
        """(H, n_rec, A) recorded-state policy -> (H, n_aug, A); it ignores the hidden part."""
        return np.tile(pi_rec, (1, self.n_hid, 1))

    def behavior_by_t(self) -> np.ndarray:
        return self.behavior[self.bphase_of_t]

    @property
    def n_est_phases(self) -> int:
        return int(self.est_phase_of_t.max()) + 1


def build_world(cfg: Config) -> World:
    return {"R1": _build_r1, "R2": _build_r2, "R3": _build_r3}[cfg.regime](cfg)


def _tilted_pair(pi, cfg):
    clip = (cfg.p_clip_lo, cfg.p_clip_hi)
    return np.concatenate([tilt_antibiotics(pi, cfg.lam, 0, clip), tilt_antibiotics(pi, cfg.lam, 1, clip)])


def _build_r1(cfg: Config) -> World:
    H, pu, A = cfg.H, cfg.p_u, N_ACTIONS
    P0 = transition_matrix()
    P1 = transition_matrix(u=1, kappa=cfg.kappa, kappa_p=cfg.kappa_p)
    reward, terminal = outcome_tables(N_FULL)
    V, Q = discounted_value_iteration(P0, reward, terminal, cfg.vi_discount)
    proto_later = eps_greedy(greedy(Q), cfg.epsilon)
    Pbar0 = (1 - pu) * P0 + pu * P1
    Q0 = (Pbar0 @ (reward + cfg.vi_discount * (~terminal) * V)).reshape(N_FULL, A)
    proto_first = eps_greedy(greedy(Q0), cfg.epsilon)
    protocol = np.repeat(proto_later[None], H, axis=0)
    protocol[0] = proto_first
    behavior = np.stack([_tilted_pair(proto_first, cfg), np.tile(proto_later, (2, 1))])
    mu_s = start_distribution(cfg.p_diabetes)
    phases = np.r_[0, np.ones(H - 1, int)].astype(int)
    return World(
        cfg=cfg, n_rec=N_FULL, n_hid=2,
        mu=np.concatenate([(1 - pu) * mu_s, pu * mu_s]),
        kernels=[sp.block_diag([P0, P1], format="csr"), sp.block_diag([P0, P0], format="csr")],
        kernel_of_t=phases, behavior=behavior, bphase_of_t=phases,
        confounded_t=phases == 0, est_phase_of_t=phases, protocol=protocol,
        forced_factor="abx", rec_reward=reward, rec_terminal=terminal,
        extra={"P0": P0, "P1": P1},
    )


def _build_r2(cfg: Config) -> World:
    H, pu, A = cfg.H, cfg.p_u, N_ACTIONS
    P0 = transition_matrix()
    P1 = transition_matrix(u=1, kappa=cfg.kappa, kappa_p=cfg.kappa_p)
    reward, terminal = outcome_tables(N_FULL)
    _, Q = discounted_value_iteration((1 - pu) * P0 + pu * P1, reward, terminal, cfg.vi_discount)
    proto = eps_greedy(greedy(Q), cfg.epsilon)
    mu_s = start_distribution(cfg.p_diabetes)
    K = sp.bmat([[(1 - pu) * P0, pu * P0], [(1 - pu) * P1, pu * P1]], format="csr")  # u' fresh
    zeros = np.zeros(H, int)
    return World(
        cfg=cfg, n_rec=N_FULL, n_hid=2,
        mu=np.concatenate([(1 - pu) * mu_s, pu * mu_s]),
        kernels=[K], kernel_of_t=zeros, behavior=_tilted_pair(proto, cfg)[None], bphase_of_t=zeros,
        confounded_t=np.ones(H, bool), est_phase_of_t=zeros,
        protocol=np.repeat(proto[None], H, axis=0), forced_factor="abx",
        rec_reward=reward, rec_terminal=terminal, extra={"P0": P0, "P1": P1},
    )


def _build_r3(cfg: Config) -> World:
    H, pd, A = cfg.H, cfg.p_diabetes, N_ACTIONS
    P = transition_matrix()
    reward_full, terminal_full = outcome_tables(N_FULL)
    _, Q_full = discounted_value_iteration(P, reward_full, terminal_full, cfg.vi_discount)
    pi_full = eps_greedy(greedy(Q_full), cfg.epsilon)
    # diabetes-blind protocol: dynamics on the 720 observed states averaged over the prior
    P_obs = (1 - pd) * P[: N_OBS * A, :N_OBS] + pd * P[N_OBS * A:, N_OBS:]
    reward, terminal = outcome_tables(N_OBS)
    _, Q_blind = discounted_value_iteration(sp.csr_matrix(P_obs), reward, terminal, cfg.vi_discount)
    pi_blind = eps_greedy(greedy(Q_blind), cfg.epsilon)
    behavior = (1 - cfg.w) * np.tile(pi_blind, (2, 1)) + cfg.w * pi_full
    zeros = np.zeros(H, int)
    return World(
        cfg=cfg, n_rec=N_OBS, n_hid=2, mu=start_distribution(pd),
        kernels=[P], kernel_of_t=zeros, behavior=behavior[None], bphase_of_t=zeros,
        confounded_t=np.ones(H, bool), est_phase_of_t=zeros,
        protocol=np.repeat(pi_blind[None], H, axis=0), forced_factor="vaso",
        rec_reward=reward, rec_terminal=terminal, extra={"pi_full": pi_full},
    )
