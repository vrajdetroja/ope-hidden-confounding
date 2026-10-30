"""Protocol, clinician and evaluation policies.

Policies are arrays of action probabilities: (S, A) when stationary, (H, S, A) when they
depend on the step. Evaluation policies see only the recorded state.
"""

from __future__ import annotations

import numpy as np

from ..toy import logit, sigmoid
from .spaces import N_ACTIONS, action_bits, action_index


def discounted_value_iteration(P, reward, terminal, discount, tol=1e-12, max_iter=200_000):
    """Optimal values and Q for the stationary discounted problem.

    P is (S * A, S); reward[s'] is collected on entering s', and an episode stops on
    entering a terminal state: Q(s, a) = sum_s' P(s' | s, a) [r(s') + discount * V(s') 1{s' not terminal}].
    """
    S = P.shape[1]
    A = P.shape[0] // S
    V = np.zeros(S)
    for _ in range(max_iter):
        Q = (P @ (reward + discount * (~terminal) * V)).reshape(S, A)
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            return V_new, Q
        V = V_new
    raise RuntimeError("value iteration did not converge")


def greedy(Q, tol=1e-9):
    """argmax with deterministic tie-breaking: the lowest action index within tol of the max."""
    return np.argmax(Q >= Q.max(axis=-1, keepdims=True) - tol, axis=-1)


def one_hot(actions, A=N_ACTIONS):
    return np.eye(A)[actions]


def eps_greedy(actions, eps, A=N_ACTIONS):
    """(1 - eps) on the greedy action plus eps spread uniformly: every action gets >= eps / A."""
    return (1 - eps) * one_hot(actions, A) + eps / A


def tilt_antibiotics(pi, lam, u, clip=(0.05, 0.95)):
    """Clinician pi_b(a | s, u): the protocol with its antibiotic logit shifted by lam * (2u - 1).

    p_0(s) = protocol P(abx on | s), clipped; p_u = sigmoid(logit p_0 + lam (2u - 1));
    pi_b(a | s, u) = p_u^abx (1 - p_u)^(1 - abx) * protocol(vent, vaso | s, abx).
    """
    abx = action_bits(np.arange(N_ACTIONS))["abx"]
    p_on = pi[..., abx == 1].sum(axis=-1)
    p_u = sigmoid(logit(np.clip(p_on, *clip)) + lam * (2 * u - 1))
    marg = np.where(abx == 1, p_on[..., None], 1 - p_on[..., None])
    rest = pi / marg                                   # protocol(vent, vaso | s, abx)
    return np.where(abx == 1, p_u[..., None], 1 - p_u[..., None]) * rest


def force_factor(pi, factor, value):
    """Move every action's probability to the same action with `factor` set to `value`
    (Namkoong et al.'s 'with / without antibiotics' construction)."""
    bits = action_bits(np.arange(N_ACTIONS))
    bits[factor] = np.full(N_ACTIONS, value)
    target = action_index(bits["abx"], bits["vent"], bits["vaso"])
    out = np.zeros_like(pi)
    for a in range(N_ACTIONS):
        out[..., target[a]] += pi[..., a]
    return out


def soften(pi, total):
    """(1 - delta) pi + delta uniform, with delta chosen so a deterministic policy keeps 1 - total
    on its action and spreads `total` over the others (as in Tang & Wiens)."""
    A = pi.shape[-1]
    delta = total * A / (A - 1)
    return (1 - delta) * pi + delta / A


def uniform(H, S, A=N_ACTIONS):
    return np.full((H, S, A), 1.0 / A)
