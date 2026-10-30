"""Learned policies: finite-horizon value iteration on the empirical MDP (the 'AI clinician').

Unseen (s, a) pairs count as death, as in Oberst & Sontag, so a greedy policy never prefers
one over an observed action; a state with no observed action gets value -1 (and, outside the
data, action 0). The pessimistic variant subtracts c / sqrt(n(s, a)) at every backup. Greedy
policies are softened before use.
"""

from __future__ import annotations

import numpy as np

from .estimators import Fit
from .sim.policies import one_hot, soften


def learn_policy(fit: Fit, H, rec_reward, rec_terminal, pessimism=0.0, soften_total=0.01,
                 gamma=1.0) -> np.ndarray:
    n_c, A = fit.idx.size, fit.A
    rr, alive = rec_reward[fit.idx], ~rec_terminal[fit.idx]
    V = np.zeros(n_c)
    greedy = np.zeros((H, fit.S), dtype=int)
    for t in reversed(range(H)):
        p = fit.phase_of_t[t]
        q = (fit.P[p] @ (rr + gamma * alive * V)).reshape(n_c, A)
        if pessimism > 0:
            q = q - pessimism / np.sqrt(np.maximum(fit.counts_c[p], 1.0))
        q = np.where(fit.seen_c[p], q, -np.inf)
        any_seen = fit.seen_c[p].any(axis=1)
        a_star = np.where(any_seen, np.argmax(q, axis=1), 0)
        V = np.where(any_seen, q[np.arange(n_c), a_star], -1.0)
        greedy[t, fit.idx] = a_star
    return soften(one_hot(greedy, A), soften_total)
