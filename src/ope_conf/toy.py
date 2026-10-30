"""Two closed-form toys that check the bound machinery before the simulator.

1. `Bandit`: one decision, binary hidden u, binary action, outcome Y in {-1, +1}. Exact truth,
   exact nominal (confounded) limit, exact population sort bound, and a sampler.
2. `namkoong_toy`: the example of Namkoong et al. (NeurIPS 2020, Section 4). One hidden coin
   U ~ Unif{0, 1} is drawn once; each of T actions is drawn independently given U with
   P(A_t = 1 | U = 1) = sqrt(G) / (1 + sqrt(G)) and P(A_t = 1 | U = 0) = 1 / (1 + sqrt(G));
   the outcome is Y = U whatever the actions. Always-treat and never-treat are truly equal
   (both 1/2), yet nominal OPE prefers always-treat even with infinite data, and the gap
   grows with T.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .msm import msm_odds_ratio, robust_backup


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logit(p):
    p = np.asarray(p, dtype=float)
    return np.log(p) - np.log1p(-p)


@dataclass(frozen=True)
class Bandit:
    q: np.ndarray       # (S,) distribution of the recorded state
    p0: np.ndarray      # (S,) protocol probability of treating (a = 1)
    theta: np.ndarray   # (S, 2, 2) P(Y = +1 | s, a, u); Y = -1 otherwise
    p_u: float
    lam: float

    @staticmethod
    def random(rng, n_states=4, lam=1.0, p_u=0.3):
        theta = rng.uniform(0.1, 0.9, size=(n_states, 2, 2))
        theta[:, 0, 1] = np.clip(theta[:, 0, 1] - 0.3, 0.02, None)  # u = 1 untreated does worse
        return Bandit(q=rng.dirichlet(np.ones(n_states)), p0=rng.uniform(0.05, 0.95, n_states),
                      theta=theta, p_u=p_u, lam=lam)

    @property
    def pu(self):
        return np.array([1 - self.p_u, self.p_u])

    def behavior(self):
        """(S, U, A): pi_b(a | s, u), the antibiotic-style logit tilt by lam * (2u - 1)."""
        p1 = sigmoid(logit(self.p0)[:, None] + self.lam * (2 * np.arange(2) - 1)[None, :])
        return np.stack([1 - p1, p1], axis=-1)

    def propensity(self):
        """(S, A): nominal e(a | s) = sum_u P(u) pi_b(a | s, u); u is independent of s."""
        return np.einsum("u,sua->sa", self.pu, self.behavior())

    def mean_y(self):
        return 2 * self.theta - 1  # (S, A, U)

    def truth(self, pi):
        """V(pi) for a policy pi (S, A) that does not see u."""
        return float(np.einsum("s,sa,u,sau->", self.q, pi, self.pu, self.mean_y()))

    def posterior_u(self):
        """(S, A, U): P(u | s, a) in the observational data."""
        joint = self.pu[None, :, None] * self.behavior()           # (S, U, A)
        return (joint / joint.sum(axis=1, keepdims=True)).transpose(0, 2, 1)

    def nominal_limit(self, pi):
        """What every nominal estimator converges to: sum_s q sum_a pi E[Y | s, a]."""
        return float(np.einsum("s,sa,sau,sau->", self.q, pi, self.posterior_u(), self.mean_y()))

    def gamma_star(self):
        e = self.propensity()
        return float(msm_odds_ratio(self.behavior(), e[:, None, :]).max())

    def population_bound(self, pi, gamma, lower=True):
        """Exact (infinite-data) sort bound: tilt each cell's outcome distribution."""
        e, post, total = self.propensity(), self.posterior_u(), 0.0
        sign = 1.0 if lower else -1.0
        for s in range(len(self.q)):
            for a in range(2):
                if pi[s, a] == 0:
                    continue
                p_plus = float(post[s, a] @ self.theta[s, a])
                p_next = np.array([1 - p_plus, p_plus])
                y = sign * np.array([-1.0, 1.0])
                total += self.q[s] * pi[s, a] * sign * robust_backup(p_next, y, e[s, a], gamma)
        return total

    def sample(self, n, rng):
        s = rng.choice(len(self.q), size=n, p=self.q)
        u = (rng.random(n) < self.p_u).astype(int)
        a = (rng.random(n) < self.behavior()[s, u, 1]).astype(int)
        y = np.where(rng.random(n) < self.theta[s, a, u], 1.0, -1.0)
        return s, a, y, u


def namkoong_toy(gamma: float, T: int) -> dict:
    """Closed form of Namkoong et al.'s Section 4 example, in the infinite-data limit."""
    r = np.sqrt(gamma)
    p1, p0 = r / (1 + r), 1 / (1 + r)    # P(A_t = 1 | U = 1), P(A_t = 1 | U = 0)
    e = 0.5 * (p1 + p0)                  # nominal propensity of a = 1; equals 1/2
    return {
        "true_treat": 0.5,
        "true_never": 0.5,
        # importance sampling: E[prod_t 1{A_t = a} / e(a) * Y]; only U = 1 has Y = 1
        "is_treat": 0.5 * (p1 / e) ** T,
        "is_never": 0.5 * ((1 - p1) / (1 - e)) ** T,
        # weighted IS (= model-based here): E[Y | all T actions equal a]
        "wis_treat": p1**T / (p1**T + p0**T),
        "wis_never": (1 - p1) ** T / ((1 - p1) ** T + (1 - p0) ** T),
        "gamma_rosenbaum_per_step": (p1 / (1 - p1)) / (p0 / (1 - p0)),  # = gamma
        "gamma_msm_per_step": (p1 / (1 - p1)) / (e / (1 - e)),          # = sqrt(gamma)
    }


def namkoong_toy_sample(gamma: float, T: int, n: int, rng) -> dict:
    """Simulate the example and apply nominal IS and WIS with estimated per-step propensities."""
    r = np.sqrt(gamma)
    u = rng.random(n) < 0.5
    p_treat = np.where(u, r / (1 + r), 1 / (1 + r))
    A = rng.random((n, T)) < p_treat[:, None]
    y = u.astype(float)
    e_hat = A.mean(axis=0)               # per-step nominal propensity of a = 1
    out = {}
    for name, act in (("treat", True), ("never", False)):
        follow = np.all(A == act, axis=1)
        e_act = e_hat if act else 1 - e_hat
        rho = follow / np.prod(e_act)
        out[f"is_{name}"] = float(np.mean(rho * y))
        out[f"wis_{name}"] = float(np.sum(rho * y) / np.sum(rho)) if follow.any() else np.nan
    return out
