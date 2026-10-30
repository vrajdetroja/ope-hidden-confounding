"""Exact transition tensor of the sim-v2 sepsis simulator, plus the hidden-u channels.

Port of sepsisSimDiabetes/MDP.py (clinicalml/gumbel-max-scm, branch sim-v2, MIT license;
see third_party/sepsis). MDP.transition applies four update steps in a fixed order
(antibiotics, ventilation, vasopressors, then drift), and within every step each vital sign
moves independently given diabetes, the treatments already on, and the action. So each step
is a small stochastic matrix per vital; multiplying them in order gives one matrix per vital,
and their Kronecker product is the exact distribution of the next vitals. The treatment bits
of the next state are the action's bits, and diabetes never changes.

The hidden-u channels act after the simulator's own step:
  outcome channel     u = 1 and antibiotics off: HR N->H and BP N->L, each w.p. kappa
  prognostic channel  u = 1: O2 N->L w.p. kappa_p, whatever the treatment
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .spaces import N_ACTIONS, N_FULL, VITAL_SIZES, VITALS, decode, outcome_tables

HR_BP_START = np.array([0.25, 0.5, 0.25])
O2_START = np.array([0.2, 0.8])
GLU_START = {0: np.array([0.05, 0.15, 0.6, 0.15, 0.05]), 1: np.array([0.01, 0.05, 0.15, 0.6, 0.19])}


def _move(n, frm, to, p):
    """Identity, except level `frm` moves to `to` with probability p."""
    M = np.eye(n)
    M[frm, frm] -= p
    M[frm, to] += p
    return M


def _drift(n, p_down, p_up):
    """Move down one level w.p. p_down and up one level w.p. p_up, clipped at the ends."""
    M = np.zeros((n, n))
    for v in range(n):
        M[v, max(0, v - 1)] += p_down
        M[v, min(n - 1, v + 1)] += p_up
        M[v, v] += 1 - p_down - p_up
    return M


def vital_kernels(diab, prev, act, u=0, kappa=0.0, kappa_p=0.0):
    """Per-vital transition matrices for one (diabetes, treatments on, action) combination.

    prev and act map 'abx', 'vent', 'vaso' to 0/1. Mirrors MDP.transition line by line.
    """
    K = {v: np.eye(VITAL_SIZES[v]) for v in VITALS}

    # 1. antibiotics (transition_antibiotics_on / _off)
    if act["abx"]:
        K["hr"] = K["hr"] @ _move(3, 2, 1, 0.5)
        K["bp"] = K["bp"] @ _move(3, 2, 1, 0.5)
        hr_fluct = bp_fluct = False
    elif prev["abx"]:
        K["hr"] = K["hr"] @ _move(3, 1, 2, 0.1)
        K["bp"] = K["bp"] @ _move(3, 1, 2, 0.1)
        hr_fluct = bp_fluct = False
    else:
        hr_fluct = bp_fluct = True

    # 2. ventilation (transition_vent_on / _off)
    if act["vent"]:
        K["o2"] = K["o2"] @ _move(2, 0, 1, 0.7)
        o2_fluct = False
    elif prev["vent"]:
        K["o2"] = K["o2"] @ _move(2, 1, 0, 0.1)
        o2_fluct = False
    else:
        o2_fluct = True

    # 3. vasopressors (transition_vaso_on / _off)
    glu_fluct = True
    if act["vaso"]:
        if diab == 0:
            M = np.array([[0.3, 0.7, 0.0], [0.0, 0.3, 0.7], [0.0, 0.0, 1.0]])
        else:
            M = np.array([[0.1, 0.5, 0.4], [0.0, 0.1, 0.9], [0.0, 0.0, 1.0]])
            K["glu"] = K["glu"] @ _drift(5, 0.0, 0.5)
        K["bp"] = K["bp"] @ M
        bp_fluct = glu_fluct = False
    elif prev["vaso"]:
        K["bp"] = K["bp"] @ _drift(3, 0.1 if diab == 0 else 0.05, 0.0)
        bp_fluct = False

    # 4. drift (transition_fluctuate); sim-v2 lets non-diabetic glucose rise to level 4
    if hr_fluct:
        K["hr"] = K["hr"] @ _drift(3, 0.1, 0.1)
    if bp_fluct:
        K["bp"] = K["bp"] @ _drift(3, 0.1, 0.1)
    if o2_fluct:
        K["o2"] = K["o2"] @ _drift(2, 0.1, 0.1)
    if glu_fluct:
        p = 0.1 if diab == 0 else 0.3
        K["glu"] = K["glu"] @ _drift(5, p, p)

    # 5. hidden-u channels, after the simulator's own step
    if u == 1 and not act["abx"] and kappa > 0:
        K["hr"] = K["hr"] @ _move(3, 1, 2, kappa)
        K["bp"] = K["bp"] @ _move(3, 1, 0, kappa)
    if u == 1 and kappa_p > 0:
        K["o2"] = K["o2"] @ _move(2, 1, 0, kappa_p)
    return K


def transition_matrix(u=0, kappa=0.0, kappa_p=0.0) -> sp.csr_matrix:
    """P[s * 8 + a, s'] over the 1,440 full states, as a CSR matrix with rows summing to one.

    Full index = 720 * diab + 8 * vitals + treatments, where vitals = ((hr*3+bp)*2+o2)*5+glu
    and treatments = 4*abx + 2*vaso + vent (state order).
    """
    vit = np.arange(90)
    rows, cols, vals = [], [], []
    for diab in (0, 1):
        for prev_code in range(8):
            prev = {"abx": prev_code // 4, "vaso": (prev_code // 2) % 2, "vent": prev_code % 2}
            for a in range(N_ACTIONS):
                act = {"abx": a // 4, "vent": (a // 2) % 2, "vaso": a % 2}
                K = vital_kernels(diab, prev, act, u, kappa, kappa_p)
                Kv = np.kron(np.kron(np.kron(K["hr"], K["bp"]), K["o2"]), K["glu"])
                next_code = 4 * act["abx"] + 2 * act["vaso"] + act["vent"]
                i, j = np.nonzero(Kv)
                rows.append((diab * 720 + vit[i] * 8 + prev_code) * N_ACTIONS + a)
                cols.append(diab * 720 + vit[j] * 8 + next_code)
                vals.append(Kv[i, j])
    P = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(N_FULL * N_ACTIONS, N_FULL))
    P.sum_duplicates()
    P.sort_indices()
    return P


def start_distribution(p_diabetes=0.2) -> np.ndarray:
    """Exact law of MDP.get_new_state(): treatments off, vitals drawn from their marginals,
    diabetes ~ Bernoulli(p_diabetes), all redrawn until the state is not absorbing."""
    f = decode(np.arange(N_FULL))
    d = f["diab"]
    p = (np.where(d == 1, p_diabetes, 1 - p_diabetes) * HR_BP_START[f["hr"]] * HR_BP_START[f["bp"]]
         * O2_START[f["o2"]] * np.where(d == 1, GLU_START[1][f["glu"]], GLU_START[0][f["glu"]]))
    p = p * ((f["abx"] == 0) & (f["vaso"] == 0) & (f["vent"] == 0))
    _, terminal = outcome_tables(N_FULL)
    p[terminal] = 0.0
    return p / p.sum()
