"""State and action encodings of the sim-v2 sepsis simulator, vectorised.

The full state index (1,440) uses the simulator's own mixed-radix order, most significant
first: diabetes, heart rate, systolic BP, oxygen, glucose, antibiotics, vasopressors,
ventilation (State.get_state_idx('full')). The observed index (720) drops diabetes, so
full = 720 * diabetes + observed. Actions are 4 * abx + 2 * vent + vaso
(Action.get_action_idx); note that vent and vaso swap places relative to the state.
"""

from __future__ import annotations

import numpy as np

N_ACTIONS = 8
FULL_FACTORS = ("diab", "hr", "bp", "o2", "glu", "abx", "vaso", "vent")
FULL_SIZES = (2, 3, 3, 2, 5, 2, 2, 2)
OBS_FACTORS = FULL_FACTORS[1:]
OBS_SIZES = FULL_SIZES[1:]
N_FULL = int(np.prod(FULL_SIZES))   # 1440
N_OBS = int(np.prod(OBS_SIZES))     # 720
VITALS = ("hr", "bp", "o2", "glu")
VITAL_SIZES = {"hr": 3, "bp": 3, "o2": 2, "glu": 5}
NORMAL = {"hr": 1, "bp": 1, "o2": 1, "glu": 2}
TREATMENTS = ("abx", "vent", "vaso")   # in action-bit order


def decode(idx, factors=FULL_FACTORS, sizes=FULL_SIZES) -> dict[str, np.ndarray]:
    idx = np.asarray(idx)
    out = {}
    for name, size in zip(reversed(factors), reversed(sizes)):
        out[name] = idx % size
        idx = idx // size
    return {name: out[name] for name in factors}


def encode(values, factors=FULL_FACTORS, sizes=FULL_SIZES) -> np.ndarray:
    idx = 0
    for name, size in zip(factors, sizes):
        idx = idx * size + np.asarray(values[name])
    return idx


def factors_for(n_states: int):
    if n_states == N_FULL:
        return FULL_FACTORS, FULL_SIZES
    if n_states == N_OBS:
        return OBS_FACTORS, OBS_SIZES
    raise ValueError(f"no encoding with {n_states} states")


def action_bits(a) -> dict[str, np.ndarray]:
    a = np.asarray(a)
    return {"abx": a // 4, "vent": (a // 2) % 2, "vaso": a % 2}


def action_index(abx, vent, vaso):
    return 4 * np.asarray(abx) + 2 * np.asarray(vent) + np.asarray(vaso)


def num_abnormal(f) -> np.ndarray:
    return sum((np.asarray(f[v]) != NORMAL[v]).astype(int) for v in VITALS)


def outcome_tables(n_states: int = N_FULL):
    """Reward for entering each state (+1 discharge, -1 death, 0 otherwise) and the terminal flag.

    Mirrors MDP.calculateReward: death if 3 or more vitals are abnormal; discharge if all
    vitals are normal and no treatment is on.
    """
    factors, sizes = factors_for(n_states)
    f = decode(np.arange(n_states), factors, sizes)
    k = num_abnormal(f)
    death = k >= 3
    discharge = (k == 0) & (f["abx"] == 0) & (f["vaso"] == 0) & (f["vent"] == 0)
    reward = np.where(death, -1.0, np.where(discharge, 1.0, 0.0))
    return reward, death | discharge
