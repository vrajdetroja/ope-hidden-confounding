"""Thin access to the untouched sim-v2 simulator in third_party/sepsis (tests only)."""

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "sepsis"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sepsisSimDiabetes.Action import Action  # noqa: E402
from sepsisSimDiabetes.MDP import MDP  # noqa: E402
from sepsisSimDiabetes.State import State  # noqa: E402


def sample_next_states(state_idx: int, action_idx: int, n: int, seed: int) -> np.ndarray:
    """n draws of the next full-state index from MDP.transition."""
    np.random.seed(seed)
    mdp = MDP(init_state_idx=state_idx, init_state_idx_type="full")
    action = Action(action_idx=action_idx)
    out = np.empty(n, dtype=int)
    for i in range(n):
        mdp.state = State(state_idx=state_idx, idx_type="full")
        mdp.transition(action)
        out[i] = mdp.state.get_state_idx(idx_type="full")
    return out


def sample_start_states(n: int, seed: int) -> np.ndarray:
    np.random.seed(seed)
    mdp = MDP()
    return np.array([mdp.get_new_state().get_state_idx(idx_type="full") for _ in range(n)])


def simulate_returns(policy_full: np.ndarray, n: int, horizon: int, seed: int) -> np.ndarray:
    """Episode returns under a (1440, 8) policy, exactly as DataGenerator.simulate scores them."""
    np.random.seed(seed)
    mdp = MDP(policy_array=policy_full, policy_idx_type="full")
    returns = np.zeros(n)
    for i in range(n):
        mdp.state = mdp.get_new_state()
        for _ in range(horizon):
            reward = mdp.transition(mdp.select_actions())
            if reward != 0:
                returns[i] = reward
                break
    return returns
