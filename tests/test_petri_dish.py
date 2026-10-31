import numpy as np
import pytest
import torch

from cellware.envs.petri_dish import LEVEL_1_TABLE, LEVEL_2_TABLE, SWAP, PetriDish


class ReferenceDish:
    """Line-for-line copy of the legacy Ray environment's dynamics, without Ray."""

    def __init__(self, size=10):
        self.MAX_HOR_VAL = size - 1
        self.MAX_VER_VAL = size - 1
        self.lvl_1_reward_table = np.array(LEVEL_1_TABLE)
        self.lvl_2_reward_table = np.array(LEVEL_2_TABLE)

    def step(self, action, current_state):
        state_observation = self.take_action(action, current_state)
        r1, r2 = self.get_reward(current_state, state_observation)
        return state_observation, r1, r2

    def get_reward(self, current_state, s):
        if all(current_state == s):
            return 0, 0
        t1, t2 = self.lvl_1_reward_table, self.lvl_2_reward_table
        r1 = t1[s[0], s[2]] + t1[s[1], s[3]]
        r2 = t2[s[0], s[2]] + t2[s[1], s[3]]
        t1, t2 = t1.copy(), t2.copy()
        t1[s[0], s[2]] = t1[s[1], s[3]] = 0
        t2[s[0], s[2]] = t2[s[1], s[3]] = 0
        self.lvl_1_reward_table, self.lvl_2_reward_table = t1, t2
        return r1, r2

    def take_action(self, action, current_state):
        t = self.lvl_1_reward_table
        state = current_state.copy()
        if (
            action == 4
            and state[2] == state[3]
            and (state[2] == self.MAX_HOR_VAL or t[state[0], state[2] + 1] == -1)
        ):
            return state
        elif (
            action == 4
            and state[0] == state[1]
            and (state[0] == self.MAX_VER_VAL or t[state[0] + 1, state[2]] == -1)
        ):
            return state
        elif action == 4 and state[2] == state[3]:
            state[1] = state[0]
            state[3] = state[2] + 1
            return state
        elif action == 4 and state[0] == state[1]:
            state[1] = state[0] + 1
            state[3] = state[2]
            return state
        if (
            (action == 0 and any(state[0:2] == 0))
            or (action == 2 and any(state[0:2] == self.MAX_VER_VAL))
            or (
                action == 0 and (t[state[0] - 1, state[2]] == -1 or t[state[1] - 1, state[3]] == -1)
            )
            or (
                action == 2 and (t[state[0] + 1, state[2]] == -1 or t[state[1] + 1, state[3]] == -1)
            )
        ):
            pass
        elif (
            (action == 1 and any(state[2:4] == 0))
            or (action == 3 and any(state[2:4] == self.MAX_HOR_VAL))
            or (
                action == 1 and (t[state[0], state[2] - 1] == -1 or t[state[1], state[3] - 1] == -1)
            )
            or (
                action == 3 and (t[state[0], state[2] + 1] == -1 or t[state[1], state[3] + 1] == -1)
            )
        ):
            pass
        elif action == 0:
            state[0:2] -= 1
        elif action == 2:
            state[0:2] += 1
        elif action == 1:
            state[2:4] -= 1
        elif action == 3:
            state[2:4] += 1
        return state


@pytest.mark.parametrize("seed", range(5))
def test_matches_legacy_dynamics_single_agent(seed):
    rng = np.random.default_rng(seed)
    batch = 16
    dish = PetriDish(batch_size=batch, num_agents=1)
    refs = [ReferenceDish() for _ in range(batch)]
    ref_states = [np.array([0, 1, 0, 0]) for _ in range(batch)]
    for _ in range(200):
        actions = rng.integers(0, 5, size=batch)
        states, r1, r2 = dish.step(torch.tensor(actions).unsqueeze(1))
        for b in range(batch):
            ref_states[b], e1, e2 = refs[b].step(actions[b], ref_states[b])
            assert states[b, 0].tolist() == ref_states[b].tolist()
            assert (r1[b, 0].item(), r2[b, 0].item()) == (e1, e2)
        for b in range(batch):
            assert np.array_equal(dish.table1[b].numpy(), refs[b].lvl_1_reward_table)


def test_shared_dish_depletes_resources_across_agents():
    rng = np.random.default_rng(1)
    dish = PetriDish(batch_size=1, num_agents=3)
    ref = ReferenceDish()
    ref_states = [np.array([0, 1, 0, 0]) for _ in range(3)]
    for _ in range(150):
        actions = rng.integers(0, 5, size=3)
        states, r1, _ = dish.step(torch.tensor(actions).unsqueeze(0))
        for n in range(3):
            ref_states[n], e1, _ = ref.step(actions[n], ref_states[n])
            assert states[0, n].tolist() == ref_states[n].tolist()
            assert r1[0, n].item() == e1


def test_swap_rotates_body_and_reset_restores_tables():
    dish = PetriDish()
    states, r1, _ = dish.step(torch.tensor([[SWAP]]))
    assert states[0, 0].tolist() == [0, 0, 0, 1]  # vertical -> horizontal
    assert r1.item() == 2  # cells (0,0) and (0,1) of the level-1 map
    dish.reset()
    assert dish.table1[0, 0, 0].item() == 1
    assert dish.state[0, 0].tolist() == [0, 1, 0, 0]
