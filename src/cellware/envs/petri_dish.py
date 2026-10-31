"""Batched, Ray-free port of the legacy "Environment Petri Dish".

Each agent is a two-cell body on an ``size x size`` grid. Its state is
``[row_a, row_b, col_a, col_b]``; the two cells are always adjacent, so the body is
either vertical (``col_a == col_b``) or horizontal (``row_a == row_b``).

Actions: 0=UP, 1=LEFT, 2=DOWN, 3=RIGHT, 4=SWAP (rotate between vertical and horizontal).
Cells with value -1 in the level-1 table are walls. Entering a cell collects its reward
in both tables and sets it to 0 there, so resources are depleted for every agent sharing
the dish.

The dynamics match ``legacy/jepa_evolution_network/environment_dish.py`` exactly, but
``B`` independent dishes are simulated at once as tensors, with ``N`` agents per dish
acting in index order within a step.
"""

from __future__ import annotations

import torch

UP, LEFT, DOWN, RIGHT, SWAP = range(5)
NUM_ACTIONS = 5
WALL = -1

LEVEL_1_TABLE = [
    [1, 1, 1, 1, 2, 2, 3, 3, 0, 0],
    [1, 1, 1, 1, 2, 2, -1, 4, -1, -1],
    [1, 1, 1, 1, -1, -1, -1, 4, 50, 0],
    [1, 1, 1, 1, -1, -1, -1, 0, 0, 0],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
    [0, 10, 20, 10, 0, 0, 20, -1, 60, 0],
    [10, 20, 30, 20, 10, 20, 30, -1, 6, 0],
    [20, 30, 90, 30, 20, 10, 20, -1, 6, -1],
    [10, 20, 30, 20, 10, 5, 5, -1, 0, 0],
    [0, 10, 20, 10, 0, 0, 0, 70, 0, 0],
]

LEVEL_2_TABLE = [
    [5, 5, 5, 5, 5, 5, 10, 20, 10, 5],
    [5, 5, 5, 5, 5, 10, 20, 30, 20, 10],
    [5, 5, 5, 5, 10, 20, 30, 40, 30, 20],
    [5, 5, 5, 10, 20, 30, 40, 90, 40, 30],
    [5, 5, 5, 10, 20, 30, 40, 90, 40, 30],
    [5, 5, 5, 10, 20, 30, 40, 90, 40, 30],
    [5, 5, 5, 10, 20, 30, 40, 90, 40, 30],
    [5, 5, 5, 10, 10, 20, 30, 40, 30, 20],
    [5, 5, 5, 5, 5, 10, 20, 30, 20, 10],
    [5, 5, 5, 5, 5, 5, 10, 20, 10, 5],
]

DEFAULT_START = (0, 1, 0, 0)


class PetriDish:
    """``batch_size`` independent dishes, each shared by ``num_agents`` agents."""

    def __init__(
        self,
        batch_size: int = 1,
        num_agents: int = 1,
        size: int = 10,
        device: str | torch.device = "cpu",
    ):
        if size != len(LEVEL_1_TABLE):
            raise ValueError(
                f"Only the built-in {len(LEVEL_1_TABLE)}x{len(LEVEL_1_TABLE)} map exists"
            )
        self.batch_size = batch_size
        self.num_agents = num_agents
        self.size = size
        self.max_index = size - 1
        self.device = torch.device(device)
        self._base1 = torch.tensor(LEVEL_1_TABLE, dtype=torch.long, device=self.device)
        self._base2 = torch.tensor(LEVEL_2_TABLE, dtype=torch.long, device=self.device)
        self.reset()

    def reset(self, start: tuple[int, int, int, int] = DEFAULT_START) -> torch.Tensor:
        self.table1 = self._base1.expand(self.batch_size, -1, -1).clone()
        self.table2 = self._base2.expand(self.batch_size, -1, -1).clone()
        self.state = (
            torch.tensor(start, dtype=torch.long, device=self.device)
            .expand(self.batch_size, self.num_agents, 4)
            .clone()
        )
        return self.state.clone()

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply ``actions`` of shape ``(B, N)``.

        Returns the new states ``(B, N, 4)`` and the level-1 and level-2 rewards ``(B, N)``.
        """
        actions = actions.to(self.device, torch.long)
        if actions.shape != (self.batch_size, self.num_agents):
            raise ValueError(f"actions must have shape {(self.batch_size, self.num_agents)}")
        r1 = torch.zeros(self.batch_size, self.num_agents, dtype=torch.long, device=self.device)
        r2 = torch.zeros_like(r1)
        for n in range(self.num_agents):
            old = self.state[:, n]
            new = self._move(old, actions[:, n])
            r1[:, n], r2[:, n] = self._collect(old, new)
            self.state[:, n] = new
        return self.state.clone(), r1, r2

    def _wall(self, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
        """Wall lookup that is safe for out-of-range indices (callers mask those out)."""
        b = torch.arange(self.batch_size, device=self.device)
        r = rows.clamp(0, self.max_index)
        c = cols.clamp(0, self.max_index)
        return self.table1[b, r, c] == WALL

    def _move(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        r0, r1, c0, c1 = s.unbind(-1)
        m = self.max_index
        vertical = c0 == c1
        horizontal = r0 == r1

        # SWAP: vertical -> horizontal grows to the right of the first cell,
        # horizontal -> vertical grows below it; blocked by the edge or a wall.
        swap_v_blocked = vertical & ((c0 == m) | self._wall(r0, c0 + 1))
        swap_h_blocked = horizontal & ((r0 == m) | self._wall(r0 + 1, c0))
        to_horizontal = vertical & ~swap_v_blocked
        to_vertical = ~vertical & ~swap_h_blocked & horizontal
        swapped = s.clone()
        swapped[:, 1] = torch.where(to_horizontal | to_vertical, r0 + to_vertical.long(), r1)
        swapped[:, 3] = torch.where(to_horizontal | to_vertical, c0 + to_horizontal.long(), c1)
        # The legacy chain checks the vertical case first, so a vertical body never
        # falls through to the horizontal branch.
        swapped = torch.where((vertical & swap_v_blocked).unsqueeze(-1), s, swapped)

        up_blocked = (r0 == 0) | (r1 == 0) | self._wall(r0 - 1, c0) | self._wall(r1 - 1, c1)
        down_blocked = (r0 == m) | (r1 == m) | self._wall(r0 + 1, c0) | self._wall(r1 + 1, c1)
        left_blocked = (c0 == 0) | (c1 == 0) | self._wall(r0, c0 - 1) | self._wall(r1, c1 - 1)
        right_blocked = (c0 == m) | (c1 == m) | self._wall(r0, c0 + 1) | self._wall(r1, c1 + 1)

        delta = torch.zeros_like(s)
        delta[:, 0:2] -= ((a == UP) & ~up_blocked).long().unsqueeze(-1)
        delta[:, 0:2] += ((a == DOWN) & ~down_blocked).long().unsqueeze(-1)
        # Legacy precedence: a blocked UP/DOWN suppresses nothing else, but an UP/DOWN
        # action never moves horizontally, so the column deltas are independent.
        delta[:, 2:4] -= ((a == LEFT) & ~left_blocked).long().unsqueeze(-1)
        delta[:, 2:4] += ((a == RIGHT) & ~right_blocked).long().unsqueeze(-1)
        moved = s + delta

        return torch.where((a == SWAP).unsqueeze(-1), swapped, moved)

    def _collect(self, old: torch.Tensor, new: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = torch.arange(self.batch_size, device=self.device)
        r0, r1, c0, c1 = new.unbind(-1)
        changed = (old != new).any(-1)
        reward1 = self.table1[b, r0, c0] + self.table1[b, r1, c1]
        reward2 = self.table2[b, r0, c0] + self.table2[b, r1, c1]
        reward1 = torch.where(changed, reward1, torch.zeros_like(reward1))
        reward2 = torch.where(changed, reward2, torch.zeros_like(reward2))
        for table in (self.table1, self.table2):
            keep_a = torch.where(changed, torch.zeros_like(r0), table[b, r0, c0])
            table[b, r0, c0] = keep_a
            keep_b = torch.where(changed, torch.zeros_like(r1), table[b, r1, c1])
            table[b, r1, c1] = keep_b
        return reward1, reward2
