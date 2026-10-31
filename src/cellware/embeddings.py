"""Sin-cos embeddings of state, action and reward (torch port of the legacy numpy code).

Ported from ``legacy/jepa_evolution_network/state_action_reward_emb.py``; outputs match
the original to floating-point precision.
"""

from __future__ import annotations

import torch


def sincos(values: torch.Tensor, embed_dim: int) -> torch.Tensor:
    """Embed ``values`` of shape ``(B, M)`` into ``(B, M, embed_dim)``."""
    if embed_dim % 2:
        raise ValueError("embed_dim must be even")
    omega = torch.arange(embed_dim // 2, dtype=torch.float64, device=values.device)
    omega = 1.0 / 10000 ** (omega / (embed_dim / 2.0))
    out = values.to(torch.float64).unsqueeze(-1) * omega
    return torch.cat([out.sin(), out.cos()], dim=-1)


def position_embedding(state: torch.Tensor, size: int, embed_dim: int) -> torch.Tensor:
    """Embed each cell of a ``(B, M, 2*P)`` state by its normalised Manhattan distance."""
    p = state.shape[-1] // 2
    max_distance = 2 * (size - 1)
    parts = [
        sincos((state[..., i] + state[..., i + p]) / max_distance, embed_dim) for i in range(p)
    ]
    return torch.cat(parts, dim=-1)


def state_action_reward_embedding(
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    current_states: torch.Tensor,
    current_actions: torch.Tensor | None = None,
    current_rewards: torch.Tensor | None = None,
    size: int = 10,
    embed_dim: int = 8,
    num_actions: int = 5,
    max_reward: float = 100.0,
) -> torch.Tensor:
    action_emb = sincos(actions / num_actions, embed_dim)
    reward_emb = sincos(rewards / max_reward, embed_dim)
    if current_actions is None:
        cur_action_emb = torch.zeros_like(action_emb)
        cur_reward_emb = torch.zeros_like(reward_emb)
    else:
        cur_action_emb = sincos(current_actions / num_actions, embed_dim)
        cur_reward_emb = sincos(current_rewards / max_reward, embed_dim)
    return torch.cat(
        [
            position_embedding(states, size, embed_dim),
            action_emb,
            reward_emb,
            position_embedding(current_states, size, embed_dim),
            cur_action_emb,
            cur_reward_emb,
        ],
        dim=-1,
    )
