"""Collective vs individual memory in a shared petri dish (re-run of the 2023 Tufts experiment).

N agents share one dish and compete for depleting rewards. Each agent has its own PPO
actor-critic. In the *individual* condition an agent learns only from its own trajectories;
in the *collective* condition every agent learns from the pooled trajectories of all agents
(the shared-memory mode of the original code). B dishes run in parallel per seed.

Usage:
    uv run python experiments/collective_vs_individual.py --condition collective --seed 0
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from cellware.envs.petri_dish import NUM_ACTIONS, WALL, PetriDish

MAX_CELL_REWARD = 90.0


def observe(env: PetriDish, t: int, horizon: int) -> torch.Tensor:
    """Per-agent observation (B, N, D): own coordinates, time, and a 3x3 view around each cell."""
    table = F.pad(env.table1, (1, 1, 1, 1), value=WALL)  # (B, H+2, W+2)
    b = torch.arange(env.batch_size).view(-1, 1, 1, 1)
    offsets = torch.arange(-1, 2)
    dr, dc = torch.meshgrid(offsets, offsets, indexing="ij")
    views = []
    for rows, cols in (
        (env.state[..., 0], env.state[..., 2]),
        (env.state[..., 1], env.state[..., 3]),
    ):
        r = rows.unsqueeze(-1).unsqueeze(-1) + 1 + dr  # (B, N, 3, 3)
        c = cols.unsqueeze(-1).unsqueeze(-1) + 1 + dc
        patch = table[b, r, c].flatten(-2)
        views += [patch.clamp(min=0) / MAX_CELL_REWARD, (patch == WALL).float()]
    coords = env.state.float() / env.max_index
    time_feat = torch.full_like(coords[..., :1], t / horizon)
    return torch.cat([coords, time_feat, *views], dim=-1)


class AgentNets(torch.nn.Module):
    """N independent actor-critic MLPs evaluated in one batched call."""

    def __init__(self, num_agents: int, obs_dim: int, hidden: int = 64):
        super().__init__()

        def layer(i, o):
            w = torch.empty(num_agents, i, o)
            for n in range(num_agents):
                torch.nn.init.orthogonal_(w[n], gain=math.sqrt(2))
            return torch.nn.Parameter(w), torch.nn.Parameter(torch.zeros(num_agents, 1, o))

        self.w1, self.b1 = layer(obs_dim, hidden)
        self.w2, self.b2 = layer(hidden, hidden)
        self.wpi, self.bpi = layer(hidden, NUM_ACTIONS)
        self.wv, self.bv = layer(hidden, 1)
        with torch.no_grad():
            self.wpi.mul_(0.01)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (N, K, D) -> logits (N, K, A), values (N, K)."""
        h = torch.tanh(x @ self.w1 + self.b1)
        h = torch.tanh(h @ self.w2 + self.b2)
        return h @ self.wpi + self.bpi, (h @ self.wv + self.bv).squeeze(-1)


def rollout(env, nets, horizon):
    env.reset()
    obs, acts, logps, rews = [], [], [], []
    for t in range(horizon):
        o = observe(env, t, horizon)  # (B, N, D)
        with torch.no_grad():
            logits, _ = nets(o.transpose(0, 1))  # (N, B, A)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        _, r1, _ = env.step(a.T)
        obs.append(o)
        acts.append(a.T)
        logps.append(dist.log_prob(a).T)
        rews.append(r1.float() / MAX_CELL_REWARD)
    return (
        torch.stack(obs),
        torch.stack(acts),
        torch.stack(logps),
        torch.stack(rews),
    )  # (T, B, N, ...)


def gae(rewards, values, gamma, lam):
    """rewards/values: (..., T) with the episode ending after the last step."""
    adv = torch.zeros_like(rewards)
    last = torch.zeros_like(rewards[..., 0])
    for t in reversed(range(rewards.shape[-1])):
        next_v = values[..., t + 1] if t + 1 < rewards.shape[-1] else torch.zeros_like(last)
        delta = rewards[..., t] + gamma * next_v - values[..., t]
        last = delta + gamma * lam * last
        adv[..., t] = last
    return adv, adv + values


def update(nets, opt, data, collective, args):
    obs, acts, logps, rews = data  # (T, B, N, ...)
    T, B, N, D = obs.shape
    # Per-agent training sets X[n]: own column (individual) or all columns (collective).
    if collective:
        X = obs.permute(2, 0, 1, 3).reshape(1, N * T * B, D).expand(N, -1, -1)
        src_shape = (N, N, T, B)  # (learner, source agent, T, B)
        A = acts.permute(2, 0, 1).reshape(1, -1).expand(N, -1)
        L = logps.permute(2, 0, 1).reshape(1, -1).expand(N, -1)
        R = rews.permute(2, 0, 1).reshape(1, N, T, B).expand(N, -1, -1, -1)
    else:
        X = obs.permute(2, 0, 1, 3).reshape(N, T * B, D)
        src_shape = (N, 1, T, B)
        A = acts.permute(2, 0, 1).reshape(N, -1)
        L = logps.permute(2, 0, 1).reshape(N, -1)
        R = rews.permute(2, 0, 1).reshape(N, 1, T, B)
    with torch.no_grad():
        _, v = nets(X)
        v = v.reshape(src_shape)
        adv, ret = gae(R.transpose(-1, -2), v.transpose(-1, -2), args.gamma, args.gae_lambda)
        adv = adv.transpose(-1, -2).reshape(N, -1)
        ret = ret.transpose(-1, -2).reshape(N, -1)
        adv = (adv - adv.mean(1, keepdim=True)) / (adv.std(1, keepdim=True) + 1e-8)
    K = X.shape[1]
    stats = {}
    for _ in range(args.epochs):
        perm = torch.randperm(K)
        for idx in perm.chunk(args.minibatches):
            logits, values = nets(X[:, idx])
            dist = torch.distributions.Categorical(logits=logits)
            ratio = (dist.log_prob(A[:, idx]) - L[:, idx]).exp()
            a_mb = adv[:, idx]
            pg = -torch.min(ratio * a_mb, ratio.clamp(1 - args.clip, 1 + args.clip) * a_mb).mean(1)
            vf = 0.5 * (values - ret[:, idx]).pow(2).mean(1)
            ent = dist.entropy().mean(1)
            loss = (pg + vf - args.entropy_coef * ent).sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(nets.parameters(), 0.5)
            opt.step()
            stats = {"entropy": ent.mean().item(), "value_loss": vf.mean().item()}
    return stats


def coordination_metrics(nets, obs, logps, rews):
    T, B, N, _ = obs.shape
    returns = rews.sum(0)  # (B, N)
    total = returns.sum(1)  # (B,)
    cv = (returns.std(1) / returns.mean(1).clamp(min=1e-8)).mean()
    with torch.no_grad():
        # every agent's action distribution on a shared sample of visited states
        sample = obs.reshape(-1, obs.shape[-1])[torch.randperm(T * B * N)[:2048]]
        logits, _ = nets(sample.unsqueeze(0).expand(N, -1, -1))
        p = logits.softmax(-1)  # (N, K, A)
        m = 0.5 * (p.unsqueeze(0) + p.unsqueeze(1))
        kl = lambda a, b: (a * (a.clamp_min(1e-12).log() - b.clamp_min(1e-12).log())).sum(-1)  # noqa: E731
        js = 0.5 * (kl(p.unsqueeze(1).expand_as(m), m) + kl(p.unsqueeze(0).expand_as(m), m))
        off_diag = ~torch.eye(N, dtype=torch.bool)
        policy_js = js.mean(-1)[off_diag].mean()
    return {
        "mean_agent_return": returns.mean().item() * MAX_CELL_REWARD,
        "dish_total_return": total.mean().item() * MAX_CELL_REWARD,
        "return_cv_across_agents": cv.item(),
        "logprob_std_across_agents": logps.std(2).mean().item(),
        "policy_js_divergence": policy_js.item(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", choices=["individual", "collective"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--agents", type=int, default=4)
    p.add_argument("--dishes", type=int, default=16)
    p.add_argument("--horizon", type=int, default=128)
    p.add_argument("--iterations", type=int, default=150)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--entropy_coef", type=float, default=0.01)
    p.add_argument("--out", type=Path, default=Path("experiments/results"))
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    env = PetriDish(batch_size=args.dishes, num_agents=args.agents)
    obs_dim = observe(env, 0, args.horizon).shape[-1]
    nets = AgentNets(args.agents, obs_dim)
    opt = torch.optim.Adam(nets.parameters(), lr=args.lr, eps=1e-5)
    collective = args.condition == "collective"

    history = []
    start = time.time()
    for it in range(args.iterations):
        data = rollout(env, nets, args.horizon)
        metrics = coordination_metrics(nets, data[0], data[2], data[3])
        metrics.update(update(nets, opt, data, collective, args))
        metrics["iteration"] = it
        history.append(metrics)
        if it % 10 == 0 or it == args.iterations - 1:
            print(
                f"[{args.condition} s{args.seed}] it {it:4d}  "
                f"agent_return {metrics['mean_agent_return']:7.1f}  "
                f"dish_total {metrics['dish_total_return']:7.1f}  "
                f"cv {metrics['return_cv_across_agents']:.3f}  "
                f"js {metrics['policy_js_divergence']:.4f}  "
                f"({time.time() - start:.0f}s)",
                flush=True,
            )

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.condition}_seed{args.seed}.json"
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    path.write_text(json.dumps({"config": config, "history": history}), encoding="utf-8")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
