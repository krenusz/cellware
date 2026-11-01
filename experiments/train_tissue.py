"""Train stem cells (one shared policy) to grow a tissue of target size and fate proportions.

All cells share one actor-critic network (parameter sharing as in MAPPO); cells differ through
their local signals and their heritable memory. Every cell receives the tissue-level reward.
The policy is trained only on decisions of uncommitted stem cells; the critic on every cell.

Knockouts switch off a signalling channel or memory inheritance, e.g.:
    uv run python experiments/train_tissue.py --seed 0 --no_contact --out experiments/results_tissue
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from cellware.tissue import NUM_ACTIONS, TissueConfig, TissueEnv


class SharedCellPolicy(torch.nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 64):
        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Tanh(),
        )
        self.pi = torch.nn.Linear(hidden, NUM_ACTIONS)
        self.v = torch.nn.Linear(hidden, 1)
        for m in self.body:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight, math.sqrt(2))
                torch.nn.init.zeros_(m.bias)
        torch.nn.init.orthogonal_(self.pi.weight, 0.01)
        torch.nn.init.zeros_(self.pi.bias)

    def forward(self, x):
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)


def rollout(env: TissueEnv, net: SharedCellPolicy):
    obs = env.reset()
    buf = {k: [] for k in ("obs", "act", "logp", "agent", "alive", "rew")}
    done = False
    while not done:
        with torch.no_grad():
            logits, _ = net(obs)
        dist = torch.distributions.Categorical(logits=logits)
        act = dist.sample()
        buf["obs"].append(obs)
        buf["act"].append(act)
        buf["logp"].append(dist.log_prob(act))
        buf["agent"].append(env.agents.clone())
        buf["alive"].append(env.occ.clone())
        obs, rew, done = env.step(act)
        buf["rew"].append(rew)
    return {k: torch.stack(v) for k, v in buf.items()}  # leading dim T


def update(net, opt, data, args, entropy_coef):
    obs, act, logp = data["obs"], data["act"], data["logp"]  # (T, B, H, W, ...)
    agent, alive = data["agent"], data["alive"]
    T = obs.shape[0]
    rew = data["rew"].view(T, -1, 1, 1).expand_as(alive).float()
    with torch.no_grad():
        _, values = net(obs)
        values = values * alive
        adv = torch.zeros_like(values)
        last = torch.zeros_like(values[0])
        for t in reversed(range(T)):
            next_v = values[t + 1] if t + 1 < T else torch.zeros_like(last)
            delta = rew[t] + args.gamma * next_v - values[t]
            last = (delta + args.gamma * args.gae_lambda * last) * alive[t]
            adv[t] = last
        ret = adv + values

    a_obs, a_act, a_logp, a_adv = obs[agent], act[agent], logp[agent], adv[agent]
    a_adv = (a_adv - a_adv.mean()) / (a_adv.std() + 1e-8)
    c_obs, c_ret = obs[alive], ret[alive]
    stats = {}
    for _ in range(args.epochs):
        for idx_a, idx_c in zip(
            torch.randperm(len(a_act)).chunk(args.minibatches),
            torch.randperm(len(c_ret)).chunk(args.minibatches),
            strict=False,
        ):
            logits, _ = net(a_obs[idx_a])
            dist = torch.distributions.Categorical(logits=logits)
            ratio = (dist.log_prob(a_act[idx_a]) - a_logp[idx_a]).exp()
            adv_mb = a_adv[idx_a]
            pg = -torch.min(ratio * adv_mb, ratio.clamp(1 - args.clip, 1 + args.clip) * adv_mb)
            _, v = net(c_obs[idx_c])
            vf = 0.5 * (v - c_ret[idx_c]).pow(2).mean()
            ent = dist.entropy().mean()
            loss = pg.mean() + vf - entropy_coef * ent
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            stats = {"entropy": ent.item(), "value_loss": vf.item()}
    return stats


def condition_name(args) -> str:
    parts = [
        name
        for name, off in (
            ("nocontact", args.no_contact),
            ("nomorphogen", args.no_morphogen),
            ("nogap", args.no_gap),
            ("nomemory", args.no_memory),
        )
        if off
    ]
    return "_".join(parts) or "full"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tissues", type=int, default=16)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--entropy_coef", type=float, default=0.01, help="final entropy bonus")
    p.add_argument(
        "--entropy_start", type=float, default=None, help="initial bonus, annealed linearly"
    )
    p.add_argument("--min_commit_size", type=int, default=1, help="totipotency window size")
    p.add_argument("--no_contact", action="store_true")
    p.add_argument("--no_morphogen", action="store_true")
    p.add_argument("--no_gap", action="store_true")
    p.add_argument("--no_memory", action="store_true", help="resample memory at division")
    p.add_argument("--spatial", action="store_true", help="add the A-core / B-rim target")
    p.add_argument("--global_readout", action="store_true", help="show global size and time")
    p.add_argument("--cue", action="store_true", help="lineage-cue task (E7)")
    p.add_argument("--threads", type=int, default=2, help="CPU threads per run")
    p.add_argument("--out", type=Path, default=Path("experiments/results_tissue"))
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    cfg = TissueConfig(
        contact=not args.no_contact,
        morphogen=not args.no_morphogen,
        gap_junction=not args.no_gap,
        inherit_memory=not args.no_memory,
        spatial_target=args.spatial,
        global_readout=args.global_readout,
        min_commit_size=args.min_commit_size,
        cue_target=args.cue,
    )
    env = TissueEnv(cfg, batch_size=args.tissues, seed=args.seed)
    net = SharedCellPolicy(env.obs_dim)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    name = condition_name(args)

    history = []
    start = time.time()
    for it in range(args.iterations):
        data = rollout(env, net)
        metrics = env.metrics()
        start_coef = args.entropy_coef if args.entropy_start is None else args.entropy_start
        frac = it / max(1, args.iterations - 1)
        coef = start_coef + (args.entropy_coef - start_coef) * frac
        metrics.update(update(net, opt, data, args, coef))
        metrics["entropy_coef"] = coef
        metrics["iteration"] = it
        history.append(metrics)
        if it % 10 == 0 or it == args.iterations - 1:
            print(
                f"[{name} s{args.seed}] it {it:4d}  score {metrics['score']:+.3f}  "
                f"size {metrics['size']:5.1f}  S/A/B {metrics['prop_stem']:.2f}/"
                f"{metrics['prop_a']:.2f}/{metrics['prop_b']:.2f}  "
                f"pattern {metrics['pattern_accuracy']:.2f}  "
                f"sisters {metrics['sister_concordance']:.2f}  "
                f"segr {metrics['segregation']:+.2f}  ({time.time() - start:.0f}s)",
                flush=True,
            )

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{name}_seed{args.seed}.json"
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    path.write_text(json.dumps({"config": config, "history": history}), encoding="utf-8")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
