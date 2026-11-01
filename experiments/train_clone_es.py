"""E10: learn the cells' fate-decision rule from LARRY clone data with evolution strategies.

Non-spatial clone environment matching the E9 lineage model, except that fate decisions come from a
shared policy network instead of fitted equations. Cells divide once a day (committed cells with
probability q); the fate state follows the pitchfork drift with constants fixed from the E6 v6
fit. Each quarter-day an uncommitted cell sees [x, age, memory (2), condition one-hot (3)] and
chooses REST, PUSH_N, PUSH_M or COMMIT_O. The cytokine condition is only an *input*: how cells
respond to it is learned. Fitness = minus the E9 distance between simulated and observed
within-condition clone statistics (full cocktail, G-CSF, M-CSF).

    uv run python experiments/train_clone_es.py --seed 0 --memory inherit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from fit_cytokine import TIME_CONDS, condition_stats, cross_stats, load_counts  # noqa: E402
from fit_larry import THRESH, distance  # noqa: E402

HERE = os.path.dirname(__file__)
STEPS_PER_DAY = 4
PUSH = 0.6  # drift added by PUSH_N / PUSH_M
REST, PUSH_N, PUSH_M, COMMIT_O = range(4)
CONDS = ("full", "G", "M")


class Policy(torch.nn.Module):
    def __init__(self, obs_dim=7, hidden=32):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, 4),
        )

    def forward(self, x):
        return self.net(x)


def simulate(net, landscape, memory, clones=1000, seed=0):
    """Returns {cond: {day: {clone: [N, M, O]}}} like fit_cytokine.load_counts."""
    g = torch.Generator().manual_seed(seed)
    k, sigma, f, q = (landscape[n] for n in ("k", "sigma", "f", "q"))
    dt = 1.0 / STEPS_PER_DAY
    x = 0.05 * torch.randn(clones, 1, generator=g)
    mem = torch.randn(clones, 1, 2, generator=g)
    fate = torch.zeros(clones, 1, dtype=torch.long)  # -1 absent, 0 undecided, 1 N, 2 M, 3 O
    cond = torch.full((clones, 1), -1, dtype=torch.long)
    age = torch.zeros(clones, 1)
    out = {c: {4: {}, 6: {}} for c in CONDS}
    for day in range(1, 7):
        for _ in range(STEPS_PER_DAY):
            free = fate == 0
            onehot = torch.nn.functional.one_hot(cond.clamp(min=0), 3).float()
            onehot = onehot * (cond >= 0)[..., None]
            obs = torch.cat([x[..., None], age[..., None] / STEPS_PER_DAY, mem, onehot], -1)
            with torch.no_grad():
                act = torch.distributions.Categorical(logits=net(obs)).sample(generator=g) \
                    if hasattr(torch.distributions.Categorical, "sample_with_generator") else \
                    torch.multinomial(torch.softmax(net(obs).view(-1, 4), -1), 1,
                                      generator=g).view(x.shape)
            push = PUSH * ((act == PUSH_N).float() - (act == PUSH_M).float())
            noise = torch.randn(x.shape, generator=g)
            x = torch.where(free, x + k * (x - x**3 + push) * dt + sigma * dt**0.5 * noise, x)
            fate = torch.where(free & (act == COMMIT_O), 3, fate)
            free = fate == 0
            fate = torch.where(free & (x > THRESH), 1, fate)
            fate = torch.where(free & (x < -THRESH), 2, fate)
            age = age + 1
        if day in (4, 6):
            pick = (fate > 0) & (torch.rand(fate.shape, generator=g) < f)
            for ci, cname in enumerate(CONDS):
                sel = pick & (cond == ci)
                for j, fv in enumerate((1, 2, 3)):
                    cnt = ((fate == fv) & sel).sum(1)
                    for cl in torch.nonzero(cnt).flatten().tolist():
                        out[cname][day].setdefault(cl, [0, 0, 0])[j] += int(cnt[cl])
        if day < 6:
            x, fate, cond = (v.repeat_interleave(2, 1) for v in (x, fate, cond))
            mem = mem.repeat_interleave(2, 1)
            age = torch.zeros_like(x)
            second = fate[:, 1::2]
            second[(second > 0) & (torch.rand(second.shape, generator=g) >= q)] = -1
            if memory == "inherit":
                mem = mem + 0.1 * torch.randn(mem.shape, generator=g)
            else:
                mem = torch.randn(mem.shape, generator=g)
            if day == 2:
                cond = torch.randint(0, 3, cond.shape, generator=g)
    return out


def fitness_loss(net, landscape, memory, obs, seed):
    sim = condition_stats(simulate(net, landscape, memory, seed=seed))
    total = 0.0
    for c in obs:
        if sim[c]["clones"] < 30:
            return 10.0
        total += distance(sim[c], obs[c])
        if c in TIME_CONDS:
            total += (sim[c]["r_d4_d6"] - obs[c]["r_d4_d6"]) ** 2
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--memory", choices=["inherit", "resample"], default="inherit")
    p.add_argument("--generations", type=int, default=60)
    p.add_argument("--pairs", type=int, default=8)
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--out", default=os.path.join(HERE, "results_clone_es"))
    args = p.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    obs = condition_stats(load_counts())
    v6 = json.load(open(os.path.join(HERE, "results_larry", "fit_v6_resample.json")))["params"]
    landscape = {n: v6[n] for n in ("k", "sigma", "f", "q")}

    net = Policy()
    theta = torch.nn.utils.parameters_to_vector(net.parameters()).detach().requires_grad_()
    opt = torch.optim.Adam([theta], lr=args.lr)
    rng = np.random.default_rng(args.seed)
    history, start = [], time.time()
    for gen in range(args.generations):
        eps = torch.randn(args.pairs, theta.numel())
        sim_seed = int(rng.integers(1 << 30))  # common random numbers within a generation
        losses = torch.zeros(2 * args.pairs)
        for i in range(args.pairs):
            for j, sign in enumerate((1.0, -1.0)):
                torch.nn.utils.vector_to_parameters(theta.detach() + sign * args.sigma * eps[i],
                                                    net.parameters())
                losses[2 * i + j] = fitness_loss(net, landscape, args.memory, obs, sim_seed)
        util = (-losses).argsort().argsort().float() / (len(losses) - 1) - 0.5
        grad = ((util[0::2] - util[1::2]).unsqueeze(1) * eps).sum(0) / (2 * args.pairs * args.sigma)
        opt.zero_grad()
        theta.grad = -grad
        opt.step()
        torch.nn.utils.vector_to_parameters(theta.detach(), net.parameters())
        centre = fitness_loss(net, landscape, args.memory, obs, seed=12345)
        history.append({"generation": gen, "loss": centre, "pop_loss": losses.mean().item()})
        if gen % 5 == 0 or gen == args.generations - 1:
            print(f"[clone-es {args.memory} s{args.seed}] gen {gen:3d}  loss {centre:.4f}  "
                  f"pop {losses.mean():.4f}  ({time.time() - start:.0f}s)", flush=True)

    torch.nn.utils.vector_to_parameters(theta.detach(), net.parameters())
    sims = [simulate(net, landscape, args.memory, seed=s) for s in range(10, 15)]
    pred = {}
    for key in ("full-G", "full-M", "G-M"):
        vals = [cross_stats(c, boot=1)[key]["r"] for c in sims]
        pred[key] = {"r_mean": float(np.mean(vals)), "r_range": [min(vals), max(vals)]}
    os.makedirs(args.out, exist_ok=True)
    name = f"clone_es_{args.memory}_seed{args.seed}"
    torch.save(net.state_dict(), os.path.join(args.out, name + ".pt"))
    result = {"config": vars(args), "landscape": landscape, "history": history,
              "final_loss": history[-1]["loss"], "fit_sim": condition_stats(sims[0]), "pred": pred}
    json.dump(result, open(os.path.join(args.out, name + ".json"), "w"), indent=1)
    print(json.dumps({"final_loss": result["final_loss"], "pred": pred}, indent=1))


if __name__ == "__main__":
    main()
