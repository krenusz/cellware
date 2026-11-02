"""E11: learned fate rule with condition-specific selection, all five cytokine conditions.

Extends E10 (train_clone_es.py) in three ways, all motivated by the E9/E10 held-out result
(sisters in G-CSF vs M-CSF agree, r = 0.51; every model predicted ~0-0.2):
  1. Selection: committed cells also consult the policy, at each daily division, to divide or
     arrest (replaces the fixed q). A cytokine can now favour a lineage without redirecting
     undecided cells.
  2. The fitness includes the cross-condition concordance for (full, G), (full, M), (G, M),
     now spent as a test and therefore usable as training data.
  3. Epo and SCF are simulated and fitted on their within-condition statistics. Their
     cross-condition concordance (pooled over every pair that includes Epo or SCF) is the new
     held-out test, computed only by `--test`.

Observation: [x, age, memory (2), condition one-hot (5), committed fate one-hot (3)].

    uv run python experiments/train_clone_e11.py --seed 0 --memory inherit
    uv run python experiments/train_clone_e11.py --test      # the single look, after all runs
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
from fit_cytokine import (  # noqa: E402
    CONDS5,
    TIME_CONDS,
    condition_stats,
    cross_stats,
    load_counts,
    pooled_pairs_r,
)
from fit_larry import THRESH, distance  # noqa: E402

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "results_clone_e11")
STEPS_PER_DAY = 4
PUSH = 0.6
REST, PUSH_N, PUSH_M, COMMIT_O = range(4)
CONDS = tuple(CONDS5.values())  # full, G, M, Epo, SCF
TRAIN_PAIRS = ("full-G", "full-M", "G-M")
W_CROSS = 2.0  # weight of the cross-condition terms in the loss
FIX_B = None  # E13 scan: fixed founder strength (--fix_b)


class Policy(torch.nn.Module):
    def __init__(self, obs_dim=12, hidden=32, trait=False, founder=False):
        super().__init__()
        # E13: strength of an exactly inherited founder N/M propensity in the fate drift (learned)
        self.founder_scale = torch.nn.Parameter(torch.tensor(1.0)) if founder else None
        # E12: strength of a heritable clone trait that tilts N vs M expansion (learned)
        self.trait_scale = torch.nn.Parameter(torch.tensor(0.5)) if trait else None
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, 6),  # 4 fate actions (undecided) + divide/arrest (committed)
        )

    def forward(self, x):
        return self.net(x)


def observe(x, age, mem, cond, fate):
    onehot = torch.nn.functional.one_hot(cond.clamp(min=0), len(CONDS)).float()
    onehot = onehot * (cond >= 0)[..., None]
    fate_oh = torch.nn.functional.one_hot((fate - 1).clamp(min=0), 3).float()
    fate_oh = fate_oh * (fate > 0)[..., None]
    return torch.cat([x[..., None], age[..., None] / STEPS_PER_DAY, mem, onehot, fate_oh], -1)


def simulate(net, landscape, memory, clones=1500, seed=0):
    """Returns {cond: {day: {clone: [N, M, O]}}} like fit_cytokine.load_counts."""
    g = torch.Generator().manual_seed(seed)
    k, sigma, f = (landscape[n] for n in ("k", "sigma", "f"))
    dt = 1.0 / STEPS_PER_DAY
    x = 0.05 * torch.randn(clones, 1, generator=g)
    mem = torch.randn(clones, 1, 2, generator=g)
    fate = torch.zeros(clones, 1, dtype=torch.long)  # -1 absent, 0 undecided, 1 N, 2 M, 3 O
    cond = torch.full((clones, 1), -1, dtype=torch.long)
    age = torch.zeros(clones, 1)
    trait = torch.randn(clones, 1, generator=g)  # E12: fixed per clone, shared by all its cells
    founder = torch.randn(clones, 1, generator=g)  # E13: founder propensity, exactly inherited
    out = {c: {4: {}, 6: {}} for c in CONDS}
    for day in range(1, 7):
        for _ in range(STEPS_PER_DAY):
            free = fate == 0
            with torch.no_grad():
                logits = net(observe(x, age, mem, cond, fate))[..., :4]
            act = torch.multinomial(torch.softmax(logits.reshape(-1, 4), -1), 1,
                                    generator=g).view(x.shape)
            push = PUSH * ((act == PUSH_N).float() - (act == PUSH_M).float())
            noise = torch.randn(x.shape, generator=g)
            if net.founder_scale is not None:  # same bias for every cell of the clone, all days
                push = push + net.founder_scale.detach() * founder
            elif FIX_B is not None:  # E13 scan: founder strength fixed, not learned
                push = push + FIX_B * founder
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
            with torch.no_grad():  # selection: committed cells divide or arrest
                logit = net(observe(x, age, mem, cond, fate))[..., 4:]
                logit = logit[..., 0] - logit[..., 1]
                if net.trait_scale is not None:  # high-trait clones expand N, low-trait expand M
                    tilt = net.trait_scale * trait.expand_as(logit)
                    logit = logit + tilt * ((fate == 1).float() - (fate == 2).float())
                p_div = torch.sigmoid(logit)
            divides = torch.rand(p_div.shape, generator=g) < p_div
            x, fate, cond, divides = (v.repeat_interleave(2, 1) for v in (x, fate, cond, divides))
            mem = mem.repeat_interleave(2, 1)
            age = torch.zeros_like(x)
            second = fate[:, 1::2]
            second[(second > 0) & ~divides[:, 1::2]] = -1
            if memory == "inherit":
                mem = mem + 0.1 * torch.randn(mem.shape, generator=g)
            else:
                mem = torch.randn(mem.shape, generator=g)
            if day == 2:
                cond = torch.randint(0, len(CONDS), cond.shape, generator=g)
    return out


def fitness_loss(net, landscape, memory, obs, obs_cross, seed):
    cnt = simulate(net, landscape, memory, seed=seed)
    sim = condition_stats(cnt)
    total = 0.0
    for c in obs:
        if sim[c]["clones"] < 20:
            return 10.0
        total += distance(sim[c], obs[c])
        if c in TIME_CONDS:
            total += (sim[c]["r_d4_d6"] - obs[c]["r_d4_d6"]) ** 2
    cross = cross_stats({c: cnt[c] for c in ("full", "G", "M")}, boot=1)
    for key in TRAIN_PAIRS:
        total += W_CROSS * (cross[key]["r"] - obs_cross[key]) ** 2
    return total


def test_look():
    raise SystemExit("the Epo/SCF test is issued only by experiments/test_oracle.py")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--memory", choices=["inherit", "resample"], default="inherit")
    p.add_argument("--generations", type=int, default=150)
    p.add_argument("--pairs", type=int, default=8)
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--test", action="store_true", help="the single look at the held-out test")
    p.add_argument("--trait", action="store_true", help="E12: heritable clone expansion trait")
    p.add_argument("--founder", action="store_true", help="E13: inherited founder fate propensity")
    p.add_argument("--fix_b", type=float, default=None, help="E13 scan: fixed founder strength")
    p.add_argument("--w_cross", type=float, default=2.0)
    args = p.parse_args()
    if args.test:
        test_look()
        return
    global FIX_B, W_CROSS
    FIX_B, W_CROSS = args.fix_b, args.w_cross
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    cnt_obs = load_counts(CONDS5)
    obs = condition_stats(cnt_obs)
    cross = cross_stats({c: cnt_obs[c] for c in ("full", "G", "M")}, boot=1)
    obs_cross = {k: cross[k]["r"] for k in TRAIN_PAIRS}
    v6 = json.load(open(os.path.join(HERE, "results_larry", "fit_v6_resample.json")))["params"]
    landscape = {n: v6[n] for n in ("k", "sigma", "f")}

    net = Policy(trait=args.trait, founder=args.founder and FIX_B is None)
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
                losses[2 * i + j] = fitness_loss(net, landscape, args.memory, obs, obs_cross,
                                                 sim_seed)
        util = (-losses).argsort().argsort().float() / (len(losses) - 1) - 0.5
        grad = ((util[0::2] - util[1::2]).unsqueeze(1) * eps).sum(0) / (2 * args.pairs * args.sigma)
        opt.zero_grad()
        theta.grad = -grad
        opt.step()
        torch.nn.utils.vector_to_parameters(theta.detach(), net.parameters())
        centre = fitness_loss(net, landscape, args.memory, obs, obs_cross, seed=12345)
        history.append({"generation": gen, "loss": centre, "pop_loss": losses.mean().item()})
        if gen % 5 == 0 or gen == args.generations - 1:
            print(f"[e11 {args.memory} s{args.seed}] gen {gen:3d}  loss {centre:.4f}  "
                  f"pop {losses.mean():.4f}  ({time.time() - start:.0f}s)", flush=True)

    torch.nn.utils.vector_to_parameters(theta.detach(), net.parameters())
    sims = [simulate(net, landscape, args.memory, seed=s) for s in range(10, 15)]
    crosses = [cross_stats({c: s[c] for c in ("full", "G", "M")}, boot=1) for s in sims]
    pooled_r = [pooled_pairs_r(s, boot=1)["r"] for s in sims]
    pred = {k: {"r_mean": float(np.mean([c[k]["r"] for c in crosses]))} for k in TRAIN_PAIRS}
    pred["pooled_epo_scf"] = {"r_mean": float(np.mean(pooled_r)),
                              "r_range": [min(pooled_r), max(pooled_r)]}
    os.makedirs(OUT, exist_ok=True)
    tag = 'e13' if args.founder else 'e12' if args.trait else 'e11'
    if FIX_B is not None:
        tag = f"e13scan_b{FIX_B:g}_w{W_CROSS:g}"
    name = f"{tag}_{args.memory}_seed{args.seed}"
    torch.save(net.state_dict(), os.path.join(OUT, name + ".pt"))
    if args.trait:
        pred["trait_scale"] = float(net.trait_scale)
    if net.founder_scale is not None:
        pred["founder_scale"] = float(net.founder_scale)
    result = {"config": vars(args), "landscape": landscape, "obs_cross": obs_cross,
              "history": history, "final_loss": history[-1]["loss"],
              "fit_sim": condition_stats(sims[0]), "pred": pred}
    json.dump(result, open(os.path.join(OUT, name + ".json"), "w"), indent=1)
    print(json.dumps({"final_loss": result["final_loss"], "pred": pred}, indent=1))


if __name__ == "__main__":
    main()
