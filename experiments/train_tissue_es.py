"""E8: train the shared cell policy with evolution strategies instead of PPO.

OpenAI-style ES with antithetic sampling and rank-normalised fitness. Fitness is the mean final
tissue score of an episode (the episode reward telescopes to the score change). Same environment
and policy network as train_tissue.py; the value head is unused.

    uv run python experiments/train_tissue_es.py --seed 0 --spatial --min_commit_size 16
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from train_tissue import SharedCellPolicy  # noqa: E402

from cellware.tissue import TissueConfig, TissueEnv  # noqa: E402


def episode(env: TissueEnv, net: SharedCellPolicy) -> torch.Tensor:
    obs = env.reset()
    done = False
    while not done:
        with torch.no_grad():
            logits, _ = net(obs)
        act = torch.distributions.Categorical(logits=logits).sample()
        obs, _, done = env.step(act)
    return env.score()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--generations", type=int, default=50)
    p.add_argument("--pairs", type=int, default=8, help="antithetic pairs per generation")
    p.add_argument("--tissues", type=int, default=8, help="tissues per fitness evaluation")
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--min_commit_size", type=int, default=16)
    p.add_argument("--spatial", action="store_true")
    p.add_argument("--no_memory", action="store_true")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--out", type=Path, default=Path("experiments/results_es"))
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    cfg = TissueConfig(spatial_target=args.spatial, min_commit_size=args.min_commit_size,
                       inherit_memory=not args.no_memory)
    env = TissueEnv(cfg, batch_size=args.tissues, seed=args.seed)
    eval_env = TissueEnv(cfg, batch_size=16, seed=args.seed + 1000)
    net = SharedCellPolicy(env.obs_dim)
    theta = torch.nn.utils.parameters_to_vector(net.parameters()).detach()
    opt = torch.optim.Adam([theta.requires_grad_()], lr=args.lr)
    name = "es" + ("_nomemory" if args.no_memory else "")

    history, start = [], time.time()
    for gen in range(args.generations):
        eps = torch.randn(args.pairs, theta.numel())
        fit = torch.zeros(2 * args.pairs)
        for i in range(args.pairs):
            for j, sign in enumerate((1.0, -1.0)):
                torch.nn.utils.vector_to_parameters(theta.detach() + sign * args.sigma * eps[i],
                                                    net.parameters())
                fit[2 * i + j] = episode(env, net).mean()
        ranks = fit.argsort().argsort().float()
        util = ranks / (len(fit) - 1) - 0.5
        grad = ((util[0::2] - util[1::2]).unsqueeze(1) * eps).sum(0) / (2 * args.pairs * args.sigma)
        opt.zero_grad()
        theta.grad = -grad
        opt.step()

        torch.nn.utils.vector_to_parameters(theta.detach(), net.parameters())
        episode(eval_env, net)
        metrics = eval_env.metrics()
        metrics.update({"generation": gen, "pop_fitness_mean": fit.mean().item(),
                        "episodes": (gen + 1) * 2 * args.pairs * args.tissues})
        history.append(metrics)
        if gen % 5 == 0 or gen == args.generations - 1:
            print(f"[{name} s{args.seed}] gen {gen:3d}  score {metrics['score']:+.3f}  "
                  f"size {metrics['size']:5.1f}  S/A/B {metrics['prop_stem']:.2f}/"
                  f"{metrics['prop_a']:.2f}/{metrics['prop_b']:.2f}  "
                  f"pattern {metrics['pattern_accuracy']:.2f}  ({time.time() - start:.0f}s)",
                  flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{name}_seed{args.seed}.json"
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    path.write_text(json.dumps({"config": config, "history": history}), encoding="utf-8")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
