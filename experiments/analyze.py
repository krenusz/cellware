"""Summarise collective-vs-individual runs: final-window stats per seed, bootstrap CIs, plots.

Usage:
    uv run python experiments/analyze.py [--results experiments/results] [--window 30]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = {
    "mean_agent_return": "Mean return per agent",
    "dish_total_return": "Total return per dish",
    "return_cv_across_agents": "CV of returns across agents",
    "logprob_std_across_agents": "Std of action log-probs across agents",
    "policy_js_divergence": "Policy divergence between agents (JS)",
}
CONDITIONS = ("individual", "collective")


def load(results: Path):
    runs = defaultdict(dict)
    for path in sorted(results.glob("*_seed*.json")):
        condition, seed = path.stem.rsplit("_seed", 1)
        runs[condition][int(seed)] = json.loads(path.read_text(encoding="utf-8"))["history"]
    return runs


def curves(history, metric):
    return np.array([h[metric] for h in history])


def bootstrap_diff(a, b, n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    diffs = rng.choice(b, (n, len(b))).mean(1) - rng.choice(a, (n, len(a))).mean(1)
    return np.percentile(diffs, [2.5, 97.5])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("experiments/results"))
    p.add_argument("--window", type=int, default=30)
    args = p.parse_args()
    runs = load(args.results)

    lines = [
        "# Collective vs individual memory: results",
        "",
        f"Seeds per condition: individual {len(runs['individual'])}, "
        f"collective {len(runs['collective'])}. Values are means over the last "
        f"{args.window} iterations of each run, then mean ± sd across seeds.",
        "",
        "| Metric | Individual | Collective | Difference (coll − ind) | 95% bootstrap CI |",
        "|---|---|---|---|---|",
    ]
    final = {}
    for metric, label in METRICS.items():
        per_cond = {
            c: np.array([curves(h, metric)[-args.window :].mean() for h in runs[c].values()])
            for c in CONDITIONS
        }
        final[metric] = per_cond
        ind, col = per_cond["individual"], per_cond["collective"]
        lo, hi = bootstrap_diff(ind, col)
        flag = "" if lo <= 0 <= hi else " *"
        lines.append(
            f"| {label} | {ind.mean():.4g} ± {ind.std(ddof=1):.2g} | "
            f"{col.mean():.4g} ± {col.std(ddof=1):.2g} | {col.mean() - ind.mean():+.4g}{flag} | "
            f"[{lo:+.4g}, {hi:+.4g}] |"
        )
    lines += ["", "\\* the 95% CI excludes zero.", ""]
    summary = "\n".join(lines)
    (args.results / "summary.md").write_text(summary, encoding="utf-8")
    print(summary.encode("ascii", "replace").decode())

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots (uv sync --extra viz)")
        return

    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 3.4))
    colors = {"individual": "#4C78A8", "collective": "#E45756"}
    kernel = np.ones(9) / 9
    for ax, (metric, label) in zip(axes, METRICS.items(), strict=True):
        for c in CONDITIONS:
            ys = np.stack([curves(h, metric) for h in runs[c].values()])
            smooth = np.stack([np.convolve(y, kernel, mode="valid") for y in ys])
            x = np.arange(smooth.shape[1]) + len(kernel) // 2
            m, s = smooth.mean(0), smooth.std(0)
            ax.plot(x, m, color=colors[c], label=c)
            ax.fill_between(x, m - s, m + s, color=colors[c], alpha=0.2)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("PPO iteration")
    axes[0].legend()
    fig.tight_layout()
    out = args.results / "curves.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
