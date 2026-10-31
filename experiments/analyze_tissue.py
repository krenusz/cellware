"""Compare tissue knockouts against the full model.

Usage:
    uv run python experiments/analyze_tissue.py [--results experiments/results_tissue] [--window 30]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = {
    "score": "Score (0 = perfect)",
    "size": "Tissue size (target 60)",
    "prop_stem": "Stem share (target 0.10)",
    "prop_a": "A share (target 0.60)",
    "prop_b": "B share (target 0.30)",
    "pattern_accuracy": "Pattern accuracy (A core, B rim)",
    "sister_concordance": "Sister fate concordance",
    "segregation": "Fate segregation",
}
ORDER = ["full", "nocontact", "nomorphogen", "nogap", "nomemory"]


def load(results: Path):
    runs = defaultdict(dict)
    for path in sorted(results.glob("*_seed*.json")):
        cond, seed = path.stem.rsplit("_seed", 1)
        runs[cond][int(seed)] = json.loads(path.read_text(encoding="utf-8"))["history"]
    return runs


def final(history, metric, window):
    vals = np.array([h[metric] for h in history[-window:]], dtype=float)
    return np.nanmean(vals)


def bootstrap_diff(a, b, n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    d = rng.choice(b, (n, len(b))).mean(1) - rng.choice(a, (n, len(a))).mean(1)
    return np.percentile(d, [2.5, 97.5])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("experiments/results_tissue"))
    p.add_argument("--window", type=int, default=30)
    args = p.parse_args()
    runs = load(args.results)
    conds = [c for c in ORDER if c in runs] + sorted(set(runs) - set(ORDER))
    first = next(iter(runs[conds[0]].values()))[0]
    metrics = {k: v for k, v in METRICS.items() if k in first}

    vals = {
        c: {m: np.array([final(h, m, args.window) for h in runs[c].values()]) for m in metrics}
        for c in conds
    }
    lines = [
        "# Tissue knockout study",
        "",
        f"Means over the last {args.window} iterations per run, then mean (sd) across seeds. "
        "Knockout rows show the difference from `full`; * marks a 95% bootstrap CI excluding 0.",
        "",
        "| Condition | n | " + " | ".join(metrics.values()) + " |",
        "|---|---|" + "---|" * len(metrics),
    ]
    for c in conds:
        cells = []
        for m in metrics:
            v = vals[c][m]
            if c == "full" or "full" not in vals:
                cells.append(f"{v.mean():.3f} ({v.std(ddof=1):.3f})")
            else:
                lo, hi = bootstrap_diff(vals["full"][m], v)
                star = "" if lo <= 0 <= hi else "*"
                cells.append(f"{v.mean() - vals['full'][m].mean():+.3f}{star}")
        lines.append(f"| {c} | {len(runs[c])} | " + " | ".join(cells) + " |")
    summary = "\n".join(lines) + "\n"
    (args.results / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    shown = [m for m in ("score", "size", "pattern_accuracy", "sister_concordance") if m in metrics]
    fig, axes = plt.subplots(1, len(shown), figsize=(4.3 * len(shown), 3.4))
    kernel = np.ones(9) / 9
    for ax, m in zip(axes, shown, strict=True):
        for c in conds:
            ys = np.stack([[h[m] for h in hist] for hist in runs[c].values()]).astype(float)
            ys = np.where(np.isnan(ys), np.nanmean(ys), ys)
            sm = np.stack([np.convolve(y, kernel, mode="valid") for y in ys])
            x = np.arange(sm.shape[1]) + len(kernel) // 2
            ax.plot(x, sm.mean(0), label=c, lw=1.6 if c == "full" else 1.0)
        ax.set_title(metrics[m], fontsize=9)
        ax.set_xlabel("PPO iteration")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.results / "curves.png", dpi=130)
    print("saved", args.results / "curves.png")


if __name__ == "__main__":
    main()
