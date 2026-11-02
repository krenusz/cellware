"""Training-health check for sweep outputs: flags runs whose result reflects a training failure.

    uv run python experiments/health_check.py "experiments/results_clone_e11/a_*.json"

A run fails if any learned scalar moved < 0.05 from its start (optimisation-limited, as in E9 v8
and E15), if a condition kept < 50 sampled clones, or if the loss still trends down over the last 60
generations by more than 2x its evaluation noise and 5% (not converged).
"""

from __future__ import annotations

import glob
import json
import sys

import numpy as np


def check(d):
    h, problems = d.get("health", {}), []
    for name, (start, end) in h.get("scalars", {}).items():
        if np.max(np.abs(np.array(end) - np.array(start))) < 0.05:
            problems.append(f"{name} did not move")
    if h.get("min_clones", 1e9) < 50:
        problems.append(f"min clones {h['min_clones']}")
    loss = np.array([g["loss"] for g in d["history"]])
    if len(loss) >= 60:  # noise-tolerant: trend over the last 60 generations vs evaluation noise
        t = np.arange(60)
        slope, icpt = np.polyfit(t, loss[-60:], 1)
        noise = np.std(loss[-60:] - (slope * t + icpt))
        drop = -slope * 60
        if drop > 2 * noise and drop > 0.05 * np.mean(loss[-60:]):
            problems.append("loss still falling")
    return problems


if __name__ == "__main__":
    files = sorted(f for pat in sys.argv[1:] for f in glob.glob(pat))
    bad = 0
    for f in files:
        p = check(json.load(open(f)))
        bad += bool(p)
        if p:
            print(f"FAIL {f}: {'; '.join(p)}")
    print(f"{len(files) - bad}/{len(files)} healthy")
