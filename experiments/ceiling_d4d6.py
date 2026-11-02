"""Sampling ceiling of the M-CSF day-4 vs day-6 clone N-fraction correlation (track A follow-up).

Toy with perfectly stable clones: each clone keeps one N share p for both days. p is drawn
from Beta(a, b) fitted (method of moments, corrected for binomial noise) to the observed clone
N shares; each clone is then sampled with its observed day-4 and day-6 N+M counts. Gives the r
that pure sampling noise allows when composition does not change at all.

    uv run python experiments/ceiling_d4d6.py
"""

from __future__ import annotations

import json
import os

import numpy as np
from fit_cytokine import load_counts
from larry_stats import test_stats

HERE = os.path.dirname(__file__)


def main():
    cnt = load_counts()["M"]
    obs = test_stats(cnt[4], cnt[6], boot=2000)
    clones = [c for c in set(cnt[4]) & set(cnt[6])
              if sum(cnt[4][c][:2]) >= 2 and sum(cnt[6][c][:2]) >= 2]
    n4 = np.array([sum(cnt[4][c][:2]) for c in clones])
    n6 = np.array([sum(cnt[6][c][:2]) for c in clones])
    k = np.array([cnt[4][c][0] + cnt[6][c][0] for c in clones])
    n = n4 + n6
    share = k / n
    m, v = share.mean(), share.var() - np.mean(share * (1 - share) / n)  # remove binomial noise
    v = max(v, 1e-4)
    common = m * (1 - m) / v - 1
    a, b = m * common, (1 - m) * common
    rng = np.random.default_rng(0)
    rs = []
    for _ in range(500):
        p = rng.beta(a, b, len(clones))
        d4 = {i: [rng.binomial(n4[i], p[i]), 0] for i in range(len(clones))}
        d6 = {i: [rng.binomial(n6[i], p[i]), 0] for i in range(len(clones))}
        for i in range(len(clones)):
            d4[i][1], d6[i][1] = n4[i] - d4[i][0], n6[i] - d6[i][0]
        rs.append(test_stats(d4, d6, boot=1)["r"])
    out = {"observed_r": obs["r"], "observed_ci": obs["r_ci"], "clones": len(clones),
           "median_cells_d4_d6": [float(np.median(n4)), float(np.median(n6))],
           "beta": [float(a), float(b)], "stable_clone_ceiling_r": float(np.median(rs)),
           "ceiling_range_95": np.percentile(rs, [2.5, 97.5]).tolist()}
    json.dump(out, open(os.path.join(HERE, "results_track_b", "ceiling_d4d6.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
