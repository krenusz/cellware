"""CAND-I power design check (pre-registered in EXPERIMENT_LOG, 2026-11-23; Bence: "Yes, seal CAND-I").

Simulation only: no file from the germline set is read. The test unit is the fish. Each fish gives one excess value
(observed minus replicate-calibrated null), simulated as normal with mean d and SD 1. The test is an exact one-sided
sign-flip test on the mean at alpha 0.05, with 4 (one sealed partition), 8 (two) or 12 (all three) eligible fish.
"""

import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_cand", "candi_power.json")
ALPHA, SIMS, SEED = 0.05, 4000, 20261123
FISH = (4, 8, 12)
EFFECTS = (0.5, 0.8, 1.0, 1.5, 2.0)


def signflip_p(x, signs):
    """Exact one-sided p per row of x (sims by fish)."""
    obs = x.mean(axis=1)
    return np.array([(((signs * row).mean(axis=1)) >= m - 1e-12).mean() for row, m in zip(x, obs)])


def main():
    rng = np.random.default_rng(SEED)
    out = {"alpha": ALPHA, "sims": SIMS, "seed": SEED, "by_fish": {}}
    for n in FISH:
        signs = np.array(list(itertools.product((-1, 1), repeat=n)), dtype=float)
        row = {"smallest_possible_p": 1 / len(signs), "power_by_effect": {}}
        for d in EFFECTS:
            x = rng.normal(d, 1, (SIMS, n))
            row["power_by_effect"][str(d)] = round(float((signflip_p(x, signs) <= ALPHA).mean()), 3)
        row["null_rejection"] = round(float((signflip_p(rng.normal(0, 1, (SIMS, n)), signs) <= ALPHA).mean()), 3)
        out["by_fish"][str(n)] = row
    out["one_partition_passes"] = out["by_fish"]["4"]["power_by_effect"]["1.0"] >= 0.8
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
