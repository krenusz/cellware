"""E19c: E19's kin-correlation ratio by clone-size tertile (presence selection check).
See the 2026-11-03 (5) batch pre-registration in EXPERIMENT_LOG.md.

    uv run python experiments/kin_corr_e19c.py
"""

from __future__ import annotations

import json
import os

import numpy as np
from kin_corr_e19 import HERE, estimate, load, run, tables


def main():
    rng = np.random.default_rng(0)
    groups = load()
    size = {c: sum(n + m for n, m in g.values()) for c, g in groups.items()}
    cut = np.percentile(list(size.values()), [100 / 3, 200 / 3])
    tert = {c: int(np.searchsorted(cut, s, side="right")) for c, s in size.items()}
    out = {"cuts": cut.tolist()}
    for t in range(3):
        sub = {c: g for c, g in groups.items() if tert[c] == t}
        out[f"tertile_{t}"] = {
            "sizes": [min(size[c] for c in sub), max(size[c] for c in sub)],
            **run(sub, False, rng),
        }
    # smallest vs largest tertile difference, clone bootstrap within each tertile
    same, cross = tables(groups, False)
    lo = [c for c in same if tert[c] == 0]
    hi = [c for c in same if tert[c] == 2]
    diffs = []
    for _ in range(1000):
        try:
            a = estimate(same, cross, list(rng.choice(lo, len(lo))))[2]
            b = estimate(same, cross, list(rng.choice(hi, len(hi))))[2]
            diffs.append(b - a)
        except ValueError:
            pass
    out["large_minus_small"] = {
        "point": out["tertile_2"]["rho_f_hat"] - out["tertile_0"]["rho_f_hat"],
        "ci": np.percentile(diffs, [2.5, 97.5]).tolist(),
    }
    wells = {}
    for c, g in groups.items():
        per = {}
        for (_, w), (n, m) in g.items():
            per[w] = per.get(w, 0) + n + m
        wells[c] = per
    strict = {c: g for c, g in groups.items() if len(wells[c]) >= 2 and min(wells[c].values()) >= 3}
    out["strict_ge3_each_well"] = run(strict, False, rng)
    json.dump(out, open(os.path.join(HERE, "results_e18", "e19c_kin.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
