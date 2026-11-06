"""C1: does the core LARRY configuration reproduce E48's simulator? (EXPERIMENT_LOG C1)

    uv run python c1_larry_equiv.py
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np

import core_larry
import e26_selection as e26
import e33_measured as e33
import e36_combined as e36
import e48_capsel6 as e48

OUT = os.path.join(os.path.dirname(__file__), "results_e18")
RES = os.path.join(os.path.dirname(__file__), "results_sis")


def main(nsim=300):
    reps = [json.load(open(f)) for f in glob.glob(os.path.join(OUT, "e48_s*.json"))]
    best = min(reps, key=lambda r: r["loss"])  # any E48 fit serves for an equivalence check
    prm = best["params"]
    cnt = e33.load_counts(e33.CONDS5)
    clones = e26.universe(cnt)
    meas = e33.measured(clones, np.random.default_rng(77))
    keys = sorted(e36.targets(cnt))
    a, b = [], []
    for s in range(nsim):
        ta = e36.targets(e48.simulate(clones, prm, np.random.default_rng(5000 + s), meas))
        tb = e36.targets(core_larry.simulate(clones, prm, np.random.default_rng(9000 + s), meas))
        a.append([ta[k] for k in keys])
        b.append([tb[k] for k in keys])
    a, b = np.array(a, float), np.array(b, float)
    d = np.abs(np.nanmean(a, 0) - np.nanmean(b, 0))
    se = np.sqrt(np.nanvar(a, 0) / nsim + np.nanvar(b, 0) / nsim)
    ok = (d <= 3 * se) | (d <= 0.01)
    res = {"best_seed": best["seed"], "n_stats": len(keys), "share_pass": float(ok.mean()),
           "fails": {keys[i]: {"e48": float(np.nanmean(a[:, i])), "core": float(np.nanmean(b[:, i])),
                               "se": float(se[i])} for i in np.flatnonzero(~ok)},
           "equivalent": bool(ok.mean() >= 0.95)}
    os.makedirs(RES, exist_ok=True)
    json.dump(res, open(os.path.join(RES, "c1_equiv_larry.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
