"""E50 readings: the E46 analysis applied to E50, the B34 privacy-by-size and B31/B32 statistics
for the best gated fit, and class parameters. The fit-set primary and B7 CRPS come from
`b29_rescore.py e50`.

    uv run python experiments/analyze_e50.py
"""

from __future__ import annotations

import glob
import json
import os

import analyze_e46
import b31_copresence as b31
import b32_small_clones as b32
import b34_private as b34
import e26_selection as e26
import e32_multinomial as e32
import e33_measured as e33
import e34_fullarm as e34
import e50_gneu as e50
import numpy as np
from clone_state_model import OUT, split
from sweep_guard import gated

KEYS = ("G_2", "G_3-5", "G_6-15", "M_2", "M_3-5", "a_lift_ge2", "b_PG_given_M2", "M2_in_any")


def collect(o):
    pv, _ = b34.private_by_size(o)
    return {**pv, **b31.stats(o, {}), **b32.presence_elsewhere(o)[0]}


def main(nsim=200):
    analyze_e46.main(name="e50", sim_mod=e50)
    e34.e33.unpack_base = e33.unpack
    cnt = e33.load_counts(e33.CONDS5)
    fit = split(cnt, None)
    co = e32.copresence(fit)
    res = [json.load(open(f)) for f in glob.glob(os.path.join(OUT, "e50_s*.json"))]
    g = [r for r in res if gated(r, co)]
    best = min(g, key=lambda r: r["loss"])
    clones = e26.universe(cnt)
    meas = e33.measured(clones, np.random.default_rng(77))
    data = collect(fit)
    sims = [collect(split(e50.simulate(clones, best["params"], np.random.default_rng(1000 + s), meas), None))
            for s in range(nsim)]
    out = {}
    for k in KEYS:
        x = np.array([s[k] for s in sims], float)
        x = x[np.isfinite(x)]
        lo, hi = np.percentile(x, [2.5, 97.5])
        out[k] = {"data": data[k], "sim_mean": float(x.mean()), "sim_95": [float(lo), float(hi)],
                  "inside": bool(lo <= data[k] <= hi)}
    rep = {
        "best_gated": best["seed"],
        "gated": len(g),
        "median_phi_C": float(np.median([r["params"]["phi_C"] for r in g])) if g else None,
        "median_b_C": float(np.median([r["params"]["b_C"] for r in g])) if g else None,
        "median_d_C": float(np.median([r["params"]["d_C"] for r in g])) if g else None,
        "median_gamma": float(np.median([r["params"]["gamma"] for r in g])) if g else None,
        "private": out,
        "explains_G_private": bool(out["G_3-5"]["inside"] and out["G_6-15"]["inside"]),
    }
    json.dump(rep, open(os.path.join(OUT, "e50_private.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "private"}, indent=1))
    for k, v in out.items():
        print(f"{k:14s} data {v['data']:.3f} | e50 {v['sim_mean']:.3f} "
              f"[{v['sim_95'][0]:.2f},{v['sim_95'][1]:.2f}] {'in ' if v['inside'] else 'OUT'}")


if __name__ == "__main__":
    main()
