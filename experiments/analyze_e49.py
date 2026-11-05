"""E49 readings: the E46 analysis applied to E49 (gate, B27, B24/B26, instruction), the B31/B32
small-clone statistics for the best gated E49 fit, sister and selection parameters. The fit-set
primary and B7 CRPS come from `b29_rescore.py e49`.

    uv run python experiments/analyze_e49.py
"""

from __future__ import annotations

import glob
import json
import os

import analyze_e46
import b31_copresence as b31
import b32_small_clones as b32
import e26_selection as e26
import e32_multinomial as e32
import e33_measured as e33
import e34_fullarm as e34
import e49_sisters_sel as e49
import numpy as np
import pandas as pd
from clone_state_model import OUT, split
from sweep_guard import gated

KEYS31 = ("a_lift_ge1", "a_lift_ge2", "a_lift_ge3", "b_PG_given_M2", "b_PM_given_G2", "b_PG_given_M6+")
KEYS32 = ("M2_n", "M2_in_any", "M2_in_Epo", "M2_in_SCF", "G3-5_in_Epo", "G3-5_in_SCF")


def main(nsim=200):
    analyze_e46.main(name="e49", sim_mod=e49)
    e34.e33.unpack_base = e33.unpack
    cnt = e33.load_counts(e33.CONDS5)
    fit = split(cnt, None)
    co = e32.copresence(fit)
    res = [json.load(open(f)) for f in glob.glob(os.path.join(OUT, "e49_s*.json"))]
    g = [r for r in res if gated(r, co)]
    best = min(g, key=lambda r: r["loss"])
    t = pd.read_csv(os.path.join(OUT, "b14_clone_day2.csv"), index_col=0)
    score = t.cebpa - t.irf8
    mstate = dict(zip(t.index, ((score - score.mean()) / score.std()).to_numpy()))
    clones = e26.universe(cnt)
    meas = e33.measured(clones, np.random.default_rng(77))
    d31, (d32, _) = b31.stats(fit, mstate), b32.presence_elsewhere(fit)
    sims = []
    for s in range(nsim):
        o = split(e49.simulate(clones, best["params"], np.random.default_rng(1000 + s), meas), None)
        sims.append({**b31.stats(o, mstate), **b32.presence_elsewhere(o)[0]})
    data = {**{k: d31[k] for k in KEYS31}, **{k: d32[k] for k in KEYS32}}
    small = {}
    for k, v in data.items():
        x = np.array([s[k] for s in sims], float)
        x = x[np.isfinite(x)]
        lo, hi = np.percentile(x, [2.5, 97.5])
        small[k] = {"data": v, "sim_mean": float(x.mean()), "sim_95": [float(lo), float(hi)],
                    "inside": bool(lo <= v <= hi)}
    rep = {
        "best_gated": best["seed"],
        "gated": len(g),
        "median_lam_K": float(np.median([r["params"]["lam_K"] for r in g])),
        "median_w_M": float(np.median([r["params"]["w_M"] for r in g])),
        "median_gamma": float(np.median([r["params"]["gamma"] for r in g])),
        "small_clones": small,
        "explains_B32": bool(small["M2_in_any"]["inside"] and small["b_PG_given_M2"]["inside"]),
        "lift_ge2_inside": small["a_lift_ge2"]["inside"],
    }
    json.dump(rep, open(os.path.join(OUT, "e49_small_clones.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "small_clones"}, indent=1))
    for k, v in small.items():
        print(f"{k:16s} data {v['data']:.3f} | e49 {v['sim_mean']:.3f} "
              f"[{v['sim_95'][0]:.2f},{v['sim_95'][1]:.2f}] {'in ' if v['inside'] else 'OUT'}")


if __name__ == "__main__":
    main()
