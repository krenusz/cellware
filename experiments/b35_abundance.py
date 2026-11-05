"""B35: does a clone's day-2 cell count predict how many wells it is present in, beyond E48?
See the B35 pre-registration in EXPERIMENT_LOG.md.

    uv run python experiments/b35_abundance.py
"""

from __future__ import annotations

import glob
import json
import os

import e26_selection as e26
import e32_multinomial as e32
import e33_measured as e33
import e34_fullarm as e34
import e48_capsel6 as e48
import numpy as np
import pandas as pd
from clone_state_model import OUT, split
from fit_cytokine import pooled
from sweep_guard import gated

CONDS = ("full", "G", "M", "Epo", "SCF")
GROUPS = (("0", 0, 0), ("1", 1, 1), ("2+", 2, 10**9))


def stats(cnt, n2):
    pres = {c: {k for k, v in pooled(cnt[c]).items() if v[0] + v[1] > 0} for c in CONDS}
    univ = set().union(*pres.values())
    out = {}
    for lab, lo, hi in GROUPS:
        ks = [k for k in univ if lo <= n2.get(int(k), 0) <= hi]
        nw = np.array([sum(k in pres[c] for c in CONDS) for k in ks])
        out[f"n_{lab}"] = len(ks)
        out[f"wells_{lab}"] = float(nw.mean()) if len(ks) else np.nan
        out[f"G_{lab}"] = float(np.mean([k in pres["G"] for k in ks])) if ks else np.nan
        out[f"M_{lab}"] = float(np.mean([k in pres["M"] for k in ks])) if ks else np.nan
        out[f"multi_{lab}"] = float((nw >= 2).mean()) if len(ks) else np.nan
    out["gradient"] = out["wells_2+"] - out["wells_0"]
    return out


def main(nsim=200):
    e34.e33.unpack_base = e33.unpack
    cnt = e33.load_counts(e33.CONDS5)
    fit = split(cnt, None)
    co = e32.copresence(fit)
    t = pd.read_csv(os.path.join(OUT, "b14_clone_day2.csv"), index_col=0)
    n2 = {int(k): int(v) for k, v in t["n2"].items()}
    data = stats(fit, n2)
    res = [json.load(open(f)) for f in glob.glob(os.path.join(OUT, "e48_s*.json"))]
    best = min((r for r in res if gated(r, co)), key=lambda r: r["loss"])
    clones = e26.universe(cnt)
    meas = e33.measured(clones, np.random.default_rng(77))
    sims = [stats(split(e48.simulate(clones, best["params"], np.random.default_rng(1000 + s), meas), None), n2)
            for s in range(nsim)]
    comp = {}
    for k, v in data.items():
        x = np.array([s[k] for s in sims], float)
        x = x[np.isfinite(x)]
        lo, hi = np.percentile(x, [2.5, 97.5])
        comp[k] = {"data": v, "sim_mean": float(x.mean()), "sim_95": [float(lo), float(hi)],
                   "inside": bool(lo <= v <= hi)}
    g = comp["gradient"]
    rep = {"e48_seed": best["seed"], "stats": comp,
           "abundance_linked": bool(g["data"] > g["sim_95"][1]
                                    and min(data["n_0"], data["n_1"], data["n_2+"]) >= 30)}
    json.dump(rep, open(os.path.join(OUT, "b35_abundance.json"), "w"), indent=1)
    for k, v in comp.items():
        print(f"{k:10s} data {v['data']:.3f} | e48 {v['sim_mean']:.3f} [{v['sim_95'][0]:.2f},{v['sim_95'][1]:.2f}] "
              f"{'in ' if v['inside'] else 'OUT'}")
    print("abundance_linked", rep["abundance_linked"])


if __name__ == "__main__":
    main()
