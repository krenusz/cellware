"""B27: how clones spread across wells (presence), data vs E38.
See the B27 pre-registration in EXPERIMENT_LOG.md.

    uv run python experiments/track_b27.py
"""

from __future__ import annotations

import json
import os

import e26_selection as e26
import e33_measured as e33
import e37_pool as e37
import numpy as np
import pandas as pd
import track_b24 as b24
from clone_state_model import OUT, split
from fit_cytokine import pooled

MYE = ("full", "G", "M")
PAIRS = (("G", "M"), ("full", "G"), ("full", "M"))


def stats(cnt, n2):
    p = {c: pooled(cnt[c]) for c in MYE}
    pres = {c: {k for k, v in p[c].items() if v[0] + v[1] >= 2} for c in MYE}
    anyp = set().union(*pres.values())
    out = {f"P4_{c}": len(pres[c]) for c in MYE}
    nw = np.array([sum(k in pres[c] for c in MYE) for k in anyp])
    for j in (1, 2, 3):
        out[f"P1_frac_{j}"] = float((nw == j).mean()) if len(nw) else np.nan
    n = len(anyp)
    for a, b in PAIRS:
        obs = len(pres[a] & pres[b])
        exp = len(pres[a]) * len(pres[b]) / n if n else np.nan
        out[f"P2_lift_{a}-{b}"] = obs / exp if exp else np.nan
        out[f"P2_n_{a}-{b}"] = obs
    univ = set().union(*[set(p[c]) for c in ("full", "G", "M")])  # clones seen in any myeloid well
    for lab, test in (("0", lambda v: v == 0), ("1", lambda v: v == 1), ("2+", lambda v: v >= 2)):
        ks = [k for k in univ if test(n2.get(int(k), 0))]
        out[f"P3_n_{lab}"] = len(ks)
        for c in ("G", "M"):
            out[f"P3_{c}_{lab}"] = float(np.mean([k in pres[c] for k in ks])) if ks else np.nan
    return out


def main(nsim=200):
    cnt = e33.load_counts(e33.CONDS5)
    t = pd.read_csv(os.path.join(OUT, "b14_clone_day2.csv"), index_col=0)
    n2 = {int(k): int(v) for k, v in t["n2"].items()}
    data = stats(split(cnt, None), n2)
    ref = b24.best_e38()
    clones = e26.universe(cnt)
    meas = e33.measured(clones, np.random.default_rng(77))
    sims = [
        stats(
            split(e37.simulate(clones, ref["params"], np.random.default_rng(1000 + s), meas), None),
            n2,
        )
        for s in range(nsim)
    ]
    rep = {"reference": f"e38_s{ref['seed']}", "data": data, "sim": {}}
    for k, v in data.items():
        x = np.array([s[k] for s in sims], float)
        x = x[np.isfinite(x)]
        lo, hi = np.percentile(x, [2.5, 97.5])
        rep["sim"][k] = {"mean": float(x.mean()), "95": [float(lo), float(hi)]}
        rep["sim"][k]["data_inside"] = bool(np.isfinite(v) and lo <= v <= hi)
    s = rep["sim"]
    readings = []
    if (
        data["P2_n_G-M"] >= 30
        and data["P2_lift_G-M"] < s["P2_lift_G-M"]["95"][0]
        and s["P4_G"]["data_inside"]
        and s["P4_M"]["data_inside"]
    ):
        readings.append("too much shared presence")
    rise = any(
        data[f"P3_n_{a}"] >= 30
        and data["P3_n_2+"] >= 30
        and data[f"P3_{c}_2+"] - data[f"P3_{c}_0"] >= 0.1
        and not (s[f"P3_{c}_2+"]["mean"] - s[f"P3_{c}_0"]["mean"] >= 0.1)
        for c in ("G", "M")
        for a in ("0",)
    )
    if rise:
        readings.append("missing founder heterogeneity")
    if any(not s[f"P4_{c}"]["data_inside"] for c in MYE):
        readings.append("marginals wrong")
    rep["readings"] = readings or ["not localised"]
    json.dump(rep, open(os.path.join(OUT, "b27_presence.json"), "w"), indent=1)
    for k, v in data.items():
        x = s[k]
        flag = "" if x["data_inside"] else "  OUT"
        print(
            f"{k:16s} data {v:8.3f}  sim {x['mean']:8.3f} "
            f"[{x['95'][0]:.3f}, {x['95'][1]:.3f}]{flag}"
        )
    print("readings:", rep["readings"])


if __name__ == "__main__":
    main()
