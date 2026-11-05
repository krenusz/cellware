"""B31: locate the G-M co-presence deficit (threshold curve, cross-presence by size, measured
day-2 state, day structure). See the B31 pre-registration in EXPERIMENT_LOG.md.

    uv run python experiments/b31_copresence.py
"""

from __future__ import annotations

import json
import os

import e26_selection as e26
import e32_multinomial as e32
import e33_measured as e33
import e34_fullarm as e34
import e46_selexp as e46
import e48_capsel6 as e48
import numpy as np
import pandas as pd
from clone_state_model import OUT, split
from fit_cytokine import pooled
from sweep_guard import gated

MYE = ("full", "G", "M")
THR = (1, 2, 3, 5)
BINS = (("2", 2, 2), ("3-5", 3, 5), ("6+", 6, 10**9))


def lift(pa, pb, univ):
    n = len(univ)
    a, b = pa & univ, pb & univ
    exp = len(a) * len(b) / n if n else np.nan
    return (len(a & b) / exp if exp else np.nan), exp


def stats(cnt, mstate):
    p = {c: pooled(cnt[c]) for c in MYE}
    size = {c: {k: v[0] + v[1] for k, v in p[c].items()} for c in MYE}
    out = {}
    for t in THR:
        pres = {c: {k for k, s in size[c].items() if s >= t} for c in MYE}
        univ = set().union(*pres.values())
        out[f"a_lift_ge{t}"], _ = lift(pres["G"], pres["M"], univ)
        out[f"a_den_ge{t}"] = min(len(pres["G"]), len(pres["M"]))
    pres = {c: {k for k, s in size[c].items() if s >= 2} for c in MYE}
    for src, dst in (("G", "M"), ("M", "G")):
        for lab, lo, hi in BINS:
            ks = [k for k, s in size[src].items() if lo <= s <= hi]
            out[f"b_P{dst}_given_{src}{lab}"] = float(np.mean([k in pres[dst] for k in ks])) if ks else np.nan
            out[f"b_den_{src}{lab}"] = len(ks)
    univ = set().union(*pres.values())
    med = np.median([mstate[k] for k in univ if k in mstate])
    for lab, test in (("low", lambda v: v <= med), ("high", lambda v: v > med)):
        u = {k for k in univ if k in mstate and test(mstate[k])}
        out[f"c_lift_{lab}"], out[f"c_exp_{lab}"] = lift(pres["G"], pres["M"], u)
    return out


def day_lifts(cnt):
    pres = {(c, d): {k for k, v in cnt[c][d].items() if v[0] + v[1] >= 2} for c in MYE for d in (4, 6)}
    univ = set().union(*pres.values())
    res = {}
    for dg in (4, 6):
        for dm in (4, 6):
            res[f"G{dg}-M{dm}"] = {"lift": lift(pres[("G", dg)], pres[("M", dm)], univ)[0],
                                   "n": len(pres[("G", dg)] & pres[("M", dm)])}
    return res


def main(nsim=200):
    e34.e33.unpack_base = e33.unpack
    cnt = e33.load_counts(e33.CONDS5)
    fit = split(cnt, None)
    co = e32.copresence(fit)
    t = pd.read_csv(os.path.join(OUT, "b14_clone_day2.csv"), index_col=0)
    score = t.cebpa - t.irf8
    mstate = dict(zip(t.index, ((score - score.mean()) / score.std()).to_numpy()))
    clones = e26.universe(cnt)
    meas = e33.measured(clones, np.random.default_rng(77))
    data = stats(fit, mstate)
    rep = {"data": data, "data_days": day_lifts(fit), "models": {}}
    for name, mod in (("e46", e46), ("e48", e48)):
        import glob
        res = [json.load(open(f)) for f in glob.glob(os.path.join(OUT, f"{name}_s*.json"))]
        best = min((r for r in res if gated(r, co)), key=lambda r: r["loss"])
        sims = [stats(split(mod.simulate(clones, best["params"], np.random.default_rng(1000 + s), meas), None), mstate)
                for s in range(nsim)]
        m = {}
        for k, v in data.items():
            x = np.array([s[k] for s in sims], float)
            x = x[np.isfinite(x)]
            lo, hi = np.percentile(x, [2.5, 97.5])
            m[k] = {"data": v, "sim_mean": float(x.mean()), "sim_95": [float(lo), float(hi)],
                    "inside": bool(np.isfinite(v) and lo <= v <= hi)}
        rep["models"][name] = {"seed": best["seed"], "stats": m}
        print(name, best["seed"], flush=True)
    json.dump(rep, open(os.path.join(OUT, "b31_copresence.json"), "w"), indent=1)
    for k, v in data.items():
        if "_den_" in k or "_exp_" in k:
            continue
        r = [rep["models"][n]["stats"][k] for n in ("e46", "e48")]
        print(f"{k:22s} data {v:6.3f} | e46 {r[0]['sim_mean']:.3f} [{r[0]['sim_95'][0]:.2f},{r[0]['sim_95'][1]:.2f}] {'in ' if r[0]['inside'] else 'OUT'}"
              f" | e48 {r[1]['sim_mean']:.3f} [{r[1]['sim_95'][0]:.2f},{r[1]['sim_95'][1]:.2f}] {'in ' if r[1]['inside'] else 'OUT'}")
    print({k: v for k, v in data.items() if "_den_" in k or "_exp_" in k})
    print(json.dumps(rep["data_days"]))


if __name__ == "__main__":
    main()
