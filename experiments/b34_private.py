"""B34: describe the well-private clones (privacy by size vs E48, day structure, day-2 presence
and state, composition). See the B34 pre-registration in EXPERIMENT_LOG.md.

    uv run python experiments/b34_private.py
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
from scipy.stats import mannwhitneyu
from sweep_guard import gated

CONDS = ("full", "G", "M", "Epo", "SCF")
BINS = (("1", 1, 1), ("2", 2, 2), ("3-5", 3, 5), ("6-15", 6, 15), ("16+", 16, 10**9))
RNG = np.random.default_rng(34)


def private_by_size(cnt):
    p = {c: pooled(cnt[c]) for c in CONDS}
    size = {c: {k: v[0] + v[1] for k, v in p[c].items() if v[0] + v[1] > 0} for c in CONDS}
    out = {}
    for src in ("M", "G"):
        others = [c for c in CONDS if c != src]
        for lab, lo, hi in BINS:
            ks = [k for k, s in size[src].items() if lo <= s <= hi]
            out[f"{src}_{lab}"] = float(np.mean([not any(k in size[c] for c in others) for k in ks])) if ks else np.nan
            out[f"{src}_{lab}_n"] = len(ks)
    return out, size


def boot_diff(a, b, n=2000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = [RNG.choice(a, len(a)).mean() - RNG.choice(b, len(b)).mean() for _ in range(n)]
    return float(a.mean() - b.mean()), np.percentile(d, [2.5, 97.5]).tolist()


def describe(cnt, size, t):
    res = {}
    for src in ("M", "G"):
        others = [c for c in CONDS if c != src]
        ks = [k for k, s in size[src].items() if 2 <= s <= 5]
        priv = [k for k in ks if not any(k in size[c] for c in others)]
        shar = [k for k in ks if any(k in size[c] for c in others)]
        d4, d6 = cnt[src][4], cnt[src][6]

        def frac6(k):
            a = sum(d4.get(k, [0, 0, 0])[:2])
            b = sum(d6.get(k, [0, 0, 0])[:2])
            return b / (a + b)

        def both(k):
            return sum(d4.get(k, [0, 0, 0])[:2]) > 0 and sum(d6.get(k, [0, 0, 0])[:2]) > 0

        def nfrac(k):
            v = pooled(cnt[src])[k]
            return v[0] / (v[0] + v[1])

        r = {"n_private": len(priv), "n_shared": len(shar)}
        r["day6_share"] = {"private": float(np.mean([frac6(k) for k in priv])),
                           "shared": float(np.mean([frac6(k) for k in shar]))}
        r["day6_share"]["diff"], r["day6_share"]["diff_95"] = boot_diff([frac6(k) for k in priv], [frac6(k) for k in shar])
        r["both_days"] = {"private": float(np.mean([both(k) for k in priv])),
                          "shared": float(np.mean([both(k) for k in shar]))}
        r["N_share"] = {"private": float(np.mean([nfrac(k) for k in priv])),
                        "shared": float(np.mean([nfrac(k) for k in shar]))}
        seen = {g: [k for k in grp if int(k) in t.index and t.loc[int(k), "n2"] > 0]
                for g, grp in (("private", priv), ("shared", shar))}
        r["day2_seen"] = {g: len(seen[g]) / max(len(grp), 1) for g, grp in (("private", priv), ("shared", shar))}
        r["day2_seen_n"] = {g: len(v) for g, v in seen.items()}
        for col in ("n2", "m"):
            a = t.loc[[int(k) for k in seen["private"]], col].to_numpy()
            b = t.loc[[int(k) for k in seen["shared"]], col].to_numpy()
            ok = len(a) >= 30 and len(b) >= 30
            r[f"day2_{col}"] = {"private_mean": float(a.mean()) if len(a) else np.nan,
                                "shared_mean": float(b.mean()) if len(b) else np.nan,
                                "p": float(mannwhitneyu(a, b).pvalue) if len(a) and len(b) else np.nan,
                                "enough": bool(ok)}
        res[src] = r
    return res


def main(nsim=200):
    e34.e33.unpack_base = e33.unpack
    cnt = e33.load_counts(e33.CONDS5)
    fit = split(cnt, None)
    co = e32.copresence(fit)
    t = pd.read_csv(os.path.join(OUT, "b14_clone_day2.csv"), index_col=0)
    sc = t.cebpa - t.irf8
    t["m"] = (sc - sc.mean()) / sc.std()
    data, size = private_by_size(fit)
    res = [json.load(open(f)) for f in glob.glob(os.path.join(OUT, "e48_s*.json"))]
    best = min((r for r in res if gated(r, co)), key=lambda r: r["loss"])
    clones = e26.universe(cnt)
    meas = e33.measured(clones, np.random.default_rng(77))
    sims = [private_by_size(split(e48.simulate(clones, best["params"], np.random.default_rng(1000 + s), meas), None))[0]
            for s in range(nsim)]
    comp = {}
    for k, v in data.items():
        if k.endswith("_n"):
            continue
        x = np.array([s[k] for s in sims], float)
        x = x[np.isfinite(x)]
        lo, hi = np.percentile(x, [2.5, 97.5])
        comp[k] = {"data": v, "n": data[k + "_n"], "sim_mean": float(x.mean()),
                   "sim_95": [float(lo), float(hi)], "inside": bool(lo <= v <= hi)}
    rep = {"e48_seed": best["seed"], "private_by_size": comp, "describe": describe(fit, size, t)}
    json.dump(rep, open(os.path.join(OUT, "b34_private.json"), "w"), indent=1)
    for k, v in comp.items():
        print(f"{k:8s} n {v['n']:4d} data {v['data']:.3f} | e48 {v['sim_mean']:.3f} "
              f"[{v['sim_95'][0]:.2f},{v['sim_95'][1]:.2f}] {'in ' if v['inside'] else 'OUT'}")
    print(json.dumps(rep["describe"], indent=1))


if __name__ == "__main__":
    main()
