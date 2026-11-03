"""B12: what makes full-M low in the data? Components vs the E30a fit, and a day split.
See the 2026-11-03 (7) batch pre-registration in EXPERIMENT_LOG.md.

    uv run python experiments/track_b12.py
"""

from __future__ import annotations

import json
import os

import e26_selection as e26
import e29_ceiling as e29
import numpy as np
from clone_state_model import OUT
from fit_cytokine import CONDS5, load_counts, pooled, test_stats

PAIRS = (("full", "M"), ("G", "M"), ("full", "G"))


def components(cnt):
    out = {}
    for a, b in PAIRS:
        pa, pb = pooled(cnt[a]), pooled(cnt[b])
        both = [k for k in pa if k in pb and sum(pa[k][:2]) >= 2 and sum(pb[k][:2]) >= 2]
        sa = np.array([pa[k][0] / (pa[k][0] + pa[k][1]) for k in both])
        sb = np.array([pb[k][0] / (pb[k][0] + pb[k][1]) for k in both])
        r = test_stats(pa, pb, boot=1)["r"] if len(both) >= 3 else np.nan
        out[f"{a}-{b}"] = {
            "n": len(both),
            f"mean_{a}": float(sa.mean()) if len(both) else np.nan,
            f"var_{a}": float(sa.var()) if len(both) else np.nan,
            f"mean_{b}": float(sb.mean()) if len(both) else np.nan,
            f"var_{b}": float(sb.var()) if len(both) else np.nan,
            "r": float(r),
        }
    return out


def day_r(cnt, a, b, day_b, rng):
    """r between condition a (all days) and condition b restricted to one day, with bootstrap CI."""
    pa, pb = pooled(cnt[a]), pooled({day_b: cnt[b][day_b]})
    keys = [k for k in pa if k in pb and sum(pa[k][:2]) >= 2 and sum(pb[k][:2]) >= 2]
    if len(keys) < 15:
        return {"n": len(keys), "r": None}
    r = test_stats({k: pa[k] for k in keys}, {k: pb[k] for k in keys}, boot=1)["r"]
    bs = []
    for _ in range(2000):
        pick = rng.choice(len(keys), len(keys))
        ka = {(keys[i], j): pa[keys[i]] for j, i in enumerate(pick)}
        kb = {(keys[i], j): pb[keys[i]] for j, i in enumerate(pick)}
        bs.append(test_stats(ka, kb, boot=1)["r"])
    return {"n": len(keys), "r": float(r), "ci": np.nanpercentile(bs, [2.5, 97.5]).tolist()}


def main():
    cnt = load_counts(CONDS5)
    clones = e26.universe(cnt)
    data = components(cnt)
    prm = json.load(open(os.path.join(OUT, "e30a_s2.json")))["params"]
    sims = [
        components(e29.simulate(clones, prm, np.random.default_rng(4000 + s))) for s in range(200)
    ]
    rep = {"a_components": {}}
    for pair, d in data.items():
        rep["a_components"][pair] = {}
        for k, v in d.items():
            s = np.array([x[pair][k] for x in sims], float)
            s = s[np.isfinite(s)]
            lo, hi = np.percentile(s, [2.5, 97.5])
            rep["a_components"][pair][k] = {
                "data": v,
                "sim_mean": float(s.mean()),
                "sim_95": [float(lo), float(hi)],
                "flag": "above" if v > hi else "below" if v < lo else "",
            }
    rng = np.random.default_rng(0)
    rep["b_day_split"] = {
        "full-M_day4": day_r(cnt, "full", "M", 4, rng),
        "full-M_day6": day_r(cnt, "full", "M", 6, rng),
        "G-M_Gday4": day_r(cnt, "M", "G", 4, rng),
        "G-M_Gday6": day_r(cnt, "M", "G", 6, rng),
    }
    json.dump(rep, open(os.path.join(OUT, "b12.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
