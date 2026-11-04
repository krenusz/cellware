"""B20: decompose B7 G (G-CSF presence selection) in data vs the E38 model.
See the B20 pre-registration in EXPERIMENT_LOG.md. Full, G-CSF and M-CSF only.

    uv run python experiments/track_b20.py
"""

from __future__ import annotations

import json
import os

import e26_selection as e26
import e33_measured as e33
import e37_pool as e37
import numpy as np
import pandas as pd
from clone_state_model import OUT
from e25_clone_types import groups, nshare
from fit_cytokine import CONDS5, load_counts

GATED = (1, 3, 4, 6, 8, 11, 12, 14, 15)  # E38 seeds passing the gate; 1 is the best


def b7g(pg, a, b):
    if not len(a) or not len(b):
        return np.nan
    return float(nshare(pg, a).mean() - nshare(pg, b).mean())


def components(cnt, mvals):
    pg, _, both, g_only, _ = groups(cnt)
    both, g_only = sorted(both), sorted(g_only)
    xb, xg = nshare(pg, both), nshare(pg, g_only)
    size = {k: pg[k][0] + pg[k][1] for k in both + g_only}
    small = lambda ks: [k for k in ks if size[k] <= 3]  # noqa: E731
    large = lambda ks: [k for k in ks if size[k] >= 4]  # noqa: E731
    mb = [mvals[k] for k in both if k in mvals]
    mg = [mvals[k] for k in g_only if k in mvals]
    return {
        "B7G": float(xb.mean() - xg.mean()),
        "C1_both_mean": float(xb.mean()),
        "C2_Gonly_mean": float(xg.mean()),
        "C3_n_both": len(both),
        "C4_n_Gonly": len(g_only),
        "C5_Gonly_Npure": float((xg == 1).mean()),
        "C6_both_N0": float((xb == 0).mean()),
        "C7_B7G_small": b7g(pg, small(both), small(g_only)),
        "C8_B7G_large": b7g(pg, large(both), large(g_only)),
        "C9_m_both_minus_Gonly": float(np.mean(mb) - np.mean(mg)) if mb and mg else np.nan,
    }


def main():
    cnt = load_counts(CONDS5)
    clones = e26.universe(cnt)
    meas = e33.measured(clones, np.random.default_rng(77))
    measured = set(pd.read_csv(os.path.join(OUT, "b14_clone_day2.csv"), index_col=0).index)
    mvals = {c: meas[i, 0] for i, c in enumerate(clones) if c in measured}
    sub = {c: cnt[c] for c in ("full", "G", "M")}
    data = components(sub, mvals)

    def sims(s, n):
        prm = json.load(open(os.path.join(OUT, f"e38_s{s}.json")))["params"]
        rows = [
            components(e37.simulate(clones, prm, np.random.default_rng(7000 + i), meas), mvals)
            for i in range(n)
        ]
        return pd.DataFrame(rows)

    best = sims(GATED[0], 200)
    others = {s: sims(s, 50).mean() for s in GATED[1:]}
    rep = {}
    for k, d in data.items():
        v = best[k].dropna().to_numpy()
        lo, hi = np.percentile(v, [2.5, 97.5])
        side = [float(o[k]) > d for o in others.values()]
        rep[k] = {
            "data": d,
            "model_mean": float(v.mean()),
            "model_95": [float(lo), float(hi)],
            "data_pct": float((v < d).mean()),
            "misfit": bool(d < lo or d > hi),
            "other_seeds_above_data": int(sum(side)),
            "other_seed_means": {s: float(o[k]) for s, o in others.items()},
        }
        rep[k]["robust"] = rep[k]["misfit"] and max(sum(side), len(side) - sum(side)) >= 7
    json.dump(rep, open(os.path.join(OUT, "b20_b7g.json"), "w"), indent=1)
    for k, r in rep.items():
        print(
            f"{k:24s} data {r['data']:7.3f} model {r['model_mean']:7.3f} "
            f"[{r['model_95'][0]:.3f}, {r['model_95'][1]:.3f}] pct {r['data_pct']:.2f} "
            f"misfit {r['misfit']} robust {r['robust']} above {r['other_seeds_above_data']}/8"
        )


if __name__ == "__main__":
    main()
