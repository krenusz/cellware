"""B17: why does E33 over-select under G-CSF (B7 G)? Components, data vs the E33 best fit.
See the 2026-11-03 (9) batch pre-registration in EXPERIMENT_LOG.md.

    uv run python experiments/track_b17.py
"""

from __future__ import annotations

import json
import os

import e26_selection as e26
import e33_measured as e33
import numpy as np
from clone_state_model import OUT
from e25_clone_types import groups
from fit_cytokine import CONDS5, load_counts


def components(cnt):
    pg, _, both, g_only, _ = groups(cnt)
    out = {"n_G_only": len(g_only), "n_both": len(both)}
    for name, ks in (("G_only", g_only), ("both", both)):
        s = np.array([pg[k][0] / (pg[k][0] + pg[k][1]) for k in ks])
        sz = np.log([pg[k][0] + pg[k][1] for k in ks])
        out[f"{name}_mean"] = float(s.mean()) if len(s) else np.nan
        out[f"{name}_sd"] = float(s.std()) if len(s) else np.nan
        out[f"{name}_allN"] = float((s == 1).mean()) if len(s) else np.nan
        out[f"{name}_logsize"] = float(sz.mean()) if len(s) else np.nan
    return out


def main():
    cnt = load_counts(CONDS5)
    clones = e26.universe(cnt)
    meas = e33.measured(clones, np.random.default_rng(77))
    prm = json.load(open(os.path.join(OUT, "e33_s3.json")))["params"]
    data = components(cnt)
    sims = [
        components(e33.simulate(clones, prm, np.random.default_rng(6000 + s), meas))
        for s in range(200)
    ]
    rep = {}
    for k, d in data.items():
        v = np.array([s[k] for s in sims], float)
        v = v[np.isfinite(v)]
        lo, hi = np.percentile(v, [2.5, 97.5])
        rep[k] = {
            "data": d,
            "sim_mean": float(v.mean()),
            "sim_95": [float(lo), float(hi)],
            "flag": "above" if d > hi else "below" if d < lo else "",
        }
    rep["params"] = {k: prm[k] for k in ("s", "tau", "w", "kappa", "beta")}
    json.dump(rep, open(os.path.join(OUT, "b17.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
