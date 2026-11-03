"""B10: neutrophil output under M-CSF in the clones seen under both G-CSF and M-CSF.
See the B10 pre-registration in EXPERIMENT_LOG.md.

    uv run python experiments/track_b10.py
"""

from __future__ import annotations

import json
import os

import e26_selection as e26
import e28_split as e28
import e29_ceiling as e29
import numpy as np
from clone_state_model import OUT
from e25_clone_types import groups
from fit_cytokine import CONDS5, load_counts
from scipy.stats import spearmanr


def stats(cnt):
    pg, pm, both, _, _ = groups(cnt)
    both = sorted(both)
    n_m = np.array([pm[k][0] for k in both])
    sg = np.array([pg[k][0] / (pg[k][0] + pg[k][1]) for k in both])
    sm = np.array([pm[k][0] / (pm[k][0] + pm[k][1]) for k in both])
    has = n_m >= 1

    def rho(a, b):
        return (
            float(spearmanr(a, b)[0]) if len(a) >= 5 and np.ptp(a) > 0 and np.ptp(b) > 0 else np.nan
        )

    return {
        "n_both": len(both),
        "frac_with_N_in_M": float(has.mean()) if len(both) else np.nan,
        "mean_N_in_M": float(n_m.mean()) if len(both) else np.nan,
        "rho_G_vs_M": rho(sg, sm),
        "rho_G_vs_M_with_N": rho(sg[has], sm[has]),
    }


def main():
    cnt = load_counts(CONDS5)
    clones = e26.universe(cnt)
    rep = {"data": stats(cnt)}
    for name, sim in (("e29wwarm_s2", e29.simulate), ("e28a_s2", e28.simulate)):
        prm = json.load(open(os.path.join(OUT, f"{name}.json")))["params"]
        if sim is e29.simulate:
            prm.setdefault("gamma", 0.0)
        runs = [stats(sim(clones, prm, np.random.default_rng(2000 + s))) for s in range(200)]
        out = {}
        for k, d in rep["data"].items():
            v = np.array([r[k] for r in runs], float)
            v = v[np.isfinite(v)]
            if not len(v):  # e.g. too few clones with an N cell under M-CSF in every simulation
                out[k] = {"data": d, "finite_sims": 0}
                continue
            lo, hi = np.percentile(v, [2.5, 97.5])
            out[k] = {
                "data": d,
                "finite_sims": len(v),
                "sim_mean": float(v.mean()),
                "sim_95": [float(lo), float(hi)],
                "above": bool(d > hi),
                "below": bool(d < lo),
            }
        rep[name] = out
    json.dump(rep, open(os.path.join(OUT, "b10.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
