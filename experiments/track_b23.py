"""B23: does day-2 Gfi1 gate the G-CSF response beyond Cebpa-Irf8 (and explain B7 G)?
See the B23 pre-registration in EXPERIMENT_LOG.md. Downloads the expression matrix if missing;
meant for GitHub Actions.

    python experiments/track_b23.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from clone_state_model import OUT
from e25_clone_types import groups
from fit_cytokine import CONDS5, load_counts
from scipy.stats import rankdata
from track_b13 import bh, day2_expression, partial_multi

GENES = ["Gfi1", "Rasgrp1", "Csf3r"]


def resid(y, covs):
    a = np.c_[np.ones(len(y)), *[rankdata(c) for c in covs]]
    v = rankdata(y)
    return v - a @ np.linalg.lstsq(a, v, rcond=None)[0]


def b7_resid(y, covs, shared):
    r = resid(y, covs)
    return r[shared].mean() - r[~shared].mean()


def one_sided(r, p, sign):
    return float(p / 2 if np.sign(r) == sign else 1 - p / 2)


def main():
    expr, found = day2_expression(GENES)
    t = pd.read_csv(os.path.join(OUT, "b14_clone_day2.csv"), index_col=0)
    cnt = load_counts(CONDS5)
    pg, pm, both, g_only, _ = groups(cnt)
    keys = sorted(k for k in (both | g_only) if k in expr and k in t.index)
    share = np.array([pg[k][0] / (pg[k][0] + pg[k][1]) for k in keys])
    size = np.log([pg[k][0] + pg[k][1] for k in keys])
    m = np.array([t.loc[k, "cebpa"] - t.loc[k, "irf8"] for k in keys])
    shared = np.array([k in both for k in keys])
    gene = {g: np.array([expr[k][g] for k in keys]) for g in found}
    rep = {
        "genes_found": found,
        "n_G_present": len(keys),
        "n_shared": int(shared.sum()),
        "n_G_only": int((~shared).sum()),
        "frac_zero": {g: float((gene[g] == 0).mean()) for g in found},
        "raw_B7G_mean_share": float(share[shared].mean() - share[~shared].mean()),
    }
    g = gene["Gfi1"]
    r, p = partial_multi(g, share, [m, size])
    rep["T1_Gfi1_Nshare_given_m"] = {"partial_rho": float(r), "p_one_sided": one_sided(r, p, 1)}
    base = b7_resid(share, [m], shared)
    plus = b7_resid(share, [m, g], shared)
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(2000):
        i = rng.integers(0, len(keys), len(keys))
        if shared[i].all() or not shared[i].any():
            continue
        b0 = b7_resid(share[i], [m[i]], shared[i])
        b1 = b7_resid(share[i], [m[i], g[i]], shared[i])
        boot.append(1 - b1 / b0)
    rep["T2_B7G_rank_resid"] = {
        "given_m": float(base),
        "given_m_Gfi1": float(plus),
        "shrink": float(1 - plus / base),
        "shrink_95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "pass": bool(1 - plus / base >= 1 / 3),
    }
    r, p = partial_multi(g, size, [m])
    rep["T3_Gfi1_Gsize_given_m"] = {"partial_rho": float(r), "p_one_sided": one_sided(r, p, -1)}
    sec, ps = {}, []
    for name in [x for x in ("Rasgrp1", "Csf3r") if x in gene]:
        r1, p1 = partial_multi(gene[name], share, [m, size])
        r3, p3 = partial_multi(gene[name], size, [m])
        sec[name] = {"T1_partial_rho": float(r1), "T1_p": float(p1)}
        sec[name] |= {"T3_partial_rho": float(r3), "T3_p": float(p3)}
        ps += [p1, p3]
    q = iter(bh(ps)) if ps else iter([])
    for name in sec:
        sec[name]["T1_bh_q"], sec[name]["T3_bh_q"] = float(next(q)), float(next(q))
    rep["T4_secondary"] = sec
    rep["readings"] = {
        "T1_pass": rep["T1_Gfi1_Nshare_given_m"]["p_one_sided"] < 0.05
        and rep["T1_Gfi1_Nshare_given_m"]["partial_rho"] > 0,
        "T2_pass": rep["T2_B7G_rank_resid"]["pass"],
    }
    json.dump(rep, open(os.path.join(OUT, "b23_gfi1.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
