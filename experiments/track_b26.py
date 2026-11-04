"""B26: what do monocyte-committed-looking clones make under Epo and SCF?
See the B26 pre-registration in EXPERIMENT_LOG.md.

    uv run python experiments/track_b26.py
"""

from __future__ import annotations

import json
import os

import e26_selection as e26
import e33_measured as e33
import e37_pool as e37
import numpy as np
import track_b24 as b24
from clone_state_model import OUT, split
from fit_cytokine import pooled

MYE = ("full", "G", "M")
EPS = ("Epo", "SCF")
STRATA = {"2-5": (2, 5), "6-15": (6, 15), "16+": (16, 10**9)}


def stats(cnt):
    p = {c: pooled(cnt[c]) for c in cnt}
    clones = set().union(*[set(p[c]) for c in MYE])
    a, b, size = [], [], {}
    for k in clones:
        pres = [c for c in MYE if k in p[c] and p[c][k][0] + p[c][k][1] >= 2]
        if len(pres) < 2:
            continue
        size[k] = sum(p[c][k][0] + p[c][k][1] for c in pres)
        (a if all(p[c][k][0] == 0 for c in pres) else b).append(k)

    def epo_scf(keys):
        present, shares = [], []
        for k in keys:
            n = m = 0
            for c in EPS:
                v = p[c].get(k, [0, 0])
                n, m = n + v[0], m + v[1]
            present.append(n + m >= 2)
            if n + m >= 2:
                shares.append(n / (n + m))
        return np.array(present), shares

    out = {"n_A": len(a), "n_B": len(b), "frac_A": len(a) / max(len(a) + len(b), 1)}
    pa, sa = epo_scf(a)
    pb, sb = epo_scf(b)
    out["C2_A_Nshare"] = float(np.mean(sa)) if sa else np.nan
    out["C2_B_Nshare"] = float(np.mean(sb)) if sb else np.nan
    out["n_A_in_EpoSCF"] = int(pa.sum())
    for name, (lo, hi) in STRATA.items():
        ia = np.array([lo <= size[k] <= hi for k in a], bool)
        ib = np.array([lo <= size[k] <= hi for k in b], bool)
        out[f"C1_{name}_A"] = float(pa[ia].mean()) if ia.any() else np.nan
        out[f"C1_{name}_B"] = float(pb[ib].mean()) if ib.any() else np.nan
        out[f"C1_{name}_nA"] = int(ia.sum())
    return out


def main(nsim=200):
    cnt = e33.load_counts(e33.CONDS5)
    data = stats(split(cnt, None))
    ref = b24.best_e38()
    clones = e26.universe(cnt)
    meas = e33.measured(clones, np.random.default_rng(77))
    sims = [
        stats(
            split(e37.simulate(clones, ref["params"], np.random.default_rng(1000 + s), meas), None)
        )
        for s in range(nsim)
    ]
    rep = {"reference": f"e38_s{ref['seed']}", "data": data, "sim": {}}
    for k, v in data.items():
        x = np.array([s[k] for s in sims], float)
        x = x[np.isfinite(x)]
        if not len(x):
            continue
        lo, hi = np.percentile(x, [2.5, 97.5])
        rep["sim"][k] = {"mean": float(x.mean()), "95": [float(lo), float(hi)]}
        rep["sim"][k]["data_inside"] = bool(np.isfinite(v) and lo <= v <= hi)
    cond = np.isfinite(data["C2_A_Nshare"]) and data["C2_A_Nshare"] >= 0.5
    pres = all(
        data[f"C1_{s}_A"] < 0.5 * data[f"C1_{s}_B"] for s in STRATA if data[f"C1_{s}_nA"] >= 10
    ) and any(data[f"C1_{s}_nA"] >= 10 for s in STRATA)
    if cond:
        e38_ok = rep["sim"].get("C2_A_Nshare", {}).get("data_inside", False)
        rep["reading"] = "condition-dependent; " + (
            "E38 reproduces fate, count is the issue"
            if e38_ok
            else "E38 misses it: class warranted"
        )
    elif pres:
        rep["reading"] = "presence problem"
    else:
        rep["reading"] = "mixed"
    json.dump(rep, open(os.path.join(OUT, "b26_mcommit.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
