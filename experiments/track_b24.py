"""B24: is B7 G an artefact (cross-contamination between wells)? Design check on the data, with
the best gated E38 fit as the reference. See the B24 pre-registration in EXPERIMENT_LOG.md.

    uv run python experiments/track_b24.py
"""

from __future__ import annotations

import glob
import json
import os

import e26_selection as e26
import e32_multinomial as e32
import e33_measured as e33
import e37_pool as e37
import numpy as np
from clone_state_model import OUT  # noqa: E402
from fit_cytokine import pooled
from sweep_guard import gated
from track_b13 import partial_multi

STRATA = {"2": (2, 2), "3-5": (3, 5), "6+": (6, 10**9)}


def stats(cnt):
    pg, pm = pooled(cnt["G"]), pooled(cnt["M"])
    gk = [k for k, v in pg.items() if v[0] + v[1] >= 2]
    gsize = np.array([pg[k][0] + pg[k][1] for k in gk])
    share = np.array([pg[k][0] / (pg[k][0] + pg[k][1]) for k in gk])
    msize = np.array([pm.get(k, [0, 0])[0] + pm.get(k, [0, 0])[1] for k in gk])
    shared = msize >= 2
    out = {"n_G": len(gk), "n_shared": int(shared.sum())}
    out["B7G"] = float(share[shared].mean() - share[~shared].mean())
    r, _ = partial_multi(share, msize, [np.log(gsize)])
    out["D1_rho"] = float(r)
    for name, (lo, hi) in STRATA.items():
        s = (gsize >= lo) & (gsize <= hi)
        ok = s & shared
        out[f"D2_{name}"] = (
            float(share[ok].mean() - share[s & ~shared].mean()) if ok.any() else np.nan
        )
        out[f"D2_{name}_n_shared"] = int(ok.sum())
    sig = shared & (gsize <= 3) & (share == 0) & (msize >= 5 * gsize)
    out["D3_frac"] = float(sig.sum() / max(shared.sum(), 1))
    return out


def d4(cnt):
    """Same-day vs other-day co-presence (data only; the simulation has no days)."""
    rows = {"same": [], "other": [], "only": []}
    for d, e in ((4, 6), (6, 4)):
        g, md, me = cnt["G"].get(d, {}), cnt["M"].get(d, {}), cnt["M"].get(e, {})
        for k, v in g.items():
            n = v[0] + v[1]
            if n < 2:
                continue
            s = v[0] / n
            a = sum(md.get(k, [0, 0])[:2])
            b = sum(me.get(k, [0, 0])[:2])
            if a >= 2:
                rows["same"].append(s)
            elif b >= 2:
                rows["other"].append(s)
            elif a + b == 0:
                rows["only"].append(s)
    base = np.mean(rows["only"])
    return {
        "n": {k: len(v) for k, v in rows.items()},
        "gap_same": float(np.mean(rows["same"]) - base) if rows["same"] else None,
        "gap_other": float(np.mean(rows["other"]) - base) if rows["other"] else None,
    }


def best_e38():
    cnt = e33.load_counts(e33.CONDS5)
    co = e32.copresence(e33.split(cnt, None))
    reps = [json.load(open(f)) for f in glob.glob(os.path.join(OUT, "e38_s*.json"))]
    reps = [r for r in reps if gated(r, co)]
    return min(reps, key=lambda r: r["loss"])


def main(nsim=200):
    cnt = e33.load_counts(e33.CONDS5)
    fit = e33.split(cnt, None)
    data = stats(fit)
    ref = best_e38()
    clones = e26.universe(cnt)
    meas = e33.measured(clones, np.random.default_rng(77))
    sims = [
        stats(
            e33.split(
                e37.simulate(clones, ref["params"], np.random.default_rng(1000 + s), meas), None
            )
        )
        for s in range(nsim)
    ]
    rep = {"reference": f"e38_s{ref['seed']}", "data": data, "sim": {}}
    for k, v in data.items():
        x = np.array([s[k] for s in sims], float)
        x = x[np.isfinite(x)]
        lo, hi = np.percentile(x, [2.5, 97.5]) if len(x) else (np.nan, np.nan)
        rep["sim"][k] = {
            "mean": float(x.mean()) if len(x) else None,
            "95": [float(lo), float(hi)],
            "data_outside": bool(np.isfinite(v) and not lo <= v <= hi),
        }
    rep["D4"] = d4(fit)
    out = {k: rep["sim"][f"D2_{k}"]["data_outside"] for k in STRATA}
    g4 = rep["D4"]
    same_twice = (
        g4["gap_same"] is not None
        and g4["gap_other"] is not None
        and abs(g4["gap_same"]) >= 2 * abs(g4["gap_other"])
    )
    contam = out["2"] and not out["6+"] and rep["sim"]["D3_frac"]["data_outside"] and same_twice
    bio = out["6+"] and not same_twice
    rep["reading"] = "contamination-like" if contam else ("biology" if bio else "mixed")
    rep["checks"] = {"D2_outside": out, "D4_same_twice_other": same_twice}
    json.dump(rep, open(os.path.join(OUT, "b24_b7_design.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
