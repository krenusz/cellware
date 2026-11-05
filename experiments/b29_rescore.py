"""B29: re-score E38 and E46 primary statistics on the fit set, with pass/fail and empirical CRPS.
See the B29 pre-registration in EXPERIMENT_LOG.md.

    uv run python experiments/b29_rescore.py
"""

from __future__ import annotations

import glob
import json
import os

import e26_selection as e26
import e29_ceiling as e29
import e32_multinomial as e32
import e33_measured as e33
import e34_fullarm as e34
import e37_pool as e37
import e46_selexp as e46
import numpy as np
from clone_state_model import OUT, split
from sweep_guard import gated

CROSS = ("G-M", "full-G", "full-M")
B7 = ("G_mean_both_minus_Gonly", "M_both_minus_Monly")
KEYS = (*CROSS, *B7)


def crps(draws, y):
    x = np.asarray(draws, float)
    x = x[np.isfinite(x)]
    return float(np.mean(np.abs(x - y)) - 0.5 * np.mean(np.abs(x[:, None] - x[None, :])))


def score(mod, prm, clones, meas, data, nsim):
    draws = {k: [] for k in KEYS}
    for s in range(nsim):
        sim = split(mod.simulate(clones, prm, np.random.default_rng(1000 + s), meas), None)
        for k, v in e29.cross_r(sim).items():
            draws[k].append(v)
        for k, v in e33.presence_stats(sim).items():
            if k in B7:
                draws[k].append(v)
    out = {}
    for k in KEYS:
        x = np.array(draws[k], float)
        x = x[np.isfinite(x)]
        lo, hi = np.percentile(x, [2.5, 97.5])
        hi_share = float((x >= data[k]).mean())
        out[k] = {
            "data": data[k],
            "sim_mean": float(x.mean()),
            "sim_95": [float(lo), float(hi)],
            "width": float(hi - lo),
            "crps": crps(x, data[k]),
            "pass": bool(2 * min(hi_share, 1 - hi_share) >= 0.05)
            if k in CROSS
            else bool(lo <= data[k] <= hi),
        }
    return out


def verdict(rows):
    n = len(rows)
    agree = {k: sum(r[k]["pass"] for r in rows) for k in KEYS}
    stable = bool(n >= 9 and all(max(v, n - v) >= 2 * n / 3 for v in agree.values()))
    return {
        "gated": n,
        "component_pass": agree,
        "primary_pass": sum(all(r[k]["pass"] for k in KEYS) for r in rows),
        "stable": stable,
        "stable_and_passing": bool(stable and all(v >= 2 * n / 3 for v in agree.values())),
        "median_crps": {k: float(np.median([r[k]["crps"] for r in rows])) for k in KEYS},
        "median_width": {k: float(np.median([r[k]["width"] for r in rows])) for k in KEYS},
    }


def main(nsim=200):
    e34.e33.unpack_base = e33.unpack
    cnt = e33.load_counts(e33.CONDS5)
    fit = split(cnt, None)
    co = e32.copresence(fit)
    clones = e26.universe(cnt)
    meas = e33.measured(clones, np.random.default_rng(77))
    data = {**e29.cross_r(fit), **{k: v for k, v in e33.presence_stats(fit).items() if k in B7}}
    rep = {"data_fit_set": data, "models": {}}
    for name, mod in (("e38", e37), ("e46", e46)):
        res = [json.load(open(f)) for f in glob.glob(os.path.join(OUT, f"{name}_s*.json"))]
        rows = []
        for r in sorted((r for r in res if gated(r, co)), key=lambda r: r["seed"]):
            sc = score(mod, r["params"], clones, meas, data, nsim)
            rows.append({"seed": r["seed"], **sc})
            print(name, r["seed"], " ".join(f"{k[:6]}:{int(sc[k]['pass'])}" for k in KEYS), flush=True)
        v = verdict(rows)
        v["sum_median_crps"] = float(sum(v["median_crps"].values()))
        rep["models"][name] = {"verdict": v, "rows": rows}
    e46v = rep["models"]["e46"]["verdict"]
    rep["full_M_heldout_artefact"] = bool(e46v["component_pass"]["full-M"] >= 2 * e46v["gated"] / 3)
    json.dump(rep, open(os.path.join(OUT, "b29_rescore.json"), "w"), indent=1)
    print(json.dumps({"data_fit_set": data, "full_M_heldout_artefact": rep["full_M_heldout_artefact"],
                      **{m: rep["models"][m]["verdict"] for m in rep["models"]}}, indent=1))


if __name__ == "__main__":
    main()
