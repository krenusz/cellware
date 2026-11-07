"""G10: is pair sorting a transitive 1D ranking of clone states? See EXPERIMENT_LOG G10.

    uv run --with openpyxl python g10_sorting.py
Data: control pairs only (G3 open, G9 train series, G7a spent summaries from the ledger).
"""

from __future__ import annotations

import csv
import json
import os

import numpy as np
from scipy.optimize import minimize

import g7_model
import g9_mech

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_gastruloid")
TAU = 0.26


def sim_map(grid=np.arange(-4.0, 4.01, 0.25), reps=300, seed=101):
    """Mean D = P_a - P_b for a 50:50 pair with state difference delta (simulator v2)."""
    rng = np.random.default_rng(seed)
    m = []
    for dl in grid:
        s = g7_model.sim_pairs(dl / 2, -dl / 2, reps, TAU, rng)
        m.append(float((s[:, 0] - s[:, 1]).mean()))
    return grid, np.array(m)


def ledger(test):
    with open(os.path.join(HERE, "test_ledger.tsv"), encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["test"] == test:
                return json.loads(r["result"])


def observations():
    obs = []  # (pair_type, series, (a, b) with D = P_a - P_b, mean, se, n)
    op = g7_model.open_pairs()
    p1 = op[op["passage"] == 1]["D"]
    obs.append(("1+3", "apr24", (3, 1), float(p1.mean()), float(p1.std() / np.sqrt(len(p1))), len(p1)))
    dd = g9_mech.drug_D()
    for (series, pair), g in dd[dd["cond"] == "ctrl"].groupby(["series", "pair"]):
        ab = (3, 1) if pair == "1+3" else (6, 1)
        obs.append((pair, series, ab, float(g["D"].mean()), float(g["D"].std() / np.sqrt(len(g))), len(g)))
    g7 = ledger("gastruloid_g7_pairs")
    for key, ab in (("3+7", (3, 7)), ("6+7", (6, 7))):
        m, lo, hi = g7[key]["D_mean"]
        obs.append((key, "may25_g7a", ab, m, (hi - lo) / 3.92, g7[key]["n"]))
    return obs


def main():
    grid, mp = sim_map()
    f = lambda dl: np.interp(dl, grid, mp)
    mu0, _ = g7_model.fit_mu()
    obs = observations()
    types = sorted({o[0] for o in obs})

    def pred(model, th, o):
        a, b = o[2]
        if model == "S0":
            return f(mu0[a] - mu0[b])
        if model == "S1":
            return f(th[0] * (mu0[a] - mu0[b]))
        st = {1: 0.0, 3: th[0], 6: th[1], 7: th[2]}
        return f(st[a] - st[b])

    def fit(model, tr):
        if model == "S0":
            return np.zeros(0)
        x0 = [1.0] if model == "S1" else [mu0[3] - mu0[1], mu0[6] - mu0[1], mu0[7] - mu0[1]]
        loss = lambda th: sum(((o[3] - pred(model, th, o)) / o[4]) ** 2 for o in tr)
        best = min((minimize(loss, np.array(x0) * s, method="Nelder-Mead", options={"maxiter": 20000}) for s in (0.5, 1, 1.5)),
                   key=lambda r: r.fun)
        return best.x

    res = {"observations": [dict(pair=o[0], series=o[1], D_mean=round(o[3], 3), se=round(o[4], 3), n=o[5]) for o in obs],
           "map_D_at_delta": {str(round(float(g), 2)): round(float(v), 3) for g, v in zip(grid[::4], mp[::4])},
           "models": {}}
    for model in ("S0", "S1", "S2"):
        cv = {}
        for t in types:
            tr = [o for o in obs if o[0] != t]
            te = [o for o in obs if o[0] == t]
            th = fit(model, tr)
            cv[t] = float(sum(-0.5 * ((o[3] - pred(model, th, o)) / o[4]) ** 2 - np.log(o[4] * np.sqrt(2 * np.pi)) for o in te))
        th = fit(model, obs)
        misses = [f"{o[0]} {o[1]}: obs {o[3]:.2f} fit {pred(model, th, o):.2f} z {(o[3] - pred(model, th, o)) / o[4]:.2f}"
                  for o in obs if abs((o[3] - pred(model, th, o)) / o[4]) > 2.5]
        res["models"][model] = {"cv_loglik": round(sum(cv.values()), 2), "cv_by_pair": {k: round(v, 2) for k, v in cv.items()},
                                "params": [round(float(x), 3) for x in th], "consistency_misses": misses}
    res = g9_mech.select(res, "gastruloid_g10_sorting", "G10")
    json.dump(res, open(os.path.join(OUT, "g10_sorting.json"), "w"), indent=1)
    print("observations:", json.dumps(res["observations"], indent=0))
    print("ranking", res["ranking_by_cv"], "best qualifying", res["best_qualifying"])
    for k in res["ranking_by_cv"]:
        v = res["models"][k]
        print(f"  {k}: CV {v['cv_loglik']} by pair {v['cv_by_pair']} params {v['params']}")
        for mm in v["consistency_misses"]:
            print("     miss:", mm)


if __name__ == "__main__":
    main()
