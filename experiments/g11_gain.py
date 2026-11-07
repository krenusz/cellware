"""G11: series gain + drug compression, conditional on each series' control. Train series only.
See EXPERIMENT_LOG G11.

    uv run --with openpyxl python g11_gain.py
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.optimize import minimize

import g9_mech

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_gastruloid")
SEL = os.path.join(HERE, "selection")
MODELS = ("R0", "R1", "R2", "R3", "R4")


def items():
    """(series, pair, drug, D_drug, se_drug, D_ctrl, se_ctrl) for every train drug condition."""
    m = g9_mech.cond_means(g9_mech.drug_D())
    out = []
    for (s, p), g in m.groupby(["series", "pair"]):
        c = g[g["cond"] == "ctrl"].iloc[0]
        for r in g[g["cond"] != "ctrl"].itertuples():
            out.append((s, p, r.cond, float(r.mean), float(r.se), float(c["mean"]), float(c["se"])))
    return out


def keys(model, its):
    if model in ("R1", "R3"):
        return sorted({i[2] for i in its})
    if model == "R2":
        return ["all"]
    if model == "R4":
        return sorted({f"{i[1]}|{i[2]}" for i in its})
    return []


def par(model, th, i):
    if model in ("R1", "R3"):
        return th.get(i[2], 1.0 if model == "R1" else 0.0)
    if model == "R2":
        return th["all"]
    if model == "R4":
        return th.get(f"{i[1]}|{i[2]}", th.get(i[2], 1.0))
    return None


def mu_var(model, th, i):
    _, _, _, _, sd, c, sc = i
    if model == "R0":
        return c, sd ** 2 + sc ** 2
    if model == "R3":
        return c + par(model, th, i), sd ** 2 + sc ** 2
    g = par(model, th, i)
    return g * c, sd ** 2 + g ** 2 * sc ** 2


def ll(model, th, its):
    t = 0.0
    for i in its:
        mu, v = mu_var(model, th, i)
        t += -0.5 * (i[3] - mu) ** 2 / v - 0.5 * np.log(2 * np.pi * v)
    return float(t)


def fit(model, its):
    ks = keys(model, its)
    if not ks:
        return {}
    x0 = np.full(len(ks), 0.0 if model == "R3" else 0.6)
    res = minimize(lambda x: -ll(model, dict(zip(ks, x)), its), x0, method="Nelder-Mead",
                   options={"maxiter": 40000, "xatol": 1e-7, "fatol": 1e-10})
    return dict(zip(ks, map(float, res.x)))


def main():
    its = items()
    series = sorted({i[0] for i in its})
    res = {"items": [dict(series=i[0], pair=i[1], drug=i[2], D=round(i[3], 3), se=round(i[4], 3),
                          D_ctrl=round(i[5], 3), se_ctrl=round(i[6], 3)) for i in its], "models": {}}
    for model in MODELS:
        cv = {}
        for s in series:
            th = fit(model, [i for i in its if i[0] != s])
            if model == "R4":  # unseen pair x drug: fall back to drug gain fitted as R1
                th = {**fit("R1", [i for i in its if i[0] != s]), **th}
            cv[s] = ll(model, th, [i for i in its if i[0] == s])
        th = fit(model, its)
        misses = []
        for i in its:
            mu, v = mu_var(model, th, i)
            z = (i[3] - mu) / np.sqrt(v)
            if abs(z) > 2.5:
                misses.append(f"{i[0]} {i[1]} {i[2]}: obs {i[3]:.2f} fit {mu:.2f} z {z:.2f}")
        res["models"][model] = {"cv_loglik": round(sum(cv.values()), 2), "cv_by_series": {k: round(v, 2) for k, v in cv.items()},
                                "params": {k: round(v, 3) for k, v in th.items()}, "consistency_misses": misses}
    ranking = sorted(res["models"], key=lambda k: -res["models"][k]["cv_loglik"])
    qual = [k for k in ranking if not res["models"][k]["consistency_misses"]]
    res["ranking_by_cv"], res["best_qualifying"] = ranking, (qual[0] if qual else None)
    chosen = ranking[0] if qual and qual[0] == ranking[0] else None
    res["look_allowed"] = bool(chosen and chosen != "R0")
    if res["look_allowed"]:
        os.makedirs(SEL, exist_ok=True)
        json.dump({"test": "gastruloid_g11_drug", "candidates": {k: res["models"][k]["cv_loglik"] for k in ranking},
                   "higher_is_better": True, "chosen": chosen, "params": res["models"][chosen]["params"],
                   "note": "G11: leave-one-series-out CV on train drug series, conditional on each series' control; "
                           "consistency within 2.5 combined SE"},
                  open(os.path.join(SEL, "G11.json"), "w"), indent=1)
    json.dump(res, open(os.path.join(OUT, "g11_gain.json"), "w"), indent=1)
    for i in res["items"]:
        print(i)
    print("ranking", ranking, "best qualifying", res["best_qualifying"], "look allowed", res["look_allowed"])
    for k in ranking:
        v = res["models"][k]
        print(f"  {k}: CV {v['cv_loglik']} {v['cv_by_series']} params {v['params']}")
        for mm in v["consistency_misses"]:
            print("     miss:", mm)


if __name__ == "__main__":
    main()
