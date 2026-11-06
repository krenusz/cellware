"""G2: elongation of clone mixtures, division of labour vs generic mixing. See EXPERIMENT_LOG G2.

    uv run python g2_elongation.py
"""

from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "gastruloid_raj", "morph_csv", "Morphologies_1_2_3_4_5_6_7.csv")
OUT = os.path.join(HERE, "results_gastruloid")
COMP = ["bulk"] + [f"c{i}" for i in range(1, 8)]
ANT, POST = {3, 5, 7}, {1, 6}


def parse(s):
    """Composition string -> fractions over COMP."""
    f = dict.fromkeys(COMP, 0.0)
    s = s.strip()
    if s == "bulk":
        f["bulk"] = 1.0
        return f
    parts = re.findall(r"(\d+)%\s*clone\s*(\d)", s)
    tot = 0.0
    for pct, c in parts:
        v = float(pct) / 100
        f[f"c{c}"] += v
        tot += v
    if len(parts) > 1:  # clone mixtures: normalise (33% x 3 -> 1/3 each)
        for k in f:
            f[k] /= tot
    else:
        f["bulk"] = 1.0 - tot
    return f


def pair_class(f):
    cl = [int(k[1]) for k in COMP[1:] if f[k] > 0]
    if len(cl) < 2 or f["bulk"] > 0:
        return "none"
    if any(c in (2, 4) for c in cl) or len(cl) > 2:
        return "unl"
    a, b = cl
    if (a in ANT and b in POST) or (a in POST and b in ANT):
        return "comp"
    return "same"


def load():
    d = pd.read_csv(DATA)
    d = d.dropna(subset=["Total", "Elongated"])
    d = d[d["Total"] > 0].copy()
    F = np.array([[parse(s)[k] for k in COMP] for s in d["Composition"]])
    cls = np.array([pair_class(parse(s)) for s in d["Composition"]])
    return d, F, cls


def nll(theta, F, cls, k, n, terms):
    e = theta[:len(COMP)]
    eta = F @ e
    for j, t in enumerate(terms):
        eta = eta + theta[len(COMP) + j] * (cls == t)
    p = 1 / (1 + np.exp(-eta))
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.sum(k * np.log(p) + (n - k) * np.log(1 - p)) + 1e-4 * np.sum(theta ** 2)


def fit(F, cls, k, n, terms):
    x0 = np.zeros(len(COMP) + len(terms))
    r = minimize(nll, x0, args=(F, cls, k, n, terms), method="BFGS")
    return r.x, r.fun


def main(boot=2000, seed=63):
    d, F, cls = load()
    k, n = d["Elongated"].to_numpy(float), d["Total"].to_numpy(float)
    terms = ["comp", "same", "unl"]
    th0, f0 = fit(F, cls, k, n, [])
    th3, f3 = fit(F, cls, k, n, terms)
    dates = d["Date"].to_numpy()
    ud = np.unique(dates)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(boot):
        pick = rng.choice(ud, len(ud))
        ix = np.concatenate([np.where(dates == u)[0] for u in pick])
        th, _ = fit(F[ix], cls[ix], k[ix], n[ix], terms)
        bs.append(th[len(COMP):])
    bs = np.array(bs)
    ci = lambda v: np.percentile(v, [2.5, 97.5]).tolist()
    dc, ds = th3[len(COMP)], th3[len(COMP) + 1]
    res = {
        "n_rows": int(len(d)), "n_dates": int(len(ud)),
        "gastruloids_per_class": {c: int(n[cls == c].sum()) for c in ("comp", "same", "unl", "none")},
        "competence_M0": dict(zip(COMP, th0.round(3).tolist())),
        "nll_M0": float(f0), "nll_M3": float(f3),
        "delta_comp": [float(dc), *ci(bs[:, 0])], "delta_same": [float(ds), *ci(bs[:, 1])],
        "delta_unl": [float(th3[len(COMP) + 2]), *ci(bs[:, 2])],
        "comp_minus_same": [float(dc - ds), *ci(bs[:, 0] - bs[:, 1])],
    }
    c_lo, s_lo = res["delta_comp"][1], res["delta_same"][1]
    diff = res["comp_minus_same"]
    if min(res["gastruloids_per_class"]["comp"], res["gastruloids_per_class"]["same"]) < 40:
        reading = "descriptive (class under 40 gastruloids)"
    elif diff[1] > 0:
        reading = "division of labour"
    elif c_lo > 0 and s_lo > 0 and diff[1] <= 0 <= diff[2]:
        reading = "generic mixing rescue"
    elif res["delta_comp"][1] <= 0 <= res["delta_comp"][2] and res["delta_same"][1] <= 0 <= res["delta_same"][2]:
        reading = "additive (no mixing effect)"
    else:
        reading = "other pattern (described, no claim)"
    res["reading"] = reading
    # secondary: clone + bulk chimeras under M0
    chim = (F[:, 0] > 0) & (F[:, 0] < 1)
    p0 = 1 / (1 + np.exp(-(F @ th0[:len(COMP)])))
    rows = []
    for comp in sorted(set(d["Composition"][chim])):
        m = (d["Composition"] == comp).to_numpy()
        rows.append({"composition": comp, "n": int(n[m].sum()), "observed": float(k[m].sum() / n[m].sum()),
                     "M0_pred": float((p0[m] * n[m]).sum() / n[m].sum())})
    res["chimeras_M0"] = rows
    # per pair observed vs M0 (pure-clone competences only would be cleaner; M0 is joint)
    pr = []
    for comp in sorted(set(d["Composition"])):
        m = (d["Composition"] == comp).to_numpy()
        if cls[m][0] in ("comp", "same"):
            pr.append({"pair": comp, "class": str(cls[m][0]), "n": int(n[m].sum()),
                       "observed": float(k[m].sum() / n[m].sum()),
                       "M0_pred": float((p0[m] * n[m]).sum() / n[m].sum())})
    res["pairs"] = pr
    json.dump(res, open(os.path.join(OUT, "g2_elongation.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
