"""G3: does clone 1 vs clone 3 A-P separation fade over passages? See EXPERIMENT_LOG G3.

    uv run --with openpyxl python g3_fade.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "data", "gastruloid_raj", "raw_ls2")
OUT = os.path.join(HERE, "results_gastruloid")
FILES = {1: "clone1+3_pass1_04222024.xlsx", 2: "clone1+3_pass2_04242024.xlsx",
         3: "clone1+3_pass3_4272024.xlsx", 4: "clone1+3_pass4_04292024.xlsx"}
N = 30


def resample(s):
    v = np.array([float(x) for x in str(s).split(",")])
    return v[np.round(np.linspace(0, len(v) - 1, N)).astype(int)]


def enrich(v):
    return (v[: N // 2].mean() - v[N // 2:].mean()) / v.mean()


def load():
    rows = []
    for p, f in FILES.items():
        d = pd.read_excel(os.path.join(RAW, f))
        ex = d[d["Condition"] == "EX"]
        for aid, g in ex.groupby("Annotation ID"):
            r = g[g["Header"] == "Channel 2"]["Values"]
            gr = g[g["Header"] == "Channel 0"]["Values"]
            if len(r) != 1 or len(gr) != 1:
                continue
            er, eg = enrich(resample(r.iloc[0])), enrich(resample(gr.iloc[0]))
            rows.append({"passage": p, "id": aid, "E_red": er, "E_green": eg, "S": er - eg})
    return pd.DataFrame(rows)


def stats(df):
    t = df["passage"].to_numpy(float)
    s = df["S"].to_numpy()
    slope = np.polyfit(t, s, 1)[0]
    m = df.groupby("passage")["S"].mean()
    ratio = m[4] / m[1]
    fit = least_squares(lambda q: q[0] * q[1] ** (m.index.to_numpy() - 1) - m.to_numpy(), [m[1], 0.8],
                        bounds=([-10, 0.0], [10, 1.5]))
    return slope, ratio, fit.x[1], m


def main(boot=2000, seed=64):
    df = load()
    slope, ratio, r, m = stats(df)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(boot):
        b = pd.concat([g.sample(len(g), replace=True, random_state=int(rng.integers(1 << 31)))
                       for _, g in df.groupby("passage")])
        bs.append(stats(b)[:3])
    bs = np.array(bs)
    ci = lambda j: np.percentile(bs[:, j], [2.5, 97.5]).tolist()
    r_ci = ci(2)
    hl = lambda x: float(np.log(2) / -np.log(x)) if 0 < x < 1 else float("inf")
    res = {
        "n_per_passage": df.groupby("passage").size().to_dict(),
        "S_mean": m.round(4).to_dict(),
        "S_sd": df.groupby("passage")["S"].std().round(4).to_dict(),
        "share_red_anterior_of_green": df.assign(a=df["S"] > 0).groupby("passage")["a"].mean().round(3).to_dict(),
        "share_red_anterior": df.assign(a=df["E_red"] > 0).groupby("passage")["a"].mean().round(3).to_dict(),
        "share_green_posterior": df.assign(a=df["E_green"] < 0).groupby("passage")["a"].mean().round(3).to_dict(),
        "slope": [float(slope), *ci(0)], "ratio_4_over_1": [float(ratio), *ci(1)],
        "retention_r": [float(r), *r_ci],
        "half_life_passages": [hl(r), hl(r_ci[0]), hl(r_ci[1])],
        "rho_per_division": {g: [float(r ** (1 / g)), float(r_ci[0] ** (1 / g)), float(r_ci[1] ** (1 / g))]
                             for g in (3, 4.5, 6)},
    }
    sl, ra = res["slope"], res["ratio_4_over_1"]
    if sl[2] < 0 and ra[2] < 1:
        res["reading"] = "fades"
    elif ra[1] <= 1 <= ra[2]:
        res["reading"] = "stable"
    else:
        res["reading"] = "other (described)"
    json.dump(res, open(os.path.join(OUT, "g3_fade.json"), "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
