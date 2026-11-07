"""C4-S2b: T vs Igfbp5 domain separation by composition (descriptive). See EXPERIMENT_LOG C4-S2b.

    uv run python c4s2b_ap_separation.py

Reads only data/gastruloid_raj/T_Igfbp5_09022026/gastruloid_T_Igfbp5.csv (spent as exploratory).
Channel 1 = T (YFP), Channel 2 = Igfbp5 (Cy5). EMD in % of length; overlap in [0, 1].
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "data", "gastruloid_raj", "T_Igfbp5_09022026", "gastruloid_T_Igfbp5.csv")
OUT = os.path.join(HERE, "results_gastruloid", "c4s2b_ap_separation.json")
NG = 200
E_H = {"5": 0.0, "6": 0.0, "bulk+5": 0.297, "bulk+6": 0.306, "5+6": 0.546}  # C3-S2 E[H] (50% mixes)


def num(s):
    return np.array([float(v) for v in str(s).split(",") if v != ""])


def metrics(x, y, t, ig):
    n = min(len(x), len(y), len(t), len(ig))
    x, y, t, ig = x[:n], y[:n], t[:n], ig[:n]
    s = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    u = s / s[-1]
    g = np.linspace(0, 1, NG)
    a = np.interp(g, u, t)
    b = np.interp(g, u, ig)
    a, b = np.clip(a - a.min(), 0, None), np.clip(b - b.min(), 0, None)
    if a.sum() <= 0 or b.sum() <= 0:
        return np.nan, np.nan
    pa, pb = a / a.sum(), b / b.sum()
    dx = 1 / (NG - 1)
    return 100 * np.abs(np.cumsum(pa) - np.cumsum(pb)).sum() * dx, float(np.minimum(pa, pb).sum())


def main():
    d = pd.read_csv(CSV)
    rows = []
    for aid, g in d.groupby("Annotation ID"):
        h = {r["Header"]: num(r["Values"]) for _, r in g.iterrows()}
        emd, ov = metrics(h["X"], h["Y"], h["Channel 1"], h["Channel 2"])
        rows.append({"id": aid, "cond": str(g["Tags"].iloc[0]), "EMD": emd, "overlap": ov})
    r = pd.DataFrame(rows)
    per = r.groupby("cond").agg(n=("EMD", "size"), EMD_median=("EMD", "median"), EMD_min=("EMD", "min"),
                                EMD_max=("EMD", "max"), overlap_median=("overlap", "median"))
    nb = per.loc[[c for c in E_H if c in per.index]]
    order = nb["EMD_median"].sort_values(ascending=False).index.tolist()
    consistent = bool(order[0] == "5+6" and set(order[-2:]) == {"5", "6"})
    rho = spearmanr([E_H[c] for c in nb.index], nb["EMD_median"]).statistic
    res = {"per_condition": per.round(3).reset_index().to_dict(orient="records"),
           "nonbulk_order_by_median_EMD": order, "reading": "consistent" if consistent else "not consistent",
           "spearman_EH_vs_median_EMD": round(float(rho), 3),
           "per_gastruloid": r.round(3).to_dict(orient="records")}
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_gastruloid"}, indent=1))


if __name__ == "__main__":
    main()
