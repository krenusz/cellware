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


AUTH = os.path.join(HERE, "..", "data", "gastruloid_raj", "rev_T_Igfbp5_sep", "separation_metrics.csv")


def metrics_bg(x, y, t, ig, q):
    """As metrics(), with background = q-th percentile (q = 0: minimum); also returns length and saturation."""
    n = min(len(x), len(y), len(t), len(ig))
    x, y, t, ig = x[:n], y[:n], t[:n], ig[:n]
    s = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    g = np.linspace(0, 1, NG)
    a, b = np.interp(g, s / s[-1], t), np.interp(g, s / s[-1], ig)
    a = np.clip(a - np.percentile(a, q), 0, None)
    b = np.clip(b - np.percentile(b, q), 0, None)
    pa, pb = a / a.sum(), b / b.sum()
    return (100 * np.abs(np.cumsum(pa) - np.cumsum(pb)).sum() / (NG - 1), float(np.minimum(pa, pb).sum()),
            float(s[-1]), float(np.mean(ig >= 65535)))


def ordering_ok(med):
    order = sorted(med, key=med.get, reverse=True)
    return order[0] == "5+6" and set(order[-2:]) == {"5", "6"}


def robust():
    d = pd.read_csv(CSV)
    rows = []
    for aid, g in d.groupby("Annotation ID"):
        h = {r["Header"]: num(r["Values"]) for _, r in g.iterrows()}
        e0, o0, L, sat = metrics_bg(h["X"], h["Y"], h["Channel 1"], h["Channel 2"], 0)
        e5, o5, _, _ = metrics_bg(h["X"], h["Y"], h["Channel 1"], h["Channel 2"], 5)
        rows.append({"id": aid, "cond": str(g["Tags"].iloc[0]), "EMD": e0, "EMD_p05": e5, "len": L, "sat": sat})
    r = pd.DataFrame(rows)
    a = pd.read_csv(AUTH)[["id", "EMD", "EMD_p05"]].rename(columns={"EMD": "EMD_auth", "EMD_p05": "EMD_p05_auth"})
    m = r.merge(a, on="id", how="inner")
    res = {"1_implementation": {"n_matched": int(len(m)),
                                "pearson_EMD": round(float(np.corrcoef(m["EMD"], m["EMD_auth"])[0, 1]), 4),
                                "max_abs_diff": round(float((m["EMD"] - m["EMD_auth"]).abs().max()), 3)}}
    nb = [c for c in E_H if c in set(r["cond"])]
    med5 = r[r["cond"].isin(nb)].groupby("cond")["EMD_p05"].median()
    res["2_background_p05"] = {"medians": med5.round(2).to_dict(), "consistent": ordering_ok(med5.to_dict()),
                               "spearman_EH": round(float(spearmanr([E_H[c] for c in med5.index], med5).statistic), 3)}
    res["3_confounds"] = {"spearman_EMD_len": round(float(spearmanr(r["EMD"], r["len"]).statistic), 3),
                          "spearman_EMD_sat": round(float(spearmanr(r["EMD"], r["sat"]).statistic), 3),
                          "median_len_by_cond": r.groupby("cond")["len"].median().round(0).to_dict(),
                          "median_sat_by_cond": r.groupby("cond")["sat"].median().round(3).to_dict()}
    # 4: power of the ordering check
    sub = r[r["cond"].isin(nb)]
    sd = float(np.sqrt(np.average(sub.groupby("cond")["EMD"].var(ddof=1), weights=sub.groupby("cond").size() - 1)))
    ns = sub.groupby("cond").size().to_dict()
    rng = np.random.default_rng(77)
    pw = {}
    for dval in (0, 0.5, 1, 2):
        hits = 0
        for _ in range(20000):
            med = {c: np.median(rng.normal(dval * E_H[c] / E_H["5+6"], 1, ns[c])) for c in nb}
            hits += ordering_ok(med)
        pw[f"d={dval}"] = round(hits / 20000, 3)
    res["4_power_ordering"] = {"pooled_within_sd_EMD": round(sd, 2), "n": ns, "P_consistent": pw}
    print(json.dumps(res, indent=1))
    json.dump(res, open(OUT.replace(".json", "_robust.json"), "w"), indent=1)


if __name__ == "__main__":
    import sys

    robust() if "--robust" in sys.argv else main()
