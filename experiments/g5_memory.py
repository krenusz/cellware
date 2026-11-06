"""G5 out-of-sample test: is gastruloid clone memory lost per division or per passage event?
Subclone polarity (all_normalized_scans.csv) vs parent polarity (G1a lines). See EXPERIMENT_LOG G5.

    uv run python g5_memory.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from g1_gastruloid import DATA, OUT, polarity

SIGN = {"head": 1.0, "tail": -1.0}  # head = anterior clone 3, tail = posterior clone 1
PARENT = {"head": "Clone 3", "tail": "Clone 1"}
PER_DIVISION_MAX = 0.92 ** 13  # 0.34


def subclone_polarity():
    d = pd.read_csv(os.path.join(DATA, "all_normalized_scans.csv"))
    rows = []
    for (cond, sub, sid), g in d.groupby(["condition", "subclone", "scan_id"]):
        v = savgol_filter(g.sort_values("position_index")["normalized_intensity"].to_numpy(), 5, 2)
        n = len(v)
        side, pas = cond.split("_")
        rows.append({"side": side, "passage": pas, "subclone": sub, "scan": sid,
                     "P": float(v[: n // 2].mean() - v[n // 2:].mean())})
    return pd.DataFrame(rows)


def cond_mean(df):
    """sign-adjusted mean over subclones (each subclone weighted equally)."""
    return float(df.groupby("subclone")["P_adj"].mean().mean())


def main(B=2000, seed=67):
    rng = np.random.default_rng(seed)
    S = subclone_polarity()
    S["P_adj"] = S["P"] * S["side"].map(SIGN)
    G = polarity()
    par = {s: G[G["clone"] == PARENT[s]]["P"].to_numpy() * SIGN[s] for s in SIGN}
    res = {"n_scans": S.groupby(["side", "passage"]).size().to_dict(),
           "n_subclones": S.groupby(["side", "passage"])["subclone"].nunique().to_dict()}
    res = {k: {f"{a}_{b}": int(v) for (a, b), v in d.items()} for k, d in res.items()}
    est, bs = {}, {}
    for s in SIGN:
        for p in ("p1", "p2"):
            est[(s, p)] = cond_mean(S[(S["side"] == s) & (S["passage"] == p)])
        est[(s, "parent")] = float(par[s].mean())
    draws = []
    for _ in range(B):
        x = {}
        for s in SIGN:
            for p in ("p1", "p2"):
                sub = S[(S["side"] == s) & (S["passage"] == p)]
                parts = [g.sample(len(g), replace=True, random_state=int(rng.integers(1 << 31)))
                         for _, g in sub.groupby("subclone")]
                x[(s, p)] = cond_mean(pd.concat(parts))
            x[(s, "parent")] = float(rng.choice(par[s], len(par[s])).mean())
        draws.append(x)
    out = {}
    for s in list(SIGN) + ["both"]:
        def ratio(x, a, b):
            if s == "both":
                return (x[("head", a)] + x[("tail", a)]) / (x[("head", b)] + x[("tail", b)])
            return x[(s, a)] / x[(s, b)]
        r1 = ratio(est, "p1", "parent")
        r2 = ratio(est, "p2", "p1")
        c1 = np.percentile([ratio(x, "p1", "parent") for x in draws], [2.5, 97.5]).tolist()
        c2 = np.percentile([ratio(x, "p2", "p1") for x in draws], [2.5, 97.5]).tolist()
        out[s] = {"p1_over_parent": [r1, *c1], "p2_over_p1": [r2, *c2]}
    res["means_sign_adjusted"] = {f"{a}_{b}": v for (a, b), v in est.items()}
    res["subclone_means"] = {f"{s}_{p}_{sc}": round(float(g["P_adj"].mean()), 3)
                             for (s, p, sc), g in S.groupby(["side", "passage", "subclone"])}
    res["ratios"] = out
    b1 = out["both"]["p1_over_parent"]
    if b1[1] > PER_DIVISION_MAX and b1[1] <= 1 <= b1[2]:
        r1 = "per passage event (per-division rejected)"
    elif b1[1] <= PER_DIVISION_MAX <= b1[2] and not (b1[1] <= 1 <= b1[2]):
        r1 = "per division"
    else:
        r1 = "other (described)"
    b2 = out["both"]["p2_over_p1"]
    r2 = ("consistent with G3 per-passage rate" if b2[1] <= 0.68 and b2[2] >= 0.50 else
          "below G3 rate: loss at the subcloning bottleneck exceeds the population rate" if b2[2] < 0.50
          else "above G3 rate")
    res["reading_1"], res["reading_2"] = r1, r2
    json.dump(res, open(os.path.join(OUT, "g5_memory.json"), "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
