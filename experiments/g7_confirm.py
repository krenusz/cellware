"""G7: sealed test of simulator v2. Read ONLY through test_oracle (one look each). See EXPERIMENT_LOG G7.
Predictions: results_gastruloid/g7_predictions.json (written by g7_model.py on open data, committed first).
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from g7_model import N, pol, resample

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "data", "gastruloid_raj", "raw_ls2")
PRED = os.path.join(HERE, "results_gastruloid", "g7_predictions.json")
PAIRS = "3+7_6+7_linescan_5032025.csv"
SERIES = {  # (clone, passage): (file, channel, sign toward parent)
    (1, 2): ("clone1%_pass2_07092024.xlsx", "Channel 0", -1),
    (3, 2): ("clone3%_pass2_07092024.xlsx", "Channel 1", 1),
    (3, 3): ("clone3%_pass3_07112024.xlsx", "Channel 1", 1),
    (3, 4): ("clone3%_pass4_7132024.xlsx", "Channel 1", 1),
}
assert N == 30


def _boot(x, f=np.mean, B=2000, seed=77):
    rng = np.random.default_rng(seed)
    return np.percentile([f(rng.choice(x, len(x))) for _ in range(B)], [2.5, 97.5]).tolist()


def g7a_pairs():
    pred = json.load(open(PRED))["G7a"]
    d = pd.read_csv(os.path.join(RAW, PAIRS))
    out = {}
    for tag, key in (("3+7_linescan", "3+7"), ("6+7_linescan", "6+7")):
        D = []
        for _, g in d[d["Tags"] == tag].groupby("Annotation ID"):
            r, gr = g[g["Header"] == "Channel 2"]["Values"], g[g["Header"] == "Channel 0"]["Values"]
            if len(r) == 1 and len(gr) == 1:
                D.append(pol(resample(r.iloc[0])) - pol(resample(gr.iloc[0])))
        D = np.array(D)
        D = D[np.isfinite(D)]
        share = float((D > 0).mean())
        p = pred[key]
        if key == "6+7":
            reading = "pass (clone 7 anterior >= 0.75)" if 1 - share >= 0.75 else "fail"
        else:
            lo, hi = p["share_red_anterior"][1:]
            reading = "pass (share inside predicted interval)" if lo <= share <= hi else "fail"
        out[key] = {"n": int(len(D)), "share_red_anterior": share,
                    "D_mean": [float(D.mean()), *_boot(D)], "predicted_share": p["share_red_anterior"],
                    "predicted_D": p["D_red_minus_green_mean"], "reading": reading}
    return out


def g7b_memory():
    pred = json.load(open(PRED))["G7b"]
    out = {}
    for (c, p), (f, ch, sgn) in SERIES.items():
        d = pd.read_excel(os.path.join(RAW, f))
        d = d[(d["Header"] == ch) & (d["Condition"].isin([0.25, 0.5]))]
        P = np.array([sgn * pol(resample(v)) for v in d["Values"]])
        P = P[np.isfinite(P)]
        k = f"clone{c}_p{p}"
        v2, v1 = pred[k]["v2"], pred[k]["v1"]
        m = float(P.mean())
        in2, in1 = v2[1] <= m <= v2[2], v1[1] <= m <= v1[2]
        out[k] = {"n": int(len(P)), "mean_signed_P": [m, *_boot(P)], "pred_v2": v2, "pred_v1": v1,
                  "inside_v2": bool(in2), "inside_v1": bool(in1)}
    c1 = out["clone1_p2"]
    out["primary_clone1_p2"] = ("v2 (clone-specific memory)" if c1["inside_v2"] and not c1["inside_v1"] else
                                "v1 (single rate)" if c1["inside_v1"] and not c1["inside_v2"] else
                                "both" if c1["inside_v1"] else "neither")
    out["clone3_inside_v2"] = int(sum(out[f"clone3_p{p}"]["inside_v2"] for p in (2, 3, 4)))
    return out
