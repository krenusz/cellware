"""G6: sealed confirmation sets for chapter 3. Read ONLY through test_oracle (one look each).
See EXPERIMENT_LOG G6.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "data", "gastruloid_raj", "raw_ls2")
N = 30
P1_MEAN = {1: -0.922, 3: 0.881, 5: 1.051, 6: -1.132}  # G1a fit-shard passage-1 means
CHANNEL = {1: "Channel 0", 6: "Channel 0", 3: "Channel 2", 5: "Channel 2"}
P4_FILES = {
    "clone1+bulk_3+bulk_5+bulk_pass4_12042024.xlsx": {"1+WT": 1, "3+WT": 3, "5+WT": 5},
    "clone1+bulk_3+bulk_5+bulk_6+bulk_pass4_01272025.xlsx": {0.01: 1, 0.03: 3, 0.05: 5, 0.06: 6},
}
PRED = [0.13, 0.31]


def resample(s):
    v = np.array([float(x) for x in str(s).split(",")])
    return v[np.round(np.linspace(0, len(v) - 1, N)).astype(int)]


def pol(v):
    sd = v.std()
    if sd == 0:
        return np.nan
    z = savgol_filter((v - v.mean()) / sd, 5, 2)
    return z[: N // 2].mean() - z[N // 2:].mean()


def enrich(v):
    return (v[: N // 2].mean() - v[N // 2:].mean()) / v.mean()


def _boot_ci(groups, stat, B=2000, seed=69):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        vals.append(stat({k: rng.choice(v, len(v)) for k, v in groups.items()}))
    return np.percentile(vals, [2.5, 97.5]).tolist()


def p4_memory():
    groups = {}
    for f, tags in P4_FILES.items():
        d = pd.read_excel(os.path.join(RAW, f))
        for tag, c in tags.items():
            rows = d[(d["Tags"] == tag) & (d["Header"] == CHANNEL[c])]["Values"]
            p = np.array([pol(resample(s)) for s in rows])
            p = p[np.isfinite(p)]
            groups[(f[-13:-5], c)] = p * np.sign(P1_MEAN[c])

    def per_clone(g):
        out = {}
        for c in P1_MEAN:
            xs = [v for (f, cc), v in g.items() if cc == c]
            if xs:
                out[c] = np.concatenate(xs).mean() / abs(P1_MEAN[c])
        return out

    def pooled(g):
        return float(np.mean(list(per_clone(g).values())))
    est = pooled(groups)
    ci = _boot_ci(groups, pooled)
    pc = per_clone(groups)
    pci = {c: _boot_ci({k: v for k, v in groups.items() if k[1] == c},
                       lambda g, c=c: per_clone(g)[c]) for c in pc}
    reading = ("confirmed" if ci[1] >= PRED[0] and ci[0] <= PRED[1]
               else "fades slower than G3" if ci[0] > PRED[1] else "fades faster than G3")
    return {"n": {f"{k[0]}_clone{k[1]}": int(len(v)) for k, v in groups.items()},
            "pooled_ratio": [est, *ci], "per_clone_ratio": {f"clone{c}": [float(pc[c]), *pci[c]] for c in pc},
            "prediction": PRED, "reading": reading}


def jan24_pairs():
    def S_of(f, tag):
        d = pd.read_excel(os.path.join(RAW, f))
        out = []
        for _, g in d[d["Tags"] == tag].groupby("Annotation ID"):
            r = g[g["Header"] == "Channel 2"]["Values"]
            gr = g[g["Header"] == "Channel 0"]["Values"]
            if len(r) == 1 and len(gr) == 1:
                out.append(enrich(resample(r.iloc[0])) - enrich(resample(gr.iloc[0])))
        return np.array(out)
    s1 = S_of("clone1+3_pass1_01172024.xlsx", "1+3_linescan")
    s4 = S_of("clone1+3_clone1+6_pass4_01272024.xlsx", "1+3")
    share1 = float((s1 > 0).mean())
    ratio = float(s4.mean() / s1.mean())
    ci = _boot_ci({"a": s1, "b": s4}, lambda g: g["b"].mean() / g["a"].mean())
    rr = ("confirmed" if ci[1] >= PRED[0] and ci[0] <= PRED[1]
          else "fades slower than G3" if ci[0] > PRED[1] else "fades faster than G3")
    return {"n_p1": int(len(s1)), "n_p4": int(len(s4)), "S_mean_p1": float(s1.mean()), "S_mean_p4": float(s4.mean()),
            "share_clone3_anterior_p1": share1, "share_clone3_anterior_p4": float((s4 > 0).mean()),
            "propensity_reading": "replicated (>= 0.80)" if share1 >= 0.80 else "not replicated",
            "p4_over_p1": [ratio, *ci], "prediction": PRED, "memory_reading": rr}
