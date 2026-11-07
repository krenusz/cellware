"""G9/G11 sealed drug partitions. Read ONLY through test_oracle (one look per partition).
See EXPERIMENT_LOG G11. Selection: selection/G11.json (train fit, committed before the look).
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from g7_model import pol, resample
from g9_mech import lines

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "data", "gastruloid_raj", "raw_ls2")
SEL = os.path.join(HERE, "selection", "G11.json")
FILES = {"A": ("1+3_ctrl_ra_sb_05032025.csv", {"clone1+3": "ctrl", "clone1+3_RA": "RA", "clone1+3_SB": "SB"})}


def _D(part):
    f, tags = FILES[part]
    d = pd.read_csv(os.path.join(RAW, f))
    rows = []
    for aid, tag, h in lines(d[d["Tags"].isin(tags)]):
        if "Channel 2" in h and "Channel 0" in h:
            D = pol(resample(h["Channel 2"])) - pol(resample(h["Channel 0"]))
            if np.isfinite(D):
                rows.append({"cond": tags[tag], "id": aid, "D": D})
    return pd.DataFrame(rows)


def _boot_mean(g, rng):
    ids = g["id"].unique()
    pick = rng.choice(ids, len(ids))
    return float(np.concatenate([g[g["id"] == i]["D"].to_numpy() for i in pick]).mean())


def g11_drug(part):
    if part not in FILES:
        raise SystemExit(f"G11 is pre-registered for partition A only, not {part}")
    sel = json.load(open(SEL))
    g = sel["params"]["all"] if sel["chosen"] == "R2" else None
    df = _D(part)
    rng = np.random.default_rng(1111)
    by = {c: df[df["cond"] == c] for c in ("ctrl", "RA", "SB")}
    se = {c: max(float(x["D"].std() / np.sqrt(len(x))), 0.05) for c, x in by.items()}
    m = {c: float(x["D"].mean()) for c, x in by.items()}
    idm = {c: float(x.groupby("id")["D"].mean().mean()) for c, x in by.items()}
    out = {"chosen_model": sel["chosen"], "gain": g,
           "n_ids": {c: int(x["id"].nunique()) for c, x in by.items()}, "n_lines": {c: int(len(x)) for c, x in by.items()},
           "D_mean": {c: round(m[c], 3) for c in m}, "D_se": {c: round(se[c], 3) for c in se}}
    boots = {c: np.array([_boot_mean(x, rng) for _ in range(2000)]) for c, x in by.items()}
    passes = 0
    for drug in ("RA", "SB"):
        r = m[drug] / m["ctrl"]
        rb = boots[drug] / boots["ctrl"]
        lo, hi = np.percentile(rb, [2.5, 97.5])
        pred = g * m["ctrl"]
        z = (m[drug] - pred) / np.sqrt(se[drug] ** 2 + g ** 2 * se["ctrl"] ** 2)
        ok = bool(hi < 1)
        passes += ok
        out[drug] = {"ratio": round(r, 3), "ratio_ci95": [round(lo, 3), round(hi, 3)], "directional_pass": ok,
                     "predicted_D": round(pred, 3), "z": round(float(z), 2), "quantitative_pass": bool(abs(z) <= 2.5),
                     "ratio_per_id": round(idm[drug] / idm["ctrl"], 3)}
    out["primary_reading"] = {2: "pass", 1: "partial", 0: "fail"}[passes]
    out["underpowered"] = any(v < 4 for v in out["n_ids"].values())
    return out
