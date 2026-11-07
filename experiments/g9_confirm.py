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


# ---------- structure counts (H9/H14: units per tag only, never outcomes) ----------

DRUG_FILES = {"A": "1+3_ctrl_ra_sb_05032025.csv",
              "B": os.path.join("..", "raw_linescans", "07282025", "linescan_drug_07282025.csv")}


def tag_counts(part):
    """Per tag: gastruloid IDs, lines, and which channel headers exist. No values are summarised."""
    d = pd.read_csv(os.path.join(RAW, DRUG_FILES[part]), usecols=["Annotation ID", "Tags", "Header"])
    out = {}
    for tag, g in d.groupby("Tags"):
        out[str(tag)] = {"ids": int(g["Annotation ID"].nunique()),
                         "headers": sorted(map(str, g["Header"].unique()))}
    return out


G12_TAGS = {"AP_control": "ctrl", "AP_ra": "drug", "AP_sb": "drug", "AP_agn": "drug"}


def g12_counts(part):
    """Units per pre-registered group (counts only)."""
    if part != "B":
        raise SystemExit("G12 is pre-registered for partition B only")
    c = tag_counts(part)
    return {"ctrl": c["AP_control"]["ids"], "drug": sum(c[t]["ids"] for t, g in G12_TAGS.items() if g == "drug")}


def g12_drug(part):
    if part != "B":
        raise SystemExit("G12 is pre-registered for partition B only")
    d = pd.read_csv(os.path.join(RAW, DRUG_FILES[part]))
    rows = []
    for aid, tag, h in lines(d[d["Tags"].isin(G12_TAGS)]):
        if "Channel 2" in h and "Channel 0" in h:
            D = pol(resample(h["Channel 2"])) - pol(resample(h["Channel 0"]))
            if np.isfinite(D):
                rows.append({"grp": G12_TAGS[tag], "id": aid, "D": D})
    x = pd.DataFrame(rows).groupby(["grp", "id"])["D"].mean().reset_index()
    c, dr = x[x["grp"] == "ctrl"]["D"].to_numpy(), x[x["grp"] == "drug"]["D"].to_numpy()
    rng = np.random.default_rng(1212)
    bd = np.array([rng.choice(c, len(c)).mean() - rng.choice(dr, len(dr)).mean() for _ in range(4000)])
    lo = float(np.percentile(bd, 5))
    g = 0.348
    se_c, se_d = c.std(ddof=1) / np.sqrt(len(c)), dr.std(ddof=1) / np.sqrt(len(dr))
    z = (dr.mean() - g * c.mean()) / np.sqrt(se_d ** 2 + g ** 2 * se_c ** 2)
    return {"n": {"ctrl": int(len(c)), "drug": int(len(dr))}, "D_ctrl": round(float(c.mean()), 3),
            "D_drug": round(float(dr.mean()), 3), "delta": round(float(c.mean() - dr.mean()), 3),
            "delta_lower95_one_sided": round(lo, 3), "primary": "pass" if lo > 0 else "fail",
            "ratio": round(float(dr.mean() / c.mean()), 3), "z_vs_R2": round(float(z), 2)}
