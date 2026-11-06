"""G1: gastruloid clone A-P propensity (sealed, G1a) and rank sorting vs fixed domains (G1b).
See EXPERIMENT_LOG G1 and its amendment.

    uv run python g1_gastruloid.py fit       # G1a fit-shard variance shares + held-out intervals
    uv run python g1_gastruloid.py mixing    # G1b crossing point vs anterior-clone fraction

Held-out gastruloids are read only in `heldout_stat`, called by test_oracle.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from test_oracle import shard_of

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "gastruloid_raj", "linescans")
OUT = os.path.join(HERE, "results_gastruloid")
os.makedirs(OUT, exist_ok=True)


# ---------- G1a ----------

def polarity(include_heldout=False):
    """Per-gastruloid polarity. Held-out (G1a sealed) gastruloids are dropped unless
    include_heldout=True, which only heldout_stat (via test_oracle) may pass."""
    d = pd.read_csv(os.path.join(DATA, "normalized_smoothed_individual_lines.csv"))
    x = d["position"] * 100 / d["position"].max()
    d["half"] = np.where(x < 50, "a", "p")
    m = d.groupby(["clone", "replicate", "gastruloid_id", "half"])["normalized_intensity"].mean().unstack()
    P = (m["a"] - m["p"]).rename("P").reset_index()
    P["heldout"] = [shard_of(f"g1:{g}", 3) == 0 for g in P["gastruloid_id"]]
    return P if include_heldout else P[~P["heldout"]].copy()


def components(P):
    """Method-of-moments shares (clone, replicate within clone, residual) for an unbalanced
    nested design."""
    cell = P.groupby(["clone", "replicate"])["P"].agg(["mean", "var", "count"]).reset_index()
    s_e2 = float(np.average(cell["var"].fillna(0), weights=np.maximum(cell["count"] - 1, 0)))
    nbar = float(cell["count"].mean())
    rep = []
    for _, g in cell.groupby("clone"):
        if len(g) > 1:
            rep.append(g["mean"].var())
    s_rep2 = max(float(np.mean(rep)) - s_e2 / nbar, 0.0) if rep else 0.0
    cm = cell.groupby("clone")["mean"].mean()
    rbar = float(cell.groupby("clone").size().mean())
    s_cl2 = max(float(cm.var()) - s_rep2 / rbar - s_e2 / (rbar * nbar), 0.0)
    tot = s_cl2 + s_rep2 + s_e2
    return {"clone": s_cl2 / tot, "replicate": s_rep2 / tot, "residual": s_e2 / tot,
            "s_clone": s_cl2 ** 0.5, "s_rep": s_rep2 ** 0.5, "s_e": s_e2 ** 0.5}


def fit(boot=2000, seed=61):
    P = polarity(include_heldout=True)  # held-out rows used for counts only (interval widths)
    F, H = P[~P["heldout"]], P[P["heldout"]]
    est = components(F)
    rng = np.random.default_rng(seed)
    design = F[["clone", "replicate"]].reset_index(drop=True)
    clones = design["clone"].unique()
    reps = design.drop_duplicates()
    bs = []
    for _ in range(boot):
        mu = dict(zip(clones, rng.normal(0, est["s_clone"], len(clones))))
        u = {(c, r): rng.normal(0, est["s_rep"]) for c, r in reps.itertuples(index=False)}
        sim = design.copy()
        sim["P"] = [mu[c] + u[(c, r)] + rng.normal(0, est["s_e"]) for c, r in design.itertuples(index=False)]
        bs.append(components(sim))
    ci = {k: np.percentile([b[k] for b in bs], [2.5, 97.5]).tolist() for k in ("clone", "replicate", "residual")}
    cell = F.groupby(["clone", "replicate"])["P"].agg(["mean", "count"])
    nh = H.groupby(["clone", "replicate"]).size()
    pred = []
    for (c, r), row in cell.iterrows():
        n_ho = int(nh.get((c, r), 0))
        half = 1.96 * est["s_e"] * np.sqrt(1 / row["count"] + 1 / max(n_ho, 1))
        pred.append({"clone": c, "replicate": r, "mean_fit": float(row["mean"]), "n_fit": int(row["count"]),
                     "n_heldout_expected": n_ho, "lo": float(row["mean"] - half), "hi": float(row["mean"] + half)})
    clone_means = F.groupby("clone")["P"].agg(["mean", "count"]).reset_index().to_dict("records")
    res = {"n_fit": int(len(F)), "n_heldout": int(len(H)), "shares": est,
           "shares_ci_parametric_boot": ci, "clone_means_fit": clone_means, "predictions": pred}
    json.dump(res, open(os.path.join(OUT, "g1a_fit.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


def heldout_stat():
    """Called by test_oracle only."""
    P = polarity(include_heldout=True)
    H = P[P["heldout"]]
    pred = json.load(open(os.path.join(OUT, "g1a_fit.json")))["predictions"]
    cells, inside = [], []
    for p in pred:
        h = H[(H["clone"] == p["clone"]) & (H["replicate"] == p["replicate"])]["P"]
        m = float(h.mean()) if len(h) else None
        ok = None if m is None else bool(p["lo"] <= m <= p["hi"])
        cells.append({"clone": p["clone"], "replicate": p["replicate"], "n": int(len(h)), "mean": m, "inside": ok})
        if len(h) >= 4:
            inside.append(ok)
    return {"n": int(len(H)), "cells": cells, "qualifying_cells": len(inside),
            "inside_fraction": float(np.mean(inside)) if inside else None,
            "heldout_shares": components(H)}


# ---------- G1b ----------

def _crossing(x, red, green):
    d = red - green
    for k in range(len(d) - 1):
        if d[k] == 0:
            return x[k]
        if d[k] * d[k + 1] < 0:
            return x[k] + (x[k + 1] - x[k]) * d[k] / (d[k] - d[k + 1])
    return np.nan


def mixing(draws=2000, seed=62):
    rows = []
    for f in sorted(glob.glob(os.path.join(DATA, "ratios_*_clone1clone3_*.csv"))):
        b = os.path.basename(f).split("_")
        a1, a3, date = int(b[1]), int(b[2]), b[4].split(".")[0]
        rows.append((f, a3 / (a1 + a3), date, a1, a3))
    rng = np.random.default_rng(seed)
    obs, sims = [], []
    for f, frac, date, a1, a3 in rows:
        d = pd.read_csv(f)
        x = d["x"].to_numpy()
        xr = _crossing(x, d["mean_red"].to_numpy(), d["mean_green"].to_numpy())
        se_r = (d["upper_red"] - d["lower_red"]).abs().to_numpy() / 2
        se_g = (d["upper_green"] - d["lower_green"]).abs().to_numpy() / 2
        s = [_crossing(x, d["mean_red"].to_numpy() + rng.normal(0, 1, len(x)) * se_r,
                       d["mean_green"].to_numpy() + rng.normal(0, 1, len(x)) * se_g) for _ in range(draws)]
        obs.append({"file": os.path.basename(f), "clone1:clone3": f"{a1}:{a3}", "f_anterior": frac,
                    "date": date, "crossing": float(xr), "crossing_ci": np.nanpercentile(s, [2.5, 97.5]).tolist(),
                    "frac_draws_without_crossing": float(np.mean(np.isnan(s)))})
        sims.append(np.array(s))
    dates = sorted({o["date"] for o in obs})

    def slope(xs):
        ok = ~np.isnan(xs)
        X = np.c_[[100 * o["f_anterior"] for o in obs], [[o["date"] == dd for dd in dates] for o in obs]][ok].astype(float)
        return np.linalg.lstsq(X, xs[ok], rcond=None)[0][0] if ok.sum() >= 4 else np.nan

    b0 = slope(np.array([o["crossing"] for o in obs]))
    bs = np.array([slope(np.array([s[i] for s in sims])) for i in range(draws)])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    n_cross = int(np.sum(~np.isnan([o["crossing"] for o in obs])))
    inc0, inc1 = lo <= 0 <= hi, lo <= 1 <= hi
    reading = ("rank sorting" if inc1 and not inc0 else "fixed domains" if inc0 and not inc1
               else "partial proportion regulation" if (lo > 0 and hi < 1) else "unresolved")
    res = {"ratios": obs, "n_with_crossing": n_cross, "slope": float(b0), "slope_ci": [float(lo), float(hi)],
           "reading": reading if n_cross >= 4 else "descriptive (fewer than 4 crossings)"}
    json.dump(res, open(os.path.join(OUT, "g1b_mixing.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "mixing"])
    {"fit": fit, "mixing": mixing}[ap.parse_args().cmd]()
