"""C7-S2a: train-only diagnostic of what limits the C7-S2 heritable share H and its power.
See EXPERIMENT_LOG C7-S2a. Reads the celltag_lsk TRAIN partition only.

    python c7s2a_diag.py [--boot 300]
"""

from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import norm

import partitions
import sis_data as sd
from c7s2_sister import CSV, GROUPS, OUT

G7 = ["Neu", "Mono", "BEM", "EMk", "DC", "Lym", "Prog"]
COARSE = {"EMk": "Other", "BEM": "Other", "DC": "Other", "Lym": "Other"}
SCALE = 1348 / 2720  # partition A / train clone ratio (all clones)


def load():
    df = pd.read_csv(CSV, usecols=["clone.id", "cell.bc", "day", "assay", "cell_type2", "edge.den"])
    df = partitions.train_only("celltag_lsk", df, unit_col="clone.id")
    df = df[df["day"] == "d5"].copy()
    df["well"] = np.where(df["assay"] == "atac", "r1", df["cell.bc"].str.extract(r"^d5(r[12])-")[0])
    df["grp"] = df["cell_type2"].map(GROUPS)
    return df[df["grp"].notna()]


def table(df, grouping, min_cells, quality, atac, med):
    d = df if atac else df[df["assay"] == "rna"]
    if quality:
        d = d[d["edge.den"] >= med]
    if grouping == "G6":
        d, fates = d[d["grp"] != "Prog"], G7[:-1]
    elif grouping == "G4":
        d, fates = d.assign(grp=d["grp"].replace(COARSE)), ["Neu", "Mono", "Prog", "Other"]
    else:
        fates = G7
    t = d.groupby(["clone.id", "well", "grp"]).size().unstack("grp", fill_value=0).reindex(columns=fates, fill_value=0)
    t = t.unstack("well", fill_value=0)
    k1, k2 = t.xs("r1", axis=1, level="well").to_numpy(), t.xs("r2", axis=1, level="well").to_numpy()
    m = (k1.sum(1) >= min_cells) & (k2.sum(1) >= min_cells)
    return k1[m], k2[m], fates


def power(h, se, n):
    n_part = n * SCALE
    se_p = se * np.sqrt(n / n_part)
    return float(1 - norm.cdf(norm.ppf(0.975) - (h - 0.5) / se_p))


def run_h(k1, k2, rng, boot):
    n = len(k1)
    g = np.zeros(n, int)
    h = sd.heritable_share(k1, k2, g, rng, 500, 100)
    hb = np.array([sd.heritable_share(k1[ix], k2[ix], g[ix], rng, 60, 30)["H"]
                   for ix in (rng.integers(0, n, n) for _ in range(boot))])
    se = float(np.nanstd(hb))
    return {"n": int(n), "H": float(h["H"]), "D_sis": float(h["D_sis"]), "D_rand": float(h["D_rand"]),
            "D_noise": float(h["D_noise"]), "ci": np.nanpercentile(hb, [2.5, 97.5]).tolist(), "se": se,
            "power_partition": power(h["H"], se, n), "cells_per_well_median": float(np.median(np.r_[k1.sum(1), k2.sum(1)]))}


def larry_r(df, k, rng, boot):
    d = df[(df["assay"] == "rna") & df["grp"].isin(["Neu", "Mono"])]
    t = d.groupby(["clone.id", "well", "grp"]).size().unstack("grp", fill_value=0).unstack("well", fill_value=0)
    n1, m1 = t[("Neu", "r1")], t[("Mono", "r1")]
    n2, m2 = t[("Neu", "r2")], t[("Mono", "r2")]
    ok = ((n1 + m1) >= k) & ((n2 + m2) >= k)
    x, y = (n1 / (n1 + m1))[ok].to_numpy(), (n2 / (n2 + m2))[ok].to_numpy()
    r = float(np.corrcoef(x, y)[0, 1])
    rb = [np.corrcoef(x[ix], y[ix])[0, 1] for ix in (rng.integers(0, len(x), len(x)) for _ in range(boot))]
    return {"k": k, "n": int(ok.sum()), "r": r, "ci": np.nanpercentile(rb, [2.5, 97.5]).tolist()}


def main(boot):
    rng = np.random.default_rng(721)
    df = load()
    med = float(df.drop_duplicates("clone.id")["edge.den"].median())
    rows = []
    for mc, grp, q, at in itertools.product((2, 3, 4, 6, 8), ("G7", "G6", "G4"), (False, True), (False, True)):
        k1, k2, fates = table(df, grp, mc, q, at, med)
        if len(k1) < 20:
            rows.append({"min_cells": mc, "grouping": grp, "quality": q, "atac_well1": at, "n": int(len(k1))})
            continue
        r = run_h(k1, k2, rng, boot)
        rows.append({"min_cells": mc, "grouping": grp, "quality": q, "atac_well1": at, **r})
        print(mc, grp, q, at, r["n"], round(r["H"], 3), [round(c, 3) for c in r["ci"]], round(r["power_partition"], 2), flush=True)
    res = {"edge_den_median": med, "grid": rows, "larry_r": [larry_r(df, k, rng, boot) for k in (2, 4)]}
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "c7s2a_diag.json"), "w"), indent=1)
    print(json.dumps(res["larry_r"], indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=300)
    main(ap.parse_args().boot)
