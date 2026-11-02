"""Tracks B3f (size-matched Static vs Dynamic fate prediction) and B3g (do sisters share
cell-state methylation within a cell type?). See the B3f / B3g pre-registration.

    uv run python experiments/track_b3fg.py
"""

from __future__ import annotations

import csv
import json
import os
import warnings

import numpy as np
import rdata
from track_b5_camellia import auc
from track_b_epiclone import dgc, top_pcs
from track_b_epiclone_m13 import ERY, MULTI, MYEL
from track_b_state import cv_predict

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "epiclone")


def load():
    r = rdata.read_rds(os.path.join(DATA, "M1-M3_seurat.rds"), default_encoding="utf8")
    md = vars(r)["meta.data"]
    keep = md.Experiment.astype(str).str.startswith("LARRY").to_numpy()
    counts = vars(r.assays["DNAm"])["counts"]
    names = np.array([str(x) for x in counts.Dimnames[0]])
    meth = (dgc(counts).T.toarray() > 0).astype(float)[keep]
    path = os.path.join(DATA, "info", "cpg_selection.csv")
    kind = {row["CpG"]: row["Type"] for row in csv.DictReader(open(path))}
    return meth, np.array([kind.get(n, "NA") for n in names]), md[keep]


def b3f(meth, types, md, rng, draws=200):
    clone = md.LARRY.fillna("").to_numpy(dtype=str)
    ctype = md.CellType.astype(str).to_numpy()
    multi = np.isin(ctype, list(MULTI))
    rows = []
    for c in np.unique(clone[clone != ""]):
        cells = clone == c
        my = (cells & np.isin(ctype, list(MYEL))).sum()
        er = (cells & np.isin(ctype, list(ERY))).sum()
        if (cells & multi).sum() >= 1 and my + er >= 5:
            rows.append((c, my / (my + er)))
    y = np.array([t[1] for t in rows])
    folds = np.random.default_rng(0).integers(0, 5, len(rows))

    def r_of(cols):
        pcs = top_pcs(meth[:, cols], 20)
        state = np.stack([pcs[(clone == c) & multi].mean(0) for c, _ in rows])
        return float(np.corrcoef(cv_predict(state, y, folds, "ridge", lam=10.0), y)[0, 1])

    static = r_of(np.flatnonzero(types == "Static"))
    dyn = np.flatnonzero(types == "Dynamic")
    n = int((types == "Static").sum())
    rs = np.array([r_of(rng.choice(dyn, n, replace=False)) for _ in range(draws)])
    return {
        "clones": len(rows),
        "static_r": static,
        "dynamic_110": {
            "median": float(np.median(rs)),
            "p5": float(np.percentile(rs, 5)),
            "p95": float(np.percentile(rs, 95)),
        },
        "static_percentile": float((rs < static).mean() * 100),
    }


def pair_auc(f, clone, group, boot, perms, rng):
    c = np.corrcoef(f) if f.ndim == 2 else -np.abs(f[:, None] - f[None, :])
    i, j = np.triu_indices(len(clone), 1)
    keep = group[i] == group[j]
    i, j = i[keep], j[keep]
    s, sis = c[i, j], clone[i] == clone[j]
    out = {"pairs": int(len(s)), "sister_pairs": int(sis.sum()), "auc": float(auc(s[sis], s[~sis]))}
    nulls = []
    for _ in range(perms):
        perm = clone.copy()
        for g in np.unique(group):
            k = group == g
            perm[k] = rng.permutation(perm[k])
        ps = perm[i] == perm[j]
        nulls.append(auc(s[ps], s[~ps]))
    out["null"] = float(np.mean(nulls))
    ids = np.unique(clone)
    pos = {k: np.flatnonzero(sis & (clone[i] == k)) for k in ids}
    neg = np.flatnonzero(~sis)
    bs = []
    for _ in range(boot):
        pick = rng.choice(ids, len(ids))
        p = np.concatenate([pos[k] for k in pick])
        if len(p):
            bs.append(auc(s[p], s[rng.choice(neg, min(len(neg), 200_000))]))
    if bs:
        out["ci"] = np.percentile(bs, [2.5, 97.5]).tolist()
    return out


def b3g(meth, types, md, rng):
    clone = md.LARRY.fillna("").to_numpy(dtype=str)
    ctype = md.CellType.astype(str).to_numpy()
    k = np.isin(ctype, list(MULTI)) & (clone != "")
    group = (
        (md.Experiment.astype(str) + "|" + md.ProcessingBatch.astype(str)).to_numpy() + "|" + ctype
    )
    clone, group, m = clone[k], group[k], meth[k]
    pt = md.Pseudotime.to_numpy(dtype=float)[k]
    # keep groups' cells only from clones with >= 2 cells in the group
    key = np.char.add(np.char.add(group.astype(str), "#"), clone)
    u, cnt = np.unique(key, return_counts=True)
    ok = np.isin(key, u[cnt >= 2]) | (
        rng.random(len(key)) < 0.3
    )  # all sisters + a non-sister sample
    clone, group, m, pt = clone[ok], group[ok], m[ok], pt[ok]
    dyn = np.flatnonzero(types == "Dynamic")
    n = int((types == "Static").sum())
    out = {"cells": int(len(clone)), "clones_with_sisters": int((cnt >= 2).sum())}
    out["dynamic"] = pair_auc(top_pcs(m[:, dyn], 20), clone, group, 300, 100, rng)
    out["static"] = pair_auc(top_pcs(m[:, types == "Static"], 20), clone, group, 300, 100, rng)
    d110 = [
        pair_auc(top_pcs(m[:, rng.choice(dyn, n, replace=False)], 20), clone, group, 0, 20, rng)
        for _ in range(20)
    ]
    out["dynamic_110"] = {
        "median_auc": float(np.median([d["auc"] for d in d110])),
        "null": float(np.mean([d["null"] for d in d110])),
        "range": [float(min(d["auc"] for d in d110)), float(max(d["auc"] for d in d110))],
    }
    good = ~np.isnan(pt)
    out["pseudotime"] = pair_auc(pt[good], clone[good], group[good], 300, 100, rng)
    return out


def main():
    warnings.filterwarnings("ignore")
    meth, types, md = load()
    rng = np.random.default_rng(0)
    out = {"b3f": b3f(meth, types, md, rng)}
    print(json.dumps(out, indent=1), flush=True)
    out["b3g"] = b3g(meth, types, md, rng)
    json.dump(out, open(os.path.join(HERE, "results_track_b", "b3fg.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
