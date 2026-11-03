"""B14: committed-like day-2 cells per clone (proNeu-like / monocyte-committed) vs outcomes.
See the B14 pre-registration in EXPERIMENT_LOG.md. Uses the cytokine-experiment counts (B13).

    python experiments/track_b14.py
"""

from __future__ import annotations

import gzip
import json
import os

import numpy as np
import pandas as pd
from e25_clone_types import groups
from fit_cytokine import CONDS5, load_counts
from scipy.stats import mannwhitneyu, spearmanr
from track_b8_b9 import CYTO, HERE
from track_b13 import fetch, partial_multi

NEU = ["Gfi1", "Cebpe", "Elane", "Mpo"]
MONO = ["Irf8", "Ly86", "Ccr2", "Csf1r"]
EXTRA = ["Cebpa"]


def day2_cells():
    """Per day-2 cell: annotation, clone, log1p expression of the gene set."""
    genes = NEU + MONO + EXTRA
    names = [g.strip() for g in gzip.open(fetch("gene_names.txt.gz"), "rt")]
    col = {names.index(g) + 1: g for g in genes}
    rows = [
        line.rstrip("\n").split("\t")
        for line in gzip.open(os.path.join(CYTO, "metadata.txt.gz"), "rt")
    ][1:]
    d2 = [i + 1 for i, r in enumerate(rows) if r[2] == "2"]
    d2set = set(d2)
    x = pd.DataFrame(0.0, index=d2, columns=genes)
    with gzip.open(fetch("normed_counts.mtx.gz"), "rt") as f:
        line = f.readline()
        while line.startswith("%"):
            line = f.readline()
        for chunk in pd.read_csv(
            f, sep=" ", header=None, names=["c", "g", "v"], chunksize=5_000_000
        ):
            chunk = chunk[chunk.g.isin(col.keys()) & chunk.c.isin(d2set)]
            for c, g, v in chunk.itertuples(index=False):
                x.at[c, col[g]] = np.log1p(v)
    m = np.loadtxt(
        gzip.open(os.path.join(CYTO, "clone_matrix.mtx.gz"), "rt"),
        comments="%",
        skiprows=3,
        dtype=int,
    )
    clone = {}
    for cl, cell in zip(m[:, 0] - 1, m[:, 1], strict=True):
        if cell in d2set:
            clone[int(cell)] = int(cl)
    x["ann"] = [rows[c - 1][3] for c in d2]
    x["clone"] = [clone.get(c, -1) for c in d2]
    return x


def classify(x):
    z = (x[NEU + MONO] - x[NEU + MONO].mean()) / x[NEU + MONO].std().replace(0, 1)
    neu, mono = z[NEU].mean(1), z[MONO].mean(1)
    und = x.ann == "Undifferentiated"
    tn, tm = np.percentile(neu[und], 90), np.percentile(mono[und], 90)
    x["N_comm"] = (x.ann == "Neu") | (und & (neu >= tn) & (x.Csf1r == 0))
    x["M_comm"] = (x.ann == "Mo") | (und & (mono >= tm) & (x.Gfi1 == 0))
    return x, {"thr_neu": float(tn), "thr_mono": float(tm)}


def boot_drop(a, b, covs, rng, n=2000):
    """Bootstrap the drop from raw Spearman(a, b) to partial given covs."""
    out = []
    for _ in range(n):
        i = rng.choice(len(a), len(a))
        try:
            raw = spearmanr(a[i], b[i])[0]
            par = partial_multi(a[i], b[i], [c[i] for c in covs])[0]
            out.append(raw - par)
        except ValueError:
            pass
    out = np.array(out)
    out = out[np.isfinite(out)]
    return np.percentile(out, [2.5, 97.5]).tolist() if len(out) else [np.nan, np.nan]


def main():
    x, thr = classify(day2_cells())
    per = (
        x[x.clone >= 0]
        .groupby("clone")
        .agg(
            frac_N=("N_comm", "mean"),
            frac_M=("M_comm", "mean"),
            cebpa=("Cebpa", "mean"),
            irf8=("Irf8", "mean"),
            n2=("ann", "size"),
        )
    )
    cnt = load_counts(CONDS5)
    pg, pm, both, g_only, m_only = groups(cnt)
    rep = {
        "thresholds": thr,
        "clones_with_day2": len(per),
        "cells_N_comm": int(x.N_comm.sum()),
        "cells_M_comm": int(x.M_comm.sum()),
    }
    # 1: N under M-CSF
    keys = [k for k, v in pm.items() if v[0] + v[1] >= 2 and k in per.index]
    has = np.array([pm[k][0] >= 1 for k in keys], float)
    size = np.log([pm[k][0] + pm[k][1] for k in keys])
    fn = per.loc[keys, "frac_N"].to_numpy()
    ce = per.loc[keys, "cebpa"].to_numpy()
    t1 = {"n": len(keys)}
    for name, v, covs in (
        ("frac_N|size", fn, [size]),
        ("cebpa|size", ce, [size]),
        ("frac_N|size,cebpa", fn, [size, ce]),
        ("cebpa|size,frac_N", ce, [size, fn]),
    ):
        r, p = partial_multi(v, has, covs)
        t1[name] = {"rho": float(r), "p": float(p)}
    rep["t1_N_under_M"] = t1
    # 2: presence groups
    t2 = {}
    for col_, a, b in (("frac_N", g_only, m_only), ("frac_M", m_only, g_only)):
        va = per.loc[[k for k in a if k in per.index], col_]
        vb = per.loc[[k for k in b if k in per.index], col_]
        t2[col_] = {
            "n": [len(va), len(vb)],
            "mean": [float(va.mean()), float(vb.mean())],
            "p_one_sided": float(mannwhitneyu(va, vb, alternative="greater").pvalue),
        }
    rep["t2_presence"] = t2
    # 3: G-M agreement among both-clones with day-2 cells
    bk = [k for k in both if k in per.index]
    sg = np.array([pg[k][0] / (pg[k][0] + pg[k][1]) for k in bk])
    sm = np.array([pm[k][0] / (pm[k][0] + pm[k][1]) for k in bk])
    f_n, f_m = per.loc[bk, "frac_N"].to_numpy(), per.loc[bk, "frac_M"].to_numpy()
    c_a, c_i = per.loc[bk, "cebpa"].to_numpy(), per.loc[bk, "irf8"].to_numpy()
    rng = np.random.default_rng(0)
    raw = spearmanr(sg, sm)[0]
    pc = partial_multi(sg, sm, [f_n, f_m])[0]
    pg_ = partial_multi(sg, sm, [c_a, c_i])[0]
    rep["t3_GM"] = {
        "n": len(bk),
        "raw_spearman": float(raw),
        "partial_committed": float(pc),
        "partial_graded": float(pg_),
        "drop_committed": float(raw - pc),
        "drop_committed_ci": boot_drop(sg, sm, [f_n, f_m], rng),
        "drop_graded": float(raw - pg_),
        "drop_graded_ci": boot_drop(sg, sm, [c_a, c_i], rng),
        "clones_with_any_committed": int(((f_n > 0) | (f_m > 0)).sum()),
    }
    t = rep["t3_GM"]
    rel = lambda d: d / raw if raw else np.nan  # noqa: E731
    rep["reading"] = (
        "committed"
        if rel(t["drop_committed"]) >= 0.3
        and t["drop_committed"] > t["drop_graded"]
        and t["drop_committed_ci"][0] > 0
        else "graded"
        if rel(t["drop_graded"]) >= 0.3
        and t["drop_graded"] > t["drop_committed"]
        and t["drop_graded_ci"][0] > 0
        else "neither"
    )
    per.to_csv(os.path.join(HERE, "results_e18", "b14_clone_day2.csv"))
    json.dump(rep, open(os.path.join(HERE, "results_e18", "b14.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
