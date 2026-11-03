"""B13: cytokine receptors and fate regulators at day 2 vs clone presence and fate.
See the B13 pre-registration in EXPERIMENT_LOG.md. Downloads the cytokine-experiment counts
(342 MB) if missing; meant for GitHub Actions.

    python experiments/track_b13.py
"""

from __future__ import annotations

import gzip
import json
import os
import urllib.request

import numpy as np
import pandas as pd
from e25_clone_types import groups
from fit_cytokine import CONDS5, load_counts, pooled
from scipy.stats import mannwhitneyu, pearsonr, rankdata, spearmanr
from track_b8_b9 import CYTO, HERE
from track_b11 import day2_features, partial_spearman

BASE = "https://kleintools.hms.harvard.edu/paper_websites/state_fate2020/"
PRIMARY = ["Csf3r", "Csf1r"]
PANEL = (
    "Csf2rb Il3ra Kit Stat3 Cebpa Cebpb Cebpe Gfi1 Spi1 Irf8 Klf4 Mafb Elane Mpo Ly86 Ccr2"
).split()


def fetch(name):
    path = os.path.join(CYTO, name)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        urllib.request.urlretrieve(BASE + "stateFate_cytokinePerturbation_" + name, path)
    return path


def day2_expression(genes):
    """{clone: {gene: mean log1p(normalised expression) over the clone's day-2 cells}}."""
    names = [g.strip() for g in gzip.open(fetch("gene_names.txt.gz"), "rt")]
    col = {g: names.index(g) + 1 for g in genes if g in names}  # MatrixMarket is 1-based
    rows = [
        line.rstrip("\n").split("\t")
        for line in gzip.open(os.path.join(CYTO, "metadata.txt.gz"), "rt")
    ][1:]
    d2 = {i + 1 for i, r in enumerate(rows) if r[2] == "2"}
    path = fetch("normed_counts.mtx.gz")
    with gzip.open(path, "rt") as f:
        header = f.readline()
        while header.startswith("%"):
            header = f.readline()
        n_cells, n_genes, _ = (int(v) for v in header.split())
        assert n_cells == len(rows) and n_genes == len(names), (
            n_cells,
            len(rows),
            n_genes,
            len(names),
        )
        want = set(col.values())
        vals = {}
        for chunk in pd.read_csv(
            f, sep=" ", header=None, names=["c", "g", "v"], chunksize=5_000_000
        ):
            chunk = chunk[chunk.g.isin(want) & chunk.c.isin(d2)]
            for c, g, v in chunk.itertuples(index=False):
                vals[(c, g)] = v
    m = np.loadtxt(
        gzip.open(os.path.join(CYTO, "clone_matrix.mtx.gz"), "rt"),
        comments="%",
        skiprows=3,
        dtype=int,
    )
    cells = {}
    for clone, cell in zip(m[:, 0] - 1, m[:, 1], strict=True):  # cell kept 1-based
        if cell in d2:
            cells.setdefault(int(clone), []).append(int(cell))
    out = {}
    for clone, cs in cells.items():
        out[clone] = {
            g: float(np.mean([np.log1p(vals.get((c, j), 0.0)) for c in cs])) for g, j in col.items()
        }
    return out, sorted(col)


def partial_multi(x, y, covs):
    """Partial Spearman given several covariates: Pearson of rank residuals (B11 v3 method)."""
    a = np.c_[np.ones(len(x)), *[rankdata(c) for c in covs]]

    def res(v):
        v = rankdata(v)
        return v - a @ np.linalg.lstsq(a, v, rcond=None)[0]

    return pearsonr(res(x), res(y))


def bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    q = p[order] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1)
    return out


def tests(expr, gene_fn):
    cnt = load_counts(CONDS5)
    pg, pm, both, g_only, m_only = groups(cnt)
    pmm = pooled(cnt["M"])
    res = {}
    # 1: G-only vs M-only (higher in G-only is the prediction)
    a = [gene_fn(expr[k]) for k in g_only if k in expr]
    b = [gene_fn(expr[k]) for k in m_only if k in expr]
    res["t1_Gonly_vs_Monly"] = {
        "n": [len(a), len(b)],
        "median": [float(np.median(a)), float(np.median(b))],
        "p_one_sided": float(mannwhitneyu(a, b, alternative="greater").pvalue),
    }
    # 2: N under M-CSF vs none, partial on log size
    keys = [k for k, v in pmm.items() if v[0] + v[1] >= 2 and k in expr]
    has = np.array([pmm[k][0] >= 1 for k in keys], float)
    size = np.log([pmm[k][0] + pmm[k][1] for k in keys])
    x = np.array([gene_fn(expr[k]) for k in keys])
    r, p = partial_spearman(x, has, size)
    res["t2_N_under_M"] = {
        "n": len(keys),
        "partial_rho": float(r),
        "p_one_sided": float(p / 2 if r > 0 else 1 - p / 2),
    }
    # 3: G-present clones, gene vs G-CSF N-share
    gk = [k for k, v in pg.items() if v[0] + v[1] >= 2 and k in expr]
    share = [pg[k][0] / (pg[k][0] + pg[k][1]) for k in gk]
    r, p = spearmanr([gene_fn(expr[k]) for k in gk], share)
    res["t3_G_Nshare"] = {
        "n": len(gk),
        "rho": float(r),
        "p_one_sided": float(p / 2 if r > 0 else 1 - p / 2),
    }
    return res


def main():
    expr, found = day2_expression(PRIMARY + PANEL)
    out = {"genes_found": found, "clones_with_day2": len(expr)}
    have = set(found)
    if {"Csf3r", "Csf1r"} <= have:
        out["primary"] = {
            "Csf3r": tests(expr, lambda e: e["Csf3r"]),
            "Csf1r": tests(expr, lambda e: e["Csf1r"]),
            "balance": tests(expr, lambda e: e["Csf3r"] - e["Csf1r"]),
        }
        pr = out["primary"]
        out["predictions"] = {
            "P1_balance_Gonly_gt_Monly": pr["balance"]["t1_Gonly_vs_Monly"]["p_one_sided"] < 0.05,
            "P2_Csf3r_N_under_M": pr["Csf3r"]["t2_N_under_M"]["p_one_sided"] < 0.05,
            "P3_balance_G_Nshare": pr["balance"]["t3_G_Nshare"]["p_one_sided"] < 0.05,
        }
    panel = {g: tests(expr, lambda e, g=g: e[g]) for g in found}
    for t in ("t1_Gonly_vs_Monly", "t2_N_under_M", "t3_G_Nshare"):
        # two-sided p for the panel screen (direction not predicted for every gene)
        p2 = [
            min(1.0, 2 * min(panel[g][t]["p_one_sided"], 1 - panel[g][t]["p_one_sided"]))
            for g in found
        ]
        for g, q in zip(found, bh(p2), strict=True):
            panel[g][t]["p_two_sided"] = float(p2[found.index(g)])
            panel[g][t]["bh_q"] = float(q)
    out["panel"] = panel
    # does the B11 priming effect survive the receptor balance?
    if {"Csf3r", "Csf1r"} <= have:
        pmm = pooled(load_counts(CONDS5)["M"])
        feats = day2_features()
        keys = [k for k, v in pmm.items() if v[0] + v[1] >= 2 and k in expr and k in feats]
        has = np.array([pmm[k][0] >= 1 for k in keys], float)
        size = np.log([pmm[k][0] + pmm[k][1] for k in keys])
        bal = np.array([expr[k]["Csf3r"] - expr[k]["Csf1r"] for k in keys])
        score = np.array([feats[k][0] for k in keys])
        r0 = partial_spearman(score, has, size)
        r1 = partial_multi(score, has, [size, bal])
        out["b11_score_given_balance"] = {
            "n": len(keys),
            "rho_size_only": float(r0[0]),
            "rho_size_and_balance": float(r1[0]),
            "p": float(r1[1]),
        }
    json.dump(out, open(os.path.join(HERE, "results_e18", "b13.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in out if k != "panel"}, indent=1))


if __name__ == "__main__":
    main()
