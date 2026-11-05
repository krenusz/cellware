"""B41: does the sisters' RNA before the split predict DC fate? SIS-seq fit set.

    uv run python b41_rna_fate.py [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

import sis_data as sd

GENES = os.path.join(sd.ROOT, "geo", "gene_counts.csv.gz")
MART = os.path.join(sd.ROOT, "full", "x", "Fig1-3_scripts_and_data", "data", "clone_split",
                    "mart_export.txt")
PROGRAMMES = {"pDC": ["Tcf4", "Spib", "Siglech", "Runx2", "Ly6d", "Bst2"],
              "cDC1": ["Id2", "Batf3", "Nfil3", "Zbtb46"],
              "cDC2": ["Irf4", "Klf4", "Relb", "Notch2"]}


def within_batch_z(M, batch):
    Z = np.zeros_like(M)
    for b in np.unique(batch):
        m = batch == b
        sdv = M[m].std(0)
        Z[m] = np.where(sdv > 0, (M[m] - M[m].mean(0)) / np.where(sdv > 0, sdv, 1), 0.0)
    return Z


def centre(x, batch):
    x = x.copy()
    for b in np.unique(batch):
        x[batch == b] -= x[batch == b].mean(0)
    return x


def loo_r2(X, y):
    X1 = np.column_stack([np.ones(len(y)), X])
    pred = np.empty(len(y))
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        w = np.linalg.lstsq(X1[m], y[m], rcond=None)[0]
        pred[i] = X1[i] @ w
    return float(1 - np.mean((y - pred) ** 2) / np.var(y))


def main(out=None, boot=1000):
    rng = np.random.default_rng(41)
    d = sd.load_seq("fit")
    g = pd.read_csv(GENES, index_col=0)
    mart = pd.read_csv(MART).dropna(subset=["EntrezGene ID"])
    sym = dict(zip(mart["EntrezGene ID"].astype(int), mart["Associated Gene Name"]))
    lib = g.sum(0)
    import gzip
    meta = {}
    for line in gzip.open(sd.SEQ_SOFT, "rt", encoding="utf-8", errors="replace"):
        if line.startswith("!Sample_title"):
            t = line.split("= ", 1)[1].strip()
        elif line.startswith("!Sample_characteristics_ch1 = cells_sorted"):
            meta[t] = float(line.split(": ", 1)[1])
    ok = ((d["k1"].sum(1) > 3) & (d["k2"].sum(1) > 3)
          & np.array([t in g.columns and lib.get(t, 0) >= 5000 and meta.get(t, 0) >= 1 for t in d["title"]]))
    titles, batch = d["title"][ok], d["batch"][ok]
    k = d["k1"][ok] + d["k2"][ok]
    n = k.sum(1, keepdims=True)
    L = centre(np.log((k + 0.5) / (n - k + 0.5)), batch)
    C = g[titles].to_numpy(float).T
    E = np.log1p(C / C.sum(1, keepdims=True) * 1e4)
    det = (C > 0).mean(0) >= 0.10
    E, genes = E[:, det], g.index[det]
    Z = within_batch_z(E, batch)
    names = np.array([sym.get(int(x), str(x)) for x in genes])
    res = {"n_clones": int(ok.sum()), "batches": {b: int((batch == b).sum()) for b in np.unique(batch)}}
    if ok.sum() < 40:
        res["reading"] = "underpowered"
    scores, used = {}, {}
    for f, gs in PROGRAMMES.items():
        idx = [i for i, s in enumerate(names) if s in gs]
        used[f] = names[idx].tolist()
        scores[f] = Z[:, idx].mean(1) if idx else np.zeros(len(Z))
    res["genes_used"] = used
    fi = {f: j for j, f in enumerate(sd.FATES)}
    tab, ci = {}, {}
    for f in PROGRAMMES:
        tab[f] = {h: sd.spearman(scores[f], L[:, fi[h]]) for h in sd.FATES}
        ci[f] = sd.bootstrap(lambda ix: sd.spearman(scores[f][ix], L[ix, fi[f]]), len(L), rng, boot)
    res["A_own_fate"] = {f: {"rho": tab[f][f], "ci": ci[f]} for f in PROGRAMMES}
    res["A_table_score_by_fate"] = tab
    var = Z.var(0)
    top = np.argsort(var)[::-1][:2000]
    U, S, _ = np.linalg.svd(Z[:, top] - Z[:, top].mean(0), full_matrices=False)
    pcs = U[:, :3] * S[:3]
    res["B_pc_var_share"] = (S[:3] ** 2 / (S ** 2).sum()).tolist()
    res["B_loo_r2_PC1_3"] = {f: loo_r2(pcs, L[:, fi[f]]) for f in sd.FATES}
    m = ok
    k1, k2 = d["k1"][m], d["k2"][m]
    l1 = centre(np.log((k1 + 0.5) / (k1.sum(1, keepdims=True) - k1 + 0.5)), batch)
    l2 = centre(np.log((k2 + 0.5) / (k2.sum(1, keepdims=True) - k2 + 0.5)), batch)
    res["C_clone_level_share"] = {f: float(np.cov(l1[:, j], l2[:, j])[0, 1] / np.var(L[:, j]))
                                  for f, j in fi.items()}
    out = out or os.path.join(os.path.dirname(__file__), "results_sis")
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, "b41.json"), "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=lambda x: round(float(x), 3)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    main(a.out, a.boot)
