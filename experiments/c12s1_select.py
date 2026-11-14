"""C12-S1: train-only selection + power check, LSC LARRY (pre-registered in EXPERIMENT_LOG "C12-S1").

Serves: C12-S1. Bence: "go with step 1".
    python c12s1_select.py          # train CV only; no sealed outcome is read; writes selection/C12-S1.json
Outcomes (presence at D40 ex vivo / D20 in vivo) are indexed ONLY for train clones, chosen with
partitions.train_only("lsc_larry", ...). Per-partition counts printed are structural (unit counts).
"""

from __future__ import annotations

import collections
import json
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import c12s0_lsc as c0
import partitions
import power

HERE = os.path.dirname(os.path.abspath(__file__))
SEL = os.path.join(HERE, "selection", "C12-S1.json")
OUT = os.path.join(HERE, "results_c12", "c12s1_train.json")
MIN_UMI, MIN_CELLS = 3, 3
OUTCOMES = ["D40_exvivo", "D20_invivo"]
rng = np.random.default_rng(1212)


def assign_max(s, cmap):
    """cell (1-based column) -> clone key by its highest-UMI Custom barcode (>= MIN_UMI); ties -> smallest ID."""
    feats = c0._features(s)
    cust = {i + 1: x[0] for i, x in enumerate(feats) if x[2] == "Custom"}
    lo, hi = min(cust), max(cust)
    cs, rs, vs = [], [], []
    for ch in pd.read_csv(os.path.join(c0.RAW, f"{s}_matrix.mtx.gz"), sep=" ", comment="%", header=None, skiprows=3,
                          chunksize=5_000_000, dtype={0: np.int64, 1: np.int64, 2: np.int64}):
        r, c, v = ch[0].to_numpy(), ch[1].to_numpy(), ch[2].to_numpy()
        k = (r >= lo) & (r <= hi) & (v >= MIN_UMI)
        rs.append(r[k]); cs.append(c[k]); vs.append(v[k])
    r, c, v = np.concatenate(rs), np.concatenate(cs), np.concatenate(vs)
    o = np.lexsort((r, -v, c))
    r, c = r[o], c[o]
    first = np.r_[True, c[1:] != c[:-1]]
    out = {}
    for cell, row in zip(c[first], r[first]):
        key = cmap.get(cust[int(row)], cust[int(row)])
        if key != "AMBIENT":
            out[int(cell)] = key
    return out


def d20_expression(cells):
    """Gene counts of the given D20 ex vivo cells (1-based columns): csr cells x genes, gene names."""
    from scipy import sparse
    feats = c0._features("D20_exvivo")
    gene_rows = {i + 1: x[1] for i, x in enumerate(feats) if x[2] == "Gene Expression"}
    assert gene_rows, "no Gene Expression rows"
    grow = np.array(sorted(gene_rows))
    rmap = np.full(grow.max() + 1, -1, dtype=np.int64)
    rmap[grow] = np.arange(len(grow))
    cellarr = np.array(cells)
    cmap_ = np.full(cellarr.max() + 1, -1, dtype=np.int64)
    cmap_[cellarr] = np.arange(len(cellarr))
    rr, cc, vv = [], [], []
    for ch in pd.read_csv(os.path.join(c0.RAW, "D20_exvivo_matrix.mtx.gz"), sep=" ", comment="%", header=None, skiprows=3,
                          chunksize=5_000_000, dtype={0: np.int64, 1: np.int64, 2: np.int64}):
        r, c, v = ch[0].to_numpy(), ch[1].to_numpy(), ch[2].to_numpy()
        k = (r <= grow.max()) & (c <= cellarr.max())
        r, c, v = r[k], c[k], v[k]
        k = (rmap[r] >= 0) & (cmap_[c] >= 0)
        rr.append(rmap[r[k]]); cc.append(cmap_[c[k]]); vv.append(v[k])
    X = sparse.csr_matrix((np.concatenate(vv), (np.concatenate(cc), np.concatenate(rr))), shape=(len(cells), len(grow)), dtype=np.float32)
    return X, [gene_rows[r] for r in grow]


def cv_auc(M, y, repeats=5, seed=0):
    aucs, oofs = [], []
    for rep in range(repeats):
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed + rep).split(M, y):
            mu, sd = M[tr].mean(0), M[tr].std(0) + 1e-9
            lr = LogisticRegression(C=1.0, max_iter=2000).fit((M[tr] - mu) / sd, y[tr])
            oof[te] = lr.predict_proba((M[te] - mu) / sd)[:, 1]
        aucs.append(roc_auc_score(y, oof)); oofs.append(oof)
    return float(np.mean(aucs)), oofs


def main():
    cmap = dict(pd.read_csv(c0.MAP, sep="\t").itertuples(index=False, name=None))
    asg = {s: assign_max(s, cmap) for s in c0.SAMPLES}
    size = {s: collections.Counter(a.values()) for s, a in asg.items()}
    units = pd.DataFrame({"clone": [k for k, n in size["D20_exvivo"].items() if n >= MIN_CELLS]})
    res = {"rule": "highest-UMI barcode per cell (>= 3 UMI), mapped through the locked clone map",
           "cells_assigned": {s: len(a) for s, a in asg.items()}}
    res["units_ge3_per_partition"] = dict(collections.Counter(units["clone"].map(lambda u: partitions.partition_of(c0.DS, u))))  # structural
    tr = partitions.train_only("lsc_larry", units, "clone")  # outcomes: train only (H11)
    tr_clones = list(tr["clone"])
    tr_set = set(tr_clones)
    cells = sorted(c for c, k in asg["D20_exvivo"].items() if k in tr_set)
    X, genes = d20_expression(cells)
    tot = np.asarray(X.sum(1)).ravel()
    nfeat = np.asarray((X > 0).sum(1)).ravel()
    mt = np.array([g.lower().startswith("mt-") for g in genes])
    pmt = np.asarray(X[:, mt].sum(1)).ravel() / np.maximum(tot, 1)
    Xn = X.multiply(1e4 / np.maximum(tot, 1)[:, None]).tocsr()
    Xn.data = np.log1p(Xn.data)
    var = np.asarray(Xn.power(2).mean(0)).ravel() - np.asarray(Xn.mean(0)).ravel() ** 2
    Z = Xn[:, np.argsort(-var)[:2000]].toarray()
    cell_clone = np.array([asg["D20_exvivo"][c] for c in cells])
    idx = {k: np.where(cell_clone == k)[0] for k in tr_clones}
    mean_expr = np.vstack([Z[idx[k]].mean(0) for k in tr_clones])
    pcs = PCA(5, random_state=0).fit_transform(mean_expr - mean_expr.mean(0))
    qc = np.column_stack([[np.log(len(idx[k])) for k in tr_clones],
                          [np.log(tot[idx[k]].mean()) for k in tr_clones],
                          [np.log(max(1.0, nfeat[idx[k]].mean())) for k in tr_clones],
                          [pmt[idx[k]].mean() for k in tr_clones]])
    lib = np.array([k.rsplit("_", 1)[0] for k in tr_clones])
    libs = sorted(set(lib))
    L = np.column_stack([(lib == x).astype(float) for x in libs[1:]]) if len(libs) > 1 else np.zeros((len(lib), 0))
    qcL = np.hstack([qc, L])                                                            # amendment 1
    res["train"] = {"n_units": len(tr_clones), "library_prefix_counts": dict(collections.Counter(lib))}
    sel = {"test": "lsc_c12s1", "higher_is_better": True,
           "note": "delta AUC (M1 - M0), M0 = log D20 size + QC, M1 = M0 + 5 PCs of clone-mean D20 expression; train CV only"}
    for s in OUTCOMES:
        y = np.array([1 if size[s].get(k, 0) > 0 else 0 for k in tr_clones])
        r = {"positives": int(y.sum()), "prevalence": round(float(y.mean()), 4)}
        if y.sum() < 30 or (len(y) - y.sum()) < 30:
            r["readout_gate"] = "FAIL (< 30 per side)"
            res["train"][s] = r
            continue
        M0, M1 = qcL, np.hstack([qcL, pcs])
        a0, o0 = cv_auc(M0, y)
        a1, o1 = cv_auc(M1, y)
        a_size, _ = cv_auc(qc[:, :1], y)
        a0_noL, _ = cv_auc(qc, y)
        a1_noL, _ = cv_auc(np.hstack([qc, pcs]), y)
        a_lib, _ = cv_auc(L, y) if L.shape[1] else (0.5, None)
        delta = a1 - a0
        p0, p1 = np.mean(o0, 0), np.mean(o1, 0)
        bs = []
        for _ in range(500):
            b = rng.integers(0, len(y), len(y))
            if y[b].min() != y[b].max():
                bs.append(roc_auc_score(y[b], p1[b]) - roc_auc_score(y[b], p0[b]))
        se = float(np.std(bs))
        plc = [cv_auc(np.hstack([qcL, pcs[rng.permutation(len(y))]]), y, repeats=1, seed=100 + j)[0] - a0 for j in range(20)]
        null = []
        for j in range(10):
            yp = rng.permutation(y)
            null.append(cv_auc(M1, yp, repeats=1, seed=200 + j)[0] - cv_auc(M0, yp, repeats=1, seed=200 + j)[0])
        effect = 0.5 * delta
        sd_pos = se * np.sqrt(y.sum())
        pw = {}
        for p, n in res["units_ge3_per_partition"].items():
            if p == "train":
                continue
            npos = max(1.0, n * y.mean())
            pw[p] = {"units": int(n), "expected_positives": round(float(npos), 1),
                     "power": round(float(power.z_power(effect, sd_pos / np.sqrt(npos))), 3)}
        r.update({"auc_size_only": round(a_size, 4), "auc_library_only": round(a_lib, 4), "delta_auc_without_library": round(a1_noL - a0_noL, 4), "auc_M0": round(a0, 4), "auc_M1": round(a1, 4), "delta_auc": round(delta, 4),
                  "boot_se": round(se, 4), "state_placebo_delta_mean": round(float(np.mean(plc)), 4),
                  "state_placebo_delta_sd": round(float(np.std(plc)), 4), "label_shuffle_delta_mean": round(float(np.mean(null)), 4),
                  "power_effect_shrunk": round(effect, 4), "power": pw, "readout_gate": "PASS"})
        res["train"][s] = r
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    sel["train"] = res["train"]
    sel["exposure"] = "C12-S0 counts + train presence fractions; no sealed outcome read"
    json.dump(sel, open(SEL, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
