"""C10-S1: within-library genetic vs non-genetic selection - TRAIN-ONLY power check.

Serves: C10-S1. Bence: "yes". Pre-registered in EXPERIMENT_LOG ("C10-S1"). Nothing sealed is read:
cells come from c10s0_placebo.allowed_cells (train lineages + unlabelled pre cells), and lineage
outcomes (post link yes/no) are read only for train lineages via partitions.train_only.
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import c10s0_placebo as pl  # noqa: E402
import partitions  # noqa: E402

DS = "resistrace"
POS = os.path.join(HERE, "..", "data", "ensembl_ref_human", "gene_pos_grch38.tsv")
OUT = os.path.join(HERE, "results", "c10s1_power_qc.json" if "--qc" in sys.argv else "c10s1_power.json")
DRUGS = ["Carboplatin_1", "Carboplatin_2", "Olaparib_1", "Olaparib_2", "NK_1", "NK_2"]
SAMPLES = DRUGS + ["Control_1", "Control_2"]
rng = np.random.default_rng(111)


def train_outcomes():
    """Train lineages of each pre library: unit -> linked (any pre cell carries prer_r_group)."""
    rows = []
    for s in SAMPLES:
        f = glob.glob(os.path.join(pl.PROC, f"*ReSisTrace_{s}_pre_treatment_cell_info.tsv.gz"))[0]
        pre = pd.read_csv(f, sep="\t", index_col=0,
                          usecols=lambda c: c in ("", "Unnamed: 0", "sisters", "drugSens", "prer_r_group",
                                                  "nCount_RNA", "nFeature_RNA", "percent.mt"))
        pre = pre[pre["drugSens"].notna()]
        pre["unit"] = [f"{s}:s{int(x)}" if pd.notna(x) else f"{s}:c{bc}" for bc, x in zip(pre.index, pre["sisters"])]
        tr = partitions.train_only(DS, pre, "unit")
        g = tr.groupby("unit")
        lk = g["prer_r_group"].apply(lambda x: x.notna().any())
        qc = pd.DataFrame({"q_count": np.log(g["nCount_RNA"].mean()), "q_feat": np.log(g["nFeature_RNA"].mean()),
                           "q_mt": g["percent.mt"].mean(), "q_ncell": np.log(g.size())})
        rows.append(pd.DataFrame({"sample": s, "linked": lk.astype(int)}).join(qc))
    return pd.concat(rows)


def kmeans(X, k, n_init=10, iters=100):
    best = None
    for _ in range(n_init):
        C = X[rng.choice(len(X), k, replace=False)]
        for _ in range(iters):
            lab = ((X[:, None, :] - C[None]) ** 2).sum(2).argmin(1)
            C2 = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
            if np.allclose(C, C2):
                break
            C = C2
        inertia = ((X - C[lab]) ** 2).sum()
        if best is None or inertia < best[0]:
            best = (inertia, lab, C)
    return best[1], best[2]


def silhouette(X, lab, n=3000):
    i = rng.choice(len(X), min(n, len(X)), replace=False)
    X, lab = X[i], lab[i]
    D = np.sqrt(((X[:, None, :] - X[None]) ** 2).sum(2))
    s = []
    for a in range(len(X)):
        same = lab == lab[a]
        if same.sum() < 2:
            continue
        ai = D[a, same].sum() / (same.sum() - 1)
        bi = min(D[a, lab == c].mean() for c in set(lab) if c != lab[a])
        s.append((bi - ai) / max(ai, bi))
    return float(np.mean(s))


def logit_fit(X, y, ridge=1e-2):
    w = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ w))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + ridge * np.eye(X.shape[1])
        w = w + np.linalg.solve(H, X.T @ (y - p) - ridge * w)
    return w


def auc(score, y):
    r = pd.Series(score).rank().values
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def oof_scores(Xs, y, folds):
    out = [np.zeros(len(y)) for _ in Xs]
    for f in range(5):
        tr, te = folds != f, folds == f
        for j, X in enumerate(Xs):
            w = logit_fit(X[tr], y[tr])
            out[j][te] = X[te] @ w
    return out


def main():
    cells = pl.allowed_cells()
    pos = pd.read_csv(POS, sep="\t")
    pos.columns = ["gene", "chrom", "start"]
    pos = pos[pos["chrom"].isin([str(c) for c in range(1, 23)])].drop_duplicates("gene").set_index("gene")
    # common genes, version-stripped
    gene_sets = []
    for s in SAMPLES:
        f = glob.glob(os.path.join(pl.MAT, f"*ReSisTrace_{s}_pre_treatment_UMIcounts.tsv.gz"))[0]
        gene_sets.append(set(pd.read_csv(f, sep="\t", usecols=[0]).iloc[:, 0]))
    genes = sorted(set.intersection(*gene_sets))
    Xs, meta = [], []
    for s in SAMPLES:
        df = cells[(s, "pre")]
        X, _ = pl.load(s, "pre", list(df.index), genes=genes)
        Xs.append(X)
        meta.append(pd.DataFrame({"sample": s, "unit": df["unit"].values}, index=df.index))
        print("loaded", s, X.shape, flush=True)
    X = np.vstack(Xs); meta = pd.concat(meta); del Xs
    # ---- CNV proxy ----
    base = [g.split(".")[0] for g in genes]
    keep = (X.mean(0) > 0.1) & np.array([b in pos.index for b in base])
    gi = np.where(keep)[0]
    order = sorted(gi, key=lambda i: (int(pos.loc[base[i], "chrom"]), pos.loc[base[i], "start"]))
    Xc = X[:, order] - X[:, order].mean(0)
    chroms = np.array([int(pos.loc[base[i], "chrom"]) for i in order])
    prof = []
    for c in range(1, 23):
        idx = np.where(chroms == c)[0]
        if len(idx) < 101:
            continue
        sub = Xc[:, idx]
        cs = np.cumsum(np.pad(sub, ((0, 0), (1, 0))), axis=1)
        sm = (cs[:, 101:] - cs[:, :-101]) / 101
        prof.append(sm[:, ::25])
    prof = np.hstack(prof)
    prof = prof - prof.mean(1, keepdims=True)
    U, Sv, Vt = np.linalg.svd(prof - prof.mean(0), full_matrices=False)
    cnv_pc = U[:, :10] * Sv[:10]
    sil = {}
    labs = {}
    for k in (2, 3, 4):
        lab, _ = kmeans(cnv_pc, k)
        labs[k] = lab
        sil[k] = round(silhouette(cnv_pc, lab), 3)
    kc = max(sil, key=sil.get)
    res = {"cnv_genes": int(len(order)), "cnv_silhouette": sil, "k_subclone": kc}
    if sil[kc] <= 0.1:
        res["stop"] = "CNV proxy does not separate subclones (silhouette <= 0.1): genetic term undefined"
        json.dump(res, open(OUT, "w"), indent=1); print(json.dumps(res, indent=1)); return
    meta["subclone"] = labs[kc]
    # ---- non-genetic state: HVG PCA k=5 ----
    m, v = X.mean(0), X.var(0)
    hvg = np.argsort(-np.where(m > 0.02, v / np.maximum(m, 1e-6), -1))[:2000]
    Z = (X[:, hvg] - X[:, hvg].mean(0)) / (X[:, hvg].std(0) + 1e-6)
    w, V = np.linalg.eigh(np.cov(Z, rowvar=False))
    pcs = Z @ V[:, ::-1][:, :5]
    for j in range(5):
        meta[f"pc{j}"] = pcs[:, j]
    # ---- lineage table (train lineages, drug libraries) ----
    lin = meta.dropna(subset=["unit"])
    agg = lin.groupby("unit").agg(sample=("sample", "first"), subclone=("subclone", lambda x: x.mode().iloc[0]),
                                  **{f"pc{j}": (f"pc{j}", "mean") for j in range(5)})
    out = train_outcomes()
    agg = agg.join(out[["linked", "q_count", "q_feat", "q_mt", "q_ncell"]], how="inner")
    res["linked_rate_per_library_train"] = agg.groupby("sample")["linked"].agg(["sum", "count"]).to_dict("index")
    d = agg[agg["sample"].isin(DRUGS)].copy()
    y = d["linked"].values.astype(float)
    L = pd.get_dummies(d["sample"]).values.astype(float)
    G = pd.get_dummies(d["subclone"]).values[:, 1:].astype(float)
    P = d[[f"pc{j}" for j in range(5)]].values
    P = (P - P.mean(0)) / P.std(0)
    Q = d[["q_count", "q_feat", "q_mt", "q_ncell"]].values.astype(float)
    Q = (Q - Q.mean(0)) / Q.std(0)
    if "--qc" in sys.argv:                      # amendment 1: QC covariates in both models
        XL = np.hstack([L, Q]); X0 = np.hstack([L, G, Q]); X1 = np.hstack([L, G, Q, P])
        res["amendment"] = "1: QC covariates (log nCount, log nFeature, percent.mt, log n pre cells) in both models"
    else:
        XL = L
        X0 = np.hstack([L, G])
        X1 = np.hstack([L, G, P])
    folds = rng.integers(0, 5, size=len(y))
    sL, s0, s1 = oof_scores([XL, X0, X1], y, folds)
    res["train_lineages_drug"] = int(len(y)); res["train_linked_drug"] = int(y.sum())
    res["AUC_library"] = round(auc(sL, y), 3)
    res["AUC_M0_library_subclone"] = round(auc(s0, y), 3)
    res["AUC_M1_plus_state"] = round(auc(s1, y), 3)
    delta = auc(s1, y) - auc(s0, y)
    boots = []
    for _ in range(500):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() < 5:
            continue
        boots.append(auc(s1[i], y[i]) - auc(s0[i], y[i]))
    se = float(np.std(boots))
    se3 = se * np.sqrt(2)
    from math import erf, sqrt
    phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))  # noqa: E731
    res["delta_auc_train"] = round(delta, 4); res["se_train"] = round(se, 4); res["se_sealed_third"] = round(se3, 4)
    res["power_shrunk_0.5"] = round(phi(0.5 * delta / se3 - 1.96), 3)
    res["power_unshrunk"] = round(phi(delta / se3 - 1.96), 3)
    res["go"] = bool(res["power_shrunk_0.5"] >= 0.8)
    json.dump(res, open(OUT, "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
