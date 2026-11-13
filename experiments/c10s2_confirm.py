"""C10-S2: pre-treatment state predicts survival beyond CNV subclone + QC; oracle look on resistrace:A.

Serves: C10-S2. Bence: "go". Pre-registered in EXPERIMENT_LOG ("C10-S2").
  fit       : train only (partitions.train_only); freezes the pipeline to selection/C10-S2_model.npz and
              writes selection/C10-S2.json (power + readout blocks, model sha256).
  counts_A  : structural count of labelled drug-library lineages in a sealed partition (no outcome column).
  test_A    : called ONLY by test_oracle (look resistrace_c10s2 --partition A).
"""
import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import c10s0_placebo as pl  # noqa: E402
import c10s1_select as c1  # noqa: E402
import partitions  # noqa: E402

DS = "resistrace"
DRUGS = c1.DRUGS
SAMPLES = c1.SAMPLES
SEL = os.path.join(HERE, "selection", "C10-S2.json")
MODEL = os.path.join(HERE, "selection", "C10-S2_model.npz")
rng = np.random.default_rng(121)


def _info(s):
    f = glob.glob(os.path.join(pl.PROC, f"*ReSisTrace_{s}_pre_treatment_cell_info.tsv.gz"))[0]
    pre = pd.read_csv(f, sep="\t", index_col=0, usecols=lambda c: c in (
        "", "Unnamed: 0", "sisters", "drugSens", "prer_r_group", "nCount_RNA", "nFeature_RNA", "percent.mt"))
    pre = pre[pre["drugSens"].notna()].copy()
    pre["unit"] = [f"{s}:s{int(x)}" if pd.notna(x) else f"{s}:c{bc}" for bc, x in zip(pre.index, pre["sisters"])]
    return pre


def _genes():
    sets = []
    for s in SAMPLES:
        f = glob.glob(os.path.join(pl.MAT, f"*ReSisTrace_{s}_pre_treatment_UMIcounts.tsv.gz"))[0]
        sets.append(set(pd.read_csv(f, sep="\t", usecols=[0]).iloc[:, 0]))
    return sorted(set.intersection(*sets))


def _cnv_profile(X, order, chroms, gmean):
    Xc = X[:, order] - gmean
    prof = []
    for c in range(1, 23):
        idx = np.where(chroms == c)[0]
        if len(idx) < 101:
            continue
        cs = np.cumsum(np.pad(Xc[:, idx], ((0, 0), (1, 0))), axis=1)
        prof.append(((cs[:, 101:] - cs[:, :-101]) / 101)[:, ::25])
    prof = np.hstack(prof)
    return prof - prof.mean(1, keepdims=True)


def _lineage_table(meta, sub, pcs, info):
    meta = meta.copy()
    meta["subclone"] = sub
    for j in range(5):
        meta[f"pc{j}"] = pcs[:, j]
    agg = meta.groupby("unit").agg(sample=("sample", "first"), subclone=("subclone", lambda x: x.mode().iloc[0]),
                                   **{f"pc{j}": (f"pc{j}", "mean") for j in range(5)})
    g = info.groupby("unit")
    q = pd.DataFrame({"q_count": np.log(g["nCount_RNA"].mean()), "q_feat": np.log(g["nFeature_RNA"].mean()),
                      "q_mt": g["percent.mt"].mean(), "q_ncell": np.log(g.size())})
    return agg.join(q, how="inner")


def _design(d, k, qmu, qsd, pmu, psd):
    L = np.column_stack([(d["sample"] == s).values for s in DRUGS]).astype(float)
    G = np.column_stack([(d["subclone"] == j).values for j in range(1, k)]).astype(float)
    Q = (d[["q_count", "q_feat", "q_mt", "q_ncell"]].values - qmu) / qsd
    P = (d[[f"pc{j}" for j in range(5)]].values - pmu) / psd
    return np.hstack([L, G, Q]), np.hstack([L, G, Q, P])


def fit():
    cells = pl.allowed_cells()                    # train lineages + unlabelled pre cells only
    genes = _genes()
    Xs, meta = [], []
    for s in SAMPLES:
        df = cells[(s, "pre")]
        X, _ = pl.load(s, "pre", list(df.index), genes=genes)
        Xs.append(X); meta.append(pd.DataFrame({"sample": s, "unit": df["unit"].values}, index=df.index))
        print("loaded", s, X.shape, flush=True)
    X = np.vstack(Xs); meta = pd.concat(meta); del Xs
    pos = pd.read_csv(c1.POS, sep="\t"); pos.columns = ["gene", "chrom", "start"]
    pos = pos[pos["chrom"].isin([str(c) for c in range(1, 23)])].drop_duplicates("gene").set_index("gene")
    base = [g.split(".")[0] for g in genes]
    keep = (X.mean(0) > 0.1) & np.array([b in pos.index for b in base])
    order = np.array(sorted(np.where(keep)[0],
                            key=lambda i: (int(pos.loc[base[i], "chrom"]), pos.loc[base[i], "start"])))
    chroms = np.array([int(pos.loc[base[i], "chrom"]) for i in order])
    gmean = X[:, order].mean(0)
    prof = _cnv_profile(X, order, chroms, gmean)
    pmean = prof.mean(0)
    U, Sv, Vt = np.linalg.svd(prof - pmean, full_matrices=False)
    cnv_V = Vt[:10].T
    cnv_pc = (prof - pmean) @ cnv_V
    sil, labs, cents = {}, {}, {}
    for k in (2, 3, 4):
        lab, C = c1.kmeans(cnv_pc, k)
        labs[k], cents[k] = lab, C
        sil[k] = c1.silhouette(cnv_pc, lab)
    k = max(sil, key=sil.get)
    m, v = X.mean(0), X.var(0)
    hvg = np.argsort(-np.where(m > 0.02, v / np.maximum(m, 1e-6), -1))[:2000]
    hmu, hsd = X[:, hvg].mean(0), X[:, hvg].std(0) + 1e-6
    Z = (X[:, hvg] - hmu) / hsd
    w, V = np.linalg.eigh(np.cov(Z, rowvar=False))
    sV = V[:, ::-1][:, :5]
    pcs = Z @ sV
    info = pd.concat([_info(s) for s in SAMPLES])
    info_tr = partitions.train_only(DS, info, "unit")
    has = meta["unit"].notna().values
    tab = _lineage_table(meta[has], labs[k][has], pcs[has], info_tr)
    linked = info_tr.groupby("unit")["prer_r_group"].apply(lambda x: x.notna().any()).astype(int)
    tab = tab.join(linked.rename("linked"), how="inner")
    d = tab[tab["sample"].isin(DRUGS)]
    qcols = ["q_count", "q_feat", "q_mt", "q_ncell"]; pcols = [f"pc{j}" for j in range(5)]
    qmu, qsd = d[qcols].values.mean(0), d[qcols].values.std(0)
    pmu, psd = d[pcols].values.mean(0), d[pcols].values.std(0)
    X0, X1 = _design(d, k, qmu, qsd, pmu, psd)
    y = d["linked"].values.astype(float)
    w0, w1 = c1.logit_fit(X0, y), c1.logit_fit(X1, y)
    insample = c1.auc(X1 @ w1, y) - c1.auc(X0 @ w0, y)
    np.savez(MODEL, hvg=hvg, hmu=hmu, hsd=hsd, sV=sV, order=order, chroms=chroms, gmean=gmean, pmean=pmean,
             cnv_V=cnv_V, cent=cents[k], k=k, qmu=qmu, qsd=qsd, pmu=pmu, psd=psd, w0=w0, w1=w1,
             genes=np.array(genes))
    sha = hashlib.sha256(open(MODEL, "rb").read()).hexdigest()
    n_tr = int(len(y))
    sel = {
        "test": "resistrace_c10s2", "higher_is_better": True,
        "candidates": {str(kk): round(float(vv), 4) for kk, vv in sil.items()}, "chosen": str(k),
        "note": "candidates = CNV-proxy subclone k scored by train silhouette (pre-registered C10-S1); "
                "statistic delta AUC (M1 - M0) on sealed drug lineages, lower 97.5% bootstrap bound > 0",
        "model_sha256": sha,
        "train": {"n_lineages": n_tr, "n_linked": int(y.sum()), "insample_delta_auc": round(float(insample), 4),
                  "cv_delta_auc_c10s1_qc": 0.0549, "cv_se_c10s1_qc": 0.0066},
        "power": {"effect": round(0.5 * 0.0549, 5), "sd": round(0.0066 * float(np.sqrt(10132)), 4), "alpha": 0.025,
                  "one_sided": True, "min_power": 0.8,
                  "sources": {"effect": "C10-S1 amendment 1 train CV delta AUC 0.0549 x 0.5 shrink",
                              "sd": "C10-S1 amendment 1 train bootstrap SE 0.0066 x sqrt(10132 lineages)"}},
        "readout_gate": {"pass": True, "sides": {"linked": int(y.sum()), "unlinked": int(n_tr - y.sum())}},
        "exposure": "pooled per-sample pre-resistant counts seen (accepted by Bence 2026-11-13)",
    }
    json.dump(sel, open(SEL, "w"), indent=1)
    print(json.dumps(sel, indent=1))


def _sealed_info(part):
    return pd.concat([partitions.sealed_rows(DS, _info(s), "unit", part) for s in DRUGS])


def counts_A(part="A"):
    """Structural: labelled drug-library lineages in the sealed partition (no outcome column used)."""
    info = _sealed_info(part)
    return {"lineages": int(info["unit"].nunique())}


def test_A(part="A"):
    M = np.load(MODEL, allow_pickle=False)
    sha = hashlib.sha256(open(MODEL, "rb").read()).hexdigest()
    assert sha == json.load(open(SEL))["model_sha256"], "frozen model changed"
    genes = list(M["genes"]); k = int(M["k"])
    info = _sealed_info(part)
    meta, Xs = [], []
    for s in DRUGS:
        sub_s = info[info["unit"].str.startswith(s + ":")]   # barcodes repeat across libraries: select per library
        bcs = list(sub_s.index)
        X, _ = pl.load(s, "pre", bcs, genes=genes)
        Xs.append(X); meta.append(pd.DataFrame({"sample": s, "unit": sub_s["unit"].values}, index=bcs))
    X = np.vstack(Xs); meta = pd.concat(meta)
    prof = _cnv_profile(X, M["order"], M["chroms"], M["gmean"])
    cnv_pc = (prof - M["pmean"]) @ M["cnv_V"]
    sub = ((cnv_pc[:, None, :] - M["cent"][None]) ** 2).sum(2).argmin(1)
    pcs = ((X[:, M["hvg"]] - M["hmu"]) / M["hsd"]) @ M["sV"]
    tab = _lineage_table(meta, sub, pcs, info)
    linked = info.groupby("unit")["prer_r_group"].apply(lambda x: x.notna().any()).astype(int)
    tab = tab.join(linked.rename("linked"), how="inner")
    X0, X1 = _design(tab, k, M["qmu"], M["qsd"], M["pmu"], M["psd"])
    y = tab["linked"].values.astype(float)
    s0, s1 = X0 @ M["w0"], X1 @ M["w1"]
    delta = c1.auc(s1, y) - c1.auc(s0, y)
    boots = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            boots.append(c1.auc(s1[i], y[i]) - c1.auc(s0[i], y[i]))
    lo = float(np.quantile(boots, 0.025))
    return {"n_lineages": int(len(y)), "n_linked": int(y.sum()), "auc_M0": round(c1.auc(s0, y), 4),
            "auc_M1": round(c1.auc(s1, y), 4), "delta_auc": round(float(delta), 4), "lower_97_5": round(lo, 4),
            "confirmed": bool(lo > 0)}


if __name__ == "__main__":
    if sys.argv[1:] == ["fit"]:
        fit()
