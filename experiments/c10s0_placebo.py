"""C10-S0 placebo (train only). See EXPERIMENT_LOG "C10-S0 placebo". Bence: "go forward with recomms".

Columns are chosen from cell_info before any expression value is read: sealed-lineage pre cells and
sealed-family post cells are never loaded. Featuriser: log CP10k -> 2000 HVG -> z -> PCA k (10/20/30),
k chosen by train pre sister-pair AUC (not by S). P1 = logF(Control_1) - logF(Control_2) on train
families; P2 = sister-group logF per pre library (train sister groups).
"""
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import partitions  # noqa: E402

DS = "resistrace"
PROC = os.path.join(HERE, "..", "data", "ext_resistrace", "processed")
MAT = os.path.join(HERE, "..", "data", "ext_resistrace", "raw_tar", "files")
OUT = os.path.join(HERE, "results", "c10s0_placebo_v5.json" if "--v5" in sys.argv else "c10s0_placebo.json")
rng = np.random.default_rng(101)


def sample_of(path):
    return re.search(r"ReSisTrace_(\w+?_\d)_(pre|post)", path).group(1)


def allowed_cells():
    """Per (sample, phase): list of (barcode, lineage unit or None, family or None)."""
    cells = {}
    for f in sorted(glob.glob(os.path.join(PROC, "*_pre_treatment_cell_info.tsv.gz"))):
        s = sample_of(f)
        pre = pd.read_csv(f, sep="\t", index_col=0,
                          usecols=lambda c: c in ("", "Unnamed: 0", "sisters", "drugSens", "prer_r_group"))
        lab = pre["drugSens"].notna()
        unit = pd.Series([f"{s}:s{int(x)}" if pd.notna(x) else f"{s}:c{bc}"
                          for bc, x in zip(pre.index, pre["sisters"])], index=pre.index)
        tr = partitions.train_only(DS, pd.DataFrame({"unit": unit[lab]}), "unit")   # H11: train lineages only
        is_tr = pd.Series(pre.index.isin(tr.index), index=pre.index)
        keep = (~lab) | is_tr
        u = unit.where(is_tr)
        cells[(s, "pre")] = pd.DataFrame({"unit": u[keep], "fam": pre.loc[keep, "prer_r_group"].where(lab[keep])})
        fam2unit = {fam: unit[bc] for bc, fam in pre["prer_r_group"].dropna().items()}
        pf = glob.glob(os.path.join(PROC, f"*ReSisTrace_{s}_post_treatment_cell_info.tsv.gz"))[0]
        post = pd.read_csv(pf, sep="\t", index_col=0, usecols=lambda c: c in ("", "Unnamed: 0", "prer_r_group"))
        famp = post["prer_r_group"]
        fam_df = pd.DataFrame({"unit": pd.Series(fam2unit)})
        train_fams = set(partitions.train_only(DS, fam_df, "unit").index)
        in_train = famp.map(lambda x: pd.notna(x) and x in train_fams)
        keep = famp.isna() | in_train
        cells[(s, "post")] = pd.DataFrame({"unit": famp[keep].map(lambda x: fam2unit[x] if pd.notna(x) else None),
                                           "fam": famp[keep]})
    return cells


def load(s, phase, barcodes, genes=None):
    f = glob.glob(os.path.join(MAT, f"*ReSisTrace_{s}_{phase}_treatment_UMIcounts.tsv.gz"))[0]
    want = set(barcodes)
    df = pd.read_csv(f, sep="\t", index_col=0, usecols=lambda c: c in ("", "Unnamed: 0") or c in want,
                     dtype={b: np.int32 for b in barcodes})
    tot = df[list(barcodes)].to_numpy(np.float32).sum(0)  # library size on all genes of this file
    if genes is not None:
        df = df.reindex(genes, fill_value=0)
    X = df[list(barcodes)].to_numpy(np.float32).T          # cells x genes
    X = np.log1p(X / np.maximum(tot[:, None], 1) * 1e4)
    return X, list(df.index)


def main():
    cells = allowed_cells()
    keys = sorted(cells)
    # pass 1: HVG on a 400-cell subsample per file
    # common gene list = genes present in all 16 matrices (QC differs per sample); read gene column only
    gene_sets = []
    for k in keys:
        f = glob.glob(os.path.join(MAT, f"*ReSisTrace_{k[0]}_{k[1]}_treatment_UMIcounts.tsv.gz"))[0]
        gene_sets.append(set(pd.read_csv(f, sep="	", usecols=[0]).iloc[:, 0]))
    gene_ref = sorted(set.intersection(*gene_sets))
    sub = []
    for k in keys:
        bcs = list(cells[k].index)
        pick = list(rng.choice(bcs, size=min(400, len(bcs)), replace=False))
        X, _ = load(*k, pick, genes=gene_ref)
        sub.append(X)
    S = np.vstack(sub)
    m, v = S.mean(0), S.var(0)
    ok = m > 0.02
    disp = np.where(ok, v / np.maximum(m, 1e-6), -1)
    hvg = np.argsort(-disp)[:2000]
    hgenes = [gene_ref[i] for i in hvg]
    mu, sd = S[:, hvg].mean(0), S[:, hvg].std(0) + 1e-6
    del sub, S
    # pass 2: all allowed cells, HVG only
    feats = {}
    for k in keys:
        X, _ = load(*k, list(cells[k].index), genes=gene_ref)
        feats[k] = ((X[:, hvg] - mu) / sd).astype(np.float32)
        print("loaded", k, feats[k].shape, flush=True)
    allX = np.vstack(list(feats.values()))
    C = np.cov(allX, rowvar=False)
    w, V = np.linalg.eigh(C)
    V = V[:, ::-1][:, :30]
    res = {"n_cells_fit": int(len(allX)), "n_genes_hvg": len(hgenes)}
    # featuriser choice: train pre sister-pair AUC
    auc = {}
    for kdim in (10, 20, 30):
        sis, rnd = [], []
        for (s, ph), df in cells.items():
            if ph != "pre":
                continue
            Z = feats[(s, ph)] @ V[:, :kdim]
            u = df["unit"].values
            grp = pd.Series(np.arange(len(u)))[pd.Series(u).str.contains(":s", na=False).values].groupby(
                pd.Series(u)[pd.Series(u).str.contains(":s", na=False)].values).apply(list)
            for idx in grp:
                if len(idx) >= 2:
                    sis.append(np.linalg.norm(Z[idx[0]] - Z[idx[1]]))
                    j = rng.integers(len(Z), size=2)
                    rnd.append(np.linalg.norm(Z[j[0]] - Z[j[1]]))
        sis, rnd = np.array(sis), np.array(rnd)
        auc[kdim] = float((sis[:, None] < rnd[None, :]).mean())
    kbest = max(auc, key=auc.get)
    V5 = "--v5" in sys.argv
    if V5:
        kbest = 5                                   # amendment 5: dof cap k <= 36/5, fixed before running
    res["sister_auc"] = auc
    res["k_chosen"] = kbest
    P = V[:, :kbest]

    def logF(pre_Z, pre_unit, post_Z, post_unit, min_post):
        units = sorted(set(u for u in post_unit if isinstance(u, str)))
        rows = []
        for u in units:
            pz = post_Z[[i for i, x in enumerate(post_unit) if x == u]]
            qz = pre_Z[[i for i, x in enumerate(pre_unit) if x == u]]
            if len(qz) == 0 or len(pz) < min_post:
                continue
            rows.append((qz.mean(0), pz))
        means = np.array([r[1].mean(0) for r in rows]); X = np.column_stack([np.ones(len(rows)),
                                                                             np.array([r[0] for r in rows])])
        n = np.array([len(r[1]) for r in rows])
        multi = [r[1] for r in rows if len(r[1]) >= 2]
        sw2 = sum(((f - f.mean(0)) ** 2).sum() for f in multi) / (sum(len(f) - 1 for f in multi) * P.shape[1])
        return rows, means, X, n, sw2

    def F_from(means, X, n, sw2):
        beta, *_ = np.linalg.lstsq(X, means, rcond=None)
        r = means - X @ beta
        dof = len(means) - X.shape[1]
        if dof <= 0:
            return np.nan
        return np.log((r ** 2).sum() / (dof * means.shape[1]) / (sw2 * np.mean(1.0 / n)))

    p1 = {}
    for est, mp in (("F2", 2), ("Fall", 1)):
        parts = {}
        for s in ("Control_1", "Control_2"):
            pre = cells[(s, "pre")]; post = cells[(s, "post")]
            parts[s] = logF(feats[(s, "pre")] @ P, list(pre["unit"]), feats[(s, "post")] @ P,
                            list(post["unit"]), mp)
        est_S = F_from(*parts["Control_1"][1:]) - F_from(*parts["Control_2"][1:])
        boots = []
        for _ in range(1000):
            vals = []
            for s in ("Control_1", "Control_2"):
                rows, means, X, n, sw2 = parts[s]
                i = rng.integers(len(rows), size=len(rows))
                vals.append(F_from(means[i], X[i], n[i], sw2))
            boots.append(vals[0] - vals[1])
        boots = np.array(boots)
        p1[est] = {"S": round(float(est_S), 3), "ci95": [round(float(np.nanquantile(boots, q)), 3) for q in (.025, .975)],
                   "families": {s: len(parts[s][0]) for s in parts}}
    res["P1_control1_minus_control2"] = p1
    p2 = {}
    for (s, ph), df in cells.items():
        if ph != "pre":
            continue
        Z = feats[(s, ph)] @ P
        u = pd.Series(df["unit"].values)
        g = [np.where(u.values == x)[0] for x in u[u.str.contains(":s", na=False)].unique()]
        g = [ix for ix in g if len(ix) >= 2]
        means = np.array([Z[ix].mean(0) for ix in g]); n = np.array([len(ix) for ix in g])
        def lf(gs):
            mm = np.array([Z[ix].mean(0) for ix in gs]); nn = np.array([len(ix) for ix in gs])
            wi = sum(((Z[ix] - Z[ix].mean(0)) ** 2).sum() for ix in gs) / ((nn - 1).sum() * P.shape[1])
            be = ((mm - mm.mean(0)) ** 2).sum() / ((len(gs) - 1) * P.shape[1])
            return float(np.log(be / (wi * np.mean(1 / nn))))
        if V5:
            r5 = np.random.default_rng(103)
            vals5 = [lf([g[i] for i in r5.choice(len(g), 39, replace=False)]) for _ in range(200)]
            p2[s] = {"groups": len(g), "logF": round(float(np.median(vals5)), 3)}
        else:
            p2[s] = {"groups": len(g), "logF": round(lf(g), 3)}
    vals = [v["logF"] for v in p2.values()]
    res["P2_pre_sister_logF"] = p2
    res["P2_range"] = round(max(vals) - min(vals), 3)
    bar = 0.5
    res["stop"] = bool(abs(p1["F2"]["S"]) >= bar or res["P2_range"] >= bar)
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
