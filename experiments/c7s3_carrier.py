"""C7-S3: does early chromatin carry a clone's fate better than early RNA? CellTag-multi LSK, day 2.5 -> day 5.
See EXPERIMENT_LOG C7-S3. Train CV + selection/power file here; the sealed look runs only via test_oracle.

    python c7s3_carrier.py [--select]
"""

from __future__ import annotations

import argparse
import json
import os

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression

import partitions
from c7s2_sister import CSV, GROUPS, OUT

HERE = os.path.dirname(os.path.abspath(__file__))
GEO = os.path.join(HERE, "..", "data", "ext_celltag_multi", "geo")
H5 = {"rna": os.path.join(GEO, "GSM6681127_d2_5_filtered_feature_bc_matrix.h5"),
      "atac": os.path.join(GEO, "GSM6675425_d2_5_filtered_peak_bc_matrix.h5")}
FATES = ["Neu", "Mono", "BEM", "EMk", "DC", "Lym", "Prog"]
CS = (0.01, 0.1, 1.0, 10.0)
NCOMP = 30
WCAP = 20
SCALE = 1348 / 2720


# ---------- data ----------
def _table():
    return pd.read_csv(CSV, usecols=["clone.id", "cell.bc", "day", "assay", "cell_type2"])


def _fates(tab, fates=FATES):
    d5 = tab[tab["day"] == "d5"].assign(grp=lambda d: d["cell_type2"].map(GROUPS))
    d5 = d5[d5["grp"].isin(fates)]
    return d5.groupby(["clone.id", "grp"]).size().unstack(fill_value=0).reindex(columns=fates, fill_value=0)


def _early(tab, mod):
    d2 = tab[(tab["day"] == "d2") & (tab["assay"] == mod)]
    return d2.assign(bc=d2["cell.bc"].str.replace(r"^d2_5-(rna|atac)-", "", regex=True))[["clone.id", "bc"]]


def _h5(mod):
    with h5py.File(H5[mod], "r") as f:
        m = f["matrix"]
        X = sp.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=tuple(m["shape"][:])).T.tocsr()
        bc = [b.decode() for b in m["barcodes"][:]]
    return X.astype(np.float32), pd.Index(bc)


# ---------- features (fit on train-clone + non-clonal cells only) ----------
class Featurizer:
    def __init__(self, mod):
        self.mod = mod

    def _prep(self, X):
        if self.mod == "rna":
            X = sp.diags(1e4 / np.maximum(np.asarray(X.sum(1)).ravel(), 1)) @ X
            X = X.log1p()
            return X[:, self.hvg]
        X = (X[:, self.top] > 0).astype(np.float32)  # amendment 4: top-50k peaks, Signac log1p(TF x IDF x 1e4)
        tf = sp.diags(1 / np.maximum(np.asarray(X.sum(1)).ravel(), 1)) @ X
        Y = tf.multiply(self.idf).tocsr() * 1e4
        Y.data = np.log1p(Y.data)
        return Y

    def fit(self, X):
        if self.mod == "rna":
            mu = np.asarray(X.mean(0)).ravel()
            var = np.asarray(X.multiply(X).mean(0)).ravel() - mu ** 2
            ok = mu > 0
            lm, lv = np.log10(mu[ok]), np.log10(np.maximum(var[ok], 1e-12))
            exp = 10 ** np.polyval(np.polyfit(lm, lv, 2), lm)
            n = X.shape[0]
            Z = (X[:, np.flatnonzero(ok)].toarray() - mu[ok]) / np.sqrt(exp)
            sv = np.clip(Z, None, np.sqrt(n)).var(0)
            self.hvg = np.flatnonzero(ok)[np.argsort(-sv)[:2000]]
            Y = self._prep(X).toarray()
            self.m, self.s = Y.mean(0), Y.std(0) + 1e-6
            self.red = PCA(NCOMP, random_state=0).fit((Y - self.m) / self.s)
        else:
            B = (X > 0).astype(np.float32)
            self.top = np.argsort(-np.asarray(B.sum(0)).ravel())[:50000]
            B = B[:, self.top]
            self.idf = np.log1p(B.shape[0] / (1 + np.asarray(B.sum(0)).ravel())).astype(np.float32)
            self.red = TruncatedSVD(NCOMP + 1, random_state=0).fit(self._prep(X))
        return self

    def transform(self, X):
        Y = self._prep(X)
        if self.mod == "rna":
            return self.red.transform((Y.toarray() - self.m) / self.s)
        return self.red.transform(Y)[:, 1:]


def clone_features(mod, keep_clones, fit_clones):
    """Mean day-2.5 feature per clone for keep_clones; featurizer fit on fit_clones' cells + non-clonal cells."""
    tab = _table()
    X, bc = _h5(mod)
    e = _early(tab, mod)
    all_clonal = set(tab.loc[tab["assay"] == mod, "cell.bc"].str.replace(r"^d2_5-(rna|atac)-", "", regex=True))
    fit_bc = set(e.loc[e["clone.id"].isin(fit_clones), "bc"]) | (set(bc) - all_clonal)
    fz = Featurizer(mod).fit(X[np.flatnonzero(bc.isin(list(fit_bc)))])
    # amendment 4: kNN smoothing (k 20, cosine) over ALL day-2.5 cells of the modality, in the fitted space
    from sklearn.neighbors import NearestNeighbors

    Fall = fz.transform(X)
    idx = NearestNeighbors(n_neighbors=20, metric="cosine").fit(Fall).kneighbors(Fall, return_distance=False)
    Fall = Fall[idx].mean(1)
    e = e[e["clone.id"].isin(keep_clones) & e["bc"].isin(bc)]
    F = Fall[bc.get_indexer(e["bc"])]
    df = pd.DataFrame(F, index=e["clone.id"].to_numpy()).groupby(level=0).mean()
    # amendment 2026-11-11: z-score clone features with fit-clone statistics (LSI scale ~1e-3 vs PCA ~1)
    ref = df.loc[df.index.isin(fit_clones)]
    df = (df - ref.mean()) / (ref.std() + 1e-9)
    return df, e.groupby("clone.id").size()


# ---------- model and score ----------
def _expand(F, K):
    rows, ys, ws = [], [], []
    n = K.sum(1)
    w = np.minimum(n, WCAP) / n
    for i in range(len(F)):
        for j in np.flatnonzero(K[i]):
            rows.append(F[i]); ys.append(j); ws.append(K[i, j] * w[i])
    return np.array(rows), np.array(ys), np.array(ws)


def fit_model(F, K, C):
    X, y, w = _expand(F, K)
    return LogisticRegression(C=C, max_iter=2000).fit(X, y, sample_weight=w)


def gains(model, F, K, base):
    P = np.full((len(F), len(FATES)), 1e-6)
    P[:, model.classes_] = model.predict_proba(F)
    P /= P.sum(1, keepdims=True)
    n = K.sum(1)
    g = ((K * (np.log2(P) - np.log2(np.maximum(base, 1e-6)))).sum(1)) / n
    return g, np.minimum(n, WCAP)


def wmean(g, w):
    return float((g * w).sum() / w.sum())


def _units(fates_tab, mod_sizes):
    k = fates_tab.reindex(mod_sizes.index).dropna()
    return k[k.sum(1) >= 2]


def cv_modality(mod, train_clones, K_all, rng):
    F, sizes = clone_features(mod, train_clones, train_clones)  # CV features: fit on train cells (unsupervised)
    K = _units(K_all, sizes)
    F = F.loc[K.index].to_numpy()
    Kn = K.to_numpy()
    folds = rng.integers(0, 5, len(K))
    res = {}
    for C in CS:
        g = np.zeros(len(K))
        for k in range(5):
            tr, te = folds != k, folds == k
            base = Kn[tr].sum(0) / Kn[tr].sum()
            g[te] = gains(fit_model(F[tr], Kn[tr], C), F[te], Kn[te], base)[0]
        res[C] = g
    w = np.minimum(Kn.sum(1), WCAP)
    best = max(CS, key=lambda c: wmean(res[c], w))
    return {"clones": K.index.to_numpy(), "g": res[best], "w": w, "C": best,
            "by_C": {str(c): wmean(res[c], w) for c in CS}, "n_early": sizes.loc[K.index].to_numpy()}


def delta_boot(r, a, rng, boot=1000):
    ids = np.union1d(r["clones"], a["clones"])
    ri, ai = pd.Series(range(len(r["clones"])), r["clones"]), pd.Series(range(len(a["clones"])), a["clones"])
    out = []
    for _ in range(boot):
        s = rng.choice(ids, len(ids))
        x, y = ri.reindex(s).dropna().astype(int).to_numpy(), ai.reindex(s).dropna().astype(int).to_numpy()
        out.append(wmean(a["g"][y], a["w"][y]) - wmean(r["g"][x], r["w"][x]))
    return np.percentile(out, [2.5, 97.5]).tolist()


def summarize(r, a, rng):
    G_r, G_a = wmean(r["g"], r["w"]), wmean(a["g"], a["w"])
    paired = np.intersect1d(r["clones"], a["clones"])
    pr = pd.Series(r["g"], r["clones"]).loc[paired]
    pa = pd.Series(a["g"], a["clones"]).loc[paired]
    strat = {}
    for name, x in (("rna", r), ("atac", a)):
        for lab, m in (("1_early", x["n_early"] == 1), ("2plus_early", x["n_early"] >= 2)):
            if m.sum() >= 5:
                strat[f"{name}_{lab}"] = {"n": int(m.sum()), "G": wmean(x["g"][m], x["w"][m])}
    return {"n_rna": int(len(r["g"])), "n_atac": int(len(a["g"])), "G_rna": G_r, "G_atac": G_a,
            "delta": G_a - G_r, "delta_ci": delta_boot(r, a, rng),
            "G_rna_ci": np.percentile([wmean(r["g"][ix], r["w"][ix]) for ix in (rng.integers(0, len(r["g"]), len(r["g"])) for _ in range(1000))], [2.5, 97.5]).tolist(),
            "G_atac_ci": np.percentile([wmean(a["g"][ix], a["w"][ix]) for ix in (rng.integers(0, len(a["g"]), len(a["g"])) for _ in range(1000))], [2.5, 97.5]).tolist(),
            "paired": {"n": int(len(paired)), "delta": float((pa - pr).mean()) if len(paired) else None},
            "stratified": strat}


def train(select=False):
    rng = np.random.default_rng(73)
    tab = partitions.train_only("celltag_lsk", _table(), unit_col="clone.id")
    train_clones = tab["clone.id"].unique()
    K_all = _fates(tab)
    r = cv_modality("rna", train_clones, K_all, rng)
    a = cv_modality("atac", train_clones, K_all, rng)
    res = {"C_rna": r["C"], "C_atac": a["C"], "by_C_rna": r["by_C"], "by_C_atac": a["by_C"], **summarize(r, a, rng),
           "sd_g_rna": float(np.std(r["g"])), "sd_g_atac": float(np.std(a["g"])),
           "neff_ratio_rna": float(r["w"].sum() ** 2 / (r["w"] ** 2).sum() / len(r["w"])),
           "neff_ratio_atac": float(a["w"].sum() ** 2 / (a["w"] ** 2).sum() / len(a["w"]))}
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "c7s3_train.json"), "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))
    if select:
        sd_pool = float(np.sqrt((res["sd_g_rna"] ** 2 + res["sd_g_atac"] ** 2) / 2))
        sel = {"test": "celltag_c7s3", "higher_is_better": True,
               "candidates": {"rna": res["G_rna"], "atac": res["G_atac"]},
               "chosen": "atac" if res["G_atac"] >= res["G_rna"] else "rna",
               "C": {"rna": r["C"], "atac": a["C"]},
               "power": {"sd": sd_pool, "effect": abs(res["delta"]), "alpha": 0.05, "one_sided": False,
                         "min_power": 0.8},
               "train_counts": {"rna": res["n_rna"], "atac": res["n_atac"]},
               "neff_ratio": {"rna": res["neff_ratio_rna"], "atac": res["neff_ratio_atac"]},
               "note": "C7-S3: Delta = G_atac - G_rna (bits/cell); counts(part) = effective n per modality"}
        json.dump(sel, open(os.path.join(HERE, "selection", "C7-S3.json"), "w"), indent=1, default=float)
        print(json.dumps(sel, indent=1, default=float))
    return res


def counts(part):
    """Effective units per modality in a sealed partition (structure only: no fate column read). Structural count
    of clones with an early sibling and >= 2 day-5 cells, scaled by the TRAIN ratios (eligible after fate-label
    exclusions / structural, and effective-n / n from the weights)."""
    sel = json.load(open(os.path.join(HERE, "selection", "C7-S3.json")))
    tab = pd.read_csv(CSV, usecols=["clone.id", "cell.bc", "day", "assay"])
    trn = partitions.train_only("celltag_lsk", tab, unit_col="clone.id")
    out = {}
    for mod in ("rna", "atac"):
        def struct(t):
            early = set(t.loc[(t["day"] == "d2") & (t["assay"] == mod), "clone.id"])
            d5 = t[t["day"] == "d5"].groupby("clone.id").size()
            return len(early & set(d5[d5 >= 2].index))
        ratio = sel["train_counts"][mod] / struct(trn)
        n = struct(partitions.sealed_rows("celltag_lsk", tab, "clone.id", part))
        out[mod] = int(np.floor(n * ratio * sel["neff_ratio"][mod]))
    return out


def test(part):
    """Sealed look (oracle only): featurizers and models fitted on all train clones, applied to the partition."""
    rng = np.random.default_rng(74)
    sel = json.load(open(os.path.join(HERE, "selection", "C7-S3.json")))
    full = _table()
    trn = partitions.train_only("celltag_lsk", full, unit_col="clone.id")
    held = partitions.sealed_rows("celltag_lsk", full, "clone.id", part)
    train_clones, held_clones = trn["clone.id"].unique(), held["clone.id"].unique()
    Kt, Kh = _fates(trn), _fates(held)
    out = {}
    for mod in ("rna", "atac"):
        F, sizes = clone_features(mod, np.r_[train_clones, held_clones], train_clones)
        Ktr = _units(Kt, sizes.reindex(train_clones).dropna())
        Khe = _units(Kh, sizes.reindex(held_clones).dropna())
        model = fit_model(F.loc[Ktr.index].to_numpy(), Ktr.to_numpy(), sel["C"][mod])
        base = Ktr.to_numpy().sum(0) / Ktr.to_numpy().sum()
        g, w = gains(model, F.loc[Khe.index].to_numpy(), Khe.to_numpy(), base)
        out[mod] = {"clones": Khe.index.to_numpy(), "g": g, "w": w, "n_early": sizes.loc[Khe.index].to_numpy()}
    s = summarize(out["rna"], out["atac"], rng)
    lo, hi = s["delta_ci"]
    s["reading"] = ("chromatin carries more" if lo > 0 else "RNA carries more" if hi < 0 else
                    "no difference detectable")
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true")
    train(ap.parse_args().select)
