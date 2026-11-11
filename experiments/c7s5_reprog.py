"""C7-S5: early chromatin vs early RNA in reprogramming (CellTag-multi iEP, day 3 -> day 12/21). TRAIN ONLY.
See EXPERIMENT_LOG C7-S5. Reads the celltag_reprog TRAIN partition only; no sealed rows are read.

    python c7s5_reprog.py
"""

from __future__ import annotations

import glob
import json
import os
import re

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

import partitions
from c7s3_carrier import NCOMP, WCAP, Featurizer, fit_model, wmean

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "ext_celltag_multi")
CSV = os.path.join(DATA, "b4.rna_atac.r1_2_master_rev1_annotated.csv")
GEO = os.path.join(DATA, "geo_reprog")
OUT = os.path.join(HERE, "results_celltag")
FATES = ["reprogramming", "transition", "dead-end"]
CS = (0.001, 0.01, 0.1, 1.0, 10.0)
NONCLONAL = 20000
BIN = 5000
SCALE_A = 1561 / 3208  # partition A / train clone ratio (structure only; no sealed rows read)


# ---------- data ----------
def _table():
    t = pd.read_csv(CSV, usecols=["clone_id", "cell.bc", "day", "assay", "cell_type"])
    return partitions.train_only("celltag_reprog", t, unit_col="clone_id")


def _fates(tab):
    f = tab[tab["day"].isin(["D12", "D21"]) & tab["cell_type"].isin(FATES)]
    return f.groupby(["clone_id", "cell_type"]).size().unstack(fill_value=0).reindex(columns=FATES, fill_value=0)


def _early(tab, mod):
    d3 = tab[(tab["day"] == "D3") & (tab["assay"] == mod.upper())]
    m = d3["cell.bc"].str.extract(r"^B4D3-(?:RNA|ATAC)-(r\d-\d+)-(.+)$")
    return pd.DataFrame({"clone_id": d3["clone_id"].to_numpy(), "bc": (m[0] + ":" + m[1]).to_numpy()})


def _lane_files(mod):
    out = {}
    for f in sorted(glob.glob(os.path.join(GEO, f"*_B4D3-{mod.upper()}-*.h5"))):
        out[re.search(r"B4D3-(?:RNA|ATAC)-(r\d-\d+)_", os.path.basename(f)).group(1)] = f
    return out


def _read(f):
    with h5py.File(f, "r") as h:
        m = h["matrix"]
        X = sp.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=tuple(m["shape"][:])).T.tocsr()
        bc = [b.decode() for b in m["barcodes"][:]]
        ids = [b.decode() for b in m["features"]["id"][:]]
    return X.astype(np.float32), bc, ids


def _bins(ids):
    out = []
    for p in ids:
        c, a, b = re.split(r"[:-]", p)
        out.append(f"{c}:{(int(a) + int(b)) // 2 // BIN}")
    return out


def load_universe(mod, clonal_bc, rng):
    """All clonal day-3 cells + up to NONCLONAL non-clonal day-3 cells (seed 75), over all lanes of the modality.
    ATAC peaks are mapped to 5 kb genome bins (peak midpoint)."""
    mats, bcs, vocab = [], [], {}
    for lane, f in _lane_files(mod).items():
        X, bc, ids = _read(f)
        if mod == "atac":
            b = _bins(ids)
            col = np.array([vocab.setdefault(x, len(vocab)) for x in b])
            M = sp.csr_matrix((np.ones(len(col), np.float32), (np.arange(len(col)), col)), shape=(len(col), 10 ** 7))
            X = X @ M
        else:
            if not vocab:
                vocab = {g: i for i, g in enumerate(ids)}
            assert len(ids) == len(vocab) and ids[0] in vocab, "RNA lanes differ in genes"
        mats.append(X)
        bcs += [f"{lane}:{x}" for x in bc]
    n_feat = len(vocab)
    X = sp.vstack([m[:, :n_feat] for m in mats]).tocsr()
    bc = pd.Index(bcs)
    clonal = bc.isin(list(clonal_bc))
    non = np.flatnonzero(~clonal)
    keep = np.sort(np.r_[np.flatnonzero(clonal), rng.choice(non, min(NONCLONAL, len(non)), replace=False)])
    return X[keep], bc[keep], {"cells_total": int(len(bc)), "clonal_in_matrix": int(clonal.sum()), "features": n_feat}


class ChunkedFeaturizer(Featurizer):
    """Same as C7-S3 Featurizer; the RNA HVG variance is computed in gene chunks (identical result, less RAM)."""

    def fit(self, X):
        if self.mod != "rna":
            return super().fit(X)
        mu = np.asarray(X.mean(0)).ravel()
        var = np.asarray(X.multiply(X).mean(0)).ravel() - mu ** 2
        ok = np.flatnonzero(mu > 0)
        lm, lv = np.log10(mu[ok]), np.log10(np.maximum(var[ok], 1e-12))
        exp = 10 ** np.polyval(np.polyfit(lm, lv, 2), lm)
        n = X.shape[0]
        Xc = X.tocsc()
        sv = np.empty(len(ok))
        for s in range(0, len(ok), 1000):
            j = ok[s:s + 1000]
            Z = (Xc[:, j].toarray().astype(np.float64) - mu[j]) / np.sqrt(exp[s:s + 1000])
            sv[s:s + 1000] = np.clip(Z, None, np.sqrt(n)).var(0)
        self.hvg = ok[np.argsort(-sv)[:2000]]
        Y = self._prep(X).toarray()
        self.m, self.s = Y.mean(0), Y.std(0) + 1e-6
        self.red = PCA(NCOMP, random_state=0).fit((Y - self.m) / self.s)
        return self


def features(mod, tab, rng):
    """Per-cell smoothed features and per-clone z-scored means (train clones only; all fit cells are train)."""
    e = _early(tab, mod)
    X, bc, info = load_universe(mod, set(e["bc"]), rng)
    fz = ChunkedFeaturizer(mod).fit(X)  # fit cells = train clonal + sampled non-clonal (no sealed cells loaded)
    F = fz.transform(X)
    idx = NearestNeighbors(n_neighbors=20, metric="cosine").fit(F).kneighbors(F, return_distance=False)
    F = F[idx].mean(1)
    e = e[e["bc"].isin(bc)]
    Fc = F[bc.get_indexer(e["bc"])]
    df = pd.DataFrame(Fc, index=e["clone_id"].to_numpy()).groupby(level=0).mean()
    df = (df - df.mean()) / (df.std() + 1e-9)
    info["clonal_cells_used"] = int(len(e))
    return df, e.groupby("clone_id").size(), (Fc, e["clone_id"].to_numpy()), info


def sibling_auc(cells, rng, n_rand=20000):
    """Fate-free validity: cosine similarity of sibling pairs vs random cross-clone pairs (AUC)."""
    F, cl = cells
    Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)
    sib = []
    for c, ix in pd.Series(range(len(cl))).groupby(cl):
        ix = ix.to_numpy()
        if len(ix) >= 2:
            ix = ix[:20]
            s = Fn[ix] @ Fn[ix].T
            sib += list(s[np.triu_indices(len(ix), 1)])
    a, b = rng.integers(0, len(cl), n_rand), rng.integers(0, len(cl), n_rand)
    m = cl[a] != cl[b]
    rnd = (Fn[a[m]] * Fn[b[m]]).sum(1)
    sib = np.array(sib)
    if len(sib) == 0:
        return {"n_sib_pairs": 0, "auc": None}
    ranks = pd.Series(np.r_[sib, rnd]).rank().to_numpy()
    auc = (ranks[:len(sib)].sum() - len(sib) * (len(sib) + 1) / 2) / (len(sib) * len(rnd))
    return {"n_sib_pairs": int(len(sib)), "auc": float(auc), "blind": bool(auc < 0.55)}


def gains(model, F, K, base):
    P = np.full((len(F), len(FATES)), 1e-6)
    P[:, model.classes_] = model.predict_proba(F)
    P /= P.sum(1, keepdims=True)
    n = K.sum(1)
    g = (K * (np.log2(P) - np.log2(np.maximum(base, 1e-6)))).sum(1) / n
    return g, np.minimum(n, WCAP)


def cv(F, sizes, K_all, rng):
    K = K_all.reindex(sizes.index).dropna()
    K = K[K.sum(1) >= 2]
    Fm, Kn = F.loc[K.index].to_numpy(), K.to_numpy()
    folds = rng.integers(0, 5, len(K))
    res = {}
    for C in CS:
        g = np.zeros(len(K))
        for k in range(5):
            tr, te = folds != k, folds == k
            base = Kn[tr].sum(0) / Kn[tr].sum()
            g[te] = gains(fit_model(Fm[tr], Kn[tr], C), Fm[te], Kn[te], base)[0]
        res[C] = g
    w = np.minimum(Kn.sum(1), WCAP)
    best = max(CS, key=lambda c: wmean(res[c], w))
    return {"clones": K.index.to_numpy(), "g": res[best], "w": w, "C": best,
            "by_C": {str(c): wmean(res[c], w) for c in CS}, "n_early": sizes.loc[K.index].to_numpy()}


def boot_ci(x, rng, f, boot=1000):
    n = len(x["g"])
    return np.percentile([f(rng.integers(0, n, n)) for _ in range(boot)], [2.5, 97.5]).tolist()


def summarize(r, a, rng):
    G_r, G_a = wmean(r["g"], r["w"]), wmean(a["g"], a["w"])
    ids = np.union1d(r["clones"], a["clones"])
    ri, ai = pd.Series(range(len(r["clones"])), r["clones"]), pd.Series(range(len(a["clones"])), a["clones"])
    d = []
    for _ in range(1000):
        s = rng.choice(ids, len(ids))
        x, y = ri.reindex(s).dropna().astype(int).to_numpy(), ai.reindex(s).dropna().astype(int).to_numpy()
        d.append(wmean(a["g"][y], a["w"][y]) - wmean(r["g"][x], r["w"][x]))
    paired = np.intersect1d(r["clones"], a["clones"])
    pr, pa = pd.Series(r["g"], r["clones"]).loc[paired], pd.Series(a["g"], a["clones"]).loc[paired]
    strat = {}
    for name, x in (("rna", r), ("atac", a)):
        for lab, m in (("1_early", x["n_early"] == 1), ("2plus_early", x["n_early"] >= 2)):
            if m.sum() >= 5:
                strat[f"{name}_{lab}"] = {"n": int(m.sum()), "G": wmean(x["g"][m], x["w"][m])}
    return {"n_rna": int(len(r["g"])), "n_atac": int(len(a["g"])), "G_rna": G_r, "G_atac": G_a,
            "G_rna_ci": boot_ci(r, rng, lambda ix: wmean(r["g"][ix], r["w"][ix])),
            "G_atac_ci": boot_ci(a, rng, lambda ix: wmean(a["g"][ix], a["w"][ix])),
            "delta": G_a - G_r, "delta_ci": np.percentile(d, [2.5, 97.5]).tolist(),
            "paired": {"n": int(len(paired)), "delta": float((pa - pr).mean()) if len(paired) else None},
            "stratified": strat}


def power_block(r, a, s):
    z2, z1 = norm.ppf(0.975), norm.ppf(0.975)
    out = {}
    neff = {}
    for name, x in (("rna", r), ("atac", a)):
        ratio = x["w"].sum() ** 2 / (x["w"] ** 2).sum() / len(x["w"])
        neff[name] = len(x["g"]) * SCALE_A * ratio
        sd = float(np.std(x["g"]))
        G = s[f"G_{name}"]
        out[f"G_{name}>0_on_A"] = float(1 - norm.cdf(z1 - G / (sd / np.sqrt(neff[name]))))
    sd_r, sd_a = float(np.std(r["g"])), float(np.std(a["g"]))
    se = np.sqrt(sd_r ** 2 / neff["rna"] + sd_a ** 2 / neff["atac"])
    out["delta_on_A"] = float(1 - norm.cdf(z2 - abs(s["delta"]) / se))
    out["neff_A"] = neff
    return out


def main():
    rng = np.random.default_rng(75)
    tab = _table()
    K_all = _fates(tab)
    res, mods = {"fates": FATES}, {}
    for mod in ("rna", "atac"):
        F, sizes, cells, info = features(mod, tab, rng)
        res[f"validity_{mod}"] = sibling_auc(cells, rng)
        res[f"universe_{mod}"] = info
        print(mod, info, res[f"validity_{mod}"], flush=True)
        mods[mod] = (F, sizes)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "c7s5_validity.json"), "w"), indent=1, default=float)
    r = cv(*mods["rna"], K_all, rng)
    a = cv(*mods["atac"], K_all, rng)
    s = summarize(r, a, rng)
    res.update({"C_rna": r["C"], "C_atac": a["C"], "by_C_rna": r["by_C"], "by_C_atac": a["by_C"], **s,
                "power": power_block(r, a, s),
                "train_fate_freq": (K_all.sum(0) / K_all.to_numpy().sum()).to_dict()})
    json.dump(res, open(os.path.join(OUT, "c7s5_train.json"), "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
