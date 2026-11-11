"""C7-S6: human HSPC replication (cHSPCTrack). Does early (d3-5) RNA carry the clone's day-10 fate, and how much
of the clone's serial reference (d7 composition -> d10) does it cover? TRAIN ONLY. See EXPERIMENT_LOG C7-S6.

    python c7s6_hspc.py
"""

from __future__ import annotations

import hashlib
import json
import os

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

import partitions
from c7s3_carrier import WCAP, fit_model, wmean
from c7s5_reprog import ChunkedFeaturizer, sibling_auc

HERE = os.path.dirname(os.path.abspath(__file__))
H5AD = os.path.join(HERE, "..", "data", "ext_chspctrack", "cells_with_clone_info.h5ad")
OUT = os.path.join(HERE, "results_hspc")
G8 = {"Neutrophils": "Neu", "Monocyte": "Mono", "Ery": "Ery", "MK": "MK", "Basophils": "Baso",
      "Eosinophil": "Eos", "Mast": "Mast", "GMP": "Prog", "MEBEMP": "Prog", "MEP": "Prog", "BEMP": "Prog",
      "BEP": "Prog"}
FATES = ["Neu", "Mono", "Ery", "MK", "Baso", "Eos", "Mast", "Prog"]
CS = (0.001, 0.01, 0.1, 1.0, 10.0)


def _fold(c):
    return int(hashlib.sha256(f"c7s6-{c}".encode()).hexdigest(), 16) % 5


def _cat(o, k):
    g = o[k]
    c = np.array([x.decode() if isinstance(x, bytes) else x for x in g["categories"][:]], dtype=object)
    codes = g["codes"][:]
    return np.where(codes >= 0, c[np.maximum(codes, 0)], None)


def load():
    with h5py.File(H5AD, "r") as f:
        o = f["obs"]
        obs = pd.DataFrame({"clone": _cat(o, "clone_id"), "day": o["sample_day_as_int"][:],
                            "cond": _cat(o, "condition"), "ct": _cat(o, "top_level_cell_type")})
    obs["row"] = np.arange(len(obs))
    clonal = obs[obs["clone"].notna()]
    train = partitions.train_only("chspctrack", clonal, unit_col="clone")
    nonclonal_early = obs[obs["clone"].isna() & (obs["day"] <= 5)]
    return train, nonclonal_early


def read_rows(rows):
    rows = np.sort(rows)
    with h5py.File(H5AD, "r") as f:
        X = f["X"]
        ip = X["indptr"][:]
        shape = tuple(X.attrs["shape"])
        data, ind, ptr = [], [], [0]
        for r in rows:
            a, b = ip[r], ip[r + 1]
            data.append(X["data"][a:b]); ind.append(X["indices"][a:b]); ptr.append(ptr[-1] + (b - a))
    M = sp.csr_matrix((np.concatenate(data), np.concatenate(ind), np.array(ptr)), shape=(len(rows), shape[1]))
    return M.astype(np.float32), rows


def fate_counts(train, day):
    d = train[(train["day"] == day)].assign(g=lambda x: x["ct"].map(G8))
    d = d[d["g"].notna()]
    return d.groupby(["clone", "g"]).size().unstack(fill_value=0).reindex(columns=FATES, fill_value=0)


def cond_onehot(train, clones, levels):
    c = train.groupby("clone")["cond"].first().reindex(clones).fillna("unknown")
    return np.array([[float(x == l) for l in levels] for x in c]), c.to_numpy()


def cv_gain(F, K, fold, base, w_cap=WCAP):
    """5-fold CV (given folds) of the multinomial model; returns per-clone gain for each C."""
    out = {}
    for C in CS:
        g = np.zeros(len(K))
        for k in range(5):
            tr, te = fold != k, fold == k
            m = fit_model(F[tr], K[tr], C)
            P = np.full((te.sum(), len(FATES)), 1e-6)
            P[:, m.classes_] = m.predict_proba(F[te])
            P /= P.sum(1, keepdims=True)
            g[te] = (K[te] * (np.log2(P) - np.log2(np.maximum(base[te], 1e-9)))).sum(1) / K[te].sum(1)
        out[C] = g
    return out


def fold_bases(K10, conds, fold):
    """Per clone: day-10 G8 frequencies of its condition over clones outside its fold."""
    B = np.zeros(K10.shape)
    for k in range(5):
        tr = fold != k
        for c in np.unique(conds):
            s = K10[tr & (conds == c)].sum(0)
            s = s if s.sum() > 0 else K10[tr].sum(0)
            B[(fold == k) & (conds == c)] = (s + 0.5) / (s + 0.5).sum()
    return B


def boot(g, w, rng, n=1000):
    return np.percentile([wmean(g[ix], w[ix]) for ix in (rng.integers(0, len(g), len(g)) for _ in range(n))],
                         [2.5, 97.5]).tolist()


def main():
    rng = np.random.default_rng(77)
    train, non = load()
    levels = sorted(train["cond"].fillna("unknown").unique())
    # ---- early features (universe = train-clone early cells + non-clonal early cells) ----
    early = train[train["day"] <= 5]
    X, rows = read_rows(np.r_[early["row"].to_numpy(), non["row"].to_numpy()])
    fz = ChunkedFeaturizer("rna").fit(X)
    Fz = fz.transform(X)
    idx = NearestNeighbors(n_neighbors=20, metric="cosine").fit(Fz).kneighbors(Fz, return_distance=False)
    Fz = Fz[idx].mean(1)
    pos = pd.Series(np.arange(len(rows)), index=rows)
    Fe = Fz[pos.loc[early["row"].to_numpy()].to_numpy()]
    res = {"validity": sibling_auc((Fe, early["clone"].to_numpy()), rng),
           "universe": {"early_train_clonal": int(len(early)), "early_nonclonal": int(len(non))}}
    print("validity", res["validity"], flush=True)
    clone_F = pd.DataFrame(Fe, index=early["clone"].to_numpy()).groupby(level=0).mean()
    clone_F = (clone_F - clone_F.mean()) / (clone_F.std() + 1e-9)
    # ---- units ----
    K10, K7 = fate_counts(train, 10), fate_counts(train, 7)
    u1 = [c for c in K10.index[K10.sum(1) >= 2] if c in clone_F.index]
    ser = [c for c in K10.index[K10.sum(1) >= 2] if c in K7.index and K7.loc[c].sum() >= 2]
    u2 = [c for c in u1 if c in set(ser)]
    res["units"] = {"U1": len(u1), "serial_pool": len(ser), "U2": len(u2)}
    print("units", res["units"], flush=True)
    # ---- G_early on U1 ----
    k1 = K10.loc[u1].to_numpy(); f1 = np.array([_fold(c) for c in u1])
    oh1, cd1 = cond_onehot(train, u1, levels)
    B1 = fold_bases(k1, cd1, f1)
    Fm = np.c_[clone_F.loc[u1].to_numpy(), oh1]
    w1 = np.minimum(k1.sum(1), WCAP)
    ge_all = cv_gain(Fm, k1, f1, B1)
    Ce = max(CS, key=lambda c: wmean(ge_all[c], w1))
    ge = ge_all[Ce]
    null_e = []
    for _ in range(5):
        P = clone_F.loc[u1].to_numpy().copy()
        for c in np.unique(cd1):
            ix = np.flatnonzero(cd1 == c); P[ix] = P[rng.permutation(ix)]
        null_e.append(wmean(cv_gain(np.c_[P, oh1], k1, f1, B1)[Ce], w1))
    lo = boot(ge, w1, rng)
    res["G_early_U1"] = {"C": Ce, "by_C": {str(c): wmean(ge_all[c], w1) for c in CS}, "G": wmean(ge, w1), "ci": lo,
                         "null_mean": float(np.mean(null_e)), "null_ok": bool(np.mean(null_e) <= 0.02),
                         "reading": "early RNA carries clone fate (human HSPC)" if lo[0] > 0 else "not detected"}
    print("G_early_U1", res["G_early_U1"], flush=True)
    # ---- G_serial on serial pool ----
    ks = K10.loc[ser].to_numpy(); fs = np.array([_fold(c) for c in ser])
    ohs, cds = cond_onehot(train, ser, levels)
    Bs = fold_bases(ks, cds, fs)
    k7 = K7.loc[ser].to_numpy()
    Fs = np.c_[k7 / k7.sum(1, keepdims=True), np.log(k7.sum(1)), ohs]
    ws = np.minimum(ks.sum(1), WCAP)
    gs_all = cv_gain(Fs, ks, fs, Bs)
    Cs = max(CS, key=lambda c: wmean(gs_all[c], ws))
    gs = gs_all[Cs]
    null_s = []
    for _ in range(5):
        P = k7.copy()
        for c in np.unique(cds):
            ix = np.flatnonzero(cds == c); P[ix] = P[rng.permutation(ix)]
        null_s.append(wmean(cv_gain(np.c_[P / P.sum(1, keepdims=True), np.log(P.sum(1)), ohs], ks, fs, Bs)[Cs], ws))
    res["G_serial_pool"] = {"C": Cs, "by_C": {str(c): wmean(gs_all[c], ws) for c in CS}, "G": wmean(gs, ws),
                            "null_mean": float(np.mean(null_s)), "null_ok": bool(np.mean(null_s) <= 0.02)}
    # ---- coverage on U2 ----
    e_s = pd.Series(ge, index=u1).loc[u2].to_numpy()
    s_s = pd.Series(gs, index=ser).loc[u2].to_numpy()
    w2 = pd.Series(w1, index=u1).loc[u2].to_numpy()
    cov = wmean(e_s, w2) / wmean(s_s, w2)
    bs = []
    for _ in range(1000):
        ix = rng.integers(0, len(u2), len(u2))
        bs.append(wmean(e_s[ix], w2[ix]) / wmean(s_s[ix], w2[ix]))
    ci = np.percentile(bs, [2.5, 97.5]).tolist()
    res["coverage_U2"] = {"n": len(u2), "G_early": wmean(e_s, w2), "G_early_ci": boot(e_s, w2, rng),
                          "G_serial": wmean(s_s, w2), "G_serial_ci": boot(s_s, w2, rng), "coverage": cov,
                          "ci": ci, "reading": "most" if ci[0] > 0.5 else "minority" if ci[1] < 0.5 else "unresolved"}
    print("coverage", res["coverage_U2"], flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "c7s6_train.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
