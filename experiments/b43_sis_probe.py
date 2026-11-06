"""B43: what carries the SIS-seq Geneformer V2 signal? Descriptive, fit clones only.

    uv run python b43_sis_probe.py
"""

from __future__ import annotations

import json
import os

import numpy as np

import core_sisseq as cs
import e55_fm as e
import sis_data as sd

OUT = e.OUT
PDC = ["Siglech", "Bst2", "Ly6d", "Ccr9", "Cox6a2", "Klk1", "Ly6c2", "Tcf4", "Spib", "Runx2"]
CDC2 = ["Sirpa", "Itgam", "Irf4", "Klf4", "Esam", "Clec4a4"]


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def pca_loo_perbatch(C, batch, L, k=3):
    """E54 recipe (per-batch z, top-2000 variance, PCA-3, OLS) with everything refitted inside LOO."""
    det = (C > 0).mean(0) >= 0.10
    E = np.log1p(C / C.sum(1, keepdims=True) * 1e4)[:, det]
    n = len(L)
    pred = np.zeros_like(L)
    for i in range(n):
        tr = np.arange(n) != i
        Z = np.zeros((n, E.shape[1]))
        for b in np.unique(batch):
            m = (batch == b) & tr
            mu, sdv = E[m].mean(0), E[m].std(0)
            sdv = np.where(sdv > 0, sdv, 1.0)
            Z[batch == b] = (E[batch == b] - mu) / sdv
        top = np.argsort(Z[tr].var(0))[::-1][:2000]
        cmu = Z[tr][:, top].mean(0)
        _, _, vt = np.linalg.svd(Z[tr][:, top] - cmu, full_matrices=False)
        P = (Z[:, top] - cmu) @ vt[:k].T
        X1 = np.c_[np.ones(n), P]
        pred[i] = X1[i] @ np.linalg.lstsq(X1[tr], L[tr], rcond=None)[0]
    return pred


def gene_select_loo(G, L, k):
    n = len(L)
    pred = np.zeros_like(L)
    for i in range(n):
        tr = np.arange(n) != i
        Gt = G[tr]
        Gc = (Gt - Gt.mean(0)) / (Gt.std(0) + 1e-9)
        for j in range(L.shape[1]):
            y = L[tr, j]
            c = np.abs(Gc.T @ ((y - y.mean()) / (y.std() + 1e-9))) / len(y)
            top = np.argsort(c)[::-1][:k]
            pred[i, j] = e.nested_ridge(Gt[:, top], y, G[i:i + 1, top])[0]
    return pred


def main():
    C, genes, F, L = e.sis_data()
    C = C.toarray()
    genes = np.array(genes)
    res = {"n": int(len(L))}

    # 1. baselines
    _, r54 = cs.scores("fit")
    res["E54_recipe_pca_on_all"] = {f: corr(r54[:, j], L[:, j]) for j, f in enumerate(sd.FATES)}
    p = pca_loo_perbatch(C, F["batch"], L)
    res["E54_recipe_pca_inside_loo"] = {f: corr(p[:, j], L[:, j]) for j, f in enumerate(sd.FATES)}

    # 2. plain-gene ridge with selection inside LOO
    det = (C > 0).mean(0) >= 0.10
    G = e._batch_centre(np.log1p(C / C.sum(1, keepdims=True) * 1e4)[:, det], F["batch"])
    gn = genes[det]
    for k in (20, 50, 200):
        p = gene_select_loo(G, L, k)
        res[f"genes_top{k}_inside_loo"] = {f: corr(p[:, j], L[:, j]) for j, f in enumerate(sd.FATES)}
        print(k, res[f"genes_top{k}_inside_loo"], flush=True)

    # 3. attribution of the V2 LOO predictions
    A = e._batch_centre(np.load(os.path.join(OUT, "sis_emb_V2-316M.npz"))["emb"].astype(float), F["batch"])
    n = len(L)
    Gs = (G - G.mean(0)) / (G.std(0) + 1e-9)
    for j, f in ((2, "pDC"), (1, "cDC2")):
        pv = np.array([e.nested_ridge(A[np.arange(n) != i], L[np.arange(n) != i, j], A[i:i + 1])[0]
                       for i in range(n)])
        pz = (pv - pv.mean()) / pv.std()
        c = Gs.T @ pz / n
        top = np.argsort(c)[::-1][:30]
        bot = np.argsort(c)[:15]
        marks = PDC if f == "pDC" else CDC2
        res[f"V2_{f}_r"] = corr(pv, L[:, j])
        res[f"V2_{f}_top30"] = [(str(gn[t]), round(float(c[t]), 2)) for t in top]
        res[f"V2_{f}_bottom15"] = [(str(gn[t]), round(float(c[t]), 2)) for t in bot]
        res[f"V2_{f}_markers_in_top30"] = [g for g in gn[top] if g in marks]
        res[f"{f}_marker_corr_with_pred"] = {g: round(float(c[list(gn).index(g)]), 2) for g in marks if g in gn}
        res[f"{f}_marker_corr_with_fate"] = {g: round(corr(G[:, list(gn).index(g)], L[:, j]), 2)
                                             for g in marks if g in gn}
    json.dump(res, open(os.path.join(OUT, "b43_probe.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))



def raw_logits(k):
    n = k.sum(1, keepdims=True)
    return np.log((k + 0.5) / (n - k + 0.5))


def loo_clean(X, Lraw, batch, fit_predict):
    """LOO with batch centring of X and y from training clones only (B43 amendment)."""
    n = len(Lraw)
    pred, yc = np.zeros_like(Lraw), np.zeros_like(Lraw)
    for i in range(n):
        tr = np.arange(n) != i
        Xc, Yc = X.astype(float).copy(), Lraw.copy()
        for b in np.unique(batch):
            m = batch == b
            Xc[m] -= X[m & tr].mean(0)
            Yc[m] -= Lraw[m & tr].mean(0)
        yc[i] = Yc[i]
        pred[i] = fit_predict(Xc[tr], Yc[tr], Xc[i:i + 1])
    return pred, yc


def fp_ridge(A, Y, B):
    return np.array([e.nested_ridge(A, Y[:, j], B)[0] for j in range(Y.shape[1])])


def fp_genes(k):
    def f(A, Y, B):
        out = []
        Ac = (A - A.mean(0)) / (A.std(0) + 1e-9)
        for j in range(Y.shape[1]):
            y = Y[:, j]
            c = np.abs(Ac.T @ ((y - y.mean()) / (y.std() + 1e-9))) / len(y)
            top = np.argsort(c)[::-1][:k]
            out.append(e.nested_ridge(A[:, top], y, B[:, top])[0])
        return np.array(out)
    return f


def fp_pca3(A, Y, B):
    sd_ = A.std(0) + 1e-9
    Z, zb = A / sd_, B / sd_
    top = np.argsort(Z.var(0))[::-1][:2000]
    mu = Z[:, top].mean(0)
    _, _, vt = np.linalg.svd(Z[:, top] - mu, full_matrices=False)
    P, pb = (Z[:, top] - mu) @ vt[:3].T, (zb[:, top] - mu) @ vt[:3].T
    W = np.linalg.lstsq(np.c_[np.ones(len(P)), P], Y, rcond=None)[0]
    return (np.c_[np.ones(1), pb] @ W)[0]


def clean():
    C, genes, F, _ = e.sis_data()
    C = C.toarray()
    Lraw = raw_logits(F["k1"] + F["k2"])
    det = (C > 0).mean(0) >= 0.10
    G = np.log1p(C / C.sum(1, keepdims=True) * 1e4)[:, det]
    feats = {"pca3": (G, fp_pca3), "genes_top50": (G, fp_genes(50)), "genes_top200": (G, fp_genes(200))}
    for m in e.MODELS:
        feats[f"gf_{m}"] = (np.load(os.path.join(OUT, f"sis_emb_{m}.npz"))["emb"].astype(float), fp_ridge)
    res = {"n": int(len(Lraw)), "note": "B43 amendment: batch centring inside LOO"}
    rng = np.random.default_rng(43)
    for name, (X, fp) in feats.items():
        p, yc = loo_clean(X, Lraw, F["batch"], fp)
        res[name] = {f: corr(p[:, j], yc[:, j]) for j, f in enumerate(sd.FATES)}
        if name in ("genes_top200", "gf_V2-316M"):
            sh = []
            for _ in range(3):
                y = Lraw.copy()
                for b in np.unique(F["batch"]):
                    m = np.where(F["batch"] == b)[0]
                    y[m] = Lraw[rng.permutation(m)]
                ps, ys = loo_clean(X, y, F["batch"], fp)
                sh.append([corr(ps[:, j], ys[:, j]) for j in range(3)])
            res[name + "_shuffled"] = sh
        print(name, res[name], res.get(name + "_shuffled"), flush=True)
    json.dump(res, open(os.path.join(OUT, "b43_clean.json"), "w"), indent=1)


if __name__ == "__main__":
    import sys
    clean() if sys.argv[1:] == ["clean"] else main()
