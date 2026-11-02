"""Track B step 1: how much of a clone's later N/M fate is predictable from its day-2 state?

In vitro LARRY (fit data, already spent as a test). For each clone with day-2 cells and >= 3
day-4/6 N+M cells, target = N / (N + M); state = mean of its day-2 cells' features. 5-fold CV by
clone; kNN regression and ridge. Features: SPRING 2D coordinates, or top PCs of expression
(`--pcs`, needs the normalised counts).

    uv run python experiments/track_b_state.py
    uv run python experiments/track_b_state.py --pcs 50
"""

from __future__ import annotations

import argparse
import gzip
import json
import os

import numpy as np
import scipy.io
import scipy.sparse

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "larry")


def load():
    with gzip.open(os.path.join(DATA, "metadata.txt.gz"), "rt") as f:
        head = f.readline().rstrip("\n").split("\t")
        rows = [line.rstrip("\n").split("\t") for line in f]
    col = {h: i for i, h in enumerate(head)}
    day = np.array([float(r[col["Time point"]]) for r in rows])
    ctype = np.array([r[col["Cell type annotation"]] for r in rows])
    xy = np.array([[float(r[col["SPRING-x"]]), float(r[col["SPRING-y"]])] for r in rows])
    clone = scipy.io.mmread(os.path.join(DATA, "clone_matrix.mtx.gz")).tocsr()
    cid = np.full(len(rows), -1)
    r, c = clone.nonzero()
    cid[r] = c
    return day, ctype, xy, cid


def pcs_of(n_pcs, keep):
    """Top PCs of log-normalised expression for the cells in `keep` (bool mask)."""
    # day-2 rows pre-extracted from stateFate_inVitro_normed_counts.mtx.gz (1-based triplets)
    t = np.loadtxt(os.path.join(DATA, "day2_normed_triplets.txt.gz"))
    m = scipy.sparse.csr_matrix((t[:, 2], (t[:, 0].astype(int) - 1, t[:, 1].astype(int) - 1)),
                                shape=(len(keep), int(t[:, 1].max())))
    x = m[np.flatnonzero(keep)]
    x.data = np.log1p(x.data)
    var = np.asarray(x.multiply(x).mean(0)).ravel() - np.asarray(x.mean(0)).ravel() ** 2
    top = np.argsort(var)[-2000:]
    x = x[:, top].toarray()
    x -= x.mean(0)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    out = np.zeros((len(keep), n_pcs))
    out[keep] = u[:, :n_pcs] * s[:n_pcs]
    return out


def cv_predict(feat, y, folds, model, k=15, lam=1.0):
    pred = np.zeros_like(y)
    for f in range(folds.max() + 1):
        tr, te = folds != f, folds == f
        mu, sd = feat[tr].mean(0), feat[tr].std(0) + 1e-9
        a, b = (feat[tr] - mu) / sd, (feat[te] - mu) / sd
        if model == "knn":
            d = ((b[:, None, :] - a[None, :, :]) ** 2).sum(-1)
            pred[te] = y[tr][np.argsort(d, 1)[:, :k]].mean(1)
        else:
            a1, b1 = np.c_[a, np.ones(len(a))], np.c_[b, np.ones(len(b))]
            w = np.linalg.solve(a1.T @ a1 + lam * np.eye(a1.shape[1]), a1.T @ y[tr])
            pred[te] = b1 @ w
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcs", type=int, default=0)
    args = ap.parse_args()
    day, ctype, xy, cid = load()
    feats = {"spring": xy}
    if args.pcs:
        feats[f"pc{args.pcs}"] = pcs_of(args.pcs, day == 2)
    late = (day > 2) & (cid >= 0)
    n = np.bincount(cid[late & (ctype == "Neutrophil")], minlength=cid.max() + 1)
    m = np.bincount(cid[late & (ctype == "Monocyte")], minlength=cid.max() + 1)
    d2 = (day == 2) & (cid >= 0)
    n2 = np.bincount(cid[d2], minlength=cid.max() + 1)
    keep = np.flatnonzero((n + m >= 3) & (n2 >= 1))
    y = n[keep] / (n[keep] + m[keep])
    folds = np.random.default_rng(0).integers(0, 5, len(keep))
    out = {"clones": len(keep), "day2_cells_per_clone": float(n2[keep].mean()),
           "late_nm_per_clone": float((n + m)[keep].mean()), "target_var": float(y.var())}
    for name, f in feats.items():
        state = np.stack([f[d2 & (cid == c)].mean(0) for c in keep])
        for model in ("knn", "ridge"):
            p = cv_predict(state, y, folds, model)
            r = float(np.corrcoef(p, y)[0, 1])
            out[f"{name}_{model}"] = {"cv_r": r, "r2": float(1 - ((y - p) ** 2).mean() / y.var())}
    os.makedirs(os.path.join(HERE, "results_track_b"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results_track_b", "step1.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
