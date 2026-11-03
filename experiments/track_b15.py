"""B15: how much of the selection state is readable from day-2 expression (cytokine experiment)?
See the 2026-11-03 (8) batch pre-registration in EXPERIMENT_LOG.md. Needs the cytokine counts
(downloaded by track_b13.fetch); meant for GitHub Actions.

    python experiments/track_b15.py
"""

from __future__ import annotations

import gzip
import json
import os

import numpy as np
import pandas as pd
import scipy.sparse
from e25_clone_types import groups
from fit_cytokine import CONDS5, load_counts
from track_b8_b9 import CYTO, HERE
from track_b11 import day2_features
from track_b13 import fetch


def day2_matrix():
    names = [g.strip() for g in gzip.open(fetch("gene_names.txt.gz"), "rt")]
    rows = [
        line.rstrip("\n").split("\t")
        for line in gzip.open(os.path.join(CYTO, "metadata.txt.gz"), "rt")
    ][1:]
    d2 = np.array([i + 1 for i, r in enumerate(rows) if r[2] == "2"])
    pos = {c: j for j, c in enumerate(d2)}
    ri, ci, vi = [], [], []
    with gzip.open(fetch("normed_counts.mtx.gz"), "rt") as f:
        line = f.readline()
        while line.startswith("%"):
            line = f.readline()
        for chunk in pd.read_csv(
            f, sep=" ", header=None, names=["c", "g", "v"], chunksize=5_000_000
        ):
            chunk = chunk[chunk.c.isin(pos.keys())]
            ri.append(chunk.c.map(pos).to_numpy())
            ci.append(chunk.g.to_numpy() - 1)
            vi.append(np.log1p(chunk.v.to_numpy()))
    x = scipy.sparse.csr_matrix(
        (np.concatenate(vi), (np.concatenate(ri), np.concatenate(ci))),
        shape=(len(d2), len(names)),
    )
    m = np.loadtxt(
        gzip.open(os.path.join(CYTO, "clone_matrix.mtx.gz"), "rt"),
        comments="%",
        skiprows=3,
        dtype=int,
    )
    clone = np.full(len(d2), -1)
    for cl, cell in zip(m[:, 0] - 1, m[:, 1], strict=True):
        if cell in pos:
            clone[pos[cell]] = cl
    return x, clone, names


def clone_features(x, clone, names, n_pcs=30):
    var = np.asarray(x.multiply(x).mean(0)).ravel() - np.asarray(x.mean(0)).ravel() ** 2
    top = np.argsort(var)[-2000:]
    d = x[:, top].toarray()
    d -= d.mean(0)
    u, s, _ = np.linalg.svd(d, full_matrices=False)
    pc = u[:, :n_pcs] * s[:n_pcs]
    cebpa = np.asarray(x[:, names.index("Cebpa")].todense()).ravel()
    out = {}
    for c in np.unique(clone[clone >= 0]):
        i = clone == c
        out[int(c)] = (pc[i].mean(0), float(cebpa[i].mean()))
    return out


def logit_cv(f, y, rng, lam=1.0, reps=20):
    """Out-of-fold probabilities (mean over repeats) from L2 logistic regression."""
    pred = np.zeros(len(y))
    for _ in range(reps):
        folds = rng.integers(0, 5, len(y))
        for k in range(5):
            tr, te = folds != k, folds == k
            mu, sd = f[tr].mean(0), f[tr].std(0) + 1e-9
            a = np.c_[(f[tr] - mu) / sd, np.ones(tr.sum())]
            b = np.c_[(f[te] - mu) / sd, np.ones(te.sum())]
            w = np.zeros(a.shape[1])
            for _ in range(200):  # Newton steps
                p = 1 / (1 + np.exp(-a @ w))
                g = a.T @ (p - y[tr]) + lam * np.r_[w[:-1], 0]
                h = (a * (p * (1 - p))[:, None]).T @ a + lam * np.diag(
                    np.r_[np.ones(len(w) - 1), 0]
                )
                step = np.linalg.solve(h + 1e-9 * np.eye(len(w)), g)
                w -= step
                if np.abs(step).max() < 1e-6:
                    break
            pred[te] += 1 / (1 + np.exp(-b @ w)) / reps
    return pred


def ridge_cv(f, y, rng, lam=1.0, reps=20):
    pred = np.zeros(len(y))
    for _ in range(reps):
        folds = rng.integers(0, 5, len(y))
        for k in range(5):
            tr, te = folds != k, folds == k
            mu, sd = f[tr].mean(0), f[tr].std(0) + 1e-9
            a = np.c_[(f[tr] - mu) / sd, np.ones(tr.sum())]
            b = np.c_[(f[te] - mu) / sd, np.ones(te.sum())]
            w = np.linalg.solve(a.T @ a + lam * np.eye(a.shape[1]), a.T @ y[tr])
            pred[te] += b @ w / reps
    return pred


def auc(p, y):
    pos, neg = p[y == 1], p[y == 0]
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def compare(preds, y, metric, rng, boots=2000):
    out = {k: metric(v, y) for k, v in preds.items()}
    for k in preds:
        if k == "cebpa":
            continue
        d = []
        for _ in range(boots):
            i = rng.choice(len(y), len(y))
            if metric is auc and not 0 < y[i].sum() < len(i):
                continue
            d.append(metric(preds[k][i], y[i]) - metric(preds["cebpa"][i], y[i]))
        out[f"{k}_minus_cebpa_ci"] = np.percentile(d, [2.5, 97.5]).tolist()
    return out


def main():
    x, clone, names = day2_matrix()
    feats = clone_features(x, clone, names)
    score = day2_features()
    cnt = load_counts(CONDS5)
    pg, pm, both, g_only, m_only = groups(cnt)
    rng = np.random.default_rng(0)
    rep = {"clones_with_day2": len(feats)}

    def run(keys, y, kind):
        f_full = np.stack([feats[k][0] for k in keys])
        f_ceb = np.array([[feats[k][1]] for k in keys])
        f_sc = np.array([[score[k][0]] for k in keys])
        fit = logit_cv if kind == "auc" else ridge_cv
        preds = {
            name: fit(f, y, np.random.default_rng(1))
            for name, f in (
                ("full", f_full),
                ("cebpa", f_ceb),
                ("priming", f_sc),
                ("full_plus_cebpa", np.c_[f_full, f_ceb]),
            )
        }
        metric = auc if kind == "auc" else (lambda p, t: float(np.corrcoef(p, t)[0, 1]))
        r = compare(preds, y, metric, rng)
        r["n"], r["positives"] = len(keys), int(y.sum()) if kind == "auc" else None
        lo = r["full_minus_cebpa_ci"][0]
        r["more_readable"] = bool(r["full"] - r["cebpa"] >= 0.05 and lo > 0)
        return r

    ka = [k for k in g_only if k in feats and k in score] + [
        k for k in m_only if k in feats and k in score
    ]
    ya = np.array([1.0] * sum(k in g_only for k in ka) + [0.0] * sum(k in m_only for k in ka))
    rep["a_Gonly_vs_Monly"] = run(ka, ya, "auc")
    kb = [k for k, v in pm.items() if v[0] + v[1] >= 2 and k in feats and k in score]
    yb = np.array([pm[k][0] >= 1 for k in kb], float)
    rep["b_N_under_M"] = run(kb, yb, "auc")
    kc = [k for k, v in pg.items() if v[0] + v[1] >= 2 and k in feats and k in score]
    yc = np.array([pg[k][0] / (pg[k][0] + pg[k][1]) for k in kc])
    rep["c_G_Nshare"] = run(kc, yc, "r")
    json.dump(rep, open(os.path.join(HERE, "results_e18", "b15.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
