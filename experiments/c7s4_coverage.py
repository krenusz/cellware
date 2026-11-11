"""C7-S4: carrier coverage. How much of a clone's heritable fate state (sister-well ceiling) does the day-2.5
snapshot capture? CellTag-multi LSK, TRAIN ONLY. See EXPERIMENT_LOG C7-S4.

    python c7s4_coverage.py
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd

import partitions
from c7s2_sister import CSV, GROUPS, OUT
from c7s3_carrier import FATES, WCAP, _fates, clone_features, fit_model, wmean

AS = (0.5, 1, 2, 5, 10, 20)
CS = (0.001, 0.01, 0.1, 1.0, 10.0)


def _fold(c):
    return int(hashlib.sha256(f"c7s4-{c}".encode()).hexdigest(), 16) % 5


def load():
    tab = pd.read_csv(CSV, usecols=["clone.id", "cell.bc", "day", "assay", "cell_type2"])
    tab = partitions.train_only("celltag_lsk", tab, unit_col="clone.id")
    d5 = tab[(tab["day"] == "d5") & (tab["assay"] == "rna")].copy()
    d5["well"] = d5["cell.bc"].str.extract(r"^d5(r[12])-")[0]
    d5["grp"] = d5["cell_type2"].map(GROUPS)
    d5 = d5[d5["grp"].isin(FATES)]
    t = d5.groupby(["clone.id", "well", "grp"]).size().unstack("grp", fill_value=0).reindex(columns=FATES, fill_value=0)
    t = t.unstack("well", fill_value=0)
    k1 = t.xs("r1", axis=1, level="well").reindex(columns=FATES, fill_value=0)
    k2 = t.xs("r2", axis=1, level="well").reindex(columns=FATES, fill_value=0)
    return tab, k1, k2


def bases(k1, k2, clones):
    """Per-fold baseline: RNA day-5 fate frequencies of train clones outside the fold."""
    tot = k1 + k2
    f = np.array([_fold(c) for c in tot.index])
    out = {}
    for k in range(5):
        s = tot[f != k].to_numpy().sum(0)
        out[k] = s / s.sum()
    return {c: out[_fold(c)] for c in clones}


def g_sister(k1, k2, base, a):
    p1 = (k2 + a * base) / (k2.sum(1, keepdims=True) + a)
    p2 = (k1 + a * base) / (k1.sum(1, keepdims=True) + a)
    lb = np.log2(np.maximum(base, 1e-9))
    bits = (k1 * (np.log2(p1) - lb)).sum(1) + (k2 * (np.log2(p2) - lb)).sum(1)
    return bits / (k1.sum(1) + k2.sum(1))


def g_cells(P, K, base):
    P = np.maximum(P, 1e-6)
    P = P / P.sum(1, keepdims=True)
    return (K * (np.log2(P) - np.log2(np.maximum(base, 1e-9)))).sum(1) / K.sum(1)


def early_probs(mod, tab, rng):
    """CV (folds by clone hash) predicted fate distributions for every train clone with an early sibling; C chosen
    by the C7-S3 criterion (CV gain on all fate cells)."""
    train_clones = tab["clone.id"].unique()
    F, sizes = clone_features(mod, train_clones, train_clones)
    K = _fates(tab).reindex(sizes.index).dropna()
    K = K[K.sum(1) >= 2]
    Fm, Kn = F.loc[K.index].to_numpy(), K.to_numpy()
    f = np.array([_fold(c) for c in K.index])
    probs, score = {}, {}
    for C in CS:
        P = np.zeros((len(K), len(FATES)))
        g = np.zeros(len(K))
        for k in range(5):
            tr, te = f != k, f == k
            m = fit_model(Fm[tr], Kn[tr], C)
            P[np.ix_(te, m.classes_)] = m.predict_proba(Fm[te])
            base = Kn[tr].sum(0) / Kn[tr].sum()
            g[te] = g_cells(P[te], Kn[te], base)
        probs[C] = pd.DataFrame(P, index=K.index)
        score[C] = wmean(g, np.minimum(Kn.sum(1), WCAP))
    best = max(CS, key=score.get)
    return probs[best], best, {str(c): s for c, s in score.items()}


def choose_a(k1, k2, B, f):
    """a chosen on the other folds, applied to each fold."""
    w = np.minimum(k1.sum(1) + k2.sum(1), WCAP)
    g = np.zeros(len(k1))
    chosen = {}
    for k in range(5):
        tr, te = f != k, f == k
        a = max(AS, key=lambda a: wmean(g_sister(k1[tr], k2[tr], B[tr], a), w[tr]))
        chosen[k] = a
        g[te] = g_sister(k1[te], k2[te], B[te], a)
    return g, chosen


def coverage(units, k1, k2, Pdf, B, rng, min_cells):
    m = (k1.loc[units].sum(1) >= min_cells) & (k2.loc[units].sum(1) >= min_cells)
    u = np.array(units)[m.to_numpy()]
    K1, K2 = k1.loc[u].to_numpy(), k2.loc[u].to_numpy()
    Bu = np.array([B[c] for c in u])
    f = np.array([_fold(c) for c in u])
    gs, a_ch = choose_a(K1, K2, Bu, f)
    ge = g_cells(Pdf.loc[u].to_numpy(), K1 + K2, Bu)
    w = np.minimum(K1.sum(1) + K2.sum(1), WCAP)
    cov = wmean(ge, w) / wmean(gs, w)
    bs = []
    for _ in range(1000):
        ix = rng.integers(0, len(u), len(u))
        bs.append(wmean(ge[ix], w[ix]) / wmean(gs[ix], w[ix]))
    ci = np.percentile(bs, [2.5, 97.5]).tolist()
    # null calibration: sister wells re-paired across random clones
    null = []
    for _ in range(10):
        p = rng.permutation(len(u))
        null.append(wmean(g_sister(K1, K2[p], Bu, float(np.median(list(a_ch.values())))), w))
    gci = lambda g: np.percentile([wmean(g[ix], w[ix]) for ix in (rng.integers(0, len(u), len(u)) for _ in range(1000))], [2.5, 97.5]).tolist()
    reading = ("snapshot carries most" if ci[0] > 0.5 else "snapshot carries a minority" if ci[1] < 0.5 else
               "unresolved")
    return {"min_cells": min_cells, "n": int(len(u)), "a_by_fold": a_ch, "null_G_sister_mean": float(np.mean(null)),
            "null_ok": bool(np.mean(null) <= 0.02),
            "G_sister": wmean(gs, w), "G_sister_ci": gci(gs), "G_early": wmean(ge, w), "G_early_ci": gci(ge),
            "coverage": cov, "coverage_ci": ci, "reading": reading}


def main():
    rng = np.random.default_rng(76)
    tab, k1, k2 = load()
    sis = k1.index[(k1.sum(1) >= 2) & (k2.sum(1) >= 2)]
    res = {"n_sister_ge2": int(len(sis))}
    for mod in ("rna", "atac"):
        Pdf, C, by_C = early_probs(mod, tab, rng)
        units = [c for c in sis if c in Pdf.index]
        B = bases(k1, k2, units)
        res[mod] = {"C": C, "by_C": by_C,
                    "primary": coverage(units, k1, k2, Pdf, B, rng, 2),
                    "secondary_ge4": coverage(units, k1, k2, Pdf, B, rng, 4)}
        print(mod, json.dumps(res[mod], default=float), flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "c7s4_coverage.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
