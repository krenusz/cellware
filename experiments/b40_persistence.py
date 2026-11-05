"""B40: does Bcor loss select clones by persistence, and by fate? SIS-skew fit set.

    uv run python b40_persistence.py [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

import sis_data as sd
from b39_sisskew import GROUPS, SIZE_BINS


def logit_fit(x, y, iters=50):
    """Logistic regression y ~ a + b x by Newton's method; returns (a, b)."""
    X = np.column_stack([np.ones_like(x), x])
    w = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-6 * np.eye(2)
        w += np.linalg.solve(H, X.T @ (y - p))
    return w


def one_arm_flags(P, N):
    pp, pn = P.sum(1) > 1, N.sum(1) > 1
    return pp ^ pn, pp & ~pn


def persistence_by_fate(P, N):
    """[rho(- arm fate j, + arm absent) for j] + [rho(+ arm fate j, - arm absent) for j]."""
    out = []
    for src, oth in ((N, P), (P, N)):
        m = src.sum(1) > 10
        frac = sd._frac(src[m])
        absent = (oth[m].sum(1) <= 1).astype(float)
        out += [sd.spearman(frac[:, j], absent) if absent.std() > 0 else np.nan for j in range(3)]
    return out


def main(out=None, boot=1000):
    rng = np.random.default_rng(40)
    d = sd.load_skew("fit")
    groups = {g: np.isin(d["well"], w) for g, w in GROUPS.items()}
    tot = d["P"].sum(1) + d["N"].sum(1)
    x_med = float(np.median(np.log(tot)))
    res = {"x_median_log_cells": x_med}

    # A: odds ratio KO vs null at the pooled median size
    def odds_ratio(ixk, ixn):
        o = []
        for ix in (ixk, ixn):
            P, N = d["P"][ix], d["N"][ix]
            y = one_arm_flags(P, N)[0].astype(float)
            a, b = logit_fit(np.log(P.sum(1) + N.sum(1)) - x_med, y)
            o.append(a)
        return float(np.exp(o[0] - o[1]))
    ik, inn = np.flatnonzero(groups["KO"]), np.flatnonzero(groups["null"])
    orr = odds_ratio(ik, inn)
    orr_ci = np.percentile([odds_ratio(rng.choice(ik, len(ik)), rng.choice(inn, len(inn)))
                            for _ in range(boot)], [2.5, 97.5]).tolist()
    res["A_odds_ratio_KO_vs_null"] = {"est": orr, "ci": orr_ci}

    for g, m in groups.items():
        P, N = d["P"][m], d["N"][m]
        one, plus_only = one_arm_flags(P, N)
        t = P.sum(1) + N.sum(1)
        b = {}
        for lo, hi in SIZE_BINS:
            mm = (t >= lo) & (t <= hi) & one
            b[f"{lo}-{hi if np.isfinite(hi) else 'inf'}"] = {
                "n_one_arm": int(mm.sum()),
                "plus_only_share": float(plus_only[mm].mean()) if mm.any() else None}
        rho = persistence_by_fate(P, N)
        rho_ci = sd.bootstrap(lambda ix: persistence_by_fate(P[ix], N[ix]), len(P), rng, boot)
        lab = [f"plus_absent_vs_minus_{f}" for f in sd.FATES] + [f"minus_absent_vs_plus_{f}" for f in sd.FATES]
        res[g] = {"n_clones": int(m.sum()),
                  "n_minus_gt10": int((N.sum(1) > 10).sum()),
                  "n_plus_gt10": int((P.sum(1) > 10).sum()),
                  "share_plus_absent_given_minus_gt10": float((P[N.sum(1) > 10].sum(1) <= 1).mean()),
                  "share_minus_absent_given_plus_gt10": float((N[P.sum(1) > 10].sum(1) <= 1).mean()),
                  "B_direction": b,
                  "C_rho": {k: {"est": rho[i], "ci": [rho_ci[0][i], rho_ci[1][i]]} for i, k in enumerate(lab)}}
    out = out or os.path.join(os.path.dirname(__file__), "results_sis")
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, "b40.json"), "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=lambda x: round(float(x), 3)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    main(a.out, a.boot)
