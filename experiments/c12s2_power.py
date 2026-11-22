"""C12-S2: simulated power (H17) of the frozen C12 sealed test, train only (pre-registered in EXPERIMENT_LOG "C12-S2").

Serves: C12-S2. Bence: "1".
    python c12s2_power.py            # rebuild train covariates, selftests, power per configuration
    python c12s2_power.py selftest   # selftests only
Outcomes are indexed ONLY for train clones (partitions.train_only). Sealed partitions enter as structural clone
counts only; no sealed outcome or expression is read.
"""

from __future__ import annotations

import collections
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # run with python -I
import c12s0_lsc as c0  # noqa: E402
import c12s1_select as s1  # noqa: E402
import partitions  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_c12", "c12s2_power_a1.json")  # amendment 1
SIZES = {"A": 304, "B": 316, "C": 298, "A+B": 620, "A+B+C": 918}  # structural counts, C12-S1
N_SIM, N_PERM, N_POOL, SHRINK, ALPHA = 1000, 200, 20000, 0.5, 0.025
TRAIN_DELTA, TRAIN_N, TRAIN_POS = 0.166, 712, 101


def build():
    """Train covariates as C12-S1 amendment 2: nuisance matrix N (M0x), state PCs P, D20 in vivo outcome y."""
    cmap = dict(pd.read_csv(c0.MAP, sep="\t").itertuples(index=False, name=None))
    asg = {s: s1.assign_max(s, cmap) for s in ("D20_exvivo", "D20_invivo")}
    size = {s: collections.Counter(a.values()) for s, a in asg.items()}
    units = pd.DataFrame({"clone": [k for k, n in size["D20_exvivo"].items() if n >= s1.MIN_CELLS]})
    struct = dict(collections.Counter(units["clone"].map(lambda u: partitions.partition_of(c0.DS, u))))  # structural
    tr_clones = list(partitions.train_only("lsc_larry", units, "clone")["clone"])  # outcomes: train only (H11)
    tr_set = set(tr_clones)
    cells = sorted(c for c, k in asg["D20_exvivo"].items() if k in tr_set)
    X, genes = s1.d20_expression(cells)
    tot = np.asarray(X.sum(1)).ravel()
    nfeat = np.asarray((X > 0).sum(1)).ravel()
    mt = np.array([g.lower().startswith("mt-") for g in genes])
    pmt = np.asarray(X[:, mt].sum(1)).ravel() / np.maximum(tot, 1)
    Xn = X.multiply(1e4 / np.maximum(tot, 1)[:, None]).tocsr()
    Xn.data = np.log1p(Xn.data)
    var = np.asarray(Xn.power(2).mean(0)).ravel() - np.asarray(Xn.mean(0)).ravel() ** 2
    Z = Xn[:, np.argsort(-var)[:2000]].toarray()
    cell_clone = np.array([asg["D20_exvivo"][c] for c in cells])
    idx = {k: np.where(cell_clone == k)[0] for k in tr_clones}
    mean_expr = np.vstack([Z[idx[k]].mean(0) for k in tr_clones])
    P = PCA(5, random_state=0).fit_transform(mean_expr - mean_expr.mean(0))
    qc = np.column_stack([[np.log(len(idx[k])) for k in tr_clones],
                          [np.log(tot[idx[k]].mean()) for k in tr_clones],
                          [np.log(max(1.0, nfeat[idx[k]].mean())) for k in tr_clones],
                          [pmt[idx[k]].mean() for k in tr_clones]])
    lib = np.array([k.rsplit("_", 1)[0] for k in tr_clones])
    libs = sorted(set(lib))
    L = np.column_stack([(lib == x).astype(float) for x in libs[1:]])
    top, ncl, htag = s1.d20_extra(cells, cmap)
    bc_umi = np.array([np.mean([np.log(top.get(cells[i], 3)) for i in idx[k]]) for k in tr_clones])
    multi = np.array([np.mean([ncl.get(cells[i], 1) >= 2 for i in idx[k]]) for k in tr_clones])
    maj = [collections.Counter(htag[cells[i]] for i in idx[k] if cells[i] in htag).most_common(1) for k in tr_clones]
    tags = sorted({m[0][0] for m in maj if m})
    H = np.column_stack([[1.0 if (m and m[0][0] == t) else 0.0 for m in maj] for t in tags[1:]])
    N = np.hstack([qc, L, bc_umi[:, None], multi[:, None], H])
    y = np.array([1 if size["D20_invivo"].get(k, 0) > 0 else 0 for k in tr_clones])
    return N, P, y, struct


def fit(M, y):
    """Logistic fit as C12-S1 (standardised, C = 1); returns (mu, sd, coef, intercept)."""
    mu, sd = M.mean(0), M.std(0) + 1e-9
    lr = LogisticRegression(C=1.0, max_iter=2000).fit((M - mu) / sd, y)
    return mu, sd, lr.coef_.ravel(), float(lr.intercept_[0])


def score(m, M):
    mu, sd, w, b = m
    return ((M - mu) / sd) @ w + b


class Truth:
    """Generative model: logistic M1x on a train bootstrap; state-PC part of the linear predictor scaled by s."""

    def __init__(self, N, P, y, rng=None):
        i = rng.integers(0, len(y), len(y)) if rng is not None else np.arange(len(y))  # None: full-train fit
        mu, sd, w, b = fit(np.hstack([N[i], P[i]]), y[i])
        k = N.shape[1]
        self.mu, self.sd, self.w, self.b, self.k = mu, sd, w, b, k

    def parts(self, N, P):
        Zs = (np.hstack([N, P]) - self.mu) / self.sd
        return Zs[:, :self.k] @ self.w[:self.k] + self.b, Zs[:, self.k:] @ self.w[self.k:]

    def draw(self, N, P, s, rng):
        e0, e1 = self.parts(N, P)
        return (rng.random(len(N)) < 1 / (1 + np.exp(-(e0 + s * e1)))).astype(int)


def oracle_delta(truth, N, P, s, rng):
    e0, e1 = truth.parts(N, P)
    y = (rng.random(len(N)) < 1 / (1 + np.exp(-(e0 + s * e1)))).astype(int)
    return roc_auc_score(y, e0 + s * e1) - roc_auc_score(y, e0)


def calibrate(truth, N, P, target, rng):
    """s such that the oracle delta AUC on a large bootstrap pool equals target (bisection)."""
    i = rng.integers(0, len(N), N_POOL)
    Np, Pp = N[i], P[i]
    lo, hi = 0.0, 1.0
    while oracle_delta(truth, Np, Pp, hi, np.random.default_rng(7)) < target and hi < 64:
        hi *= 2
    for _ in range(30):
        mid = (lo + hi) / 2
        if oracle_delta(truth, Np, Pp, mid, np.random.default_rng(7)) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def frozen_test(m0, m1, N, P, y, rng):
    """Delta AUC (M1x - M0x) on the partition and its state-row permutation p value."""
    if y.min() == y.max():
        return 0.0, 1.0
    s0 = score(m0, N)
    a0 = roc_auc_score(y, s0)
    obs = roc_auc_score(y, score(m1, np.hstack([N, P]))) - a0
    A = np.hstack([np.ones((len(N), 1)), N])  # amendment 1: permute state residuals given M0x covariates
    B = np.linalg.lstsq(A, P, rcond=None)[0]
    fitP, res = A @ B, P - A @ B
    ge = sum(roc_auc_score(y, score(m1, np.hstack([N, fitP + res[rng.permutation(len(y))]]))) - a0 >= obs
             for _ in range(N_PERM))
    return float(obs), (1 + ge) / (1 + N_PERM)


def simulate(N, P, y, s, n_part, n_sim, seed):
    rng = np.random.default_rng(seed)
    passes, deltas = [], []
    for _ in range(n_sim):
        truth = Truth(N, P, y, rng)
        it = rng.integers(0, len(y), len(y))
        yt = truth.draw(N[it], P[it], s, rng)
        m0 = fit(N[it], yt)
        m1 = fit(np.hstack([N[it], P[it]]), yt)
        ip = rng.integers(0, len(y), n_part)
        yp = truth.draw(N[ip], P[ip], s, rng)
        d, p = frozen_test(m0, m1, N[ip], P[ip], yp, rng)
        passes.append(p < ALPHA)
        deltas.append(d)
    return {"power": round(float(np.mean(passes)), 3), "mean_delta": round(float(np.mean(deltas)), 4), "n_sim": n_sim}


def main(selftest_only=False):
    N, P, y, struct = build()
    cv0, _ = s1.cv_auc(N, y)
    cv1, _ = s1.cv_auc(np.hstack([N, P]), y)
    train_delta = cv1 - cv0
    rng = np.random.default_rng(1222)
    shuffle = [s1.cv_auc(np.hstack([N, P]), yp, repeats=1, seed=200 + j)[0] - s1.cv_auc(N, yp, repeats=1, seed=200 + j)[0]
               for j, yp in enumerate(rng.permutation(y) for _ in range(10))]
    res = {"structural_units": struct, "train": {"n_units": len(y), "positives": int(y.sum()),
           "cv_delta_auc": round(train_delta, 4), "label_shuffle_delta_mean": round(float(np.mean(shuffle)), 4)}}
    assert len(y) == TRAIN_N and int(y.sum()) == TRAIN_POS, f"train units {len(y)} / positives {y.sum()} differ from C12-S1"
    assert abs(train_delta - TRAIN_DELTA) < 0.005, f"train delta {train_delta:.4f} differs from C12-S1 {TRAIN_DELTA}"
    target = SHRINK * min(train_delta, train_delta - float(np.mean(shuffle)))
    ref = Truth(N, P, y)
    s_eff = calibrate(ref, N, P, target, np.random.default_rng(1))
    s_big = calibrate(ref, N, P, 0.25, np.random.default_rng(1))
    res["calibration"] = {"target_delta": round(target, 4), "s_effect": round(s_eff, 4), "s_planted": round(s_big, 4)}
    st = {"null_false_pass": simulate(N, P, y, 0.0, SIZES["A"], N_SIM, 11),
          "planted_power": simulate(N, P, y, s_big, SIZES["A"], 200, 12)}
    st["pass"] = st["null_false_pass"]["power"] <= 0.05 and st["planted_power"]["power"] >= 0.95
    res["selftest"] = st
    print(json.dumps(res, indent=1), flush=True)
    if not st["pass"]:
        res["verdict"] = "SELFTEST FAIL: no power reported"
    elif not selftest_only:
        res["power"] = {k: simulate(N, P, y, s_eff, n, N_SIM, 100 + j) for j, (k, n) in enumerate(SIZES.items())}
        single = [k for k in ("A", "B", "C") if res["power"][k]["power"] >= 0.8]
        pooled = [k for k in ("A+B", "A+B+C") if res["power"][k]["power"] >= 0.8]
        res["verdict"] = (f"single partition powered: {single}" if single else
                          f"only pooled powered: {pooled} (Bence's call)" if pooled else "nothing reaches 0.8: C12 stays train-only")
    res["caveat"] = "in-sample train PC scores used; a real look projects partition clones through train PCA (not modelled)"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(selftest_only=len(sys.argv) > 1 and sys.argv[1] == "selftest")
