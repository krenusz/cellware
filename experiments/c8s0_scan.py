"""C8-S0: H19 power scan for chapter 8 candidates K1-K3. TRAIN ONLY (no sealed rows read).
See EXPERIMENT_LOG C8-S0.   python c8s0_scan.py
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

import partitions
from c7s3_carrier import WCAP, fit_model, wmean

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_c8")
CS = (0.001, 0.01, 0.1, 1.0, 10.0)
Z = norm.ppf(0.975)


def _fold(c):
    return int(hashlib.sha256(f"c8s0-{c}".encode()).hexdigest(), 16) % 5


def power_mean(d, w, scale, two_sided=True):
    neff = w.sum() ** 2 / (w ** 2).sum() * scale
    m = wmean(d, w)
    se = np.std(d) / np.sqrt(neff)
    return float(1 - norm.cdf(Z - abs(m) / se)), float(neff)


def power_rho(rho, n):
    return float(1 - norm.cdf(Z - np.arctanh(max(rho, 0)) * np.sqrt(max(n - 3, 1))))


def cv_binomial(F, K, folds):
    """K = [success, other] counts. Returns per-C CV predicted success probability and CV gain."""
    out = {}
    for C in CS:
        p = np.zeros(len(K))
        for k in range(5):
            tr, te = folds != k, folds == k
            m = fit_model(F[tr], K[tr], C)
            P = np.zeros((te.sum(), 2))
            P[:, m.classes_] = m.predict_proba(F[te])
            p[te] = P[:, 0]
        out[C] = p
    return out


def scan_binary(F, K, scale, label):
    folds = np.array([_fold(c) for c in K.index])
    Kn = K.to_numpy()
    preds = cv_binomial(F.loc[K.index].to_numpy(), Kn, folds)
    frac = Kn[:, 0] / Kn.sum(1)
    w = np.minimum(Kn.sum(1), WCAP)
    base = frac.mean()

    def gain(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        b = np.clip(base, 1e-6, 1 - 1e-6)
        return wmean((Kn[:, 0] * (np.log2(p) - np.log2(b)) + Kn[:, 1] * (np.log2(1 - p) - np.log2(1 - b))) / Kn.sum(1), w)

    C = max(CS, key=lambda c: gain(preds[c]))
    rho = float(spearmanr(preds[C], frac).statistic)
    n_b = len(K) * scale
    return {"label": label, "n_train": int(len(K)), "C": C, "rho": rho, "gain_bits": gain(preds[C]),
            "by_C_rho": {str(c): float(spearmanr(preds[c], frac).statistic) for c in CS},
            "n_B_est": n_b, "power_B": power_rho(rho, n_b)}


def k1_hspc():
    import c7s6_hspc as c6
    import c7s7_confirm as c7

    obs = c7._hspc_obs()
    clonal = obs[obs["clone"].notna()]
    trn = partitions.train_only("chspctrack", clonal, unit_col="clone")
    e5 = sorted(trn.loc[trn["day"] == 5, "exp"].astype(str).unique())  # the day-5-early experiment (exp 7)
    print("K1 day-5 experiments:", e5, "all:", sorted(trn["exp"].astype(str).unique()), flush=True)
    trn = trn[trn["exp"].astype(str) == "7"]  # as pre-registered (exp 7); exp 6 also has day-5 cells, not used
    non = obs[obs["clone"].isna() & (obs["day"] <= 5)]
    et = trn[trn["day"] <= 5]
    cf = c7._hspc_features(et, et, non)
    cf = (cf - cf.mean()) / (cf.std() + 1e-9)
    K10 = c6.fate_counts(trn, 10)
    u = [c for c in K10.index[K10.sum(1) >= 2] if c in cf.index]
    k = K10.loc[u].to_numpy()
    f = np.array([_fold(c) for c in u])
    levels = sorted(trn["cond"].unique())
    oh, cd = c6.cond_onehot(trn, u, levels)
    Bc = c6.fold_bases(k, cd, f)  # condition baseline
    Bp = c6.fold_bases(k, np.array(["pool"] * len(u)), f)  # exp-7 pooled baseline
    n = k.sum(1)
    g_cond = (k * (np.log2(Bc) - np.log2(Bp))).sum(1) / n
    ge = c6.cv_gain(np.c_[cf.loc[u].to_numpy(), oh], k, f, Bc)
    w = np.minimum(n, WCAP)
    C = max(c6.CS, key=lambda c: wmean(ge[c], w))
    g_state = ge[C]
    d = g_state - g_cond
    pw, neff = power_mean(d, w, 0.5)
    return {"label": "K1 instruction vs clone state (HSPC exp 7)", "n_train": len(u), "C": C,
            "conditions": {c: int((cd == c).sum()) for c in np.unique(cd)},
            "G_cond": wmean(g_cond, w), "G_state_given_cond": wmean(g_state, w), "diff": wmean(d, w),
            "sd_diff": float(np.std(d)), "neff_B_est": neff, "power_B": pw}


def k2_reprog():
    import c7s5_reprog as c5

    rng = np.random.default_rng(75)
    tab = c5._table()
    F, sizes, _, _ = c5.features("rna", tab, rng)
    K = c5._fates(tab).reindex(sizes.index).dropna()
    K = K[K.sum(1) >= 2]
    K2 = pd.DataFrame({"success": K["reprogramming"], "other": K["transition"] + K["dead-end"]})
    return scan_binary(F, K2, 1647 / 3208, "K2 early predictor of reprogramming success")


def k3_lsk():
    import c7s3_carrier as c3

    tab = partitions.train_only("celltag_lsk", c3._table(), unit_col="clone.id")
    tc = tab["clone.id"].unique()
    F, sizes = c3.clone_features("rna", tc, tc)
    K = c3._fates(tab).reindex(sizes.index).dropna()[["Neu", "Mono"]]
    K = K[K.sum(1) >= 2]
    K2 = pd.DataFrame({"success": K["Neu"], "other": K["Mono"]})
    return scan_binary(F, K2, 1365 / 2720, "K3 early predictor of Neu vs Mono bias (LSK)")


def main():
    res = {}
    for name, fn in (("K1", k1_hspc), ("K2", k2_reprog), ("K3", k3_lsk)):
        res[name] = fn()
        print(name, json.dumps(res[name], default=float), flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "c8s0_scan.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
