"""C9-S0: H19 (amended) power scan for chapter 9 candidates H1-H4.
Train statistics only; sealed partitions are touched ONLY for structural counts (clone/day/exp/assay columns,
no fate or cell-type column loaded). See EXPERIMENT_LOG C9-S0.   python c9s0_scan.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

import partitions
from c7s3_carrier import WCAP, wmean
from c8s0_scan import CS, _fold, cv_binomial

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_c9")
Z = norm.ppf(0.975)
FISHER_SD = float(np.sqrt(1.06))


def pw_mean(eff, sd, n):
    return float(1 - norm.cdf(Z - eff / (sd / np.sqrt(max(n, 1)))))


def pw_rho(rho, n):
    return float(1 - norm.cdf(Z - np.arctanh(max(rho, 0)) / (FISHER_SD / np.sqrt(max(n, 1)))))


# ---------------- structural counts (no fate columns) ----------------
def _hspc_struct_obs():
    import h5py

    import c7s6_hspc as c6

    with h5py.File(c6.H5AD, "r") as f:
        o = f["obs"]
        obs = pd.DataFrame({"clone": c6._cat(o, "clone_id"), "day": o["sample_day_as_int"][:],
                            "exp": c6._cat(o, "exp_id")})
    obs = obs[obs["clone"].notna()]
    obs["exp"] = obs["exp"].astype(str)
    return obs


def _hspc_struct(x, exps):
    x = x[x["exp"].isin(exps)]
    pc = x.groupby(["clone", "day"]).size().unstack(fill_value=0)
    early = pc[[c for c in pc.columns if c <= 5]].sum(axis=1)
    return int(((early >= 1) & (pc.get(10, 0) >= 2)).sum())


def hspc_counts(exps, part="B"):
    obs = _hspc_struct_obs()
    trn = partitions.train_only("chspctrack", obs, unit_col="clone")
    held = partitions.sealed_rows("chspctrack", obs, "clone", part)
    return _hspc_struct(trn, exps), _hspc_struct(held, exps)


def _lsk_struct(t):
    early = set(t.loc[(t["day"] == "d2") & (t["assay"] == "atac"), "clone.id"])
    d5 = t[t["day"] == "d5"].groupby("clone.id").size()
    return len(early & set(d5[d5 >= 2].index))


def lsk_counts(part="C"):
    import c7s3_carrier as c3

    t = pd.read_csv(c3.CSV, usecols=["clone.id", "day", "assay"])
    return (_lsk_struct(partitions.train_only("celltag_lsk", t, unit_col="clone.id")),
            _lsk_struct(partitions.sealed_rows("celltag_lsk", t, "clone.id", part)))


# ---------------- HSPC train ----------------
def hspc_train():
    import c7s6_hspc as c6
    import c7s7_confirm as c7

    obs = c7._hspc_obs()
    obs["exp"] = obs["exp"].astype(str)
    clonal = obs[obs["clone"].notna()]
    trn = partitions.train_only("chspctrack", clonal, unit_col="clone")
    print("train days per exp:", {e: sorted(g["day"].unique().tolist()) for e, g in trn.groupby("exp")}, flush=True)
    non = obs[obs["clone"].isna() & (obs["day"] <= 5)]
    et = trn[trn["day"] <= 5]
    cf = c7._hspc_features(et, et, non)
    cf = (cf - cf.mean()) / (cf.std() + 1e-9)
    return c6, trn, cf


def state_vs_cond(c6, trn, cf, exps, label):
    t = trn[trn["exp"].isin(exps)]
    K10 = c6.fate_counts(t, 10)
    u = [c for c in K10.index[K10.sum(1) >= 2] if c in cf.index]
    k = K10.loc[u].to_numpy()
    f = np.array([_fold(c) for c in u])
    levels = sorted(t["cond"].unique())
    oh, cd = c6.cond_onehot(t, u, levels)
    ex = t.groupby("clone")["exp"].first().reindex(u).to_numpy()
    Bc = c6.fold_bases(k, cd, f)
    Bp = c6.fold_bases(k, ex, f)  # experiment pool
    n = k.sum(1)
    w = np.minimum(n, WCAP)
    g_cond = (k * (np.log2(Bc) - np.log2(Bp))).sum(1) / n
    ge = c6.cv_gain(np.c_[cf.loc[u].to_numpy(), oh], k, f, Bc)
    C = max(c6.CS, key=lambda c: wmean(ge[c] - g_cond, w))
    d = ge[C] - g_cond
    eff, sd = wmean(d, w), float(np.std(d))
    neff_ratio = float(w.sum() ** 2 / (w ** 2).sum() / len(u))
    st, sb = hspc_counts(exps)
    n_b = sb * (len(u) / st) * neff_ratio
    by_exp = {e: {"n": int((ex == e).sum()), "diff": wmean(d[ex == e], w[ex == e])} for e in sorted(set(ex))}
    return {"label": label, "n_train": len(u), "C": C, "G_cond": wmean(g_cond, w), "G_state": wmean(ge[C], w),
            "diff": eff, "sd": sd, "by_exp": by_exp, "struct_train": st, "struct_B": sb, "neff_B": n_b,
            "power_B": pw_mean(eff, sd, n_b)}


def mege_split(c6, trn, cf, label):
    K10 = c6.fate_counts(trn, 10)
    K = pd.DataFrame({"MegE": K10["Ery"] + K10["MK"], "GM": K10["Neu"] + K10["Mono"]})
    K = K[(K.sum(1) >= 2) & K.index.isin(cf.index)]
    u = list(K.index)
    Kn = K.to_numpy()
    levels = sorted(trn["cond"].unique())
    oh, cd = c6.cond_onehot(trn, u, levels)
    folds = np.array([_fold(c) for c in u])
    preds = cv_binomial(np.c_[cf.loc[u].to_numpy(), oh], Kn, folds)
    frac = Kn[:, 0] / Kn.sum(1)

    def centred(v):
        s = pd.Series(v).groupby(cd).transform("mean").to_numpy()
        return v - s

    by = {str(c): float(spearmanr(centred(preds[c]), centred(frac)).statistic) for c in CS}
    C = max(CS, key=lambda c: by[str(c)])
    rho = by[str(C)]
    exps = sorted(trn["exp"].unique())
    st, sb = hspc_counts(exps)
    n_b = sb * (len(u) / st) - 3
    return {"label": label, "n_train": len(u), "C": C, "rho_within_stratum": rho, "by_C": by,
            "rho_raw": float(spearmanr(preds[C], frac).statistic), "mege_frac_mean": float(frac.mean()),
            "struct_train": st, "struct_B": sb, "n_B": n_b, "power_B": pw_rho(rho, n_b),
            "note": "C chosen by the test statistic itself on train (optimistic); selection step would use CV gain"}


def lsk_atac(label):
    import c7s3_carrier as c3
    from c8s1_confirm import _nm_units

    tab = partitions.train_only("celltag_lsk", c3._table(), unit_col="clone.id")
    tc = tab["clone.id"].unique()
    F, sizes = c3.clone_features("atac", tc, tc)
    K = _nm_units(tab, sizes)
    Kn = K.to_numpy()
    folds = np.array([_fold(c) for c in K.index])
    preds = cv_binomial(F.loc[K.index].to_numpy(), Kn, folds)
    frac = Kn[:, 0] / Kn.sum(1)
    by = {str(c): float(spearmanr(preds[c], frac).statistic) for c in CS}
    C = max(CS, key=lambda c: by[str(c)])
    rho = by[str(C)]
    st, sc = lsk_counts("C")
    n_c = sc * (len(K) / st) - 3
    return {"label": label, "n_train": int(len(K)), "C": C, "rho": rho, "by_C": by, "struct_train": st,
            "struct_C": sc, "n_C": n_c, "power_C": pw_rho(rho, n_c), "last_partition": True}


def main():
    res = {}
    c6, trn, cf = hspc_train()
    res["H1"] = state_vs_cond(c6, trn, cf, ["6", "7"], "H1 HSPC state vs condition, exp 6+7")
    print("H1", json.dumps(res["H1"], default=float), flush=True)
    res["H2"] = state_vs_cond(c6, trn, cf, sorted(trn["exp"].unique()), "H2 HSPC state vs condition, all exps")
    print("H2", json.dumps(res["H2"], default=float), flush=True)
    res["H3"] = mege_split(c6, trn, cf, "H3 HSPC MegE vs GM split beyond condition")
    print("H3", json.dumps(res["H3"], default=float), flush=True)
    res["H4"] = lsk_atac("H4 LSK early chromatin -> Neu vs Mono (celltag_lsk:C, last)")
    print("H4", json.dumps(res["H4"], default=float), flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "c9s0_scan.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
