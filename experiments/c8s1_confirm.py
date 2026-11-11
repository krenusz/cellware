"""C8-S1: chapter 8 train selection (select_*) and the sealed tests (test_*, counts_*) for test_oracle.py.
T1 (K3) LSK early RNA -> Neu vs Mono bias; T2 (K1) HSPC exp 7 clone state vs cytokine condition.
See EXPERIMENT_LOG C8-S1 and CHAPTER8_PLAN.md.   python c8s1_confirm.py   (train selection only)
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import partitions
from c7s3_carrier import WCAP, fit_model, wmean
from c7s7_confirm import ALPHA, _boot_g, _lb
from c8s0_scan import CS, _fold, cv_binomial

HERE = os.path.dirname(os.path.abspath(__file__))
SEL = os.path.join(HERE, "selection")
EXP = "7"
FISHER_SD = float(np.sqrt(1.06))


def _write(exp, sel):
    os.makedirs(SEL, exist_ok=True)
    json.dump(sel, open(os.path.join(SEL, f"{exp}.json"), "w"), indent=1, default=float)
    print(exp, json.dumps(sel, indent=1, default=float), flush=True)


def _power_z(effect, sd, n):
    from scipy.stats import norm

    return float(1 - norm.cdf(norm.ppf(1 - ALPHA) - effect / (sd / np.sqrt(n))))


# =============================== T1: LSK Neu vs Mono ===============================
def _nm_units(tab, sizes):
    import c7s3_carrier as c3

    K = c3._fates(tab).reindex(sizes.index).dropna()[["Neu", "Mono"]]
    return K[K.sum(1) >= 2]


def _lsk_struct(t):
    early = set(t.loc[(t["day"] == "d2") & (t["assay"] == "rna"), "clone.id"])
    d5 = t[t["day"] == "d5"].groupby("clone.id").size()
    return len(early & set(d5[d5 >= 2].index))


def select_lsk():
    import c7s3_carrier as c3

    tab = partitions.train_only("celltag_lsk", c3._table(), unit_col="clone.id")
    tc = tab["clone.id"].unique()
    F, sizes = c3.clone_features("rna", tc, tc)
    K = _nm_units(tab, sizes)
    Kn = K.to_numpy()
    folds = np.array([_fold(c) for c in K.index])
    preds = cv_binomial(F.loc[K.index].to_numpy(), Kn, folds)
    frac = Kn[:, 0] / Kn.sum(1)
    w = np.minimum(Kn.sum(1), WCAP)
    b = np.clip(frac.mean(), 1e-6, 1 - 1e-6)

    def gain(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return wmean((Kn[:, 0] * (np.log2(p) - np.log2(b)) + Kn[:, 1] * (np.log2(1 - p) - np.log2(1 - b))) / Kn.sum(1), w)

    by = {str(c): gain(preds[c]) for c in CS}
    C = max(CS, key=lambda c: by[str(c)])
    rho = float(spearmanr(preds[C], frac).statistic)
    ratio = len(K) / _lsk_struct(tab)
    _write("C8-S1a", {"test": "celltag_c8s1_lsk", "higher_is_better": True, "candidates": by, "chosen": str(C), "C": C,
                      "power": {"sd": FISHER_SD, "effect": float(np.arctanh(rho)), "alpha": ALPHA, "one_sided": True,
                                "min_power": 0.8},
                      "train": {"n": int(len(K)), "cv_rho": rho, "by_C_rho": {str(c): float(spearmanr(preds[c], frac).statistic) for c in CS},
                                "unit_ratio": ratio, "power_B_ratio_est": _power_z(np.arctanh(rho), FISHER_SD, len(K) * 1365 / 2720 - 3)},
                      "note": "T1 (K3): Spearman rho(pred P(Neu), observed Neu fraction) > 0 one-sided; power in Fisher-z units"})


def counts_lsk(part):
    import c7s3_carrier as c3

    sel = json.load(open(os.path.join(SEL, "C8-S1a.json")))
    tab = c3._table()[["clone.id", "day", "assay"]]  # no fate column read
    n = _lsk_struct(partitions.sealed_rows("celltag_lsk", tab, "clone.id", part))
    return {"rna": int(np.floor(n * sel["train"]["unit_ratio"])) - 3}


def test_lsk(part):
    import c7s3_carrier as c3

    rng = np.random.default_rng(91)
    sel = json.load(open(os.path.join(SEL, "C8-S1a.json")))
    full = c3._table()
    trn = partitions.train_only("celltag_lsk", full, unit_col="clone.id")
    held = partitions.sealed_rows("celltag_lsk", full, "clone.id", part)
    tc, hc = trn["clone.id"].unique(), held["clone.id"].unique()
    F, sizes = c3.clone_features("rna", np.r_[tc, hc], tc)
    Kt = _nm_units(trn, sizes.reindex(tc).dropna())
    Kh = _nm_units(held, sizes.reindex(hc).dropna())
    m = fit_model(F.loc[Kt.index].to_numpy(), Kt.to_numpy(), sel["C"])
    P = np.zeros((len(Kh), 2))
    P[:, m.classes_] = m.predict_proba(F.loc[Kh.index].to_numpy())
    p = P[:, 0]
    frac = Kh["Neu"].to_numpy() / Kh.sum(1).to_numpy()
    rho = float(spearmanr(p, frac).statistic)
    bs = [spearmanr(p[ix], frac[ix]).statistic for ix in (rng.integers(0, len(p), len(p)) for _ in range(2000))]
    bs = np.array([b for b in bs if np.isfinite(b)])
    lo = _lb(bs)
    return {"T1_rho": {"n": int(len(p)), "rho": rho, "lower_one_sided": lo,
                       "ci95": np.percentile(bs, [2.5, 97.5]).tolist(), "confirmed": bool(lo > 0)}}


# =============================== T2: HSPC exp 7 state vs condition ===============================
def _hspc_setup(held_part=None):
    import c7s6_hspc as c6
    import c7s7_confirm as c7

    obs = c7._hspc_obs()
    clonal = obs[obs["clone"].notna()]
    trn = partitions.train_only("chspctrack", clonal, unit_col="clone")
    trn = trn[trn["exp"].astype(str) == EXP]
    non = obs[obs["clone"].isna() & (obs["day"] <= 5)]
    et = trn[trn["day"] <= 5]
    held = eh = None
    if held_part is not None:
        held = partitions.sealed_rows("chspctrack", clonal, "clone", held_part)
        held = held[held["exp"].astype(str) == EXP]
        eh = held[held["day"] <= 5]
    cf = c7._hspc_features(et, et if eh is None else pd.concat([et, eh]), non)
    ref = cf.loc[cf.index.isin(et["clone"])]
    cf = (cf - ref.mean()) / (ref.std() + 1e-9)
    return c6, trn, held, cf


def _hspc_struct(x):
    pc = x.groupby(["clone", "day"]).size().unstack(fill_value=0)
    early = pc[[c for c in pc.columns if c <= 5]].sum(axis=1)
    return int(((early >= 1) & (pc.get(10, 0) >= 2)).sum())


def select_hspc():
    c6, trn, _, cf = _hspc_setup()
    K10 = c6.fate_counts(trn, 10)
    u = [c for c in K10.index[K10.sum(1) >= 2] if c in cf.index]
    k = K10.loc[u].to_numpy()
    f = np.array([_fold(c) for c in u])
    levels = sorted(trn["cond"].unique())
    oh, cd = c6.cond_onehot(trn, u, levels)
    Bc = c6.fold_bases(k, cd, f)
    Bp = c6.fold_bases(k, np.array(["pool"] * len(u)), f)
    n = k.sum(1)
    w = np.minimum(n, WCAP)
    g_cond = (k * (np.log2(Bc) - np.log2(Bp))).sum(1) / n
    ge = c6.cv_gain(np.c_[cf.loc[u].to_numpy(), oh], k, f, Bc)
    by = {str(c): wmean(ge[c] - g_cond, w) for c in c6.CS}
    C = max(c6.CS, key=lambda c: by[str(c)])
    d = ge[C] - g_cond
    neff = w.sum() ** 2 / (w ** 2).sum()
    eff, sd = wmean(d, w), float(np.std(d))
    _write("C8-S1b", {"test": "hspc_c8s1", "higher_is_better": True, "candidates": by, "chosen": str(C), "C": C,
                      "levels": levels,
                      "power": {"sd": sd, "effect": eff, "alpha": ALPHA, "one_sided": True, "min_power": 0.8},
                      "train": {"n": len(u), "G_cond": wmean(g_cond, w), "G_state": wmean(ge[C], w), "diff": eff, "sd": sd,
                                "neff_ratio": float(neff / len(u)), "struct_ratio": len(u) / _hspc_struct(trn),
                                "power_B_ratio_est": _power_z(eff, sd, neff * 0.5)},
                      "note": "T2 (K1): mean(g_state - g_cond) > 0 one-sided, exp 7 only; tested only if T1 confirmed"})


def counts_hspc(part):
    import c7s7_confirm as c7

    sel = json.load(open(os.path.join(SEL, "C8-S1b.json")))
    obs = c7._hspc_obs()[["clone", "day", "exp"]]  # no fate column read
    held = partitions.sealed_rows("chspctrack", obs[obs["clone"].notna()], "clone", part)
    n = _hspc_struct(held[held["exp"].astype(str) == EXP])
    return {"rna": int(np.floor(n * sel["train"]["struct_ratio"] * sel["train"]["neff_ratio"]))}


def test_hspc(part):
    rng = np.random.default_rng(92)
    sel = json.load(open(os.path.join(SEL, "C8-S1b.json")))
    c6, trn, held, cf = _hspc_setup(part)
    levels = sel["levels"]
    Kt, Kh = c6.fate_counts(trn, 10), c6.fate_counts(held, 10)
    ut = [c for c in Kt.index[Kt.sum(1) >= 2] if c in cf.index]
    uh = [c for c in Kh.index[Kh.sum(1) >= 2] if c in cf.index]
    oht, cdt = c6.cond_onehot(trn, ut, levels)
    ohh, cdh = c6.cond_onehot(held, uh, levels)
    kt, kh = Kt.loc[ut].to_numpy(), Kh.loc[uh].to_numpy()
    model = fit_model(np.c_[cf.loc[ut].to_numpy(), oht], kt, sel["C"])
    pool = (kt.sum(0) + 0.5) / (kt.sum(0) + 0.5).sum()
    Bc = np.zeros(kh.shape)
    for i, c in enumerate(cdh):
        s = kt[cdt == c].sum(0)
        Bc[i] = (s + 0.5) / (s + 0.5).sum() if s.sum() > 0 else pool
    P = np.full(kh.shape, 1e-6)
    P[:, model.classes_] = model.predict_proba(np.c_[cf.loc[uh].to_numpy(), ohh])
    P /= P.sum(1, keepdims=True)
    n = kh.sum(1)
    w = np.minimum(n, WCAP)
    g_state = (kh * (np.log2(P) - np.log2(Bc))).sum(1) / n
    g_cond = (kh * (np.log2(Bc) - np.log2(pool))).sum(1) / n
    d = g_state - g_cond
    bs = _boot_g(d, w, rng)
    lo = _lb(bs)
    rep = {k: {"G": wmean(v, w), "ci95": np.percentile(_boot_g(v, w, rng), [2.5, 97.5]).tolist()}
           for k, v in (("G_state", g_state), ("G_cond", g_cond))}
    return {"T2_state_minus_cond": {"n": len(uh), "diff": wmean(d, w), "lower_one_sided": lo,
                                    "ci95": np.percentile(bs, [2.5, 97.5]).tolist(), "confirmed": bool(lo > 0),
                                    "strata_unseen_in_train": sorted(set(cdh) - set(cdt)), **rep}}


if __name__ == "__main__":
    select_lsk()
    select_hspc()
