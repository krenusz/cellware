"""C9-S1: chapter 9 train selection (select_hspc) and the sealed test (test_hspc, counts_hspc) for test_oracle.py.
Human HSPC (cHSPCTrack): early RNA -> erythroid/MK vs neutrophil/monocyte share, within experiment x condition.
See EXPERIMENT_LOG C9-S1 and CHAPTER9_PLAN.md.   python c9s1_confirm.py   (train selection only)
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import partitions
from c7s3_carrier import WCAP, fit_model, wmean
from c7s7_confirm import ALPHA, _lb
from c8s0_scan import CS, _fold, cv_binomial

HERE = os.path.dirname(os.path.abspath(__file__))
SEL = os.path.join(HERE, "selection")
FISHER_SD = float(np.sqrt(1.06))


def _write(sel):
    os.makedirs(SEL, exist_ok=True)
    json.dump(sel, open(os.path.join(SEL, "C9-S1.json"), "w"), indent=1, default=float)
    print("C9-S1", json.dumps(sel, indent=1, default=float), flush=True)


def _power_z(effect, n):
    from scipy.stats import norm

    return float(1 - norm.cdf(norm.ppf(1 - ALPHA) - effect / (FISHER_SD / np.sqrt(max(n, 1)))))


def _units(c6, x, cf):
    K10 = c6.fate_counts(x, 10)
    K = pd.DataFrame({"MegE": K10["Ery"] + K10["MK"], "GM": K10["Neu"] + K10["Mono"]})
    return K[(K.sum(1) >= 2) & K.index.isin(cf.index)]


def _centred(v, strata):
    return v - pd.Series(v).groupby(strata).transform("mean").to_numpy()


def _obs():
    import c7s7_confirm as c7

    obs = c7._hspc_obs()
    obs["exp"] = obs["exp"].astype(str)
    return obs


def _exps(trn):
    return sorted(e for e, g in trn.groupby("exp") if (g["day"] <= 5).any() and (g["day"] == 10).any())


def select_hspc():
    import c7s6_hspc as c6
    import c7s7_confirm as c7

    from c9s0_scan import _hspc_struct, _hspc_struct_obs

    obs = _obs()
    clonal = obs[obs["clone"].notna()]
    trn = partitions.train_only("chspctrack", clonal, unit_col="clone")
    exps = _exps(trn)
    trn = trn[trn["exp"].isin(exps)]
    non = obs[obs["clone"].isna() & (obs["day"] <= 5)]
    et = trn[trn["day"] <= 5]
    cf = c7._hspc_features(et, et, non)
    cf = (cf - cf.mean()) / (cf.std() + 1e-9)
    K = _units(c6, trn, cf)
    u = list(K.index)
    Kn = K.to_numpy()
    levels = sorted(trn["cond"].unique())
    oh, cd = c6.cond_onehot(trn, u, levels)
    folds = np.array([_fold(c) for c in u])
    preds = cv_binomial(np.c_[cf.loc[u].to_numpy(), oh], Kn, folds)
    frac = Kn[:, 0] / Kn.sum(1)
    w = np.minimum(Kn.sum(1), WCAP)
    b = np.clip(frac.mean(), 1e-6, 1 - 1e-6)

    def gain(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return wmean((Kn[:, 0] * (np.log2(p) - np.log2(b)) + Kn[:, 1] * (np.log2(1 - p) - np.log2(1 - b))) / Kn.sum(1), w)

    by = {str(c): gain(preds[c]) for c in CS}
    C = max(CS, key=lambda c: by[str(c)])
    rho = float(spearmanr(_centred(preds[C], cd), _centred(frac, cd)).statistic)
    ex = trn.groupby("clone")["exp"].first().reindex(u).to_numpy()
    per_exp = {e: {"n": int((ex == e).sum()),
                   "rho": float(spearmanr(_centred(preds[C], cd)[ex == e], _centred(frac, cd)[ex == e]).statistic)}
               for e in exps}
    so = _hspc_struct_obs()
    st = _hspc_struct(partitions.train_only("chspctrack", so, unit_col="clone"), exps)
    ratio = len(u) / st
    _write({"test": "hspc_c9s1", "higher_is_better": True, "candidates": by, "chosen": str(C), "C": C,
            "levels": levels, "exps": exps,
            "power": {"sd": FISHER_SD, "effect": float(np.arctanh(rho)), "alpha": ALPHA, "one_sided": True,
                      "min_power": 0.8},
            "train": {"n": len(u), "rho_within": rho, "rho_raw": float(spearmanr(preds[C], frac).statistic),
                      "by_C_rho_within": {str(c): float(spearmanr(_centred(preds[c], cd), _centred(frac, cd)).statistic) for c in CS},
                      "per_exp": per_exp, "units_with_any_GM": int((Kn[:, 1] > 0).sum()), "mege_share_mean": float(frac.mean()),
                      "struct_train": st, "unit_ratio": ratio},
            "note": "C9-S1: within-stratum Spearman rho > 0 one-sided; C by CV gain (H12); Fisher-z power"})


def counts_hspc(part):
    from c9s0_scan import _hspc_struct, _hspc_struct_obs

    sel = json.load(open(os.path.join(SEL, "C9-S1.json")))
    so = _hspc_struct_obs()  # clone/day/exp only; no fate column
    n = _hspc_struct(partitions.sealed_rows("chspctrack", so, "clone", part), sel["exps"])
    return {"rna": int(np.floor(n * sel["train"]["unit_ratio"])) - 3}


def test_hspc(part):
    import c7s6_hspc as c6
    import c7s7_confirm as c7

    sel = json.load(open(os.path.join(SEL, "C9-S1.json")))
    exps, levels = sel["exps"], sel["levels"]
    obs = _obs()
    clonal = obs[obs["clone"].notna()]
    trn = partitions.train_only("chspctrack", clonal, unit_col="clone")
    held = partitions.sealed_rows("chspctrack", clonal, "clone", part)
    trn, held = trn[trn["exp"].isin(exps)], held[held["exp"].isin(exps)]
    non = obs[obs["clone"].isna() & (obs["day"] <= 5)]
    et, eh = trn[trn["day"] <= 5], held[held["day"] <= 5]
    cf = c7._hspc_features(et, pd.concat([et, eh]), non)
    ref = cf.loc[cf.index.isin(et["clone"])]
    cf = (cf - ref.mean()) / (ref.std() + 1e-9)
    Kt, Kh = _units(c6, trn, cf), _units(c6, held, cf)
    ut, uh = list(Kt.index), list(Kh.index)
    oht, _ = c6.cond_onehot(trn, ut, levels)
    ohh, cdh = c6.cond_onehot(held, uh, levels)
    m = fit_model(np.c_[cf.loc[ut].to_numpy(), oht], Kt.to_numpy(), sel["C"])
    P = np.zeros((len(uh), 2))
    P[:, m.classes_] = m.predict_proba(np.c_[cf.loc[uh].to_numpy(), ohh])
    p = P[:, 0]
    Kn = Kh.to_numpy()
    frac = Kn[:, 0] / Kn.sum(1)
    pc, fc = _centred(p, cdh), _centred(frac, cdh)
    rho = float(spearmanr(pc, fc).statistic)
    rng = np.random.default_rng(94)
    bs = [spearmanr(pc[ix], fc[ix]).statistic for ix in (rng.integers(0, len(p), len(p)) for _ in range(2000))]
    bs = np.array([x for x in bs if np.isfinite(x)])
    lo = _lb(bs)
    ex = held.groupby("clone")["exp"].first().reindex(uh).to_numpy()
    per_exp = {e: {"n": int((ex == e).sum()), "rho": float(spearmanr(pc[ex == e], fc[ex == e]).statistic)}
               for e in exps if (ex == e).sum() > 3}
    return {"rho_within": {"n": len(uh), "rho": rho, "lower_one_sided": lo,
                           "ci95": np.percentile(bs, [2.5, 97.5]).tolist(), "confirmed": bool(lo > 0),
                           "rho_raw": float(spearmanr(p, frac).statistic), "per_exp": per_exp,
                           "units_with_any_GM": int((Kn[:, 1] > 0).sum()),
                           "strata_unseen_in_train": sorted(set(cdh) - set(levels))}}


if __name__ == "__main__":
    select_hspc()
