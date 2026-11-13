"""C9-S1: chapter 9 train selection (select_hspc). WITHDRAWN 2026-11-11: the outcome was degenerate, so no
sealed look was run. The sealed test and count functions (test_hspc, counts_hspc) were removed on 2026-11-13
(Bence: "go 1"), so no code path to the sealed HSPC rows remains for a withdrawn test; they are in git
history (f4b93bb and earlier).
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
from c7s3_carrier import WCAP, wmean
from c7s7_confirm import ALPHA
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


if __name__ == "__main__":
    select_hspc()
