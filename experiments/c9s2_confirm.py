"""C9-S2: LSK day-2.5 chromatin -> day-5 Neu vs Mono split. Train selection (select_lsk) and the sealed test
(test_lsk, counts_lsk) for test_oracle.py. See EXPERIMENT_LOG C9-S2.   python c9s2_confirm.py   (train only)
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.stats import spearmanr

import partitions
from c7s3_carrier import WCAP, fit_model, wmean
from c7s7_confirm import ALPHA, _lb
from c8s0_scan import CS, _fold, cv_binomial
from c8s1_confirm import _nm_units

HERE = os.path.dirname(os.path.abspath(__file__))
SEL = os.path.join(HERE, "selection")
FISHER_SD = float(np.sqrt(1.06))
MIN_SIDE = 30


def _struct_tab():
    import c7s3_carrier as c3

    import pandas as pd

    return pd.read_csv(c3.CSV, usecols=["clone.id", "day", "assay"])  # no fate column


def select_lsk():
    import c7s3_carrier as c3
    from c9s0_scan import _lsk_struct

    tab = partitions.train_only("celltag_lsk", c3._table(), unit_col="clone.id")
    tc = tab["clone.id"].unique()
    F, sizes = c3.clone_features("atac", tc, tc)
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
    rhos = {str(c): float(spearmanr(preds[c], frac).statistic) for c in CS}
    rho = rhos[str(C)]
    any_neu, any_mono = int((Kn[:, 0] > 0).sum()), int((Kn[:, 1] > 0).sum())
    gate = {"any_neu": any_neu, "any_mono": any_mono, "rho_range_over_C": max(rhos.values()) - min(rhos.values()),
            "pass": bool(any_neu >= MIN_SIDE and any_mono >= MIN_SIDE and max(rhos.values()) - min(rhos.values()) > 1e-3)}
    st = _lsk_struct(partitions.train_only("celltag_lsk", _struct_tab(), unit_col="clone.id"))
    sel = {"test": "celltag_c9s2_atac", "higher_is_better": True, "candidates": by, "chosen": str(C), "C": C,
           "power": {"sd": FISHER_SD, "effect": float(np.arctanh(max(rho, 0))), "alpha": ALPHA, "one_sided": True,
                     "min_power": 0.8},
           "train": {"n": int(len(K)), "cv_rho": rho, "by_C_rho": rhos, "neu_share_mean": float(frac.mean()),
                     "struct_train": st, "unit_ratio": len(K) / st},
           "readout_gate": gate,
           "note": "C9-S2: Spearman rho(pred P(Neu), observed Neu share) > 0 one-sided; C by CV gain; Fisher-z power"}
    os.makedirs(SEL, exist_ok=True)
    json.dump(sel, open(os.path.join(SEL, "C9-S2.json"), "w"), indent=1, default=float)
    print("C9-S2", json.dumps(sel, indent=1, default=float), flush=True)


def counts_lsk(part):
    from c9s0_scan import _lsk_struct

    sel = json.load(open(os.path.join(SEL, "C9-S2.json")))
    if not sel["readout_gate"]["pass"]:
        return {"atac": 0}
    n = _lsk_struct(partitions.sealed_rows("celltag_lsk", _struct_tab(), "clone.id", part))
    return {"atac": int(np.floor(n * sel["train"]["unit_ratio"])) - 3}


def test_lsk(part):
    import c7s3_carrier as c3

    sel = json.load(open(os.path.join(SEL, "C9-S2.json")))
    full = c3._table()
    trn = partitions.train_only("celltag_lsk", full, unit_col="clone.id")
    held = partitions.sealed_rows("celltag_lsk", full, "clone.id", part)
    tc, hc = trn["clone.id"].unique(), held["clone.id"].unique()
    F, sizes = c3.clone_features("atac", np.r_[tc, hc], tc)
    Kt = _nm_units(trn, sizes.reindex(tc).dropna())
    Kh = _nm_units(held, sizes.reindex(hc).dropna())
    m = fit_model(F.loc[Kt.index].to_numpy(), Kt.to_numpy(), sel["C"])
    P = np.zeros((len(Kh), 2))
    P[:, m.classes_] = m.predict_proba(F.loc[Kh.index].to_numpy())
    p = P[:, 0]
    frac = Kh["Neu"].to_numpy() / Kh.sum(1).to_numpy()
    rho = float(spearmanr(p, frac).statistic)
    rng = np.random.default_rng(95)
    bs = [spearmanr(p[ix], frac[ix]).statistic for ix in (rng.integers(0, len(p), len(p)) for _ in range(2000))]
    bs = np.array([x for x in bs if np.isfinite(x)])
    lo = _lb(bs)
    return {"rho_atac": {"n": int(len(p)), "rho": rho, "lower_one_sided": lo,
                         "ci95": np.percentile(bs, [2.5, 97.5]).tolist(), "confirmed": bool(lo > 0)}}


if __name__ == "__main__":
    select_lsk()
