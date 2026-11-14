"""C9-S3: power rescue for C9-S2 (LSK day-2.5 chromatin -> day-5 Neu vs Mono) on train + SPENT A/B only.

Serves: C9-S3. Bence: "think about c9-s2 and try to rescue the power". Pre-registered in EXPERIMENT_LOG ("C9-S3").
celltag_lsk:C (last partition) rows are dropped before any feature or fate step; A and B are spent (C7-S7a, C8-S1a)
and used here for fitting and out-of-sample effect estimates only, never as a test (H7).
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.stats import spearmanr

import partitions
import power
from c7s3_carrier import WCAP, fit_model, wmean
from c8s0_scan import CS, _fold, cv_binomial
from c8s1_confirm import _nm_units

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_c9", "c9s3_power.json")
DS = "celltag_lsk"
N_C = 97          # oracle's effective unit count for C under C9-S2's unit definition (counts_lsk)
C_FIXED = 0.01    # C9-S2 chosen C (R0)
rng = np.random.default_rng(96)


def _parts():
    import c7s3_carrier as c3

    full = c3._table()
    full = full[full["clone.id"].map(lambda u: partitions.partition_of(DS, u) != "C")]   # C dropped first
    return {"train": partitions.train_only(DS, full, "clone.id"),
            "A": partitions.sealed_rows(DS, full, "clone.id", "A"),      # spent (C7-S7a)
            "B": partitions.sealed_rows(DS, full, "clone.id", "B")}      # spent (C8-S1a)


def _gain(p, Kn):
    frac = Kn[:, 0] / Kn.sum(1)
    w = np.minimum(Kn.sum(1), WCAP)
    b = np.clip(frac.mean(), 1e-6, 1 - 1e-6)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return wmean((Kn[:, 0] * (np.log2(p) - np.log2(b)) + Kn[:, 1] * (np.log2(1 - p) - np.log2(1 - b))) / Kn.sum(1), w)


def fit_predict(train_tab, test_tab, choose_c):
    import c7s3_carrier as c3
    import pandas as pd

    tc, hc = train_tab["clone.id"].unique(), test_tab["clone.id"].unique()
    F, sizes = c3.clone_features("atac", np.r_[tc, hc], tc)
    Kt = _nm_units(train_tab, sizes.reindex(tc).dropna())
    Kh = _nm_units(test_tab, sizes.reindex(hc).dropna())
    C = C_FIXED
    if choose_c:
        folds = np.array([_fold(c) for c in Kt.index])
        preds = cv_binomial(F.loc[Kt.index].to_numpy(), Kt.to_numpy(), folds)
        by = {c: _gain(preds[c], Kt.to_numpy()) for c in CS}
        C = max(CS, key=by.get)
    m = fit_model(F.loc[Kt.index].to_numpy(), Kt.to_numpy(), C)
    P = np.zeros((len(Kh), 2))
    P[:, m.classes_] = m.predict_proba(F.loc[Kh.index].to_numpy())
    return pd.DataFrame({"p": P[:, 0], "neu": Kh["Neu"].to_numpy(), "mono": Kh["Mono"].to_numpy()}), C, len(Kt)


def s_spearman(d):
    return float(spearmanr(d["p"], d["neu"] / (d["neu"] + d["mono"])).statistic)


def s_wlogit(d):
    n = (d["neu"] + d["mono"]).to_numpy()
    x = np.log(np.clip(d["p"], 1e-6, 1 - 1e-6) / (1 - np.clip(d["p"], 1e-6, 1 - 1e-6))).to_numpy()
    q = (d["neu"].to_numpy() + 0.5) / (n + 1)
    y = np.log(q / (1 - q))
    w = np.minimum(n, WCAP).astype(float)
    xm, ym = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - xm) * (y - ym), weights=w)
    return float(cov / np.sqrt(np.average((x - xm) ** 2, weights=w) * np.average((y - ym) ** 2, weights=w)))


def assess(d, stat):
    s = stat(d)
    bs = []
    for _ in range(2000):
        b = d.iloc[rng.integers(0, len(d), len(d))]
        v = stat(b)
        if np.isfinite(v):
            bs.append(v)
    se = float(np.std(bs))
    pw = power.mean_power(s, se * np.sqrt(len(d)), {"atac": N_C}, alpha=0.025, one_sided=True)
    return {"n_oos": int(len(d)), "stat": round(s, 4), "se_oos": round(se, 4),
            "se_C": round(se * np.sqrt(len(d) / N_C), 4), "power_C": round(float(pw), 3), "go": bool(pw >= 0.8)}


def main():
    import pandas as pd

    P = _parts()
    res = {}
    # R0: current design, train -> A and B
    d0, c0, n0 = fit_predict(P["train"], pd.concat([P["A"], P["B"]]), choose_c=False)
    res["R0_train_only_spearman"] = {**assess(d0, s_spearman), "C": c0, "n_train": n0}
    print("R0", res["R0_train_only_spearman"], flush=True)
    # R1/R2: pooled training, leave-one-spent-partition-out
    dB, cB, nB = fit_predict(pd.concat([P["train"], P["A"]]), P["B"], choose_c=True)
    dA, cA, nA = fit_predict(pd.concat([P["train"], P["B"]]), P["A"], choose_c=True)
    d1 = pd.concat([dA, dB], ignore_index=True)
    res["R1_pooled_spearman"] = {**assess(d1, s_spearman), "C": {"->A": cA, "->B": cB}, "n_train": {"->A": nA, "->B": nB}}
    res["R2_pooled_wlogit"] = {**assess(d1, s_wlogit), "C": {"->A": cA, "->B": cB}}
    res["per_partition_spearman"] = {"A": round(s_spearman(dA), 4), "B": round(s_spearman(dB), 4)}
    res["R0_wlogit_for_reference"] = assess(d0, s_wlogit)
    go = [k for k in ("R1_pooled_spearman", "R2_pooled_wlogit", "R0_train_only_spearman") if res[k]["go"]]
    res["go_candidates"] = go
    res["chosen"] = max(go, key=lambda k: res[k]["power_C"]) if go else None
    res["N_C_effective"] = N_C
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
