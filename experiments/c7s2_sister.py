"""C7-S2: is blood fate a heritable clone state in CellTag-multi? Day-5 sister wells (B38 method).
See EXPERIMENT_LOG C7-S2. Train analysis + selection/power file; the sealed look runs only via test_oracle.

    python c7s2_sister.py [--synthetic syn0|syn1]     # train only
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

import partitions
import sis_data as sd

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "data", "ext_celltag_multi", "hsc.rna_atac.r1_2_master_v2.csv")
DATASET = "celltag_lsk"
GROUPS = {"Neutro": "Neu", "Mono": "Mono", "Baso/Eos/Mast": "BEM", "Ery/Meg": "EMk", "MEP": "EMk",
          "Ccr7_DC": "DC", "pDC": "DC", "Lym": "Lym", "MPP": "Prog", "MPP/GMP": "Prog"}
FATES = ["Neu", "Mono", "BEM", "EMk", "DC", "Lym", "Prog"]
MIN_CELLS = 4
OUT = os.path.join(HERE, "results_celltag")


def _load():
    df = pd.read_csv(CSV, usecols=["clone.id", "cell.bc", "day", "assay", "cell_type2"])
    return df[(df["day"] == "d5") & (df["assay"] == "rna")]


def _counts(df, fates):
    df = df.assign(well=df["cell.bc"].str.extract(r"^d5(r[12])-")[0], grp=df["cell_type2"].map(GROUPS))
    df = df[df["well"].notna() & df["grp"].isin(fates)]
    tab = df.groupby(["clone.id", "well", "grp"]).size().unstack("grp", fill_value=0).reindex(columns=fates, fill_value=0)
    tab = tab.unstack("well", fill_value=0)
    k1 = tab.xs("r1", axis=1, level="well").to_numpy()
    k2 = tab.xs("r2", axis=1, level="well").to_numpy()
    m = (k1.sum(1) >= MIN_CELLS) & (k2.sum(1) >= MIN_CELLS)
    return k1[m], k2[m]


def analyse(k1, k2, fates, rng, boot=1000):
    n = len(k1)
    grp = np.zeros(n, int)
    h = sd.heritable_share(k1, k2, grp, rng)
    hb = [sd.heritable_share(k1[ix], k2[ix], grp[ix], rng, 100, 50)["H"]
          for ix in (rng.integers(0, n, n) for _ in range(boot))]
    sig = sd.clone_logit_sd(k1, k2)
    sig_ci = sd.bootstrap(lambda ix: sd.clone_logit_sd(k1[ix], k2[ix]), n, rng, boot)
    return {"n_clones": int(n), **{k: float(v) for k, v in h.items()},
            "H_ci": np.nanpercentile(hb, [2.5, 97.5]).tolist(), "H_se_boot": float(np.nanstd(hb)),
            "sigma_logit": dict(zip(fates, sig)),
            "sigma_ci": {f: [sig_ci[0][j], sig_ci[1][j]] for j, f in enumerate(fates)}}


def _loglik(k1, k2):
    """Per-cell log-likelihood of each well given the other (memory) vs population fractions (no memory)."""
    pi = (k1 + k2).sum(0) / (k1 + k2).sum()
    out = {}
    for name in ("memory", "no_memory"):
        ll, nc = 0.0, 0
        for a, b in ((k1, k2), (k2, k1)):
            p = (a + pi) / (a.sum(1, keepdims=True) + 1) if name == "memory" else np.tile(pi, (len(a), 1))
            ll += float((b * np.log(np.maximum(p, 1e-12))).sum())
            nc += int(b.sum())
        out[name] = ll / nc
    return out


def train(synthetic=None, boot=1000):
    rng = np.random.default_rng(72)
    df = partitions.train_only("celltag_lsk", _load(), unit_col="clone.id")
    res = {"synthetic": synthetic}
    for label, fates in (("all", FATES), ("differentiated", FATES[:-1])):
        k1, k2 = _counts(df, fates)
        if synthetic:
            k1, k2 = sd.synthetic(k1, k2, np.zeros(len(k1), int), synthetic, rng)
        res[label] = analyse(k1, k2, fates, rng, boot)
        if label == "all":
            res["selection_scores"] = _loglik(k1, k2)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, f"c7s2_train{'_' + synthetic if synthetic else ''}.json"), "w"), indent=1)
    print(json.dumps(res, indent=1, default=float))
    return res


def write_selection(res):
    a = res["all"]
    sel = {"test": "celltag_c7s2", "higher_is_better": True, "candidates": res["selection_scores"],
           "chosen": max(res["selection_scores"], key=res["selection_scores"].get),
           "power": {"sd": a["H_se_boot"] * np.sqrt(a["n_clones"]), "effect": a["H"] - 0.5, "alpha": 0.025,
                     "one_sided": True, "min_power": 0.8},
           "note": "C7-S2: H > 0.5 one-sided (95% lower bound); sd = train bootstrap SE x sqrt(n_train)"}
    os.makedirs(os.path.join(HERE, "selection"), exist_ok=True)
    json.dump(sel, open(os.path.join(HERE, "selection", "C7-S2.json"), "w"), indent=1, default=float)
    print(json.dumps(sel, indent=1, default=float))


def _eligible_structural(df):
    w = df.assign(well=df["cell.bc"].str.extract(r"^d5(r[12])-")[0]).dropna(subset=["well"])
    t = w.groupby(["clone.id", "well"]).size().unstack(fill_value=0)
    return int(((t.get("r1", 0) >= MIN_CELLS) & (t.get("r2", 0) >= MIN_CELLS)).sum())


def counts(part):
    """Units per sealed partition (structure only: no fate column read). The structural count is an upper
    bound (fate-label exclusions need the fate column), so it is scaled by the TRAIN ratio
    eligible-after-exclusions / eligible-structural (conservative, rounded down)."""
    df = pd.read_csv(CSV, usecols=["clone.id", "cell.bc", "day", "assay"])
    df = df[(df["day"] == "d5") & (df["assay"] == "rna")]
    tr = partitions.train_only(DATASET, _load(), unit_col="clone.id")
    ratio = len(_counts(tr, FATES)[0]) / _eligible_structural(tr)
    n = _eligible_structural(partitions.sealed_rows(DATASET, df, "clone.id", part))
    return {"clones": int(np.floor(n * ratio))}


def test(part):
    """Sealed look (oracle only): the pre-registered statistics on one partition."""
    rng = np.random.default_rng(73)
    df = partitions.sealed_rows(DATASET, _load(), "clone.id", part)
    out = {}
    for label, fates in (("all", FATES), ("differentiated", FATES[:-1])):
        k1, k2 = _counts(df, fates)
        out[label] = analyse(k1, k2, fates, rng)
    a = out["all"]
    out["reading"] = ("fixed" if a["H"] >= 0.7 and a["H_ci"][0] > 0.5 else
                      "partly fixed" if a["H_ci"][0] > 0 else "no clone memory")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", choices=["syn0", "syn1"])
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--select", action="store_true", help="write selection/C7-S2.json from the train run")
    a = ap.parse_args()
    r = train(a.synthetic, a.boot)
    if a.select and not a.synthetic:
        write_selection(r)
