"""C7-S7: chapter 7 confirmatory looks (one per dataset, partition A, fixed-sequence across the chapter).
See EXPERIMENT_LOG C7-S7. `python c7s7_confirm.py select [lsk|reprog|hspc]` writes the train selection/power
files (train only); the sealed statistics run only through test_oracle (test_lsk / test_reprog / test_hspc).

  C7-S7a celltag_lsk:    H (>= 2 cells per well) > 0.5;  G_rna > 0;  G_atac > 0
  C7-S7b celltag_reprog: G_rna > 0
  C7-S7c chspctrack:     G_early (RNA, d3-5 -> d10, stratum = experiment x condition) > 0
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

import partitions

HERE = os.path.dirname(os.path.abspath(__file__))
SEL = os.path.join(HERE, "selection")
ALPHA = 0.025  # one-sided, each test; fixed sequence across the chapter


def _lb(vals):
    return float(np.percentile(vals, 100 * ALPHA))


def _boot_g(g, w, rng, n=2000):
    from c7s3_carrier import wmean

    return [wmean(g[ix], w[ix]) for ix in (rng.integers(0, len(g), len(g)) for _ in range(n))]


def _power(G, sd, n):
    return float(1 - norm.cdf(norm.ppf(1 - ALPHA) - G / (sd / np.sqrt(n))))


def _write(exp, sel):
    os.makedirs(SEL, exist_ok=True)
    json.dump(sel, open(os.path.join(SEL, f"{exp}.json"), "w"), indent=1, default=float)
    print(exp, json.dumps(sel, indent=1, default=float), flush=True)


# =============================== C7-S7a: LSK ===============================
def _lsk_counts2(df):
    import c7s2_sister as s2

    df = df.assign(well=df["cell.bc"].str.extract(r"^d5(r[12])-")[0], grp=df["cell_type2"].map(s2.GROUPS))
    df = df[df["well"].notna() & df["grp"].isin(s2.FATES)]
    tab = df.groupby(["clone.id", "well", "grp"]).size().unstack("grp", fill_value=0).reindex(columns=s2.FATES, fill_value=0)
    tab = tab.unstack("well", fill_value=0)
    k1, k2 = tab.xs("r1", axis=1, level="well").to_numpy(), tab.xs("r2", axis=1, level="well").to_numpy()
    m = (k1.sum(1) >= 2) & (k2.sum(1) >= 2)
    return k1[m], k2[m]


def select_lsk():
    import c7s2_sister as s2
    import c7s3_carrier as c3

    rng = np.random.default_rng(81)
    d5 = partitions.train_only("celltag_lsk", s2._load(), unit_col="clone.id")
    k1, k2 = _lsk_counts2(d5)
    h = s2.analyse(k1, k2, s2.FATES, rng, boot=300)
    tab = partitions.train_only("celltag_lsk", c3._table(), unit_col="clone.id")
    K_all = c3._fates(tab)
    tc = tab["clone.id"].unique()
    out = {}
    for mod in ("rna", "atac"):
        r = c3.cv_modality(mod, tc, K_all, rng)
        neff = r["w"].sum() ** 2 / (r["w"] ** 2).sum()
        G, sd = c3.wmean(r["g"], r["w"]), float(np.std(r["g"]))
        out[mod] = {"C": r["C"], "by_C": r["by_C"], "G": G, "sd": sd, "n": int(len(r["g"])),
                    "neff_ratio": float(neff / len(r["g"])), "power_A_ratio_est": _power(G, sd, neff * 1348 / 2720)}
    hp = _power(h["H"] - 0.5, h["H_se_boot"] * np.sqrt(h["n_clones"]), h["n_clones"] * 1348 / 2720)
    a = out["atac"]
    _write("C7-S7a", {"test": "celltag_c7s7_lsk", "higher_is_better": True,
                      "candidates": a["by_C"], "chosen": str(a["C"]),
                      "C": {"rna": out["rna"]["C"], "atac": a["C"]},
                      "power": {"sd": a["sd"], "effect": a["G"], "alpha": ALPHA, "one_sided": True, "min_power": 0.8},
                      "train": {"H_ge2": {k: h[k] for k in ("n_clones", "H", "H_ci", "H_se_boot")}, "H_power_A_est": hp,
                                **out},
                      "note": "power gate on the weakest test (G_atac > 0); counts = effective ATAC units"})


def counts_lsk(part):
    import c7s2_sister as s2

    sel = json.load(open(os.path.join(SEL, "C7-S7a.json")))
    tab = pd.read_csv(s2.CSV, usecols=["clone.id", "cell.bc", "day", "assay"])

    def struct(t):
        early = set(t.loc[(t["day"] == "d2") & (t["assay"] == "atac"), "clone.id"])
        d5 = t[t["day"] == "d5"].groupby("clone.id").size()
        return len(early & set(d5[d5 >= 2].index))

    trn = partitions.train_only("celltag_lsk", tab, unit_col="clone.id")
    ratio = sel["train"]["atac"]["n"] / struct(trn)
    n = struct(partitions.sealed_rows("celltag_lsk", tab, "clone.id", part))
    return {"atac": int(np.floor(n * ratio * sel["train"]["atac"]["neff_ratio"]))}


def test_lsk(part):
    import c7s2_sister as s2
    import c7s3_carrier as c3

    rng = np.random.default_rng(82)
    sel = json.load(open(os.path.join(SEL, "C7-S7a.json")))
    out = {}
    held5 = partitions.sealed_rows("celltag_lsk", s2._load(), "clone.id", part)
    k1, k2 = _lsk_counts2(held5)
    h = s2.analyse(k1, k2, s2.FATES, rng)
    lo = h["H"] - norm.ppf(1 - ALPHA) * h["H_se_boot"]
    out["H_ge2"] = {"n": h["n_clones"], "H": h["H"], "ci95": h["H_ci"], "lower_one_sided": lo,
                    "confirmed": bool(lo > 0.5)}
    full = c3._table()
    trn = partitions.train_only("celltag_lsk", full, unit_col="clone.id")
    held = partitions.sealed_rows("celltag_lsk", full, "clone.id", part)
    tc, hc = trn["clone.id"].unique(), held["clone.id"].unique()
    Kt, Kh = c3._fates(trn), c3._fates(held)
    for mod in ("rna", "atac"):
        F, sizes = c3.clone_features(mod, np.r_[tc, hc], tc)
        Ktr = c3._units(Kt, sizes.reindex(tc).dropna())
        Khe = c3._units(Kh, sizes.reindex(hc).dropna())
        model = c3.fit_model(F.loc[Ktr.index].to_numpy(), Ktr.to_numpy(), sel["C"][mod])
        base = Ktr.to_numpy().sum(0) / Ktr.to_numpy().sum()
        g, w = c3.gains(model, F.loc[Khe.index].to_numpy(), Khe.to_numpy(), base)
        bs = _boot_g(g, w, rng)
        out[f"G_{mod}"] = {"n": int(len(g)), "G": c3.wmean(g, w), "lower_one_sided": _lb(bs),
                           "ci95": np.percentile(bs, [2.5, 97.5]).tolist(), "confirmed": bool(_lb(bs) > 0)}
    return out


# =============================== C7-S7b: reprogramming ===============================
def _reprog_full():
    import c7s5_reprog as c5

    return pd.read_csv(c5.CSV, usecols=["clone_id", "cell.bc", "day", "assay", "cell_type"])


def select_reprog():
    import c7s5_reprog as c5
    from c7s3_carrier import wmean

    rng = np.random.default_rng(75)
    tab = c5._table()
    F, sizes, _, _ = c5.features("rna", tab, rng)
    r = c5.cv(F, sizes, c5._fates(tab), rng)
    neff = r["w"].sum() ** 2 / (r["w"] ** 2).sum()
    G, sd = wmean(r["g"], r["w"]), float(np.std(r["g"]))
    _write("C7-S7b", {"test": "celltag_c7s7_reprog", "higher_is_better": True, "candidates": r["by_C"],
                      "chosen": str(r["C"]), "C": r["C"],
                      "power": {"sd": sd, "effect": G, "alpha": ALPHA, "one_sided": True, "min_power": 0.8},
                      "train": {"G": G, "sd": sd, "n": int(len(r["g"])), "neff_ratio": float(neff / len(r["g"])),
                                "power_A_ratio_est": _power(G, sd, neff * 1561 / 3208)},
                      "note": "C7-S7b: G_rna > 0 one-sided; counts = effective RNA units"})


def counts_reprog(part):
    sel = json.load(open(os.path.join(SEL, "C7-S7b.json")))
    t = _reprog_full()[["clone_id", "day", "assay"]]

    def struct(x):
        early = set(x.loc[(x["day"] == "D3") & (x["assay"] == "RNA"), "clone_id"])
        f = x[x["day"].isin(["D12", "D21"])].groupby("clone_id").size()
        return len(early & set(f[f >= 2].index))

    ratio = sel["train"]["n"] / struct(partitions.train_only("celltag_reprog", t, unit_col="clone_id"))
    n = struct(partitions.sealed_rows("celltag_reprog", t, "clone_id", part))
    return {"rna": int(np.floor(n * ratio * sel["train"]["neff_ratio"]))}


def test_reprog(part):
    import c7s5_reprog as c5
    from c7s3_carrier import fit_model, wmean
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(75)
    sel = json.load(open(os.path.join(SEL, "C7-S7b.json")))
    full = _reprog_full()
    trn = partitions.train_only("celltag_reprog", full, unit_col="clone_id")
    held = partitions.sealed_rows("celltag_reprog", full, "clone_id", part)
    e_all = c5._early(full, "rna")
    et, eh = c5._early(trn, "rna"), c5._early(held, "rna")
    X, bc, _ = c5.load_universe("rna", set(e_all["bc"]), rng)  # every clonal cell known; non-clonal sampled
    other = set(e_all["bc"]) - set(et["bc"]) - set(eh["bc"])
    keep = ~bc.isin(list(other))  # other partitions' clonal cells excluded
    X, bc = X[np.flatnonzero(keep)], bc[keep]
    fit_rows = np.flatnonzero(~bc.isin(list(eh["bc"])))  # train clonal + non-clonal
    fz = c5.ChunkedFeaturizer("rna").fit(X[fit_rows])
    Fz = fz.transform(X)
    idx = NearestNeighbors(n_neighbors=20, metric="cosine").fit(Fz).kneighbors(Fz, return_distance=False)
    Fz = Fz[idx].mean(1)
    e = pd.concat([et, eh])
    e = e[e["bc"].isin(bc)]
    df = pd.DataFrame(Fz[bc.get_indexer(e["bc"])], index=e["clone_id"].to_numpy()).groupby(level=0).mean()
    ref = df.loc[df.index.isin(et["clone_id"])]
    df = (df - ref.mean()) / (ref.std() + 1e-9)
    Kt, Kh = c5._fates(trn), c5._fates(held)
    Kt = Kt.reindex(df.index.intersection(Kt.index))
    Kt = Kt[Kt.sum(1) >= 2]
    Kh = Kh.reindex(df.index.intersection(Kh.index))
    Kh = Kh[Kh.sum(1) >= 2]
    model = fit_model(df.loc[Kt.index].to_numpy(), Kt.to_numpy(), sel["C"])
    base = Kt.to_numpy().sum(0) / Kt.to_numpy().sum()
    g, w = c5.gains(model, df.loc[Kh.index].to_numpy(), Kh.to_numpy(), base)
    bs = _boot_g(g, w, rng)
    return {"G_rna": {"n": int(len(g)), "G": wmean(g, w), "lower_one_sided": _lb(bs),
                      "ci95": np.percentile(bs, [2.5, 97.5]).tolist(), "confirmed": bool(_lb(bs) > 0)}}


# =============================== C7-S7c: human HSPC ===============================
def _hspc_obs():
    import h5py

    import c7s6_hspc as c6

    with h5py.File(c6.H5AD, "r") as f:
        o = f["obs"]
        obs = pd.DataFrame({"clone": c6._cat(o, "clone_id"), "day": o["sample_day_as_int"][:],
                            "cond": c6._cat(o, "condition"), "ct": c6._cat(o, "top_level_cell_type"),
                            "exp": c6._cat(o, "exp_id")})
    obs["cond"] = obs["exp"].astype(str) + "|" + obs["cond"].fillna("unknown").astype(str)
    obs["row"] = np.arange(len(obs))
    return obs


def _hspc_features(early_fit, early_all, non):
    import c7s6_hspc as c6
    from sklearn.neighbors import NearestNeighbors

    X, rows = c6.read_rows(np.r_[early_all["row"].to_numpy(), non["row"].to_numpy()])
    pos = pd.Series(np.arange(len(rows)), index=rows)
    fit_idx = np.sort(pos.loc[np.r_[early_fit["row"].to_numpy(), non["row"].to_numpy()]].to_numpy())
    fz = c6.ChunkedFeaturizer("rna").fit(X[fit_idx])
    Fz = fz.transform(X)
    idx = NearestNeighbors(n_neighbors=20, metric="cosine").fit(Fz).kneighbors(Fz, return_distance=False)
    Fz = Fz[idx].mean(1)
    Fe = Fz[pos.loc[early_all["row"].to_numpy()].to_numpy()]
    return pd.DataFrame(Fe, index=early_all["clone"].to_numpy()).groupby(level=0).mean()


def select_hspc():
    import c7s6_hspc as c6
    from c7s3_carrier import WCAP, wmean

    obs = _hspc_obs()
    clonal = obs[obs["clone"].notna()]
    trn = partitions.train_only("chspctrack", clonal, unit_col="clone")
    non = obs[obs["clone"].isna() & (obs["day"] <= 5)]
    et = trn[trn["day"] <= 5]
    cf = _hspc_features(et, et, non)
    cf = (cf - cf.mean()) / (cf.std() + 1e-9)
    K10 = c6.fate_counts(trn, 10)
    u1 = [c for c in K10.index[K10.sum(1) >= 2] if c in cf.index]
    levels = sorted(trn["cond"].unique())
    k1 = K10.loc[u1].to_numpy()
    f1 = np.array([c6._fold(c) for c in u1])
    oh, cd = c6.cond_onehot(trn, u1, levels)
    B = c6.fold_bases(k1, cd, f1)
    w = np.minimum(k1.sum(1), WCAP)
    ge = c6.cv_gain(np.c_[cf.loc[u1].to_numpy(), oh], k1, f1, B)
    by = {str(c): wmean(ge[c], w) for c in c6.CS}
    C = max(c6.CS, key=lambda c: wmean(ge[c], w))
    g = ge[C]
    neff = w.sum() ** 2 / (w ** 2).sum()
    G, sd = wmean(g, w), float(np.std(g))
    _write("C7-S7c", {"test": "hspc_c7s7", "higher_is_better": True, "candidates": by, "chosen": str(C), "C": C,
                      "levels": levels,
                      "power": {"sd": sd, "effect": G, "alpha": ALPHA, "one_sided": True, "min_power": 0.8},
                      "train": {"G": G, "sd": sd, "n": len(u1), "neff_ratio": float(neff / len(u1)),
                                "power_A_ratio_est": _power(G, sd, neff * 0.5)},
                      "note": "C7-S7c: G_early > 0 one-sided; stratum = experiment x condition (C7-S6 amendment 1)"})


def counts_hspc(part):
    sel = json.load(open(os.path.join(SEL, "C7-S7c.json")))
    obs = _hspc_obs()
    clonal = obs[obs["clone"].notna()]

    def struct(x):
        pc = x.groupby(["clone", "day"]).size().unstack(fill_value=0)
        early = pc[[c for c in pc.columns if c <= 5]].sum(axis=1)
        return int(((early >= 1) & (pc.get(10, 0) >= 2)).sum())

    ratio = sel["train"]["n"] / struct(partitions.train_only("chspctrack", clonal, unit_col="clone"))
    n = struct(partitions.sealed_rows("chspctrack", clonal, "clone", part))
    return {"rna": int(np.floor(n * ratio * sel["train"]["neff_ratio"]))}


def test_hspc(part):
    import c7s6_hspc as c6
    from c7s3_carrier import WCAP, fit_model, wmean

    rng = np.random.default_rng(83)
    sel = json.load(open(os.path.join(SEL, "C7-S7c.json")))
    obs = _hspc_obs()
    clonal = obs[obs["clone"].notna()]
    trn = partitions.train_only("chspctrack", clonal, unit_col="clone")
    held = partitions.sealed_rows("chspctrack", clonal, "clone", part)
    non = obs[obs["clone"].isna() & (obs["day"] <= 5)]
    et, eh = trn[trn["day"] <= 5], held[held["day"] <= 5]
    cf = _hspc_features(et, pd.concat([et, eh]), non)
    ref = cf.loc[cf.index.isin(et["clone"])]
    cf = (cf - ref.mean()) / (ref.std() + 1e-9)
    levels = sel["levels"]
    Kt, Kh = c6.fate_counts(trn, 10), c6.fate_counts(held, 10)
    ut = [c for c in Kt.index[Kt.sum(1) >= 2] if c in cf.index]
    uh = [c for c in Kh.index[Kh.sum(1) >= 2] if c in cf.index]
    oht, cdt = c6.cond_onehot(trn, ut, levels)
    ohh, cdh = c6.cond_onehot(held, uh, levels)
    kt, kh = Kt.loc[ut].to_numpy(), Kh.loc[uh].to_numpy()
    model = fit_model(np.c_[cf.loc[ut].to_numpy(), oht], kt, sel["C"])
    allb = kt.sum(0) + 0.5
    B = np.zeros(kh.shape)
    for i, c in enumerate(cdh):
        s = kt[cdt == c].sum(0)
        B[i] = (s + 0.5) / (s + 0.5).sum() if s.sum() > 0 else allb / allb.sum()
    P = np.full((len(uh), kt.shape[1]), 1e-6)
    P[:, model.classes_] = model.predict_proba(np.c_[cf.loc[uh].to_numpy(), ohh])
    P /= P.sum(1, keepdims=True)
    g = (kh * (np.log2(P) - np.log2(B))).sum(1) / kh.sum(1)
    w = np.minimum(kh.sum(1), WCAP)
    bs = _boot_g(g, w, rng)
    return {"G_rna": {"n": len(uh), "G": wmean(g, w), "lower_one_sided": _lb(bs),
                      "ci95": np.percentile(bs, [2.5, 97.5]).tolist(), "confirmed": bool(_lb(bs) > 0),
                      "strata_unseen_in_train": sorted(set(cdh) - set(cdt))}}


if __name__ == "__main__":
    which = sys.argv[2] if len(sys.argv) > 2 else "all"
    if sys.argv[1] == "select":
        if which in ("all", "lsk"):
            select_lsk()
        if which in ("all", "reprog"):
            select_reprog()
        if which in ("all", "hspc"):
            select_hspc()
