"""C17-S1C-S1: time-split predictability in the CRISPR brain split clones (train only).

Bence 2026-11-19: "yes" (pre-register the S1 time-split test). Serves: C17-S1 (C17-S1R: grow the systems; neural
class). Pre-registered in EXPERIMENT_LOG (C17-S1C-S1). Unit = split clone (partitions.crest_split_clone), eligible =
>= 3 in vivo cells (arm 1) and >= 4 organoid cells (arm 2).

Statistic P (leave-one-out kernel predictability): each clone's arm-2 composition f2 is predicted from the other
clones' f2, weighted by the similarity of their arm-1 compositions f1 (Gaussian kernel, bandwidth = median pairwise
squared distance of f1). P = 1 - mean ||f2 - f2_hat||^2 / mean ||f2 - mean of the other clones' f2||^2.
Null: f1 rows permuted within strata = embryo x arm-1 size bin (<= 5, >= 6) x arm-2 size bin (<= 7, >= 8), so
embryo and clone-size effects cannot create a signal. Reported: P_excess = P - mean null P, one-sided p.

Stages, in order (each refuses to run unless the previous gate passed):
    python -I experiments/c17s1c_s1.py selftest  # offline
    python -I experiments/c17s1c_s1.py smoke     # synthetic: null, embryo-offset null, size-only null, planted
    python -I experiments/c17s1c_s1.py power     # synthetic, at the train arm sizes (results_c17/c17s1c_sizes.json)
    python -I experiments/c17s1c_s1.py readout   # train: type frequencies per arm, collapse rule (H21)
    python -I experiments/c17s1c_s1.py train     # train: P, P_excess, p, clone bootstrap CI (within embryo)
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results_c17"
DS = "crest_split_clone"
ALPHA = 0.025
GEO = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE210nnn/GSE210139/suppl/"
SHA = "a3dbceaac047ade7c1c07d6b9caba78cb7e2bfc35810d4c8ac7894019a0c2462"  # recorded at download (SHA256SUMS)


# --- statistic ------------------------------------------------------------------------------------------------


def _sqdist(F):
    s = (F**2).sum(1)
    return np.maximum(s[:, None] + s[None, :] - 2 * F @ F.T, 0.0)


def kernel(F1):
    d = _sqdist(F1)
    off = d[~np.eye(len(d), dtype=bool)]
    h = np.median(off)
    if h <= 0:
        h = off.mean() if off.mean() > 0 else 1.0
    W = np.exp(-d / h)
    np.fill_diagonal(W, 0.0)
    return W


def predictability(W, F2, ids=None):
    """ids (amendment 1): original clone id per row; copies of a row's clone are excluded from its prediction
    and its population mean (bootstrap resamples)."""
    n = len(F2)
    if ids is None:
        hat = W @ F2 / W.sum(1, keepdims=True)
        pop = (F2.sum(0)[None, :] - F2) / (n - 1)
    else:
        same = ids[:, None] == ids[None, :]
        W = np.where(same, 0.0, W)
        hat = W @ F2 / W.sum(1, keepdims=True)
        keep = (~same).astype(float)
        pop = keep @ F2 / keep.sum(1, keepdims=True)
    return 1.0 - ((F2 - hat) ** 2).sum(1).mean() / ((F2 - pop) ** 2).sum(1).mean()


def strata_of(embryo, a1, a2):
    return np.array([f"{e}|{int(x >= 6)}|{int(y >= 8)}" for e, x, y in zip(embryo, a1, a2, strict=True)])


def perm_test(F1, F2, strata, n_perm, rng, ids=None):
    W = kernel(F1)
    obs = predictability(W, F2, ids)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    null = np.empty(n_perm)
    for b in range(n_perm):
        idx = np.arange(len(F1))
        for g in groups:
            idx[g] = rng.permutation(g)
        null[b] = predictability(W[np.ix_(idx, idx)], F2, ids)
    p = (1 + (null >= obs).sum()) / (n_perm + 1)
    return {"P": float(obs), "null_mean": float(null.mean()), "P_excess": float(obs - null.mean()), "p": float(p)}


# --- synthetic generator ----------------------------------------------------------------------------------------


def simulate(a1, a2, embryo, rng, H, sigma=1.0, K=6, d=2, emb_offset=0.0, size_effect=0.0):
    """Clone latent shared between arms with correlation H (H = 0: no memory). Arm-specific loadings, so the two
    arms use different type vocabularies. emb_offset and size_effect add non-memory structure (smoke nulls)."""
    n = len(a1)
    base = np.log(np.linspace(1.0, 0.25, K))
    A1, A2 = rng.normal(size=(K, d)) / np.sqrt(d), rng.normal(size=(K, d)) / np.sqrt(d)
    z = rng.normal(size=(n, d))
    u1 = np.sqrt(H) * z + np.sqrt(1 - H) * rng.normal(size=(n, d))
    u2 = np.sqrt(H) * z + np.sqrt(1 - H) * rng.normal(size=(n, d))
    emb = {k: rng.normal(size=K) * emb_offset for k in np.unique(embryo)}
    eo = np.array([emb[k] for k in embryo])
    tilt = np.zeros(K)
    tilt[0] = size_effect

    def z_(a):
        return (a - a.mean()) / a.std() if a.std() > 0 else 0.0 * a

    L1 = base + sigma * u1 @ A1.T + eo + z_(a1)[:, None] * tilt
    L2 = base + sigma * u2 @ A2.T + eo + z_(a2)[:, None] * tilt

    def draw(L, m):
        P = np.exp(L - L.max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        return np.array([rng.multinomial(k, p) / k for k, p in zip(m, P, strict=True)])

    return draw(L1, a1), draw(L2, a2)


def sizes():
    s = json.load(open(OUT / "c17s1c_sizes.json", encoding="utf-8"))
    return np.array(s["arm1"]), np.array(s["arm2"]), np.array(s["embryo"])


def rate(a1, a2, emb, rng, n_sim, n_perm, **kw):
    st = strata_of(emb, a1, a2)
    res = [perm_test(*simulate(a1, a2, emb, rng, **kw), st, n_perm, rng) for _ in range(n_sim)]
    return {"reject_rate": float(np.mean([r["p"] <= ALPHA for r in res])),
            "P_excess_mean": float(np.mean([r["P_excess"] for r in res]))}


def smoke(n_sim=200, n_perm=199):
    rng = np.random.default_rng(20261119)
    a1, a2, emb = sizes()
    out = {"null": rate(a1, a2, emb, rng, n_sim, n_perm, H=0.0),
           "null_embryo_offset": rate(a1, a2, emb, rng, n_sim, n_perm, H=0.0, emb_offset=1.0),
           "null_size_only": rate(a1, a2, emb, rng, n_sim, n_perm, H=0.0, size_effect=1.0),
           "planted_H0.9": rate(a1, a2, emb, rng, n_sim, n_perm, H=0.9, sigma=1.5)}
    out["pass"] = bool(all(out[k]["reject_rate"] <= 0.05 for k in ("null", "null_embryo_offset", "null_size_only"))
                       and out["planted_H0.9"]["reject_rate"] >= 0.9)
    return out


def power(n_sim=200, n_perm=199):
    rng = np.random.default_rng(20261120)
    a1, a2, emb = sizes()
    grid = {}
    for H in (0.33, 0.66, 0.81):
        for sigma in (0.5, 1.0, 1.5):
            for K in (6, 8):
                grid[f"H{H}_s{sigma}_K{K}"] = rate(a1, a2, emb, rng, n_sim, n_perm, H=H, sigma=sigma, K=K)
    prim = min(grid["H0.66_s1.0_K6"]["reject_rate"], grid["H0.66_s1.0_K8"]["reject_rate"])
    return {"grid": grid, "primary_power_H0.66_sigma1": prim, "gate_pass": bool(prim >= 0.8)}


# --- real data (train only; run only after the smoke and power gates) -------------------------------------------


def _gate(name, key):
    p = OUT / f"c17s1c_s1_{name}.json"
    if not p.exists() or not json.load(open(p, encoding="utf-8")).get(key):
        sys.exit(f"refused: {name} gate not passed (pre-registered order: smoke, power, readout, train)")


def train_cells():
    sys.path.insert(0, str(HERE))  # -I drops the script dir; these modules are ours
    import c17s1c_crest as cc  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415
    from partitions import train_only  # noqa: PLC0415

    path = cc.RAW / cc.FILE
    if not path.exists():  # runner: fetch the 2.7 MB GEO table into its own directory
        path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(GEO + cc.FILE, headers={"User-Agent": "cellware/1.0"})
        with urllib.request.urlopen(req, timeout=300) as r:
            path.write_bytes(r.read())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SHA, "barcode table changed since download (H4)"
    df = cc.load(path)
    cl = cc.split_clones(df)
    cl = cl[(cl["arm1"] >= 3) & (cl["arm2"] >= 4)].assign(unit=lambda c: c["embryo"] + ":" + c["pattern"])
    cl = train_only(DS, cl, "unit")
    rows = []
    for r in cl.itertuples():
        for arm, key in ((1, f"pandaE11_rep{r.embryo}"), (2, f"OGN_rep{r.embryo}")):
            g = df[(df["pattern.v2"] == r.pattern) & (df["sample_key"] == key)]
            rows += [{"unit": r.unit, "embryo": r.embryo, "arm": arm, "type": str(t)} for t in g["Cell.type"]]
    return pd.DataFrame(rows)


def collapse(freq, min_share=0.05, cap=8):
    """Pre-registered collapse: keep types with >= 5% of the arm's train cells, at most 8; the rest = 'other'."""
    keep = [t for t, f in sorted(freq.items(), key=lambda x: -x[1]) if f >= min_share][:cap]
    return {t: (t if t in keep else "other") for t in freq}


def readout():
    _gate("smoke", "pass")
    _gate("power", "gate_pass")
    cells = train_cells()
    res = {}
    for arm in (1, 2):
        c = cells[cells["arm"] == arm]
        freq = (c["type"].value_counts() / len(c)).round(4).to_dict()
        res[f"arm{arm}"] = {"cells": len(c), "types": len(freq), "frequency": freq,
                            "kept": sorted(set(collapse(freq).values()) - {"other"})}
    res["units"] = int(cells["unit"].nunique())
    res["gate_pass"] = bool(res["units"] >= 30 and all(len(res[f"arm{a}"]["kept"]) >= 2 for a in (1, 2)))
    return res


def train(n_perm=9999, n_boot=1000):
    _gate("readout", "gate_pass")
    cells = train_cells()
    units = sorted(cells["unit"].unique())
    F, n = {}, {}
    for arm in (1, 2):
        c = cells[cells["arm"] == arm]
        m = collapse((c["type"].value_counts() / len(c)).to_dict())
        tab = c.assign(t=c["type"].map(m)).groupby(["unit", "t"]).size().unstack(fill_value=0)
        tab = tab.reindex(index=units, columns=sorted(set(m.values())), fill_value=0)
        n[arm] = tab.sum(1).to_numpy()
        F[arm] = tab.to_numpy() / n[arm][:, None]
    emb = np.array([u.split(":")[0] for u in units])
    st = strata_of(emb, n[1], n[2])
    rng = np.random.default_rng(20261121)
    res = perm_test(F[1], F[2], st, n_perm, rng)
    boot = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(np.flatnonzero(emb == e), (emb == e).sum()) for e in np.unique(emb)])
        boot.append(perm_test(F[1][idx], F[2][idx], st[idx], 99, rng)["P_excess"])
    res["P_excess_ci95"] = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
    res["units"] = len(units)
    res["per_embryo_units"] = {e: int((emb == e).sum()) for e in np.unique(emb)}
    return res


def boot_ci(F1, F2, emb, st, rng, n_boot=1000, n_perm=99):
    """Amendment 1: resample clones within embryo; copies never predict each other."""
    out = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(np.flatnonzero(emb == e), (emb == e).sum()) for e in np.unique(emb)])
        out.append(perm_test(F1[idx], F2[idx], st[idx], n_perm, rng, ids=idx)["P_excess"])
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def ci_check(n_sim=40, n_boot=200, n_perm=49):
    rng = np.random.default_rng(20261122)
    a1, a2, emb = sizes()
    st = strata_of(emb, a1, a2)
    res = {}
    for name, kw in (("null", {"H": 0.0}), ("planted", {"H": 0.81, "sigma": 1.5})):
        hits = []
        for _ in range(n_sim):
            F1, F2 = simulate(a1, a2, emb, rng, **kw)
            pt = perm_test(F1, F2, st, 199, rng)["P_excess"]
            lo, hi = boot_ci(F1, F2, emb, st, rng, n_boot, n_perm)
            hits.append(lo <= 0 if name == "null" else lo <= pt <= hi)
        res[name] = float(np.mean(hits))
    res["pass"] = bool(res["null"] >= 0.9 and res["planted"] >= 0.8)
    return res


def ci():
    chk = ci_check()
    res = {"check": chk}
    if not chk["pass"]:
        return res
    cells = train_cells()
    units = sorted(cells["unit"].unique())
    F, n = {}, {}
    for arm in (1, 2):
        c = cells[cells["arm"] == arm]
        m = collapse((c["type"].value_counts() / len(c)).to_dict())
        tab = c.assign(t=c["type"].map(m)).groupby(["unit", "t"]).size().unstack(fill_value=0)
        tab = tab.reindex(index=units, columns=sorted(set(m.values())), fill_value=0)
        n[arm] = tab.sum(1).to_numpy()
        F[arm] = tab.to_numpy() / n[arm][:, None]
    emb = np.array([u.split(":")[0] for u in units])
    st = strata_of(emb, n[1], n[2])
    rng = np.random.default_rng(20261123)
    res["P_excess_point_recheck"] = perm_test(F[1], F[2], st, 999, rng)["P_excess"]
    res["P_excess_ci95"] = boot_ci(F[1], F[2], emb, st, rng)
    return res


def selftest():
    rng = np.random.default_rng(0)
    a1, a2 = np.full(60, 6), np.full(60, 8)
    emb = np.array(["1", "2", "3"] * 20)
    st = strata_of(emb, a1, a2)
    strong = perm_test(*simulate(a1, a2, emb, rng, H=1.0, sigma=3.0), st, 199, rng)
    none = perm_test(*simulate(a1, a2, emb, rng, H=0.0, sigma=3.0), st, 199, rng)
    assert strong["p"] <= 0.01 and strong["P_excess"] > 0.05, strong
    assert none["p"] > 0.01, none
    assert collapse({"a": 0.6, "b": 0.37, "c": 0.03}) == {"a": "a", "b": "b", "c": "other"}
    F1, F2 = simulate(a1, a2, emb, rng, H=0.0, sigma=3.0)
    dup = np.r_[np.arange(60), np.arange(30)]  # 30 clones duplicated; copy-aware P stays near 0 without signal
    naive = predictability(kernel(F1[dup]), F2[dup])
    aware = predictability(kernel(F1[dup]), F2[dup], ids=dup)
    assert abs(aware) < 0.1 and naive >= aware - 0.02, (naive, aware)
    W = kernel(F1[:4][[0, 0, 1, 2]])  # the copy pair (rows 0, 1) must get no weight on each other
    hat_ids = np.array([0, 0, 1, 2])
    assert np.where(hat_ids[:, None] == hat_ids[None, :], 0.0, W)[0, 1] == 0.0 and W[0, 1] == 1.0
    return {"ok": True, "strong": strong, "none": none}


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    res = {"selftest": selftest, "smoke": smoke, "power": power, "readout": readout, "train": train,
           "ci": ci}[stage]()
    if stage != "selftest":
        OUT.mkdir(exist_ok=True)
        json.dump(res, open(OUT / f"c17s1c_s1_{stage}.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
