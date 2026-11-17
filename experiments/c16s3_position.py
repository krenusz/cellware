"""C16-S3: axial-position control for the C16 hidden inherited state (pre-registered in EXPERIMENT_LOG).
TRAIN embryos only. Region scores are expression-inferred axial identity (authors' regressors), not measured position.

S3a: C16-S2 expression link with region scores in the base.
S3b: does z_rel predict a cell's own P/L label beyond its own and its close relatives' region scores?

Run (GitHub Actions, --petracer; two parallel jobs):
    python -I experiments/c16s3_position.py a
    python -I experiments/c16s3_position.py b
Smoke only: python -I experiments/c16s3_position.py --synthetic OUTDIR
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c16s0_hidden as h  # noqa: E402
import c16s2_link as k  # noqa: E402

OUT = HERE / "results_c16"
REG = ["ap", "dv", "nap", "ap2", "dv2", "apdv", "m_ap", "m_dv", "m_nap"]
RAW3 = ["ap", "dv", "nap"]
MIN_REL = 5
SEED = 20261120
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def save(name, res, outdir):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    (Path(outdir) / f"{name}.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    (Path(outdir) / f"{name}.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")


def fit_m1(forests, start=None):
    """M1 on all train trees, warm-started at the C16-S0 estimate (1 start)."""
    allF = h.merge([f.raw for f in forests.values()])
    if start is None:
        p = json.loads((OUT / "c16s0_hidden.json").read_text(encoding="utf-8"))["m1_all"]
        start = np.r_[np.log([p["alpha0"], p["beta0"], p["alpha1"], p["beta1"], p["mu"]]), 0.0, 0.0, 0.0]
    r = minimize(lambda th: -h.loglik(th, allF, 1), start, method="L-BFGS-B", bounds=[(-9, 3)] * 5 + [(-8, 8)] * 3)
    say(f"M1 all train (warm start): {h.params(r.x)}")
    return r.x


def zrel_table(forests, ids, theta):
    rows = []
    for e, F in forests.items():
        z = k.zrel(theta, F)
        for v, cid in enumerate(ids[e]):
            if cid is not None:
                rows.append((cid, e, int(F.obs[v]), z[v]))
    return pd.DataFrame(rows, columns=["cell", "emb", "lab", "z"]).set_index("cell")


# ---------- S3a ----------
def part_a(forests, ids, D, base, region, outdir, rng, theta=None):
    theta = fit_m1(forests) if theta is None else theta
    Z = zrel_table(forests, ids, theta)
    T = Z.join(base, how="inner").join(region[REG], how="left").join(D[k.PCS], how="inner")
    T[REG] = T[REG].fillna(0.0)
    miss = T.index.difference(region.index)
    T.loc[miss, ["m_ap", "m_dv", "m_nap"]] = 1.0
    bcols = list(base.columns)
    say(f"S3a leaves with PCs: {len(T)}; without a region row: {len(miss)}")
    pl = T.copy()
    pl["planted"] = pl["z"] + rng.normal(0, pl["z"].std(), len(pl))
    sm1 = k.deltas(pl, bcols + REG + k.PCS, ["planted"])
    ok1 = any(r["in_rule"] for r in sm1.values()) and all(r["delta"] > 0.05 for r in sm1.values() if r["in_rule"])
    sh = T.copy()
    sh["z"] = sh.groupby(["emb", "lab"])["z"].transform(lambda s: rng.permutation(s.to_numpy()))
    v2 = [r["delta"] for r in k.deltas(sh, bcols + REG, k.PCS).values() if r["in_rule"]]
    m2 = float(np.mean(v2)) if v2 else 0.0
    lim2 = max(0.03, 2 * float(np.std(v2, ddof=1)) / np.sqrt(len(v2))) if len(v2) > 1 else 0.03
    ok2 = m2 <= lim2  # amendment 1: one-sided (a false gain is the failure)
    say(f"SMOKE (i) planted column: min delta {min((r['delta'] for r in sm1.values() if r['in_rule']), default=float('nan')):+.3f} "
        f"-> {'PASS' if ok1 else 'FAIL'}")
    say(f"SMOKE (ii) shuffled z_rel: mean delta {m2:+.3f} (limit {lim2:.3f}) -> {'PASS' if ok2 else 'FAIL'}")
    per = k.deltas(T, bcols + REG, k.PCS)
    reg_only = k.deltas(T, bcols, REG)
    m, pos, n = k.summary(per)
    if not (ok1 and ok2):
        verdict = "INVALID (smoke failed)"
    elif not n:
        verdict = "DATA-LIMITED"
    elif m > 0.05 and pos >= min(4, n):
        verdict = "SURVIVES AXIAL IDENTITY (exploratory)"
    elif m <= 0.02:
        verdict = "ABSORBED BY AXIAL IDENTITY"
    else:
        verdict = "PARTLY ABSORBED"
    for e, r in per.items():
        say(f"  {e}: delta rho beyond region {r['delta']:+.3f}; region alone {reg_only[e]['delta']:+.3f} "
            f"(P {r['n_P']}, L {r['n_L']}){'' if r['in_rule'] else ' [not in rule]'}")
    say(f"S3a mean delta {m:+.3f}, positive {pos}/{n} -> {verdict}")
    res = dict(verdict=verdict, per_embryo=per, region_alone=reg_only, mean=m, positive=pos, n=n,
               smoke=dict(planted=sm1, shuffled_mean=m2, limit=lim2, pass_=bool(ok1 and ok2)))
    save("c16s3a_position", res, outdir)
    return res


# ---------- S3b ----------
def relatives_mean(F, cell_ids, region):
    """Per labelled leaf: mean raw region scores of other labelled leaves (with a region row) under the
    lowest ancestor holding >= MIN_REL of them."""
    n = len(F.parent)
    vals = np.zeros((n, 3))
    cnt = np.zeros(n)
    has = np.zeros(n, bool)
    R = region.reindex([c if c is not None else "" for c in cell_ids])
    ok = R["m_ap"].notna().to_numpy() & (F.obs >= 0)
    raw = R[RAW3].to_numpy(float)
    miss = R[["m_ap", "m_dv", "m_nap"]].to_numpy(float)
    raw = np.where(miss > 0, np.nan, raw)
    for j in range(3):
        col = np.where(ok & ~np.isnan(raw[:, j]), raw[:, j], 0.0)
        vals[:, j] = col
    cnt[ok] = 1
    has[ok] = True
    S, C = vals.copy(), cnt.copy()
    nn = np.zeros((n, 3))
    nn[ok] = ~np.isnan(raw[ok])
    NN = nn.copy()
    for v in range(n - 1, -1, -1):  # parents precede children
        p = F.parent[v]
        if p >= 0:
            S[p] += S[v]
            C[p] += C[v]
            NN[p] += NN[v]
    out = np.full((n, 3), np.nan)
    for v in np.where(has)[0]:
        a = F.parent[v]
        while a >= 0 and C[a] - 1 < MIN_REL:
            a = F.parent[a]
        if a < 0:  # no ancestor with enough relatives: left out of S3b
            continue
        num = S[a] - vals[v]
        den = NN[a] - nn[v]
        out[v] = np.where(den > 0, num / np.maximum(den, 1), np.nan)
    return out


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def auc_deltas(T, bcols, extra):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    per = {}
    for e in sorted(T["emb"].unique()):
        tr, te = T[T.emb != e], T[T.emb == e]
        if te["y"].nunique() < 2 or min((te.y == 0).sum(), (te.y == 1).sum()) < 30:
            per[e] = dict(delta=float("nan"), n=len(te), in_rule=False)
            continue
        auc = {}
        for name, cols in (("base", bcols), ("full", bcols + extra)):
            sc = StandardScaler().fit(tr[cols])
            m = LogisticRegression(max_iter=2000).fit(sc.transform(tr[cols]), tr["y"])
            auc[name] = roc_auc_score(te["y"], m.predict_proba(sc.transform(te[cols]))[:, 1])
        per[e] = dict(delta=auc["full"] - auc["base"], auc_base=auc["base"], auc_full=auc["full"], n=len(te), in_rule=True)
    return per


def part_b(forests, ids, region, outdir, rng, theta=None):
    theta = fit_m1(forests) if theta is None else theta
    rows = []
    for e, F in forests.items():
        z = k.zrel(theta, F)
        rel = relatives_mean(F, ids[e], region)
        for v, cid in enumerate(ids[e]):
            if cid is not None and cid in region.index and not np.isnan(rel[v]).all():
                rows.append((cid, e, int(F.obs[v]), z[v], *rel[v]))
    T = pd.DataFrame(rows, columns=["cell", "emb", "y", "z", "r_ap", "r_dv", "r_nap"]).set_index("cell")
    T = T[~T.index.duplicated()].join(region[REG], how="inner")
    for c in ["r_ap", "r_dv", "r_nap"]:
        T["m_" + c] = T[c].isna().astype(float)
        T[c] = T[c].fillna(0.0)
    T["lz"] = logit(T["z"].to_numpy())
    bcols = REG + ["r_ap", "r_dv", "r_nap", "m_r_ap", "m_r_dv", "m_r_nap"]
    say(f"S3b leaves with region rows and relatives: {len(T)}")
    for e, g in T.groupby("emb"):
        say(f"  {e}: P {int((g.y == 0).sum())}, L {int((g.y == 1).sum())}")
    sim = T.copy()
    sim["y"] = (rng.random(len(sim)) < sim["z"].to_numpy()).astype(int)
    sm1 = auc_deltas(sim, bcols, ["lz"])
    ok1 = any(r["in_rule"] for r in sm1.values()) and all(r["delta"] > 0.02 for r in sm1.values() if r["in_rule"])
    sh = T.copy()
    sh["lz"] = sh.groupby("emb")["lz"].transform(lambda s: rng.permutation(s.to_numpy()))
    v2 = [r["delta"] for r in auc_deltas(sh, bcols, ["lz"]).values() if r["in_rule"]]
    m2 = float(np.mean(v2)) if v2 else 0.0
    lim2 = max(0.01, 2 * float(np.std(v2, ddof=1)) / np.sqrt(len(v2))) if len(v2) > 1 else 0.01
    ok2 = m2 <= lim2  # amendment 1: one-sided (a false gain is the failure)
    say(f"SMOKE (i) y ~ Bernoulli(z_rel): min delta AUC {min((r['delta'] for r in sm1.values() if r['in_rule']), default=float('nan')):+.3f} "
        f"-> {'PASS' if ok1 else 'FAIL'}")
    say(f"SMOKE (ii) shuffled z_rel: mean delta AUC {m2:+.4f} (limit {lim2:.4f}) -> {'PASS' if ok2 else 'FAIL'}")
    per = auc_deltas(T, bcols, ["lz"])
    v = [r["delta"] for r in per.values() if r["in_rule"]]
    m = float(np.mean(v)) if v else float("nan")
    pos = sum(x > 0 for x in v)
    if not (ok1 and ok2):
        verdict = "INVALID (smoke failed)"
    elif not v:
        verdict = "DATA-LIMITED"
    elif m > 0.02 and pos >= len(v) - 1:
        verdict = "BEYOND AXIAL POSITION (exploratory)"
    elif m <= 0:
        verdict = "ABSORBED BY AXIAL POSITION"
    else:
        verdict = "PARTLY"
    for e, r in per.items():
        if r["in_rule"]:
            say(f"  {e}: AUC position-only {r['auc_base']:.3f}, + z_rel {r['auc_full']:.3f}, delta {r['delta']:+.3f} (n {r['n']})")
        else:
            say(f"  {e}: not in rule (n {r['n']})")
    say(f"S3b mean delta AUC {m:+.3f}, positive {pos}/{len(v)} -> {verdict}")
    res = dict(verdict=verdict, per_embryo=per, mean=m, positive=pos, n=len(v),
               smoke=dict(simulated=sm1, shuffled_mean=m2, limit=lim2, pass_=bool(ok1 and ok2)))
    save("c16s3b_position", res, outdir)
    return res


# ---------- synthetic path test ----------
def synthetic(outdir):
    import c16s1_time as t

    rng = np.random.default_rng(4)
    raw = t.rand_forests(rng, n_emb=6, n_leaves=800)
    truth = np.r_[np.log([0.03, 0.3, 0.3, 0.03, 0.08]), 0.0, 0.0, 0.0]
    forests, ids, cells = {}, {}, []
    for e, f in raw.items():
        obs = h.simulate(f, truth, 1, rng)
        forests[e] = h.Emb((f.raw[0], f.raw[1], list(obs)))
        ids[e] = [f"{e}_c{v}" if obs[v] >= 0 else None for v in range(len(obs))]
        cells += [c for c in ids[e] if c is not None]
    D = pd.DataFrame(rng.normal(size=(len(cells), 30)), index=cells, columns=k.PCS)
    base = pd.DataFrame(dict(x1=rng.normal(size=len(cells))), index=cells)
    reg = pd.DataFrame(rng.normal(size=(len(cells), 3)), index=cells, columns=RAW3)
    reg["ap2"], reg["dv2"], reg["apdv"] = reg.ap ** 2, reg.dv ** 2, reg.ap * reg.dv
    reg["m_ap"] = reg["m_dv"] = reg["m_nap"] = 0.0
    th = fit_m1(forests, start=truth)
    ra = part_a(forests, ids, D, base, reg, outdir, rng, th)
    LOG.clear()
    rb = part_b(forests, ids, reg, outdir, rng, th)
    assert ra["smoke"]["pass_"] and rb["smoke"]["pass_"], (ra["smoke"], rb["smoke"])
    assert rb["verdict"].startswith("BEYOND"), rb["verdict"]  # labels follow the tree, random region carries nothing
    print("SYNTHETIC OK", ra["verdict"], "|", rb["verdict"])


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--synthetic":
        return synthetic(sys.argv[2])
    part = sys.argv[1]
    rng = np.random.default_rng(SEED + (part == "b"))
    import c14s3_region as r

    region = r.load_region()
    region = region[~region.index.duplicated()]
    forests, ids = k.load_train_ids()
    if part == "a":
        D, _, s1 = r.readout_with_pcs()
        D = D[~D.index.duplicated()]
        part_a(forests, ids, D, k.base_design(D, s1), region, OUT, rng)
    else:
        part_b(forests, ids, region, OUT, rng)


if __name__ == "__main__":
    main()
