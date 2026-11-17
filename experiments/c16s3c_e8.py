"""C16-S3c: the E8.0 check (pre-registered in EXPERIMENT_LOG). TRAIN embryos only.

(i)  position fitted within each embryo (5-fold CV grouped by clade): AUC position-only vs + logit(z_rel).
(ii) matched family: AUC of relatives' lateral fraction for own label, within strata of family size and tree distance.

Run (GitHub Actions, --petracer; two parallel jobs):
    python -I experiments/c16s3c_e8.py i
    python -I experiments/c16s3c_e8.py ii
Smoke only: python -I experiments/c16s3c_e8.py --synthetic OUTDIR
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c16s2_link as k  # noqa: E402
import c16s3_position as s3  # noqa: E402

OUT = HERE / "results_c16"
SEED = 20261121
GROUP_STEPS = 4
N_LO, N_HI = 5, 20
D_STRATA = [(1, 2), (3, 4), (5, 8)]
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def save(name, res, outdir):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    (Path(outdir) / f"{name}.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    (Path(outdir) / f"{name}.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")


def family(F):
    """Per node: n_rel, d (generations up to the lowest ancestor with >= 5 other labelled leaves), f (lateral
    fraction among them), group (ancestor GROUP_STEPS induced steps below the root)."""
    n = len(F.parent)
    lab = F.obs >= 0
    C = lab.astype(float)
    Lc = (F.obs == 1).astype(float)
    for v in range(n - 1, -1, -1):
        p = F.parent[v]
        if p >= 0:
            C[p] += C[v]
            Lc[p] += Lc[v]
    steps = np.zeros(n, int)
    for v in range(n):
        if F.parent[v] >= 0:
            steps[v] = steps[F.parent[v]] + 1
    nrel = np.full(n, np.nan)
    d = np.full(n, np.nan)
    f = np.full(n, np.nan)
    grp = np.full(n, -1)
    for v in np.where(lab)[0]:
        a, dist = F.parent[v], F.length[v]
        while a >= 0 and C[a] - 1 < N_LO:
            dist += F.length[a] if F.parent[a] >= 0 else 0
            a = F.parent[a]
        if a >= 0:
            nrel[v] = C[a] - 1
            d[v] = dist
            f[v] = (Lc[a] - (F.obs[v] == 1)) / (C[a] - 1)
        g = v
        while steps[g] > GROUP_STEPS:
            g = F.parent[g]
        grp[v] = g
    return nrel, d, f, grp


def auc(y, s):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


# ---------- part (i) ----------
def cv_delta(T, bcols, extra):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    per = {}
    for e, g in T.groupby("emb"):
        if min((g.y == 0).sum(), (g.y == 1).sum()) < 30 or g["grp"].nunique() < 5:
            per[e] = dict(in_rule=False, n=len(g))
            continue
        pred = {"base": np.zeros(len(g)), "full": np.zeros(len(g))}
        for tr, te in GroupKFold(5).split(g, groups=g["grp"]):
            a, b = g.iloc[tr], g.iloc[te]
            if a["y"].nunique() < 2:
                continue
            for name, cols in (("base", bcols), ("full", bcols + extra)):
                sc = StandardScaler().fit(a[cols])
                m = LogisticRegression(max_iter=2000).fit(sc.transform(a[cols]), a["y"])
                pred[name][te] = m.predict_proba(sc.transform(b[cols]))[:, 1]
        ab, af = auc(g["y"], pred["base"]), auc(g["y"], pred["full"])
        per[e] = dict(in_rule=True, n=len(g), auc_base=ab, auc_full=af, delta=af - ab)
    return per


def part_i(forests, ids, region, outdir, rng, theta=None):
    theta = s3.fit_m1(forests) if theta is None else theta
    rows = []
    for e, F in forests.items():
        z = k.zrel(theta, F)
        rel = s3.relatives_mean(F, ids[e], region)
        _, _, _, grp = family(F)
        for v, cid in enumerate(ids[e]):
            if cid is not None and cid in region.index and not np.isnan(rel[v]).all():
                rows.append((cid, e, int(F.obs[v]), z[v], f"{e}:{grp[v]}", *rel[v]))
    T = pd.DataFrame(rows, columns=["cell", "emb", "y", "z", "grp", "r_ap", "r_dv", "r_nap"]).set_index("cell")
    T = T[~T.index.duplicated()].join(region[s3.REG], how="inner")
    for c in ["r_ap", "r_dv", "r_nap"]:
        T["m_" + c] = T[c].isna().astype(float)
        T[c] = T[c].fillna(0.0)
    T["lz"] = s3.logit(T["z"].to_numpy())
    bcols = s3.REG + ["r_ap", "r_dv", "r_nap", "m_r_ap", "m_r_dv", "m_r_nap"]
    sim = T.copy()
    sim["y"] = (rng.random(len(sim)) < sim["z"].to_numpy()).astype(int)
    sm1 = cv_delta(sim, bcols, ["lz"])
    ok1 = any(r["in_rule"] for r in sm1.values()) and all(r["delta"] > 0.02 for r in sm1.values() if r["in_rule"])
    sh = T.copy()
    sh["lz"] = sh.groupby("emb")["lz"].transform(lambda s: rng.permutation(s.to_numpy()))
    v2 = [r["delta"] for r in cv_delta(sh, bcols, ["lz"]).values() if r["in_rule"]]
    m2 = float(np.mean(v2)) if v2 else 0.0
    lim2 = max(0.01, 2 * float(np.std(v2, ddof=1)) / np.sqrt(len(v2))) if len(v2) > 1 else 0.01
    ok2 = m2 <= lim2
    say(f"SMOKE (1) y ~ Bernoulli(z_rel): min delta {min((r['delta'] for r in sm1.values() if r['in_rule']), default=float('nan')):+.3f} "
        f"-> {'PASS' if ok1 else 'FAIL'}")
    say(f"SMOKE (2) shuffled z_rel: mean delta {m2:+.4f} (limit {lim2:.4f}) -> {'PASS' if ok2 else 'FAIL'}")
    per = cv_delta(T, bcols, ["lz"])
    v = [r["delta"] for r in per.values() if r["in_rule"]]
    pos = sum(x > 0.02 for x in v)
    if not (ok1 and ok2):
        verdict = "INVALID (smoke failed)"
    elif pos >= min(5, len(v)) and v:
        verdict = "BIAS BEYOND WITHIN-EMBRYO POSITION (exploratory)"
    else:
        verdict = "NOT MET"
    for e, r in per.items():
        if r["in_rule"]:
            say(f"  {e}: within-embryo AUC position-only {r['auc_base']:.3f}, + z_rel {r['auc_full']:.3f}, delta {r['delta']:+.3f} (n {r['n']})")
        else:
            say(f"  {e}: not in rule (n {r['n']})")
    say(f"part (i): delta > 0.02 in {pos}/{len(v)} -> {verdict}")
    res = dict(verdict=verdict, per_embryo=per, smoke=dict(simulated=sm1, shuffled_mean=m2, limit=lim2, pass_=bool(ok1 and ok2)))
    save("c16s3c_i", res, outdir)
    return res


# ---------- part (ii) ----------
def strata_table(forests):
    rows = []
    for e, F in forests.items():
        nrel, d, f, _ = family(F)
        for v in np.where(F.obs >= 0)[0]:
            if N_LO <= nrel[v] <= N_HI:
                for lo, hi in D_STRATA:
                    if lo <= d[v] <= hi:
                        rows.append((e, int(F.obs[v]), f[v], f"d{lo}-{hi}", nrel[v]))
    return pd.DataFrame(rows, columns=["emb", "y", "f", "stratum", "nrel"])


def strata_auc(S):
    out = {}
    for (e, st), g in S.groupby(["emb", "stratum"]):
        if min((g.y == 0).sum(), (g.y == 1).sum()) >= 30:
            out.setdefault(e, {})[st] = dict(auc=auc(g["y"], g["f"]), n=len(g), mean_nrel=float(g["nrel"].mean()))
    return out


def part_ii(forests, outdir, rng):
    S = strata_table(forests)
    perm = S.copy()
    perm["y"] = perm.groupby("emb")["y"].transform(lambda s: rng.permutation(s.to_numpy()))
    pa, lim = {}, {}
    for st, g in perm.groupby("stratum"):
        n0, n1 = int((g.y == 0).sum()), int((g.y == 1).sum())
        pa[st] = auc(g["y"], g["f"])
        lim[st] = max(0.05, 3 * np.sqrt((n0 + n1 + 1) / (12 * max(n0, 1) * max(n1, 1))))  # amendment 1: null SE of AUC
    ok = all(abs(a - 0.5) <= lim[s] for s, a in pa.items() if not np.isnan(a))
    say("SMOKE permuted labels: pooled stratum AUC " + ", ".join(f"{s} {a:.3f} (limit +-{lim[s]:.3f})" for s, a in sorted(pa.items())) + f" -> {'PASS' if ok else 'FAIL'}")
    A = strata_auc(S)
    shared = sorted(set.intersection(*[set(v) for v in A.values()])) if A else []
    summ = {}
    for e in sorted(A):
        for st, r in sorted(A[e].items()):
            say(f"  {e} {st}: AUC {r['auc']:.3f} (n {r['n']}, mean family size {r['mean_nrel']:.1f})")
        if shared:
            w = np.array([A[e][s]["n"] for s in shared], float)
            summ[e] = float(np.sum(w * np.array([A[e][s]["auc"] for s in shared])) / w.sum())
    say(f"strata shared by all embryos: {shared}")
    say("summary (leaf-weighted AUC over shared strata): " + ", ".join(f"{e} {v:.3f}" for e, v in summ.items()))
    res = dict(smoke_pass=bool(ok), smoke=pa, per_embryo_stratum=A, shared=shared, summary=summ,
               n_leaves={e: int((S.emb == e).sum()) for e in S.emb.unique()})
    save("c16s3c_ii", res, outdir)
    return res


def synthetic(outdir):
    import c16s0_hidden as h
    import c16s1_time as t

    rng = np.random.default_rng(5)
    raw = t.rand_forests(rng, n_emb=6, n_leaves=800)
    truth = np.r_[np.log([0.03, 0.3, 0.3, 0.03, 0.08]), 0.0, 0.0, 0.0]
    forests, ids, cells = {}, {}, []
    for e, f in raw.items():
        obs = h.simulate(f, truth, 1, rng)
        forests[e] = h.Emb((f.raw[0], f.raw[1], list(obs)))
        ids[e] = [f"{e}_c{v}" if obs[v] >= 0 else None for v in range(len(obs))]
        cells += [c for c in ids[e] if c is not None]
    reg = pd.DataFrame(rng.normal(size=(len(cells), 3)), index=cells, columns=s3.RAW3)
    reg["ap2"], reg["dv2"], reg["apdv"] = reg.ap ** 2, reg.dv ** 2, reg.ap * reg.dv
    reg["m_ap"] = reg["m_dv"] = reg["m_nap"] = 0.0
    th = s3.fit_m1(forests, start=truth)
    ri = part_i(forests, ids, reg, outdir, rng, th)
    LOG.clear()
    rii = part_ii(forests, outdir, rng)
    assert ri["smoke"]["pass_"] and rii["smoke_pass"], (ri["smoke"], rii["smoke"])
    assert ri["verdict"].startswith("BIAS"), ri["verdict"]
    print("SYNTHETIC OK", ri["verdict"], "| shared strata", rii["shared"])


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--synthetic":
        return synthetic(sys.argv[2])
    part = sys.argv[1]
    rng = np.random.default_rng(SEED + len(part))
    forests, ids = k.load_train_ids()
    if part == "i":
        import c14s3_region as r

        region = r.load_region()
        part_i(forests, ids, region[~region.index.duplicated()], OUT, rng)
    else:
        part_ii(forests, OUT, rng)


if __name__ == "__main__":
    main()
