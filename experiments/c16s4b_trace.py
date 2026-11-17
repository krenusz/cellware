"""C16-S4b: K3', a lineage trace beyond the fate axis (pre-registered in EXPERIMENT_LOG). TRAIN embryos only.

K3'a: is z_rel readable in expression beyond base + region + the cell's own P-vs-L likeness (s_PL)?
K3'd: is z_rel readable in the 30 PCs projected orthogonal to the P/L direction u?
Both LOEO, fold-specific s_PL and u. Gene overlap of the K3'd readout with the P/L classifier.
Null: 50 shuffles of z_rel within embryo x label. Power for a one-embryo B look (K3' first, alpha 0.025).

Run (GitHub Actions, --petracer): python -I experiments/c16s4b_trace.py
Smoke only: python -I experiments/c16s4b_trace.py --synthetic OUTDIR
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c16s2_link as k  # noqa: E402
import c16s3_position as s3  # noqa: E402
import power  # noqa: E402

OUT = HERE / "results_c16"
PCS = k.PCS
N_SHUF = 50
SEED = 20261123
STAGE_MATCHED = ["E8.5-R2", "E8.5-R3", "E9.0-R1", "E9.5-R1"]
LOG = []
_G = {}


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def folds(T):
    """Per held-out embryo: logit s_PL for all rows, and orthogonal standardised PCs for all rows."""
    from sklearn.linear_model import LogisticRegression

    X = T[PCS].to_numpy(float)
    out = {}
    for e in sorted(T["emb"].unique()):
        tr = (T["emb"] != e).to_numpy()
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[~(sd > 0)] = 1
        Xs = (X - mu) / sd
        m = LogisticRegression(max_iter=2000).fit(Xs[tr], T["lab"].to_numpy()[tr])
        u = m.coef_[0] / np.linalg.norm(m.coef_[0])
        s_pl = s3.logit(m.predict_proba(Xs)[:, 1])
        out[e] = dict(spl=s_pl, perp=Xs - np.outer(Xs @ u, u), u=u, sd=sd)
    return out


def fold_stats(T, F, base_cols, zcol="z", extra=None):
    """Per embryo: delta rho for K3'a and K3'd, cell-weighted over P and L."""
    B = T[base_cols].to_numpy(float)
    X = T[PCS].to_numpy(float)
    z = T[zcol].to_numpy(float)
    lab = T["lab"].to_numpy()
    emb = T["emb"].to_numpy()
    ex = None if extra is None else T[extra].to_numpy(float).reshape(len(T), -1)
    per = {}
    for e, f in F.items():
        acc = {"a": 0.0, "d": 0.0}
        w = 0
        ok = True
        n = {}
        for lb in (0, 1):
            te = (emb == e) & (lab == lb)
            tr = (emb != e) & (lab == lb)
            n[lb] = int(te.sum())
            if te.sum() < k.MIN_PER_LABEL or tr.sum() < k.MIN_PER_LABEL or z[te].std() == 0:
                ok = False
                continue
            Ba = np.c_[B, f["spl"]]
            Xa = X if ex is None else np.c_[X, ex]
            Xd = f["perp"] if ex is None else np.c_[f["perp"], ex]

            def rho(cols, tr=tr, te=te):
                return np.nan_to_num(spearmanr(k.ridge_pred(cols[tr], z[tr], cols[te]), z[te])[0])

            acc["a"] += te.sum() * (rho(np.c_[Ba, Xa]) - rho(Ba))
            acc["d"] += te.sum() * (rho(np.c_[B, Xd]) - rho(B))
            w += te.sum()
        per[e] = dict(a=acc["a"] / w if w else float("nan"), d=acc["d"] / w if w else float("nan"),
                      n_P=n[0], n_L=n[1], in_rule=ok)
    return per


def _shuf(seed):
    T, F, bcols = _G["T"], _G["F"], _G["bcols"]
    rng = np.random.default_rng(seed)
    S = T.copy()
    S["z"] = S.groupby(["emb", "lab"])["z"].transform(lambda s: rng.permutation(s.to_numpy()))
    return fold_stats(S, F, bcols)


def shuffles(T, F, bcols, n=N_SHUF, procs=4):
    import multiprocessing as mp

    _G.update(T=T, F=F, bcols=bcols)
    seeds = [SEED + 1000 + i for i in range(n)]
    if sys.platform != "win32" and procs > 1:
        with mp.get_context("fork").Pool(procs) as pool:
            res = pool.map(_shuf, seeds)
    else:
        res = [_shuf(s) for s in seeds]
    null = {}
    for e in res[0]:
        for t in ("a", "d"):
            v = [r[e][t] for r in res if r[e]["in_rule"]]
            null.setdefault(t, {})[e] = dict(mean=float(np.mean(v)) if v else float("nan"),
                                             sd=float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"))
    return null


def verdict(per, t):
    v = [r[t] for r in per.values() if r["in_rule"]]
    if not v:
        return "DATA-LIMITED", float("nan"), 0, 0
    m, pos = float(np.mean(v)), sum(x > 0 for x in v)
    if m > 0.05 and pos >= min(4, len(v)):
        return "TRACE BEYOND THE FATE AXIS (exploratory)", m, pos, len(v)
    if m <= 0.02:
        return "ABSORBED", m, pos, len(v)
    return "PARTLY", m, pos, len(v)


def b_power(per, null, t, rng):
    vals_all = {e: r[t] for e, r in per.items() if r["in_rule"]}
    nsd = max(null[t][e]["sd"] for e in STAGE_MATCHED if e in null[t] and not np.isnan(null[t][e]["sd"]))
    out = dict(null_sd=nsd)
    for scen, keys in (("all", list(vals_all)), ("stage-matched", [e for e in STAGE_MATCHED if e in vals_all])):
        v = np.array([vals_all[e] for e in keys])
        sims = v.mean() + v.std(ddof=1) * np.sqrt(1 + 1 / len(v)) * rng.standard_t(len(v) - 1, 100_000)
        out[scen] = dict(n=len(v), mean=float(v.mean()), sd=float(v.std(ddof=1)),
                         power=power.sim_power(sims, nsd, alpha=0.025, one_sided=True))
    out["passes"] = all(out[s]["power"] >= 0.8 for s in ("all", "stage-matched"))
    return out


def gene_overlap(T, F, comps, names, top=50):
    """Gene weights via PCA loadings: P/L classifier vs the K3'd readout (ridge on orthogonal PCs), per label."""
    from sklearn.linear_model import LogisticRegression

    X = T[PCS].to_numpy(float)
    mu, sd = X.mean(0), X.std(0)
    sd[~(sd > 0)] = 1
    Xs = (X - mu) / sd
    m = LogisticRegression(max_iter=2000).fit(Xs, T["lab"])
    u = m.coef_[0] / np.linalg.norm(m.coef_[0])
    P = np.eye(len(u)) - np.outer(u, u)
    w_pl = comps.T @ (m.coef_[0] / sd)
    res = {}
    top_pl = set(np.argsort(-np.abs(w_pl))[:top])
    for lb, nm in ((0, "P"), (1, "L")):
        g = (T["lab"] == lb).to_numpy()
        A = Xs[g] @ P
        A1 = np.c_[np.ones(g.sum()), A]
        reg = np.eye(A1.shape[1])
        reg[0, 0] = 0
        beta = np.linalg.solve(A1.T @ A1 + reg, A1.T @ T["z"].to_numpy()[g])[1:]
        w_tr = comps.T @ ((P @ beta) / sd)
        top_tr = set(np.argsort(-np.abs(w_tr))[:top])
        res[nm] = dict(jaccard=len(top_pl & top_tr) / len(top_pl | top_tr),
                       top_trace=[str(names[i]) for i in np.argsort(-np.abs(w_tr))[:15]],
                       top_trace_sign=[int(np.sign(w_tr[i])) for i in np.argsort(-np.abs(w_tr))[:15]])
    res["top_pl"] = [str(names[i]) for i in np.argsort(-np.abs(w_pl))[:15]]
    return res


def analyse(T, bcols, comps, names, outdir, rng):
    F = folds(T)
    say(f"leaves with PCs: {len(T)}")
    # smoke 1: planted column orthogonal to the fate axis and to s_PL
    pl = T.copy()
    noise = rng.normal(0, pl["z"].std(), len(pl))
    pl["planted"] = pl["z"] + noise
    sm1 = fold_stats(pl, F, bcols, extra="planted")
    ok1 = any(r["in_rule"] for r in sm1.values()) and all(r["d"] > 0.05 for r in sm1.values() if r["in_rule"])
    say(f"SMOKE (1) planted trace: min K3'd delta {min((r['d'] for r in sm1.values() if r['in_rule']), default=float('nan')):+.3f} "
        f"-> {'PASS' if ok1 else 'FAIL'}")
    null = shuffles(T, F, bcols)
    ok2 = True
    for t in ("a", "d"):
        v = [null[t][e]["mean"] for e in null[t] if not np.isnan(null[t][e]["mean"])]
        sds = [null[t][e]["sd"] for e in null[t] if not np.isnan(null[t][e]["sd"])]
        lim = 2 * float(np.sqrt(np.mean(np.square(sds)))) / np.sqrt(N_SHUF) + 2 * float(np.std(v, ddof=1)) / np.sqrt(len(v))
        okt = float(np.mean(v)) <= max(lim, 0.01)
        ok2 &= okt
        say(f"SMOKE (2) shuffled z_rel, K3'{t}: mean {np.mean(v):+.4f} (limit {max(lim, 0.01):.4f}) -> {'PASS' if okt else 'FAIL'}; "
            "per-embryo null sd " + ", ".join(f"{e} {null[t][e]['sd']:.4f}" for e in null[t]))
    per = fold_stats(T, F, bcols)
    res = dict(per_embryo=per, null=null, smoke=dict(planted=sm1, pass_=bool(ok1 and ok2)))
    for t, label in (("a", "K3'a (beyond own P/L likeness)"), ("d", "K3'd (orthogonal to the fate axis)")):
        vd, m, pos, n = ("INVALID (smoke failed)", float("nan"), 0, 0) if not (ok1 and ok2) else verdict(per, t)
        for e, r in per.items():
            say(f"  {e} {label}: delta {r[t]:+.3f} (null sd {null[t][e]['sd']:.4f}; P {r['n_P']}, L {r['n_L']})"
                f"{'' if r['in_rule'] else ' [not in rule]'}")
        bp = b_power(per, null, t, rng)
        say(f"{label}: mean {m:+.3f}, positive {pos}/{n} -> {vd}; B power (alpha 0.025, null sd {bp['null_sd']:.4f}): "
            f"all {bp['all']['power']:.3f}, stage-matched {bp['stage-matched']['power']:.3f} -> {'PASSES' if bp['passes'] else 'fails'} the gate")
        res[t] = dict(verdict=vd, mean=m, positive=pos, n=n, b_power=bp)
    enter = "K3'd" if res["d"]["b_power"]["passes"] and res["d"]["mean"] > 0 else ("K3'a" if res["a"]["b_power"]["passes"] and res["a"]["mean"] > 0 else "none")
    res["enters_look"] = enter
    say(f"K3' entering a B look (fixed rule): {enter}")
    if comps is not None:
        go = gene_overlap(T, F, comps, names)
        res["genes"] = go
        say(f"gene overlap (Jaccard, top 50) trace vs P/L classifier: P {go['P']['jaccard']:.2f}, L {go['L']['jaccard']:.2f}")
    Path(outdir).mkdir(parents=True, exist_ok=True)
    (Path(outdir) / "c16s4b_trace.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    (Path(outdir) / "c16s4b_trace.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    return res


def synthetic(outdir):
    rng = np.random.default_rng(6)
    rows = []
    for e in ["E8.0-R1", "E8.0-R3", "E8.5-R2", "E8.5-R3", "E9.0-R1", "E9.5-R1"]:
        n = 1500
        lab = rng.integers(0, 2, n)
        z = np.clip(rng.beta(2, 2, n) * 0.6 + 0.4 * lab, 0, 1)
        for j in range(n):
            rows.append((f"{e}_c{j}", e, lab[j], z[j]))
    T = pd.DataFrame(rows, columns=["cell", "emb", "lab", "z"]).set_index("cell")
    X = rng.normal(size=(len(T), 30))
    X[:, 0] += 2.0 * T["lab"].to_numpy()  # fate axis
    X[:, 1] += 1.5 * (T["z"].to_numpy() - 0.5)  # trace orthogonal-ish to the fate axis
    for i, p in enumerate(PCS):
        T[p] = X[:, i]
    T["x1"] = rng.normal(size=len(T))
    comps = rng.normal(size=(30, 200))
    names = np.array([f"g{i}" for i in range(200)])
    res = analyse(T, ["x1"], comps, names, outdir, rng)
    assert res["smoke"]["pass_"], res["smoke"]
    assert res["d"]["mean"] > 0.05, res["d"]
    print("SYNTHETIC OK", res["a"]["verdict"], "|", res["d"]["verdict"], "| enters", res["enters_look"])


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--synthetic":
        return synthetic(sys.argv[2])
    rng = np.random.default_rng(SEED)
    import c14s3_region as r
    import c14s4_genes as g4

    region = r.load_region()
    region = region[~region.index.duplicated()]
    forests, ids = k.load_train_ids()
    keep = {}
    D, _, s1 = r.readout_with_pcs(keep)
    D = D[~D.index.duplicated()]
    base = k.base_design(D, s1)
    theta = s3.fit_m1(forests)
    Z = s3.zrel_table(forests, ids, theta)
    T = Z.join(base, how="inner").join(region[s3.REG], how="left").join(D[PCS], how="inner")
    T[s3.REG] = T[s3.REG].fillna(0.0)
    names = g4.gene_names()[keep["hvg"]]
    analyse(T, list(base.columns) + s3.REG, keep["comps"], names, OUT, rng)


if __name__ == "__main__":
    main()
