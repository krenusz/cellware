"""C14-S3: train-only regional confound check of the expression-state gain (pre-registered in EXPERIMENT_LOG).

Models per type, leave-one-embryo-out (as C14-S1): B base, BP base+PCs, BR base+region, BRP base+region+PCs.
d = BP-B, dR = BRP-BR, r = BR-B, retained = dR/d (embryo means of per-embryo cell-weighted delta AUC).
Region block: lean-set AP, DV, neural-AP scores (expression-inferred), squares, AP*DV, missing flags.
Train embryos only: the scores file is filtered with partitions.train_only on the cell-id column before any
score column is read. Readout, HVG and PCs as C14-S1b (c14s1b_expr).

Run (GitHub Actions, --petracer): python -I experiments/c14s3_region.py
Smoke only (synthetic, scratch): python -I experiments/c14s3_region.py --synthetic OUTDIR
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from partitions import train_only  # noqa: E402

DS = "petracer_embryo"
SCORES = HERE.parent / "data" / "ext_petracer" / "raw" / "E7.5-E10_spatial_scores.csv"
OUT = HERE / "results_c14"
SEED = 20261116
PCS = [f"pc{i}" for i in range(30)]
REG = ["ap", "dv", "nap", "ap2", "dv2", "apdv", "m_ap", "m_dv", "m_nap"]
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def embryo_of(cell):
    return "-".join(str(cell).split("-")[:2])


def load_region(cells=None):
    ids = pd.read_csv(SCORES, usecols=[0]).iloc[:, 0].astype(str)
    u = pd.DataFrame({"cell": ids, "embryo": ids.map(embryo_of), "row": np.arange(len(ids))})
    tr = train_only(DS, u, unit_col="embryo")  # sealed rows dropped before any score is read
    keep = set(tr["row"].to_numpy() + 1)  # +1 for the header line
    df = pd.read_csv(SCORES, skiprows=lambda i: i != 0 and i not in keep, index_col=0)
    df.index = df.index.astype(str)
    assert set(df.index.map(embryo_of)) <= set(tr["embryo"]), "non-train rows read"
    return region_block(df.rename(columns={"ap_score": "ap", "dv_score": "dv", "neural_ap_score": "nap"}))


def region_block(df):
    R = pd.DataFrame(index=df.index)
    for c in ["ap", "dv", "nap"]:
        v = pd.to_numeric(df[c], errors="coerce")
        R["m_" + c] = v.isna().astype(float)
        R[c] = v.fillna(0.0)
    R["ap2"], R["dv2"], R["apdv"] = R["ap"] ** 2, R["dv"] ** 2, R["ap"] * R["dv"]
    return R[REG]


def readout_with_pcs():
    """C14-S1b readout + 30 PCs (same code path as c14s1b_expr.main, steps 1-2)."""
    import c14s1_select as s1
    import c14s1b_expr as e
    D, embryos = s1.build()
    obs, _, raw = s1.g0.load_obs()
    del raw
    totals = obs["total_counts"].to_dict()
    del obs
    n_genes = s = ss = None
    n_cells = 0
    for emb in embryos:
        if emb == e.HVG_SKIP:
            continue
        bc, scale = e.cell_scale(emb, totals)
        ok = ~np.isnan(scale)
        n_cells += int(ok.sum())
        for ch, dims in e.stream(e.files(emb)[1]):
            if n_genes is None:
                n_genes = dims[0]
                s, ss = np.zeros(n_genes), np.zeros(n_genes)
            c = ch["c"].to_numpy() - 1
            k = ok[c]
            x = np.log1p(ch["v"].to_numpy()[k] * scale[c[k]])
            g = ch["g"].to_numpy()[k] - 1
            s += np.bincount(g, x, n_genes)
            ss += np.bincount(g, x * x, n_genes)
    hvg = np.argsort(-(ss / n_cells - (s / n_cells) ** 2))[:e.N_HVG]
    gpos = -np.ones(n_genes, np.int64)
    gpos[hvg] = np.arange(e.N_HVG)
    rows = {c: i for i, c in enumerate(D.index)}
    M = np.zeros((len(D), e.N_HVG), np.float32)
    for emb in embryos:
        bc, scale = e.cell_scale(emb, totals)
        rid = np.array([rows.get(x, -1) for x in bc])
        for ch, _ in e.stream(e.files(emb)[1]):
            c = ch["c"].to_numpy() - 1
            g = gpos[ch["g"].to_numpy() - 1]
            k = (rid[c] >= 0) & (g >= 0)
            M[rid[c[k]], g[k]] += np.log1p(ch["v"].to_numpy()[k] * scale[c[k]]).astype(np.float32)
        say(f"matrix pass: {emb} done")
    mu, sd = M.mean(0), M.std(0)
    sd[~(sd > 0)] = 1
    M = np.clip((M - mu) / sd, -10, 10)
    pcs = PCA(n_components=len(PCS), random_state=0).fit_transform(M)
    del M
    for i, name in enumerate(PCS):
        D[name] = pcs[:, i]
    return D, embryos, s1


def design(s1, df, parts, planted=None):
    out = s1.X(df, [], None)
    if "reg" in parts:
        out = pd.concat([out, df[REG]], axis=1)
    if "pcs" in parts:
        out = pd.concat([out, df[PCS]], axis=1)
    if planted is not None:
        out["planted"] = planted.loc[df.index]
    return out


def loeo(s1, D, embryos, ycol, parts, planted=None):
    p = pd.Series(np.nan, index=D.index)
    for _, g in D.groupby("T"):
        Xg = design(s1, g, parts, planted)
        for emb in embryos:
            te = (g["embryo"] == emb).to_numpy()
            if te.sum() == 0 or g.loc[~te, ycol].nunique() < 2:
                continue
            p.loc[g.index[te]] = s1.g0.fit_pred(Xg[~te], g.loc[~te, ycol], Xg[te])
    return p


def per_embryo(s1, D, embryos, ycol, pb, pf):
    E = D.assign(p_base=pb, p_full=pf).dropna(subset=["p_base", "p_full"])
    out = {}
    for emb in embryos:
        sub = E[E.embryo == emb]
        if len(sub):
            v = s1.pooled_delta(sub, ycol)
            if v == v:
                out[emb] = round(float(v), 4)
    return out


def contrasts(s1, D, embryos, ycol, planted=None):
    p = {k: loeo(s1, D, embryos, ycol, parts, planted if "pcs" in parts else None)
         for k, parts in {"B": [], "BP": ["pcs"], "BR": ["reg"], "BRP": ["reg", "pcs"]}.items()}
    d = per_embryo(s1, D, embryos, ycol, p["B"], p["BP"])
    dR = per_embryo(s1, D, embryos, ycol, p["BR"], p["BRP"])
    r = per_embryo(s1, D, embryos, ycol, p["B"], p["BR"])
    m = {k: float(np.mean(list(v.values()))) if v else float("nan") for k, v in [("d", d), ("dR", dR), ("r", r)]}
    return dict(d=d, dR=dR, r=r, mean_d=m["d"], mean_dR=m["dR"], mean_r=m["r"],
                retained=m["dR"] / m["d"] if m["d"] else float("nan"))


def decide(c):
    pos = sum(v > 0 for v in c["dR"].values())
    n = len(c["dR"])
    if c["retained"] >= 0.5 and pos >= min(5, n):
        return "SURVIVES", pos, n
    if c["retained"] < 0.25 or pos < n / 2:
        return "MOSTLY REGIONAL", pos, n
    return "PARTLY REGIONAL", pos, n


def run(D, embryos, s1, outdir):
    rng = np.random.default_rng(SEED)
    z = lambda v: (v - v.mean()) / (v.std() or 1)  # noqa: E731
    D["y_reg"] = ((z(D["ap"]) + z(D["dv"]) + rng.normal(0, 1, len(D))) > 0).astype(int)
    s_reg = contrasts(s1, D, embryos, "y_reg")
    planted = D["y"] + rng.normal(0, 1.5, len(D))
    s_pl = contrasts(s1, D, embryos, "y", planted=planted)
    ok1, ok2 = s_reg["mean_dR"] < 0.02, s_pl["mean_dR"] > 0.05
    say(f"SMOKE (i) region-only outcome: dR {s_reg['mean_dR']:+.4f} (d {s_reg['mean_d']:+.4f}) -> {'PASS' if ok1 else 'FAIL'}")
    say(f"SMOKE (ii) planted state: dR {s_pl['mean_dR']:+.4f} -> {'PASS' if ok2 else 'FAIL'}")
    c = contrasts(s1, D, embryos, "y")
    verdict, pos, n = decide(c)
    if not (ok1 and ok2):
        verdict = "INVALID (smoke test failed)"
    say(f"d (BP-B) {c['mean_d']:+.4f}; dR (BRP-BR) {c['mean_dR']:+.4f}; r (BR-B) {c['mean_r']:+.4f}; "
        f"retained {c['retained']:.2f}; dR>0 in {pos}/{n} embryos -> {verdict}")
    say("per embryo dR: " + ", ".join(f"{k} {v:+.3f}" for k, v in c["dR"].items()))
    res = dict(verdict=verdict, dR_positive=pos, n_embryos=n, smoke_region=s_reg, smoke_planted=s_pl, smoke_pass=bool(ok1 and ok2),
               region_missing_frac={k: float(D["m_" + k].mean()) for k in ["ap", "dv", "nap"]}, **c)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    (Path(outdir) / "c14s3_region.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    (Path(outdir) / "c14s3_region.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    return res


def synthetic(outdir):
    """Tiny synthetic path test: fake readout, PCs carry region + a non-regional signal."""
    import c14s1_select as s1
    rng = np.random.default_rng(1)
    rows = []
    for emb in ["E8.0-R1", "E8.5-R2", "E9.0-R1", "E9.5-R1", "E8.0-R3", "E8.5-R3"]:
        for t in ["T1", "T2"]:
            n = 400
            ap, dv, st = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
            y = ((0.8 * ap + 1.2 * st + rng.normal(size=n)) > 0).astype(int)
            df = pd.DataFrame({"embryo": emb, "T": t, "y": y, "cluster": [f"{emb}{t}{i // 4}" for i in range(n)],
                               "total_counts": rng.integers(1000, 5000, n), "detection_rate": rng.random(n),
                               "clade": rng.integers(6, 50, n), "depth": rng.integers(1, 9, n), "stage": 8.5,
                               "lineage": "L", "germ_layer": "G", "ap_score": ap, "dv_score": dv, "neural_ap_score": np.nan})
            for i in range(len(PCS)):
                df[PCS[i]] = (ap if i == 0 else st if i == 1 else dv if i == 2 else rng.normal(size=n)) + 0.1 * rng.normal(size=n)
            rows.append(df)
    D = pd.concat(rows, ignore_index=True)
    D.index = [f"c{i}" for i in range(len(D))]
    D = D.join(region_block(D.rename(columns={"ap_score": "ap", "dv_score": "dv", "neural_ap_score": "nap"})))
    res = run(D, sorted(D.embryo.unique()), s1, outdir)
    assert res["smoke_pass"], "synthetic smoke failed"
    assert 0.3 < res["retained"] < 1.0, f"synthetic retained {res['retained']}"
    print("SYNTHETIC OK", res["verdict"], round(res["retained"], 2))


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--synthetic":
        return synthetic(sys.argv[2])
    D, embryos, s1 = readout_with_pcs()
    R = load_region()
    D = D.join(R, how="left")
    miss = D["ap"].isna()
    say(f"readout cells {len(D)}; without a score row {int(miss.sum())}")
    D[REG] = D[REG].fillna(0.0)
    D.loc[miss, ["m_ap", "m_dv", "m_nap"]] = 1.0
    run(D, embryos, s1, OUT)


if __name__ == "__main__":
    main()
