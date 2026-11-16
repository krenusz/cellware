"""C14-S4: train-only, exploratory. Which genes and programs carry the expression-state gain (P over base)?

Pre-registered in EXPERIMENT_LOG (C14-S4). Readout, HVG and PCs exactly as C14-S1b/S3 (c14s3_region.readout_with_pcs,
which now also keeps the PCA loadings). Per type, logistic as in S1 (StandardScaler + LogisticRegression C=1).
  (a) per-type gain: LOEO delta AUC (BP - B), cell-weighted over embryos
  (b) gene weights per type: w_g = sum_k loading_kg * beta_k / sd(pc_k), fit on all train embryos; positive = toward L1
      stability: refit without each embryo, overlap of the top-50 |w| genes with the all-embryo fit
  (c) shared vs type-specific: cosine of gene-weight vectors between the top types; shared program = mean of unit |w|
  (d) PC importance: leave-one-PC-out drop of the pooled LOEO gain for the 8 PCs with the largest mean |beta|
Smoke (synthetic, scratch): a planted gene drives y through one PC; it must rank in the top 20 of its type.

Run (GitHub Actions, --petracer): python -I experiments/c14s4_genes.py
Smoke only: python -I experiments/c14s4_genes.py --synthetic OUTDIR
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c14s3_region as s3  # noqa: E402

OUT = HERE / "results_c14"
PCS = s3.PCS
TOP_TYPES, TOP_GENES, STAB_K, N_DROP = 8, 20, 50, 8
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def gene_names():
    f = sorted(HERE.parent.joinpath("data", "ext_petracer", "train_matrices").glob("*_features.tsv.gz"))[0]
    with gzip.open(f, "rt") as fh:
        return np.array([ln.split("\t")[1] for ln in fh])


def fit_coef(s1, g):
    Xg = s3.design(s1, g, ["pcs"])
    sc = StandardScaler().fit(Xg)
    m = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xg), g["y"])
    cols = list(Xg.columns)
    idx = [cols.index(p) for p in PCS]
    beta_std = m.coef_[0][idx]
    return beta_std, beta_std / sc.scale_[idx]


def gains(s1, D, embryos, drop=None):
    """Per-type LOEO gain (cell-weighted over embryos) and the embryo-mean pooled gain."""
    pb = s3.loeo(s1, D, embryos, "y", [])
    Dd = D
    if drop is not None:
        Dd = D.copy()
        Dd[drop] = 0.0  # a constant column carries no information after scaling
    pf = s3.loeo(s1, Dd, embryos, "y", ["pcs"])
    E = D.assign(p_base=pb, p_full=pf).dropna(subset=["p_base", "p_full"])
    rows = {}
    for T, g in E.groupby("T"):
        num = den = 0.0
        for _, h in g.groupby("embryo"):
            d = s1.g0.auc_delta(pd.DataFrame({"y": h["y"], "p_base": h["p_base"], "p_full": h["p_full"]}))
            if d is not None:
                num += d * len(h)
                den += len(h)
        if den:
            rows[T] = (num / den, int(den))
    per_emb = s3.per_embryo(s1, D, embryos, "y", pb, pf)
    return rows, float(np.mean(list(per_emb.values()))) if per_emb else float("nan")


def analyse(D, embryos, s1, comps, genes, outdir):
    tg, mean_gain = gains(s1, D, embryos)
    order = sorted(tg, key=lambda t: -tg[t][0])
    say(f"pooled LOEO gain (embryo mean) {mean_gain:+.4f} over {len(tg)} types")
    top = order[:TOP_TYPES]
    res_types, W, Bstd = {}, {}, {}
    for T in top:
        g = D[D["T"] == T]
        bstd, b = fit_coef(s1, g)
        w = comps.T @ b
        W[T], Bstd[T] = w, np.abs(bstd)
        topk = set(np.argsort(-np.abs(w))[:STAB_K])
        stab = []
        for emb in embryos:
            h = g[g.embryo != emb]
            if h["y"].nunique() < 2 or len(h) == len(g):
                continue
            _, bb = fit_coef(s1, h)
            stab.append(len(topk & set(np.argsort(-np.abs(comps.T @ bb))[:STAB_K])) / STAB_K)
        pos, neg = np.argsort(-w)[:TOP_GENES], np.argsort(w)[:TOP_GENES]
        L1 = str(g["L1"].iloc[0]) if "L1" in g else "L1"
        L2 = str(g["L2"].iloc[0]) if "L2" in g else "L2"
        res_types[T] = dict(gain=round(tg[T][0], 4), cells=tg[T][1], L1=L1, L2=L2,
                            toward_L1=[str(genes[i]) for i in pos], toward_L2=[str(genes[i]) for i in neg],
                            stability_top50=round(float(np.mean(stab)), 3) if stab else None)
        say(f"{T} gain {tg[T][0]:+.3f} n={tg[T][1]} [{L1} vs {L2}] stab {res_types[T]['stability_top50']}: "
            f"L1 {', '.join(res_types[T]['toward_L1'][:8])} | L2 {', '.join(res_types[T]['toward_L2'][:8])}")
    U = {T: W[T] / (np.linalg.norm(W[T]) or 1) for T in top}
    cos = {f"{a} | {b}": round(float(U[a] @ U[b]), 3) for i, a in enumerate(top) for b in top[i + 1:]}
    shared = np.mean([np.abs(U[T]) for T in top], axis=0)
    shared_genes = [str(genes[i]) for i in np.argsort(-shared)[:TOP_GENES]]
    mean_abs_cos = float(np.mean(np.abs(list(cos.values())))) if cos else float("nan")
    say(f"mean |cosine| between top types' gene weights {mean_abs_cos:.3f}; shared |w| top: {', '.join(shared_genes[:10])}")
    imp = np.mean([Bstd[T] for T in top], axis=0)
    drops = {}
    for k in np.argsort(-imp)[:N_DROP]:
        _, m_drop = gains(s1, D, embryos, drop=PCS[k])
        drops[PCS[k]] = round(mean_gain - m_drop, 4)
        top_load = [str(genes[i]) for i in np.argsort(-np.abs(comps[k]))[:10]]
        say(f"drop {PCS[k]}: gain falls by {drops[PCS[k]]:+.4f}; top loadings {', '.join(top_load)}")
    res = dict(mean_gain=mean_gain, type_gains={t: round(v[0], 4) for t, v in tg.items()}, top_types=res_types,
               cosine=cos, mean_abs_cosine=mean_abs_cos, shared_top_genes=shared_genes, pc_drop=drops,
               pc_top_loadings={PCS[k]: [str(genes[i]) for i in np.argsort(-np.abs(comps[k]))[:15]] for k in range(len(PCS))})
    Path(outdir).mkdir(parents=True, exist_ok=True)
    (Path(outdir) / "c14s4_genes.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    (Path(outdir) / "c14s4_genes.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    return res


def synthetic(outdir):
    """Planted gene 'G7' loads on pc1, and pc1 drives y in type T1 only; G7 must rank top 20 toward L1 in T1."""
    import c14s1_select as s1
    rng = np.random.default_rng(2)
    n_g = 200
    genes = np.array([f"G{i}" for i in range(n_g)])
    comps = rng.normal(0, 0.05, (len(PCS), n_g))
    comps[1, 7] = 1.0
    comps /= np.linalg.norm(comps, axis=1, keepdims=True)
    rows = []
    for emb in ["E8.0-R1", "E8.5-R2", "E9.0-R1", "E9.5-R1", "E8.0-R3", "E8.5-R3"]:
        for t in ["T1", "T2"]:
            n = 400
            P = rng.normal(size=(n, len(PCS)))
            y = ((1.5 * P[:, 1] * (t == "T1") + rng.normal(size=n)) > 0).astype(int)
            df = pd.DataFrame({"embryo": emb, "T": t, "y": y, "cluster": [f"{emb}{t}{i // 4}" for i in range(n)],
                               "total_counts": rng.integers(1000, 5000, n), "detection_rate": rng.random(n),
                               "clade": rng.integers(6, 50, n), "depth": rng.integers(1, 9, n), "stage": 8.5,
                               "lineage": "L", "germ_layer": "G", "L1": "A", "L2": "B"})
            for i, p in enumerate(PCS):
                df[p] = P[:, i]
            rows.append(df)
    D = pd.concat(rows, ignore_index=True)
    D.index = [f"c{i}" for i in range(len(D))]
    res = analyse(D, sorted(D.embryo.unique()), s1, comps, genes, outdir)
    assert "G7" in res["top_types"]["T1"]["toward_L1"], "planted gene not recovered"
    assert res["pc_drop"].get("pc1", 0) > 0.05, "planted PC drop too small"
    print("SYNTHETIC OK", res["top_types"]["T1"]["toward_L1"][:3], res["pc_drop"].get("pc1"))


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--synthetic":
        return synthetic(sys.argv[2])
    keep = {}
    D, embryos, s1 = s3.readout_with_pcs(keep=keep)
    genes = gene_names()[keep["hvg"]]
    comps = keep["comps"]
    say(f"readout cells {len(D)}; types {D['T'].nunique()}; HVG {len(genes)}")
    analyse(D, embryos, s1, comps, genes, OUT)


if __name__ == "__main__":
    main()
