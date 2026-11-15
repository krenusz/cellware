"""C14-S2: expression state beyond cell type predicts clade fate; one oracle look on petracer_embryo:A.

Serves: C14-S2. Bence: "1". Pre-registered in EXPERIMENT_LOG ("C14-S2").
  fit       : TRAIN embryos only (partitions.train_only via the C14 loaders). Runs on GitHub Actions
              (sweeps/c14s2_fit.txt). Freezes readout spec, HVG, z-score, PCA and per-type logistic
              models to results_c14/c14s2_model.json and writes results_c14/c14s2_selection.json
              (copied to selection/ locally and committed before the look).
  counts_A  : structural count of embryos in a sealed partition, from the public embryo names.
  test_A    : called ONLY by test_oracle (look petracer_c14s2 --partition A).
"""
from __future__ import annotations

import collections
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c14s0_gate as g0  # noqa: E402
import partitions  # noqa: E402
from partitions import train_only  # noqa: E402,F401  (train reads go through g0.load_obs -> train_only)

DS = "petracer_embryo"
RAW = HERE.parent / "data" / "ext_petracer"
TRAIN_MAT = RAW / "train_matrices"
SEALED_MAT = {"A": RAW / "sealed_A_matrices"}
OUT = HERE / "results_c14"
SEL = HERE / "selection" / "C14-S2.json"
MODEL = HERE / "selection" / "C14-S2_model.json"
N_HVG, N_PC, MIN_CL, CHUNK = 2000, 30, 30, 20_000_000
HVG_SKIP = "E9.5-R1"
SMOKE_EMB = "E8.5-R2"
TRAIN_SD, CRIT = 0.0374, 1.959964
NUM = ["ltc", "det", "lsz", "dep", "stage"]


# ---------- shared pieces ----------
def rel_table(obs, embryos, raw):
    ctype, lin = obs["cell_type"].to_dict(), obs["lineage"].to_dict()
    recs = []
    for key, tree in g0.iter_train_trees(raw, embryos):  # yields only keys of the given embryos
        r, _, _ = g0.relatives(tree, ctype, lin)
        recs += [(key,) + x for x in r]
    R = pd.DataFrame(recs, columns=["tree", "cell", "anc", "clade", "depth", "relcnt"]).set_index("cell").join(obs)
    R["cluster"] = R["tree"] + ":" + R["anc"]
    R["stage"] = R["embryo"].map(g0.stage_of)
    return R


def outcome(g, L1, L2):
    a = g["relcnt"].map(lambda c: c.get(L1, 0))
    b = g["relcnt"].map(lambda c: c.get(L2, 0))
    ok = (a + b >= 3) & (a != b)
    h = g[ok].copy()
    h["y"] = (a[ok] > b[ok]).astype(int)
    return h


def train_readout(R):
    rows, spec = [], {}
    for T, g in R.groupby("cell_type"):
        tot = collections.Counter()
        for c in g["relcnt"]:
            tot.update(c)
        if len(tot) < 2:
            continue
        (L1, n1), (L2, n2) = tot.most_common(2)
        allr = sum(tot.values())
        if n1 / allr < g0.MIN_SHARE or n2 / allr < g0.MIN_SHARE:
            continue
        h = outcome(g, L1, L2)
        if len(h) < g0.MIN_CELLS or h["y"].nunique() < 2:
            continue
        if min(h.loc[h.y == 1, "cluster"].nunique(), h.loc[h.y == 0, "cluster"].nunique()) < MIN_CL:
            continue
        h["T"] = T
        spec[T] = [L1, L2]
        rows.append(h.drop(columns=["relcnt"]))
    return pd.concat(rows), spec


def mat_files(d, emb):
    (b,) = Path(d).glob(f"*_{emb}_barcodes.tsv.gz")
    (f,) = Path(d).glob(f"*_{emb}_features.tsv.gz")
    (m,) = Path(d).glob(f"*_{emb}_matrix.mtx.gz")
    return b, f, m


def stream(m):
    skip = 0
    with gzip.open(m, "rt") as fh:
        for line in fh:
            skip += 1
            if not line.startswith("%"):
                dims = [int(x) for x in line.split()]
                break
    for ch in pd.read_csv(m, sep=" ", header=None, names=["g", "c", "v"], skiprows=skip, chunksize=CHUNK,
                          dtype={"g": np.int32, "c": np.int32, "v": np.float32}):
        yield ch, dims


def scale_of(b, totals):
    bc = pd.read_csv(b, header=None)[0].astype(str).to_numpy()
    return bc, 1e4 / pd.Series(bc).map(totals).to_numpy(dtype=float)


def extract(d, embryos, cells, totals, hvg_idx):
    """cells x HVG log-normalised matrix (rows in the order of `cells`)."""
    rows = {c: i for i, c in enumerate(cells)}
    gpos = None
    # dense float32 filled chunk by chunk (each cell x gene entry appears once); avoids COO concatenation OOM
    M = np.zeros((len(cells), len(hvg_idx)), np.float32)
    for emb in embryos:
        b, _, m = mat_files(d, emb)
        bc, scale = scale_of(b, totals)
        rid = np.array([rows.get(x, -1) for x in bc])
        for ch, dims in stream(m):
            if gpos is None:
                gpos = -np.ones(dims[0], np.int64)
                gpos[np.asarray(hvg_idx)] = np.arange(len(hvg_idx))
            c = ch["c"].to_numpy() - 1
            g = gpos[ch["g"].to_numpy() - 1]
            keep = (rid[c] >= 0) & (g >= 0)
            M[rid[c[keep]], g[keep]] += np.log1p(ch["v"].to_numpy()[keep] * scale[c[keep]]).astype(np.float32)
    matched = len(set(cells) & set(np.concatenate([scale_of(mat_files(d, e)[0], totals)[0] for e in embryos])))
    return M, matched / max(len(cells), 1)


def design(df, cols=None, pcs=None):
    num = pd.DataFrame({"ltc": np.log1p(df["total_counts"]), "det": df["detection_rate"],
                        "lsz": np.log(df["clade"]), "dep": df["depth"], "stage": df["stage"]}, index=df.index)
    X = pd.concat([num, pd.get_dummies(df[["lineage", "germ_layer"]].astype(str)).astype(float)], axis=1)
    if pcs is not None:
        X = pd.concat([X, pd.DataFrame(pcs, index=df.index, columns=[f"pc{i}" for i in range(pcs.shape[1])])], axis=1)
    if cols is not None:
        X = X.reindex(columns=cols, fill_value=0.0)
    return X


def fit_lr(X, y):
    from sklearn.linear_model import LogisticRegression
    mu, sd = np.array(X.mean(axis=0), dtype=float), np.array(X.std(axis=0), dtype=float)
    sd[~(sd > 0)] = 1
    m = LogisticRegression(max_iter=2000, C=1.0).fit((X.to_numpy() - mu) / sd, y)
    return dict(cols=list(X.columns), mu=mu.tolist(), sd=sd.tolist(), coef=m.coef_[0].tolist(), b=float(m.intercept_[0]))


def score(m, X):
    Z = (X.reindex(columns=m["cols"], fill_value=0.0).to_numpy() - np.array(m["mu"])) / np.array(m["sd"])
    return 1 / (1 + np.exp(-(Z @ np.array(m["coef"]) + m["b"])))


def project(model, M):
    Z = np.clip((M - np.array(model["gmu"])) / np.array(model["gsd"]), -10, 10)
    return (Z - np.array(model["pca_mean"])) @ np.array(model["pca_V"]).T


def deltas(D):
    from sklearn.metrics import roc_auc_score
    per = {}
    for e, ge in D.groupby("embryo"):
        num = den = 0.0
        for _, g in ge.groupby("T"):
            if g["y"].nunique() < 2:
                continue
            num += (roc_auc_score(g["y"], g["p_full"]) - roc_auc_score(g["y"], g["p_base"])) * len(g)
            den += len(g)
        if den:
            per[e] = num / den
    return per


def fit_models(D, pcs):
    models = {}
    for T, g in D.groupby("T"):
        if g["y"].nunique() < 2:
            continue
        models[T] = dict(base=fit_lr(design(g), g["y"]), full=fit_lr(design(g, pcs=pcs[D.index.get_indexer(g.index)]), g["y"]))
    return models


def apply_models(D, pcs, models):
    D = D.copy()
    D["p_base"] = np.nan
    D["p_full"] = np.nan
    for T, g in D.groupby("T"):
        if T not in models:
            continue
        D.loc[g.index, "p_base"] = score(models[T]["base"], design(g))
        D.loc[g.index, "p_full"] = score(models[T]["full"], design(g, pcs=pcs[D.index.get_indexer(g.index)]))
    return D.dropna(subset=["p_base", "p_full"])


# ---------- fit (train only; GitHub Actions) ----------
def fit():
    from sklearn.decomposition import PCA
    OUT.mkdir(exist_ok=True)
    obs, embryos, raw = g0.load_obs()  # train_only inside
    totals = obs["total_counts"].to_dict()
    R = rel_table(obs, embryos, raw)
    del raw
    D, spec = train_readout(R)
    print(f"train readout: types {len(spec)}, cells {len(D)}, clusters {D['cluster'].nunique()}", flush=True)

    # HVG on six train embryos (as S1b)
    s = ss = None
    n = 0
    for emb in embryos:
        if emb == HVG_SKIP:
            continue
        b, _, m = mat_files(TRAIN_MAT, emb)
        bc, scale = scale_of(b, totals)
        ok = ~np.isnan(scale)
        n += int(ok.sum())
        for ch, dims in stream(m):
            if s is None:
                s, ss = np.zeros(dims[0]), np.zeros(dims[0])
            c = ch["c"].to_numpy() - 1
            k = ok[c]
            x = np.log1p(ch["v"].to_numpy()[k] * scale[c[k]])
            gi = ch["g"].to_numpy()[k] - 1
            s += np.bincount(gi, x, len(s))
            ss += np.bincount(gi, x * x, len(s))
    var = ss / n - (s / n) ** 2
    hvg = np.sort(np.argsort(-var)[:N_HVG])
    feats = pd.read_csv(mat_files(TRAIN_MAT, embryos[0])[1], sep="\t", header=None)[0].astype(str).to_numpy()

    M, matched = extract(TRAIN_MAT, embryos, list(D.index), totals, hvg)
    gmu, gsd = M.mean(0), M.std(0)
    gsd[gsd == 0] = 1
    Z = np.clip((M - gmu) / gsd, -10, 10)
    pca = PCA(n_components=N_PC, random_state=0).fit(Z)
    pcs = pca.transform(Z)
    del M, Z

    # smoke: frozen-model path on one train embryo left out of the model fit
    keep = (D["embryo"] != SMOKE_EMB).to_numpy()
    sm = fit_models(D[keep], pcs[keep])
    smoke = deltas(apply_models(D[~keep], pcs[~keep], sm)).get(SMOKE_EMB)
    print(f"smoke: frozen path on held-out train embryo {SMOKE_EMB}: delta {smoke}", flush=True)

    models = fit_models(D, pcs)
    model = dict(spec=spec, hvg=hvg.tolist(), hvg_genes=feats[hvg].tolist(), gmu=gmu.tolist(), gsd=gsd.tolist(),
                 pca_mean=pca.mean_.tolist(), pca_V=pca.components_.tolist(), models=models)
    mtxt = json.dumps(model)
    (OUT / "c14s2_model.json").write_text(mtxt, encoding="utf-8")
    sides = {"L1_majority_clusters": int(D.loc[D.y == 1, "cluster"].nunique()),
             "L2_majority_clusters": int(D.loc[D.y == 0, "cluster"].nunique())}
    sel = {
        "test": "petracer_c14s2", "higher_is_better": True,
        "candidates": {"P": 0.0903, "F": 0.0813, "U": 0.0761, "S": 0.0160},
        "chosen": "P",
        "note": "candidates = C14-S1a/S1b train LOEO delta AUC; P = base + 30 expression PCs (train-fit HVG/z/PCA); "
                "primary = mean of A embryo deltas, known-sd z > 1.96, sd = max(0.0374, A sd)",
        "model_sha256": hashlib.sha256(mtxt.encode("utf-8")).hexdigest(),
        "train": {"types": len(spec), "cells": int(len(D)), "clusters": int(D["cluster"].nunique()),
                  "matched": matched, "smoke_heldout_embryo": SMOKE_EMB, "smoke_delta": smoke},
        "power": {"effect": 0.0556, "sd": TRAIN_SD, "alpha": 0.025, "one_sided": True, "min_power": 0.8,
                  "sources": {"effect": "C14-S1b train embryo-mean LOEO delta 0.0741 x 0.75 shrink",
                              "sd": "C14-S1b sd of 6 train embryo LOEO deltas"}},
        "readout_gate": {"pass": True, "sides": sides},
        "exposure": "train results; C13 headings; two re-analysis abstracts (topic level)",
    }
    (OUT / "c14s2_selection.json").write_text(json.dumps(sel, indent=1), encoding="utf-8")
    print(json.dumps({k: sel[k] for k in ("model_sha256", "train", "readout_gate")}, indent=1))


# ---------- oracle side (sealed; called only via test_oracle) ----------
NAMES = g0.NAMES


def counts_A(part="A"):
    """Structural: number of embryos in the sealed partition, from public names only."""
    return {"embryos": sum(1 for e in NAMES if partitions.partition_of(DS, e) == part)}


def _sealed_obs(part):
    import h5py
    (h5,) = sorted((RAW / "raw").glob("*.h5td"))
    f = h5py.File(h5, "r")
    cats = [g0._s(c) for c in f["obs/embryo/categories"][()]]
    emb = pd.DataFrame({"embryo": pd.Categorical.from_codes(f["obs/embryo/codes"][()], cats).astype(str)})
    emb["row"] = np.arange(len(emb))
    sel = partitions.sealed_rows(DS, emb, "embryo", part)
    mask = np.zeros(len(emb), bool)
    mask[sel["row"].to_numpy()] = True

    def col(name):
        g = f[f"obs/{name}"]
        if isinstance(g, h5py.Group):
            cc = [g0._s(c) for c in g["categories"][()]]
            return pd.Categorical.from_codes(g["codes"][mask], cc).astype(str)
        return g[mask]

    obs = pd.DataFrame({"cell": [g0._s(i) for i in f["obs/_index"][mask]]})
    for c in ["embryo", "cell_type", "lineage", "germ_layer", "total_counts", "detection_rate"]:
        obs[c] = col(c)
    raw = f["obst"][()]
    f.close()
    return obs.set_index("cell"), sorted(set(obs["embryo"])), raw


def test_A(part="A"):
    mtxt = MODEL.read_text(encoding="utf-8")
    assert hashlib.sha256(mtxt.encode("utf-8")).hexdigest() == json.loads(SEL.read_text())["model_sha256"], \
        "frozen model changed"
    model = json.loads(mtxt)
    obs, embryos, raw = _sealed_obs(part)
    R = rel_table(obs, embryos, raw)
    del raw
    rows = []
    for T, (L1, L2) in model["spec"].items():
        g = R[R["cell_type"] == T]
        if len(g):
            h = outcome(g, L1, L2)
            h["T"] = T
            rows.append(h.drop(columns=["relcnt"]))
    D = pd.concat(rows)
    feats = pd.read_csv(mat_files(SEALED_MAT[part], embryos[0])[1], sep="\t", header=None)[0].astype(str).to_numpy()
    assert feats[model["hvg"]].tolist() == model["hvg_genes"], "gene order differs from train"
    M, matched = extract(SEALED_MAT[part], embryos, list(D.index), obs["total_counts"].to_dict(), model["hvg"])
    if matched < 0.9:
        return {"stop": f"adequacy: only {matched:.3f} of readout cells matched", "confirmed": False}
    pcs = project(model, M)
    del M
    S = apply_models(D, pcs, model["models"])
    per = deltas(S)
    vals = np.array(list(per.values()))
    if len(vals) < 3:
        return {"stop": f"adequacy: {len(vals)} embryos with a delta", "per_embryo": per, "confirmed": False}
    sd = max(TRAIN_SD, float(vals.std(ddof=1)))
    z = float(vals.mean() / (sd / np.sqrt(len(vals))))
    from sklearn.metrics import roc_auc_score
    pooled_num = pooled_den = 0.0
    per_type = {}
    for T, g in S.groupby("T"):
        if g["y"].nunique() < 2:
            continue
        d = roc_auc_score(g["y"], g["p_full"]) - roc_auc_score(g["y"], g["p_base"])
        per_type[T] = round(float(d), 4)
        pooled_num += d * len(g)
        pooled_den += len(g)
    return {"n_embryos": int(len(vals)), "per_embryo": {k: round(float(v), 4) for k, v in per.items()},
            "mean_delta": round(float(vals.mean()), 4), "sd_used": round(sd, 4), "z": round(z, 3),
            "confirmed": bool(z > CRIT), "embryos_positive": int((vals > 0).sum()),
            "pooled_delta": round(pooled_num / pooled_den, 4), "cells": int(len(S)),
            "clusters": int(S["cluster"].nunique()), "matched": round(matched, 4), "per_type": per_type}


if __name__ == "__main__":
    if sys.argv[1:] == ["fit"]:
        fit()
