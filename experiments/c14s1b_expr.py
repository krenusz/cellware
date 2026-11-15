"""C14-S1b: expression state (30 PCs of 2,000 HVGs) beyond cell type, TRAIN embryos only.

Pre-registered in EXPERIMENT_LOG (C14-S1b). Readout, LOEO, smoke tests and embryo-level power are the
C14-S1a code (c14s1_select), which filters to train embryos with partitions.train_only first.
Matrices: data/ext_petracer/train_matrices (7 train embryos only; sealed embryos never downloaded).

1. HVG: highest variance of log1p(count / author total_counts * 1e4) over all train cells of the six
   train embryos other than E9.5-R1 (zeros included).
2. Readout-set cells x HVG matrix from all 7 train embryos; z-score per gene (clip 10); PCA 30.
3. Feature set P = base + 30 PCs; smoke (i) shuffled |delta| < 0.01, (ii) planted delta > 0.05;
   embryo-level power on A; selection rule as pre-registered (compare with U from S1a).

Run: .venv/Scripts/python.exe -I experiments/c14s1b_expr.py
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c14s1_select as s1  # noqa: E402
from partitions import train_only  # noqa: E402,F401  (guard: partitioned reads go through train_only)
from power import sim_power  # noqa: E402,F401

MAT = HERE.parent / "data" / "ext_petracer" / "train_matrices"
OUT = HERE / "results_c14"
SEL = HERE / "selection" / "C14-S1b.json"
N_HVG, N_PC, CHUNK = 2000, 30, 20_000_000
HVG_SKIP = "E9.5-R1"
PCS = [f"pc{i}" for i in range(N_PC)]
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def files(emb):
    (b,) = MAT.glob(f"*_{emb}_barcodes.tsv.gz")
    (m,) = MAT.glob(f"*_{emb}_matrix.mtx.gz")
    return b, m


def header_lines(m):
    n = 0
    with gzip.open(m, "rt") as fh:
        for line in fh:
            n += 1
            if not line.startswith("%"):
                return n, [int(x) for x in line.split()]


def stream(m):
    skip, dims = header_lines(m)
    for ch in pd.read_csv(m, sep=" ", header=None, names=["g", "c", "v"], skiprows=skip,
                          chunksize=CHUNK, dtype={"g": np.int32, "c": np.int32, "v": np.float32}):
        yield ch, dims


def cell_scale(emb, totals):
    b, _ = files(emb)
    bc = pd.read_csv(b, header=None)[0].astype(str).to_numpy()
    tot = pd.Series(bc).map(totals).to_numpy(dtype=float)  # NaN for cells not in train obs
    return bc, 1e4 / tot


def main():
    OUT.mkdir(exist_ok=True)
    D, embryos = s1.build()
    # totals for every train cell (not only readout cells) come from the train obs table
    obs, _, raw = s1.g0.load_obs()
    del raw
    totals = obs["total_counts"].to_dict()
    del obs

    # 1. HVG on six embryos
    n_genes = None
    s = ss = None
    n_cells = 0
    for emb in embryos:
        if emb == HVG_SKIP:
            continue
        bc, scale = cell_scale(emb, totals)
        ok_cells = ~np.isnan(scale)
        n_cells += int(ok_cells.sum())
        for ch, dims in stream(files(emb)[1]):
            if n_genes is None:
                n_genes = dims[0]
                s, ss = np.zeros(n_genes), np.zeros(n_genes)
            c = ch["c"].to_numpy() - 1
            keep = ok_cells[c]
            x = np.log1p(ch["v"].to_numpy()[keep] * scale[c[keep]])
            g = ch["g"].to_numpy()[keep] - 1
            s += np.bincount(g, x, n_genes)
            ss += np.bincount(g, x * x, n_genes)
        say(f"HVG pass: {emb} done ({int(ok_cells.sum())} train cells)")
    var = ss / n_cells - (s / n_cells) ** 2
    hvg = np.argsort(-var)[:N_HVG]
    gpos = -np.ones(n_genes, np.int64)
    gpos[hvg] = np.arange(N_HVG)
    say(f"HVG chosen on {n_cells} cells; genes {n_genes}")

    # 2. readout cells x HVG
    rows = {c: i for i, c in enumerate(D.index)}
    R, C, V = [], [], []
    matched = 0
    for emb in embryos:
        bc, scale = cell_scale(emb, totals)
        rid = np.array([rows.get(x, -1) for x in bc])
        matched += int((rid >= 0).sum())
        for ch, _ in stream(files(emb)[1]):
            c = ch["c"].to_numpy() - 1
            g = gpos[ch["g"].to_numpy() - 1]
            keep = (rid[c] >= 0) & (g >= 0)
            R.append(rid[c[keep]])
            C.append(g[keep])
            V.append(np.log1p(ch["v"].to_numpy()[keep] * scale[c[keep]]).astype(np.float32))
        say(f"matrix pass: {emb} done")
    frac = matched / len(D)
    say(f"readout cells matched to a matrix barcode: {matched}/{len(D)} ({frac:.3f})")
    if frac < 0.9:
        say("STOP: adequacy rule (< 90% matched)")
        (OUT / "c14s1b_expr.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
        return
    M = sparse.csr_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))),
                          shape=(len(D), N_HVG)).toarray()
    mu, sd = M.mean(0), M.std(0)
    sd[sd == 0] = 1
    M = np.clip((M - mu) / sd, -10, 10)
    pcs = PCA(n_components=N_PC, random_state=0).fit_transform(M)
    del M
    for i, name in enumerate(PCS):
        D[name] = pcs[:, i]

    orig_X = s1.X

    def X(df, parts, planted=None):
        out = orig_X(df, [p for p in parts if p != "pcs"], planted)
        if "pcs" in parts:
            out = pd.concat([out, df[PCS]], axis=1)
        return out

    s1.X = X
    s1.LOG = LOG
    rng = np.random.default_rng(s1.SEED + 1)
    P = {"P": ["pcs"]}

    D["y_shuf"] = D["y"]
    for _, g in D.groupby(["T", "embryo"]):
        cy = g.groupby("cluster")["y"].first()
        perm = pd.Series(rng.permutation(cy.to_numpy()), index=cy.index)
        D.loc[g.index, "y_shuf"] = g["cluster"].map(perm).to_numpy()
    shuf = s1.run_sets(D, embryos, "y_shuf", sets=P)["P"][0]
    planted = D["y"] + rng.normal(0, 1.5, len(D))
    pl = s1.run_sets(D, embryos, "y", planted=planted, sets=P)["P"][0]
    smoke = abs(shuf) < 0.01 and pl > 0.05
    say(f"SMOKE shuffled {shuf:+.4f}, planted {pl:+.4f} -> {'PASS' if smoke else 'FAIL'}")

    d, E = s1.run_sets(D, embryos, "y", sets=P)["P"]
    per_emb = {e: s1.pooled_delta(E[E.embryo == e]) for e in embryos if (E.embryo == e).any()}
    pw, pwh, m, sdv, nsd = s1.embryo_power(E, rng)
    say(f"SET P: LOEO delta {d:+.4f}; per embryo " + ", ".join(f"{e} {v:+.3f}" for e, v in per_emb.items())
        + f"; embryo-level power on A {pw:.3f} (half {pwh:.3f}); sim mean {m:.4f} sd {sdv:.4f}, null sd {nsd:.4f}")
    u_file = OUT / "c14s1_select.json"
    u = json.loads(u_file.read_text())["sets"]["U"]["loeo_delta"] if u_file.exists() else None
    passP = bool(smoke and d >= 0.02 and pwh >= 0.8)
    if passP and (u is None or d >= u):
        sel = "P carried to the A look (needs A expression matrices: separate download, Bence OK)"
    elif passP:
        sel = "P passes but U's delta is larger: U stays"
    else:
        sel = "P fails the rule"
    say(f"U delta from S1a: {u}; SELECTION (S1b): {sel}")
    res = dict(smoke_shuffled=shuf, smoke_planted=pl, smoke_pass=bool(smoke), loeo_delta=d, per_embryo=per_emb,
               power_A=pw, power_A_half=pwh, sim_mean=m, sim_sd=sdv, null_sd=nsd, u_delta_s1a=u,
               selection=sel, n_hvg=N_HVG, n_pc=N_PC, matched=frac)
    (OUT / "c14s1b_expr.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    SEL.write_text(json.dumps(dict(experiment="C14-S1b", rule="pre-registered C14-S1b", **res), indent=1,
                              default=str), encoding="utf-8")
    (OUT / "c14s1b_expr.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
