"""B22: do M-CSF-only clones carry a monocyte-DC progenitor (MDP) signature?
See the B22 pre-registration in EXPERIMENT_LOG.md.

    uv run python experiments/track_b22.py
"""

from __future__ import annotations

import gzip
import json
import os

import numpy as np
import pandas as pd
from clone_state_model import OUT
from fit_cytokine import DATA
from track_b13 import fetch, partial_multi

GENES = ["Flt3", "Cd74", "H2-Aa", "H2-Ab1", "H2-Eb1"]


def load():
    names = [g.strip() for g in gzip.open(fetch("gene_names.txt.gz"), "rt")]
    col = {names.index(g) + 1: g for g in GENES if g in names}
    with gzip.open(os.path.join(DATA, "metadata.txt.gz"), "rt") as f:
        rows = [line.rstrip("\n").split("\t") for line in f][1:]
    meta = pd.DataFrame(
        {"day": [r[2] for r in rows], "kind": [r[3] for r in rows], "cond": [r[4] for r in rows]},
        index=np.arange(1, len(rows) + 1),
    )
    with gzip.open(os.path.join(DATA, "clone_matrix.mtx.gz"), "rt") as f:
        m = np.loadtxt(f, comments="%", skiprows=3, dtype=int)
    meta["clone"] = -1
    meta.loc[m[:, 1], "clone"] = m[:, 0] - 1
    sel = (meta.clone >= 0) & ((meta.day == "2") | ((meta.cond == "M-csf") & (meta.kind == "Mo")))
    want = meta.index[sel]
    x = pd.DataFrame(0.0, index=want, columns=list(col.values()))
    wset = set(want)
    with gzip.open(fetch("normed_counts.mtx.gz"), "rt") as f:
        line = f.readline()
        while line.startswith("%"):
            line = f.readline()
        for chunk in pd.read_csv(
            f, sep=" ", header=None, names=["c", "g", "v"], chunksize=5_000_000
        ):
            chunk = chunk[chunk.g.isin(col.keys()) & chunk.c.isin(wset)]
            for c, g, v in chunk.itertuples(index=False):
                x.at[c, col[g]] = np.log1p(v)
    return meta, x


def score(x, idx):
    z = (x.loc[idx] - x.loc[idx].mean()) / x.loc[idx].std().replace(0, 1)
    return z.mean(1)


def main():
    meta, x = load()
    mature = meta[(meta.day != "2") & meta.kind.isin(["Neu", "Mo"]) & (meta.clone >= 0)]
    size = mature.groupby(["clone", "cond"]).size().unstack(fill_value=0)
    d2 = x.index[meta.loc[x.index, "day"] == "2"]
    mono = x.index[meta.loc[x.index, "day"] != "2"]
    s2 = score(x, d2).groupby(meta.loc[d2, "clone"]).mean()
    sm = score(x, mono).groupby(meta.loc[mono, "clone"]).mean()
    t = pd.read_csv(os.path.join(OUT, "b14_clone_day2.csv"), index_col=0)
    t["m"] = t.cebpa - t.irf8
    g = size["G-csf"] if "G-csf" in size else 0
    f = size["full cocktail"] if "full cocktail" in size else 0
    mc = size["M-csf"]
    cl = size.index[(mc >= 2) & size.index.isin(s2.index) & size.index.isin(t.index)]
    only = ((g[cl] == 0) & (f[cl] == 0)).astype(int)
    shared = (g[cl] >= 2) | (f[cl] >= 2)
    keep = cl[((only == 1) | shared).to_numpy()]
    y = only[keep].to_numpy()

    def covs(k):
        return [t.loc[k, "m"].to_numpy(), np.log(mc[k].to_numpy())]

    r1, p1 = partial_multi(s2[keep].to_numpy(), y, covs(keep))
    k2 = keep[keep.isin(sm.index)]
    r2, p2 = partial_multi(sm[k2].to_numpy(), only[k2].to_numpy(), covs(k2))
    top = (s2[keep] >= np.percentile(s2[keep], 75)).to_numpy()
    rep = {
        "genes": list(x.columns),
        "n_M_only": int(y.sum()),
        "n_M_shared": int(len(y) - y.sum()),
        "T1_day2_MDP": {"partial_rho": float(r1), "p": float(p1)},
        "T2_mono_DClike": {"partial_rho": float(r2), "p": float(p2), "clones": len(k2)},
        "frac_M_only_in_top_quartile": float(y[top].mean()),
        "frac_M_only_overall": float(y.mean()),
        "MDP_mean_M_only": float(s2[keep][y == 1].mean()),
        "MDP_mean_M_shared": float(s2[keep][y == 0].mean()),
    }
    json.dump(rep, open(os.path.join(OUT, "b22_mdp.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
