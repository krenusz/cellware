"""B16: does the day-2 Cebpa - Irf8 score predict clone N-share in the main in vitro LARRY data?
See the 2026-11-03 (8) batch pre-registration in EXPERIMENT_LOG.md.

    uv run python experiments/track_b16.py
"""

from __future__ import annotations

import gzip
import json
import os

import numpy as np
import pandas as pd
from track_b13 import partial_multi
from track_b_state import DATA, HERE, load

GENES = ["Cebpa", "Irf8", "Gfi1"]


def main():
    day, ctype, _, cid = load()
    names = [
        g.strip()
        for g in gzip.open(os.path.join(DATA, "stateFate_inVitro_gene_names.txt.gz"), "rt")
    ]
    col = {names.index(g) + 1: g for g in GENES}
    d2 = np.flatnonzero((day == 2) & (cid >= 0)) + 1  # 1-based cell rows
    expr = pd.DataFrame(0.0, index=d2, columns=GENES)
    want = set(d2)
    for chunk in pd.read_csv(
        os.path.join(DATA, "day2_normed_triplets.txt.gz"),
        sep=r"\s+",
        header=None,
        names=["c", "g", "v"],
        chunksize=5_000_000,
    ):
        chunk = chunk[chunk.g.isin(col.keys()) & chunk.c.isin(want)]
        for c, g, v in chunk.itertuples(index=False):
            expr.at[int(c), col[int(g)]] = np.log1p(v)
    expr["clone"] = cid[d2 - 1]
    per = expr.groupby("clone").agg(
        Cebpa=("Cebpa", "mean"), Irf8=("Irf8", "mean"), Gfi1=("Gfi1", "mean"), n2=("Cebpa", "size")
    )
    late = (day > 2) & (cid >= 0)
    n = np.bincount(cid[late & (ctype == "Neutrophil")], minlength=cid.max() + 1)
    m = np.bincount(cid[late & (ctype == "Monocyte")], minlength=cid.max() + 1)
    keys = [c for c in per.index if n[c] + m[c] >= 3]
    y = np.array([n[c] / (n[c] + m[c]) for c in keys])
    size = np.log([n[c] + m[c] for c in keys])
    n2 = np.log(per.loc[keys, "n2"].to_numpy())
    rep = {"clones": len(keys)}
    for name, v in (
        ("Cebpa_minus_Irf8", (per.Cebpa - per.Irf8).loc[keys].to_numpy()),
        ("Cebpa", per.loc[keys, "Cebpa"].to_numpy()),
        ("Irf8", per.loc[keys, "Irf8"].to_numpy()),
        ("Gfi1", per.loc[keys, "Gfi1"].to_numpy()),
    ):
        r, p = partial_multi(v, y, [size, n2])
        rep[name] = {"partial_rho": float(r), "p": float(p)}
    rep["supported"] = bool(
        rep["Cebpa_minus_Irf8"]["partial_rho"] > 0 and rep["Cebpa_minus_Irf8"]["p"] < 0.05
    )
    json.dump(rep, open(os.path.join(HERE, "results_e18", "b16.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
