"""Track B3b: methylation state of a clone's multipotent cells vs its myeloid / erythroid bias.

EPI-Clone datasets M.1-M.3 (Figshare 10.6084/m9.figshare.24204750), LARRY mice 1-4 (main +
replicate). Progenitor compartment only. For clones with >= 1 multipotent cell (HSC/MPP1-4)
and >= 5 committed progenitors, target = myeloid share GMP / (GMP + MEP + EryP + MkP).
State = mean over the clone's multipotent cells of the top-20 methylation PCs. 5-fold CV by
clone, ridge; null = same with clone labels of the state permuted.

    uv run python experiments/track_b_epiclone_m13.py
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import rdata
from track_b_epiclone import dgc, top_pcs
from track_b_state import cv_predict

HERE = os.path.dirname(__file__)
MULTI = {"HSC/MPP1", "MPP2", "MPP3", "MPP4"}
MYEL, ERY = {"GMP"}, {"MEP", "EryP", "MkP 1", "MkP 2"}


def main():
    warnings.filterwarnings("ignore")
    r = rdata.read_rds(os.path.join(HERE, "..", "data", "epiclone", "M1-M3_seurat.rds"),
                       default_encoding="utf8")
    md = vars(r)["meta.data"]
    keep = md.Experiment.astype(str).str.startswith("LARRY").to_numpy()
    meth = (dgc(vars(r.assays["DNAm"])["counts"]).T.toarray() > 0).astype(float)
    pcs = top_pcs(meth[keep], 20)
    md = md[keep]
    clone = md.LARRY.fillna("").to_numpy(dtype=str)
    ctype = md.CellType.astype(str).to_numpy()
    multi = np.isin(ctype, list(MULTI))
    rows = []
    for c in np.unique(clone[clone != ""]):
        cells = clone == c
        my = (cells & np.isin(ctype, list(MYEL))).sum()
        er = (cells & np.isin(ctype, list(ERY))).sum()
        if (cells & multi).sum() >= 1 and my + er >= 5:
            rows.append((c, my / (my + er), (cells & multi).sum(), my + er))
    y = np.array([t[1] for t in rows])
    state = np.stack([pcs[(clone == c) & multi].mean(0) for c, *_ in rows])
    folds = np.random.default_rng(0).integers(0, 5, len(rows))
    p = cv_predict(state, y, folds, "ridge", lam=10.0)
    perm = np.random.default_rng(1).permutation(len(rows))
    p0 = cv_predict(state[perm], y, folds, "ridge", lam=10.0)
    # confound check: the clone's multipotent cell-type composition alone
    comp = np.stack([[((clone == c) & (ctype == t)).sum() for t in sorted(MULTI)]
                     for c, *_ in rows])
    comp = comp / comp.sum(1, keepdims=True)
    pc = cv_predict(comp, y, folds, "ridge", lam=10.0)
    pb = cv_predict(np.c_[comp, state], y, folds, "ridge", lam=10.0)
    n = np.array([t[3] for t in rows])
    noise = np.mean(y * (1 - y) / np.maximum(n - 1, 1))
    out = {"clones": len(rows), "multipotent_cells_per_clone": float(np.mean([t[2] for t in rows])),
           "committed_per_clone": float(n.mean()), "target_mean": float(y.mean()),
           "target_var": float(y.var()), "noise_ceiling_r2": float(1 - noise / y.var()),
           "methylation": {"cv_r": float(np.corrcoef(p, y)[0, 1]),
                           "r2": float(1 - ((y - p) ** 2).mean() / y.var())},
           "permuted_null": {"cv_r": float(np.corrcoef(p0, y)[0, 1])},
           "composition_only": {"cv_r": float(np.corrcoef(pc, y)[0, 1])},
           "composition_plus_methylation": {"cv_r": float(np.corrcoef(pb, y)[0, 1])}}
    json.dump(out, open(os.path.join(HERE, "results_track_b", "epiclone_m13.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
