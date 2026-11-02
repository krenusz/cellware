"""Track B3: does a clone's DNA-methylation state predict its fate better than its protein state?

EPI-Clone dataset M.4 (Scherer et al. 2025 Nature; Figshare 10.6084/m9.figshare.25472467):
mouse haematopoiesis, LARRY clones, single-cell targeted DNA methylation (scTAM-seq) plus 85
surface proteins in the same cells. For clones with >= 1 progenitor cell (HSC/MPP1, MPP3/4,
GMP, MEP, MkP) and >= 3 mature cells, target = lymphoid share of mature cells (pre/pro-B vs
macrophages, mature myeloid, DCs). State = mean over the clone's progenitor cells of the top-k
PCs of methylation or protein. 5-fold CV by clone, ridge (as track B step 1).

    uv run python experiments/track_b_epiclone.py
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import rdata
import scipy.sparse
from track_b_state import cv_predict

HERE = os.path.dirname(__file__)
PROG = {"HSC/MPP1", "MPP3/4", "GMP", "MEP", "MkP"}
LYMPH, MYEL = {"pre/pro-B"}, {"macrophages", "mature myeloid", "DCs"}


def dgc(m):
    """R dgCMatrix namespace -> scipy csc (features x cells)."""
    shape = tuple(int(v) for v in m.Dim)
    return scipy.sparse.csc_matrix((np.asarray(m.x), np.asarray(m.i), np.asarray(m.p)), shape)


def top_pcs(x, k):
    x = x - x.mean(0)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    return u[:, :k] * s[:k]


def main():
    warnings.filterwarnings("ignore")
    r = rdata.read_rds(os.path.join(HERE, "..", "data", "epiclone", "LARRY_mature.RDS"),
                       default_encoding="utf8")
    md = vars(r)["meta.data"]
    meth = (dgc(vars(r.assays["DNAm"])["counts"]).T.toarray() > 0).astype(float)  # cells x CpG
    prot = np.nan_to_num(np.asarray(r.assays["AB"].data, dtype=float).T)  # cells x 85
    clone = md.LARRY.fillna("").to_numpy(dtype=str)
    ctype = md.CellType.astype(str).to_numpy()
    is_prog = np.isin(ctype, list(PROG))
    feats = {"methylation": top_pcs(meth, 20), "protein": top_pcs(prot, 20)}
    rows = []
    for c in md.LARRY.dropna().unique():
        cells = clone == c
        ly = (cells & np.isin(ctype, list(LYMPH))).sum()
        my = (cells & np.isin(ctype, list(MYEL))).sum()
        if (cells & is_prog).sum() >= 1 and ly + my >= 3:
            rows.append((c, ly / (ly + my), (cells & is_prog).sum()))
    y = np.array([t[1] for t in rows])
    folds = np.random.default_rng(0).integers(0, 5, len(rows))
    out = {"clones": len(rows), "progenitor_cells_per_clone": float(np.mean([t[2] for t in rows])),
           "target_mean": float(y.mean()), "target_var": float(y.var())}
    for name, f in feats.items():
        state = np.stack([f[(clone == c) & is_prog].mean(0) for c, _, _ in rows])
        p = cv_predict(state, y, folds, "ridge", lam=10.0)
        out[name] = {"cv_r": float(np.corrcoef(p, y)[0, 1]),
                     "r2": float(1 - ((y - p) ** 2).mean() / y.var())}
    os.makedirs(os.path.join(HERE, "results_track_b"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results_track_b", "epiclone.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
