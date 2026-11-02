"""Track B3d: expression analogue of track_b_epiclone_m13 on the EPI-Clone scRNA-seq object.

Same LARRY transplant experiments, own cells: target = myeloid share Myeloid Progenitors /
(Myeloid Progenitors + MEP + EryP + MkP) among the clone's committed progenitors (>= 5); state =
mean top-20 expression PCs (log-normalised, 2000 most variable genes) of the clone's HSC/MPP1,
MPP3, MPP4 cells. Composition-only baseline and permuted null as in m13.

    uv run python experiments/track_b_epiclone_rna.py
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import rdata
import rds_raw
from track_b_epiclone import top_pcs
from track_b_state import cv_predict

HERE = os.path.dirname(__file__)
MULTI = ["HSC/MPP1", "MPP3", "MPP4"]
ERY = ["MEP", "EryP", "MkP"]


def main():
    warnings.filterwarnings("ignore")
    obj = rds_raw.attrs(rdata.parser.parse_file(
        os.path.join(HERE, "..", "data", "epiclone", "scRNA_seurat.RDS"), extension="rds").object)
    md = obj["meta.data"]
    clone = rds_raw.column(md, "LARRY").astype(str)
    ctype = rds_raw.column(md, "CellType").astype(str)
    x = rds_raw.dgc(rds_raw.attrs(rds_raw.named_list_item(obj["assays"], "RNA"))["counts"])
    multi = np.isin(ctype, MULTI)
    xm = x.T.tocsr()[multi]
    xm = xm.multiply(1e4 / np.maximum(xm.sum(1), 1)).tocsr()
    xm.data = np.log1p(xm.data)
    var = np.asarray(xm.multiply(xm).mean(0)).ravel() - np.asarray(xm.mean(0)).ravel() ** 2
    pcs = np.zeros((len(clone), 20))
    pcs[multi] = top_pcs(xm[:, np.argsort(var)[-2000:]].toarray(), 20)
    rows = []
    for c in set(clone) - {"", "nan", "NA"}:
        cells = clone == c
        my = (cells & (ctype == "Myeloid Progenitors")).sum()
        er = (cells & np.isin(ctype, ERY)).sum()
        if (cells & multi).sum() >= 1 and my + er >= 5:
            rows.append((c, my / (my + er), my + er))
    y = np.array([t[1] for t in rows])
    n = np.array([t[2] for t in rows])
    state = np.stack([pcs[(clone == c) & multi].mean(0) for c, *_ in rows])
    comp = np.stack([[((clone == c) & (ctype == t)).sum() for t in MULTI] for c, *_ in rows])
    comp = comp / comp.sum(1, keepdims=True)
    folds = np.random.default_rng(0).integers(0, 5, len(rows))
    perm = np.random.default_rng(1).permutation(len(rows))
    noise = np.mean(y * (1 - y) / np.maximum(n - 1, 1))
    out = {"clones": len(rows), "committed_per_clone": float(n.mean()),
           "target_mean": float(y.mean()), "noise_ceiling_r2": float(1 - noise / y.var())}
    for name, st in (("expression", state), ("permuted_null", state[perm]),
                     ("composition_only", comp),
                     ("composition_plus_expression", np.c_[comp, state])):
        p = cv_predict(st, y, folds, "ridge", lam=10.0)
        out[name] = {"cv_r": float(np.corrcoef(p, y)[0, 1]),
                     "r2": float(1 - ((y - p) ** 2).mean() / y.var())}
    p = cv_predict(state, y, folds, "ridge", lam=10.0)
    np.savez(os.path.join(HERE, "results_track_b", "epiclone_rna_pred.npz"), y=y, p=p)
    json.dump(out, open(os.path.join(HERE, "results_track_b", "epiclone_rna.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
