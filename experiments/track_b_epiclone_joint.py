"""Track B3c: same clones, two readouts. Does methylation or expression of a clone's multipotent
cells predict its myeloid / erythroid bias better, and do they add up?

EPI-Clone M.1-M.3 (methylation) and the EPI-Clone scRNA-seq object (Figshare 24260743), both
from the same LARRY transplant experiments, different cells. Clones matched by LARRY barcode.
Target as in track_b_epiclone_m13 (from the methylation dataset's committed progenitors).
State: top-20 PCs of methylation, or of log-normalised expression (2000 most variable genes),
averaged over the clone's multipotent cells in each dataset. 5-fold CV by clone, ridge.

    uv run python experiments/track_b_epiclone_joint.py
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import rdata
import rds_raw
from track_b_epiclone import dgc, top_pcs
from track_b_epiclone_m13 import ERY, MULTI, MYEL
from track_b_state import cv_predict

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "epiclone")


def load(name, assay):
    r = rdata.read_rds(os.path.join(DATA, name), default_encoding="utf8")
    md = vars(r)["meta.data"]
    keep = md.Experiment.astype(str).str.startswith("LARRY").to_numpy()
    x = dgc(vars(r.assays[assay])["counts"]).T.tocsr()[keep]
    md = md[keep]
    return md.LARRY.fillna("").to_numpy(dtype=str), md.CellType.astype(str).to_numpy(), x


def main():
    warnings.filterwarnings("ignore")
    cm, tm, xm = load("M1-M3_seurat.rds", "DNAm")
    pm = top_pcs((xm.toarray() > 0).astype(float), 20)
    # the scRNA object needs the raw-tree reader (rdata cannot convert it)
    obj = rds_raw.attrs(rdata.parser.parse_file(os.path.join(DATA, "scRNA_seurat.RDS"),
                                                extension="rds").object)
    md = obj["meta.data"]
    cr = rds_raw.column(md, "LARRY").astype(str)
    tr = rds_raw.column(md, "CellType").astype(str)
    xr = rds_raw.dgc(rds_raw.attrs(rds_raw.named_list_item(obj["assays"], "RNA"))["counts"])
    xr = xr.T.tocsr()
    # target from the RNA dataset's own committed progenitors (cells independent of the state)
    rna_target = {}
    for c in set(cr) - {"", "nan", "NA"}:
        cells = cr == c
        my = (cells & (tr == "Myeloid Progenitors")).sum()
        er = (cells & np.isin(tr, ["MEP", "EryP", "MkP"])).sum()
        if my + er >= 5:
            rna_target[c] = my / (my + er)
    multi_r = np.isin(tr, list(MULTI))
    xr = xr[multi_r]
    xr = xr.multiply(1e4 / np.maximum(xr.sum(1), 1)).tocsr()
    xr.data = np.log1p(xr.data)
    var = np.asarray(xr.multiply(xr).mean(0)).ravel() - np.asarray(xr.mean(0)).ravel() ** 2
    pr = top_pcs(xr[:, np.argsort(var)[-2000:]].toarray(), 20)
    cr = cr[multi_r]
    multi_m = np.isin(tm, list(MULTI))
    rows = []
    for c in sorted(set(cm[multi_m]) & set(cr) & set(rna_target)):
        cells = cm == c
        my = (cells & np.isin(tm, list(MYEL))).sum()
        er = (cells & np.isin(tm, list(ERY))).sum()
        if my + er >= 5:
            rows.append((c, my / (my + er)))
    y = np.array([t[1] for t in rows])
    y_rna = np.array([rna_target[c] for c, _ in rows])
    sm = np.stack([pm[(cm == c) & multi_m].mean(0) for c, _ in rows])
    sr = np.stack([pr[cr == c].mean(0) for c, _ in rows])
    folds = np.random.default_rng(0).integers(0, 5, len(rows))
    out = {"clones": len(rows), "target_agreement_r": float(np.corrcoef(y, y_rna)[0, 1])}
    for tname, yy in (("target_methylation_cells", y), ("target_rna_cells", y_rna)):
        for name, st in (("methylation", sm), ("expression", sr), ("both", np.c_[sm, sr])):
            p = cv_predict(st, yy, folds, "ridge", lam=10.0)
            out[f"{tname}/{name}"] = {"cv_r": float(np.corrcoef(p, yy)[0, 1]),
                                      "r2": float(1 - ((yy - p) ** 2).mean() / yy.var())}
    json.dump(out, open(os.path.join(HERE, "results_track_b", "epiclone_joint.json"), "w"),
              indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
