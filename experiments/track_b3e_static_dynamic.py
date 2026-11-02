"""Track B3e: does EPI-Clone's fate prediction come from Static (clone-record) or Dynamic
(cell-state) CpGs? Same design as track_b_epiclone_m13; see the B3e pre-registration.

    uv run python experiments/track_b3e_static_dynamic.py
"""

from __future__ import annotations

import csv
import json
import os
import warnings

import numpy as np
import rdata
from track_b_epiclone import dgc, top_pcs
from track_b_epiclone_m13 import ERY, MULTI, MYEL
from track_b_state import cv_predict

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "epiclone")


def main():
    warnings.filterwarnings("ignore")
    r = rdata.read_rds(os.path.join(DATA, "M1-M3_seurat.rds"), default_encoding="utf8")
    md = vars(r)["meta.data"]
    keep = md.Experiment.astype(str).str.startswith("LARRY").to_numpy()
    counts = vars(r.assays["DNAm"])["counts"]
    names = np.array([str(x) for x in counts.Dimnames[0]])
    meth = (dgc(counts).T.toarray() > 0).astype(float)[keep]
    kind = {
        row["CpG"]: row["Type"]
        for row in csv.DictReader(open(os.path.join(DATA, "info", "cpg_selection.csv")))
    }
    types = np.array([kind.get(n, "NA") for n in names])
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
            rows.append((c, my / (my + er)))
    y = np.array([t[1] for t in rows])
    folds = np.random.default_rng(0).integers(0, 5, len(rows))
    comp = np.stack([[((clone == c) & (ctype == t)).sum() for t in sorted(MULTI)] for c, _ in rows])
    comp = comp / comp.sum(1, keepdims=True)
    rng = np.random.default_rng(0)
    out = {
        "clones": len(rows),
        "cpgs": {t: int((types == t).sum()) for t in ("Static", "Dynamic", "NA")},
    }
    for name, sel in (
        ("all", np.ones(len(types), bool)),
        ("static", types == "Static"),
        ("dynamic", types == "Dynamic"),
    ):
        pcs = top_pcs(meth[:, sel], 20)
        state = np.stack([pcs[(clone == c) & multi].mean(0) for c, _ in rows])
        p = cv_predict(state, y, folds, "ridge", lam=10.0)
        pb = cv_predict(np.c_[comp, state], y, folds, "ridge", lam=10.0)
        idx = rng.integers(0, len(y), (3000, len(y)))
        bs = np.array([np.corrcoef(p[i], y[i])[0, 1] for i in idx])
        out[name] = {
            "cv_r": float(np.corrcoef(p, y)[0, 1]),
            "ci": np.percentile(bs, [2.5, 97.5]).tolist(),
            "with_composition": float(np.corrcoef(pb, y)[0, 1]),
        }
        out[name + "_pred"] = p.tolist()
    ps, pd = np.array(out["static_pred"]), np.array(out["dynamic_pred"])
    idx = rng.integers(0, len(y), (3000, len(y)))
    d = np.array([np.corrcoef(ps[i], y[i])[0, 1] - np.corrcoef(pd[i], y[i])[0, 1] for i in idx])
    out["static_minus_dynamic_ci"] = np.percentile(d, [2.5, 97.5]).tolist()
    for k in ("all_pred", "static_pred", "dynamic_pred"):
        out.pop(k)
    json.dump(
        out, open(os.path.join(HERE, "results_track_b", "b3e_static_dynamic.json"), "w"), indent=1
    )
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
