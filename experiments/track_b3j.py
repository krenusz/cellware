"""Track B3j: which CpGs carry the fate signal net of mouse and clone size? Plus EPI-Clone's own
clone-association p by site class. See the 2026-11-03 batch pre-registration.

    uv run python experiments/track_b3j.py
"""

from __future__ import annotations

import csv
import json
import os
import warnings

import numpy as np
from scipy.stats import fisher_exact, spearmanr
from track_b3fg import DATA, load
from track_b3i import resid
from track_b_epiclone_m13 import ERY, MULTI, MYEL

HERE = os.path.dirname(__file__)


def bh(p):
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def main():
    warnings.filterwarnings("ignore")
    meth, types, md = load()
    clone = md.LARRY.fillna("").to_numpy(dtype=str)
    ctype = md.CellType.astype(str).to_numpy()
    batch = (md.Experiment.astype(str) + "|" + md.ProcessingBatch.astype(str)).to_numpy()
    multi = np.isin(ctype, list(MULTI))
    rows = []
    for c in np.unique(clone[clone != ""]):
        cells = clone == c
        my = (cells & np.isin(ctype, list(MYEL))).sum()
        er = (cells & np.isin(ctype, list(ERY))).sum()
        if (cells & multi).sum() >= 1 and my + er >= 5:
            b, n = np.unique(batch[cells], return_counts=True)
            rows.append((c, my / (my + er), b[n.argmax()], cells.sum(), (cells & multi).sum()))
    y = np.array([t[1] for t in rows])
    mouse = np.array([t[2] for t in rows])
    mice = sorted(set(mouse))
    cov = np.c_[
        np.stack([mouse == m for m in mice[1:]], 1).astype(float),
        np.log([t[3] for t in rows]),
        np.log([t[4] for t in rows]),
    ]
    yr = resid(y, cov)
    x = np.stack([meth[(clone == c) & multi].mean(0) for c, *_ in rows])
    rho, p = np.zeros(x.shape[1]), np.ones(x.shape[1])
    for j in range(x.shape[1]):
        if x[:, j].std() > 0:
            rho[j], p[j] = spearmanr(resid(x[:, j], cov), yr)
    q = bh(p)
    with open(os.path.join(DATA, "info", "panel_info_dropout_pwm.tsv")) as f:
        head = next(csv.reader(f, delimiter="	"))
        # written by R with row names: one more field per row than in the header
        panel = list(csv.DictReader(f, fieldnames=["row", *head], delimiter="	"))
    sel = list(csv.DictReader(open(os.path.join(DATA, "info", "cpg_selection.csv"))))
    order = [r["CpG"] for r in sel]
    kinds = np.array([r["Type"] if r["Type"] not in ("", "NA") else "NA" for r in sel])
    assert len(order) == len(types) and (kinds == types).all(), "panel order != assay order"
    pinfo = {r["AmpID"]: r for r in panel}
    cloneass = {
        r["CpG"]: float(r["pvals_cloneass"]) for r in sel if r["pvals_cloneass"] not in ("NA", "")
    }
    hits = np.flatnonzero(q < 0.10)
    out = {
        "clones": len(rows),
        "cpgs_tested": int((p < 1).sum()),
        "fdr10_hits": int(len(hits)),
        "min_q": float(q.min()),
        "p_lt_0.01": int((p < 0.01).sum()),
        "expected_p_lt_0.01": round(0.01 * x.shape[1], 1),
    }
    top = np.argsort(p)[:15]
    out["top15"] = [
        {
            "cpg": order[j],
            "type": types[j],
            "rho": float(rho[j]),
            "p": float(p[j]),
            "q": float(q[j]),
            "gene": pinfo.get(order[j], {}).get("gene"),
            "enhancer": pinfo.get(order[j], {}).get("enhancer_annotation"),
            "class": pinfo.get(order[j], {}).get("Type"),
            "chrom_state": pinfo.get(order[j], {}).get("ChromState"),
        }
        for j in top
    ]
    if len(hits):
        a = [
            [int((types[hits] == "Dynamic").sum()), int((types[hits] == "Static").sum())],
            [int((types == "Dynamic").sum()), int((types == "Static").sum())],
        ]
        a[1] = [a[1][0] - a[0][0], a[1][1] - a[0][1]]
        out["hits_dynamic_vs_static_fisher"] = {"table": a, "p": float(fisher_exact(a)[1])}
    for t in ("Static", "Dynamic"):
        v = [cloneass[order[j]] for j in np.flatnonzero(types == t) if order[j] in cloneass]
        out[f"cloneass_p_lt_0.05_{t}"] = {"n": len(v), "share": float(np.mean(np.array(v) < 0.05))}
    json.dump(out, open(os.path.join(HERE, "results_track_b", "b3j.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
