"""Track B3k: within cell types, are clones with more MPP-like multipotent methylation more
myeloid? Pre-defined whole-class index. See the 2026-11-03 (2) batch pre-registration.

    uv run python experiments/track_b3k.py
"""

from __future__ import annotations

import csv
import json
import os
import warnings

import numpy as np
from scipy.stats import spearmanr
from track_b3fg import DATA, load
from track_b3i import resid
from track_b_epiclone_m13 import ERY, MULTI, MYEL

HERE = os.path.dirname(__file__)


def main():
    warnings.filterwarnings("ignore")
    meth, types, md = load()
    sel = [r["CpG"] for r in csv.DictReader(open(os.path.join(DATA, "info", "cpg_selection.csv")))]
    with open(os.path.join(DATA, "info", "panel_info_dropout_pwm.tsv")) as f:
        head = next(csv.reader(f, delimiter="\t"))
        cls = {
            r["AmpID"]: r["Type"]
            for r in csv.DictReader(f, fieldnames=["row", *head], delimiter="\t")
        }
    kind = np.array([cls.get(c, "") for c in sel])
    mpp, hsc = kind == "MPP_high", kind == "HSC_high"
    raw = meth[:, mpp].mean(1) - meth[:, hsc].mean(1)
    clone = md.LARRY.fillna("").to_numpy(dtype=str)
    ctype = md.CellType.astype(str).to_numpy()
    batch = (md.Experiment.astype(str) + "|" + md.ProcessingBatch.astype(str)).to_numpy()
    multi = np.isin(ctype, list(MULTI))
    cent = raw.copy()
    for g in np.unique(np.char.add(np.char.add(batch.astype(str), "#"), ctype.astype(str))):
        k = (np.char.add(np.char.add(batch.astype(str), "#"), ctype.astype(str)) == g) & multi
        cent[k] = raw[k] - raw[k].mean()
    rows = []
    for c in np.unique(clone[clone != ""]):
        cells = clone == c
        my = (cells & np.isin(ctype, list(MYEL))).sum()
        er = (cells & np.isin(ctype, list(ERY))).sum()
        if (cells & multi).sum() >= 1 and my + er >= 5:
            b, n = np.unique(batch[cells], return_counts=True)
            comp = [((cells & (ctype == t)).sum()) for t in sorted(MULTI)]
            rows.append(
                (c, my / (my + er), b[n.argmax()], cells.sum(), (cells & multi).sum(), comp)
            )
    y = np.array([t[1] for t in rows])
    mouse = np.array([t[2] for t in rows])
    mice = sorted(set(mouse))
    comp = np.array([t[5] for t in rows], float)
    comp = comp / comp.sum(1, keepdims=True)
    cov = np.c_[
        np.stack([mouse == m for m in mice[1:]], 1).astype(float),
        np.log([t[3] for t in rows]),
        np.log([t[4] for t in rows]),
        comp[:, 1:],
    ]
    out = {"clones": len(rows), "mpp_high_cpgs": int(mpp.sum()), "hsc_high_cpgs": int(hsc.sum())}
    idx = np.random.default_rng(0).integers(0, len(y), (2000, len(y)))
    for name, v in (("centred", cent), ("uncentred", raw)):
        x = np.array([v[(clone == c) & multi].mean() for c, *_ in rows])
        xr, yr = resid(x, cov), resid(y, cov)
        bs = [spearmanr(xr[i], yr[i])[0] for i in idx]
        per = {}
        for m in mice:
            k = mouse == m
            cm = cov[k][:, len(mice) - 1 :]
            per[m] = float(spearmanr(resid(x[k], cm), resid(y[k], cm))[0])
        out[name] = {
            "partial_spearman": float(spearmanr(xr, yr)[0]),
            "ci": np.percentile(bs, [2.5, 97.5]).tolist(),
            "per_mouse": per,
            "positive_mice": int(sum(v > 0 for v in per.values())),
        }
    json.dump(out, open(os.path.join(HERE, "results_track_b", "b3k.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
