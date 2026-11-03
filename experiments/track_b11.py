"""B11: are clones that make neutrophils under M-CSF neutrophil-primed at day 2?
See the 2026-11-03 (6) batch pre-registration in EXPERIMENT_LOG.md.

    uv run python experiments/track_b11.py
"""

from __future__ import annotations

import gzip
import json
import os

import numpy as np
from fit_cytokine import CONDS5, load_counts, pooled
from scipy.stats import mannwhitneyu, spearmanr
from track_b8_b9 import CYTO, HERE


def day2_features():
    """{clone: (neutrophil-priming score, day-2 Neu share)} as in B9."""
    rows = [
        line.rstrip("\n").split("\t")
        for line in gzip.open(os.path.join(CYTO, "metadata.txt.gz"), "rt")
    ][1:]
    m = np.loadtxt(
        gzip.open(os.path.join(CYTO, "clone_matrix.mtx.gz"), "rt"),
        comments="%",
        skiprows=3,
        dtype=int,
    )
    d2 = [i for i, r in enumerate(rows) if r[2] == "2"]
    xy = np.array([[float(rows[i][5]), float(rows[i][6])] for i in d2])
    ct = np.array([rows[i][3] for i in d2])
    cn, cm = xy[ct == "Neu"].mean(0), xy[ct == "Mo"].mean(0)
    pos = {i: j for j, i in enumerate(d2)}
    cells = {}
    for clone, cell in zip(m[:, 0] - 1, m[:, 1] - 1, strict=True):
        if cell in pos:
            cells.setdefault(clone, []).append(pos[cell])
    out = {}
    for c, j in cells.items():
        j = np.array(j)
        score = np.mean(np.linalg.norm(xy[j] - cm, axis=1) - np.linalg.norm(xy[j] - cn, axis=1))
        out[c] = (float(score), float(np.mean(ct[j] == "Neu")))
    return out


def partial_spearman(x, y, z):
    """Spearman of x and y after removing a linear fit on z from the ranks of each."""
    rx, ry, rz = (np.argsort(np.argsort(v)).astype(float) for v in (x, y, z))
    a = np.c_[np.ones_like(rz), rz]

    def res(v):
        return v - a @ np.linalg.lstsq(a, v, rcond=None)[0]

    return spearmanr(res(rx), res(ry))


def main():
    pm = pooled(load_counts(CONDS5)["M"])
    feats = day2_features()
    keys = [k for k, v in pm.items() if v[0] + v[1] >= 2 and k in feats]
    has = np.array([pm[k][0] >= 1 for k in keys], float)
    size = np.log([pm[k][0] + pm[k][1] for k in keys])
    score = np.array([feats[k][0] for k in keys])
    neu = np.array([feats[k][1] for k in keys])
    out = {"clones": len(keys), "with_N": int(has.sum())}
    for name, v in (("neu_priming_score", score), ("day2_neu_share", neu)):
        r = partial_spearman(v, has, size)
        out[name] = {
            "partial_spearman": float(r[0]),
            "p": float(r[1]),
            "median_with_N": float(np.median(v[has == 1])),
            "median_none": float(np.median(v[has == 0])),
            "mannwhitney_p": float(mannwhitneyu(v[has == 1], v[has == 0]).pvalue),
        }
    out["supported"] = bool(
        out["neu_priming_score"]["partial_spearman"] > 0 and out["neu_priming_score"]["p"] < 0.05
    )
    json.dump(out, open(os.path.join(HERE, "results_e18", "b11.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
