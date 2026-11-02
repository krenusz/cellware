"""Track B3i: does B3's methylation -> fate prediction survive mouse-grouped CV and control for
clone size? See the B3i pre-registration.

    uv run python experiments/track_b3i.py
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
from track_b3fg import load
from track_b_epiclone import top_pcs
from track_b_epiclone_m13 import ERY, MULTI, MYEL
from track_b_state import cv_predict

HERE = os.path.dirname(__file__)


def resid(v, cov):
    c1 = np.c_[cov, np.ones(len(cov))]
    return v - c1 @ np.linalg.lstsq(c1, v, rcond=None)[0]


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
    rand = np.random.default_rng(0).integers(0, 5, len(rows))
    lomo = np.array([mice.index(m) for m in mouse])
    rng = np.random.default_rng(1)
    idx = rng.integers(0, len(y), (2000, len(y)))

    def r_ci(p, t):
        bs = [np.corrcoef(p[i], t[i])[0, 1] for i in idx]
        return {"r": float(np.corrcoef(p, t)[0, 1]), "ci": np.percentile(bs, [2.5, 97.5]).tolist()}

    out = {
        "clones": len(rows),
        "clones_per_mouse": {m: int((mouse == m).sum()) for m in mice},
        "baseline_mouse_size": r_ci(cv_predict(cov, y, rand, "ridge", lam=10.0), y),
    }
    yr = resid(y, cov)
    for name, sel in (
        ("all", np.ones(len(types), bool)),
        ("static", types == "Static"),
        ("dynamic", types == "Dynamic"),
    ):
        pcs = top_pcs(meth[:, sel], 20)
        state = np.stack([pcs[(clone == c) & multi].mean(0) for c, *_ in rows])
        p = cv_predict(state, y, rand, "ridge", lam=10.0)
        out[name] = {
            "random_folds": r_ci(p, y),
            "leave_one_mouse_out": r_ci(cv_predict(state, y, lomo, "ridge", lam=10.0), y),
            "partial_mouse_size": r_ci(resid(p, cov), yr),
        }
    json.dump(out, open(os.path.join(HERE, "results_track_b", "b3i.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
