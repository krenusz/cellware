"""Track B: does fate predictability rise with the number of sampled day-2 cells per clone?

Fits ridge on top-50 PCs once (5-fold CV by clone), then reports CV r within strata of day-2
cells per clone, and saves every clone's CV-predicted N propensity (used by the B2 simulator).

    uv run python experiments/track_b_depth.py
"""

from __future__ import annotations

import json
import os

import numpy as np
from track_b_state import HERE, cv_predict, load, pcs_of


def main():
    day, ctype, _, cid = load()
    pcs = pcs_of(50, day == 2)
    late = (day > 2) & (cid >= 0)
    nc = cid.max() + 1
    n = np.bincount(cid[late & (ctype == "Neutrophil")], minlength=nc)
    m = np.bincount(cid[late & (ctype == "Monocyte")], minlength=nc)
    d2 = (day == 2) & (cid >= 0)
    n2 = np.bincount(cid[d2], minlength=nc)
    keep = np.flatnonzero((n + m >= 3) & (n2 >= 1))
    y = n[keep] / (n[keep] + m[keep])
    state = np.stack([pcs[d2 & (cid == c)].mean(0) for c in keep])
    pred = cv_predict(state, y, np.random.default_rng(0).integers(0, 5, len(keep)), "ridge")
    out = {}
    for lo, hi in ((1, 1), (2, 2), (3, 99)):
        s = (n2[keep] >= lo) & (n2[keep] <= hi)
        out[f"day2_cells_{lo}-{hi}"] = {"clones": int(s.sum()),
                                        "cv_r": float(np.corrcoef(pred[s], y[s])[0, 1])}
    os.makedirs(os.path.join(HERE, "results_track_b"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results_track_b", "depth.json"), "w"), indent=1)
    np.save(os.path.join(HERE, "results_track_b", "propensity.npy"),
            np.clip(pred, 0.01, 0.99).astype(np.float32))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
