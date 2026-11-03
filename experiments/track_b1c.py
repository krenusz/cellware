"""Track B1c: does day-2 expression -> fate (B1, r 0.66) survive library-grouped CV and
covariates (starting population, day-2 library, clone sizes)? See the 2026-11-03 batch
pre-registration.

    uv run python experiments/track_b1c.py
"""

from __future__ import annotations

import gzip
import json
import os

import numpy as np
from track_b3i import resid
from track_b_state import DATA, cv_predict, load, pcs_of

HERE = os.path.dirname(__file__)


def main():
    day, ctype, _, cid = load()
    with gzip.open(os.path.join(DATA, "metadata.txt.gz"), "rt") as f:
        rows = [line.rstrip("\n").split("\t") for line in f][1:]
    lib = np.array([r[0] for r in rows])
    pop = np.array([r[3] for r in rows])
    late = (day > 2) & (cid >= 0)
    n = np.bincount(cid[late & (ctype == "Neutrophil")], minlength=cid.max() + 1)
    m = np.bincount(cid[late & (ctype == "Monocyte")], minlength=cid.max() + 1)
    d2 = (day == 2) & (cid >= 0)
    n2 = np.bincount(cid[d2], minlength=cid.max() + 1)
    keep = np.flatnonzero((n + m >= 3) & (n2 >= 1))
    y = n[keep] / (n[keep] + m[keep])
    feat = pcs_of(50, day == 2)
    state = np.stack([feat[d2 & (cid == c)].mean(0) for c in keep])
    libs, pops = [], []
    for c in keep:
        v, k = np.unique(lib[d2 & (cid == c)], return_counts=True)
        libs.append(v[k.argmax()])
        pops.append(pop[d2 & (cid == c)][0])
    libs, pops = np.array(libs), np.array(pops)
    ul = sorted(set(libs))
    cov = np.c_[
        np.stack([libs == u for u in ul[1:]], 1).astype(float),
        (pops == "Lin-Kit+Sca1+").astype(float),
        np.log((n + m)[keep]),
        np.log(n2[keep]),
    ]
    rand = np.random.default_rng(0).integers(0, 5, len(keep))
    lolo = np.array([ul.index(u) for u in libs])
    idx = np.random.default_rng(1).integers(0, len(y), (2000, len(y)))

    def r_ci(p, t):
        bs = [np.corrcoef(p[i], t[i])[0, 1] for i in idx]
        return {"r": float(np.corrcoef(p, t)[0, 1]), "ci": np.percentile(bs, [2.5, 97.5]).tolist()}

    p = cv_predict(state, y, rand, "ridge")
    out = {
        "clones": len(keep),
        "clones_per_library": {u: int((libs == u).sum()) for u in ul},
        "lsk_share": float((pops == "Lin-Kit+Sca1+").mean()),
        "random_folds": r_ci(p, y),
        "leave_one_library_out": r_ci(cv_predict(state, y, lolo, "ridge"), y),
        "partial_covariates": r_ci(resid(p, cov), resid(y, cov)),
        "covariates_alone": r_ci(cv_predict(cov, y, rand, "ridge"), y),
    }
    json.dump(out, open(os.path.join(HERE, "results_track_b", "b1c.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
