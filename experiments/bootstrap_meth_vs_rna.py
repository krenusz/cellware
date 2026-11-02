"""Bootstrap CI for CV r of methylation (EPI-Clone M.1-M.3) vs expression (EPI-Clone scRNA).

Resamples clones within each analysis independently (different clone sets) and reports each r
with a 95% CI and the CI of the difference.

    uv run python experiments/bootstrap_meth_vs_rna.py
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(__file__)
R = os.path.join(HERE, "results_track_b")


def boot_r(y, p, rng, n=5000):
    idx = rng.integers(0, len(y), (n, len(y)))
    return np.array([np.corrcoef(p[i], y[i])[0, 1] for i in idx])


def main():
    rng = np.random.default_rng(0)
    m, e = (
        np.load(os.path.join(R, "epiclone_m13_pred.npz")),
        np.load(os.path.join(R, "epiclone_rna_pred.npz")),
    )
    bm, be = boot_r(m["y"], m["p"], rng), boot_r(e["y"], e["p"], rng)
    d = bm - be
    out = {
        "methylation": {
            "r": float(np.corrcoef(m["p"], m["y"])[0, 1]),
            "ci": np.percentile(bm, [2.5, 97.5]).tolist(),
        },
        "expression": {
            "r": float(np.corrcoef(e["p"], e["y"])[0, 1]),
            "ci": np.percentile(be, [2.5, 97.5]).tolist(),
        },
        "difference_ci": np.percentile(d, [2.5, 97.5]).tolist(),
        "p_diff_le_0": float((d <= 0).mean()),
    }
    json.dump(out, open(os.path.join(R, "meth_vs_rna_bootstrap.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
