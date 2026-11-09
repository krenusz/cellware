"""C6-S5: one new rule, s-independent adhesion (clones sort only by taxis on the Wnt pole). See EXPERIMENT_LOG C6-S5
(pre-registered, 9c3cf8c). Step 0 on synthetic mixes trims the grid; then c6s4_fit.main runs the identical LOCO CV
(and, only on pass, Q2/Q3/smoke/S1/S2) with the overrides passed through the environment.

    uv run --with scikit-image --with numba python c6s5_rule.py
"""

from __future__ import annotations

import importlib
import json
import os
from multiprocessing import Pool

import numpy as np

import c6s3_break as b

EXTRA = {"w": 1e9}  # adhesion exp(-|ds|/w) = 1 for every pair
U0 = 0.1
GRID0 = [(b0, chi) for b0 in (0.0, 0.05, 0.15) for chi in (1.0, 4.0, 16.0)]


def job0(a):
    b0, chi, seed = a
    rng = np.random.default_rng(seed)
    s = np.r_[2 + rng.standard_normal(150), -2 + rng.standard_normal(150)]
    prm = {**b.params("A", "4", 1.0), "b0": b0, "chi": chi, "u0": U0, **EXTRA}
    x, P, pr, sh = b.simulate_A(s, np.ones(300), prm, rng)
    return b.polar_index(x, P), float((s[P] < 0).mean()), sh["moment_elong"]


def main():
    step0 = {}
    with Pool(10) as p:
        for g in GRID0:
            r = np.array(p.map(job0, [g + (900 + k,) for k in range(5)])).mean(0)
            step0[str(g)] = {"polar": float(r[0]), "P_on_low_s": float(r[1]), "elong": float(r[2])}
            print("step0", g, {k: round(v, 2) for k, v in step0[str(g)].items()}, flush=True)
    keep = [g for g in GRID0 if step0[str(g)]["polar"] >= 0.5]
    out = os.path.join(b.OUT, "c6s5_step0.json")
    json.dump({"extra": EXTRA, "u0": U0, "step0": step0, "kept": keep}, open(out, "w"), indent=1)
    if not keep:
        print("no grid point keeps one axis: CV not runnable -> FAIL (pre-registered)")
        return
    os.environ["C6_EXTRA"] = json.dumps(EXTRA)
    os.environ["C6_GRID"] = json.dumps(keep)
    os.environ["C6_TAG"] = "c6s5_rule"
    fit = importlib.import_module("c6s4_fit")
    fit.STEP0 = step0
    fit.main()


if __name__ == "__main__":
    main()
