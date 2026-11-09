"""C6-S7d: de Jong Potts port (c6s7c_potts.simulate, unchanged) run 3-10x longer. See EXPERIMENT_LOG C6-S7de.

    uv run --with scikit-image --with numba python c6s7d_potts_long.py
"""

from __future__ import annotations

import json
import os
import time
from multiprocessing import Pool

import c6s2_space as c2
import c6s7_checks as ck
import c6s7c_potts as pc

OUT = c2.OUT
MCS = (30000, 100000)
GRID = [dict(mcs=m, lam_F=l) for m in MCS for l in (20.0, 80.0)]


def main():
    t0 = time.time()
    res = {"base": pc.BASE, "grid": GRID}
    pool = Pool(10)
    MOD = "c6s7c_potts"
    lam0 = {}
    for mi, m in enumerate(MCS):
        lam0[m] = ck.run0(pool, MOD, {**pc.BASE, "mcs": m}, 1.0, 50000 + 100 * mi)
        print("lam0", m, {k: round(x, 3) for k, x in lam0[m].items()}, flush=True)
    res["lam0"] = {str(k): v for k, v in lam0.items()}
    s0, passing = {}, []
    for gi, g in enumerate(GRID):
        v = ck.step0_point(pool, MOD, {**pc.BASE, **g}, lam0[g["mcs"]], 51000 + 20 * gi)
        if ck.criteria(v, 300):
            passing.append(g)
        s0[json.dumps(g)] = v
        print("step0", g, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()}, flush=True)
    res["step0"] = s0
    res["step0_passing"] = passing
    if passing:
        best = max(passing, key=lambda g: s0[json.dumps(g)]["polar"])
        res["chosen"] = best
        print("chosen", best, flush=True)
        ck.downstream(pool, MOD, {**pc.BASE, **best}, res)
    else:
        res["note"] = "no step-0 point passes (pre-registered)"
        print(res["note"])
    res["minutes"] = (time.time() - t0) / 60
    json.dump(res, open(os.path.join(OUT, "c6s7d_potts_long.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
