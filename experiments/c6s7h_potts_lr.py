"""C6-S7h: attached Potts (C6-S7g energies) plus Guan's long-range 1/d^2 pull among posterior cells as a CPM force
term. See EXPERIMENT_LOG C6-S7h.

    uv run --with scikit-image --with numba --with matplotlib python c6s7h_potts_lr.py
"""

from __future__ import annotations

import json
import os
import time
from multiprocessing import Pool

import c6s2_space as c2
import c6s7_checks as ck
import c6s7c_potts as pc
import c6s7f_potts_pcrawl as pf
import c6s7g_potts_wet as pg

OUT = c2.OUT
N = pc.N
MCS = 100000
TABLES = ("W1", "W4")
BETAS = (300.0, 1000.0, 3000.0)
LAMS = (20.0, 80.0)
simulate = pg.simulate
multipolar = pg.multipolar


def main():
    t0 = time.time()
    res = {"tables": TABLES, "betas": BETAS, "mcs": MCS, "lams": LAMS}
    pool = Pool(10)
    st1, pics, go = {}, [], []
    for wi, wn in enumerate(TABLES):
        for bi, be in enumerate(BETAS):
            p = {**pc.BASE, **pg.W[wn], "mcs": MCS, "beta_lr": be}
            v, r = pg.run1(pool, p, 1.0, 80000 + 100 * wi + 20 * bi)
            key = f"{wn}_b{be:.0f}"
            v["pass1"] = bool(v["polar"] >= 0.5 and v["multipolar_rate"] <= 0.2 and v["attached"] >= 0.95)
            st1[key] = v
            if v["pass1"]:
                go.append((wn, be))
            lat, P = r[0][1]
            pics.append((lat, P, f"{wn}, beta {be:.0f}: polar {r[0][0]['polar']:.2f}"))
            print("stage1", key, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()}, flush=True)
    res["stage1"] = st1
    pf.picture(pics, os.path.join(OUT, "c6s7h_lattice.png"))
    st2, passing = {}, []
    for gi, (wn, be) in enumerate(go):
        for lam in LAMS:
            p = {**pc.BASE, **pg.W[wn], "mcs": MCS, "beta_lr": be, "lam_F": lam}
            v1, _ = pg.run1(pool, p, 1.0, 81000 + 40 * gi + (lam > 50) * 20)
            vc, _ = pg.run1(pool, p, 0.0, 81500 + 40 * gi + (lam > 50) * 20)
            l0 = st1[f"{wn}_b{be:.0f}"]
            v = {**v1, "elong_c0": vc["elong"], "elong_lam0": l0["elong"], "polar_lam0": l0["polar"]}
            ok = ck.criteria(v, 300) and v["attached"] >= 0.95
            v["attached_ok"] = bool(v["attached"] >= 0.95)
            key = f"{wn}_b{be:.0f}_lam{lam:.0f}"
            st2[key] = v
            if ok:
                passing.append((wn, be, lam))
            print("stage2", key, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()}, flush=True)
    res["stage2"] = st2
    res["step0_passing"] = passing
    if passing:
        wn, be, lam = max(passing, key=lambda q: st2[f"{q[0]}_b{q[1]:.0f}_lam{q[2]:.0f}"]["polar"])
        res["chosen"] = {"W": wn, "beta_lr": be, "lam_F": lam}
        print("chosen", res["chosen"], flush=True)
        json.dump(res, open(os.path.join(OUT, "c6s7h_potts_lr.json"), "w"), indent=1)
        ck.downstream(pool, "c6s7h_potts_lr", {**pc.BASE, **pg.W[wn], "mcs": MCS, "beta_lr": be, "lam_F": lam}, res)
    else:
        res["note"] = "step 0 fails at stage 1 (pre-registered)" if not go else "no step-0 point passes stage 2"
        print(res["note"])
    res["minutes"] = (time.time() - t0) / 60
    json.dump(res, open(os.path.join(OUT, "c6s7h_potts_lr.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
