"""C6-S7g: Potts port with energies in the partial-wetting range (types stay attached). See EXPERIMENT_LOG C6-S7g.

    uv run --with scikit-image --with numba --with matplotlib python c6s7g_potts_wet.py
"""

from __future__ import annotations

import json
import os
import time
from multiprocessing import Pool

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import c6s2_space as c2
import c6s3_break as b
import c6s7_checks as ck
import c6s7c_potts as pc
import c6s7f_potts_pcrawl as pf

OUT = c2.OUT
N = pc.N
W = {"W1": dict(J_AA=10.0, J_PP=10.0, J_Am=20.0, J_Pm=20.0, J_AP=20.0),
     "W2": dict(J_AA=10.0, J_PP=10.0, J_Am=20.0, J_Pm=20.0, J_AP=30.0),
     "W3": dict(J_AA=10.0, J_PP=10.0, J_Am=20.0, J_Pm=25.0, J_AP=25.0),
     "W4": dict(J_AA=10.0, J_PP=10.0, J_Am=25.0, J_Pm=20.0, J_AP=25.0)}
MCS = (30000, 100000)
LAMS = (20.0, 80.0)


def attached(n):
    pr = pc.LAST["pairs"]
    g = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    return float(np.bincount(lab).max() / n)


def multipolar(x, P):
    return pf.domains(P, pc.LAST["pairs"]) > 1


def simulate(s, c, prm, rng):
    x, P, sh = pc.simulate(s, c, prm, rng)
    sh = {**sh, "attached": attached(len(x))}
    return x, P, sh


def job1(a):
    p, comp, seed = a
    rng = np.random.default_rng(seed)
    s = np.r_[2 + rng.standard_normal(N // 2), -2 + rng.standard_normal(N // 2)]
    t0 = time.time()
    x, P, sh = simulate(s, np.full(N, comp), p, rng)
    return {"polar": b.polar_index(x, P), "multipolar": bool(multipolar(x, P)), "attached": sh["attached"],
            "elong": sh["moment_elong"], "sec": time.time() - t0}, (pc.LAST["lat"], P)


def run1(pool, p, comp, seed0, reps=6):
    r = pool.map(job1, [(p, comp, seed0 + k) for k in range(reps)])
    v = {k: float(np.mean([a[0][k] for a in r])) for k in r[0][0]}
    v["multipolar_rate"] = v.pop("multipolar")
    return v, r


def main():
    t0 = time.time()
    res = {"W": W, "mcs": MCS, "lams": LAMS}
    pool = Pool(10)
    st1, pics, go = {}, [], []
    for wi, (wn, w) in enumerate(W.items()):
        for mi, m in enumerate(MCS):
            v, r = run1(pool, {**pc.BASE, **w, "mcs": m}, 1.0, 70000 + 100 * wi + 20 * mi)
            key = f"{wn}_{m}"
            v["pass1"] = bool(v["polar"] >= 0.5 and v["multipolar_rate"] <= 0.2 and v["attached"] >= 0.95)
            st1[key] = v
            if v["pass1"]:
                go.append((wn, m))
            if m == MCS[-1]:
                lat, P = r[0][1]
                pics.append((lat, P, f"{wn}, 100k MCS: polar {r[0][0]['polar']:.2f}"))
            print("stage1", key, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()}, flush=True)
    res["stage1"] = st1
    pf.picture(pics, os.path.join(OUT, "c6s7g_lattice.png"))
    st2, passing = {}, []
    for gi, (wn, m) in enumerate(go):
        for lam in LAMS:
            p = {**pc.BASE, **W[wn], "mcs": m, "lam_F": lam}
            v1, _ = run1(pool, p, 1.0, 71000 + 40 * gi + (lam > 50) * 20)
            vc, _ = run1(pool, p, 0.0, 71500 + 40 * gi + (lam > 50) * 20)
            l0 = st1[f"{wn}_{m}"]
            v = {**v1, "elong_c0": vc["elong"], "elong_lam0": l0["elong"], "polar_lam0": l0["polar"]}
            ok = ck.criteria(v, 300) and v["attached"] >= 0.95
            v["attached_ok"] = bool(v["attached"] >= 0.95)
            key = f"{wn}_{m}_lam{lam:.0f}"
            st2[key] = v
            if ok:
                passing.append((wn, m, lam))
            print("stage2", key, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()}, flush=True)
    res["stage2"] = st2
    res["step0_passing"] = passing
    if passing:
        wn, m, lam = max(passing, key=lambda q: st2[f"{q[0]}_{q[1]}_lam{q[2]:.0f}"]["polar"])
        res["chosen"] = {"W": wn, "mcs": m, "lam_F": lam}
        print("chosen", res["chosen"], flush=True)
        json.dump(res, open(os.path.join(OUT, "c6s7g_potts_wet.json"), "w"), indent=1)
        ck.downstream(pool, "c6s7g_potts_wet", {**pc.BASE, **W[wn], "mcs": m, "lam_F": lam}, res)
    else:
        res["note"] = ("step 0 fails at stage 1 (pre-registered)" if not go else "no step-0 point passes stage 2")
        print(res["note"])
    res["minutes"] = (time.time() - t0) / 60
    json.dump(res, open(os.path.join(OUT, "c6s7g_potts_wet.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
