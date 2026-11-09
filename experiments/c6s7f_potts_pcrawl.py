"""C6-S7f: Potts cluster-count check (part A) and posterior-only crawling (part B). See EXPERIMENT_LOG C6-S7f.

    uv run --with scikit-image --with numba --with matplotlib python c6s7f_potts_pcrawl.py
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
import c6s7_sort as m7

OUT = c2.OUT
N = pc.N
MCS = 100000
LAMS = (20.0, 80.0)


def domains(P, pairs, min_frac=0.1):
    iP = np.flatnonzero(P)
    m = len(iP)
    if m < 2:
        return 0
    idx = -np.ones(len(P), int)
    idx[iP] = np.arange(m)
    q = pairs[P[pairs[:, 0]] & P[pairs[:, 1]]]
    g = coo_matrix((np.ones(len(q)), (idx[q[:, 0]], idx[q[:, 1]])), shape=(m, m))
    _, lab = connected_components(g, directed=False)
    return int(np.sum(np.bincount(lab) >= min_frac * m))


def multipolar(x, P):
    return domains(P, pc.LAST["pairs"]) > 1


def simulate(s, c, prm, rng):
    return pc.simulate(s, c, prm, rng)


def jobA(seed):
    rng = np.random.default_rng(seed)
    s = np.r_[2 + rng.standard_normal(N // 2), -2 + rng.standard_normal(N // 2)]
    x, P, sh = pc.simulate(s, np.ones(N), {**pc.BASE, "mcs": MCS}, rng)
    return {"polar": b.polar_index(x, P), "multipolar_dist": bool(pc.multipolar(x, P)),
            "n_domains_contact": domains(P, pc.LAST["pairs"]), "multipolar_contact": bool(multipolar(x, P)),
            "elong": sh["moment_elong"]}, (pc.LAST["lat"], P)


def picture(items, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(items), figsize=(4 * len(items), 4))
    for ax, (lat, P, title) in zip(np.atleast_1d(axs), items):
        img = np.zeros(lat.shape + (3,))
        img[:] = 1.0
        typ = np.r_[-1, P.astype(int)]
        t = typ[lat]
        img[t == 0] = (0.35, 0.55, 0.85)  # anterior
        img[t == 1] = (0.90, 0.45, 0.20)  # posterior
        edge = np.zeros(lat.shape, bool)
        edge[:-1] |= lat[:-1] != lat[1:]
        edge[:, :-1] |= lat[:, :-1] != lat[:, 1:]
        img[edge & (lat > 0)] *= 0.6
        ys, xs = np.nonzero(lat)
        ax.imshow(img[ys.min() - 3: ys.max() + 4, xs.min() - 3: xs.max() + 4], interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    t0 = time.time()
    res = {"mcs": MCS, "lams": LAMS}
    pool = Pool(10)
    # ---- part A: same seeds as C6-S7d lam_F 0 at 100k MCS ----
    ra = pool.map(jobA, [50100 + k for k in range(6)])
    res["partA"] = [r[0] for r in ra]
    res["partA_mean"] = {k: float(np.mean([r[0][k] for r in ra])) for k in ra[0][0]}
    print("partA", json.dumps(res["partA"]), flush=True)
    pics = [(lat, P, "no crawling, 100k MCS, seed %d: %d contact domains" % (50100 + k, r["n_domains_contact"]))
            for k, (r, (lat, P)) in enumerate(ra[:3])]
    picture(pics, os.path.join(OUT, "c6s7f_lattice.png"))
    lam0 = {"polar": res["partA_mean"]["polar"], "multipolar_rate": res["partA_mean"]["multipolar_contact"],
            "elong": res["partA_mean"]["elong"]}
    # ---- part B: posterior-only crawling ----
    MOD = "c6s7f_potts_pcrawl"
    s0, passing = {}, []
    for gi, lam in enumerate(LAMS):
        g = {"mcs": MCS, "lam_F": lam, "crawl": "P"}
        v = ck.step0_point(pool, MOD, {**pc.BASE, **g}, lam0, 61000 + 20 * gi)
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
        json.dump(res, open(os.path.join(OUT, "c6s7f_potts_pcrawl.json"), "w"), indent=1)
        ck.downstream(pool, MOD, {**pc.BASE, **best}, res)
    else:
        res["note"] = "no step-0 point passes (pre-registered)"
        print(res["note"])
    res["minutes"] = (time.time() - t0) / 60
    json.dump(res, open(os.path.join(OUT, "c6s7f_potts_pcrawl.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
