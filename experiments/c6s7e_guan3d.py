"""C6-S7e: Guan force law (as C6-S7b, P-only receivers) in 3-D with a random mixed start, or in 2-D with P cells
starting outside (radial prepattern). See EXPERIMENT_LOG C6-S7de (pre-registered).

    uv run --with scikit-image --with numba python c6s7e_guan3d.py
"""

from __future__ import annotations

import json
import os
import time
from multiprocessing import Pool

import numpy as np

import c6s2_space as c2
import c6s3_break as b
import c6s7_checks as ck
import c6s7_sort as m7
import c6s7b_guan as gb

OUT = c2.OUT
N = 300
BASE = {**gb.BASE, "recv": "P"}
STAGE1 = [dict(a_PP=pp, a_AP=ap, beta=be, geom=ge) for ge in ("3d", "outer") for pp in (0.8625, 0.95)
          for ap in (0.8625, 0.95) for be in (0.125, 0.5)]
LAMS = (0.05, 0.2)
_LAST = {}


def multipolar(x, P):
    x3 = _LAST.get("x3")
    if x3 is not None and len(x3) == len(x):
        return m7.multipolar(x3, P, r=2.6)
    return m7.multipolar(x, P, r=2.6)


def _unit(v):
    return v / (np.linalg.norm(v, axis=1)[:, None] + 1e-12)


def _view2d(x, P):
    xc = x - x.mean(0)
    u = xc[P].mean(0) - xc[~P].mean(0) if P.any() and (~P).any() else np.eye(x.shape[1])[0]
    u = u / (np.linalg.norm(u) + 1e-12)
    res = xc - np.outer(xc @ u, u)
    w, v = np.linalg.eigh(np.cov(res.T))
    return np.c_[xc @ u, res @ v[:, -1]]


def simulate(s, c, prm, rng):
    n = len(s)
    P = c2.identities(s, prm["g"], prm["a_star"])
    dim = 3 if prm["geom"] == "3d" else 2
    if dim == 3:
        R0 = 1.1 * n ** (1 / 3)
        x = _unit(rng.standard_normal((n, 3))) * (R0 * rng.random(n) ** (1 / 3))[:, None]
    else:
        R0 = np.sqrt(n) * 1.05
        r = np.sort(R0 * np.sqrt(rng.random(n)))
        th = 2 * np.pi * rng.random(n)
        rad = np.empty(n)
        order = np.r_[rng.permutation(np.flatnonzero(~P)), rng.permutation(np.flatnonzero(P))]
        rad[order] = r  # P cells take the outermost radii
        x = np.c_[rad * np.cos(th), rad * np.sin(th)]
    PP = P[:, None] & P[None, :]
    AA = ~P[:, None] & ~P[None, :]
    alpha = np.where(PP, prm["a_PP"], np.where(AA, prm["a_AA"], prm["a_AP"]))
    beta = np.where(P[:, None] & P[None, :], prm["beta"], 0.0)
    np.fill_diagonal(beta, 0.0)
    pol = _unit(rng.standard_normal((n, dim)))
    links = np.zeros((0, 2), int)
    cosc = np.cos(np.radians(prm["cone"]))
    iu = np.triu_indices(n, 1)
    for step in range(int(round(prm["T"] / prm["dt"]))):
        d = x[:, None, :] - x[None, :, :]
        dist = np.sqrt((d ** 2).sum(-1))
        np.fill_diagonal(dist, np.inf)
        coef = np.where(dist < 2.0, 1.0 - dist / (2.0 * alpha), -beta / dist ** 2)
        F = np.einsum("mn,mnk->mk", coef / dist, d)
        if prm["lam_F"] > 0:
            if step % prm["t_int"] == 0:
                cm = dist[iu] < prm["r_link"]
                i, j = iu[0][cm], iu[1][cm]
                same = P[i] == P[j]
                ci, cj = i[same], j[same]
                e = _unit(x[cj] - x[ci])
                src, dst, ee = np.r_[ci, cj], np.r_[cj, ci], np.r_[e, -e]
                inside = np.abs(np.sum(ee * pol[src], 1)) >= cosc
                src, dst = src[inside], dst[inside]
                perm = rng.permutation(len(src))
                src, dst = src[perm], dst[perm]
                order = np.argsort(src, kind="stable")
                src, dst = src[order], dst[order]
                links = np.zeros((0, 2), int)
                if len(src):
                    first = np.r_[0, np.flatnonzero(np.diff(src)) + 1]
                    rank = np.arange(len(src)) - np.repeat(first, np.diff(np.r_[first, len(src)]))
                    links = np.c_[src, dst][rank < prm["n_links"]]
                if len(links):
                    acc = np.zeros((n, dim))
                    k = np.zeros(n)
                    np.add.at(acc, links[:, 0], pol[links[:, 1]])
                    np.add.at(k, links[:, 0], 1)
                    mm = k > 0
                    pol[mm] = _unit(prm["memory"] * pol[mm] + (1 - prm["memory"]) * acc[mm] / k[mm, None])
            if len(links):
                la, lb = links[:, 0], links[:, 1]
                fl = (prm["lam_F"] * c[la])[:, None] * _unit(x[lb] - x[la])
                np.add.at(F, la, fl)
                np.add.at(F, lb, -fl)
        x = x + prm["dt"] / prm["eta"] * F + np.sqrt(prm["dt"] * prm["kappa_r"]) / prm["eta"] * rng.standard_normal(x.shape)
    if dim == 3:
        _LAST["x3"] = x
        x2 = _view2d(x, P)
    else:
        _LAST["x3"] = None
        x2 = x
    return x2, P, b.shape_particles(x2)


def main():
    t0 = time.time()
    res = {"base": BASE, "stage1_grid": STAGE1, "lams": LAMS}
    pool = Pool(10)
    MOD = "c6s7e_guan3d"
    st1, st2, passing = {}, {}, []
    for gi, g in enumerate(STAGE1):
        v = ck.run0(pool, MOD, {**BASE, **g}, 1.0, 40000 + 20 * gi)
        st1[json.dumps(g)] = v
        print("stage1", g, {k: round(x, 3) for k, x in v.items()}, flush=True)
    res["stage1"] = st1
    go = [g for g in STAGE1 if st1[json.dumps(g)]["polar"] >= 0.5 and st1[json.dumps(g)]["multipolar_rate"] <= 0.2]
    res["stage1_passing"] = go
    if not go:
        res["note"] = "step 0 fails at stage 1: no point with polar >= 0.5 and multipolar <= 20% (pre-registered)"
    for gi, g in enumerate(go):
        for lam in LAMS:
            v = ck.step0_point(pool, MOD, {**BASE, **g, "lam_F": lam}, st1[json.dumps(g)], 41000 + 40 * gi + (lam > 0.1) * 20)
            key = json.dumps({**g, "lam_F": lam})
            if ck.criteria(v, 10):
                passing.append({**g, "lam_F": lam})
            st2[key] = v
            print("stage2", key, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()}, flush=True)
    res["stage2"] = st2
    res["step0_passing"] = passing
    if passing:
        best = max(passing, key=lambda g: st2[json.dumps(g)]["polar"])
        res["chosen"] = best
        print("chosen", best, flush=True)
        ck.downstream(pool, MOD, {**BASE, **best}, res)
    else:
        res.setdefault("note", "no step-0 point passes stage 2 (pre-registered)")
        print(res["note"])
    res["minutes"] = (time.time() - t0) / 60
    json.dump(res, open(os.path.join(OUT, "c6s7e_guan3d.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
