"""C6-S7c: de Jong 2024 Cellular Potts model ported (binary types, J medium 10 / like 10 / unlike 30, T 25, same-type
crawling links with Vicsek polarity), identity from clone state. See EXPERIMENT_LOG C6-S7bc (pre-registered).
The authors' C++ code was read as text only (data/ext_plos_gastruloid/repo); this is an independent numba port.

    uv run --with scikit-image --with numba python c6s7c_potts.py
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

OUT = c2.OUT
N = 300
KI = 16  # max links touching one cell
BASE = dict(side=5, A0=25.0, lam_a=5.0, T=25.0, J_like=10.0, J_unlike=30.0, J_med=10.0, t_int=20, n_links=3,
            cone=45.0, memory=0.99, g=c2.G_FB, a_star=c2.A_STAR, lam_F=0.0, mcs=3000)
GRID = [dict(mcs=m, lam_F=l) for m in (3000, 10000) for l in (20.0, 80.0)]

_K = None
LAST = {}


def _kernel():
    global _K
    if _K is not None:
        return _K
    import numba as nb

    @nb.njit(cache=True)
    def jf(a, b_, typ, J):
        if a == b_:
            return 0.0
        return J[typ[a], typ[b_]]

    @nb.njit(cache=True)
    def dlen(q, nx, ny, ox, oy, inc, ninc, la, lb, lw, sx, sy, area):
        tot = 0.0
        for t in range(ninc[q]):
            li = inc[q, t]
            o = lb[li] if la[li] == q else la[li]
            cx = sx[o] / area[o]
            cy = sy[o] / area[o]
            tot += lw[li] * (np.sqrt((nx - cx) ** 2 + (ny - cy) ** 2) - np.sqrt((ox - cx) ** 2 + (oy - cy) ** 2))
        return tot

    @nb.njit(cache=True)
    def sweep(lat, area, sx, sy, typ, J, T, lam_a, A0, inc, ninc, la, lb, lw, n_mcs, seed):
        np.random.seed(seed)
        L0, L1 = lat.shape
        DX = np.array([1, -1, 0, 0, 1, 1, -1, -1])
        DY = np.array([0, 0, 1, -1, 1, -1, 1, -1])
        # sample only the cells' bounding box (+2): sites outside are medium among medium, always no-ops;
        # attempts per MCS = box area, so the per-site attempt rate is unchanged
        y0, y1, x0, x1 = L0, 0, L1, 0
        for yy in range(L0):
            for xx in range(L1):
                if lat[yy, xx] > 0:
                    y0 = min(y0, yy); y1 = max(y1, yy); x0 = min(x0, xx); x1 = max(x1, xx)
        y0 = max(1, y0 - 2); y1 = min(L0 - 2, y1 + 2); x0 = max(1, x0 - 2); x1 = min(L1 - 2, x1 + 2)
        n_att = n_mcs * (y1 - y0 + 1) * (x1 - x0 + 1)
        for _ in range(n_att):
            y = np.random.randint(y0, y1 + 1)
            x = np.random.randint(x0, x1 + 1)
            k = np.random.randint(8)
            src = lat[y + DY[k], x + DX[k]]
            tgt = lat[y, x]
            if src == tgt:
                continue
            if tgt > 0 and area[tgt] <= 1:
                continue
            dH = 0.0
            for q in range(8):
                nb_ = lat[y + DY[q], x + DX[q]]
                dH += jf(src, nb_, typ, J) - jf(tgt, nb_, typ, J)
            if tgt > 0:
                dH += lam_a * ((area[tgt] - 1 - A0) ** 2 - (area[tgt] - A0) ** 2)
                if ninc[tgt] > 0:
                    a = area[tgt]
                    dH += dlen(tgt, (sx[tgt] - x) / (a - 1), (sy[tgt] - y) / (a - 1), sx[tgt] / a, sy[tgt] / a,
                               inc, ninc, la, lb, lw, sx, sy, area)
            if src > 0:
                dH += lam_a * ((area[src] + 1 - A0) ** 2 - (area[src] - A0) ** 2)
                if ninc[src] > 0:
                    a = area[src]
                    dH += dlen(src, (sx[src] + x) / (a + 1), (sy[src] + y) / (a + 1), sx[src] / a, sy[src] / a,
                               inc, ninc, la, lb, lw, sx, sy, area)
            if dH <= 0 or np.random.random() < np.exp(-dH / T):
                lat[y, x] = src
                if tgt > 0:
                    area[tgt] -= 1
                    sx[tgt] -= x
                    sy[tgt] -= y
                if src > 0:
                    area[src] += 1
                    sx[src] += x
                    sy[src] += y
        return lat

    _K = sweep
    return _K


def multipolar(x, P):
    return m7.multipolar(x, P, r=6.5)


def _links(lat, n, P, pol, prm, rng, cen):
    pr = b._contacts(lat)  # 0-based cell index pairs
    same = P[pr[:, 0]] == P[pr[:, 1]]
    if prm.get("crawl", "both") == "P":  # C6-S7f: posterior-only crawling
        same &= P[pr[:, 0]]
    ci, cj = pr[same, 0], pr[same, 1]
    e = cen[cj] - cen[ci]
    e /= np.linalg.norm(e, axis=1)[:, None] + 1e-9
    src, dst, ee = np.r_[ci, cj], np.r_[cj, ci], np.r_[e, -e]
    pv = np.c_[np.cos(pol), np.sin(pol)]
    inside = np.abs(np.sum(ee * pv[src], 1)) >= np.cos(np.radians(prm["cone"]))
    src, dst = src[inside], dst[inside]
    perm = rng.permutation(len(src))
    src, dst = src[perm], dst[perm]
    order = np.argsort(src, kind="stable")
    src, dst = src[order], dst[order]
    if not len(src):
        return np.zeros((0, 2), int), pol
    first = np.r_[0, np.flatnonzero(np.diff(src)) + 1]
    rank = np.arange(len(src)) - np.repeat(first, np.diff(np.r_[first, len(src)]))
    links = np.c_[src, dst][rank < prm["n_links"]]
    acc = np.zeros((n, 2))
    k = np.zeros(n)
    np.add.at(acc, links[:, 0], pv[links[:, 1]])
    np.add.at(k, links[:, 0], 1)
    mm = k > 0
    newv = pv.copy()
    newv[mm] = prm["memory"] * pv[mm] + (1 - prm["memory"]) * acc[mm] / k[mm, None]
    return links, np.arctan2(newv[:, 1], newv[:, 0])


def simulate(s, c, prm, rng):
    sweep = _kernel()
    lat, n = b._init_lattice(len(s), prm["side"], rng)
    s, c = s[:n], c[:n]
    P = c2.identities(s, prm["g"], prm["a_star"])
    typ = np.r_[2, P.astype(np.int64)]  # 0 anterior, 1 posterior, 2 medium
    # per-type energies (C6-S7g); defaults reproduce the symmetric de Jong table
    jaa, jpp = prm.get("J_AA", prm["J_like"]), prm.get("J_PP", prm["J_like"])
    jap = prm.get("J_AP", prm["J_unlike"])
    jam, jpm = prm.get("J_Am", prm["J_med"]), prm.get("J_Pm", prm["J_med"])
    J = np.array([[jaa, jap, jam],
                  [jap, jpp, jpm],
                  [jam, jpm, 0.0]])
    ys, xs = np.nonzero(lat)
    ids = lat[ys, xs]
    area = np.bincount(ids, minlength=n + 1).astype(np.float64)
    sx = np.bincount(ids, xs, minlength=n + 1).astype(np.float64)
    sy = np.bincount(ids, ys, minlength=n + 1).astype(np.float64)
    pol = 2 * np.pi * rng.random(n)
    empty = (np.zeros((n + 1, KI), np.int64), np.zeros(n + 1, np.int64), np.zeros(1, np.int64),
             np.zeros(1, np.int64), np.zeros(1))
    inc, ninc, la, lb, lw = empty
    for t in range(0, prm["mcs"], prm["t_int"]):
        if prm["lam_F"] > 0:
            cen = np.c_[sx[1:], sy[1:]] / np.maximum(area[1:], 1)[:, None]
            links, pol = _links(lat, n, P, pol, prm, rng, cen)
            if len(links):
                la, lb = links[:, 0] + 1, links[:, 1] + 1
                lw = prm["lam_F"] * c[links[:, 0]]
                inc = np.zeros((n + 1, KI), np.int64)
                ninc = np.zeros(n + 1, np.int64)
                for li in range(len(la)):
                    for q in (la[li], lb[li]):
                        if ninc[q] < KI:
                            inc[q, ninc[q]] = li
                            ninc[q] += 1
            else:
                inc, ninc, la, lb, lw = empty
        lat = sweep(lat, area, sx, sy, typ, J, prm["T"], prm["lam_a"], prm["A0"], inc, ninc, la.astype(np.int64),
                    lb.astype(np.int64), lw.astype(np.float64), int(prm["t_int"]), int(rng.integers(2 ** 31)))
    xy, _ = b._cells(lat, n)
    LAST["lat"], LAST["pairs"] = lat, b._contacts(lat)  # C6-S7f readouts
    return xy, P, b.shape_from_mask(lat > 0)


def main():
    t0 = time.time()
    res = {"base": BASE, "grid": GRID}
    pool = Pool(10)
    MOD = "c6s7c_potts"
    lam0 = {}
    for mi, m in enumerate((3000, 10000)):
        lam0[m] = ck.run0(pool, MOD, {**BASE, "mcs": m}, 1.0, 30000 + 100 * mi)
        print("lam0", m, {k: round(x, 3) for k, x in lam0[m].items()}, flush=True)
    res["lam0"] = {str(k): v for k, v in lam0.items()}
    s0, passing = {}, []
    for gi, g in enumerate(GRID):
        v = ck.step0_point(pool, MOD, {**BASE, **g}, lam0[g["mcs"]], 31000 + 20 * gi)
        ok = ck.criteria(v, 60)
        s0[json.dumps(g)] = v
        if ok:
            passing.append(g)
        print("step0", g, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()}, flush=True)
    res["step0"] = s0
    res["step0_passing"] = passing
    if not passing:
        res["note"] = "no step-0 point passes (pre-registered)"
        res["minutes"] = (time.time() - t0) / 60
        json.dump(res, open(os.path.join(OUT, "c6s7c_potts.json"), "w"), indent=1)
        print(res["note"])
        return
    best = max(passing, key=lambda g: s0[json.dumps(g)]["polar"])
    res["chosen"] = best
    print("chosen", best, flush=True)
    ck.downstream(pool, MOD, {**BASE, **best}, res)
    res["minutes"] = (time.time() - t0) / 60
    json.dump(res, open(os.path.join(OUT, "c6s7c_potts.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("chosen", "Q1", "Q2", "Q3", "S2")}, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
