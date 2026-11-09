"""C6-S7b: Guan/Hormoz force law exactly (Biophysical model supplement, two pre-differentiated types), identity from
clone state, plus de Jong same-type crawling for elongation. See EXPERIMENT_LOG C6-S7bc (pre-registered).

    uv run --with scikit-image --with numba python c6s7b_guan.py
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
L = 1.0  # cell radius l
BASE = dict(dt=0.2, T=150.0, eta=1.0, kappa_r=0.1, K_alpha=0.175, K_beta=0.125, r_link=2.2, t_int=20, n_links=3,
            cone=45.0, memory=0.99, g=c2.G_FB, a_star=c2.A_STAR, a_AA=0.775, lam_F=0.0)
STAGE1 = [dict(a_PP=pp, a_AP=ap, beta=be, recv=rv) for pp in (0.8625, 0.95) for ap in (0.8625, 0.95)
          for be in (0.125, 0.5) for rv in ("P", "all")]
LAMS = (0.05, 0.2)


def multipolar(x, P):
    return m7.multipolar(x, P, r=2.6)


def simulate(s, c, prm, rng):
    n = len(s)
    P = c2.identities(s, prm["g"], prm["a_star"])
    R0 = np.sqrt(n) * 1.05
    r = R0 * np.sqrt(rng.random(n))
    th = 2 * np.pi * rng.random(n)
    x = np.c_[r * np.cos(th), r * np.sin(th)]
    PP = P[:, None] & P[None, :]
    AA = ~P[:, None] & ~P[None, :]
    alpha = np.where(PP, prm["a_PP"], np.where(AA, prm["a_AA"], prm["a_AP"]))
    recv = P[:, None] if prm["recv"] == "P" else np.ones((n, 1), bool)
    beta = np.where(recv & P[None, :], prm["beta"], 0.0)  # beta[m, n]: pull on m from sender n
    np.fill_diagonal(beta, 0.0)
    pol = 2 * np.pi * rng.random(n)
    links = np.zeros((0, 2), int)
    cosc = np.cos(np.radians(prm["cone"]))
    steps = int(round(prm["T"] / prm["dt"]))
    iu = np.triu_indices(n, 1)
    for step in range(steps):
        d = x[:, None, :] - x[None, :, :]  # r_m - r_n
        dist = np.sqrt((d ** 2).sum(-1))
        np.fill_diagonal(dist, np.inf)
        short = dist < 2 * L
        coef = np.where(short, 1.0 - dist / (2 * L * alpha), -beta / dist ** 2)
        F = np.einsum("mn,mnk->mk", coef / dist, d)
        if prm["lam_F"] > 0:
            if step % prm["t_int"] == 0:
                cm = dist[iu] < prm["r_link"]
                i, j = iu[0][cm], iu[1][cm]
                same = P[i] == P[j]
                ci, cj = i[same], j[same]
                e = x[cj] - x[ci]
                e /= np.linalg.norm(e, axis=1)[:, None] + 1e-9
                src, dst, ee = np.r_[ci, cj], np.r_[cj, ci], np.r_[e, -e]
                pv = np.c_[np.cos(pol), np.sin(pol)]
                inside = np.abs(np.sum(ee * pv[src], 1)) >= cosc
                src, dst = src[inside], dst[inside]
                perm = rng.permutation(len(src))
                src, dst = src[perm], dst[perm]
                order = np.argsort(src, kind="stable")
                src, dst = src[order], dst[order]
                if len(src):
                    first = np.r_[0, np.flatnonzero(np.diff(src)) + 1]
                    rank = np.arange(len(src)) - np.repeat(first, np.diff(np.r_[first, len(src)]))
                    keep = rank < prm["n_links"]
                    links = np.c_[src[keep], dst[keep]]
                else:
                    links = np.zeros((0, 2), int)
                if len(links):
                    acc = np.zeros((n, 2))
                    k = np.zeros(n)
                    np.add.at(acc, links[:, 0], pv[links[:, 1]])
                    np.add.at(k, links[:, 0], 1)
                    mm = k > 0
                    newv = pv.copy()
                    newv[mm] = prm["memory"] * pv[mm] + (1 - prm["memory"]) * acc[mm] / k[mm, None]
                    pol = np.arctan2(newv[:, 1], newv[:, 0])
            if len(links):
                la, lb = links[:, 0], links[:, 1]
                dl = x[lb] - x[la]
                ul = dl / (np.linalg.norm(dl, axis=1)[:, None] + 1e-9)
                fl = (prm["lam_F"] * c[la])[:, None] * ul
                np.add.at(F, la, fl)
                np.add.at(F, lb, -fl)
        x = x + prm["dt"] / prm["eta"] * F + np.sqrt(prm["dt"] * prm["kappa_r"]) / prm["eta"] * rng.standard_normal(x.shape)
    return x, P, b.shape_particles(x)


def main():
    t0 = time.time()
    res = {"base": BASE, "stage1_grid": STAGE1, "lams": LAMS}
    pool = Pool(10)
    st1, st2, passing = {}, {}, []
    MOD = "c6s7b_guan"
    for gi, g in enumerate(STAGE1):
        v = ck.run0(pool, MOD, {**BASE, **g}, 1.0, 20000 + 20 * gi)
        st1[json.dumps(g)] = v
        print("stage1", g, {k: round(x, 3) for k, x in v.items()}, flush=True)
    res["stage1"] = st1
    go = [g for g in STAGE1 if st1[json.dumps(g)]["polar"] >= 0.5 and st1[json.dumps(g)]["multipolar_rate"] <= 0.2]
    res["stage1_passing"] = go
    if not go:
        res["note"] = "step 0 fails at stage 1: no point with polar >= 0.5 and multipolar <= 20% (pre-registered)"
    for gi, g in enumerate(go):
        for lam in LAMS:
            p = {**BASE, **g, "lam_F": lam}
            v = ck.step0_point(pool, MOD, p, st1[json.dumps(g)], 21000 + 40 * gi + (lam > 0.1) * 20)
            key = json.dumps({**g, "lam_F": lam})
            ok = ck.criteria(v, 10)
            st2[key] = v
            if ok:
                passing.append({**g, "lam_F": lam})
            print("stage2", key, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()}, flush=True)
    res["stage2"] = st2
    res["step0_passing"] = passing
    if not passing:
        res.setdefault("note", "no step-0 point passes stage 2 (pre-registered)")
        res["minutes"] = (time.time() - t0) / 60
        json.dump(res, open(os.path.join(OUT, "c6s7b_guan.json"), "w"), indent=1)
        print(res["note"])
        return
    best = max(passing, key=lambda g: st2[json.dumps(g)]["polar"])
    res["chosen"] = best
    print("chosen", best, flush=True)
    ck.downstream(pool, MOD, {**BASE, **best}, res)
    res["minutes"] = (time.time() - t0) / 60
    json.dump(res, open(os.path.join(OUT, "c6s7b_guan.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("chosen", "Q1", "Q2", "Q3", "S2")}, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
