"""C6-S7: state-first sorting redesign (exploratory under H17). See EXPERIMENT_LOG C6-S7 (pre-registered, 47b34b2)
and lit/reading_c6_mechanism.md.

    uv run --with scikit-image --with numba python c6s7_sort.py

Rules: (1) identity from clone state via proportion feedback (no Wnt field); (2) binary sorting, like 1 / unlike 1/3;
(3) long-range attraction among posterior cells (Guan/Hormoz 2025); (4) same-type crawling with Vicsek polarity
(de Jong 2024 best variant), strength lam_F x competence. No sealed file is read.
"""

from __future__ import annotations

import json
import os
import time
from multiprocessing import Pool

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.stats import norm, spearmanr

import c6s2_space as c2
import c6s3_break as b
import c6s4_fit as f4

OUT = c2.OUT
N = 300
BASE = dict(steps=1500, dt=0.05, noise=0.3, k_rep=4.0, k_adh=3.0, r_adh=2.6, unlike=1 / 3, R_lr=8.0,
            t_int=20, n_links=3, cone=45.0, memory=0.99, g=c2.G_FB, a_star=c2.A_STAR)
GRID0 = [(beta, lam) for beta in (0.0, 0.05, 0.2) for lam in (0.0, 0.5, 2.0)]


def simulate_S(s, c, prm, rng):
    n = len(s)
    P = c2.identities(s, prm["g"], prm["a_star"])
    R0 = np.sqrt(n) * 1.05
    r = R0 * np.sqrt(rng.random(n))
    th = 2 * np.pi * rng.random(n)
    x = np.c_[r * np.cos(th), r * np.sin(th)]
    pol = 2 * np.pi * rng.random(n)
    links = np.zeros((0, 2), int)
    cosc = np.cos(np.radians(prm["cone"]))
    iP = np.flatnonzero(P)
    for step in range(prm["steps"]):
        pr = cKDTree(x).query_pairs(prm["r_adh"], output_type="ndarray")
        i, j = pr[:, 0], pr[:, 1]
        d = x[j] - x[i]
        dist = np.linalg.norm(d, axis=1) + 1e-9
        e = d / dist[:, None]
        # cohesion per type pair (C6-S7a); defaults reproduce C6-S7 (like 1, unlike prm["unlike"])
        wgt = np.where(P[i] & P[j], prm.get("w_PP", 1.0),
                       np.where(~P[i] & ~P[j], prm.get("w_AA", 1.0), prm.get("w_AP", prm["unlike"])))
        fm = np.where(dist < 2.0, -prm["k_rep"] * (2.0 - dist), prm["k_adh"] * wgt * (dist - 2.0))
        F = np.zeros_like(x)
        np.add.at(F, i, fm[:, None] * e)
        np.add.at(F, j, -fm[:, None] * e)
        # (3) long-range attraction among posterior cells
        if prm["beta"] > 0 and len(iP) > 1:
            lp = cKDTree(x[iP]).query_pairs(prm["R_lr"], output_type="ndarray")
            if len(lp):
                a_, b_ = iP[lp[:, 0]], iP[lp[:, 1]]
                sx = np.zeros_like(x)
                cnt = np.zeros(n)
                np.add.at(sx, a_, x[b_])
                np.add.at(sx, b_, x[a_])
                np.add.at(cnt, a_, 1)
                np.add.at(cnt, b_, 1)
                has = cnt > 0
                v = np.zeros_like(x)
                v[has] = sx[has] / cnt[has, None] - x[has]
                nv = np.linalg.norm(v, axis=1)
                ok = nv > 1e-9
                F[ok] += prm["beta"] * v[ok] / nv[ok, None]
        # (4) same-type crawling: refresh links + Vicsek polarity every t_int steps
        if prm["lam_F"] > 0:
            if step % prm["t_int"] == 0:
                same = P[i] == P[j]
                ci, cj, ce = i[same], j[same], e[same]
                src = np.r_[ci, cj]
                dst = np.r_[cj, ci]
                ee = np.r_[ce, -ce]
                pv = np.c_[np.cos(pol), np.sin(pol)]
                inside = np.abs(np.sum(ee * pv[src], 1)) >= cosc
                src, dst = src[inside], dst[inside]
                perm = rng.permutation(len(src))
                src, dst = src[perm], dst[perm]
                order = np.argsort(src, kind="stable")
                src, dst = src[order], dst[order]
                first = np.r_[0, np.flatnonzero(np.diff(src)) + 1] if len(src) else np.zeros(0, int)
                rank = np.arange(len(src)) - np.repeat(first, np.diff(np.r_[first, len(src)])) if len(src) else np.zeros(0, int)
                keep = rank < prm["n_links"]
                links = np.c_[src[keep], dst[keep]]
                if len(links):
                    acc = np.zeros((n, 2))
                    k = np.zeros(n)
                    np.add.at(acc, links[:, 0], pv[links[:, 1]])
                    np.add.at(k, links[:, 0], 1)
                    m = k > 0
                    newv = pv.copy()
                    newv[m] = prm["memory"] * pv[m] + (1 - prm["memory"]) * acc[m] / k[m, None]
                    pol = np.arctan2(newv[:, 1], newv[:, 0])
            if len(links):
                la, lb = links[:, 0], links[:, 1]
                dl = x[lb] - x[la]
                ul = dl / (np.linalg.norm(dl, axis=1)[:, None] + 1e-9)
                fl = (prm["lam_F"] * c[la])[:, None] * ul
                np.add.at(F, la, fl)
                np.add.at(F, lb, -fl)
        x = x + prm["dt"] * F + np.sqrt(prm["dt"]) * prm["noise"] * rng.standard_normal(x.shape)
    return x, P, b.shape_particles(x)


def multipolar(x, P, r=2.6, min_frac=0.1):
    iP = np.flatnonzero(P)
    if len(iP) < 2:
        return False
    pr = cKDTree(x[iP]).query_pairs(r, output_type="ndarray")
    m = len(iP)
    g = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(m, m)) if len(pr) else coo_matrix((m, m))
    _, lab = connected_components(g, directed=False)
    sizes = np.bincount(lab)
    return bool(np.sum(sizes >= min_frac * m) > 1)


def prm_of(beta, lam):
    return {**BASE, "beta": beta, "lam_F": lam}


def _sim(parts, prm, seed):
    rng = np.random.default_rng(seed)
    s, c, lab = b.gastruloid(parts, N, rng)
    x, P, sh = simulate_S(s, c, prm, rng)
    return x, P, sh, lab


def job0(a):
    beta, lam, comp, seed = a
    rng = np.random.default_rng(seed)
    s = np.r_[2 + rng.standard_normal(N // 2), -2 + rng.standard_normal(N // 2)]
    t0 = time.time()
    x, P, sh = simulate_S(s, np.full(N, comp), prm_of(beta, lam), rng)
    return b.polar_index(x, P), multipolar(x, P), sh["moment_elong"], time.time() - t0


def job_clone(a):
    beta, lam, k, seed = a
    x, P, sh, lab = _sim([(k, 0.5, c2.Z[k], 1.0), ("bulk", 0.5, None, 1.0)], prm_of(beta, lam), seed)
    return c2.polarity_of(f4.frac_profile(c2.axis_coords(x, P), lab == k)), multipolar(x, P), b.polar_index(x, P)


def job_q2(a):
    beta, lam, fr, seed = a
    x, P, sh, lab = _sim([("3", fr, c2.Z["3"], 1.0), ("1", 1 - fr, c2.Z["1"], 1.0)], prm_of(beta, lam), seed)
    t = c2.axis_coords(x, P)
    return f4.frac_profile(t, lab == "3"), f4.frac_profile(t, lab == "1")


def job_shape(a):
    beta, lam, parts, seed = a
    return _sim(parts, prm_of(beta, lam), seed)[2]


def main():
    t0 = time.time()
    res = {"base": BASE}
    pool = Pool(10)
    # ---- step 0 (synthetic) ----
    s0 = {}
    for gi, (beta, lam) in enumerate(GRID0):
        r1 = pool.map(job0, [(beta, lam, 1.0, 7000 + 10 * gi + k) for k in range(6)])
        r0 = pool.map(job0, [(beta, lam, 0.0, 7500 + 10 * gi + k) for k in range(6)])
        s0[(beta, lam)] = {"polar": float(np.mean([v[0] for v in r1])), "multipolar_rate": float(np.mean([v[1] for v in r1])),
                           "elong": float(np.mean([v[2] for v in r1])), "elong_c0": float(np.mean([v[2] for v in r0])),
                           "sec": float(np.mean([v[3] for v in r1]))}
        print("step0", (beta, lam), {k: round(v, 3) for k, v in s0[(beta, lam)].items()}, flush=True)
    passing = []
    for (beta, lam), v in s0.items():
        base_el = s0[(beta, 0.0)]["elong"]
        a_ = v["polar"] >= 0.5 and v["multipolar_rate"] <= 0.2
        b_ = lam > 0 and v["elong"] - base_el >= 0.15
        c_ = lam > 0 and abs(v["elong_c0"] - base_el) <= 0.05
        d_ = v["sec"] <= 10
        v.update({"S0a": bool(a_), "S0b": bool(b_), "S0c": bool(c_), "S0d": bool(d_)})
        if a_ and b_ and c_ and d_:
            passing.append((beta, lam))
    res["step0"] = {str(k): v for k, v in s0.items()}
    res["step0_passing"] = [list(p) for p in passing]
    if not passing:
        res["note"] = "no step-0 point passes: model fails, no data checks (pre-registered)"
        res["minutes"] = (time.time() - t0) / 60
        json.dump(res, open(os.path.join(OUT, "c6s7_sort.json"), "w"), indent=1)
        print(res["note"])
        return
    beta, lam = sorted(passing)[0]
    res["chosen"] = {"beta": beta, "lam_F": lam}
    print("chosen", beta, lam, flush=True)
    # ---- Q1 clone placement (prediction) ----
    sim, mp, pi = {}, {}, {}
    for ki, k in enumerate(f4.KS):
        v = pool.map(job_clone, [(beta, lam, k, 71000 + 100 * ki + r) for r in range(40)])
        sim[k] = float(np.mean([a[0] for a in v]))
        mp[k] = float(np.mean([a[1] for a in v]))
        pi[k] = float(np.mean([a[2] for a in v]))
    ks = f4.KS
    signs = int(sum(np.sign(sim[k]) == np.sign(f4.TGT[k]) for k in ks))
    rho = float(spearmanr([sim[k] for k in ks], [f4.TGT[k] for k in ks]).correlation)
    res["Q1"] = {"sim": sim, "target": f4.TGT, "signs": signs, "spearman": rho, "multipolar_rate": mp, "polar": pi,
                 "pass": bool(signs == 5 and rho >= 0.9)}
    print("Q1", json.dumps(res["Q1"]), flush=True)
    # ---- Q2 ----
    xs = (np.arange(c2.N_BINS) + 0.5) * 100 / c2.N_BINS
    q2 = []
    for i, fr in enumerate((0.2, 1 / 3, 0.5, 2 / 3, 0.8)):
        v = pool.map(job_q2, [(beta, lam, fr, 72000 + 100 * i + r) for r in range(20)])
        q2.append({"f_anterior": fr, "crossing": float(c2.crossing(xs, np.mean([a[0] for a in v], 0), np.mean([a[1] for a in v], 0)))})
    ok = [o for o in q2 if not np.isnan(o["crossing"])]
    sl = float(np.polyfit([100 * o["f_anterior"] for o in ok], [o["crossing"] for o in ok], 1)[0]) if len(ok) >= 3 else float("nan")
    lo, hi = c2.G1B["slope_ci"]
    res["Q2"] = {"crossings": q2, "slope": sl, "g1b_ci": [lo, hi], "pass": bool(lo <= sl <= hi)}
    # ---- Q3 ----
    q3 = {}
    for i, fr in enumerate((0.0, 0.2, 0.5, 0.8, 1.0)):
        parts = ([("bulk", 1.0, None, 1.0)] if fr == 0 else [("low", 1.0, 0.0, 0.2)] if fr == 1
                 else [("low", fr, 0.0, 0.2), ("bulk", 1 - fr, None, 1.0)])
        v = pool.map(job_shape, [(beta, lam, parts, 73000 + 100 * i + r) for r in range(20)])
        q3[str(fr)] = {m: float(np.mean([a[m] for a in v])) for m in v[0]}
    y = [q3[str(fr)]["moment_elong"] for fr in (0.0, 0.2, 0.5, 0.8, 1.0)]
    res["Q3"] = {"by_fraction": q3, "monotone_fall": bool(all(a >= c for a, c in zip(y, y[1:])))}
    print("Q2", sl, res["Q2"]["pass"], "Q3", [round(v, 3) for v in y], flush=True)
    # ---- S2 prediction + H17 power ----
    s2 = {}
    for name, z6 in (("z0", 0.0), ("z_clone6", c2.Z["6"])):
        parts = {"bulk": [("bulk", 1.0, None, 1.0)], "6a": [("6a", 1.0, z6, 0.2)],
                 "mix": [("6a", 0.5, z6, 0.2), ("bulk", 0.5, None, 1.0)]}
        m = {}
        for j, (key, pp) in enumerate(parts.items()):
            v = pool.map(job_shape, [(beta, lam, pp, 74000 + 1000 * j + 5000 * (name != "z0") + r) for r in range(100)])
            m[key] = float(np.mean([a["one_minus_circ"] for a in v]))
            m[key + "_moment"] = float(np.mean([a["moment_elong"] for a in v]))
        gap = m["bulk"] - m["6a"]
        frac = (m["mix"] - 0.5 * (m["bulk"] + m["6a"])) / gap if gap > 0 else float("nan")
        eff = frac * f4.PURE_GAP_DATA
        s2[name] = {**m, "frac_of_gap": frac, "effect_data_units": eff, "power": float(norm.sf(norm.ppf(0.95) - eff / f4.S2_SE))}
    res["S2"] = {"by_state": s2, "prediction": s2["z0"]["effect_data_units"], "power": s2["z0"]["power"],
                 "H17_label": "powered" if s2["z0"]["power"] >= 0.8 else "exploratory"}
    res["minutes"] = (time.time() - t0) / 60
    json.dump(res, open(os.path.join(OUT, "c6s7_sort.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("chosen", "Q2", "Q3", "S2")}, indent=1))


if __name__ == "__main__":
    main()
