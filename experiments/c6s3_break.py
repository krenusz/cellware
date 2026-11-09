"""C6-S3: two symmetry-breaking rules for the 2-D gastruloid. See EXPERIMENT_LOG C6-S3 (pre-registered).

    uv run --with numba --with scikit-image python c6s3_break.py smoke A|B [setting]
    uv run --with numba --with scikit-image python c6s3_break.py checks A|B [setting]

A: particle model (c6s2_space) + Wnt/T wave-pinning polarity on the contact graph + s-weighted taxis up the Wnt
   gradient + rule-3 extension.
B: Cellular Potts lattice, contact energy J0 + Jd |s_i - s_j|, area constraint, rank-rule identity (proportion
   feedback), rule 3 as a directed copy bias for posterior cells.
No sealed file is read.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

import c6s2_space as c2
import core_model as cm

OUT = c2.OUT
N_BINS = c2.N_BINS

# ---------------- shared readouts ----------------


def shape_from_mask(b, px=1.0):
    from skimage.measure import perimeter
    b = ndimage.binary_fill_holes(b)
    lab, nl = ndimage.label(b)
    if nl > 1:
        b = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    A = b.sum() * px * px
    P = perimeter(b) * px
    pts = np.argwhere(b).astype(float)
    ev = np.linalg.eigvalsh(np.cov(pts.T))
    return {"one_minus_circ": float(1 - 4 * np.pi * A / (P ** 2 + 1e-9)), "moment_elong": float(1 - np.sqrt(ev[0] / ev[1]))}


def shape_particles(x, px=0.5, sig=1.5):
    lo = x.min(0) - 6
    sz = np.ceil((x.max(0) + 6 - lo) / px).astype(int) + 1
    img = np.zeros(sz[::-1])
    p = ((x - lo) / px).astype(int)
    np.add.at(img, (p[:, 1], p[:, 0]), 1)
    img = ndimage.gaussian_filter(img, sig / px)
    return shape_from_mask(img > 0.5 * np.percentile(img[img > 0], 90), px)


def homotypic_pairs(pairs, P):
    return float(np.mean(P[pairs[:, 0]] == P[pairs[:, 1]])) if len(pairs) else float("nan")


def polar_index(x, P):
    return c2.sorting_index(x, P)


# ---------------- variant A: Wnt/T wave-pinning on the particle model ----------------

A_SETTINGS = {
    "1": dict(D=0.3, b0=0.05, mass=1.6, chi=1.0, w_sub=5, dt_u=0.1),
    "2": dict(D=0.15, b0=0.05, mass=1.6, chi=2.0, w_sub=10, dt_u=0.1),
    "3": dict(D=0.15, b0=0.05, mass=1.6, chi=4.0, w_sub=10, dt_u=0.1),
    # amendment C6-S3a (2026-11-09): settings 1-3 had V < 2, outside the bistable regime (no high Wnt state possible)
    "4": dict(D=0.3, b0=0.05, mass=2.9, chi=1.0, w_sub=10, dt_u=0.1, u0=0.5),
    "5": dict(D=0.3, b0=0.05, mass=2.9, chi=2.0, w_sub=10, dt_u=0.1, u0=0.5),
    "6": dict(D=0.3, b0=0.05, mass=2.9, chi=4.0, w_sub=10, dt_u=0.1, u0=0.5),
}
A_BASE = {**c2.DEFAULT, "steps": 1500, "k_adh": 3.0, "w": 0.5, "noise": 0.3, "k_ext": 0.0}


def simulate_A(s, c, prm, rng):
    n = len(s)
    R0 = np.sqrt(n) * 1.05
    r = R0 * np.sqrt(rng.random(n))
    th = 2 * np.pi * rng.random(n)
    x = np.c_[r * np.cos(th), r * np.sin(th)]
    u = prm.get("u0", 0.1) * rng.random(n)
    bs = prm["b0"] * c2.sigmoid(-s)
    M = prm["mass"] * n
    tax = np.tanh(-s / 2.0)  # + posterior-leaning (moves up the Wnt gradient)
    for _ in range(prm["steps"]):
        pr = cKDTree(x).query_pairs(2.4, output_type="ndarray")
        i, j = pr[:, 0], pr[:, 1]
        d = x[j] - x[i]
        dist = np.linalg.norm(d, axis=1) + 1e-9
        e = d / dist[:, None]
        # Wnt: wave-pinning with a global pool V (total mass M fixed)
        for _ in range(prm["w_sub"]):
            V = max(M - u.sum(), 0.0) / n
            lap = np.zeros(n)
            np.add.at(lap, i, u[j] - u[i])
            np.add.at(lap, j, u[i] - u[j])
            u = np.maximum(u + prm["dt_u"] * (prm["D"] * lap + (bs + u * u / (1 + u * u)) * V - u), 0)
        # mechanics: s-based adhesion (rule 1)
        a = np.exp(-np.abs(s[i] - s[j]) / prm["w"])
        f = np.where(dist < 2.0, -prm["k_rep"] * (2.0 - dist), prm["k_adh"] * a * (dist - 2.0))
        F = np.zeros_like(x)
        np.add.at(F, i, f[:, None] * e)
        np.add.at(F, j, -f[:, None] * e)
        # taxis up the local Wnt gradient
        gr = np.zeros_like(x)
        du = (u[j] - u[i])[:, None] * e
        np.add.at(gr, i, du)
        np.add.at(gr, j, du)
        F += prm["chi"] * tax[:, None] * gr
        P = u > 0.5 * u.max()
        if prm["k_ext"] > 0 and P.any() and (~P).any():
            F += _extension(x, P, c, i, j, d, prm["k_ext"])
        x = x + prm["dt"] * F + np.sqrt(prm["dt"]) * prm["noise"] * rng.standard_normal(x.shape)
    P = u > 0.5 * u.max()
    pr = cKDTree(x).query_pairs(2.4, output_type="ndarray")
    return x, P, pr, shape_particles(x)


def _extension(x, P, c, i, j, d, k):
    F = np.zeros_like(x)
    ca, cp = x[~P].mean(0), x[P].mean(0)
    ax = cp - ca
    rad = np.sqrt(np.mean(np.sum((x - x.mean(0)) ** 2, 1)))
    coh = min(1.0, np.linalg.norm(ax) / (rad + 1e-9))
    ax = ax / (np.linalg.norm(ax) + 1e-9)
    perp = np.array([-ax[1], ax[0]])
    pp = P[i] & P[j]
    if pp.any():
        ip, jp, dp = i[pp], j[pp], d[pp]
        st = k * coh * 0.5 * (c[ip] + c[jp])
        along = np.sign(dp @ ax)[:, None] * ax[None, :]
        across = np.sign(dp @ perp)[:, None] * perp[None, :]
        g = st[:, None] * (along - 0.5 * across)
        np.add.at(F, ip, -g)
        np.add.at(F, jp, g)
    return F


# ---------------- variant B: Cellular Potts ----------------

B_SETTINGS = {
    "1": dict(J0=2.0, Jd=4.0, Jm=8.0, T=4.0, lam=1.0, mcs=1500),
    "2": dict(J0=2.0, Jd=8.0, Jm=10.0, T=6.0, lam=1.0, mcs=2500),
    "3": dict(J0=2.0, Jd=8.0, Jm=10.0, T=8.0, lam=1.0, mcs=4000),
}
B_BASE = dict(n=300, area=16, side=4, k_ext=0.0, g=c2.G_FB, a_star=c2.A_STAR)

_NB = None


def _numba_kernels():
    global _NB
    if _NB is not None:
        return _NB
    import numba as nb

    @nb.njit(cache=True)
    def jfun(a, b, s, J0, Jd, Jm):
        if a == b:
            return 0.0
        if a == 0 or b == 0:
            return Jm
        return J0 + Jd * abs(s[a] - s[b])

    @nb.njit(cache=True)
    def sweep(lat, area, s, isP, cc, J0, Jd, Jm, T, lam, A0, kext, axx, axy, cx, cy, n_att, seed):
        np.random.seed(seed)
        L0, L1 = lat.shape
        dx = np.array([1, -1, 0, 0])
        dy = np.array([0, 0, 1, -1])
        for _ in range(n_att):
            y = np.random.randint(1, L0 - 1)
            x = np.random.randint(1, L1 - 1)
            k = np.random.randint(4)
            ys, xs = y + dy[k], x + dx[k]
            src = lat[ys, xs]
            tgt = lat[y, x]
            if src == tgt:
                continue
            dH = 0.0
            for q in range(4):
                nb_ = lat[y + dy[q], x + dx[q]]
                dH += jfun(src, nb_, s, J0, Jd, Jm) - jfun(tgt, nb_, s, J0, Jd, Jm)
            if tgt > 0:
                dH += lam * ((area[tgt] - 1 - A0) ** 2 - (area[tgt] - A0) ** 2)
            if src > 0:
                dH += lam * ((area[src] + 1 - A0) ** 2 - (area[src] - A0) ** 2)
                if isP[src] and kext > 0.0:
                    # copy direction (src -> target pixel) along the axis, pointing away from the body centre
                    vx, vy = x - xs, y - ys
                    proj = vx * axx + vy * axy
                    side = (x - cx) * axx + (y - cy) * axy
                    if proj * side > 0:
                        dH -= kext * cc[src]
            if dH <= 0 or np.random.random() < np.exp(-dH / T):
                lat[y, x] = src
                if tgt > 0:
                    area[tgt] -= 1
                if src > 0:
                    area[src] += 1
        return lat

    _NB = (jfun, sweep)
    return _NB


def _init_lattice(n, side, rng):
    R = np.sqrt(n * side * side / np.pi)
    L = int(2 * R * 3) + 10
    lat = np.zeros((L, L), np.int64)
    c0 = L / 2
    idx = 1
    g = int(np.ceil(R / side)) + 1
    spots = [(c0 + gy * side, c0 + gx * side) for gy in range(-g, g + 1) for gx in range(-g, g + 1)
             if (gy * side) ** 2 + (gx * side) ** 2 <= R * R]
    rng.shuffle(spots)
    for (yy, xx) in spots[:n]:
        lat[int(yy):int(yy) + side, int(xx):int(xx) + side] = idx
        idx += 1
    return lat, idx - 1


def _cells(lat, n):
    ys, xs = np.nonzero(lat)
    ids = lat[ys, xs]
    cnt = np.bincount(ids, minlength=n + 1).astype(float)
    cy = np.bincount(ids, ys, minlength=n + 1) / np.maximum(cnt, 1)
    cx = np.bincount(ids, xs, minlength=n + 1) / np.maximum(cnt, 1)
    return np.c_[cx, cy][1:], cnt[1:]


def _contacts(lat):
    a = np.r_[lat[:, :-1].ravel(), lat[:-1, :].ravel()]
    b = np.r_[lat[:, 1:].ravel(), lat[1:, :].ravel()]
    m = (a != b) & (a > 0) & (b > 0)
    pr = np.unique(np.sort(np.c_[a[m], b[m]], 1), axis=0)
    return pr - 1


def simulate_B(s, c, prm, rng):
    _, sweep = _numba_kernels()
    lat, n = _init_lattice(len(s), prm["side"], rng)
    s = s[:n]
    c = c[:n]
    P = c2.identities(s, prm["g"], prm["a_star"])
    s1 = np.r_[0.0, s]
    isP = np.r_[False, P]
    cc = np.r_[0.0, c]
    area = np.r_[0, np.bincount(lat.ravel(), minlength=n + 1)[1:]].astype(np.int64)
    n_att = int(2 * area.sum())
    chunk = 50
    for t in range(0, prm["mcs"], chunk):
        axx = axy = 0.0
        cx = cy = lat.shape[0] / 2
        if prm["k_ext"] > 0:
            xy, _ = _cells(lat, n)
            ca, cp = xy[~P].mean(0), xy[P].mean(0)
            ax = cp - ca
            rad = np.sqrt(np.mean(np.sum((xy - xy.mean(0)) ** 2, 1)))
            coh = min(1.0, np.linalg.norm(ax) / (rad + 1e-9))
            ax = ax / (np.linalg.norm(ax) + 1e-9) * coh
            axx, axy = float(ax[0]), float(ax[1])
            cx, cy = float(xy[:, 0].mean()), float(xy[:, 1].mean())
        for k in range(chunk):
            lat = sweep(lat, area, s1, isP, cc, prm["J0"], prm["Jd"], prm["Jm"], prm["T"], prm["lam"],
                        float(prm["area"]), float(prm["k_ext"]), axx, axy, cx, cy, n_att,
                        int(rng.integers(2 ** 31)))
    xy, _ = _cells(lat, n)
    return xy, P, _contacts(lat), shape_from_mask(lat > 0), n


# ---------------- runners ----------------

def run(variant, s, c, prm, rng):
    if variant == "A":
        x, P, pr, sh = simulate_A(s, c, prm, rng)
        return x, P, pr, sh, len(s)
    return simulate_B(s, c, prm, rng)


def params(variant, setting, k_ext=0.0):
    if variant == "A":
        return {**A_BASE, **A_SETTINGS[setting], "k_ext": k_ext}
    return {**B_BASE, **B_SETTINGS[setting], "k_ext": k_ext}


K_EXT = {"A": 1.0, "B": 3.0}


def smoke(variant, setting, reps=6, seed=631):
    rng = np.random.default_rng(seed)
    n = 300
    res = {"variant": variant, "setting": setting}

    def mix():
        return np.r_[2 + rng.standard_normal(n // 2), -2 + rng.standard_normal(n // 2)]

    t0 = time.time()
    p1, e0 = [], []
    for _ in range(reps):
        x, P, pr, sh, m = run(variant, mix(), np.ones(n), params(variant, setting), rng)
        p1.append((polar_index(x, P), homotypic_pairs(pr, P)))
        e0.append(sh["moment_elong"])
    sec = (time.time() - t0) / reps
    p1 = np.mean(p1, 0)
    res["P1"] = {"polar": float(p1[0]), "homotypic": float(p1[1]), "pass": bool(p1[0] >= 0.5 and p1[1] >= 0.75)}
    p2 = []
    for _ in range(reps):
        x, P, pr, sh, m = run(variant, rng.standard_normal(n), np.ones(n), params(variant, setting), rng)
        p2.append(polar_index(x, P))
    res["P2"] = {"polar": float(np.mean(p2)), "pass": bool(np.mean(p2) >= 0.5)}
    e1, e2 = [], []
    for _ in range(reps):
        x, P, pr, sh, m = run(variant, mix(), np.ones(n), params(variant, setting, K_EXT[variant]), rng)
        e1.append(sh["moment_elong"])
        x, P, pr, sh, m = run(variant, mix(), np.zeros(n), params(variant, setting, K_EXT[variant]), rng)
        e2.append(sh["moment_elong"])
    res["E1"] = {"k0": float(np.mean(e0)), "k": float(np.mean(e1)), "pass": bool(np.mean(e1) - np.mean(e0) >= 0.15)}
    res["E2"] = {"c0": float(np.mean(e2)), "pass": bool(abs(np.mean(e2) - np.mean(e0)) <= 0.05)}
    res["R"] = {"s_per_gastruloid": sec, "pass": bool(sec <= 20)}
    res["all_pass"] = all(res[k]["pass"] for k in ("P1", "P2", "E1", "E2", "R"))
    return res


def gastruloid(parts, n, rng):
    counts = [int(round(n * f)) for _, f, _, _ in parts]
    counts[-1] = n - sum(counts[:-1])
    s, c, lab = [], [], []
    for (name, _, z, comp), k in zip(parts, counts):
        mu = 0.0 if z is None else float(cm.state_copy(np.array([z]), c2.RHO_G, rng)[0])
        s.append(mu + rng.standard_normal(k))
        c.append(np.full(k, comp))
        lab += [name] * k
    return np.concatenate(s), np.concatenate(c), np.array(lab)


def checks(variant, setting, seed=641):
    rng = np.random.default_rng(seed)
    prm = params(variant, setting, K_EXT[variant])
    n = 300
    out = {"variant": variant, "setting": setting, "params": prm}
    sim, fit = {}, {}
    for k, z in c2.Z.items():
        vals = []
        for _ in range(40):
            s, c, lab = gastruloid([(k, 0.5, z, 1.0), ("bulk", 0.5, None, 1.0)], n, rng)
            x, P, pr, sh, m = run(variant, s, c, prm, rng)
            vals.append(c2.polarity_of(c2.profile(c2.axis_coords(x, P), lab[:m] == k)))
        sim[k], fit[k] = float(np.mean(vals)), c2.C3S1["fit_shard"][k]["mean"]
    ks = sorted(sim)
    signs = int(sum(np.sign(sim[k]) == np.sign(fit[k]) for k in ks))
    rho = float(spearmanr([sim[k] for k in ks], [fit[k] for k in ks]).correlation)
    out["Q1"] = {"sim": sim, "fit": fit, "sign_matches": signs, "spearman": rho, "pass": bool(signs == 5 and rho >= 0.9)}
    xs = (np.arange(N_BINS) + 0.5) * 100 / N_BINS
    cr = []
    for f in (0.2, 1 / 3, 0.5, 2 / 3, 0.8):
        red, green = [], []
        for _ in range(20):
            s, c, lab = gastruloid([("3", f, c2.Z["3"], 1.0), ("1", 1 - f, c2.Z["1"], 1.0)], n, rng)
            x, P, pr, sh, m = run(variant, s, c, prm, rng)
            t = c2.axis_coords(x, P)
            red.append(c2.profile(t, lab[:m] == "3"))
            green.append(c2.profile(t, lab[:m] == "1"))
        cr.append({"f_anterior": f, "crossing": float(c2.crossing(xs, np.mean(red, 0), np.mean(green, 0)))})
    ok = [o for o in cr if not np.isnan(o["crossing"])]
    b = float(np.polyfit([100 * o["f_anterior"] for o in ok], [o["crossing"] for o in ok], 1)[0]) if len(ok) >= 3 else float("nan")
    lo, hi = c2.G1B["slope_ci"]
    out["Q2"] = {"crossings": cr, "slope": b, "g1b_ci": [lo, hi], "pass": bool(lo <= b <= hi)}
    q3 = {}
    for f in (0.0, 0.2, 0.5, 0.8, 1.0):
        vals = []
        for _ in range(20):
            parts = ([("bulk", 1.0, None, 1.0)] if f == 0 else [("low", 1.0, 0.0, 0.2)] if f == 1
                     else [("low", f, 0.0, 0.2), ("bulk", 1 - f, None, 1.0)])
            s, c, lab = gastruloid(parts, n, rng)
            vals.append(run(variant, s, c, prm, rng)[3])
        q3[str(f)] = {m_: float(np.mean([v[m_] for v in vals])) for m_ in vals[0]}
    y = [q3[str(f)]["moment_elong"] for f in (0.0, 0.2, 0.5, 0.8, 1.0)]
    out["Q3"] = {"by_fraction": q3, "monotone_fall_moment": bool(all(a >= b_ for a, b_ in zip(y, y[1:])))}
    return out


if __name__ == "__main__":
    mode, variant = sys.argv[1], sys.argv[2]
    setting = sys.argv[3] if len(sys.argv) > 3 else "1"
    t0 = time.time()
    r = smoke(variant, setting) if mode == "smoke" else checks(variant, setting)
    r["minutes"] = (time.time() - t0) / 60
    path = os.path.join(OUT, f"c6s3_{mode}_{variant}{setting}.json")
    json.dump(r, open(path, "w"), indent=1)
    print(json.dumps({k: v for k, v in r.items() if k not in ("params",)}, indent=1))
