"""C6-S2: 2-D cell-level gastruloid simulator on the core model. See EXPERIMENT_LOG C6-S2 (pre-registered).

    uv run --with scikit-image python c6s2_space.py smoke     # S-a..S-d (synthetic)
    uv run --with scikit-image python c6s2_space.py checks    # Q1-Q3 (open/spent summaries only)

Cells are soft discs (radius 1) with overdamped dynamics and noise. Each cell carries its clone's state copy
z' (core_model.state_copy per gastruloid, rho_g from C3-S1) plus cell noise -> A-P score s (high = anterior),
and a clone competence c. Local rules:
  1 differential adhesion: attraction between neighbours scales with exp(-|s_i - s_j| / w)
  2 proportion feedback: posterior fraction q = (1 - g) * mean(sigmoid(-s)) + g * a*; the q lowest-s cells are P
  3 posterior-driven extension: P-P neighbour pairs are pushed apart along the A->P axis and drawn together across
    it, with strength k * c * coherence (A-P centroid separation / body radius)
No rescue term. No sealed file is read.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

import core_model as cm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_gastruloid")
C3S1 = json.load(open(os.path.join(OUT, "c3s1_core_gastruloid.json")))
G1B = json.load(open(os.path.join(OUT, "g1b_mixing.json")))
Z = {k: float(v) for k, v in C3S1["states_z"].items()}
RHO_G = float(C3S1["rho_g"])
G_FB, A_STAR = 0.834, 0.377  # G5 proportion feedback
N_BINS = 30

DEFAULT = dict(n=300, steps=800, dt=0.05, w=0.5, k_rep=4.0, k_adh=1.0, r_adh=2.6, noise=0.15,
               k_ext=0.6, g=G_FB, a_star=A_STAR)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def identities(s, g, a_star):
    """Rule 2: posterior fraction from proportion feedback; P = the q lowest-s cells."""
    q = (1 - g) * float(np.mean(sigmoid(-s))) + g * a_star
    n_p = int(round(q * len(s)))
    P = np.zeros(len(s), bool)
    if n_p > 0:
        P[np.argsort(s)[:n_p]] = True
    return P


def simulate(s, c, prm, rng, return_traj=False):
    """s: (n,) A-P scores; c: (n,) competences. Returns positions (n, 2) and posterior mask."""
    n = len(s)
    P = identities(s, prm["g"], prm["a_star"])
    R0 = np.sqrt(n) * 1.05
    r = R0 * np.sqrt(rng.random(n))
    th = 2 * np.pi * rng.random(n)
    x = np.c_[r * np.cos(th), r * np.sin(th)]
    for _ in range(prm["steps"]):
        pairs = cKDTree(x).query_pairs(prm["r_adh"], output_type="ndarray")
        i, j = pairs[:, 0], pairs[:, 1]
        d = x[j] - x[i]
        dist = np.linalg.norm(d, axis=1) + 1e-9
        u = d / dist[:, None]
        # repulsion inside contact, adhesion spring out to r_adh (rule 1)
        a = np.exp(-np.abs(s[i] - s[j]) / prm["w"])
        f = np.where(dist < 2.0, -prm["k_rep"] * (2.0 - dist), prm["k_adh"] * a * (dist - 2.0))
        F = np.zeros_like(x)
        np.add.at(F, i, f[:, None] * u)
        np.add.at(F, j, -f[:, None] * u)
        # rule 3: posterior-driven extension along the A->P axis
        if prm["k_ext"] > 0 and P.any() and (~P).any():
            ca, cp = x[~P].mean(0), x[P].mean(0)
            ax = cp - ca
            rad = np.sqrt(np.mean(np.sum((x - x.mean(0)) ** 2, 1)))
            coh = min(1.0, np.linalg.norm(ax) / (rad + 1e-9))
            ax = ax / (np.linalg.norm(ax) + 1e-9)
            perp = np.array([-ax[1], ax[0]])
            pp = P[i] & P[j]
            if pp.any():
                ip, jp, dp = i[pp], j[pp], d[pp]
                str_ = prm["k_ext"] * coh * 0.5 * (c[ip] + c[jp])
                along = np.sign(dp @ ax)[:, None] * ax[None, :]
                across = (dp @ perp)[:, None] * perp[None, :] / (np.abs(dp @ perp)[:, None] + 1e-9)
                g_ = str_[:, None] * (along - 0.5 * across)
                np.add.at(F, ip, -g_)
                np.add.at(F, jp, g_)
        x = x + prm["dt"] * F + np.sqrt(prm["dt"]) * prm["noise"] * rng.standard_normal(x.shape)
    return x, P


# ---------------- readouts ----------------

def axis_coords(x, P):
    """Long axis by PCA; oriented so posterior cells sit at the high end. Returns t in [0, 100] (anterior = 0)."""
    xc = x - x.mean(0)
    w, v = np.linalg.eigh(np.cov(xc.T))
    t = xc @ v[:, -1]
    if P.any() and (~P).any() and t[P].mean() < t[~P].mean():
        t = -t
    return 100 * (t - t.min()) / (t.max() - t.min() + 1e-9)


def profile(t, mask):
    b = np.minimum((t / 100 * N_BINS).astype(int), N_BINS - 1)
    dens = np.bincount(b[mask], minlength=N_BINS).astype(float)
    sd = dens.std()
    return (dens - dens.mean()) / sd if sd > 0 else dens * 0


def polarity_of(z):
    return z[: N_BINS // 2].mean() - z[N_BINS // 2:].mean()


def crossing(xs, red, green):
    d = red - green
    for k in range(len(d) - 1):
        if d[k] == 0:
            return xs[k]
        if d[k] * d[k + 1] < 0:
            return xs[k] + (xs[k + 1] - xs[k]) * d[k] / (d[k] - d[k + 1])
    return np.nan


def shape(x, px=0.5):
    """Rasterise discs; return 1 - circularity, geodesic aspect (skeleton length^2 / area), moment elongation."""
    from scipy import ndimage
    from skimage.measure import perimeter
    from skimage.morphology import skeletonize
    lo = x.min(0) - 2
    sz = np.ceil((x.max(0) + 2 - lo) / px).astype(int) + 1
    img = np.zeros(sz[::-1], bool)
    yy, xx = np.mgrid[-2:3, -2:3]
    disc = (xx ** 2 + yy ** 2) * px * px <= 1.15 ** 2
    for p in ((x - lo) / px).astype(int):
        ys, xs_ = yy[disc] + p[1], xx[disc] + p[0]
        img[np.clip(ys, 0, sz[1] - 1), np.clip(xs_, 0, sz[0] - 1)] = True
    img = ndimage.binary_fill_holes(ndimage.binary_closing(img, iterations=2))
    lab, nl = ndimage.label(img)
    if nl > 1:
        img = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    A = img.sum() * px * px
    Pm = perimeter(img) * px
    circ = 4 * np.pi * A / (Pm ** 2 + 1e-9)
    L = skeletonize(img).sum() * px
    pts = np.argwhere(img).astype(float)
    ev = np.linalg.eigvalsh(np.cov(pts.T))
    return {"one_minus_circ": float(1 - circ), "geo_aspect": float(L ** 2 / A), "moment_elong": float(1 - np.sqrt(ev[0] / ev[1]))}


def homotypic(x, P, r=2.4):
    """Fraction of neighbour contacts between cells of the same identity (random mixing ~0.52 at q 0.4)."""
    pr = cKDTree(x).query_pairs(r, output_type="ndarray")
    return float(np.mean(P[pr[:, 0]] == P[pr[:, 1]]))


def sorting_index(x, P):
    if not P.any() or P.all():
        return 0.0
    rad = np.sqrt(np.mean(np.sum((x - x.mean(0)) ** 2, 1)))
    return float(np.linalg.norm(x[P].mean(0) - x[~P].mean(0)) / rad)


# ---------------- gastruloid builders ----------------

def gastruloid(parts, prm, rng):
    """parts: list of (label, frac, z or None for bulk, competence). Returns s, c, labels."""
    n = prm["n"]
    counts = [int(round(n * f)) for _, f, _, _ in parts]
    counts[-1] = n - sum(counts[:-1])
    s, c, lab = [], [], []
    for (name, _, z, comp), k in zip(parts, counts):
        mu = 0.0 if z is None else float(cm.state_copy(np.array([z]), RHO_G, rng)[0])
        s.append(mu + rng.standard_normal(k))
        c.append(np.full(k, comp))
        lab += [name] * k
    return np.concatenate(s), np.concatenate(c), np.array(lab)


# ---------------- smoke tests ----------------

def smoke():
    res = {}
    rng = np.random.default_rng(601)
    # S-a: sorting rises with adhesion contrast (smaller w); synthetic 50:50, states +-2
    def mix(rng):
        return gastruloid([("A", 0.5, None, 1.0), ("B", 0.5, None, 1.0)], DEFAULT, rng)
    sa = {}
    for w in (5.0, 0.5):
        vals = []
        for _ in range(6):
            s = np.r_[2 + rng.standard_normal(150), -2 + rng.standard_normal(150)]
            x, P = simulate(s, np.ones(300), {**DEFAULT, "w": w, "k_ext": 0.0}, rng)
            vals.append((sorting_index(x, P), homotypic(x, P)))
        sa[w] = np.mean(vals, 0).tolist()
    res["S_a_sorting"] = {"w5_index_homotypic": sa[5.0], "w0.5_index_homotypic": sa[0.5], "pass": sa[0.5][0] > sa[5.0][0]}
    # S-b: elongation rises with k for a sorted 50:50 mix
    sb = {}
    for k in (0.0, 0.6, 1.2):
        vals = []
        for _ in range(6):
            s = np.r_[2 + rng.standard_normal(150), -2 + rng.standard_normal(150)]
            x, P = simulate(s, np.ones(300), {**DEFAULT, "k_ext": k}, rng)
            vals.append(shape(x)["one_minus_circ"])
        sb[k] = float(np.mean(vals))
    res["S_b_elong_vs_k"] = {**{str(k): v for k, v in sb.items()}, "pass": sb[0.0] < sb[0.6] < sb[1.2]}
    # S-c: c = 0 stays round (compare with k = 0 baseline)
    vals = []
    for _ in range(6):
        s = np.r_[2 + rng.standard_normal(150), -2 + rng.standard_normal(150)]
        x, P = simulate(s, np.zeros(300), DEFAULT, rng)
        vals.append(shape(x)["one_minus_circ"])
    res["S_c_c0"] = {"one_minus_circ": float(np.mean(vals)), "k0_baseline": sb[0.0],
                     "pass": abs(float(np.mean(vals)) - sb[0.0]) < 0.5 * (sb[1.2] - sb[0.0])}
    # S-d: run time
    t0 = time.time()
    s = rng.standard_normal(300)
    simulate(s, np.ones(300), DEFAULT, rng)
    dt = time.time() - t0
    res["S_d_seconds"] = {"s": dt, "pass": dt < 5}
    return res


# ---------------- data checks ----------------

def q1(reps=100, seed=611):
    rng = np.random.default_rng(seed)
    sim, fit = {}, {}
    for k, z in Z.items():
        P_ = []
        for _ in range(reps):
            s, c, lab = gastruloid([(k, 0.5, z, 1.0), ("bulk", 0.5, None, 1.0)], DEFAULT, rng)
            x, P = simulate(s, c, DEFAULT, rng)
            P_.append(polarity_of(profile(axis_coords(x, P), lab == k)))
        sim[k] = float(np.mean(P_))
        fit[k] = C3S1["fit_shard"][k]["mean"]
    ks = sorted(Z)
    signs = sum(np.sign(sim[k]) == np.sign(fit[k]) for k in ks)
    rho = spearmanr([sim[k] for k in ks], [fit[k] for k in ks]).correlation
    return {"sim": sim, "fit": fit, "sign_matches": int(signs), "spearman": float(rho),
            "pass": bool(signs == 5 and rho >= 0.9)}


def q2(reps=40, seed=612):
    rng = np.random.default_rng(seed)
    out = []
    xs = (np.arange(N_BINS) + 0.5) * 100 / N_BINS
    for f_ant in (0.2, 1 / 3, 0.5, 2 / 3, 0.8):  # clone-3 (anterior) fraction
        red, green = [], []
        for _ in range(reps):
            s, c, lab = gastruloid([("3", f_ant, Z["3"], 1.0), ("1", 1 - f_ant, Z["1"], 1.0)], DEFAULT, rng)
            x, P = simulate(s, c, DEFAULT, rng)
            t = axis_coords(x, P)
            red.append(profile(t, lab == "3"))
            green.append(profile(t, lab == "1"))
        cr = crossing(xs, np.mean(red, 0), np.mean(green, 0))
        out.append({"f_anterior": f_ant, "crossing": float(cr)})
    ok = [o for o in out if not np.isnan(o["crossing"])]
    b = float(np.polyfit([100 * o["f_anterior"] for o in ok], [o["crossing"] for o in ok], 1)[0]) if len(ok) >= 3 else float("nan")
    lo, hi = G1B["slope_ci"]
    return {"crossings": out, "slope": b, "g1b_slope": G1B["slope"], "g1b_ci": [lo, hi], "pass": bool(lo <= b <= hi)}


def q3(reps=40, seed=613):
    rng = np.random.default_rng(seed)
    out = {}
    for f in (0.0, 0.2, 0.5, 0.8, 1.0):
        vals = []
        for _ in range(reps):
            parts = [("low", f, 0.0, 0.2), ("bulk", 1 - f, None, 1.0)] if 0 < f < 1 else (
                [("bulk", 1.0, None, 1.0)] if f == 0 else [("low", 1.0, 0.0, 0.2)])
            s, c, lab = gastruloid(parts, DEFAULT, rng)
            x, P = simulate(s, c, DEFAULT, rng)
            vals.append(shape(x))
        out[str(f)] = {m: float(np.mean([v[m] for v in vals])) for m in vals[0]}
        out[str(f)]["sd_one_minus_circ"] = float(np.std([v["one_minus_circ"] for v in vals], ddof=1))
    y = [out[str(f)]["one_minus_circ"] for f in (0.0, 0.2, 0.5, 0.8, 1.0)]
    lin = [y[0] + (y[-1] - y[0]) * f for f in (0.0, 0.2, 0.5, 0.8, 1.0)]
    return {"by_fraction": out, "monotone_fall": bool(all(a >= b for a, b in zip(y, y[1:]))),
            "minus_linear_at_20_50_80": [float(y[i] - lin[i]) for i in (1, 2, 3)]}


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    os.makedirs(OUT, exist_ok=True)
    if mode == "smoke":
        r = smoke()
    else:
        t0 = time.time()
        r = {"params": DEFAULT, "Q1": q1(), "Q2": q2(), "Q3": q3()}
        r["minutes"] = (time.time() - t0) / 60
    path = os.path.join(OUT, f"c6s2_{mode}.json")
    json.dump(r, open(path, "w"), indent=1)
    print(json.dumps(r, indent=1))
