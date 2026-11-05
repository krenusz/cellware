"""R1: can KO instruction (tau) and KO selection (gamma) be recovered from SIS-skew statistics?

Simulator of SIS-skew clones (see EXPERIMENT_LOG R1). Steps:
    uv run python r1_skew_sim.py calibrate   # nuisance values matched to fit-set marginals
    uv run python r1_skew_sim.py recover     # 4,000 train / 300 test sims, ridge, readings
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

import sis_data as sd
from b39_sisskew import GROUPS, SIZE_BINS

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "results_sis")
SIGMA_Z = np.array([1.5, 2.0, 2.7])  # B38 clone-level logit SDs (cDC1, cDC2, pDC)
N_NULL, N_KO = 454, 831  # fit-set clone counts (B39)
PDC = 2

NUIS = {  # name: (low, high) for the calibration search
    "mu_cDC2_W": (-2.0, 2.0), "mu_pDC_W": (-4.5, 1.5),  # W (HSPC) and C (LSK) differ in
    "mu_cDC2_C": (-2.0, 3.5), "mu_pDC_C": (-4.5, 1.5),  # starting population: own fate mix
    "c0": (0.5, 6.0), "t": (0.1, 0.6),
    "m_g": (0.0, 4.0), "s_g": (0.2, 2.0), "s_line": (0.2, 2.0), "s_eps": (0.1, 3.0),
    "lam_L": (-2.5, 2.5), "lam_K": (-1.0, 4.0), "delta": (-1.0, 1.0),
    "tau": (-1.5, 1.5), "gamma": (-1.0, 1.0),
}


def simulate(p, n, ko, rng, delta=None):
    """Return P (+ arm cells x 3), N (- arm cells x 3), keeping clones with > 1 cell in total."""
    delta = p["delta"] if delta is None else delta
    w = "C" if ko else "W"
    mu = np.array([0.0, p[f"mu_cDC2_{w}"], p[f"mu_pDC_{w}"]])
    m = int(n * 1.6) + 50
    z = mu + rng.standard_normal((m, 3)) * SIGMA_Z
    zs = (z[:, PDC] - z[:, PDC].mean()) / z[:, PDC].std()
    c = 1 + rng.poisson(p["c0"], m)
    b = rng.binomial(c, p["t"])
    g = p["m_g"] + p["s_g"] * rng.standard_normal(m)
    lam_p = p["lam_L"] + (p["lam_K"] + p["gamma"] * zs if ko else 0.0)
    cmax = c.max()
    line = np.exp(g[:, None] + p["s_line"] * rng.standard_normal((m, cmax)))
    idx = np.arange(cmax)[None, :]
    in_p = idx < b[:, None]
    in_n = (idx >= b[:, None]) & (idx < c[:, None])
    mean_p = (line * in_p).sum(1) * np.exp(lam_p)
    mean_n = (line * in_n).sum(1)
    n_p, n_n = rng.poisson(mean_p), rng.poisson(mean_n)
    shift = np.zeros(3)
    shift[PDC] = delta + (p["tau"] if ko else 0.0)
    lp = z + shift + p["s_eps"] * rng.standard_normal((m, 3))
    ln = z + p["s_eps"] * rng.standard_normal((m, 3))
    sm = lambda x: np.exp(x - x.max(1, keepdims=True)) / np.exp(x - x.max(1, keepdims=True)).sum(1, keepdims=True)  # noqa: E731
    P = rng.multinomial(n_p, sm(lp)).astype(float)
    N = rng.multinomial(n_n, sm(ln)).astype(float)
    keep = np.flatnonzero((P.sum(1) + N.sum(1)) > 1)[:n]
    return P[keep], N[keep]


# ------------------------------------------------------------------ summary statistics

def _rank(x):
    r = np.empty(len(x))
    r[np.argsort(x, kind="mergesort")] = np.arange(len(x))
    return r


def _rho(x, y):
    if len(x) < 5 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(_rank(x), _rank(y))[0, 1])


def stats(P, N):
    """Summary vector (fixed order): pooled arm fractions (6), B39 decomposition (9), size-ratio rho
    (3), one-arm shares by size (4), +-only share of one-arm by size (4), B40 C rhos (6),
    log cell ratio + / - (1), mean sister TV of both-arm clones (1)."""
    s = []
    s += list(P.sum(0) / max(P.sum(), 1)) + list(N.sum(0) / max(N.sum(), 1))
    both = (P.sum(1) > 3) & (N.sum(1) > 3)
    Pb, Nb = P[both], N[both]
    if both.sum() >= 5:
        wp, wn = Pb.sum(1) / Pb.sum(), Nb.sum(1) / Nb.sum()
        pp, pn = sd._frac(Pb), sd._frac(Nb)
        s += list((wn[:, None] * (pp - pn)).sum(0)) + list(((wp - wn)[:, None] * pn).sum(0)) \
            + list(((wp - wn)[:, None] * (pp - pn)).sum(0))
        m = Nb.sum(1) > 10
        lr = np.log(Pb[m].sum(1) / Nb[m].sum(1))
        s += [_rho(lr, sd._frac(Nb[m])[:, j]) for j in range(3)]
        tvb = float(0.5 * np.abs(pp - pn).sum(1).mean())
    else:
        s += [0.0] * 12
        tvb = 0.0
    tot = P.sum(1) + N.sum(1)
    pres_p, pres_n = P.sum(1) > 1, N.sum(1) > 1
    one = pres_p ^ pres_n
    for lo, hi in SIZE_BINS:
        mm = (tot >= lo) & (tot <= hi)
        s.append(float(one[mm].mean()) if mm.any() else 0.0)
    for lo, hi in SIZE_BINS:
        mm = (tot >= lo) & (tot <= hi) & one
        s.append(float((pres_p & ~pres_n)[mm].mean()) if mm.any() else 0.5)
    for src, oth in ((N, P), (P, N)):
        m = src.sum(1) > 10
        f = sd._frac(src[m]) if m.any() else np.zeros((0, 3))
        ab = (oth[m].sum(1) <= 1).astype(float)
        s += [_rho(f[:, j], ab) for j in range(3)]
    s.append(float(np.log(max(P.sum(), 1) / max(N.sum(), 1))))
    s.append(tvb)
    return np.array(s)


def size_bin_shares(P, N):
    tot = P.sum(1) + N.sum(1)
    return np.array([np.mean((tot >= lo) & (tot <= hi)) for lo, hi in SIZE_BINS])


def target_vector(P, N):
    return np.concatenate([stats(P, N), size_bin_shares(P, N)])


def both_groups(p, rng, delta_ko=None):
    Pn, Nn = simulate(p, N_NULL, False, rng)
    Pk, Nk = simulate(p, N_KO, True, rng, delta=delta_ko)
    return Pn, Nn, Pk, Nk


# ------------------------------------------------------------------ calibration

CAL_IDX = list(range(0, 6)) + list(range(18, 22)) + list(range(22, 26)) + [32, 33] + list(range(34, 38))


def observed():
    d = sd.load_skew("fit")
    out = {}
    for g, wells in GROUPS.items():
        m = np.isin(d["well"], wells)
        out[g] = target_vector(d["P"][m], d["N"][m])
    return out


def calibrate(n_draws=6000, n_local=6000, seed=11):
    rng = np.random.default_rng(seed)
    obs = observed()
    tgt = np.concatenate([obs["null"][CAL_IDX], obs["KO"][CAL_IDX]])
    scale = np.where(np.arange(len(tgt)) % len(CAL_IDX) == 14, 0.3, 0.1)  # log ratio on its own scale

    def loss(p):
        Pn, Nn, Pk, Nk = both_groups(p, rng)
        v = np.concatenate([target_vector(Pn, Nn)[CAL_IDX], target_vector(Pk, Nk)[CAL_IDX]])
        return float((((v - tgt) / scale) ** 2).sum())

    draw = lambda: {k: rng.uniform(*r) for k, r in NUIS.items()}  # noqa: E731
    best, bl = None, np.inf
    for _ in range(n_draws):
        p = draw()
        lval = loss(p)
        if lval < bl:
            best, bl = p, lval
    for i in range(n_local):
        w = 0.15 * (1 - i / n_local) + 0.02
        p = {k: float(np.clip(best[k] + w * (r[1] - r[0]) * rng.standard_normal(), *r))
             for k, r in NUIS.items()}
        lval = np.mean([loss(p) for _ in range(2)])
        if lval < bl:
            best, bl = p, lval
    Pn, Nn, Pk, Nk = both_groups(best, rng)
    res = {"params": best, "loss": bl,
           "check": {"null_obs": obs["null"][CAL_IDX].round(3).tolist(),
                     "null_sim": target_vector(Pn, Nn)[CAL_IDX].round(3).tolist(),
                     "KO_obs": obs["KO"][CAL_IDX].round(3).tolist(),
                     "KO_sim": target_vector(Pk, Nk)[CAL_IDX].round(3).tolist()},
           "cal_index_meaning": "arm fractions +3/-3, one-arm by size 4, +-only of one-arm by size 4, "
                                "log cell ratio, sister TV, size-bin shares 4"}
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "r1_calibration.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


# ------------------------------------------------------------------ recovery

def recover(n_train=4000, n_test=300, alpha=1.0, seed=12):
    rng = np.random.default_rng(seed)
    base = json.load(open(os.path.join(OUT, "r1_calibration.json")))["params"]
    pri = {"tau": (-1.5, 1.5), "gamma": (-1.0, 1.0), "delta": (-1.0, 1.0)}

    def one(dshift=0.0):
        th = {k: rng.uniform(*r) for k, r in pri.items()}
        p = {**base, **th}
        Pn, Nn, Pk, Nk = both_groups(p, rng, delta_ko=th["delta"] + dshift)
        return np.concatenate([stats(Pn, Nn), stats(Pk, Nk)]), np.array([th["tau"], th["gamma"], th["delta"]])

    tr = [one() for _ in range(n_train)]
    te = [one() for _ in range(n_test)]
    te_mis = [one(0.5) for _ in range(n_test)]
    X, Y = np.array([a for a, _ in tr]), np.array([b for _, b in tr])
    mu_x, sd_x = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu_x) / sd_x
    W = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (Y - Y.mean(0)))

    def pred(rows):
        Xt = (np.array([a for a, _ in rows]) - mu_x) / sd_x
        return Xt @ W + Y.mean(0), np.array([b for _, b in rows])

    Yh, Yt = pred(te)
    names = ["tau", "gamma", "delta"]
    res = {"base_params": base, "alpha": alpha, "n_train": n_train, "n_test": n_test}
    for j, nm in enumerate(names):
        err = Yh[:, j] - Yt[:, j]
        res[nm] = {"R2": float(1 - np.mean(err ** 2) / np.var(Yt[:, j])), "RMSE": float(np.sqrt(np.mean(err ** 2))),
                   "err_corr_with_true": {o: float(np.corrcoef(err, Yt[:, k])[0, 1]) for k, o in enumerate(names) if k != j}}
    Yhm, Ytm = pred(te_mis)
    res["misspecified_label_shift_0.5"] = {nm: {"bias": float(np.mean(Yhm[:, j] - Ytm[:, j])),
                                               "RMSE": float(np.sqrt(np.mean((Yhm[:, j] - Ytm[:, j]) ** 2)))}
                                          for j, nm in enumerate(names)}
    json.dump(res, open(os.path.join(OUT, "r1_recovery.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calibrate", "recover"])
    a = ap.parse_args()
    calibrate() if a.cmd == "calibrate" else recover()
