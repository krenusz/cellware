"""E45: E38 (separate clone pool) with a limited number of sister cells per clone split between
the main-pool wells (G, M, Epo, SCF), plus a free M-well weight. See the E45 pre-registration in
EXPERIMENT_LOG.md.

A Poisson number of founders split between wells is the same as independent Poisson founders per
well, so competition between wells needs an under-dispersed sister count. Each clone has
K = 1 + Poisson(lam_K) sisters at the split; each sister goes to G, Epo or SCF with probability
r, to M with probability r * w_M, or is lost. r is set so that the expected founders per well,
(1 + lam_K) * r, equals E38's nu; large lam_K approaches E38's independent founders (w_M = 1).
The full arm keeps E37's separate pool.

    uv run python experiments/e45_sisters.py seed
"""

from __future__ import annotations

import sys

import e33_measured as e33
import e34_fullarm as e34
import e36_combined as e36
import e37_pool as e37
import numpy as np
from e20_gap import RHO_F

CONDS = e33.CONDS
MAIN = ("G", "M", "Epo", "SCF")


def founders(prm, u, rng):
    """Founders per main-pool well: K sisters per clone, split multinomially (with a lost bin)."""
    n = len(u)
    lam_k, w_m = prm["lam_K"], prm["w_M"]
    out = {c: np.zeros(n, int) for c in MAIN}
    for grp, scale in ((~u, 1.0), (u, prm["eps"])):
        idx = np.flatnonzero(grp)
        if not len(idx):
            continue
        r = prm["nu"] * scale / (1 + lam_k)
        r = min(r, 0.999 / (3 + w_m))  # probabilities must leave room for the lost bin
        p = np.array([r, r * w_m, r, r, 0.0])
        p[-1] = 1 - p[:-1].sum()
        k = 1 + rng.poisson(lam_k, len(idx))
        alloc = rng.multinomial(k, p)
        for j, c in enumerate(MAIN):
            out[c][idx] = alloc[:, j]
    return out


def simulate(clones, prm, rng, meas):
    n = len(clones)
    m, fn, fm = meas[:, 0], meas[:, 1], meas[:, 2]
    w = prm["w"]
    z = w * m + np.sqrt(1 - w**2) * rng.standard_normal(n)
    a = prm["sigma_a"] * rng.standard_normal(n)
    ra = prm["rho_a"]
    a_full = ra * a + np.sqrt(1 - ra**2) * prm["sigma_a"] * rng.standard_normal(n)
    cn = np.clip(prm["kappa"] * fn, 0, 1)
    cm = np.clip(prm["kappa"] * fm, 0, 1 - cn)
    free = 1 - cn - cm
    u = rng.random(n) < prm["phi"]
    eps = prm["eps"]
    fk = founders(prm, u, rng)
    out = {}
    for c in CONDS:
        full = c == "full"
        k = rng.poisson(prm["nu_full"] * np.where(u, 1.0, eps)) if full else fk[c]
        zb = np.sqrt(RHO_F) * z + np.sqrt(1 - RHO_F) * rng.standard_normal(n)
        p = 1 / (1 + np.exp(-(prm["beta"] * zb + prm["tau"][c])))
        lam = k * np.exp(np.clip(prm["mu"][c] + (a_full if full else a), -30, 8))
        kn = rng.poisson(np.clip(lam * (free * p + cn) * prm["s"][c], 0, 1e4))
        km = rng.poisson(np.clip(lam * (free * (1 - p) + cm), 0, 1e4))
        keep = np.flatnonzero(kn + km > 0)
        out[c] = {4: {clones[i]: [int(kn[i]), int(km[i]), 0] for i in keep}}
    return out


def unpack(x):
    prm = e37.unpack(x[:23])
    prm["lam_K"] = float(20 / (1 + np.exp(-x[23])))  # 0..20 extra sisters
    prm["w_M"] = float(np.exp(np.log(4) * np.tanh(x[24])))  # 0.25..4
    return prm


if __name__ == "__main__":
    e34.e33.unpack_base = e33.unpack  # e36.unpack_signed builds on the E33 parameters
    e33.simulate, e33.unpack = simulate, unpack
    e33.TARGETS, e33.EXTRA_SE, e33.SIMS = e36.targets, e36.extra_se, 10
    e33.main(int(sys.argv[1]), maxiter=500, dim=25, name="e45")
