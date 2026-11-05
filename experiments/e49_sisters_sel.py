"""E49: E45 (a limited number of sister cells per clone split between the main-pool wells) plus
E48's lineage-matched selection (gamma <= 0.6). E45 reproduced the presence structure (G-M
co-presence, M-present clones) but lost B7 G; E48 fits B7 G with plausible selection but leaves
small clones too shared (B31, B32). See the E49 pre-registration in EXPERIMENT_LOG.md.

    uv run python experiments/e49_sisters_sel.py seed
"""

from __future__ import annotations

import sys

import e33_measured as e33
import e34_fullarm as e34
import e36_combined as e36
import e37_pool as e37
import e45_sisters as e45
import numpy as np
from e20_gap import RHO_F

CONDS = e33.CONDS
SIGN = {"G": 1.0, "M": -1.0}
GAMMA_MAX = 0.6


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
    fk = e45.founders(prm, u, rng)
    out = {}
    cl = np.asarray(clones)
    for c in CONDS:
        full = c == "full"
        k = rng.poisson(prm["nu_full"] * np.where(u, 1.0, eps)) if full else fk[c]
        zb = np.sqrt(RHO_F) * z + np.sqrt(1 - RHO_F) * rng.standard_normal(n)
        p = 1 / (1 + np.exp(-(prm["beta"] * zb + prm["tau"][c])))
        sel = prm["gamma"] * SIGN.get(c, 0.0) * zb
        lam = k * np.exp(np.clip(prm["mu"][c] + (a_full if full else a) + sel, -30, 8))
        kn = rng.poisson(np.clip(lam * (free * p + cn) * prm["s"][c], 0, 1e4))
        km = rng.poisson(np.clip(lam * (free * (1 - p) + cm), 0, 1e4))
        keep = np.flatnonzero(kn + km > 0)
        ids, xn, xm = cl[keep].tolist(), kn[keep].tolist(), km[keep].tolist()
        out[c] = {4: {k: [x, y, 0] for k, x, y in zip(ids, xn, xm)}}
    return out


def unpack(x):
    prm = e45.unpack(x[:25])
    prm["gamma"] = float(GAMMA_MAX / (1 + np.exp(-x[25])))
    return prm


if __name__ == "__main__":
    e34.e33.unpack_base = e33.unpack  # e36.unpack_signed builds on the E33 parameters
    e33.simulate, e33.unpack = simulate, unpack
    e33.TARGETS, e33.EXTRA_SE, e33.SIMS = e36.targets, e36.extra_se, 10
    e33.main(int(sys.argv[1]), maxiter=500, dim=26, name="e49")
