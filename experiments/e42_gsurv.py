"""E42: E38 (separate clone pool) with clone-level survival under G-CSF that drops strongly
monocyte-biased clones. See the E42 pre-registration in EXPERIMENT_LOG.md.

Each G-CSF founder of a clone survives with q = exp(-h * softplus(z0 - z)): clones with fate
state z well above z0 are untouched, and clones below z0 (monocyte-biased; low z means low
neutrophil probability) are lost at a rate that grows with h. h = 0 recovers E38 exactly.

    uv run python experiments/e42_gsurv.py seed
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


def survival(z, prm):
    return np.exp(-prm["h_G"] * np.logaddexp(0.0, prm["z0_G"] - z))


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
    out = {}
    for c in CONDS:
        full = c == "full"
        rate = prm["nu_full"] * np.where(u, 1.0, eps) if full else prm["nu"] * np.where(u, eps, 1.0)
        if c == "G":
            rate = rate * survival(z, prm)
        k = rng.poisson(rate)
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
    prm["h_G"] = float(6 / (1 + np.exp(-x[23])))  # 0..6
    prm["z0_G"] = float(-3 + 4 / (1 + np.exp(-x[24])))  # -3..1
    return prm


if __name__ == "__main__":
    e34.e33.unpack_base = e33.unpack  # e36.unpack_signed builds on the E33 parameters
    e33.simulate, e33.unpack = simulate, unpack
    e33.TARGETS, e33.EXTRA_SE, e33.SIMS = e36.targets, e36.extra_se, 10
    e33.main(int(sys.argv[1]), maxiter=500, dim=25, name="e42")
