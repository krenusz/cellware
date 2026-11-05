"""E50: E48 (lineage-matched selection, gamma <= 0.6) plus a class of neutrophil-committed clones
that grow only under G-CSF (B34: G-private clones are neutrophil-dominated at every size). See the
E50 pre-registration in EXPERIMENT_LOG.md.

A clone is in class C with probability phi_C, with a probability that rises with its measured
day-2 neutrophil priming: logit P(C) = logit(phi_C) + b_C * z (z is the clone's latent fate state,
built from the measured score m as in E33). A class-C clone has no founders in the full, M, Epo or
SCF wells; in the G well it draws Poisson(nu) founders, expands with mu_G + a + d_C and makes
neutrophils only. Other clones are simulated as in E46/E48.

    uv run python experiments/e50_gneu.py seed
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
    lc = np.log(prm["phi_C"] / (1 - prm["phi_C"])) + prm["b_C"] * z
    cls = rng.random(n) < 1 / (1 + np.exp(-lc))
    out = {}
    cl = np.asarray(clones)
    for c in CONDS:
        full = c == "full"
        rate = prm["nu_full"] * np.where(u, 1.0, eps) if full else prm["nu"] * np.where(u, eps, 1.0)
        k = rng.poisson(rate)
        zb = np.sqrt(RHO_F) * z + np.sqrt(1 - RHO_F) * rng.standard_normal(n)
        p = 1 / (1 + np.exp(-(prm["beta"] * zb + prm["tau"][c])))
        sel = prm["gamma"] * SIGN.get(c, 0.0) * zb
        lam = k * np.exp(np.clip(prm["mu"][c] + (a_full if full else a) + sel, -30, 8))
        kn = rng.poisson(np.clip(lam * (free * p + cn) * prm["s"][c], 0, 1e4))
        km = rng.poisson(np.clip(lam * (free * (1 - p) + cm), 0, 1e4))
        if c == "G":  # class C: own founders, neutrophils only
            kc = rng.poisson(prm["nu"], n)
            lcz = kc * np.exp(np.clip(prm["mu"]["G"] + a + prm["d_C"], -30, 8))
            kn = np.where(cls, rng.poisson(np.clip(lcz * prm["s"]["G"], 0, 1e4)), kn)
            km = np.where(cls, 0, km)
        else:
            kn = np.where(cls, 0, kn)
            km = np.where(cls, 0, km)
        keep = np.flatnonzero(kn + km > 0)
        ids, xn, xm = cl[keep].tolist(), kn[keep].tolist(), km[keep].tolist()
        out[c] = {4: {k: [x, y, 0] for k, x, y in zip(ids, xn, xm)}}
    return out


def unpack(x):
    prm = e37.unpack(x[:23])
    prm["gamma"] = float(GAMMA_MAX / (1 + np.exp(-x[23])))
    prm["phi_C"] = float(0.001 + 0.299 / (1 + np.exp(-x[24])))  # 0.001..0.3
    prm["b_C"] = float(3 / (1 + np.exp(-x[25])))  # 0..3, more likely for neutrophil-primed
    prm["d_C"] = float(4 * np.tanh(x[26]))  # log-expansion offset, -4..4
    return prm


if __name__ == "__main__":
    e34.e33.unpack_base = e33.unpack  # e36.unpack_signed builds on the E33 parameters
    e33.simulate, e33.unpack = simulate, unpack
    e33.TARGETS, e33.EXTRA_SE, e33.SIMS = e36.targets, e36.extra_se, 10
    e33.main(int(sys.argv[1]), maxiter=500, dim=27, name="e50")
