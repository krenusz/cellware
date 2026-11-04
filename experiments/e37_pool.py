"""E37: E36 (signed) with the full arm drawn from a separate split of the clone pool.
See the E37 pre-registration in EXPERIMENT_LOG.md.

    uv run python experiments/e37_pool.py seed [warm_source]
"""

from __future__ import annotations

import json
import os
import sys

import e33_measured as e33
import e34_fullarm as e34
import e36_combined as e36
import numpy as np
from clone_state_model import OUT
from e20_gap import RHO_F

CONDS = e33.CONDS


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
    prm = e36.unpack_signed(x[:21])
    prm["phi"] = float(1 / (1 + np.exp(-x[21])))
    prm["eps"] = float(1 / (1 + np.exp(-x[22])))
    return prm


def warm_x(src):
    """An E36 fit mapped into the 23-dim space, with phi 0.5 and eps at the upper clip."""
    p = json.load(open(os.path.join(OUT, f"{src}.json")))["params"]
    x = list(e36.warm_x(src, True))
    x[19] = e36.logit(p["nu_full"] / 5)
    x[20] = e36.logit(p["rho_a"])
    return np.clip([*x, 0.0, 3.0], -3, 3)


if __name__ == "__main__":
    seed = int(sys.argv[1])
    e34.e33.unpack_base = e33.unpack  # e36.unpack_signed builds on the E33 parameters
    e33.simulate, e33.unpack = simulate, unpack
    e33.TARGETS, e33.EXTRA_SE, e33.SIMS = e36.targets, e36.extra_se, 10
    src = sys.argv[2] if len(sys.argv) > 2 else "e36s_s2"
    x0 = warm_x(src) if seed < 3 else None
    e33.main(seed, maxiter=500, dim=23, name="e37", x0=x0)
