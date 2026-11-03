"""E34: E33 with the full arm as a separate culture (own founder rate, own expansion trait).
See the 2026-11-03 (9) batch pre-registration in EXPERIMENT_LOG.md.

    uv run python experiments/e34_fullarm.py seed
"""

from __future__ import annotations

import sys

import e33_measured as e33
import numpy as np
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
    out = {}
    for c in CONDS:
        full = c == "full"
        k = rng.poisson(prm["nu_full"] if full else prm["nu"], n)
        zb = np.sqrt(RHO_F) * z + np.sqrt(1 - RHO_F) * rng.standard_normal(n)
        p = 1 / (1 + np.exp(-(prm["beta"] * zb + prm["tau"][c])))
        lam = k * np.exp(np.clip(prm["mu"][c] + (a_full if full else a), -30, 8))
        kn = rng.poisson(np.clip(lam * (free * p + cn) * prm["s"][c], 0, 1e4))
        km = rng.poisson(np.clip(lam * (free * (1 - p) + cm), 0, 1e4))
        keep = np.flatnonzero(kn + km > 0)
        out[c] = {4: {clones[i]: [int(kn[i]), int(km[i]), 0] for i in keep}}
    return out


def unpack(x):
    prm = e33.unpack_base(x[:19])
    prm["nu_full"] = float(5 / (1 + np.exp(-x[19])))
    prm["rho_a"] = float(1 / (1 + np.exp(-x[20])))
    return prm


if __name__ == "__main__":
    e33.unpack_base = e33.unpack
    e33.simulate, e33.unpack = simulate, unpack
    e33.main(int(sys.argv[1]) if len(sys.argv) > 1 else 0, maxiter=500, dim=21, name="e34")
