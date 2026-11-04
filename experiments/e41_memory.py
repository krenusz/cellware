"""E41: E38 (separate clone pool) with day-resolved sampling and fading fate memory.
See the E41 pre-registration in EXPERIMENT_LOG.md.

    uv run python experiments/e41_memory.py seed [warm_source]
"""

from __future__ import annotations

import json
import os
import sys

import e33_measured as e33
import e34_fullarm as e34
import e36_combined as e36
import e37_pool as e37
import numpy as np
from clone_state_model import OUT
from e20_gap import RHO_F
from larry_stats import test_stats

CONDS = e33.CONDS
# Fit-set day-4 share of mature N+M cells (E41 pre-registration).
DAY4 = {"full": 0.041, "G": 0.504, "M": 0.155, "Epo": 0.938, "SCF": 1.0}
R46_OBS_SE = 0.16


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
    eps, r46 = prm["eps"], prm["r46"]
    out = {}
    for c in CONDS:
        full = c == "full"
        rate = prm["nu_full"] * np.where(u, 1.0, eps) if full else prm["nu"] * np.where(u, eps, 1.0)
        k = rng.poisson(rate)
        zb = np.sqrt(RHO_F) * z + np.sqrt(1 - RHO_F) * rng.standard_normal(n)
        lam = k * np.exp(np.clip(prm["mu"][c] + (a_full if full else a), -30, 8))
        out[c] = {}
        for day, share in ((4, DAY4[c]), (6, 1 - DAY4[c])):
            zd = zb if day == 4 else r46 * zb + np.sqrt(1 - r46**2) * rng.standard_normal(n)
            p = 1 / (1 + np.exp(-(prm["beta"] * zd + prm["tau"][c])))
            ld = lam * share
            kn = rng.poisson(np.clip(ld * (free * p + cn) * prm["s"][c], 0, 1e4))
            km = rng.poisson(np.clip(ld * (free * (1 - p) + cm), 0, 1e4))
            keep = np.flatnonzero(kn + km > 0)
            out[c][day] = {clones[i]: [int(kn[i]), int(km[i]), 0] for i in keep}
    return out


def unpack(x):
    prm = e37.unpack(x[:23])
    prm["r46"] = float(1 / (1 + np.exp(-x[23])))
    return prm


def targets(cnt):
    t = e36.targets(cnt)
    t["r_G46"] = float(test_stats(cnt["G"][4], cnt["G"][6], boot=1)["r"])
    return t


def extra_se(obs):
    return {**e36.extra_se(obs), "r_G46": R46_OBS_SE}


def warm_x(src):
    """An E40 fit mapped into the 24-dim space (its phi, eps and fixed s_G kept), r46 0.9."""
    p = json.load(open(os.path.join(OUT, f"{src}.json")))["params"]
    x = list(e37.warm_x(src))
    x[21], x[22] = e36.logit(p["phi"]), e36.logit(p["eps"])
    return np.clip([*x, e36.logit(0.9)], -3, 3)


WARM = ["e40g2.5_s0", "e40g2.5_s1", "e40g2.5_s0", "e40g4_s0", "e40g4_s1", "e40g4_s2"]
WARM += ["e40g6.5_s0", "e40g6.5_s2", "e40g6.5_s0"]


if __name__ == "__main__":
    seed = int(sys.argv[1])
    e34.e33.unpack_base = e33.unpack  # e36.unpack_signed builds on the E33 parameters
    e33.simulate, e33.unpack = simulate, unpack
    e33.TARGETS, e33.EXTRA_SE, e33.SIMS = targets, extra_se, 10
    x0 = warm_x(WARM[seed]) if seed < len(WARM) else None
    e33.main(seed, maxiter=500, dim=24, name="e41", x0=x0)
