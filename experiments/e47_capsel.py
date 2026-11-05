"""E47: E46 (lineage-matched selective expansion) with gamma capped at a biologically plausible
0.35, i.e. at most about one extra doubling per SD of fate bias between the G-CSF and M-CSF wells
over day 2 to day 4 (Rieger 2009: GMP pedigrees almost death-free, cycle ~12 h). See the E47
pre-registration in EXPERIMENT_LOG.md.

    uv run python experiments/e47_capsel.py seed
"""

from __future__ import annotations

import sys

import e33_measured as e33
import e34_fullarm as e34
import e36_combined as e36
import e37_pool as e37
import e46_selexp as e46
import numpy as np

GAMMA_MAX = 0.35
simulate = e46.simulate


def unpack(x):
    prm = e37.unpack(x[:23])
    prm["gamma"] = float(GAMMA_MAX / (1 + np.exp(-x[23])))
    return prm


if __name__ == "__main__":
    e34.e33.unpack_base = e33.unpack  # e36.unpack_signed builds on the E33 parameters
    e33.simulate, e33.unpack = simulate, unpack
    e33.TARGETS, e33.EXTRA_SE, e33.SIMS = e36.targets, e36.extra_se, 10
    e33.main(int(sys.argv[1]), maxiter=500, dim=24, name="e47")
