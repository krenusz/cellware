"""E48: E46 (lineage-matched selective expansion) with gamma capped at 0.6, the generous end of
the range allowed by Rieger 2009's supplement (max ~2 doublings/day; a +1 SD clone may grow at
most about one doubling faster than average over 48 h). E47 used the strict end (0.35). See the
E48 pre-registration in EXPERIMENT_LOG.md.

    uv run python experiments/e48_capsel6.py seed
"""

from __future__ import annotations

import sys

import e33_measured as e33
import e34_fullarm as e34
import e36_combined as e36
import e37_pool as e37
import e46_selexp as e46
import numpy as np

GAMMA_MAX = 0.6
simulate = e46.simulate


def unpack(x):
    prm = e37.unpack(x[:23])
    prm["gamma"] = float(GAMMA_MAX / (1 + np.exp(-x[23])))
    return prm


if __name__ == "__main__":
    e34.e33.unpack_base = e33.unpack  # e36.unpack_signed builds on the E33 parameters
    e33.simulate, e33.unpack = simulate, unpack
    e33.TARGETS, e33.EXTRA_SE, e33.SIMS = e36.targets, e36.extra_se, 10
    e33.main(int(sys.argv[1]), maxiter=500, dim=24, name="e48")
