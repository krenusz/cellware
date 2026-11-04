"""E40: E38's model with G-CSF neutrophil survival s_G fixed, for a profile along it.
See the E40 pre-registration in EXPERIMENT_LOG.md.

    uv run python experiments/e40_profile.py seed s_G
"""

from __future__ import annotations

import sys

import e33_measured as e33
import e34_fullarm as e34
import e36_combined as e36
import e37_pool as e37

if __name__ == "__main__":
    seed, s_g = int(sys.argv[1]), float(sys.argv[2])

    def unpack(x):
        prm = e37.unpack(x)
        prm["s"]["G"] = s_g
        return prm

    e34.e33.unpack_base = e33.unpack  # e36.unpack_signed builds on the E33 parameters
    e33.simulate, e33.unpack = e37.simulate, unpack
    e33.TARGETS, e33.EXTRA_SE, e33.SIMS = e36.targets, e36.extra_se, 10
    e33.main(seed, maxiter=500, dim=23, name=f"e40g{s_g:g}")
