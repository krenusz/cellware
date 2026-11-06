"""E53: E48 refitted with the core model's LARRY configuration (core_larry.simulate) in place of
e46/e48's simulate. Everything else as E48. See the E53 pre-registration in EXPERIMENT_LOG.md.

    uv run python experiments/e53_core.py seed
"""

from __future__ import annotations

import sys

import core_larry
import e33_measured as e33
import e34_fullarm as e34
import e36_combined as e36
import e48_capsel6 as e48

simulate = core_larry.simulate
unpack = e48.unpack

if __name__ == "__main__":
    e34.e33.unpack_base = e33.unpack  # e36.unpack_signed builds on the E33 parameters
    e33.simulate, e33.unpack = simulate, unpack
    e33.TARGETS, e33.EXTRA_SE, e33.SIMS = e36.targets, e36.extra_se, 10
    e33.main(int(sys.argv[1]), maxiter=500, dim=24, name="e53")
