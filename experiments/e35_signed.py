"""E35: E33 with literature-signed instruction (tau_G >= tau_full >= tau_M).
See the E35 pre-registration in EXPERIMENT_LOG.md.

    uv run python experiments/e35_signed.py seed
"""

from __future__ import annotations

import sys

import e33_measured as e33
import numpy as np


def unpack(x):
    prm = e33.unpack_base(x)
    t0 = prm["tau"]["full"]
    prm["tau"]["G"] = float(t0 + 8 / (1 + np.exp(-x[3])))
    prm["tau"]["M"] = float(t0 - 8 / (1 + np.exp(-x[4])))
    return prm


if __name__ == "__main__":
    e33.unpack_base = e33.unpack
    e33.unpack = unpack
    e33.main(int(sys.argv[1]) if len(sys.argv) > 1 else 0, maxiter=500, dim=19, name="e35")
