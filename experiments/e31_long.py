"""E31: E30a run longer (500 generations), seeds 0-2 warm-started from E30a s2, seeds 3-5 cold.
See the 2026-11-03 (7) batch pre-registration in EXPERIMENT_LOG.md.

    uv run python experiments/e31_long.py seed
"""

from __future__ import annotations

import sys

import e30_refit

if __name__ == "__main__":
    seed = int(sys.argv[1])
    e30_refit.main("a", seed, name="e31", warm="e30a_s2" if seed < 3 else None, maxiter=500)
