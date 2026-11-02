"""Median / range summary of a sweep grid, healthy runs only (health_check.check).

    uv run python experiments/summarise_grid.py "experiments/results_clone_e11/a_*.json"

Groups by the run name with the trailing _seedN removed.
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np
from health_check import check

STATS = {"within": lambda d: d["within_loss"], "GM": lambda d: d["pred"]["G-M"]["r_mean"],
         "M_d4d6": lambda d: d["fit_sim"]["M"]["r_d4_d6"],
         "M_only": lambda d: d["fit_sim"]["M"]["m_only"]}

groups = {}
for f in sorted(f for pat in sys.argv[1:] for f in glob.glob(pat)):
    d = json.load(open(f))
    key = re.sub(r"_seed\d+$", "", os.path.basename(f)[:-5])
    groups.setdefault(key, []).append(d)
for key, runs in groups.items():
    ok = [d for d in runs if not check(d)]
    line = f"{key:28s} healthy {len(ok):2d}/{len(runs)}"
    for name, fn in STATS.items():
        v = np.array([fn(d) for d in ok]) if ok else np.array([np.nan])
        line += f"  {name} {np.median(v):.2f} [{v.min():.2f},{v.max():.2f}]"
    if ok and "revert" in ok[0]["pred"]:
        line += "  r0 " + ",".join(f"{d['pred']['revert'][0]:.1f}" for d in ok)
    print(line)
