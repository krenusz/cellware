"""C6-S7m: C6-S7l data checks (Q1-Q3, S2) at arm c (mediolateral all-cell crawling, t_on 100k, lam_F 20).
Sensitivity only. See EXPERIMENT_LOG C6-S7m.

    python c6s7m_armc.py --plan 2 | --part TAG | (no args = merge)
"""

from __future__ import annotations

import json
import os
import sys

import c6s7l_axis as L

L.PARTS = os.path.join(L.OUT, "c6s7m_parts")
L.S1_JSON = os.path.join(L.OUT, "c6s7m_point.json")  # holds {"chosen": arm c point}; committed before the run
L.CHECKS_JSON = os.path.join(L.OUT, "c6s7m_checks.json")

if __name__ == "__main__":
    a = sys.argv[1:]
    if "--plan" in a:
        L.MODE["plan"] = []
        L.collect2()
        print("\n".join(L.MODE["plan"]))
    elif "--part" in a:
        L.MODE["only"] = a[a.index("--part") + 1]
        try:
            L.collect2()
        except L._Done:
            pass
    else:
        L.stage2()
