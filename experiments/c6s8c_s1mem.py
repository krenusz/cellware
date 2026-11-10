"""C6-S8c: S1 power with clone-specific passage-4 memory (spent G6a/G6b calibrations). Synthetic only.
See EXPERIMENT_LOG C6-S8c. Chunked on GitHub Actions (--part TAG), merged locally (no args).

    python c6s8c_s1mem.py --plan | --part TAG | (no args = merge)
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.stats import binom

import c6s2_space as c2
import c6s4_fit as f4
import c6s7_checks as ck
import c6s7l_axis as L
import c6s8a_power as A

OUT = c2.OUT
L.PARTS = os.path.join(OUT, "c6s8c_parts")
MOD = A.MOD
VARIANTS = {"V1": (-0.06, 0.60), "V2": (0.24, 0.18), "V3": (0.68, 0.68)}  # (clone 1, clone 6) passage-4 retention
COMBOS = [("a", "V1"), ("a", "V2"), ("c", "V1"), ("c", "V2"), ("a", "V3")]
N = 120
CH = 8


def job(a):
    p, m1, m6, seed = a
    z1, z6 = c2.Z["1"] * m1, c2.Z["6"] * m6
    x, P, sh, lab = ck._sim(MOD, [("1", 0.5, float(z1), 1.0), ("6", 0.5, float(z6), 1.0)], p, seed)
    t = c2.axis_coords(x, P)
    return bool(c2.polarity_of(f4.frac_profile(t, lab == "1")) > c2.polarity_of(f4.frac_profile(t, lab == "6")))


def collect():
    raw = {}
    for ci, (arm, v) in enumerate(COMBOS):
        m1, m6 = VARIANTS[v]
        tag = f"S1_{arm}_{v}"
        raw[tag] = L.gather(job, tag, [(A.ARMS[arm], m1, m6, 121000 + 2000 * ci + 2 * j) for j in range(N)], CH)
    return raw


def main():
    raw = collect()
    kcrit = int(binom.ppf(0.95, 34, 0.5)) + 1
    res = {"k_crit": kcrit, "n_sims": N, "variants": VARIANTS, "by_combo": {}}
    for arm, v in COMBOS:
        r = raw[f"S1_{arm}_{v}"]
        sh = float(np.mean(r))
        res["by_combo"][f"{arm}_{v}"] = {"share_1_anterior_of_6": sh, "se": float(np.sqrt(sh * (1 - sh) / len(r))),
                                         "n": len(r), "power_n34": float(binom.sf(kcrit - 1, 34, sh))}
    gate = [res["by_combo"][f"{a}_{v}"]["power_n34"] for a, v in COMBOS if v in ("V1", "V2")]
    res["gate"] = {"min_power_V1V2": min(gate), "S1_proposable": bool(min(gate) >= 0.8),
                   "V3_power": res["by_combo"]["a_V3"]["power_n34"]}
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(OUT, "c6s8c_s1mem.json"), "w"), indent=1)


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--plan" in a:
        L.MODE["plan"] = []
        collect()
        print("\n".join(L.MODE["plan"]))
    elif "--part" in a:
        L.MODE["only"] = a[a.index("--part") + 1]
        try:
            collect()
        except L._Done:
            pass
    else:
        main()
