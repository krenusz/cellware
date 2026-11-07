"""C5-S2: one memory law across blood and gastruloid clones? Interval checks on used estimates.

    uv run python c5s2_memory_law.py

All inputs are numbers already in EXPERIMENT_LOG (E19, E19c, E41b, B38, G3, G5, C3-S1); nothing sealed is read.
"""

from __future__ import annotations

import json
import os

import numpy as np

import core_model as cm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_gastruloid", "c5s2_memory_law.json")

# Used estimates (point, lo, hi) with their log sources
E19C_SMALL = (0.95, 0.85, 1.00)     # LARRY cross-well sharing, clones of 2-4 cells
E19C_LARGE = (0.82, 0.72, 0.90)     # clones of 12-143 cells
E19C_DIFF = (-0.13, -0.25, -0.01)   # large - small
G5_P1_RATIO = (1.21, 1.005, 1.50)   # subclone / parent polarity, passage-1 subclones, >= 13 divisions
G5_DIVS = 13
G3_R = (0.60, 0.50, 0.68)           # retention per passage (mean polarity)
G5_P2_OVER_P1 = (0.37, 0.23, 0.51)  # subclones one passage later
C3S1_RETENTION = 0.456              # sqrt(rho_P) from C3-S1
LARRY_XWELL = (0.88, 0.82, 0.92)    # E19
SIS_H = (0.71, 0.59, 0.78)          # B38 / #13


def k1(d):
    """Implied per-division retention intervals: blood (E19c, d extra divisions) and gastruloid (G5)."""
    s = E19C_SMALL[0]
    blood = [((s + E19C_DIFF[i]) / s) ** (1 / d) for i in (1, 0, 2)]  # lo, point, hi
    # gastruloid: mean shrink sqrt(rho_div)^13 >= ratio lower bound (capped at 1: no amplification in the law)
    lb = min(G5_P1_RATIO[1], 1.0)
    gast_lo = lb ** (2 / G5_DIVS)
    return {"blood_rho_div": [round(v, 4) for v in blood], "gastruloid_rho_div_lower": round(gast_lo, 4),
            "overlap": bool(blood[2] >= gast_lo)}


def sim_check(rho_div, d, reps=20000, seed=5):
    """Simulate the law: cross-well correlation after extra d divisions per sister, vs no extra divisions."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(reps)
    a0, b0 = cm.state_copy(z, 1.0, rng), cm.state_copy(z, 1.0, rng)
    a = cm.memory_through(z, d, [], rho_div, {}, rng)
    b = cm.memory_through(z, d, [], rho_div, {}, rng)
    return round(float(np.corrcoef(a, b)[0, 1] / np.corrcoef(a0, b0)[0, 1]), 4)


def main():
    res = {"K1": {}, "K1_sim": {}}
    for d in (3, 4, 5, 6):
        res["K1"][f"d={d}"] = k1(d)
        pt = res["K1"][f"d={d}"]["blood_rho_div"][1]
        # law check: with rho_div = blood point, the simulated large/small sharing ratio = rho_div^d
        res["K1_sim"][f"d={d}"] = {"rho_div": pt, "sim_ratio": sim_check(pt, d),
                                   "target_ratio": round((E19C_SMALL[0] + E19C_DIFF[0]) / E19C_SMALL[0], 4)}
    res["K1_pass_at_d4"] = res["K1"]["d=4"]["overlap"]
    lo = max(G3_R[1], G5_P2_OVER_P1[1])
    hi = min(G3_R[2], G5_P2_OVER_P1[2])
    res["K2"] = {"G3_r": G3_R, "G5_p2_over_p1": G5_P2_OVER_P1, "C3S1_retention": C3S1_RETENTION,
                 "common_interval": [lo, hi] if lo <= hi else None,
                 "C3S1_inside": bool(lo <= C3S1_RETENTION <= hi) if lo <= hi else False}
    res["K2_pass"] = bool(lo <= hi and lo <= C3S1_RETENTION <= hi)
    res["K3_described"] = {"LARRY_cross_well": LARRY_XWELL, "SIS_heritable_share": SIS_H,
                           "note": "SIS lower than LARRY; consistent with per-division loss over a longer culture, "
                                   "not testable without SIS division counts"}
    k1p, k2p = res["K1_pass_at_d4"], res["K2_pass"]
    res["reading"] = ("one law holds on used data" if k1p and k2p else
                      ("per-division memory differs between blood progenitors and pluripotent cells" if not k1p else "") +
                      ("; " if (not k1p and not k2p) else "") +
                      ("passaging loss is series-specific (G8)" if not k2p else ""))
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
