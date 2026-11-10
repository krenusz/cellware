"""C6-S8a: power inventory for an unseen test at the C6-S7l model (arm a) and C6-S7m (arm c). Synthetic only.
See EXPERIMENT_LOG C6-S8a. Chunked on GitHub Actions (--part TAG), merged locally (no args).

    python c6s8a_power.py --plan | --part TAG | (no args = merge)
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.stats import binom, norm

import c6s2_space as c2
import c6s4_fit as f4
import c6s7_checks as ck
import c6s7l_axis as L
import core_model as cm

OUT = c2.OUT
L.PARTS = os.path.join(OUT, "c6s8a_parts")
MOD = "c6s7l_axis"
ARMS = {"a": L.pparams("a", 150000, 20.0), "c": L.pparams("c", 100000, 20.0)}
MODEL_GAP = {"a": 0.401, "c": 0.352}  # pure bulk - pure 6a, 1 - circularity (C6-S7l / C6-S7m S2)
R_S1 = 0.46
N_S1 = 200
C3 = (0.42, 0.50)  # clone 3 competence from open pure-clone elongated fractions (same-date; all dates)
FRS = (0.2, 0.5, 0.8)
COUNTS = (9, 16, 18)  # 3.1 at 20/50/80% (locked file, label columns only)
SD_DATA = 0.084  # per-gastruloid SD of 1 - circularity (C4-S4 spent)
PURE_GAP_DATA = f4.PURE_GAP_DATA  # 0.180


def job_s1(a):
    p, r, seed = a
    rng = np.random.default_rng(seed)
    z1, z6 = cm.memory_through(np.array([c2.Z["1"], c2.Z["6"]]), 0, ["passage"] * 3, 1.0, {"passage": r * r}, rng)
    x, P, sh, lab = ck._sim(MOD, [("1", 0.5, float(z1), 1.0), ("6", 0.5, float(z6), 1.0)], p, seed + 1)
    t = c2.axis_coords(x, P)
    return bool(c2.polarity_of(f4.frac_profile(t, lab == "1")) > c2.polarity_of(f4.frac_profile(t, lab == "6")))


def collect():
    raw = {}
    for ai, (arm, p) in enumerate(ARMS.items()):
        raw[f"S1_{arm}"] = L.gather(job_s1, f"S1_{arm}", [(p, R_S1, 101000 + 5000 * ai + 2 * j) for j in range(N_S1)], 10)
        for zi, (zn, z) in enumerate((("z0", 0.0), ("z3", c2.Z["3"]))):
            for ci, c3 in enumerate(C3):
                for fi, fr in enumerate(FRS):
                    parts = [("3s1", fr, z, c3), ("bulk", 1 - fr, None, 1.0)]
                    tag = f"C_{arm}_{zn}_c{ci}_f{fi}"
                    seed0 = 111000 + 10000 * ai + 1000 * zi + 300 * ci + 100 * fi
                    raw[tag] = L.gather(ck.job_shape, tag, [(MOD, p, parts, seed0 + r) for r in range(20)], 10)
    return raw


def main():
    raw = collect()
    res = {"arms": {k: {kk: v for kk, v in p.items() if kk in ("t_on", "lam_F", "orient", "crawl")} for k, p in ARMS.items()}}
    kcrit = int(binom.ppf(0.95, 34, 0.5)) + 1
    s1 = {}
    for arm in ARMS:
        sh = float(np.mean(raw[f"S1_{arm}"]))
        s1[arm] = {"share_1_anterior_of_6": sh, "power_n34": float(binom.sf(kcrit - 1, 34, sh))}
    res["S1"] = {"r": R_S1, "k_crit": kcrit, "by_arm": s1, "min_power": min(v["power_n34"] for v in s1.values())}
    fx = np.repeat(FRS, COUNTS)
    se = SD_DATA / np.sqrt(np.sum((fx - fx.mean()) ** 2))
    cres = {}
    for arm in ARMS:
        for zn in ("z0", "z3"):
            for ci, c3 in enumerate(C3):
                xs, ys = [], []
                for fi, fr in enumerate(FRS):
                    v = [a["one_minus_circ"] for a in raw[f"C_{arm}_{zn}_c{ci}_f{fi}"]]
                    xs += [fr] * len(v)
                    ys += v
                slope_model = float(np.polyfit(xs, ys, 1)[0])
                slope_data = slope_model * PURE_GAP_DATA / MODEL_GAP[arm]
                power = float(norm.sf(norm.ppf(0.95) - (-slope_data) / se))  # one-sided, negative slope
                cres[f"{arm}_{zn}_c3={c3}"] = {"means": [float(np.mean([a["one_minus_circ"] for a in
                                                                        raw[f"C_{arm}_{zn}_c{ci}_f{fi}"]]))
                                                         for fi in range(len(FRS))],
                                               "slope_model": slope_model, "slope_data_units": slope_data,
                                               "power": power}
    res["C"] = {"counts": dict(zip(map(str, FRS), COUNTS)), "slope_se": float(se), "by_combo": cres,
                "min_power": min(v["power"] for v in cres.values()),
                "note": "partly exposed: direction known from the authors' README; % plate-confounded"}
    res["gate"] = {"S1_robust": bool(res["S1"]["min_power"] >= 0.8), "C_robust": bool(res["C"]["min_power"] >= 0.8)}
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(OUT, "c6s8a_power.json"), "w"), indent=1)


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
