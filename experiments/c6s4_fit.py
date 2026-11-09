"""C6-S4: fit clone-to-pole coupling of variant A4 by leave-one-clone-out CV; then out-of-fit Q2/Q3 and fixed-model
predictions for the sealed sets S1/S2 (no sealed file is read). See EXPERIMENT_LOG C6-S4 (pre-registered, 3ae6a5e).

    uv run --with scikit-image --with numba python c6s4_fit.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.stats import binom, norm, spearmanr

import core_model as cm
import c6s2_space as c2
import c6s3_break as b

OUT = c2.OUT
N = 300
U0 = 0.1  # step 0: u0 0.1 keeps one axis at more grid points than 0.5 (synthetic only)
# step 0 trim (synthetic +-2 mix, polar >= 0.5): only these 2 of the 12 pre-registered points survive
GRID = [(0.05, 1.0), (0.05, 4.0)]
STEP0 = {"(0.05,1)": 0.80, "(0.05,4)": 0.57, "(0.05,16)": 0.24, "(0.15,1)": 0.21, "(0.15,4)": 0.15, "(0.15,16)": 0.17,
         "(0.4,1)": 0.13, "(0.4,4)": 0.15, "(0.4,16)": 0.19, "(1,1)": 0.12, "(1,4)": 0.15, "(1,16)": 0.16}
FS = c2.C3S1["fit_shard"]
KS = sorted(FS)
TGT = {k: FS[k]["mean"] for k in KS}
NW = {k: FS[k]["n"] for k in KS}
PURE_GAP_DATA = 0.222 - 0.042  # C4-S4 spent pure plate, 1 - circularity: bulk - 6a
S2_SE = 0.077 / (norm.ppf(0.95) + norm.ppf(0.84))  # SE implied by the C6-S1 power table (0.84 at +0.077)


def prm_of(b0, chi, k_ext=1.0):
    return {**b.params("A", "4", k_ext), "b0": b0, "chi": chi, "u0": U0}


def frac_profile(t, mask):
    """Clone share per bin (what a line scan of the clone's label measures), z-normalised per gastruloid."""
    bins = np.minimum((t / 100 * c2.N_BINS).astype(int), c2.N_BINS - 1)
    tot = np.bincount(bins, minlength=c2.N_BINS).astype(float)
    lab = np.bincount(bins[mask], minlength=c2.N_BINS).astype(float)
    f = np.where(tot > 0, lab / np.maximum(tot, 1), np.nan)
    f = np.where(np.isnan(f), np.nanmean(f), f)
    sd = f.std()
    return (f - f.mean()) / sd if sd > 0 else f * 0


def _sim(parts, prm, seed):
    rng = np.random.default_rng(seed)
    s, c, lab = b.gastruloid(parts, N, rng)
    x, P, pr, sh = b.simulate_A(s, c, prm, rng)
    return x, P, sh, lab


def job_clone(a):
    b0, chi, k, seed = a
    x, P, sh, lab = _sim([(k, 0.5, c2.Z[k], 1.0), ("bulk", 0.5, None, 1.0)], prm_of(b0, chi), seed)
    return c2.polarity_of(frac_profile(c2.axis_coords(x, P), lab == k))


def job_q2(a):
    b0, chi, f, seed = a
    x, P, sh, lab = _sim([("3", f, c2.Z["3"], 1.0), ("1", 1 - f, c2.Z["1"], 1.0)], prm_of(b0, chi), seed)
    t = c2.axis_coords(x, P)
    return frac_profile(t, lab == "3"), frac_profile(t, lab == "1"), c2.profile(t, lab == "3"), c2.profile(t, lab == "1")


def job_shape(a):
    b0, chi, parts, seed = a
    return _sim(parts, prm_of(b0, chi), seed)[2]


def job_s1(a):
    b0, chi, r, seed = a
    rng = np.random.default_rng(seed)
    z1, z6 = cm.memory_through(np.array([c2.Z["1"], c2.Z["6"]]), 0, ["passage"] * 3, 1.0, {"passage": r * r}, rng)
    x, P, sh, lab = _sim([("1", 0.5, z1, 1.0), ("6", 0.5, z6, 1.0)], prm_of(b0, chi), seed + 1)
    t = c2.axis_coords(x, P)
    return c2.polarity_of(frac_profile(t, lab == "1")) > c2.polarity_of(frac_profile(t, lab == "6"))


def loss(sim, ks):
    return sum(NW[k] * (sim[k] - TGT[k]) ** 2 for k in ks)


def main():
    t0 = time.time()
    res = {"grid": GRID, "u0": U0, "step0_polar_synthetic": STEP0, "targets": TGT}
    pool = Pool(10)
    sims, sds = {}, {}
    for gi, (b0, chi) in enumerate(GRID):
        sims[gi], sds[gi] = {}, {}
        for k in KS:
            v = pool.map(job_clone, [(b0, chi, k, 64000 + 1000 * gi + 100 * KS.index(k) + r) for r in range(40)])
            sims[gi][k], sds[gi][k] = float(np.mean(v)), float(np.std(v, ddof=1))
        print("grid", (b0, chi), {k: round(sims[gi][k], 3) for k in KS}, flush=True)
    res["sim_by_grid"] = {str(GRID[g]): {"mean": sims[g], "sd": sds[g]} for g in sims}
    # leave-one-clone-out CV
    cv = {}
    for k in KS:
        rest = [j for j in KS if j != k]
        g = min(sims, key=lambda g: loss(sims[g], rest))
        cv[k] = {"chosen": str(GRID[g]), "pred": sims[g][k], "target": TGT[k], "null": float(np.mean([TGT[j] for j in rest]))}
    signs = int(sum(np.sign(cv[k]["pred"]) == np.sign(TGT[k]) for k in KS))
    mse = float(np.mean([(cv[k]["pred"] - TGT[k]) ** 2 for k in KS]))
    mse0 = float(np.mean([(cv[k]["null"] - TGT[k]) ** 2 for k in KS]))
    rho = float(spearmanr([cv[k]["pred"] for k in KS], [TGT[k] for k in KS]).correlation)
    res["CV"] = {"by_clone": cv, "signs": signs, "mse": mse, "mse_null": mse0, "spearman": rho,
                 "pass": bool(signs == 5 and mse < mse0 and rho >= 0.9)}
    print("CV", json.dumps(res["CV"], indent=1), flush=True)
    if not res["CV"]["pass"]:
        res["note"] = "CV failed: no fixed model, no S1/S2 prediction (pre-registered)."
        res["minutes"] = (time.time() - t0) / 60
        json.dump(res, open(os.path.join(OUT, "c6s4_fit.json"), "w"), indent=1)
        return res
    g = min(sims, key=lambda g: loss(sims[g], KS))
    b0, chi = GRID[g]
    res["fixed"] = {"b0": b0, "chi": chi, "u0": U0}
    # Q2 out of fit
    xs = (np.arange(c2.N_BINS) + 0.5) * 100 / c2.N_BINS
    q2 = []
    for i, f in enumerate((0.2, 1 / 3, 0.5, 2 / 3, 0.8)):
        v = pool.map(job_q2, [(b0, chi, f, 65000 + 100 * i + r) for r in range(20)])
        fr = c2.crossing(xs, np.mean([a[0] for a in v], 0), np.mean([a[1] for a in v], 0))
        de = c2.crossing(xs, np.mean([a[2] for a in v], 0), np.mean([a[3] for a in v], 0))
        q2.append({"f_anterior": f, "crossing_fraction": float(fr), "crossing_density": float(de)})
    lo, hi = c2.G1B["slope_ci"]
    out_q2 = {"crossings": q2, "g1b_ci": [lo, hi]}
    for key in ("crossing_fraction", "crossing_density"):
        ok = [o for o in q2 if not np.isnan(o[key])]
        sl = float(np.polyfit([100 * o["f_anterior"] for o in ok], [o[key] for o in ok], 1)[0]) if len(ok) >= 3 else float("nan")
        out_q2[key + "_slope"] = sl
    out_q2["pass"] = bool(lo <= out_q2["crossing_fraction_slope"] <= hi)
    res["Q2"] = out_q2
    # Q3 out of fit
    q3 = {}
    for i, f in enumerate((0.0, 0.2, 0.5, 0.8, 1.0)):
        parts = ([("bulk", 1.0, None, 1.0)] if f == 0 else [("low", 1.0, 0.0, 0.2)] if f == 1
                 else [("low", f, 0.0, 0.2), ("bulk", 1 - f, None, 1.0)])
        v = pool.map(job_shape, [(b0, chi, parts, 66000 + 100 * i + r) for r in range(20)])
        q3[str(f)] = {m: float(np.mean([a[m] for a in v])) for m in v[0]}
    y = [q3[str(f)]["moment_elong"] for f in (0.0, 0.2, 0.5, 0.8, 1.0)]
    res["Q3"] = {"by_fraction": q3, "pass": bool(all(a >= c for a, c in zip(y, y[1:])))}
    # smoke P1/E1 regression at the fixed coupling
    b.A_SETTINGS["fit"] = {**b.A_SETTINGS["4"], "b0": b0, "chi": chi, "u0": U0}
    sm = b.smoke("A", "fit")
    res["smoke"] = {k: sm[k] for k in ("P1", "P2", "E1", "E2")}
    res["smoke_P1_E1_pass"] = bool(sm["P1"]["polar"] >= 0.5 and sm["E1"]["pass"])
    print("Q2/Q3/smoke", out_q2["crossing_fraction_slope"], res["Q3"]["pass"], res["smoke_P1_E1_pass"], flush=True)
    # S1 prediction + H14 power (n 34, one-sided binomial, alpha 0.05)
    kcrit = int(binom.ppf(0.95, 34, 0.5)) + 1
    s1 = {}
    for i, r in enumerate((0.46, 0.60, 0.68)):
        v = pool.map(job_s1, [(b0, chi, r, 67000 + 10000 * i + 2 * j) for j in range(400)])
        sh = float(np.mean(v))
        s1[str(r)] = {"share_1_anterior_of_6": sh, "power_n34": float(binom.sf(kcrit - 1, 34, sh))}
    low = min(s1.values(), key=lambda d: d["share_1_anterior_of_6"])
    res["S1"] = {"by_retention": s1, "k_crit": kcrit, "prediction": low["share_1_anterior_of_6"],
                 "power": low["power_n34"], "H14_pass": bool(low["power_n34"] >= 0.8)}
    # S2 prediction + H14 power
    s2 = {}
    for name, z6 in (("z0_as_Q3", 0.0), ("z_clone6", c2.Z["6"])):
        parts = {"bulk": [("bulk", 1.0, None, 1.0)], "6a": [("6a", 1.0, z6, 0.2)],
                 "mix": [("6a", 0.5, z6, 0.2), ("bulk", 0.5, None, 1.0)]}
        m = {}
        for j, (key, pp) in enumerate(parts.items()):
            v = pool.map(job_shape, [(b0, chi, pp, 68000 + 1000 * j + 5000 * (name != "z0_as_Q3") + r) for r in range(100)])
            m[key] = float(np.mean([a["one_minus_circ"] for a in v]))
        gap = m["bulk"] - m["6a"]
        frac = (m["mix"] - 0.5 * (m["bulk"] + m["6a"])) / gap if gap > 0 else float("nan")
        eff = frac * PURE_GAP_DATA
        s2[name] = {**m, "mix_over_member_mean_frac_of_gap": frac, "effect_data_units": eff,
                    "power": float(norm.sf(norm.ppf(0.95) - eff / S2_SE))}
    res["S2"] = {"by_state": s2, "se": S2_SE, "prediction": s2["z0_as_Q3"]["effect_data_units"],
                 "power": s2["z0_as_Q3"]["power"], "H14_pass": bool(s2["z0_as_Q3"]["power"] >= 0.8)}
    res["minutes"] = (time.time() - t0) / 60
    json.dump(res, open(os.path.join(OUT, "c6s4_fit.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("fixed", "Q2", "Q3", "smoke_P1_E1_pass", "S1", "S2")}, indent=1))
    return res


if __name__ == "__main__":
    main()
