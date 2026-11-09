"""C6-S7a: amendment, asymmetric cohesion (Guan/Hormoz top settings). See EXPERIMENT_LOG C6-S7a (pre-registered,
9ce3d22). Reuses c6s7_sort.simulate_S; same step-0 criteria, data checks and H17 S2 power as C6-S7.

    uv run --with scikit-image --with numba python c6s7a_sort.py
"""

from __future__ import annotations

import json
import os
import time
from multiprocessing import Pool

import numpy as np
from scipy.stats import norm, spearmanr

import c6s2_space as c2
import c6s3_break as b
import c6s4_fit as f4
import c6s7_sort as m7

OUT = c2.OUT
N = 300
LAM = 2.0
GRID0 = [dict(w_PP=wp, w_AP=wa, beta=be) for wp in (0.3, 0.6) for wa in (0.1, 0.33) for be in (0.2, 1.0)]


def prm(g, lam=LAM):
    return {**m7.BASE, "w_AA": 1.0, "R_lr": 40.0, "lam_F": lam, **g}


def _sim(parts, p, seed):
    rng = np.random.default_rng(seed)
    s, c, lab = b.gastruloid(parts, N, rng)
    x, P, sh = m7.simulate_S(s, c, p, rng)
    return x, P, sh, lab


def job0(a):
    p, comp, seed = a
    rng = np.random.default_rng(seed)
    s = np.r_[2 + rng.standard_normal(N // 2), -2 + rng.standard_normal(N // 2)]
    t0 = time.time()
    x, P, sh = m7.simulate_S(s, np.full(N, comp), p, rng)
    return b.polar_index(x, P), m7.multipolar(x, P), sh["moment_elong"], time.time() - t0


def job_clone(a):
    p, k, seed = a
    x, P, sh, lab = _sim([(k, 0.5, c2.Z[k], 1.0), ("bulk", 0.5, None, 1.0)], p, seed)
    return c2.polarity_of(f4.frac_profile(c2.axis_coords(x, P), lab == k)), m7.multipolar(x, P), b.polar_index(x, P)


def job_q2(a):
    p, fr, seed = a
    x, P, sh, lab = _sim([("3", fr, c2.Z["3"], 1.0), ("1", 1 - fr, c2.Z["1"], 1.0)], p, seed)
    t = c2.axis_coords(x, P)
    return f4.frac_profile(t, lab == "3"), f4.frac_profile(t, lab == "1")


def job_shape(a):
    p, parts, seed = a
    return _sim(parts, p, seed)[2]


def main():
    t0 = time.time()
    res = {"grid": GRID0, "lam_F": LAM}
    pool = Pool(10)
    s0, passing = {}, []
    for gi, g in enumerate(GRID0):
        r1 = pool.map(job0, [(prm(g), 1.0, 9000 + 20 * gi + k) for k in range(6)])
        rc = pool.map(job0, [(prm(g), 0.0, 9400 + 20 * gi + k) for k in range(6)])
        rl = pool.map(job0, [(prm(g, 0.0), 1.0, 9800 + 20 * gi + k) for k in range(6)])
        v = {"polar": float(np.mean([a[0] for a in r1])), "multipolar_rate": float(np.mean([a[1] for a in r1])),
             "elong": float(np.mean([a[2] for a in r1])), "elong_c0": float(np.mean([a[2] for a in rc])),
             "elong_lam0": float(np.mean([a[2] for a in rl])), "polar_lam0": float(np.mean([a[0] for a in rl])),
             "sec": float(np.mean([a[3] for a in r1]))}
        v["S0a"] = bool(v["polar"] >= 0.5 and v["multipolar_rate"] <= 0.2)
        v["S0b"] = bool(v["elong"] - v["elong_lam0"] >= 0.15)
        v["S0c"] = bool(abs(v["elong_c0"] - v["elong_lam0"]) <= 0.05)
        v["S0d"] = bool(v["sec"] <= 10)
        s0[json.dumps(g)] = v
        if v["S0a"] and v["S0b"] and v["S0c"] and v["S0d"]:
            passing.append(g)
        print("step0", g, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()}, flush=True)
    res["step0"] = s0
    res["step0_passing"] = passing
    if not passing:
        res["note"] = "no step-0 point passes: state-first design fails in this particle model (pre-registered)"
        res["minutes"] = (time.time() - t0) / 60
        json.dump(res, open(os.path.join(OUT, "c6s7a_sort.json"), "w"), indent=1)
        print(res["note"])
        return
    g = max(passing, key=lambda g: s0[json.dumps(g)]["polar"])
    p = prm(g)
    res["chosen"] = g
    print("chosen", g, flush=True)
    sim, mp, pi = {}, {}, {}
    for ki, k in enumerate(f4.KS):
        v = pool.map(job_clone, [(p, k, 91000 + 100 * ki + r) for r in range(40)])
        sim[k] = float(np.mean([a[0] for a in v]))
        mp[k] = float(np.mean([a[1] for a in v]))
        pi[k] = float(np.mean([a[2] for a in v]))
    ks = f4.KS
    signs = int(sum(np.sign(sim[k]) == np.sign(f4.TGT[k]) for k in ks))
    rho = float(spearmanr([sim[k] for k in ks], [f4.TGT[k] for k in ks]).correlation)
    res["Q1"] = {"sim": sim, "target": f4.TGT, "signs": signs, "spearman": rho, "multipolar_rate": mp, "polar": pi,
                 "pass": bool(signs == 5 and rho >= 0.9)}
    print("Q1", json.dumps(res["Q1"]), flush=True)
    xs = (np.arange(c2.N_BINS) + 0.5) * 100 / c2.N_BINS
    q2 = []
    for i, fr in enumerate((0.2, 1 / 3, 0.5, 2 / 3, 0.8)):
        v = pool.map(job_q2, [(p, fr, 92000 + 100 * i + r) for r in range(20)])
        q2.append({"f_anterior": fr, "crossing": float(c2.crossing(xs, np.mean([a[0] for a in v], 0), np.mean([a[1] for a in v], 0)))})
    ok = [o for o in q2 if not np.isnan(o["crossing"])]
    sl = float(np.polyfit([100 * o["f_anterior"] for o in ok], [o["crossing"] for o in ok], 1)[0]) if len(ok) >= 3 else float("nan")
    lo, hi = c2.G1B["slope_ci"]
    res["Q2"] = {"crossings": q2, "slope": sl, "g1b_ci": [lo, hi], "pass": bool(lo <= sl <= hi)}
    q3 = {}
    for i, fr in enumerate((0.0, 0.2, 0.5, 0.8, 1.0)):
        parts = ([("bulk", 1.0, None, 1.0)] if fr == 0 else [("low", 1.0, 0.0, 0.2)] if fr == 1
                 else [("low", fr, 0.0, 0.2), ("bulk", 1 - fr, None, 1.0)])
        v = pool.map(job_shape, [(p, parts, 93000 + 100 * i + r) for r in range(20)])
        q3[str(fr)] = {mm: float(np.mean([a[mm] for a in v])) for mm in v[0]}
    y = [q3[str(fr)]["moment_elong"] for fr in (0.0, 0.2, 0.5, 0.8, 1.0)]
    res["Q3"] = {"by_fraction": q3, "monotone_fall": bool(all(a >= c for a, c in zip(y, y[1:])))}
    print("Q2", sl, res["Q2"]["pass"], "Q3", [round(v, 3) for v in y], flush=True)
    s2 = {}
    for name, z6 in (("z0", 0.0), ("z_clone6", c2.Z["6"])):
        parts = {"bulk": [("bulk", 1.0, None, 1.0)], "6a": [("6a", 1.0, z6, 0.2)],
                 "mix": [("6a", 0.5, z6, 0.2), ("bulk", 0.5, None, 1.0)]}
        mres = {}
        for j, (key, pp) in enumerate(parts.items()):
            v = pool.map(job_shape, [(p, pp, 94000 + 1000 * j + 5000 * (name != "z0") + r) for r in range(100)])
            mres[key] = float(np.mean([a["one_minus_circ"] for a in v]))
            mres[key + "_moment"] = float(np.mean([a["moment_elong"] for a in v]))
        gap = mres["bulk"] - mres["6a"]
        frac = (mres["mix"] - 0.5 * (mres["bulk"] + mres["6a"])) / gap if gap > 0 else float("nan")
        eff = frac * f4.PURE_GAP_DATA
        s2[name] = {**mres, "frac_of_gap": frac, "effect_data_units": eff, "power": float(norm.sf(norm.ppf(0.95) - eff / f4.S2_SE))}
    res["S2"] = {"by_state": s2, "prediction": s2["z0"]["effect_data_units"], "power": s2["z0"]["power"],
                 "H17_label": "powered" if s2["z0"]["power"] >= 0.8 else "exploratory"}
    res["minutes"] = (time.time() - t0) / 60
    json.dump(res, open(os.path.join(OUT, "c6s7a_sort.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("chosen", "Q2", "Q3", "S2")}, indent=1))


if __name__ == "__main__":
    main()
