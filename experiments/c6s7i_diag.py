"""C6-S7i: diagnosis of the C6-S7h Q2 (boundary follows proportions) and Q3 (no competence effect) failures.
Diagnosis only, not a test. See EXPERIMENT_LOG C6-S7i.

    uv run --with scikit-image --with numba --with matplotlib python c6s7i_diag.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

import c6s2_space as c2
import c6s3_break as b
import c6s4_fit as f4
import c6s7_checks as ck
import c6s7c_potts as pc
import c6s7g_potts_wet as pg

OUT = c2.OUT
MOD = "c6s7h_potts_lr"
N = 300
FRS = (0.2, 1 / 3, 0.5, 2 / 3, 0.8)
P_CHOSEN = {**pc.BASE, **pg.W["W1"], "mcs": 100000, "beta_lr": 300.0, "lam_F": 80.0}
XS = (np.arange(c2.N_BINS) + 0.5) * 100 / c2.N_BINS


def _theta(s, q):
    lo, hi = -20.0, 20.0
    for _ in range(60):
        th = 0.5 * (lo + hi)
        if np.mean(c2.sigmoid(-s - th)) > q:
            lo = th
        else:
            hi = th
    return 0.5 * (lo + hi)


def ident_geom(fr, rule, seed):
    rng = np.random.default_rng(seed)
    s, _, lab = b.gastruloid([("3", fr, c2.Z["3"], 1.0), ("1", 1 - fr, c2.Z["1"], 1.0)], N, rng)
    if rule == "feedback":
        P = c2.identities(s, c2.G_FB, c2.A_STAR)
    elif rule == "g0":
        P = c2.identities(s, 0.0, c2.A_STAR)
    else:  # independent: no ranking; expected posterior fraction = feedback q
        q = (1 - c2.G_FB) * float(np.mean(c2.sigmoid(-s))) + c2.G_FB * c2.A_STAR
        P = rng.random(N) < c2.sigmoid(-s - _theta(s, q))
    t = np.empty(N)
    iA, iP = rng.permutation(np.flatnonzero(~P)), rng.permutation(np.flatnonzero(P))
    order = np.r_[iA, iP]
    t[order] = (np.arange(N) + 0.5) * 100 / N
    return f4.frac_profile(t, lab == "3"), f4.frac_profile(t, lab == "1")


def slope_of(rule, reps=200):
    cr = []
    for i, fr in enumerate(FRS):
        v = [ident_geom(fr, rule, 95000 + 1000 * i + r) for r in range(reps)]
        cr.append(float(c2.crossing(XS, np.mean([a[0] for a in v], 0), np.mean([a[1] for a in v], 0))))
    ok = [(fr, c) for fr, c in zip(FRS, cr) if not np.isnan(c)]
    sl = float(np.polyfit([100 * a for a, _ in ok], [c for _, c in ok], 1)[0]) if len(ok) >= 3 else float("nan")
    return {"crossings": cr, "slope": sl}


def job_q2c(a):
    p, fr, seed = a
    x, P, sh, lab = ck._sim(MOD, [("3", fr, c2.Z["3"], 1.0), ("1", 1 - fr, c2.Z["1"], 1.0)], p, seed)
    t = c2.axis_coords(x, P)
    out = {}
    for k in ("3", "1"):
        m = lab == k
        out[k] = {"p_post": float(P[m].mean()), "t_in_A": float(t[m & ~P].mean()) if (m & ~P).any() else float("nan"),
                  "t_in_P": float(t[m & P].mean()) if (m & P).any() else float("nan")}
    return out


PARTS = os.path.join(OUT, "c6s7i_parts")
CH = 10
ONLY = None  # set by --part: compute just this chunk file and stop


class _Done(Exception):
    pass


class _Lazy:
    p = None

    def map(self, f, a):
        if self.p is None:
            self.p = Pool(int(os.environ.get("NPROC", "10")))  # worker count does not change results
        return self.p.map(f, a)


def gather(pool, f, tag, args):
    """Chunks of 10 seeds; each chunk is loaded from c6s7i_parts/ if a GitHub Actions job already wrote it."""
    out = []
    for k in range(0, len(args), CH):
        name = f"{tag}_{k}"
        fp = os.path.join(PARTS, name + ".json")
        if ONLY is not None and name != ONLY:
            if os.path.exists(fp):
                out += json.load(open(fp))
            else:
                out += [None] * len(args[k:k + CH])
            continue
        if os.path.exists(fp):
            out += json.load(open(fp))
        else:
            r = pool.map(f, args[k:k + CH])
            if ONLY is not None:
                os.makedirs(PARTS, exist_ok=True)
                json.dump(r, open(fp, "w"))
                raise _Done
            out += r
    if any(v is None for v in out):  # --part run: placeholders for chunks owned by other jobs
        out = [v for v in out if v is not None] or [_DUMMY.get(f.__name__)]
    return out


_DUMMY = {"job_q2c": {"3": {"p_post": 0, "t_in_A": 0, "t_in_P": 0}, "1": {"p_post": 0, "t_in_A": 0, "t_in_P": 0}},
          "job_shape": {"one_minus_circ": 0.0, "moment_elong": 0.0, "attached": 0.0}}


def main(smoke=False):
    t0 = time.time()
    res = {}
    # ---- D1a/b: identity-only geometry ----
    for rule in ("feedback", "g0", "independent"):
        res["D1_" + rule] = slope_of(rule, 20 if smoke else 200)
        print("D1", rule, json.dumps(res["D1_" + rule]), flush=True)
    sl = res["D1_feedback"]["slope"]
    res["D1a_reading"] = ("identity rule" if abs(sl - 0.568) <= 0.2 else "mechanics/readout" if abs(sl) <= 0.2
                          else "mixed")
    print("D1a reading", res["D1a_reading"], flush=True)
    if smoke:
        return
    pool = _Lazy()
    # ---- D1c (conditional) ----
    if res["D1a_reading"] == "mechanics/readout":
        d1c = []
        for i, fr in enumerate(FRS):
            v = gather(pool, job_q2c, f"D1c_{i}", [(P_CHOSEN, fr, 92000 + 100 * i + r) for r in range(20)])
            d1c.append({"f_anterior": fr, **{k: {kk: float(np.nanmean([a[k][kk] for a in v])) for kk in v[0][k]}
                                               for k in ("3", "1")}})
            print("D1c", json.dumps(d1c[-1]), flush=True)
        res["D1c"] = d1c
    # ---- D2: saturation of elongation in c x lam_F ----
    d2 = {}
    for li, lam in enumerate((20.0, 80.0)):
        for ci, c in enumerate((0.0, 0.1, 0.2, 0.5, 1.0)):
            p = {**P_CHOSEN, "lam_F": lam}
            v = gather(pool, ck.job_shape, f"D2_{li}_{ci}", [(MOD, p, [("pure", 1.0, 0.0, c)], 96000 + 100 * (5 * li + ci) + r)
                                                             for r in range(10)])
            d2[f"lam{lam:.0f}_c{c}"] = {mm: float(np.mean([a[mm] for a in v])) for mm in v[0]}
            print("D2", f"lam{lam:.0f}_c{c}", json.dumps(d2[f"lam{lam:.0f}_c{c}"]), flush=True)
    e = lambda lam, c: d2[f"lam{lam:.0f}_c{c}"]["moment_elong"]
    res["D2"] = d2
    res["D2_criteria"] = {"c0.2_vs_c1_at_80": e(80, 0.2) - e(80, 1.0), "c1_minus_c0.2_at_20": e(20, 1.0) - e(20, 0.2),
                          "c0.2x80_vs_c1x20": e(80, 0.2) - e(20, 1.0)}
    cr = res["D2_criteria"]
    res["D2_saturation_supported"] = bool(abs(cr["c0.2_vs_c1_at_80"]) <= 0.05 and cr["c1_minus_c0.2_at_20"] >= 0.15
                                          and abs(cr["c0.2x80_vs_c1x20"]) <= 0.08)
    print("D2 criteria", json.dumps(cr), res["D2_saturation_supported"], flush=True)
    # ---- descriptive: Q3 at lam_F 20 (same fractions and seeds as C6-S7h) ----
    p20 = {**P_CHOSEN, "lam_F": 20.0}
    q3 = {}
    for i, fr in enumerate((0.0, 0.2, 0.5, 0.8, 1.0)):
        parts = ([("bulk", 1.0, None, 1.0)] if fr == 0 else [("low", 1.0, 0.0, 0.2)] if fr == 1
                 else [("low", fr, 0.0, 0.2), ("bulk", 1 - fr, None, 1.0)])
        v = gather(pool, ck.job_shape, f"Q3_{i}", [(MOD, p20, parts, 93000 + 100 * i + r) for r in range(20)])
        q3[str(fr)] = {mm: float(np.mean([a[mm] for a in v])) for mm in v[0]}
    y = [q3[str(fr)]["moment_elong"] for fr in (0.0, 0.2, 0.5, 0.8, 1.0)]
    res["Q3_lam20_descriptive"] = {"by_fraction": q3, "moment_elong": y,
                                   "monotone_fall": bool(all(a >= c for a, c in zip(y, y[1:])))}
    print("Q3 lam20", [round(v, 3) for v in y], flush=True)
    res["minutes"] = (time.time() - t0) / 60
    json.dump(res, open(os.path.join(OUT, "c6s7i_diag.json"), "w"), indent=1)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--part":
        ONLY = sys.argv[2]
        try:
            main()
        except _Done:
            pass
    else:
        main(smoke="--smoke" in sys.argv)
