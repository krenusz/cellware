"""C6-S7l: crawling after onset, all-cell random (a) vs mediolateral P-only (b) / all-cell (c); axis alignment.
See EXPERIMENT_LOG C6-S7l.
Runs as chunks on GitHub Actions (--part TAG), merged locally (no args = stage 1 merge, --stage 2 = data checks).

    python c6s7l_axis.py --plan 1          # print chunk tags for the sweep file
    python c6s7l_axis.py --part TAG [--stage 2]
    python c6s7l_axis.py [--stage 2]         # merge (computes any missing chunk locally)
"""

from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.stats import norm, spearmanr

import c6s2_space as c2
import c6s3_break as b
import c6s4_fit as f4
import c6s7_checks as ck
import c6s7c_potts as pc
import c6s7f_potts_pcrawl as pf
import c6s7g_potts_wet as pg

OUT = c2.OUT
MOD = "c6s7l_axis"
N = pc.N
simulate = pg.simulate
multipolar = pg.multipolar
BASE_K = {**pc.BASE, **pg.W["W1"], "beta_lr": 300.0, "mcs": 250000}
ARMS = {"a": {}, "b": {"crawl": "P", "orient": "ml"}, "c": {"orient": "ml"}}
POINTS = ([("a", t, l) for t in (100000, 150000) for l in (20.0, 40.0)] + [("b", 100000, l) for l in (20.0, 40.0, 80.0)]
          + [("c", 100000, l) for l in (20.0, 40.0)])
SNAP = 5000
PARTS = os.path.join(OUT, "c6s7l_parts")
S1_JSON = os.path.join(OUT, "c6s7l_step0.json")
MODE = {"only": None, "plan": None}


class _Done(Exception):
    pass


class _Lazy:
    p = None

    def map(self, f, a):
        if self.p is None:
            self.p = Pool(int(os.environ.get("NPROC", "10")))  # worker count does not change results
        return self.p.map(f, a)


POOL = _Lazy()


def gather(f, tag, args, ch):
    out = []
    for k in range(0, len(args), ch):
        name = f"{tag}_{k}"
        fp = os.path.join(PARTS, name + ".json")
        if MODE["plan"] is not None:
            MODE["plan"].append(name)
            continue
        if MODE["only"] is not None and name != MODE["only"]:
            continue
        if os.path.exists(fp):
            out += json.load(open(fp))
            continue
        r = POOL.map(f, args[k:k + ch])
        if MODE["only"] is not None:
            os.makedirs(PARTS, exist_ok=True)
            json.dump(r, open(fp, "w"), default=_np)
            raise _Done
        out += r
    return out


def _np(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    raise TypeError(type(o))


# ---------------- stage 1: step 0 ----------------

def job_s0(a):
    p, comp, seed, snap = a
    rng = np.random.default_rng(seed)
    s = np.r_[2 + rng.standard_normal(N // 2), -2 + rng.standard_normal(N // 2)]
    pc.LAST.pop("traj", None)
    x, P, sh = simulate(s, np.full(N, comp), {**p, "snap": snap} if snap else p, rng)
    out = {"polar": float(b.polar_index(x, P)), "multipolar": bool(multipolar(x, P)), "attached": sh["attached"],
           "elong": sh["moment_elong"], "align": align(x, P)}
    if snap:
        tr = [(t, float(b.polar_index(xy, P)), int(pf.domains(P, pairs)), float(el))
              for t, xy, pairs, el in pc.LAST["traj"]]
        tp = next((a[0] for a in tr if a[1] >= 0.5 and a[2] == 1), None)
        e0, e1 = tr[0][3], tr[-1][3]
        te = next((a[0] for a in tr if a[3] >= e0 + 0.5 * (e1 - e0)), None) if e1 > e0 else None
        out.update({"t_pole": tp, "t_elong": te, "traj": tr})
    return out


def align(x, P):
    """|cos| between the mask's long axis and the pole direction (P centroid - A centroid)."""
    pts = np.argwhere(pc.LAST["lat"] > 0)[:, ::-1].astype(float)  # (x, y) like the cell centroids
    w, v = np.linalg.eigh(np.cov(pts.T))
    d = x[P].mean(0) - x[~P].mean(0)
    return float(abs(v[:, -1] @ d) / (np.linalg.norm(d) + 1e-9))


def pparams(arm, t_on, lam):
    return {**BASE_K, **ARMS[arm], "t_on": t_on, "lam_F": lam}


def collect1():
    raw = {"lam0": gather(job_s0, "S0_lam0", [({**BASE_K, "lam_F": 0.0}, 1.0, 99900 + r, 0) for r in range(10)], 5)}
    for i, (arm, t_on, lam) in enumerate(POINTS):
        p = pparams(arm, t_on, lam)
        raw[f"{i}_c1"] = gather(job_s0, f"S0_{i}_c1", [(p, 1.0, 99000 + 50 * i + r, SNAP) for r in range(10)], 5)
        raw[f"{i}_c0"] = gather(job_s0, f"S0_{i}_c0", [(p, 0.0, 99500 + 30 * i + r, 0) for r in range(10)], 5)
    return raw


def mean(r, k):
    return float(np.mean([a[k] for a in r]))


def stage1():
    raw = collect1()
    lam0 = {k: mean(raw["lam0"], k) for k in ("polar", "multipolar", "attached", "elong", "align")}
    res = {"lam0": lam0, "points": {}}
    passing = []
    for i, (arm, t_on, lam) in enumerate(POINTS):
        r1, r0 = raw[f"{i}_c1"], raw[f"{i}_c0"]
        v = {k: mean(r1, k) for k in ("polar", "multipolar", "attached", "elong", "align")}
        v["elong_c0"] = mean(r0, "elong")
        pole_first = sum(a["t_pole"] is not None and a["t_elong"] is not None and a["t_pole"] < a["t_elong"]
                         for a in r1)
        v.update({"pole_first": int(pole_first), "t_pole": [a["t_pole"] for a in r1],
                  "t_elong": [a["t_elong"] for a in r1]})
        v["S0a"] = bool(v["polar"] >= 0.5 and v["multipolar"] <= 0.2 and v["attached"] >= 0.95)
        v["S0b"] = bool(v["elong"] - lam0["elong"] >= 0.15)
        v["S0c"] = bool(abs(v["elong_c0"] - lam0["elong"]) <= 0.05)
        v["S0e"] = bool(pole_first >= 7)
        v["S0f"] = bool(v["align"] >= 0.8)
        v["pass"] = v["S0a"] and v["S0b"] and v["S0c"] and v["S0e"] and v["S0f"]
        res["points"][f"{arm}_t{t_on // 1000}k_lam{lam:.0f}"] = v
        if v["pass"]:
            passing.append((lam, arm, t_on))
        print("step0", arm, t_on, lam, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()
                                   if k not in ("t_pole", "t_elong")}, flush=True)
    res["passing"] = passing
    if passing:
        lam, arm, t_on = min(passing)
        res["chosen"] = {"arm": arm, "t_on": t_on, "lam_F": lam}
        print("chosen", res["chosen"], flush=True)
    else:
        res["note"] = "no step-0 point passes (pre-registered); no further amendment without Bence"
        print(res["note"], flush=True)
    picture(raw, res, os.path.join(OUT, "c6s7l_step0.png"))
    json.dump(res, open(S1_JSON, "w"), indent=1, default=_np)


def picture(raw, res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = list(res["points"])
    fig, axs = plt.subplots(2, len(keys), figsize=(2.6 * len(keys), 5), sharey="row")
    for j, (i, key) in enumerate(zip(range(len(POINTS)), keys)):
        for a in raw[f"{i}_c1"]:
            t = np.array([q[0] for q in a["traj"]]) / 1000
            axs[0, j].plot(t, [q[1] for q in a["traj"]], color="tab:orange", alpha=0.5)
            axs[1, j].plot(t, [q[3] for q in a["traj"]], color="tab:blue", alpha=0.5)
        axs[0, j].axhline(0.5, ls="--", color="grey")
        axs[0, j].axvline(POINTS[i][1] / 1000, ls=":", color="k")
        axs[1, j].axvline(POINTS[i][1] / 1000, ls=":", color="k")
        axs[0, j].set_title(f"{key}\npole first {res['points'][key]['pole_first']}/10", fontsize=8)
        axs[1, j].set_xlabel("MCS (k)")
    axs[0, 0].set_ylabel("polar index")
    axs[1, 0].set_ylabel("elongation")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


# ---------------- stage 2: data checks at the chosen point (logic as c6s7_checks.downstream) ----------------

def chosen_p():
    ch = json.load(open(S1_JSON))["chosen"]
    return pparams(ch["arm"], ch["t_on"], ch["lam_F"])


def collect2():
    p = chosen_p()
    raw = {}
    for ki, k in enumerate(f4.KS):
        raw[f"Q1_{k}"] = gather(ck.job_clone, f"Q1_{k}", [(MOD, p, k, 91000 + 100 * ki + r) for r in range(40)], 10)
    for i, fr in enumerate((0.2, 1 / 3, 0.5, 2 / 3, 0.8)):
        raw[f"Q2_{i}"] = gather(ck.job_q2, f"Q2_{i}", [(MOD, p, fr, 92000 + 100 * i + r) for r in range(20)], 10)
    for i, fr in enumerate((0.0, 0.2, 0.5, 0.8, 1.0)):
        parts = ([("bulk", 1.0, None, 1.0)] if fr == 0 else [("low", 1.0, 0.0, 0.2)] if fr == 1
                 else [("low", fr, 0.0, 0.2), ("bulk", 1 - fr, None, 1.0)])
        raw[f"Q3_{i}"] = gather(ck.job_shape, f"Q3_{i}", [(MOD, p, parts, 93000 + 100 * i + r) for r in range(20)], 10)
    for name, z6 in (("z0", 0.0), ("z_clone6", c2.Z["6"])):
        parts = {"bulk": [("bulk", 1.0, None, 1.0)], "6a": [("6a", 1.0, z6, 0.2)],
                 "mix": [("6a", 0.5, z6, 0.2), ("bulk", 0.5, None, 1.0)]}
        for j, (key, pp) in enumerate(parts.items()):
            raw[f"S2_{name}_{key}"] = gather(ck.job_shape, f"S2_{name}_{key}",
                                             [(MOD, p, pp, 94000 + 1000 * j + 5000 * (name != "z0") + r)
                                              for r in range(100)], 10)
    return raw


def stage2():
    raw = collect2()
    res = {"chosen": json.load(open(S1_JSON))["chosen"]}
    ks = f4.KS
    sim = {k: float(np.mean([a[0] for a in raw[f"Q1_{k}"]])) for k in ks}
    signs = int(sum(np.sign(sim[k]) == np.sign(f4.TGT[k]) for k in ks))
    rho = float(spearmanr([sim[k] for k in ks], [f4.TGT[k] for k in ks]).correlation)
    res["Q1"] = {"sim": sim, "target": f4.TGT, "signs": signs, "spearman": rho,
                 "multipolar_rate": {k: float(np.mean([a[1] for a in raw[f"Q1_{k}"]])) for k in ks},
                 "polar": {k: float(np.mean([a[2] for a in raw[f"Q1_{k}"]])) for k in ks},
                 "pass": bool(signs == 5 and rho >= 0.9 - 1e-9)}
    xs = (np.arange(c2.N_BINS) + 0.5) * 100 / c2.N_BINS
    q2 = []
    for i, fr in enumerate((0.2, 1 / 3, 0.5, 2 / 3, 0.8)):
        v = raw[f"Q2_{i}"]
        q2.append({"f_anterior": fr, "crossing": float(c2.crossing(xs, np.mean([np.asarray(a[0]) for a in v], 0),
                                                                     np.mean([np.asarray(a[1]) for a in v], 0)))})
    ok = [o for o in q2 if not np.isnan(o["crossing"])]
    sl = float(np.polyfit([100 * o["f_anterior"] for o in ok], [o["crossing"] for o in ok], 1)[0]) if len(ok) >= 3 \
        else float("nan")
    lo, hi = c2.G1B["slope_ci"]
    res["Q2"] = {"crossings": q2, "slope": sl, "g1b_ci": [lo, hi], "pass": bool(lo <= sl <= hi)}
    q3 = {str(fr): {mm: float(np.mean([a[mm] for a in raw[f"Q3_{i}"]])) for mm in raw[f"Q3_{i}"][0]}
          for i, fr in enumerate((0.0, 0.2, 0.5, 0.8, 1.0))}
    y = [q3[str(fr)]["moment_elong"] for fr in (0.0, 0.2, 0.5, 0.8, 1.0)]
    res["Q3"] = {"by_fraction": q3, "moment_elong": y, "monotone_fall": bool(all(a >= c for a, c in zip(y, y[1:])))}
    s2 = {}
    for name in ("z0", "z_clone6"):
        m = {key: float(np.mean([a["one_minus_circ"] for a in raw[f"S2_{name}_{key}"]])) for key in ("bulk", "6a", "mix")}
        gap = m["bulk"] - m["6a"]
        frac = (m["mix"] - 0.5 * (m["bulk"] + m["6a"])) / gap if gap > 0 else float("nan")
        eff = frac * f4.PURE_GAP_DATA
        s2[name] = {**m, "pure_gap_model": gap, "frac_of_gap": frac, "effect_data_units": eff,
                    "power": float(norm.sf(norm.ppf(0.95) - eff / f4.S2_SE)) if not np.isnan(eff) else 0.0}
    robust = all(s2[n]["power"] >= 0.8 for n in s2)
    res["S2"] = {"by_state": s2, "robust_powered": bool(robust),
                 "H17_label": "powered (S2 look proposable to Bence)" if robust else "exploratory"}
    for k in ("Q1", "Q2", "Q3", "S2"):
        print(k, json.dumps(res[k], default=_np)[:900], flush=True)
    json.dump(res, open(os.path.join(OUT, "c6s7l_checks.json"), "w"), indent=1, default=_np)


if __name__ == "__main__":
    a = sys.argv[1:]
    stage = 2 if "--stage" in a and a[a.index("--stage") + 1] == "2" else 1
    if "--plan" in a:
        MODE["plan"] = []
        (collect1 if a[a.index("--plan") + 1] == "1" else collect2)()
        print("\n".join(MODE["plan"]))
    elif "--part" in a:
        MODE["only"] = a[a.index("--part") + 1]
        try:
            (collect2 if MODE["only"][:2] in ("Q1", "Q2", "Q3", "S2") else collect1)()
        except _Done:
            pass
    else:
        (stage2 if stage == 2 else stage1)()
