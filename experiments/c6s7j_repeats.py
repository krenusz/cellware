"""C6-S7j: 30-seed repeats of the C6-S7h chosen setting (J1) and order of events, pole vs elongation (J2).
Synthetic only. See EXPERIMENT_LOG C6-S7j.

    uv run --with scikit-image --with numba --with matplotlib python c6s7j_repeats.py
"""

from __future__ import annotations

import json
import os
import time
from multiprocessing import Pool

import numpy as np

import c6s2_space as c2
import c6s3_break as b
import c6s7c_potts as pc
import c6s7f_potts_pcrawl as pf
import c6s7g_potts_wet as pg

OUT = c2.OUT
N = pc.N
P_CHOSEN = {**pc.BASE, **pg.W["W1"], "mcs": 100000, "beta_lr": 300.0, "lam_F": 80.0}
SNAP = 2000


def jobJ1(a):
    p, comp, seed = a
    v, _ = pg.job1((p, comp, seed))
    return v


def ci(x, rng):
    x = np.asarray(x, float)
    bs = [rng.choice(x, len(x)).mean() for _ in range(2000)]
    return [float(x.mean()), float(x.std(ddof=1)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def jobJ2(seed):
    rng = np.random.default_rng(seed)
    s = np.r_[2 + rng.standard_normal(N // 2), -2 + rng.standard_normal(N // 2)]
    pc.LAST.pop("traj", None)
    x, P, sh = pc.simulate(s, np.ones(N), {**P_CHOSEN, "snap": SNAP}, rng)
    tr = []
    for t, xy, pairs, el in pc.LAST["traj"]:
        tr.append((t, float(b.polar_index(xy, P)), int(pf.domains(P, pairs)), float(el)))
    return tr


def main():
    t0 = time.time()
    rng = np.random.default_rng(0)
    res = {"setting": {"W": "W1", "beta_lr": 300.0, "lam_F": 80.0, "mcs": 100000}}
    pool = Pool(10)
    # ---- J1 ----
    arms = {"crawl_c1": (P_CHOSEN, 1.0, 97000), "lam0": ({**P_CHOSEN, "lam_F": 0.0}, 1.0, 97100),
            "crawl_c0": (P_CHOSEN, 0.0, 97200)}
    j1 = {}
    for name, (p, comp, s0) in arms.items():
        r = pool.map(jobJ1, [(p, comp, s0 + k) for k in range(30)])
        el = [a["elong"] for a in r]
        j1[name] = {"polar": ci([a["polar"] for a in r], rng), "multipolar_rate": ci([a["multipolar"] for a in r], rng),
                    "attached": ci([a["attached"] for a in r], rng), "elong": ci(el, rng),
                    "AR_mean": float(np.mean([1 / (1 - e) for e in el])),
                    "single_run_S0a_fail": float(np.mean([(a["polar"] < 0.5) or a["multipolar"] for a in r])),
                    "raw": r}
        print("J1", name, {k: v for k, v in j1[name].items() if k != "raw"}, flush=True)
    m = {k: j1[k]["polar"][0] for k in j1}
    e = {k: j1[k]["elong"][0] for k in j1}
    crit = {"S0a": bool(m["crawl_c1"] >= 0.5 and j1["crawl_c1"]["multipolar_rate"][0] <= 0.2),
            "S0b": bool(e["crawl_c1"] - e["lam0"] >= 0.15), "S0c": bool(abs(e["crawl_c0"] - e["lam0"]) <= 0.05)}
    j1["criteria"] = crit
    j1["reading"] = "robust" if all(crit.values()) else "fragile"
    print("J1 reading", crit, j1["reading"], flush=True)
    res["J1"] = j1
    # ---- J2 ----
    trs = pool.map(jobJ2, [97500 + k for k in range(10)])
    runs = []
    for tr in trs:
        ts = [a[0] for a in tr]
        tp = next((a[0] for a in tr if a[1] >= 0.5 and a[2] == 1), None)
        e0, e1 = tr[0][3], tr[-1][3]
        te = next((a[0] for a in tr if a[3] >= e0 + 0.5 * (e1 - e0)), None) if e1 > e0 else None
        runs.append({"t_pole": tp, "t_elong": te, "traj": tr, "first": None if tp is None or te is None
                     else ("pole" if tp < te else "elong" if te < tp else "tie")})
    nf = sum(r["first"] == "pole" for r in runs)
    ne = sum(r["first"] == "elong" for r in runs)
    res["J2"] = {"runs": runs, "pole_first": nf, "elong_first": ne,
                 "reading": "pole first" if nf >= 8 else "elongation first" if ne >= 8 else "mixed"}
    print("J2", [(r["t_pole"], r["t_elong"]) for r in runs], res["J2"]["reading"], flush=True)
    picture(runs, os.path.join(OUT, "c6s7j_order.png"))
    res["minutes"] = (time.time() - t0) / 60
    json.dump(res, open(os.path.join(OUT, "c6s7j_repeats.json"), "w"), indent=1)


def picture(runs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    for r in runs:
        t = np.array([a[0] for a in r["traj"]]) / 1000
        ax[0].plot(t, [a[1] for a in r["traj"]], color="tab:orange", alpha=0.5)
        ax[1].plot(t, [a[3] for a in r["traj"]], color="tab:blue", alpha=0.5)
    ax[0].axhline(0.5, ls="--", color="grey")
    ax[0].set(xlabel="MCS (thousands)", ylabel="polar index", title="pole")
    ax[1].set(xlabel="MCS (thousands)", ylabel="moment elongation", title="elongation")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
