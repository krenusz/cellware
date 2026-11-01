"""E9: environment vs inheritance on the Weinreb 2020 cytokine-perturbation experiment.

The E6 lineage model (fit_larry.py) with cells split at day 2 into three conditions (full cocktail,
G-CSF, M-CSF); each condition adds a drift bias b_c to the fate state after the split
(b_full = 0). Fitted on within-condition statistics only; `--test` makes the single look at the
cross-condition sister concordance (EXPERIMENT_LOG.md, E9).

    uv run python experiments/fit_cytokine.py --memory resample
    uv run python experiments/fit_cytokine.py --test      # after both fits exist
"""

import argparse
import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from fit_larry import THRESH, distance  # noqa: E402
from larry_stats import fit_stats, test_stats  # noqa: E402

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "larry_cytokine")
OUT = os.path.join(HERE, "results_larry")
CONDS = {"full cocktail": "full", "G-csf": "G", "M-csf": "M"}
STEPS_PER_DAY = 20
# day-4 vs day-6 correlation is fitted only where enough clones have cells on both days
# (full cocktail: 4 clones, G-CSF: 39, M-CSF: 131)
TIME_CONDS = {"M"}
PARAMS = [
    ("k", 0.2, 5.0), ("sigma", 0.05, 1.5), ("s0", 0.0, 1.5), ("s_mem", 0.0, 1.0),
    ("theta", -2.0, 2.0), ("f", 0.02, 0.6), ("q", 0.0, 1.0),
    ("bG", -1.5, 1.5), ("bM", -1.5, 1.5),
]


# ----------------------------------------------------------------------------- data
def load_counts():
    """{cond: {day: {clone: [N, M, O]}}} for days 4 and 6, mature cells only."""
    with gzip.open(os.path.join(DATA, "metadata.txt.gz"), "rt") as f:
        rows = [line.rstrip("\n").split("\t") for line in f][1:]
    with gzip.open(os.path.join(DATA, "clone_matrix.mtx.gz"), "rt") as f:
        m = np.loadtxt(f, comments="%", skiprows=3, dtype=int)  # clones x cells
    out = {c: {4: {}, 6: {}} for c in CONDS.values()}
    for clone, cell in zip(m[:, 0] - 1, m[:, 1] - 1, strict=True):
        day, kind, cond = rows[cell][2], rows[cell][3], rows[cell][4]
        if day == "2" or kind == "Undifferentiated" or cond not in CONDS:
            continue
        j = 0 if kind == "Neu" else 1 if kind == "Mo" else 2
        out[CONDS[cond]][int(day)].setdefault(clone, [0, 0, 0])[j] += 1
    return out


def pooled(by_day):
    tot = {}
    for d in by_day.values():
        for c, v in d.items():
            t = tot.setdefault(c, [0, 0, 0])
            for j in range(3):
                t[j] += v[j]
    return tot


def condition_stats(cnt):
    """Within-condition fit statistics, keyed by condition."""
    stats = {}
    for c, by_day in cnt.items():
        s = fit_stats(pooled(by_day))
        s["r_d4_d6"] = test_stats(by_day[4], by_day[6], boot=1)["r"]
        stats[c] = s
    return stats


def cross_stats(cnt, boot=2000):
    pairs = [("full", "G"), ("full", "M"), ("G", "M")]
    return {f"{a}-{b}": test_stats(pooled(cnt[a]), pooled(cnt[b]), boot=boot) for a, b in pairs}


# ------------------------------------------------------------------------ simulator
def simulate(p, memory, clones=1500, seed=0):
    rng = np.random.default_rng(seed)
    k, sigma, s0, s_mem, theta, f, q, b_g, b_m = (p[n] for n, _, _ in PARAMS)
    bias = np.array([0.0, b_g, b_m])  # full, G, M
    shape = (clones, 1)
    m, o = rng.normal(0, s0, shape), rng.normal(0, s0, shape)
    x = rng.normal(0, 0.05, shape)
    fate = np.zeros(shape, dtype=np.int8)  # -1 absent, 0 undecided, 1 N, 2 M, 3 O
    cond = np.full(shape, -1, dtype=np.int8)  # -1 before the split
    dt = 1.0 / STEPS_PER_DAY
    out = {c: {4: {}, 6: {}} for c in ("full", "G", "M")}
    for day in range(1, 7):
        b = np.where(cond >= 0, bias[cond.clip(0)], 0.0)
        for _ in range(STEPS_PER_DAY):
            free = fate == 0
            x = np.where(free, x + k * (x - x**3 + m + b) * dt
                         + sigma * np.sqrt(dt) * rng.standard_normal(x.shape), x)
            done = free & (np.abs(x) > THRESH)
            is_o = rng.random(x.shape) < 1 / (1 + np.exp(-4 * (o - theta)))
            fate = np.where(done, np.where(is_o, 3, np.where(x > 0, 1, 2)), fate).astype(np.int8)
        if day in (4, 6):
            pick = (fate > 0) & (rng.random(fate.shape) < f)
            for ci, cname in enumerate(("full", "G", "M")):
                sel = pick & (cond == ci)
                for j, fv in enumerate((1, 2, 3)):
                    cnt = ((fate == fv) & sel).sum(1)
                    for cl in np.nonzero(cnt)[0]:
                        out[cname][day].setdefault(cl, [0, 0, 0])[j] += int(cnt[cl])
        if day < 6:
            x, fate, cond = (np.repeat(v, 2, axis=1) for v in (x, fate, cond))
            m, o = np.repeat(m, 2, axis=1), np.repeat(o, 2, axis=1)
            if q < 1.0:
                second = fate[:, 1::2]
                second[(second > 0) & (rng.random(second.shape) >= q)] = -1
            if memory == "inherit":
                m = m + rng.normal(0, s_mem, m.shape)
                o = o + rng.normal(0, s_mem, o.shape)
            else:  # resample
                m, o = rng.normal(0, s0, m.shape), rng.normal(0, s0, o.shape)
            if day == 2:
                cond = rng.integers(0, 3, cond.shape).astype(np.int8)
    return out


# --------------------------------------------------------------------------- fitting
def loss(vec, memory, obs, seeds=(0, 1)):
    p = {n: float(np.clip(v, lo, hi)) for v, (n, lo, hi) in zip(vec, PARAMS, strict=True)}
    total = 0.0
    for s in seeds:
        sim = condition_stats(simulate(p, memory, seed=s))
        for c in obs:
            if sim[c]["clones"] < 30:
                return 10.0
            total += distance(sim[c], obs[c])
            if c in TIME_CONDS:
                total += (sim[c]["r_d4_d6"] - obs[c]["r_d4_d6"]) ** 2
    return total / len(seeds)


def fit(memory, obs, starts, maxfev):
    from scipy.optimize import minimize

    best, best_l = None, np.inf
    for i, st in enumerate(starts):
        v0 = np.array([st.get(n, (lo + hi) / 2) for n, lo, hi in PARAMS])
        res = minimize(loss, v0, args=(memory, obs), method="Nelder-Mead",
                       options={"maxfev": maxfev, "xatol": 1e-3, "fatol": 1e-5})
        print(f"  start {i}: {res.fun:.4f}", flush=True)
        if res.fun < best_l:
            best, best_l = res.x, res.fun
    lo, hi = [q[1] for q in PARAMS], [q[2] for q in PARAMS]
    best = np.clip(best, lo, hi)
    return {n: float(v) for v, (n, _, _) in zip(best, PARAMS, strict=True)}, float(best_l)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory", choices=["resample", "inherit"], default="resample")
    ap.add_argument("--maxfev", type=int, default=200)
    ap.add_argument("--test", action="store_true", help="the single look at the held-out test")
    args = ap.parse_args()
    cnt = load_counts()
    obs = condition_stats(cnt)
    json.dump(obs, open(os.path.join(OUT, "cytokine_fit.json"), "w"), indent=1)
    if args.test:
        result = {"observed": cross_stats(cnt)}
        for mem in ("resample", "inherit"):
            r = json.load(open(os.path.join(OUT, f"fit_v7_{mem}.json")))
            result[mem] = r["pred"]
        json.dump(result, open(os.path.join(OUT, "cytokine_test.json"), "w"), indent=1)
        print(json.dumps(result, indent=1))
        return
    print("fit set:", {c: s["clones"] for c, s in obs.items()})
    v6 = json.load(open(os.path.join(OUT, f"fit_v6_{args.memory}.json")))["params"]
    starts = [{**v6, "bG": 0.3, "bM": -0.3}, {**v6, "bG": 0.0, "bM": 0.0},
              {**v6, "bG": 0.8, "bM": -0.8}]
    params, best = fit(args.memory, obs, starts, args.maxfev)
    sims = [simulate(params, args.memory, seed=s) for s in range(10, 15)]
    pred = {}
    for key in ("full-G", "full-M", "G-M"):
        vals = [cross_stats(c, boot=1)[key]["r"] for c in sims]
        pred[key] = {"r_mean": float(np.mean(vals)), "r_range": [min(vals), max(vals)]}
    result = {"memory": args.memory, "params": params, "loss": best,
              "fit_sim": condition_stats(sims[0]), "pred": pred}
    json.dump(result, open(os.path.join(OUT, f"fit_v7_{args.memory}.json"), "w"), indent=1)
    print(json.dumps({k: result[k] for k in ("memory", "params", "loss")}, indent=1))


if __name__ == "__main__":
    main()
