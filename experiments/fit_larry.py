"""E6: fit a non-spatial cellware lineage model to LARRY clone statistics (well 1 only).

Founder cells divide once a day for 6 days. Each cell has a fate state x on the pitchfork
landscape dx = k (x - x^3 + m) dt + sigma dW and commits at |x| > 0.7 (x > 0: N, x < 0: M);
at commitment it becomes O instead with probability sigmoid(4 (o - theta)). Memory (m, o) is drawn
per founder with spread s0 and copied to daughters with noise s_mem (`--memory resample` draws it
afresh at every division instead; `--memory decay` keeps a share a of it and refills
the rest; `--memory none` sets m = o = 0 with only theta for O).
At day 2 cells are split at random into two wells. Mature cells are sampled with probability f
at days 4 and 6.

    uv run python experiments/fit_larry.py --memory inherit --out experiments/results_larry
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from larry_stats import fit_stats, test_stats  # noqa: E402

STEPS_PER_DAY = 20
THRESH = 0.7
# name, lower, upper
PARAMS = [
    ("k", 0.2, 5.0),        # drift speed of the fate state
    ("sigma", 0.05, 1.5),   # fate-state noise
    ("s0", 0.0, 1.5),       # founder memory spread
    ("s_mem", 0.0, 1.0),    # memory noise at division
    ("theta", -2.0, 2.0),   # O threshold
    ("f", 0.02, 0.6),       # sampling probability per mature cell
    ("a", 0.0, 1.0),        # memory carried over per division (--memory decay only)
    ("eps", 0.0, 0.25),     # N/M label swap at annotation (--label_noise only)
    ("q", 0.0, 1.0),        # chance a committed cell still divides (--late only)
]


LABEL_NOISE = False
USE_TIME = False  # v5: also fit the day-4 vs day-6 fate-bias correlation within well 1
LATE = False  # v6: committed cells divide only with probability q


def simulate(p, memory, clones=1500, seed=0):
    if not LABEL_NOISE:
        p = {**p, "eps": 0.0}
    p = {**p, "q": p.get("q", 1.0) if LATE else 1.0}
    rng = np.random.default_rng(seed)
    k, sigma, s0, s_mem, theta, f, a, eps, q = (p[n] for n, _, _ in PARAMS)
    if memory == "none":
        s0 = s_mem = 0.0
    shape = (clones, 1)
    m = rng.normal(0, s0, shape)
    o = rng.normal(0, s0, shape)
    x = rng.normal(0, 0.05, shape)
    fate = np.zeros(shape, dtype=np.int8)  # -1 absent, 0 undecided, 1 N, 2 M, 3 O
    well = np.zeros(shape, dtype=np.int8)
    dt = 1.0 / STEPS_PER_DAY
    samples = []
    for day in range(1, 7):
        for _ in range(STEPS_PER_DAY):
            free = fate == 0
            x = np.where(free, x + k * (x - x**3 + m) * dt
                         + sigma * np.sqrt(dt) * rng.standard_normal(x.shape), x)
            done = free & (np.abs(x) > THRESH)
            is_o = rng.random(x.shape) < 1 / (1 + np.exp(-4 * (o - theta)))
            fate = np.where(done, np.where(is_o, 3, np.where(x > 0, 1, 2)), fate).astype(np.int8)
        if day in (4, 6):
            pick = (fate > 0) & (rng.random(fate.shape) < f)
            seen = fate.copy()
            swap = (seen < 3) & (seen > 0) & (rng.random(seen.shape) < eps)
            seen[swap] = 3 - seen[swap]  # 1 <-> 2
            samples.append((seen, well.copy(), pick))
        if day < 6:  # divide
            x, fate, well = (np.repeat(v, 2, axis=1) for v in (x, fate, well))
            if q < 1.0:  # second daughter of a committed cell exists only with probability q
                second = fate[:, 1::2]
                skip = (second > 0) & (rng.random(second.shape) >= q)
                second[skip] = -1
            m, o = np.repeat(m, 2, axis=1), np.repeat(o, 2, axis=1)
            if memory == "inherit":
                m = m + rng.normal(0, s_mem, m.shape)
                o = o + rng.normal(0, s_mem, o.shape)
            elif memory == "decay":  # keep a share of the parent's memory, refill the variance
                keep, fresh = a, np.sqrt(max(1 - a * a, 0.0)) * s0
                m = keep * m + rng.normal(0, 1, m.shape) * fresh
                o = keep * o + rng.normal(0, 1, o.shape) * fresh
            elif memory == "resample":
                m, o = rng.normal(0, s0, m.shape), rng.normal(0, s0, o.shape)
            if day == 2:
                well = rng.integers(1, 3, well.shape).astype(np.int8)
    cnt = {1: {}, 2: {}, "d4": {}, "d6": {}}
    for (fate_s, well_s, pick), tag in zip(samples, ("d4", "d6"), strict=True):
        for w in (1, 2, tag):
            sel = pick & (well_s == (1 if isinstance(w, str) else w))
            for j, name in enumerate((1, 2, 3)):
                c = ((fate_s == name) & sel).sum(1)
                for cl in np.nonzero(c)[0]:
                    cnt[w].setdefault(cl, [0, 0, 0])[j] += int(c[cl])
    return cnt


def distance(sim, obs):
    keys = ["n_only", "m_only", "mixed_nm", "has_o", "o_share"]
    d = sum((sim[q] - obs[q]) ** 2 for q in keys)
    hs, ho = np.array(sim["nfrac_hist"], float), np.array(obs["nfrac_hist"], float)
    d += np.sum((hs / hs.sum() - ho / ho.sum()) ** 2)
    d += ((np.log(sim["mean_size"]) - np.log(obs["mean_size"])) / 2) ** 2
    if USE_TIME and "r_d4_d6" in sim:
        d += (sim["r_d4_d6"] - obs["r_d4_d6"]) ** 2
    return float(d)


def loss(vec, memory, obs, seeds=(0, 1)):
    p = {n: float(np.clip(v, lo, hi)) for v, (n, lo, hi) in zip(vec, PARAMS, strict=True)}
    total = 0.0
    for s in seeds:
        cnt = simulate(p, memory, seed=s)
        if len([v for v in cnt[1].values() if sum(v) >= 2]) < 50:
            return 10.0
        sim = fit_stats(cnt[1])
        if USE_TIME:
            sim["r_d4_d6"] = test_stats(cnt["d4"], cnt["d6"], boot=1)["r"]
        total += distance(sim, obs)
    return total / len(seeds)


def search(memory, obs, n_random, n_local, seed):
    rng = np.random.default_rng(seed)
    lo = np.array([q[1] for q in PARAMS])
    hi = np.array([q[2] for q in PARAMS])
    best, best_l = None, np.inf
    for i in range(n_random):
        v = lo + rng.random(len(PARAMS)) * (hi - lo)
        lv = loss(v, memory, obs)
        if lv < best_l:
            best, best_l = v, lv
            print(f"  random {i}: {best_l:.4f}", flush=True)
    step = (hi - lo) * 0.15
    for i in range(n_local):  # simple adaptive local search
        v = np.clip(best + rng.normal(0, 1, len(PARAMS)) * step, lo, hi)
        lv = loss(v, memory, obs)
        if lv < best_l:
            best, best_l = v, lv
            print(f"  local {i}: {best_l:.4f}", flush=True)
        elif i % 20 == 19:
            step *= 0.8
    return {n: float(v) for v, (n, _, _) in zip(best, PARAMS, strict=True)}, best_l


def refine(memory, obs, starts, maxfev):
    """Nelder-Mead from each start (a dict of params); returns the best."""
    from scipy.optimize import minimize

    best, best_l = None, np.inf
    for i, st in enumerate(starts):
        v0 = np.array([st.get(n, (lo + hi) / 2) for n, lo, hi in PARAMS])
        res = minimize(loss, v0, args=(memory, obs), method="Nelder-Mead",
                       options={"maxfev": maxfev, "xatol": 1e-3, "fatol": 1e-5})
        print(f"  start {i}: {res.fun:.4f}", flush=True)
        if res.fun < best_l:
            best, best_l = res.x, res.fun
    lo = [q[1] for q in PARAMS]
    hi = [q[2] for q in PARAMS]
    best = np.clip(best, lo, hi)
    return {n: float(v) for v, (n, _, _) in zip(best, PARAMS, strict=True)}, float(best_l)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory", choices=["inherit", "decay", "resample", "none"], default="inherit")
    ap.add_argument("--random", type=int, default=150)
    ap.add_argument("--local", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--out", default="experiments/results_larry")
    ap.add_argument("--starts", nargs="*", default=[],
                    help="earlier fit JSONs whose params seed Nelder-Mead (with --random starts)")
    ap.add_argument("--maxfev", type=int, default=300)
    ap.add_argument("--label_noise", action="store_true", help="fit an N/M annotation error")
    ap.add_argument("--use_time", action="store_true", help="fit day-4 vs day-6 correlation")
    ap.add_argument("--late", action="store_true", help="fit q, division of committed cells")
    args = ap.parse_args()
    global LABEL_NOISE, USE_TIME, LATE
    LABEL_NOISE, USE_TIME, LATE = args.label_noise, args.use_time, args.late
    obs = json.load(open(os.path.join(args.out, "larry_fit.json")))
    if args.starts:
        rng = np.random.default_rng(args.seed)
        starts = [json.load(open(f))["params"] for f in args.starts]
        starts += [{n: lo + rng.random() * (hi - lo) for n, lo, hi in PARAMS}
                   for _ in range(args.random)]
        params, best = refine(args.memory, obs, starts, args.maxfev)
    else:
        params, best = search(args.memory, obs, args.random, args.local, args.seed)
    sims = [simulate(params, args.memory, seed=s) for s in range(10, 15)]
    fit = [fit_stats(c[1]) | {"r_d4_d6": test_stats(c["d4"], c["d6"], boot=1)["r"]}
           for c in sims]
    pred = [test_stats(c[1], c[2], boot=200) for c in sims]
    result = {
        "memory": args.memory, "tag": args.tag, "params": params, "loss": best,
        "fit_sim": fit[0], "fit_obs": obs,
        "pred_r": [q["r"] for q in pred], "pred_agree": [q["agree"] for q in pred],
    }
    path = os.path.join(args.out, f"fit_{args.tag}_{args.memory}.json")
    json.dump(result, open(path, "w"), indent=1)
    print(json.dumps({k: result[k] for k in ("memory", "params", "loss", "pred_r")}, indent=1))


if __name__ == "__main__":
    main()
