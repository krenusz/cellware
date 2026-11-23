"""CAND-I-E1: exploratory wider-grid look at the train fish (pre-registered in EXPERIMENT_LOG, 2026-11-23).

Bence: "Exploratory wider grid". Train fish only, through candi_train.load_train (partitions.train_only). The drift grid
is widened, and the synthetic gate is rebuilt from each fish's own fitted noise, drift and clone frequencies. A second
noise fit treats replicate zeros as missing. Exploratory: this step supports no claim.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import candi_train as c  # noqa: E402

OUT = os.path.join(HERE, "results_cand", "candi_explore.json")
SEED = 20261125
c.NE_GRID = np.logspace(-0.5, 4, 19)
GATE_COHORTS, LINK = 200, 0.8
MAX_FALSE_POS, MIN_RECOVERY = 0.10, 0.5


def nonzero(reps):
    out = reps.copy()
    out[out == 0] = np.nan
    return out


def arm(X, x0, obs, noise, rng, n_cal=c.N_CAL, n_null=c.N_NULL):
    return c.null_excess(X, x0, obs, noise, rng, n_cal, n_null)


def fit_fish(reps, depth, rng):
    X = np.nanmean(reps, axis=2)
    obs = c.score(X)[0]
    if not np.isfinite(obs):
        return None
    try:
        a, b = c.fit_noise(reps)
        a2, b2 = c.fit_noise(nonzero(reps))
    except ValueError:
        return None
    r = np.sum(np.isfinite(reps[0]), axis=1).clip(1)
    x0 = np.clip(X.mean(axis=1), 1e-9, None)
    x0 = x0 / x0.sum()
    cal = arm(X, x0, obs, lambda y, g: c.noise_calibrated(y, a, b, r, g), rng)
    sens = arm(X, x0, obs, lambda y, g: c.noise_calibrated(y, a2, b2, r, g), rng)
    mul = arm(X, x0, obs, lambda y, g: c.noise_multinomial(y, depth, g), rng)
    edge = {float(c.NE_GRID[0]), float(c.NE_GRID[-1])}
    row = {"clones": int((X.mean(axis=1) >= c.MIN_SIZE).sum()), "samples": int(X.shape[1]), "rho": obs, "noise_a": a,
           "noise_b": b, "noise_a_nonzero": a2, "noise_b_nonzero": b2, "excess_calibrated": cal[0], "ne_calibrated": cal[1],
           "excess_nonzero": sens[0], "ne_nonzero": sens[1], "excess_multinomial": mul[0], "ne_multinomial": mul[1],
           "ne_at_grid_edge": cal[1] in edge or sens[1] in edge or mul[1] in edge}
    return row, {"x0": x0, "t": X.shape[1], "a": a, "b": b, "ne": cal[1], "r": r}


def gate(params, rng):
    rates = {}
    edge = {float(c.NE_GRID[0]), float(c.NE_GRID[-1])}
    for label, extra in (("false_positive", 0.0), ("recovery", LINK)):
        hits = 0
        for _ in range(GATE_COHORTS):
            vals = []
            for p in params:
                X = c.drift(p["x0"], p["ne"], p["t"], rng, extra)
                reps = np.stack([c.noise_calibrated(X, p["a"], p["b"], np.ones(p["t"]), rng) for _ in range(3)], axis=2)
                Y = reps.mean(axis=2)
                obs = c.score(Y)[0]
                if not np.isfinite(obs):
                    continue
                try:
                    a, b = c.fit_noise(reps)
                except ValueError:
                    continue
                x0 = np.clip(Y.mean(axis=1), 1e-9, None)
                x0 = x0 / x0.sum()
                r = np.full(p["t"], 3)
                out = arm(Y, x0, obs, lambda y, g: c.noise_calibrated(y, a, b, r, g), rng)
                if out[1] not in edge:
                    vals.append(out[0])
            hits += len(vals) >= c.MIN_FISH and c.signflip(vals) <= c.ALPHA
        rates[label] = hits / GATE_COHORTS
    rates["passed"] = rates["false_positive"] <= MAX_FALSE_POS and rates["recovery"] >= MIN_RECOVERY
    return rates


def read(rows, checks):
    if len(rows) < c.MIN_FISH:
        return {"label": "E3_inconclusive", "n_fish": len(rows)}
    v = c.verdict([r["excess_calibrated"] for r in rows], [r["excess_multinomial"] for r in rows])
    p_nonzero = c.signflip([r["excess_nonzero"] for r in rows])
    label = "E3_inconclusive"
    if checks["passed"] and v["label"] == "R1_link_survives" and p_nonzero <= c.ALPHA:
        label = "E1_consistent_with_source"
    elif checks["passed"] and v["label"] == "R2_noise_lead" and p_nonzero > c.ALPHA:
        label = "E2_noise_lead"
    return {**v, "label": label, "p_nonzero": p_nonzero, "mean_excess_nonzero": float(np.mean([r["excess_nonzero"] for r in rows]))}


def main():
    rng = np.random.default_rng(SEED)
    df = c.load_train()
    fish, params = {}, []
    for unit, g in df.groupby("unit"):
        if g["sample"].nunique() < c.MIN_T:
            continue
        reps, depth = c.fish_arrays(g)
        res = fit_fish(reps, depth, rng)
        if res is None:
            fish[str(unit)] = {"dropped": "too few clones"}
            continue
        fish[str(unit)] = res[0]
        params.append(res[1])
    used = [r for r in fish.values() if "dropped" not in r and not r["ne_at_grid_edge"]]
    checks = gate(params, rng)
    out = {"status": "done", "exploratory": True, "grid": [float(c.NE_GRID[0]), float(c.NE_GRID[-1])], "gate": checks,
           "fish": fish, "reading": read(used, checks)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print(json.dumps({k: v for k, v in out.items() if k != "fish"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
