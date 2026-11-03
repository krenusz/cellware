"""E33: E30a with measured day-2 inputs (graded Cebpa-Irf8 score, committed fractions).
See the E33 pre-registration in EXPERIMENT_LOG.md.

    uv run python experiments/e33_measured.py seed
"""

from __future__ import annotations

import json
import os
import sys

import e26_selection as e26
import e29_ceiling as e29
import e30_refit as e30
import numpy as np
import pandas as pd
from clone_state_model import OUT, split
from e20_gap import OBS, RHO_F
from e25_clone_types import presence_stats
from fit_cytokine import CONDS5, load_counts, pooled_pairs_r
from scipy.optimize import differential_evolution
from track_b10 import stats as b10_stats

CONDS = e26.CONDS
GATE_N = ("G", "M")  # present-clone gate; full excluded (E32)


def measured(clones, rng):
    """Per clone (m, frac_N, frac_M); unmeasured clones draw jointly from the measured ones."""
    t = pd.read_csv(os.path.join(OUT, "b14_clone_day2.csv"), index_col=0)
    score = t.cebpa - t.irf8
    t["m"] = (score - score.mean()) / score.std()
    pool = t[["m", "frac_N", "frac_M"]].to_numpy()
    out = np.empty((len(clones), 3))
    for i, c in enumerate(clones):
        out[i] = (
            t.loc[c, ["m", "frac_N", "frac_M"]].to_numpy()
            if c in t.index
            else pool[rng.integers(len(pool))]
        )
    return out


def simulate(clones, prm, rng, meas):
    n = len(clones)
    m, fn, fm = meas[:, 0], meas[:, 1], meas[:, 2]
    w = prm["w"]
    z = w * m + np.sqrt(1 - w**2) * rng.standard_normal(n)
    a = prm["sigma_a"] * rng.standard_normal(n)
    cn = np.clip(prm["kappa"] * fn, 0, 1)
    cm = np.clip(prm["kappa"] * fm, 0, 1 - cn)
    free = 1 - cn - cm
    out = {}
    for c in CONDS:
        k = rng.poisson(prm["nu"], n)
        zb = np.sqrt(RHO_F) * z + np.sqrt(1 - RHO_F) * rng.standard_normal(n)
        p = 1 / (1 + np.exp(-(prm["beta"] * zb + prm["tau"][c])))
        lam = k * np.exp(np.clip(prm["mu"][c] + a, -30, 8))
        kn = rng.poisson(np.clip(lam * (free * p + cn) * prm["s"][c], 0, 1e4))
        km = rng.poisson(np.clip(lam * (free * (1 - p) + cm), 0, 1e4))
        keep = np.flatnonzero(kn + km > 0)
        out[c] = {4: {clones[i]: [int(kn[i]), int(km[i]), 0] for i in keep}}
    return out


def unpack(x):
    prm = e29.unpack(np.r_[np.asarray(x[:17]), 0.0])  # E30a: 16 + nu, gamma 0
    prm["w"] = float(1 / (1 + np.exp(-x[17])))
    prm["kappa"] = float(2 / (1 + np.exp(-x[18])))
    return prm


def pct(v, q):
    return np.percentile(v, q).tolist() if len(v) else [np.nan] * len(q)


TARGETS = None  # hook: loss targets (default e30.targets); E36 adds full-arm co-presence
EXTRA_SE = None  # hook: {target: SE} for targets not covered by the bootstrap
SIMS = None  # hook: simulations per loss evaluation (default e30.SIMS)


def main(seed, maxiter=300, dim=19, name="e33", x0=None):
    cnt = load_counts(CONDS5)
    clones = e26.universe(cnt)
    meas = measured(clones, np.random.default_rng(77))
    fitset = split(cnt, None)
    targets = TARGETS or e30.targets
    sims = SIMS or e30.SIMS
    obs = targets(fitset)
    se = e30.bootstrap_se(fitset, np.random.default_rng(seed))
    if EXTRA_SE:
        se.update(EXTRA_SE(obs))

    def sim_targets(prm):
        ts = [
            targets(split(simulate(clones, prm, np.random.default_rng(s), meas), None))
            for s in range(sims)
        ]
        return {
            k: float(np.nanmean([t[k] for t in ts]))
            if any(np.isfinite(t[k]) for t in ts)
            else np.nan
            for k in obs
        }

    def loss(x):
        st = sim_targets(unpack(x))
        z = [((st[k] - obs[k]) / se[k]) ** 2 if np.isfinite(st[k]) else 100.0 for k in obs]
        tot = float(np.sum(z))
        return tot if np.isfinite(tot) else 1e9

    res = differential_evolution(
        loss,
        [(-3, 3)] * dim,
        popsize=6,
        maxiter=int(os.environ.get("E33_MAXITER", maxiter)),
        seed=seed,
        polish=False,
        tol=1e-6,
        init="latinhypercube",
        x0=x0,
    )
    prm = unpack(res.x)
    st = sim_targets(prm)
    gate = {
        c: {
            "mean_sim": st[f"{c}_mean"],
            "mean_data": obs[f"{c}_mean"],
            "mean_ok": bool(abs(st[f"{c}_mean"] - obs[f"{c}_mean"]) <= 1.96 * se[f"{c}_mean"]),
            "n_sim": st[f"{c}_n"],
            "n_data": obs[f"{c}_n"],
            "n_ok": bool(abs(st[f"{c}_n"] - obs[f"{c}_n"]) <= 0.2 * obs[f"{c}_n"]),
        }
        for c in CONDS
    }
    adequate = all(gate[c]["mean_ok"] for c in ("full", "G", "M")) and all(
        gate[c]["n_ok"] for c in GATE_N
    )
    rep = {
        "seed": seed,
        "loss": float(res.fun),
        "generations": int(res.nit),
        "converged": bool(res.success),
        "params": prm,
        "z": {k: float((st[k] - obs[k]) / se[k]) for k in obs},
        "gate": gate,
        "adequate": bool(adequate),
    }
    cross, pres, shard1, b10 = [], [], [], []
    for s in range(200):
        sim = simulate(clones, prm, np.random.default_rng(1000 + s), meas)
        cross.append(e29.cross_r(sim))
        pres.append(presence_stats(sim))
        shard1.append(pooled_pairs_r(split(sim, 1), boot=1)["r"])
        b10.append(b10_stats(sim)["frac_with_N_in_M"])
    for k, o in OBS.items():
        v = np.array([c[k] for c in cross], float)
        v = v[np.isfinite(v)]
        hi = float((v >= o).mean()) if len(v) else 0.0
        rep[k] = {
            "obs": o,
            "sim_mean": float(v.mean()) if len(v) else np.nan,
            "sim_95": pct(v, [2.5, 97.5]),
            "p_ge_obs": hi,
            "p_two_sided": float(2 * min(hi, 1 - hi)),
        }
    b7 = json.load(open(os.path.join(OUT, "b7_selection.json")))
    for k in ("G_mean_both_minus_Gonly", "M_both_minus_Monly"):
        v = np.array([p[k] for p in pres], float)
        v = v[np.isfinite(v)]
        lo, hi = pct(v, [2.5, 97.5])
        rep[k] = {
            "data": b7[k]["data"],
            "sim_mean": float(v.mean()) if len(v) else np.nan,
            "sim_95": [lo, hi],
            "inside": bool(lo <= b7[k]["data"] <= hi),
        }
    v = np.array(b10, float)
    v = v[np.isfinite(v)]
    rep["b10_frac_with_N_in_M"] = {"data": 0.205, "sim_mean": float(v.mean()) if len(v) else None}
    s1 = np.array(shard1, float)
    rep["shard1_pred"] = {
        "mean": float(np.nanmean(s1)),
        "sim_95": pct(s1[np.isfinite(s1)], [2.5, 97.5]),
    }
    rep["inputs_identified"] = bool(prm["w"] > 0.2 or prm["kappa"] > 0.2)
    rep["passes"] = bool(
        adequate
        and rep["G-M"]["p_two_sided"] >= 0.05
        and rep["G_mean_both_minus_Gonly"]["inside"]
        and rep["M_both_minus_Monly"]["inside"]
    )
    json.dump(rep, open(os.path.join(OUT, f"{name}_s{seed}.json"), "w"), indent=1)
    print(
        json.dumps(
            {k: rep[k] for k in ("loss", "adequate", "passes", "inputs_identified", "G-M")},
            indent=1,
        )
    )


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
