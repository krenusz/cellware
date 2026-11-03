"""E30: the E28/E29 presence-aware model refitted with an SE-scaled loss, differential evolution
and an adequacy gate. See the 2026-11-03 (6) batch pre-registration in EXPERIMENT_LOG.md.

    uv run python experiments/e30_refit.py a|b [seed]   # a: gamma fixed 0, b: gamma free
"""

from __future__ import annotations

import json
import os
import sys

import e26_selection as e26
import e28_split as e28
import e29_ceiling as e29
import numpy as np
from clone_state_model import OUT, split, within_stats
from e20_gap import OBS
from e25_clone_types import groups, presence_stats
from fit_cytokine import CONDS5, load_counts, pooled_pairs_r
from scipy.optimize import differential_evolution
from track_b10 import stats as b10_stats

CONDS = e26.CONDS
GATED = ("full", "G", "M")
SIMS = 5


def targets(cnt):
    """Loss targets: within (3 per condition), presence (2 per condition), overlap."""
    w, p = within_stats(cnt), e26.presence(cnt)
    t = {}
    for c in CONDS:
        t[f"{c}_mean"], t[f"{c}_var"], t[f"{c}_mixed"] = (float(v) for v in w[c])
        t[f"{c}_n"], t[f"{c}_logsize"] = float(p[c][0]), float(p[c][1])
    t["overlap"] = e28.overlap(cnt)
    return t


def bootstrap_se(cnt, rng, boots=500):
    """Clone-bootstrap SE of every target; SE sqrt(n) for present-clone counts."""
    clones = e26.universe(cnt)
    draws = []
    for _ in range(boots):
        pick = rng.choice(len(clones), len(clones))
        mult = np.bincount(pick, minlength=len(clones))
        keep = {clones[i]: m for i, m in enumerate(mult) if m}
        # resampled data: each clone repeated by its multiplicity under a new key
        bc = {
            c: {
                d: {(k, j): v for k, v in by.items() if k in keep for j in range(keep[k])}
                for d, by in days.items()
            }
            for c, days in cnt.items()
        }
        draws.append(targets(bc))
    se = {k: float(np.nanstd([d[k] for d in draws])) for k in draws[0]}
    t = targets(cnt)
    for c in CONDS:
        se[f"{c}_n"] = float(np.sqrt(max(t[f"{c}_n"], 1)))
        # amendment: floor for share terms (Epo's mixed fraction is exactly 0, bootstrap SE ~0)
        for s in ("mean", "var", "mixed"):
            se[f"{c}_{s}"] = max(se[f"{c}_{s}"], 0.25 / np.sqrt(max(t[f"{c}_n"], 1)))
    return {k: max(v, 1e-3) for k, v in se.items()}


def unpack(x, variant):
    prm = e29.unpack(np.asarray(x))
    if variant == "a":
        prm["gamma"] = 0.0
    return prm


def main(variant, seed=0):
    cnt = load_counts(CONDS5)
    clones = e26.universe(cnt)
    fitset = split(cnt, None)
    obs = targets(fitset)
    se = bootstrap_se(fitset, np.random.default_rng(seed))

    def sim_targets(prm):
        ts = [
            targets(split(e29.simulate(clones, prm, np.random.default_rng(s)), None))
            for s in range(SIMS)
        ]
        return {
            k: float(np.nanmean([t[k] for t in ts]))
            if any(np.isfinite(t[k]) for t in ts)
            else np.nan
            for k in obs
        }

    def loss(x):
        st = sim_targets(unpack(x, variant))
        z = [((st[k] - obs[k]) / se[k]) ** 2 if np.isfinite(st[k]) else 100.0 for k in obs]
        tot = float(np.sum(z))
        return tot if np.isfinite(tot) else 1e9

    dim = 19 if variant == "b" else 18
    res = differential_evolution(
        lambda x: loss(np.r_[x, 0.0]) if variant == "a" else loss(x),
        [(-3, 3)] * dim,
        popsize=6,
        maxiter=int(os.environ.get("E30_MAXITER", 120)),
        seed=seed,
        polish=False,
        tol=1e-6,
        init="latinhypercube",
    )
    xbest = np.r_[res.x, 0.0] if variant == "a" else res.x
    prm = unpack(xbest, variant)
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
    adequate = all(gate[c]["mean_ok"] and gate[c]["n_ok"] for c in GATED)
    rep = {
        "variant": variant,
        "seed": seed,
        "loss": float(res.fun),
        "generations": int(res.nit),
        "params": prm,
        "z": {k: float((st[k] - obs[k]) / se[k]) for k in obs},
        "gate": gate,
        "adequate": bool(adequate),
    }
    cross, pres, shard1, b10 = [], [], [], []
    for s in range(200):
        sim = e29.simulate(clones, prm, np.random.default_rng(1000 + s))
        cross.append(e29.cross_r(sim))
        pres.append(presence_stats(sim))
        shard1.append(pooled_pairs_r(split(sim, 1), boot=1)["r"])
        b10.append(b10_stats(sim)["frac_with_N_in_M"])
    for k, o in OBS.items():
        v = np.array([c[k] for c in cross], float)
        v = v[np.isfinite(v)]
        hi = float((v >= o).mean())
        rep[k] = {
            "obs": o,
            "sim_mean": float(v.mean()),
            "sim_95": np.percentile(v, [2.5, 97.5]).tolist(),
            "p_ge_obs": hi,
            "p_two_sided": float(2 * min(hi, 1 - hi)),
        }
    b7 = json.load(open(os.path.join(OUT, "b7_selection.json")))
    for k in ("G_mean_both_minus_Gonly", "M_both_minus_Monly"):
        v = np.array([p[k] for p in pres], float)
        v = v[np.isfinite(v)]
        lo, hi = np.percentile(v, [2.5, 97.5])
        rep[k] = {
            "data": b7[k]["data"],
            "sim_mean": float(v.mean()),
            "sim_95": [float(lo), float(hi)],
            "inside": bool(lo <= b7[k]["data"] <= hi),
        }
    v = np.array(b10, float)
    v = v[np.isfinite(v)]
    rep["b10_frac_with_N_in_M"] = {
        "data": 0.205,
        "sim_mean": float(v.mean()),
        "sim_95": np.percentile(v, [2.5, 97.5]).tolist(),
    }
    _, _, both, g_only, _ = groups(e29.simulate(clones, prm, np.random.default_rng(999)))
    rep["overlap_check"] = len(both) / max(len(both) + len(g_only), 1)
    rep["shard1_pred"] = {
        "mean": float(np.nanmean(shard1)),
        "sim_95": np.nanpercentile(shard1, [2.5, 97.5]).tolist(),
    }
    rep["passes"] = bool(
        adequate
        and all(rep[k]["p_two_sided"] >= 0.05 for k in OBS)
        and rep["G_mean_both_minus_Gonly"]["inside"]
        and rep["M_both_minus_Monly"]["inside"]
    )
    json.dump(rep, open(os.path.join(OUT, f"e30{variant}_s{seed}.json"), "w"), indent=1)
    print(json.dumps({k: rep[k] for k in ("loss", "adequate", "passes", "G-M")}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0)
