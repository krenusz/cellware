"""E32: E30a with multinomial founder allocation (K = 1 + Poisson founders split among wells) and
pairwise co-presence in the loss. See the E32 pre-registration in EXPERIMENT_LOG.md.

    uv run python experiments/e32_multinomial.py seed
"""

from __future__ import annotations

import itertools
import json
import os
import sys

import e26_selection as e26
import e29_ceiling as e29
import e30_refit as e30
import numpy as np
from clone_state_model import OUT, split
from e20_gap import OBS, RHO_F
from e25_clone_types import presence_stats
from fit_cytokine import CONDS5, load_counts, pooled, pooled_pairs_r
from scipy.optimize import differential_evolution
from track_b10 import stats as b10_stats

CONDS = e26.CONDS
PAIRS = list(itertools.combinations(CONDS, 2))
GATE_PAIRS = ("full-G", "full-M", "G-M")


def simulate(clones, prm, rng):
    """As e29.simulate (gamma 0), but founders per well come from one multinomial split."""
    n = len(clones)
    z = rng.standard_normal(n)
    a = prm["sigma_a"] * rng.standard_normal(n)
    total = 1 + rng.poisson(prm["lam_k"], n)
    k = rng.multinomial(total, prm["w"])  # n x 5, columns in CONDS order
    out = {}
    for j, c in enumerate(CONDS):
        zb = np.sqrt(RHO_F) * z + np.sqrt(1 - RHO_F) * rng.standard_normal(n)
        p = 1 / (1 + np.exp(-(prm["beta"] * zb + prm["tau"][c])))
        lam = k[:, j] * np.exp(np.clip(prm["mu"][c] + a, -30, 8))
        kn = rng.poisson(np.clip(lam * p * prm["s"][c], 0, 1e4))
        km = rng.poisson(np.clip(lam * (1 - p), 0, 1e4))
        keep = np.flatnonzero(kn + km > 0)
        out[c] = {4: {clones[i]: [int(kn[i]), int(km[i]), 0] for i in keep}}
    return out


def pct(v, q):
    """Percentiles that tolerate an empty array (a fit with no clones for a statistic)."""
    return np.percentile(v, q).tolist() if len(v) else [np.nan] * len(q)


def copresence(cnt):
    pres = {c: {k for k, v in pooled(cnt[c]).items() if v[0] + v[1] >= 2} for c in CONDS}
    return {f"{a}-{b}": float(len(pres[a] & pres[b])) for a, b in PAIRS}


def targets(cnt):
    t = e30.targets(cnt)
    t.update({f"co_{k}": v for k, v in copresence(cnt).items()})
    return t


def unpack(x):
    """16 E26b parameters, lambda_K, 4 well logits (full = 0)."""
    prm = e26.unpack(np.asarray(x[:16]), "b")
    prm["lam_k"] = float(5 / (1 + np.exp(-x[16])))
    logit = np.r_[0.0, 3 * np.tanh(np.asarray(x[17:21]))]
    prm["w"] = (np.exp(logit) / np.exp(logit).sum()).tolist()
    prm["gamma"] = 0.0
    return prm


def main(seed, maxiter=300):
    cnt = load_counts(CONDS5)
    clones = e26.universe(cnt)
    fitset = split(cnt, None)
    obs = targets(fitset)
    se = e30.bootstrap_se(fitset, np.random.default_rng(seed))
    for k in obs:
        if k.startswith("co_"):
            se[k] = float(np.sqrt(max(obs[k], 4)))

    def sim_targets(prm):
        ts = [
            targets(split(simulate(clones, prm, np.random.default_rng(s)), None))
            for s in range(e30.SIMS)
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
        [(-3, 3)] * 21,
        popsize=6,
        maxiter=int(os.environ.get("E32_MAXITER", maxiter)),
        seed=seed,
        polish=False,
        tol=1e-6,
        init="latinhypercube",
    )
    prm = unpack(res.x)
    st = sim_targets(prm)
    full_obs = targets(cnt)  # whole data, for the co-presence gate (data numbers 29 / 84 / 78)
    full_sim = [targets(simulate(clones, prm, np.random.default_rng(500 + s))) for s in range(20)]
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
    co = {}
    for p in GATE_PAIRS:
        sim_n = float(np.mean([t[f"co_{p}"] for t in full_sim]))
        co[p] = {
            "data": full_obs[f"co_{p}"],
            "sim": sim_n,
            "ok": bool(abs(sim_n - full_obs[f"co_{p}"]) <= 0.3 * full_obs[f"co_{p}"]),
        }
    adequate = all(gate[c]["mean_ok"] and gate[c]["n_ok"] for c in e30.GATED) and all(
        v["ok"] for v in co.values()
    )
    rep = {
        "seed": seed,
        "loss": float(res.fun),
        "generations": int(res.nit),
        "converged": bool(res.success),
        "params": prm,
        "z": {k: float((st[k] - obs[k]) / se[k]) for k in obs},
        "gate": gate,
        "copresence_gate": co,
        "copresence_all": {
            p: [full_obs[f"co_{p}"], float(np.mean([t[f"co_{p}"] for t in full_sim]))]
            for p in (f"{a}-{b}" for a, b in PAIRS)
        },
        "adequate": bool(adequate),
    }
    cross, pres, shard1, b10 = [], [], [], []
    for s in range(200):
        sim = simulate(clones, prm, np.random.default_rng(1000 + s))
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
            "sim_95": [float(lo), float(hi)],
            "inside": bool(lo <= b7[k]["data"] <= hi),
        }
    v = np.array(b10, float)
    v = v[np.isfinite(v)]
    rep["b10_frac_with_N_in_M"] = {"data": 0.205, "sim_mean": float(v.mean()) if len(v) else None}
    rep["shard1_pred"] = {
        "mean": float(np.nanmean(shard1)),
        "sim_95": pct(np.array(shard1, float)[np.isfinite(shard1)], [2.5, 97.5]),
    }
    rep["passes"] = bool(
        adequate
        and all(rep[k]["p_two_sided"] >= 0.05 for k in OBS)
        and rep["G_mean_both_minus_Gonly"]["inside"]
        and rep["M_both_minus_Monly"]["inside"]
    )
    json.dump(rep, open(os.path.join(OUT, f"e32_s{seed}.json"), "w"), indent=1)
    print(
        json.dumps({k: rep[k] for k in ("loss", "adequate", "passes", "G-M", "full-M")}, indent=1)
    )


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
