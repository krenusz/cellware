"""E28: E26 + founders split across condition wells (k_c ~ Poisson(nu)), presence overlap.
See the E28 pre-registration in EXPERIMENT_LOG.md.

    uv run python experiments/e28_split.py a|b [start]   # a: overlap in the loss, b: without
"""

from __future__ import annotations

import json
import os
import sys

import e26_selection as e26
import numpy as np
from clone_state_model import OUT, split, within_stats
from e20_gap import OBS, RHO_F
from e25_clone_types import groups, presence_stats
from fit_cytokine import CONDS5, load_counts, pooled, pooled_pairs_r, test_stats
from scipy.optimize import minimize

CONDS = e26.CONDS
OVERLAP = 0.141  # clones present under G and M / clones present under G (data, B7 groups)


def simulate(clones, prm, rng):
    """As e26.simulate, but each condition gets k_c ~ Poisson(nu) founders of the clone."""
    n = len(clones)
    z = rng.standard_normal(n)
    a = prm["sigma_a"] * rng.standard_normal(n)
    out = {}
    for c in CONDS:
        k = rng.poisson(prm["nu"], n)
        zb = np.sqrt(RHO_F) * z + np.sqrt(1 - RHO_F) * rng.standard_normal(n)
        p = 1 / (1 + np.exp(-(prm["beta"] * zb + prm["tau"][c])))
        lam = k * np.exp(np.clip(prm["mu"][c] + a, -30, 8))
        kn = rng.poisson(np.clip(lam * p * prm["s"][c], 0, 1e4))
        km = rng.poisson(np.clip(lam * (1 - p), 0, 1e4))
        keep = np.flatnonzero(kn + km > 0)
        out[c] = {4: {clones[i]: [int(kn[i]), int(km[i]), 0] for i in keep}}
    return out


def overlap(cnt):
    _, _, both, g_only, _ = groups(cnt)
    return len(both) / max(len(both) + len(g_only), 1)


def unpack(x):
    prm = e26.unpack(x[:-1], "b")
    prm["nu"] = float(5 / (1 + np.exp(-x[-1])))
    return prm


def main(variant, only=None):
    cnt = load_counts(CONDS5)
    clones = e26.universe(cnt)
    fitset = split(cnt, None)
    obs, obs_p = within_stats(fitset), e26.presence(fitset)

    def loss(x):
        if not np.all(np.isfinite(x)):
            return 1e6
        prm = unpack(x)
        tot = 0.0
        st, pr, ov = [], [], []
        for s in range(5):
            sim = split(simulate(clones, prm, np.random.default_rng(s)), None)
            st.append(within_stats(sim))
            pr.append(e26.presence(sim))
            ov.append(overlap(sim))
        for c in CONDS:
            for j, w in ((0, 1), (1, 4), (2, 1)):
                vals = [q[c][j] for q in st if np.isfinite(q[c][j])]
                # a condition with no present clones has no within stats: fixed penalty
                tot += (np.mean(vals) - obs[c][j]) ** 2 * w if vals else 1.0
            tot += ((np.mean([q[c][0] for q in pr]) - obs_p[c][0]) / obs_p[c][0]) ** 2
            tot += 0.25 * (np.mean([q[c][1] for q in pr]) - obs_p[c][1]) ** 2
        if variant == "a":
            tot += ((np.mean(ov) - OVERLAP) / 0.05) ** 2
        return tot if np.isfinite(tot) else 1e6

    starts = []
    for r in range(3):
        rng = np.random.default_rng(r)
        x = [rng.uniform(-1, 1), rng.uniform(-1, 1), 0.1, 0.1, -0.4, 0.0, 0.0]
        x += list(rng.normal(0, 0.3, 4))
        x += [
            float(
                np.arctanh(
                    np.clip((np.log(max(obs_p[c][0], 1) / len(clones)) + 2.5 + 3) / 7, -0.95, 0.95)
                )
            )
            for c in CONDS
        ]
        x += [rng.uniform(-2, 0)]
        starts.append(x)
    best = None
    for r, x0 in enumerate(starts):
        if only is not None and r != only:
            continue
        res = minimize(
            loss,
            x0,
            method="Nelder-Mead",
            options={
                "maxiter": int(os.environ.get("E28_MAXITER", 4000)),
                "xatol": 1e-4,
                "fatol": 1e-7,
                "adaptive": True,
            },
        )
        print(json.dumps({"loss": float(res.fun)}), flush=True)
        if best is None or res.fun < best.fun:
            best = res
    prm = unpack(best.x)
    cross, pres, shard1, ovs = [], [], [], []
    for s in range(200):
        sim = simulate(clones, prm, np.random.default_rng(1000 + s))
        cross.append(
            {
                f"{a}-{b}": test_stats(pooled(sim[a]), pooled(sim[b]), boot=1)["r"]
                for a, b in (("full", "G"), ("full", "M"), ("G", "M"))
            }
        )
        pres.append(presence_stats(sim))
        shard1.append(pooled_pairs_r(split(sim, 1), boot=1)["r"])
        ovs.append(overlap(sim))
    rep = {"variant": variant, "loss": float(best.fun), "params": prm}
    rep["overlap"] = {
        "obs": OVERLAP,
        "sim_mean": float(np.mean(ovs)),
        "sim_95": np.percentile(ovs, [2.5, 97.5]).tolist(),
        "within_0.05": bool(abs(np.mean(ovs) - OVERLAP) <= 0.05),
    }
    for k, o in OBS.items():
        v = np.array([c[k] for c in cross])
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
    rep["shard1_pred"] = {
        "mean": float(np.mean(shard1)),
        "sim_95": np.percentile(shard1, [2.5, 97.5]).tolist(),
    }
    rep["passes"] = bool(
        all(rep[k]["p_two_sided"] >= 0.05 for k in OBS)
        and rep["G_mean_both_minus_Gonly"]["inside"]
        and rep["M_both_minus_Monly"]["inside"]
    )
    name = f"e28{variant}" + ("" if only is None else f"_s{only}")
    json.dump(rep, open(os.path.join(OUT, f"{name}.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)
