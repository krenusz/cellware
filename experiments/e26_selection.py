"""E26: selection-aware simulator; counts are simulated, so presence emerges from fate and size.
See the 2026-11-03 (4) batch pre-registration in EXPERIMENT_LOG.md.

    uv run python experiments/e26_selection.py a [start]   # pure selection (shared tau)
    uv run python experiments/e26_selection.py b [start]   # instruction + selection
With a start index (0-2) only that start is fitted and written to e26{variant}_s{start}.json
(one GitHub Actions job per start); the best start is picked afterwards.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from clone_state_model import OUT, split, within_stats
from e20_gap import OBS, RHO_F
from e25_clone_types import presence_stats
from fit_cytokine import CONDS5, load_counts, pooled, pooled_pairs_r, test_stats
from scipy.optimize import minimize

CONDS = ["full", "G", "M", "Epo", "SCF"]


def universe(cnt):
    return sorted({k for days in cnt.values() for by in days.values() for k in by})


def simulate(clones, prm, rng):
    """{cond: {4: {clone: [N, M, 0]}}} for clones with N + M > 0."""
    n = len(clones)
    z = rng.standard_normal(n)
    a = prm["sigma_a"] * rng.standard_normal(n)
    out = {}
    for c in CONDS:
        zb = np.sqrt(RHO_F) * z + np.sqrt(1 - RHO_F) * rng.standard_normal(n)
        p = 1 / (1 + np.exp(-(prm["beta"] * zb + prm["tau"][c])))
        # optimiser line searches visit extreme values: cap rates so Poisson draws stay valid
        lam = np.exp(np.clip(prm["mu"][c] + a, -30, 8))
        rn = np.nan_to_num(lam * p * min(prm["s"][c], 1e3), nan=0.0, posinf=1e4)
        rm = np.nan_to_num(lam * (1 - p), nan=0.0, posinf=1e4)
        kn = rng.poisson(np.clip(rn, 0, 1e4))
        km = rng.poisson(np.clip(rm, 0, 1e4))
        keep = np.flatnonzero(kn + km > 0)
        out[c] = {4: {clones[i]: [int(kn[i]), int(km[i]), 0] for i in keep}}
    return out


def presence(cnt):
    s = {}
    for c, days in cnt.items():
        sz = np.array([v[0] + v[1] for v in pooled(days).values()])
        sz = sz[sz >= 2]
        s[c] = (len(sz), float(np.log(sz).mean()) if len(sz) else 0.0)
    return s


def unpack(x, variant):
    """Bounded transforms, so no optimiser step can leave the valid region."""
    th = np.tanh
    prm = {"beta": 10 / (1 + np.exp(-x[0])), "sigma_a": 3 / (1 + np.exp(-x[1]))}
    i = 2
    if variant == "a":
        prm["tau"] = dict.fromkeys(CONDS, float(10 * th(x[i])))
        i += 1
    else:
        prm["tau"] = {c: float(10 * th(v)) for c, v in zip(CONDS, x[i : i + 5], strict=True)}
        i += 5
    prm["s"] = {
        "full": 1.0,
        **{c: float(np.exp(6 * th(v))) for c, v in zip(CONDS[1:], x[i : i + 4], strict=True)},
    }
    i += 4
    prm["mu"] = {c: float(-3 + 7 * th(v)) for c, v in zip(CONDS, x[i : i + 5], strict=True)}
    return prm


def main(variant, only=None):
    cnt = load_counts(CONDS5)
    clones = universe(cnt)
    fitset = split(cnt, None)
    obs, obs_p = within_stats(fitset), presence(fitset)

    def loss(x):
        if not np.all(np.isfinite(x)):
            return 1e6
        prm = unpack(x, variant)
        tot = 0.0
        st, pr = [], []
        for s in range(5):
            sim = split(simulate(clones, prm, np.random.default_rng(s)), None)
            st.append(within_stats(sim))
            pr.append(presence(sim))
        for c in CONDS:
            tot += sum(
                (np.mean([q[c][j] for q in st]) - obs[c][j]) ** 2 * w
                for j, w in ((0, 1), (1, 4), (2, 1))
            )
            n_sim = np.mean([q[c][0] for q in pr])
            tot += ((n_sim - obs_p[c][0]) / obs_p[c][0]) ** 2
            tot += 0.25 * (np.mean([q[c][1] for q in pr]) - obs_p[c][1]) ** 2
        return tot

    starts = []
    for r in range(3):
        rng = np.random.default_rng(r)
        x = [rng.uniform(-1, 1), rng.uniform(-1, 1)]
        x += [0.0] if variant == "a" else [0.1, 0.1, -0.4, 0.0, 0.0]
        x += list(rng.normal(0, 0.3, 4))
        x += [
            float(
                np.arctanh(
                    np.clip((np.log(max(obs_p[c][0], 1) / len(clones)) + 2.5 + 3) / 7, -0.95, 0.95)
                )
            )
            for c in CONDS
        ]
        starts.append(x)
    best = None
    for r, x0 in enumerate(starts):
        if only is not None and r != only:
            continue
        res = minimize(
            loss,
            x0,
            method="Nelder-Mead",
            options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-7, "adaptive": True},
        )
        res = minimize(
            loss, res.x, method="Powell", options={"maxfev": 2000, "xtol": 1e-3, "ftol": 1e-6}
        )
        print(json.dumps({"loss": float(res.fun)}), flush=True)
        if best is None or res.fun < best.fun:
            best = res
    prm = unpack(best.x, variant)
    cross, pres, shard1 = [], [], []
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
    rep = {
        "variant": variant,
        "loss": float(best.fun),
        "params": prm,
        "presence_fit": {
            c: {"obs": obs_p[c], "sim_last": presence(split(sim, None))[c]} for c in CONDS
        },
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
        v = np.array([p[k] for p in pres])
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
        and rep["G-M"]["p_ge_obs"] >= 0.025
        and rep["G_mean_both_minus_Gonly"]["inside"]
        and rep["M_both_minus_Monly"]["inside"]
    )
    name = f"e26{variant}" + ("" if only is None else f"_s{only}")
    json.dump(rep, open(os.path.join(OUT, f"{name}.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)
