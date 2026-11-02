"""E18: drifting heritable clone-state model (see the E18 pre-registration in EXPERIMENT_LOG.md).

uv run python experiments/clone_state_model.py calibrate   # rho_cam from the B5 AUC
uv run python experiments/clone_state_model.py fit         # fit + predictions over the grid
"""

from __future__ import annotations

import csv
import gzip
import json
import os
import sys

import numpy as np
from fit_cytokine import CONDS5, load_counts, pooled, pooled_pairs_r, test_stats
from scipy.optimize import minimize
from test_oracle import shard_of
from track_b5_camellia import auc

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "results_e18")
AUC_OBS, AUC_CEIL = 0.626, 0.952
G_CAM, G_L = (10, 20, 40), (1, 3)


def b5_design():
    rows = list(
        csv.DictReader(
            gzip.open(os.path.join(HERE, "camellia_sample_sheet.tsv.gz"), "rt"), delimiter="\t"
        )
    )
    rows = [
        r
        for r in rows
        if r["large_clone_id"] and r["pass_metQC"] == "True" and r["pass_rnaQC"] == "True"
    ]
    clone = np.array(
        [r["mouse"] + "|" + r["large_clone_id"] for r in rows]
    )  # labels repeat across mice
    group = np.array([r["mouse"] + "|" + r["tissue"] for r in rows])
    return clone, group, np.array([r["tissue"] for r in rows])


def toy_auc(rho, noise, clone, group, rng, dim=20):
    ids = {c: k for k, c in enumerate(sorted(set(clone)))}
    shared = rng.standard_normal((len(ids), dim))
    state = np.sqrt(rho) * shared[[ids[c] for c in clone]] + np.sqrt(1 - rho) * rng.standard_normal(
        (len(clone), dim)
    )
    f = state + noise * rng.standard_normal(state.shape)
    c = np.corrcoef(f)
    i, j = np.triu_indices(len(f), 1)
    keep = group[i] == group[j]
    i, j = i[keep], j[keep]
    sis = clone[i] == clone[j]
    return auc(c[i, j][sis], c[i, j][~sis])


def mean_auc(rho, noise, clone, group, reps=30):
    rng = np.random.default_rng(0)
    return float(np.mean([toy_auc(rho, noise, clone, group, rng) for _ in range(reps)]))


def calibrate():
    clone, group, _ = b5_design()
    noise = next(n for n in np.arange(0.2, 20, 0.1) if mean_auc(1.0, n, clone, group) <= AUC_CEIL)
    rhos = np.linspace(0.02, 1, 50)
    aucs = np.array([mean_auc(r, noise, clone, group) for r in rhos])
    rho_cam = float(np.interp(AUC_OBS, aucs, rhos))
    out = {
        "cells": len(clone),
        "noise": float(noise),
        "rho_cam": rho_cam,
        "curve": {f"{r:.2f}": float(a) for r, a in zip(rhos, aucs, strict=True)},
    }
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "calibration.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "curve"}, indent=1))
    return rho_cam


def split(cnt, shard):
    """Keep clones of one Epo/SCF shard (1 = held out) or all others (shard=None -> fit set)."""
    keep = (lambda k: shard_of(k, 2) == 1) if shard == 1 else (lambda k: shard_of(k, 2) != 1)
    return {
        c: {d: {k: v for k, v in by.items() if keep(k)} for d, by in days.items()}
        for c, days in cnt.items()
    }


def within_stats(cnt):
    s = {}
    for c, days in cnt.items():
        p = pooled(days)
        f = np.array([v[0] / (v[0] + v[1]) for v in p.values() if v[0] + v[1] >= 2])
        s[c] = (f.mean(), f.var(), np.mean((f > 0) & (f < 1)))
    return s


def simulate(cnt, beta, sigma_w, tau, rho_l, rng):
    """Same clones and N+M counts per (cond, day); N drawn from the clone-state model."""
    clones = sorted({k for days in cnt.values() for by in days.values() for k in by})
    z = dict(zip(clones, rng.standard_normal(len(clones)), strict=True))
    out = {c: {d: {} for d in days} for c, days in cnt.items()}
    for c, days in cnt.items():
        zb = {k: np.sqrt(rho_l) * z[k] + np.sqrt(1 - rho_l) * rng.standard_normal() for k in z}
        for d, by in days.items():
            for k, v in by.items():
                n = v[0] + v[1]
                p = 1 / (1 + np.exp(-(beta * zb[k] + tau[c] + sigma_w * rng.standard_normal(n))))
                kn = int((rng.random(n) < p).sum())
                out[c][d][k] = [kn, n - kn, v[2]]
    return out


def fit(rho_cam):
    cnt = load_counts(CONDS5)
    fitset = split(cnt, None)
    obs = within_stats(fitset)
    conds = list(cnt)

    def loss(x, rho_l):
        beta, sigma_w = abs(x[0]), abs(x[1])
        tau = dict(zip(conds, x[2:], strict=True))
        sims = [
            within_stats(simulate(fitset, beta, sigma_w, tau, rho_l, np.random.default_rng(s)))
            for s in range(3)
        ]
        return sum(
            (np.mean([s[c][j] for s in sims]) - obs[c][j]) ** 2 * w
            for c in conds
            for j, w in ((0, 1), (1, 4), (2, 1))
        )

    results = []
    for g_cam in G_CAM:
        d = 1 - rho_cam ** (1 / g_cam)
        for g_l in G_L:
            rho_l = (1 - d) ** g_l
            x0 = [1.5, 1.0] + [float(np.log(obs[c][0] / (1 - obs[c][0]))) for c in conds]
            res = minimize(
                loss,
                x0,
                args=(rho_l,),
                method="Nelder-Mead",
                options={"maxiter": 1500, "xatol": 1e-3, "fatol": 1e-6, "initial_simplex": None},
            )
            beta, sigma_w = abs(res.x[0]), abs(res.x[1])
            tau = dict(zip(conds, res.x[2:], strict=True))
            cross, pooled_r = [], []
            for s in range(20):
                sim = simulate(cnt, beta, sigma_w, tau, rho_l, np.random.default_rng(100 + s))
                cross.append(
                    {
                        f"{a}-{b}": test_stats(pooled(sim[a]), pooled(sim[b]), boot=1)["r"]
                        for a, b in (("full", "G"), ("full", "M"), ("G", "M"))
                    }
                )
                pooled_r.append(pooled_pairs_r(split(sim, 1), boot=1)["r"])
            results.append(
                {
                    "g_cam": g_cam,
                    "g_l": g_l,
                    "d": d,
                    "rho_l": rho_l,
                    "loss": float(res.fun),
                    "beta": beta,
                    "sigma_w": sigma_w,
                    "tau": tau,
                    "cross": {k: float(np.mean([c[k] for c in cross])) for k in cross[0]},
                    "epo_scf_shard1": [
                        float(np.mean(pooled_r)),
                        float(np.percentile(pooled_r, 2.5)),
                        float(np.percentile(pooled_r, 97.5)),
                    ],
                }
            )
            print(json.dumps(results[-1]), flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(
        {"rho_cam": rho_cam, "grid": results}, open(os.path.join(OUT, "fit.json"), "w"), indent=1
    )


def rho_from_auc(a, noise, clone, group):
    rhos = np.linspace(0.0, 1, 51)
    aucs = np.array([mean_auc(max(r, 1e-6), noise, clone, group) for r in rhos])
    return float(np.interp(a, aucs, rhos))


def calibrate_tissue(b5c_path):
    """E18b: per-tissue rho from per-tissue AUC (point and CI ends), same ceiling noise."""
    b5c = json.load(open(b5c_path))["per_tissue"]
    noise = json.load(open(os.path.join(OUT, "calibration.json")))["noise"]
    clone, group, tissue = b5_design()
    out = {}
    for t in ("BM", "FL"):
        k = tissue == t
        r = b5c[t]
        out[t] = {
            "auc": r["auc"],
            "rho": rho_from_auc(r["auc"], noise, clone[k], group[k]),
            "rho_ci": [rho_from_auc(a, noise, clone[k], group[k]) for a in r["ci"]],
        }
    json.dump(out, open(os.path.join(OUT, "calibration_tissue.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


def fit_b(g_fl=(8, 11), g_l=(1, 3), restarts=3, sims=5):
    """E18b refit: d from the FL rho over g_FL, g_L in {1, 3}; restarts + common random numbers."""
    cal = json.load(open(os.path.join(OUT, "calibration_tissue.json")))
    cnt = load_counts(CONDS5)
    fitset = split(cnt, None)
    obs = within_stats(fitset)
    conds = list(cnt)

    def loss(x, rho_l):
        tau = dict(zip(conds, x[2:], strict=True))
        st = [
            within_stats(
                simulate(fitset, abs(x[0]), abs(x[1]), tau, rho_l, np.random.default_rng(s))
            )
            for s in range(sims)
        ]
        return sum(
            (np.mean([q[c][j] for q in st]) - obs[c][j]) ** 2 * w
            for c in conds
            for j, w in ((0, 1), (1, 4), (2, 1))
        )

    rho_fl = cal["FL"]["rho"]
    results = []
    for g1 in g_fl:
        d = 1 - max(rho_fl, 1e-6) ** (1 / g1)
        for gl in g_l:
            rho_l = (1 - d) ** gl
            best = None
            for rs in range(restarts):
                rng = np.random.default_rng(rs)
                x0 = [rng.uniform(1, 8), rng.uniform(0.1, 1.5)] + [
                    float(np.log(obs[c][0] / (1 - obs[c][0]))) for c in conds
                ]
                res = minimize(
                    loss,
                    x0,
                    args=(rho_l,),
                    method="Nelder-Mead",
                    options={"maxiter": 2000, "xatol": 1e-3, "fatol": 1e-6},
                )
                if best is None or res.fun < best.fun:
                    best = res
            beta, sigma_w = abs(best.x[0]), abs(best.x[1])
            tau = dict(zip(conds, best.x[2:], strict=True))
            cross, pooled_r = [], []
            for s in range(20):
                sim = simulate(cnt, beta, sigma_w, tau, rho_l, np.random.default_rng(100 + s))
                cross.append(
                    {
                        f"{a}-{b}": test_stats(pooled(sim[a]), pooled(sim[b]), boot=1)["r"]
                        for a, b in (("full", "G"), ("full", "M"), ("G", "M"))
                    }
                )
                pooled_r.append(pooled_pairs_r(split(sim, 1), boot=1)["r"])
            results.append(
                {
                    "g_fl": g1,
                    "g_l": gl,
                    "d": d,
                    "rho_l": rho_l,
                    "loss": float(best.fun),
                    "beta": beta,
                    "sigma_w": sigma_w,
                    "tau": tau,
                    "cross": {k: float(np.mean([c[k] for c in cross])) for k in cross[0]},
                    "epo_scf_shard1": [
                        float(np.mean(pooled_r)),
                        float(np.percentile(pooled_r, 2.5)),
                        float(np.percentile(pooled_r, 97.5)),
                    ],
                }
            )
            print(json.dumps(results[-1]), flush=True)
    json.dump(
        {"calibration": cal, "grid": results}, open(os.path.join(OUT, "fit_b.json"), "w"), indent=1
    )


def fit_c(rhos=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95), restarts=2, sims=3):
    """E18c: fate-state sharing rho_f profiled directly; the Camellia record does not enter."""
    cnt = load_counts(CONDS5)
    fitset = split(cnt, None)
    obs = within_stats(fitset)
    conds = list(cnt)

    def loss(x, rho_f):
        tau = dict(zip(conds, x[2:], strict=True))
        st = [
            within_stats(
                simulate(fitset, abs(x[0]), abs(x[1]), tau, rho_f, np.random.default_rng(s))
            )
            for s in range(sims)
        ]
        return sum(
            (np.mean([q[c][j] for q in st]) - obs[c][j]) ** 2 * w
            for c in conds
            for j, w in ((0, 1), (1, 4), (2, 1))
        )

    results = []
    for rho_f in rhos:
        best = None
        for rs in range(restarts):
            rng = np.random.default_rng(rs)
            x0 = [rng.uniform(1, 8), rng.uniform(0.1, 1.5)] + [
                float(np.log(obs[c][0] / (1 - obs[c][0]))) for c in conds
            ]
            res = minimize(
                loss,
                x0,
                args=(rho_f,),
                method="Nelder-Mead",
                options={"maxiter": 1500, "xatol": 1e-3, "fatol": 1e-6},
            )
            if best is None or res.fun < best.fun:
                best = res
        beta, sigma_w = abs(best.x[0]), abs(best.x[1])
        tau = dict(zip(conds, best.x[2:], strict=True))
        cross = []
        for s in range(20):
            sim = simulate(cnt, beta, sigma_w, tau, rho_f, np.random.default_rng(100 + s))
            cross.append(
                {
                    f"{a}-{b}": test_stats(pooled(sim[a]), pooled(sim[b]), boot=1)["r"]
                    for a, b in (("full", "G"), ("full", "M"), ("G", "M"))
                }
            )
        results.append(
            {
                "rho_f": rho_f,
                "loss": float(best.fun),
                "beta": beta,
                "sigma_w": sigma_w,
                "tau": tau,
                "cross": {k: float(np.mean([c[k] for c in cross])) for k in cross[0]},
                "cross_sd": {k: float(np.std([c[k] for c in cross])) for k in cross[0]},
            }
        )
        print(json.dumps(results[-1]), flush=True)
    json.dump({"grid": results}, open(os.path.join(OUT, "fit_c.json"), "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "calibrate":
        calibrate()
    elif sys.argv[1] == "calibrate_tissue":
        calibrate_tissue(sys.argv[2])
    elif sys.argv[1] == "fit_b":
        fit_b()
    elif sys.argv[1] == "fit_c":
        fit_c()
    else:
        fit(json.load(open(os.path.join(OUT, "calibration.json")))["rho_cam"])
