"""E21b (warm-started per-condition gain) and E22 (second clone trait under single cytokines).
See the 2026-11-03 (2) batch pre-registration in EXPERIMENT_LOG.md.

    uv run python experiments/e22_models.py e21b
    uv run python experiments/e22_models.py e22
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from clone_state_model import OUT, split, within_stats
from e20_gap import OBS, RHO_F
from fit_cytokine import CONDS5, load_counts, pooled, test_stats
from scipy.optimize import minimize

SINGLE = {"G", "M", "Epo", "SCF"}
SIMS = 5


def simulate(cnt, beta, sigma_w, tau, gamma, rng, urng):
    """e20_gap.simulate plus gamma * u (clone trait) under single cytokines; u from its own rng,
    so gamma = 0 reproduces e20_gap.simulate exactly."""
    clones = sorted({k for days in cnt.values() for by in days.values() for k in by})
    z = dict(zip(clones, rng.standard_normal(len(clones)), strict=True))
    u = dict(zip(clones, urng.standard_normal(len(clones)), strict=True))
    out = {c: {d: {} for d in days} for c, days in cnt.items()}
    for c, days in cnt.items():
        zb = {k: np.sqrt(RHO_F) * z[k] + np.sqrt(1 - RHO_F) * rng.standard_normal() for k in z}
        g = gamma if c in SINGLE else 0.0
        for d, by in days.items():
            for k, v in by.items():
                n = v[0] + v[1]
                lin = beta[c] * zb[k] + g * u[k] + tau[c] + sigma_w * rng.standard_normal(n)
                kn = int((rng.random(n) < 1 / (1 + np.exp(-lin))).sum())
                out[c][d][k] = [kn, n - kn, v[2]]
    return out


def unpack(x, conds, kind):
    nc = len(conds)
    beta0, sigma_w = abs(x[0]), abs(x[1])
    tau = dict(zip(conds, x[2 : 2 + nc], strict=True))
    gains = [1.0] * nc
    gamma = 0.0
    if kind == "e21b":
        gains = [1.0] + [abs(v) for v in x[2 + nc :]]
    elif kind == "e22":
        gamma = float(x[2 + nc])
    return {c: beta0 * gn for c, gn in zip(conds, gains, strict=True)}, sigma_w, tau, gamma


def sim(cnt, params, s):
    beta, sigma_w, tau, gamma = params
    return simulate(
        cnt, beta, sigma_w, tau, gamma, np.random.default_rng(s), np.random.default_rng(10_000 + s)
    )


def main(kind):
    cnt = load_counts(CONDS5)
    fitset = split(cnt, None)
    obs = within_stats(fitset)
    conds = list(cnt)

    def loss_of(params):
        st = [within_stats(sim(fitset, params, s)) for s in range(SIMS)]
        return sum(
            (np.mean([q[c][j] for q in st]) - obs[c][j]) ** 2 * w
            for c in conds
            for j, w in ((0, 1), (1, 4), (2, 1))
        )

    e20 = json.load(open(os.path.join(OUT, "e20.json")))
    x0 = [e20["beta"]["full"], e20["sigma_w"]] + [e20["tau"][c] for c in conds]
    ref = loss_of(unpack(np.array(x0), conds, "e20"))
    x0 += [1.0] * (len(conds) - 1) if kind == "e21b" else [0.0]

    def loss(x):
        return loss_of(unpack(x, conds, kind))

    res = minimize(
        loss, x0, method="Nelder-Mead", options={"maxiter": 1500, "xatol": 1e-3, "fatol": 1e-6}
    )
    print(json.dumps({"stage": "nelder-mead", "loss": float(res.fun), "ref_e20": ref}), flush=True)
    res = minimize(
        loss, res.x, method="Powell", options={"maxfev": 1500, "xtol": 1e-3, "ftol": 1e-6}
    )
    print(json.dumps({"stage": "powell", "loss": float(res.fun)}), flush=True)
    params = unpack(res.x, conds, kind)
    cross = []
    for s in range(200):
        out = sim(cnt, params, 1000 + s)
        cross.append(
            {
                f"{a}-{b}": test_stats(pooled(out[a]), pooled(out[b]), boot=1)["r"]
                for a, b in (("full", "G"), ("full", "M"), ("G", "M"))
            }
        )
    beta, sigma_w, tau, gamma = params
    rep = {
        "kind": kind,
        "loss": float(res.fun),
        "ref_e20_loss": float(ref),
        "beta": beta,
        "sigma_w": sigma_w,
        "tau": tau,
        "gamma": gamma,
    }
    for k, o in OBS.items():
        v = np.array([c[k] for c in cross])
        hi = float((v >= o).mean())
        rep[k] = {
            "obs": o,
            "sim_mean": float(v.mean()),
            "sim_sd": float(v.std()),
            "sim_95": np.percentile(v, [2.5, 97.5]).tolist(),
            "p_ge_obs": hi,
            "p_two_sided": float(2 * min(hi, 1 - hi)),
        }
    json.dump(rep, open(os.path.join(OUT, f"{kind}.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
