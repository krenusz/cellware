"""E20: is the G-CSF vs M-CSF gap real with rho_f fixed at E19's 0.88? E21 (if so): per-condition
gain. See the 2026-11-03 batch pre-registration in EXPERIMENT_LOG.md.

    uv run python experiments/e20_gap.py e20
    uv run python experiments/e20_gap.py e21
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from clone_state_model import OUT, split, within_stats
from fit_cytokine import CONDS5, load_counts, pooled, test_stats
from scipy.optimize import minimize

RHO_F = 0.88
OBS = {"full-G": 0.69, "full-M": 0.24, "G-M": 0.51}


def simulate(cnt, beta, sigma_w, tau, rho, rng):
    """As clone_state_model.simulate, with beta a per-condition dict."""
    clones = sorted({k for days in cnt.values() for by in days.values() for k in by})
    z = dict(zip(clones, rng.standard_normal(len(clones)), strict=True))
    out = {c: {d: {} for d in days} for c, days in cnt.items()}
    for c, days in cnt.items():
        zb = {k: np.sqrt(rho) * z[k] + np.sqrt(1 - rho) * rng.standard_normal() for k in z}
        for d, by in days.items():
            for k, v in by.items():
                n = v[0] + v[1]
                p = 1 / (1 + np.exp(-(beta[c] * zb[k] + tau[c] + sigma_w * rng.standard_normal(n))))
                kn = int((rng.random(n) < p).sum())
                out[c][d][k] = [kn, n - kn, v[2]]
    return out


def unpack(x, conds, gains):
    beta0, sigma_w = abs(x[0]), abs(x[1])
    tau = dict(zip(conds, x[2 : 2 + len(conds)], strict=True))
    g = (
        dict(zip(conds, [1.0] + [abs(v) for v in x[2 + len(conds) :]], strict=True))
        if gains
        else {}
    )
    beta = {c: beta0 * g.get(c, 1.0) for c in conds}
    return beta, sigma_w, tau


def fit(gains, restarts=3, sims=3):
    cnt = load_counts(CONDS5)
    fitset = split(cnt, None)
    obs = within_stats(fitset)
    conds = list(cnt)

    def loss(x):
        beta, sigma_w, tau = unpack(x, conds, gains)
        st = [
            within_stats(simulate(fitset, beta, sigma_w, tau, RHO_F, np.random.default_rng(s)))
            for s in range(sims)
        ]
        return sum(
            (np.mean([q[c][j] for q in st]) - obs[c][j]) ** 2 * w
            for c in conds
            for j, w in ((0, 1), (1, 4), (2, 1))
        )

    best = None
    for rs in range(restarts):
        rng = np.random.default_rng(rs)
        x0 = [rng.uniform(1, 8), rng.uniform(0.1, 1.5)]
        x0 += [float(np.log(obs[c][0] / (1 - obs[c][0]))) for c in conds]
        if gains:
            x0 += [1.0] * (len(conds) - 1)
        res = minimize(
            loss, x0, method="Nelder-Mead", options={"maxiter": 3000, "xatol": 1e-3, "fatol": 1e-6}
        )
        print(json.dumps({"restart": rs, "loss": float(res.fun)}), flush=True)
        if best is None or res.fun < best.fun:
            best = res
    beta, sigma_w, tau = unpack(best.x, conds, gains)
    cross = []
    for s in range(200):
        sim = simulate(cnt, beta, sigma_w, tau, RHO_F, np.random.default_rng(1000 + s))
        cross.append(
            {
                f"{a}-{b}": test_stats(pooled(sim[a]), pooled(sim[b]), boot=1)["r"]
                for a, b in (("full", "G"), ("full", "M"), ("G", "M"))
            }
        )
    out = {"rho_f": RHO_F, "loss": float(best.fun), "beta": beta, "sigma_w": sigma_w, "tau": tau}
    for k, o in OBS.items():
        v = np.array([c[k] for c in cross])
        hi = float((v >= o).mean())
        out[k] = {
            "obs": o,
            "sim_mean": float(v.mean()),
            "sim_sd": float(v.std()),
            "sim_95": np.percentile(v, [2.5, 97.5]).tolist(),
            "p_ge_obs": hi,
            "p_two_sided": float(2 * min(hi, 1 - hi)),
        }
    return out


if __name__ == "__main__":
    name = sys.argv[1]
    res = fit(gains=name == "e21")
    json.dump(res, open(os.path.join(OUT, f"{name}.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
