"""E25: mixture of clone types (fixed / M-only / plastic). B7: are the G-M clones a selected subset?
See the 2026-11-03 (3) batch pre-registration in EXPERIMENT_LOG.md.

    uv run python experiments/e25_clone_types.py b7
    uv run python experiments/e25_clone_types.py e25
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from clone_state_model import OUT, split, within_stats
from e20_gap import OBS, RHO_F
from e20_gap import simulate as sim20
from fit_cytokine import CONDS5, load_counts, pooled, test_stats
from scipy.optimize import minimize
from scipy.stats import mannwhitneyu


def simulate(cnt, beta, sigma_w, tau, pi_fix, pi_m, rng, crng):
    """e20_gap.simulate with clone classes drawn from crng; pi = 0 reproduces it exactly."""
    clones = sorted({k for days in cnt.values() for by in days.values() for k in by})
    z = dict(zip(clones, rng.standard_normal(len(clones)), strict=True))
    u = crng.random(len(clones))
    kind = {
        k: ("m" if v < pi_m else "fix" if v < pi_m + pi_fix else "plastic")
        for k, v in zip(clones, u, strict=True)
    }
    out = {c: {d: {} for d in days} for c, days in cnt.items()}
    for c, days in cnt.items():
        zb = {k: np.sqrt(RHO_F) * z[k] + np.sqrt(1 - RHO_F) * rng.standard_normal() for k in z}
        for d, by in days.items():
            for k, v in by.items():
                n = v[0] + v[1]
                t = tau["full"] if kind[k] == "fix" else tau[c]
                lin = beta[c] * zb[k] + t + sigma_w * rng.standard_normal(n)
                draw = rng.random(n)
                kn = 0 if kind[k] == "m" else int((draw < 1 / (1 + np.exp(-lin))).sum())
                out[c][d][k] = [kn, n - kn, v[2]]
    return out


def nshare(p, keys):
    return np.array([p[k][0] / (p[k][0] + p[k][1]) for k in keys])


def groups(cnt):
    pg, pm = pooled(cnt["G"]), pooled(cnt["M"])
    g_ok = {k for k, v in pg.items() if v[0] + v[1] >= 2}
    m_ok = {k for k, v in pm.items() if v[0] + v[1] >= 2}
    return pg, pm, g_ok & m_ok, g_ok - m_ok, m_ok - g_ok


def presence_stats(cnt):
    pg, pm, both, g_only, m_only = groups(cnt)
    return {
        "G_both_minus_Gonly": float(np.median(nshare(pg, both)) - np.median(nshare(pg, g_only))),
        "M_both_minus_Monly": float(np.mean(nshare(pm, both)) - np.mean(nshare(pm, m_only))),
        "G_mean_both_minus_Gonly": float(np.mean(nshare(pg, both)) - np.mean(nshare(pg, g_only))),
    }


def b7():
    cnt = load_counts(CONDS5)
    pg, pm, both, g_only, m_only = groups(cnt)
    out = {"n_both": len(both), "n_G_only": len(g_only), "n_M_only": len(m_only)}
    for name, p, a, b in (("G", pg, both, g_only), ("M", pm, both, m_only)):
        x, y = nshare(p, a), nshare(p, b)
        out[f"Nshare_in_{name}"] = {
            "both_median": float(np.median(x)),
            "both_mean": float(x.mean()),
            "single_median": float(np.median(y)),
            "single_mean": float(y.mean()),
            "mannwhitney_p": float(mannwhitneyu(x, y).pvalue),
            "both_size_median": float(np.median([p[k][0] + p[k][1] for k in a])),
            "single_size_median": float(np.median([p[k][0] + p[k][1] for k in b])),
        }
    obs = presence_stats(cnt)
    e = json.load(open(os.path.join(OUT, "e20.json")))
    sims = [
        presence_stats(
            sim20(cnt, e["beta"], e["sigma_w"], e["tau"], RHO_F, np.random.default_rng(1000 + s))
        )
        for s in range(200)
    ]
    for k, v in obs.items():
        a = np.array([s[k] for s in sims])
        lo, hi = np.percentile(a, [2.5, 97.5])
        out[k] = {
            "data": v,
            "e20_mean": float(a.mean()),
            "e20_95": [float(lo), float(hi)],
            "outside": bool(not lo <= v <= hi),
        }
    json.dump(out, open(os.path.join(OUT, "b7_selection.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


def e25():
    cnt = load_counts(CONDS5)
    fitset = split(cnt, None)
    obs = within_stats(fitset)
    conds = list(cnt)
    e = json.load(open(os.path.join(OUT, "e20.json")))

    def unpack(x):
        b = abs(x[0])
        pf, pm = 1 / (1 + np.exp(-x[7])), 1 / (1 + np.exp(-x[8]))
        s = pf + pm
        if s > 0.99:
            pf, pm = pf / s * 0.99, pm / s * 0.99
        return (
            {c: b for c in conds},
            abs(x[1]),
            dict(zip(conds, x[2:7], strict=True)),
            float(pf),
            float(pm),
        )

    def run(fs, params, s):
        beta, sw, tau, pf, pm = params
        return simulate(
            fs, beta, sw, tau, pf, pm, np.random.default_rng(s), np.random.default_rng(30_000 + s)
        )

    def loss_p(params):
        st = [within_stats(run(fitset, params, s)) for s in range(5)]
        return sum(
            (np.mean([q[c][j] for q in st]) - obs[c][j]) ** 2 * w
            for c in conds
            for j, w in ((0, 1), (1, 4), (2, 1))
        )

    x0 = [e["beta"]["full"], e["sigma_w"]] + [e["tau"][c] for c in conds]
    ref = loss_p(
        ({c: x0[0] for c in conds}, x0[1], dict(zip(conds, x0[2:7], strict=True)), 0.0, 0.0)
    )
    best = None
    for start in ([-3.0, -3.0], [0.0, -3.0], [-3.0, -1.0]):  # pi ~ 0, fixed-heavy, M-only-heavy
        res = minimize(
            lambda x: loss_p(unpack(x)),
            x0 + start,
            method="Nelder-Mead",
            options={"maxiter": 1200, "xatol": 1e-3, "fatol": 1e-6},
        )
        res = minimize(
            lambda x: loss_p(unpack(x)),
            res.x,
            method="Powell",
            options={"maxfev": 1200, "xtol": 1e-3, "ftol": 1e-6},
        )
        print(
            json.dumps(
                {
                    "start": start,
                    "loss": float(res.fun),
                    "pi": unpack(res.x)[3:],
                    "ref_e20": float(ref),
                }
            ),
            flush=True,
        )
        if best is None or res.fun < best.fun:
            best = res
    params = unpack(best.x)
    cross = []
    for s in range(200):
        q = run(cnt, params, 1000 + s)
        cross.append(
            {
                f"{a}-{b}": test_stats(pooled(q[a]), pooled(q[b]), boot=1)["r"]
                for a, b in (("full", "G"), ("full", "M"), ("G", "M"))
            }
        )
    beta, sw, tau, pf, pm = params
    rep = {
        "loss": float(best.fun),
        "ref_e20_loss": float(ref),
        "beta": beta["full"],
        "sigma_w": sw,
        "tau": tau,
        "pi_fix": pf,
        "pi_M_only": pm,
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
    json.dump(rep, open(os.path.join(OUT, "e25_model.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    b7() if sys.argv[1] == "b7" else e25()
