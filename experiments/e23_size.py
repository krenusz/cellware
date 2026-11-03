"""E23: clone size vs N-share coupling per condition (data vs E20 model); model E23m if needed.
See the E23 pre-registration in EXPERIMENT_LOG.md.

    uv run python experiments/e23_size.py diag
    uv run python experiments/e23_size.py model
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
from scipy.stats import spearmanr


def share_size(cnt_c):
    p = pooled(cnt_c)
    v = np.array(
        [(a[0] / (a[0] + a[1]), np.log(a[0] + a[1])) for a in p.values() if a[0] + a[1] >= 2]
    )
    return v


def coupling(cnt):
    return {c: float(spearmanr(*share_size(cnt[c]).T)[0]) for c in cnt}


def e20_params(conds):
    e = json.load(open(os.path.join(OUT, "e20.json")))
    return e["beta"], e["sigma_w"], e["tau"]


def diag():
    cnt = load_counts(CONDS5)
    conds = list(cnt)
    rng = np.random.default_rng(0)
    out = {}
    for c in conds:
        v = share_size(cnt[c])
        bs = [spearmanr(*v[rng.integers(0, len(v), len(v))].T)[0] for _ in range(2000)]
        out[c] = {
            "clones": len(v),
            "rho": float(spearmanr(*v.T)[0]),
            "ci": np.percentile(bs, [2.5, 97.5]).tolist(),
        }
    beta, sigma_w, tau = e20_params(conds)
    sims = [
        coupling(sim20(cnt, beta, sigma_w, tau, RHO_F, np.random.default_rng(1000 + s)))
        for s in range(200)
    ]
    for c in conds:
        v = np.array([s[c] for s in sims])
        out[c]["e20_sim_mean"] = float(v.mean())
        out[c]["e20_sim_95"] = np.percentile(v, [2.5, 97.5]).tolist()
        lo, hi = out[c]["e20_sim_95"]
        out[c]["trigger"] = bool(
            (out[c]["ci"][0] > 0 or out[c]["ci"][1] < 0) and not lo <= out[c]["rho"] <= hi
        )
    out["run_model"] = any(out[c]["trigger"] for c in conds)
    json.dump(out, open(os.path.join(OUT, "e23_diag.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


def sizes(cnt):
    """Standardised log(N+M) per clone within each condition (pooled over days)."""
    s = {}
    for c, days in cnt.items():
        p = pooled(days)
        lg = {k: np.log(max(v[0] + v[1], 1)) for k, v in p.items()}
        mu, sd = np.mean(list(lg.values())), np.std(list(lg.values())) + 1e-9
        s[c] = {k: (v - mu) / sd for k, v in lg.items()}
    return s


def simulate(cnt, beta, sigma_w, tau, delta, size, rng):
    """e20_gap.simulate plus delta_c * standardised clone size; delta = 0 reproduces it."""
    clones = sorted({k for days in cnt.values() for by in days.values() for k in by})
    z = dict(zip(clones, rng.standard_normal(len(clones)), strict=True))
    out = {c: {d: {} for d in days} for c, days in cnt.items()}
    for c, days in cnt.items():
        zb = {k: np.sqrt(RHO_F) * z[k] + np.sqrt(1 - RHO_F) * rng.standard_normal() for k in z}
        for d, by in days.items():
            for k, v in by.items():
                n = v[0] + v[1]
                lin = (
                    beta[c] * zb[k]
                    + delta[c] * size[c].get(k, 0.0)
                    + tau[c]
                    + sigma_w * rng.standard_normal(n)
                )
                kn = int((rng.random(n) < 1 / (1 + np.exp(-lin))).sum())
                out[c][d][k] = [kn, n - kn, v[2]]
    return out


def model():
    cnt = load_counts(CONDS5)
    fitset = split(cnt, None)
    obs, obs_cp = within_stats(fitset), coupling(fitset)
    conds = list(cnt)
    size_fit, size_all = sizes(fitset), sizes(cnt)
    beta0, sigma0, tau0 = e20_params(conds)

    def unpack(x):
        b = abs(x[0])
        return (
            {c: b for c in conds},
            abs(x[1]),
            dict(zip(conds, x[2:7], strict=True)),
            dict(zip(conds, x[7:12], strict=True)),
        )

    def loss(x):
        beta, sw, tau, delta = unpack(x)
        sims = [
            simulate(fitset, beta, sw, tau, delta, size_fit, np.random.default_rng(s))
            for s in range(5)
        ]
        st, cp = [within_stats(q) for q in sims], [coupling(q) for q in sims]
        base = sum(
            (np.mean([q[c][j] for q in st]) - obs[c][j]) ** 2 * w
            for c in conds
            for j, w in ((0, 1), (1, 4), (2, 1))
        )
        return base + sum((np.mean([q[c] for q in cp]) - obs_cp[c]) ** 2 for c in conds)

    x0 = [beta0["full"], sigma0] + [tau0[c] for c in conds] + [0.0] * len(conds)
    ref = loss(np.array(x0))
    res = minimize(
        loss, x0, method="Nelder-Mead", options={"maxiter": 1500, "xatol": 1e-3, "fatol": 1e-6}
    )
    print(json.dumps({"stage": "nm", "loss": float(res.fun), "ref_delta0": float(ref)}), flush=True)
    res = minimize(
        loss, res.x, method="Powell", options={"maxfev": 1500, "xtol": 1e-3, "ftol": 1e-6}
    )
    beta, sw, tau, delta = unpack(res.x)
    cross = []
    for s in range(200):
        q = simulate(cnt, beta, sw, tau, delta, size_all, np.random.default_rng(1000 + s))
        cross.append(
            {
                f"{a}-{b}": test_stats(pooled(q[a]), pooled(q[b]), boot=1)["r"]
                for a, b in (("full", "G"), ("full", "M"), ("G", "M"))
            }
        )
    rep = {
        "loss": float(res.fun),
        "ref_delta0_loss": float(ref),
        "beta": beta["full"],
        "sigma_w": sw,
        "tau": tau,
        "delta": delta,
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
    json.dump(rep, open(os.path.join(OUT, "e23_model.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    diag() if sys.argv[1] == "diag" else model()
