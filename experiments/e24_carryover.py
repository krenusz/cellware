"""E24: cells committed before the day-2 split carry their fate into every condition.
See the E24 pre-registration in EXPERIMENT_LOG.md.

    uv run python experiments/e24_carryover.py diag
    uv run python experiments/e24_carryover.py model
"""

from __future__ import annotations

import gzip
import json
import os
import sys

import numpy as np
from clone_state_model import OUT, split, within_stats
from e20_gap import OBS, RHO_F
from fit_cytokine import CONDS5, DATA, load_counts, pooled, test_stats
from scipy.optimize import minimize
from scipy.stats import spearmanr


def day2_committed():
    """{clone: [Neu, Mo]} among day-2 cells (before the split)."""
    with gzip.open(os.path.join(DATA, "metadata.txt.gz"), "rt") as f:
        rows = [line.rstrip("\n").split("\t") for line in f][1:]
    with gzip.open(os.path.join(DATA, "clone_matrix.mtx.gz"), "rt") as f:
        m = np.loadtxt(f, comments="%", skiprows=3, dtype=int)  # same layout as load_counts
    out = {}
    for clone, cell in zip(m[:, 0] - 1, m[:, 1] - 1, strict=True):
        if rows[cell][2] == "2" and rows[cell][3] in ("Neu", "Mo"):
            out.setdefault(clone, [0, 0])[0 if rows[cell][3] == "Neu" else 1] += 1
    return out


def diag():
    cnt = load_counts(CONDS5)
    d2 = day2_committed()
    pg, pm = pooled(cnt["G"]), pooled(cnt["M"])
    rng = np.random.default_rng(0)
    out = {"clones_with_day2_neu_mo": len(d2)}
    for name, p in (("G", pg), ("M", pm)):
        v = np.array(
            [
                (d2[k][0] / sum(d2[k]), p[k][0] / (p[k][0] + p[k][1]))
                for k in d2
                if sum(d2[k]) >= 2 and k in p and p[k][0] + p[k][1] >= 2
            ]
        )
        bs = [spearmanr(*v[rng.integers(0, len(v), len(v))].T)[0] for _ in range(2000)]
        out[f"day2_vs_{name}"] = {
            "clones": len(v),
            "rho": float(spearmanr(*v.T)[0]),
            "ci": np.percentile(bs, [2.5, 97.5]).tolist(),
        }
    for name, keep in (
        ("no_day2_committed", lambda k: k not in d2),
        ("with_day2_committed", lambda k: k in d2),
    ):
        a = {k: v for k, v in pg.items() if keep(k)}
        b = {k: v for k, v in pm.items() if keep(k)}
        t = test_stats(a, b, boot=2000)
        out[f"GM_{name}"] = {"clones": t["clones"], "r": t["r"], "ci": t["r_ci"]}
    json.dump(out, open(os.path.join(OUT, "e24_diag.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


def simulate(cnt, beta, sigma_w, tau, phi, rng, crng):
    """e20_gap.simulate plus carry-over; phi = 0 reproduces it exactly (extra draws use crng)."""
    clones = sorted({k for days in cnt.values() for by in days.values() for k in by})
    z = dict(zip(clones, rng.standard_normal(len(clones)), strict=True))
    out = {c: {d: {} for d in days} for c, days in cnt.items()}
    zfull = None
    for c, days in cnt.items():
        zb = {k: np.sqrt(RHO_F) * z[k] + np.sqrt(1 - RHO_F) * rng.standard_normal() for k in z}
        if c == "full":
            zfull = zb
        for d, by in days.items():
            for k, v in by.items():
                n = v[0] + v[1]
                lin = beta[c] * zb[k] + tau[c] + sigma_w * rng.standard_normal(n)
                u = rng.random(n)
                if phi[c] > 0:
                    carried = crng.random(n) < phi[c]
                    lin_full = (
                        beta["full"] * zfull[k] + tau["full"] + sigma_w * crng.standard_normal(n)
                    )
                    lin = np.where(carried, lin_full, lin)
                kn = int((u < 1 / (1 + np.exp(-lin))).sum())
                out[c][d][k] = [kn, n - kn, v[2]]
    return out


def model():
    cnt = load_counts(CONDS5)
    fitset = split(cnt, None)
    obs = within_stats(fitset)
    conds = list(cnt)
    e20 = json.load(open(os.path.join(OUT, "e20.json")))

    def unpack(x):
        b = abs(x[0])
        return (
            {c: b for c in conds},
            abs(x[1]),
            dict(zip(conds, x[2:7], strict=True)),
            {c: float(1 / (1 + np.exp(-v))) for c, v in zip(conds, x[7:12], strict=True)},
        )

    def run(fs, params, s):
        beta, sw, tau, phi = params
        return simulate(
            fs, beta, sw, tau, phi, np.random.default_rng(s), np.random.default_rng(20_000 + s)
        )

    def loss_p(params):
        st = [within_stats(run(fitset, params, s)) for s in range(5)]
        return sum(
            (np.mean([q[c][j] for q in st]) - obs[c][j]) ** 2 * w
            for c in conds
            for j, w in ((0, 1), (1, 4), (2, 1))
        )

    x0 = [e20["beta"]["full"], e20["sigma_w"]] + [e20["tau"][c] for c in conds]
    ref = loss_p(
        (
            {c: x0[0] for c in conds},
            x0[1],
            dict(zip(conds, x0[2:7], strict=True)),
            dict.fromkeys(conds, 0.0),
        )
    )
    x0 += [-3.0] * len(conds)
    res = minimize(
        lambda x: loss_p(unpack(x)),
        x0,
        method="Nelder-Mead",
        options={"maxiter": 1500, "xatol": 1e-3, "fatol": 1e-6},
    )
    print(json.dumps({"stage": "nm", "loss": float(res.fun), "ref_e20": float(ref)}), flush=True)
    res = minimize(
        lambda x: loss_p(unpack(x)),
        res.x,
        method="Powell",
        options={"maxfev": 1500, "xtol": 1e-3, "ftol": 1e-6},
    )
    params = unpack(res.x)
    cross = []
    for s in range(200):
        q = run(cnt, params, 1000 + s)
        cross.append(
            {
                f"{a}-{b}": test_stats(pooled(q[a]), pooled(q[b]), boot=1)["r"]
                for a, b in (("full", "G"), ("full", "M"), ("G", "M"))
            }
        )
    beta, sw, tau, phi = params
    rep = {
        "loss": float(res.fun),
        "ref_e20_loss": float(ref),
        "beta": beta["full"],
        "sigma_w": sw,
        "tau": tau,
        "phi": phi,
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
    json.dump(rep, open(os.path.join(OUT, "e24_model.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    diag() if sys.argv[1] == "diag" else model()
