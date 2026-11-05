"""R2 / E51: improved SIS-skew simulator and semi-automatic ABC (see EXPERIMENT_LOG R2, E51).

    uv run python r2_skew_sbi.py bank      # 20,000 prior simulations at fit-set sizes
    uv run python r2_skew_sbi.py recover   # R2: 300 test simulations
    uv run python r2_skew_sbi.py fit       # E51: fit set, posterior + predictive check
    uv run python r2_skew_sbi.py predict   # E51: held-out predictive ranges (before the look)
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

import sis_data as sd
from b39_sisskew import GROUPS
from r1_skew_sim import N_KO, N_NULL, SIGMA_Z, size_bin_shares, stats

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "results_sis")
BANK = os.path.join(OUT, "r2_bank.npz")
PRIOR = {
    "mu_cDC2_W": (-2, 2), "mu_pDC_W": (-5, 1), "mu_cDC2_C": (-1, 4), "mu_pDC_C": (-4, 1),
    "c0": (0.5, 6), "t_W": (0.1, 0.7), "t_C": (0.1, 0.7), "q": (0, 0.6),
    "m_g": (-0.5, 3), "s_g": (0.2, 2), "s_line": (0.2, 2), "s_eps": (0.3, 2),
    "lam_L": (-2, 2), "lam_K": (-1, 4),
    "delta_cDC2": (-1.5, 1.5), "delta_pDC": (-1.5, 1.5),
    "tau_cDC2": (-2, 2), "tau_pDC": (-2, 2), "gamma_cDC2": (-1, 1), "gamma_pDC": (-1, 1),
}
NAMES = list(PRIOR)
MAIN = ["gamma_cDC2", "gamma_pDC", "tau_cDC2", "tau_pDC"]
HO_NULL, HO_KO = N_NULL // 2, N_KO // 2  # expected held-out sizes (shard 0 = half the fit set)


def _sm(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def simulate(p, n, ko, rng):
    w = "C" if ko else "W"
    mu = np.array([0.0, p[f"mu_cDC2_{w}"], p[f"mu_pDC_{w}"]])
    m = int(n * 1.8) + 60
    z = mu + rng.standard_normal((m, 3)) * SIGMA_Z
    rel = z[:, 1:] - z[:, :1]
    zs = (rel - rel.mean(0)) / rel.std(0)
    c = 1 + rng.poisson(p["c0"], m)
    b = rng.binomial(c, p[f"t_{w}"])
    g = p["m_g"] + p["s_g"] * rng.standard_normal(m)
    lam_p = p["lam_L"] + ((p["lam_K"] + p["gamma_cDC2"] * zs[:, 0] + p["gamma_pDC"] * zs[:, 1]) if ko else 0.0)
    cmax = c.max()
    line = np.exp(g[:, None] + p["s_line"] * rng.standard_normal((m, cmax)))
    line *= rng.random((m, cmax)) >= p["q"]
    idx = np.arange(cmax)[None, :]
    in_p = idx < b[:, None]
    in_n = (idx >= b[:, None]) & (idx < c[:, None])
    n_p = rng.poisson((line * in_p).sum(1) * np.exp(lam_p))
    n_n = rng.poisson((line * in_n).sum(1))
    shift = np.array([0.0, p["delta_cDC2"], p["delta_pDC"]])
    if ko:
        shift = shift + np.array([0.0, p["tau_cDC2"], p["tau_pDC"]])
    P = rng.multinomial(n_p, _sm(z + shift + p["s_eps"] * rng.standard_normal((m, 3)))).astype(float)
    N = rng.multinomial(n_n, _sm(z + p["s_eps"] * rng.standard_normal((m, 3)))).astype(float)
    keep = np.flatnonzero((P.sum(1) + N.sum(1)) > 1)[:n]
    return P[keep], N[keep]


def svec(P, N):
    return np.concatenate([stats(P, N), size_bin_shares(P, N)])


def sim_stats(p, rng, n_null=N_NULL, n_ko=N_KO):
    return np.concatenate([svec(*simulate(p, n_null, False, rng)), svec(*simulate(p, n_ko, True, rng))])


def draw(rng):
    return {k: rng.uniform(*r) for k, r in PRIOR.items()}


def observed_fit():
    d = sd.load_skew("fit")
    return np.concatenate([svec(d["P"][np.isin(d["well"], GROUPS[g])], d["N"][np.isin(d["well"], GROUPS[g])])
                           for g in ("null", "KO")])


def bank(n=20000, seed=21):
    rng = np.random.default_rng(seed)
    th, S = np.zeros((n, len(NAMES))), []
    for i in range(n):
        p = draw(rng)
        th[i] = [p[k] for k in NAMES]
        S.append(sim_stats(p, rng))
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(BANK, theta=th, S=np.array(S))
    print("bank", th.shape, np.array(S).shape)


class ABC:
    """Semi-automatic ABC: ridge projection, 2% nearest, local-linear adjustment."""

    def __init__(self, th, S, alpha=1.0, frac=0.02):
        self.th, self.frac = th, frac
        self.mx, self.sx = S.mean(0), S.std(0) + 1e-9
        Z = (S - self.mx) / self.sx
        self.my = th.mean(0)
        self.W = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (th - self.my))
        self.proj = Z @ self.W + self.my
        self.ps = self.proj.std(0) + 1e-9

    def project(self, s):
        return ((s - self.mx) / self.sx) @ self.W + self.my

    def posterior(self, s):
        t = self.project(s)
        dist = np.sqrt((((self.proj - t) / self.ps) ** 2).sum(1))
        k = int(len(dist) * self.frac)
        ix = np.argsort(dist)[:k]
        h = dist[ix].max() + 1e-12
        wts = 1 - (dist[ix] / h) ** 2
        X = np.column_stack([np.ones(k), self.proj[ix] - t])
        sw = np.sqrt(wts)[:, None]
        B = np.linalg.lstsq(X * sw, self.th[ix] * sw, rcond=None)[0]
        adj = self.th[ix] - (self.proj[ix] - t) @ B[1:]
        lo = np.array([PRIOR[k][0] for k in NAMES])
        hi = np.array([PRIOR[k][1] for k in NAMES])
        return np.clip(adj, lo, hi), wts, ix


def summarise(adj, wts):
    out = {}
    for j, nm in enumerate(NAMES):
        o = np.argsort(adj[:, j])
        cw = np.cumsum(wts[o]) / wts.sum()
        q = lambda a: float(adj[o, j][np.searchsorted(cw, a)])  # noqa: E731
        out[nm] = {"median": q(0.5), "lo": q(0.025), "hi": q(0.975)}
    return out


def load_abc(exclude=0):
    z = np.load(BANK)
    th, S = z["theta"], z["S"]
    return ABC(th[exclude:], S[exclude:]), th, S


def recover(n_test=300, seed=22):
    rng = np.random.default_rng(seed)
    abc, _, _ = load_abc()
    rows = []
    for _ in range(n_test):
        p = draw(rng)
        s = sim_stats(p, rng)
        post = summarise(*abc.posterior(s)[:2])
        rows.append((p, post))
    res = {"n_test": n_test}
    for nm in NAMES:
        tru = np.array([p[nm] for p, _ in rows])
        med = np.array([q[nm]["median"] for _, q in rows])
        cov = np.mean([q[nm]["lo"] <= p[nm] <= q[nm]["hi"] for p, q in rows])
        width = np.mean([q[nm]["hi"] - q[nm]["lo"] for _, q in rows])
        res[nm] = {"RMSE": float(np.sqrt(np.mean((med - tru) ** 2))), "coverage95": float(cov),
                   "mean_width": float(width), "R2": float(1 - np.mean((med - tru) ** 2) / np.var(tru))}
    for nm in MAIN:
        r = res[nm]
        r["reading"] = ("recoverable" if r["RMSE"] <= 0.3 and 0.90 <= r["coverage95"] <= 0.99
                        else "miscalibrated" if r["coverage95"] < 0.90 else "not recoverable")
    json.dump(res, open(os.path.join(OUT, "r2_recovery.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in MAIN}, indent=1))


def predictive(adj, wts, rng, n_draws, n_null, n_ko):
    ix = rng.choice(len(adj), n_draws, p=wts / wts.sum())
    sims = np.array([sim_stats(dict(zip(NAMES, adj[i])), rng, n_null, n_ko) for i in ix])
    return np.percentile(sims, 2.5, axis=0), np.percentile(sims, 97.5, axis=0)


def fit(seed=23):
    rng = np.random.default_rng(seed)
    abc, _, _ = load_abc()
    s = observed_fit()
    adj, wts, _ = abc.posterior(s)
    post = summarise(adj, wts)
    lo, hi = predictive(adj, wts, rng, 200, N_NULL, N_KO)
    inside = (s >= lo) & (s <= hi)
    res = {"posterior": post, "fit_adequacy_share_inside": float(inside.mean()),
           "misses": [int(i) for i in np.flatnonzero(~inside)],
           "obs": s.tolist(), "pp_lo": lo.tolist(), "pp_hi": hi.tolist()}
    g = {nm: post[nm] for nm in ("gamma_cDC2", "gamma_pDC")}
    res["reading_selection"] = ("selection" if any(v["lo"] > 0 or v["hi"] < 0 for v in g.values())
                                else "none (negligible)" if all(-0.25 <= v["lo"] and v["hi"] <= 0.25 for v in g.values())
                                else "undetermined")
    np.savez_compressed(os.path.join(OUT, "e51_posterior.npz"), adj=adj, wts=wts)
    json.dump(res, open(os.path.join(OUT, "e51_fit.json"), "w"), indent=1)
    print(json.dumps({"posterior": post, "adequacy": res["fit_adequacy_share_inside"],
                      "misses": res["misses"], "reading_selection": res["reading_selection"]}, indent=1))


# held-out statistics: index within the 76-vector (null block 0-37, KO block 38-75)
def heldout_index():
    dec = list(range(38 + 6, 38 + 15))                      # KO decomposition (9)
    rho = list(range(15, 18)) + list(range(38 + 15, 38 + 18))  # size-ratio rho null + KO (6)
    one = list(range(18, 22)) + list(range(38 + 18, 38 + 22))  # one-arm by size null + KO (8)
    pers = list(range(38 + 26, 38 + 32))                    # KO persistence-by-fate (6)
    return dec + rho + one + pers


def predict(seed=24):
    rng = np.random.default_rng(seed)
    z = np.load(os.path.join(OUT, "e51_posterior.npz"))
    lo, hi = predictive(z["adj"], z["wts"], rng, 400, HO_NULL, HO_KO)
    ix = heldout_index()
    res = {"n_null": HO_NULL, "n_ko": HO_KO, "index": ix,
           "lo": [float(lo[i]) for i in ix], "hi": [float(hi[i]) for i in ix]}
    json.dump(res, open(os.path.join(OUT, "e51_heldout_prediction.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bank", "recover", "fit", "predict"])
    a = ap.parse_args()
    {"bank": bank, "recover": recover, "fit": fit, "predict": predict}[a.cmd]()
