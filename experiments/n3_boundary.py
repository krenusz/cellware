# QS2-N3-B (EXPERIMENT_LOG "## QS2-N3-B"). Serves: C12 CLOSED. Synthetic only; committed unchanged from the run.
"""Synthetic only: does N3's frozen statistic (within-embryo Spearman rho of log arm counts vs a binomial-thinning
null conditioned on eligibility) see tracking at all? No data read."""
import numpy as np
from scipy.stats import spearmanr

rng = np.random.default_rng(7)
EMB = (39, 38, 46)


def clones(model, n_elig, a=1.6, sg=1.0, rho=1.0, k=None):
    out = []
    while len(out) < n_elig:
        g = rng.normal(0, sg)
        S = 1 + rng.poisson(np.exp(a + g))
        if model == "notrack":  # arm-2 founders independent of S
            m = 1 + rng.poisson(np.exp(a + rng.normal(0, sg)) * 0.5)
        else:
            m = rng.binomial(S, 0.5)
        n1 = rng.binomial(S, 0.5)
        g2 = rho * g + np.sqrt(max(0, 1 - rho ** 2)) * rng.normal(0, sg) if model == "herit" else 0.0
        mu = m * np.exp(0.0 + g2 * (model == "herit"))
        n2 = rng.poisson(mu) if k is None else rng.poisson(rng.gamma(k, mu / k)) if mu > 0 else 0
        if n1 >= 1 and n2 >= 1 and n1 >= 3 and n2 >= 4:
            out.append((n1, n2))
    return np.array(out)


def stat(data):
    return np.mean([spearmanr(np.log(d[:, 0]), np.log(d[:, 1]))[0] for d in data])


def null_draw(data):
    res = []
    for d in data:
        N = d.sum(1)
        p = d[:, 0].sum() / N.sum()
        x = np.empty_like(d)
        for c, n in enumerate(N):
            while True:
                n1 = rng.binomial(n, p)
                if n1 >= 3 and n - n1 >= 4:
                    break
            x[c] = (n1, n - n1)
        res.append(x)
    return res


def reject(model, reps=60, B=199, **kw):
    up = lo = 0
    for _ in range(reps):
        data = [clones(model, n, **kw) for n in EMB]
        T = stat(data)
        null = np.array([stat(null_draw(data)) for _ in range(B)])
        up += (1 + np.sum(null >= T)) / (B + 1) < 0.05
        lo += (1 + np.sum(null <= T)) / (B + 1) < 0.05
    return up / reps, lo / reps


for name, model, kw in [("full tracking, pure sampling (g'=g, Poisson)", "track", {}),
                        ("heritable growth rho 0.8, Poisson", "herit", {"rho": 0.8}),
                        ("heritable growth rho 0.8, NB k=2", "herit", {"rho": 0.8, "k": 2}),
                        ("no tracking (founders independent)", "notrack", {})]:
    print(name, "upper/lower rejection:", reject(model, **kw))
