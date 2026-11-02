"""E19: kin-correlation estimate of fate-state sharing (rho_f) in LARRY in vitro.
See the E19 pre-registration in EXPERIMENT_LOG.md.

    uv run python experiments/kin_corr_e19.py
"""

from __future__ import annotations

import gzip
import json
import os

import numpy as np
from scipy.optimize import brentq
from scipy.stats import multivariate_normal, norm

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "larry")


def load():
    """{clone: {(day, well): [N, M]}} for mature N / M cells at days 4 and 6."""
    with gzip.open(os.path.join(DATA, "metadata.txt.gz"), "rt") as f:
        rows = [line.rstrip("\n").split("\t") for line in f][1:]
    with gzip.open(os.path.join(DATA, "clone_matrix.mtx.gz"), "rt") as f:
        m = np.loadtxt(f, comments="%", skiprows=3, dtype=int)  # cells x clones
    out = {}
    for cell, clone in zip(m[:, 0] - 1, m[:, 1] - 1, strict=True):
        day, kind, well = rows[cell][2], rows[cell][4], rows[cell][5]
        if day == "2.0" or kind not in ("Neutrophil", "Monocyte"):
            continue
        k = out.setdefault(int(clone), {}).setdefault((day, well), [0, 0])
        k[0 if kind == "Neutrophil" else 1] += 1
    return {c: g for c, g in out.items() if len({w for _, w in g}) >= 2}


def tables(groups, same_day_cross=False):
    """Per clone, normalised pair tables [NN, NM+MN, MM]: same well + same day, and cross-well."""
    same, cross = {}, {}
    for c, g in groups.items():
        s = np.zeros(3)
        for n, mm in g.values():
            s += [n * (n - 1) / 2, n * mm, mm * (mm - 1) / 2]
        x = np.zeros(3)
        keys = list(g)
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                (da, wa), (db, wb) = keys[a], keys[b]
                if wa == wb or (same_day_cross and da != db):
                    continue
                (n1, m1), (n2, m2) = g[keys[a]], g[keys[b]]
                x += [n1 * n2, n1 * m2 + m1 * n2, m1 * m2]
        if s.sum() > 0 and x.sum() > 0:
            same[c], cross[c] = s / s.sum(), x / x.sum()
    return same, cross


def tetrachoric(t):
    """Latent correlation from a pooled symmetric pair table [NN, discordant, MM]."""
    p_n = t[0] + t[1] / 2
    th = norm.ppf(1 - p_n)

    def f(r):
        return multivariate_normal([0, 0], [[1, r], [r, 1]]).cdf([-th, -th]) - t[0]

    return float(brentq(f, -0.99, 0.999))


def estimate(same, cross, clones):
    s = np.mean([same[c] for c in clones], 0)
    x = np.mean([cross[c] for c in clones], 0)
    rs, rx = tetrachoric(s), tetrachoric(x)
    return rs, rx, rx / rs


def run(groups, same_day_cross, rng, boot=1000):
    same, cross = tables(groups, same_day_cross)
    clones = sorted(same)
    rs, rx, ratio = estimate(same, cross, clones)
    bs = []
    for _ in range(boot):
        pick = list(rng.choice(clones, len(clones)))
        try:
            bs.append(estimate(same, cross, pick))
        except ValueError:
            pass
    bs = np.array(bs)
    ci = lambda k: np.percentile(bs[:, k], [2.5, 97.5]).tolist()  # noqa: E731
    return {
        "clones": len(clones),
        "r_same_well": rs,
        "r_same_well_ci": ci(0),
        "r_cross_well": rx,
        "r_cross_well_ci": ci(1),
        "rho_f_hat": ratio,
        "rho_f_hat_ci": ci(2),
    }


def main():
    rng = np.random.default_rng(0)
    groups = load()
    out = {"primary_cross_any_day": run(groups, False, rng)}
    out["secondary_cross_same_day"] = run(groups, True, rng)
    json.dump(out, open(os.path.join(HERE, "results_e18", "e19_kin.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
