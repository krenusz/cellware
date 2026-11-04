"""B21: fluctuation-test analysis of fate memory (beta-binomial overdispersion, replicate
libraries, day-4 vs day-6 decay). See the B21 pre-registration in EXPERIMENT_LOG.md.

    uv run python experiments/track_b21.py
"""

from __future__ import annotations

import gzip
import itertools
import json
import os

import numpy as np
from clone_state_model import OUT
from fit_cytokine import CONDS, DATA
from scipy.optimize import minimize
from scipy.stats import betabinom

BOOT = 1000


def by_library():
    """{(cond, day, library): {clone: [N, M]}}, mature N and M cells only."""
    with gzip.open(os.path.join(DATA, "metadata.txt.gz"), "rt") as f:
        rows = [line.rstrip("\n").split("\t") for line in f][1:]
    with gzip.open(os.path.join(DATA, "clone_matrix.mtx.gz"), "rt") as f:
        m = np.loadtxt(f, comments="%", skiprows=3, dtype=int)
    out = {}
    for clone, cell in zip(m[:, 0] - 1, m[:, 1] - 1, strict=True):
        lib, day, kind, cond = rows[cell][0], rows[cell][2], rows[cell][3], rows[cell][4]
        if day == "2" or kind not in ("Neu", "Mo") or cond not in CONDS:
            continue
        key = (CONDS[cond], int(day), lib)
        out.setdefault(key, {}).setdefault(int(clone), [0, 0])[kind == "Mo"] += 1
    return out


def merge(libs):
    tot = {}
    for d in libs:
        for c, v in d.items():
            t = tot.setdefault(c, [0, 0])
            t[0] += v[0]
            t[1] += v[1]
    return tot


def bb_rho(k, n):
    def nll(p):
        mu, rho = 1 / (1 + np.exp(-p[0])), 1 / (1 + np.exp(-p[1]))
        a, b = mu * (1 - rho) / rho, (1 - mu) * (1 - rho) / rho
        return -betabinom.logpmf(k, n, a, b).sum()

    res = minimize(nll, [0.0, -1.0], method="Nelder-Mead")
    return float(1 / (1 + np.exp(-res.x[1])))


def t1(cnt, rng):
    v = np.array([x for x in cnt.values() if sum(x) >= 2])
    k, n = v[:, 0], v.sum(1)
    est = bb_rho(k, n)
    bs = [bb_rho(k[i], n[i]) for i in (rng.integers(len(k), size=len(k)) for _ in range(BOOT))]
    return {
        "rho": est,
        "ci": np.percentile(bs, [2.5, 97.5]).tolist(),
        "clones": len(k),
        "mean_N_share": float(k.sum() / n.sum()),
    }


def pairs(a, b):
    out = {}
    for c in set(a) & set(b):
        x, y = a[c], b[c]
        if sum(x) >= 2 and sum(y) >= 2:
            out.setdefault(c, []).append((x[0] / sum(x), y[0] / sum(y)))
    return out


def r_rows(rows):
    p = np.array([q for v in rows for q in v])
    return float(np.corrcoef(p[:, 0], p[:, 1])[0, 1]) if len(p) >= 3 else np.nan


def boot_r(rows, rng):
    keys = list(rows)
    if len(keys) < 3:
        return [np.nan, np.nan], np.array([])
    bs = np.array(
        [
            r_rows([rows[keys[i]] for i in rng.integers(len(keys), size=len(keys))])
            for _ in range(BOOT)
        ]
    )
    return np.nanpercentile(bs, [2.5, 97.5]).tolist(), bs


def main():
    lib = by_library()
    rng = np.random.default_rng(0)
    rep = {"libraries": {f"{c}{d}:{name}": len(v) for (c, d, name), v in sorted(lib.items())}}
    groups = {}
    for (c, d, _name), v in lib.items():
        groups.setdefault((c, d), []).append(v)
    rep["T1"] = {f"{c}{d}": t1(merge(v), rng) for (c, d), v in sorted(groups.items())}
    rep["T2"], boots = {}, {}
    for (c, d), v in sorted(groups.items()):
        if len(v) < 2:
            continue
        rows = {}
        for a, b in itertools.combinations(v, 2):
            for k, q in pairs(a, b).items():
                rows.setdefault(k, []).extend(q)
        ci, bs = boot_r(rows, rng)
        rep["T2"][f"{c}{d}"] = {"r": r_rows(rows.values()), "ci": ci, "clones": len(rows)}
        boots[(c, d)] = bs
    rep["T3"] = {}
    for c in ("G", "M"):
        rows = pairs(merge(groups[(c, 4)]), merge(groups[(c, 6)]))
        ci, bs = boot_r(rows, rng)
        r46 = r_rows(rows.values())
        same = np.nanmean([rep["T2"][f"{c}{d}"]["r"] for d in (4, 6)])
        bsame = np.nanmean([boots[(c, 4)][: len(bs)], boots[(c, 6)][: len(bs)]], axis=0)
        ratio = bs / bsame
        R = r46 / same

        def half(x):
            return 2 * np.log(2) / -np.log(x) if 0 < x < 1 else np.inf

        rep["T3"][c] = {
            "r_d4_d6": r46,
            "r_d4_d6_ci": ci,
            "clones": len(rows),
            "r_same_day": float(same),
            "R": float(R),
            "R_ci": np.nanpercentile(ratio, [2.5, 97.5]).tolist(),
            "half_life_days": float(half(R)),
            "half_life_ci": [float(half(v)) for v in np.nanpercentile(ratio, [2.5, 97.5])],
        }
    for d in (4, 6):
        rows = pairs(merge(groups[("G", d)]), merge(groups[("M", d)]))
        rep.setdefault("T4", {})[f"G{d}-M{d}"] = {"r": r_rows(rows.values()), "clones": len(rows)}
    json.dump(rep, open(os.path.join(OUT, "b21_fluctuation.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "libraries"}, indent=1))


if __name__ == "__main__":
    main()
