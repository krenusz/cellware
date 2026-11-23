"""CAND-I-T1: train-only test of the clone size vs drift link (pre-registered in EXPERIMENT_LOG, 2026-11-23).

Bence: "Yes, train look". Runs a synthetic gate first and opens the table only if it passes. Rows of non-train fish are
removed by partitions.train_only before any read count is parsed. Control fish never match the identifier pattern.

For each eligible train fish: rank correlation between clone size and a normalised drift score, minus the same
correlation under two neutral nulls (replicate-calibrated noise; multinomial noise). The fish is the test unit.
"""

import csv
import hashlib
import io
import itertools
import json
import os
import re
import sys
import urllib.request

import numpy as np
import pandas as pd
from scipy.special import digamma

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from partitions import train_only  # noqa: E402

OUT = os.path.join(HERE, "results_cand", "candi_train.json")
COMMIT = "99b327ca17bfd092063dbf911e088f993e036172"
SHA256 = "6985096f252b99bd2e53e98daef23089a40bdc52345db4eaf6095105fc08de80"
URL = f"https://raw.githubusercontent.com/Gagnon-lab/weber-shrader/{COMMIT}/experiments/full_pipeline/final_data.csv"
ID = re.compile(r"fish(\d+)_s(\d+)_rep(\d+)", re.I)
MIN_T, MIN_SIZE, MIN_CLONES, MIN_FISH, ALPHA, SEED = 6, 0.005, 5, 6, 0.05, 20261123
NE_GRID = np.logspace(1, 4.5, 15)
N_CAL, N_NULL = 20, 300
GATE_COHORTS, GATE_CAL, GATE_NULL = 200, 3, 30
MAX_FALSE_POS, MIN_RECOVERY, MAX_R2_UNDER_LINK = 0.10, 0.5, 0.10
# (barcodes, sample indices) of the 8 eligible train fish, from CAND-I-T0 (structure only)
SHAPES = ((12, 11), (15, 12), (15, 6), (7, 11), (7, 16), (13, 9), (10, 17), (11, 6))


def rank(x):
    return pd.Series(x).rank().to_numpy()


def score(X):
    """(rank correlation of log size with drift score, mean drift score) over clones with mean frequency >= MIN_SIZE."""
    size = X.mean(axis=1)
    keep = size >= MIN_SIZE
    if keep.sum() < MIN_CLONES:
        return np.nan, np.nan
    a, b = X[keep, :-1], X[keep, 1:]
    m = (a + b) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where((m > 0) & (m < 1), (b - a) ** 2 / (m * (1 - m)), np.nan)
    ok = np.isfinite(f).any(axis=1)
    if ok.sum() < MIN_CLONES:
        return np.nan, np.nan
    d = np.nanmean(f[ok], axis=1)
    rx, ry = rank(np.log(size[keep][ok])), rank(d)
    if rx.std() == 0 or ry.std() == 0:
        return np.nan, float(d.mean())
    return float(np.corrcoef(rx, ry)[0, 1]), float(d.mean())


def fit_noise(reps):
    """log replicate variance = a + b log mean, over replicated clone-sample cells with mean >= MIN_SIZE / 5."""
    n = np.sum(np.isfinite(reps), axis=2)
    use = n >= 2
    r = reps[use]
    mu, s2 = np.nanmean(r, axis=1), np.nanvar(r, axis=1, ddof=1)
    ok = (mu >= MIN_SIZE / 5) & (s2 > 0)
    if ok.sum() < 10:
        raise ValueError("too few replicated cells")
    # E[log s2] = log var + digamma(df / 2) - log(df / 2): undo this bias so that noise is not underestimated
    df = n[use][ok] - 1
    bias = digamma(df / 2) - np.log(df / 2)
    b, a = np.polyfit(np.log(mu[ok]), np.log(s2[ok]) - bias, 1)
    return float(a), float(np.clip(b, 0, 2.5))


def drift(x0, ne, t, rng, extra=0.0):
    x, out = x0.copy(), [x0.copy()]
    weight = rank(x0) / len(x0)
    for _ in range(t - 1):
        g = rng.gamma(ne * x + 1e-6)
        x = g / g.sum()
        if extra:
            x = x * np.exp(rng.normal(0, extra * weight))
            x = x / x.sum()
        out.append(x)
    return np.array(out).T


def noise_calibrated(X, a, b, r, rng):
    x = np.clip(X, 1e-12, None)
    var = np.exp(a) * x**b / r
    y = rng.gamma(x**2 / var, var / x)
    return y / y.sum(axis=0, keepdims=True)


def noise_multinomial(X, depth, rng):
    return np.array([rng.multinomial(int(d), p / p.sum()) / d for p, d in zip(X.T, depth)]).T


def null_excess(X, x0, obs, noise, rng, n_cal, n_null):
    t = X.shape[1]
    target = score(X)[1]
    gaps = []
    for ne in NE_GRID:
        vals = [score(noise(drift(x0, ne, t, rng), rng))[1] for _ in range(n_cal)]
        gaps.append(abs(np.nanmean(vals) - target) if np.isfinite(vals).any() else np.inf)
    ne = NE_GRID[int(np.argmin(gaps))]
    sims = np.array([score(noise(drift(x0, ne, t, rng), rng))[0] for _ in range(n_null)])
    sims = sims[np.isfinite(sims)]
    return float(obs - sims.mean()), float(ne), int(len(sims))


def analyse(reps, depth, rng, n_cal=N_CAL, n_null=N_NULL):
    X = np.nanmean(reps, axis=2)
    obs = score(X)[0]
    if not np.isfinite(obs):
        return None
    try:
        a, b = fit_noise(reps)
    except ValueError:
        return None
    r = np.sum(np.isfinite(reps[0]), axis=1).clip(1)
    x0 = np.clip(X.mean(axis=1), 1e-9, None)
    x0 = x0 / x0.sum()
    cal = null_excess(X, x0, obs, lambda y, g: noise_calibrated(y, a, b, r, g), rng, n_cal, n_null)
    mul = null_excess(X, x0, obs, lambda y, g: noise_multinomial(y, depth, g), rng, n_cal, n_null)
    edge = {float(NE_GRID[0]), float(NE_GRID[-1])}
    present = np.isfinite(reps)
    return {"clones": int((X.mean(axis=1) >= MIN_SIZE).sum()), "samples": int(X.shape[1]), "rho": obs,
            "noise_a": a, "noise_b": b, "excess_calibrated": cal[0], "ne_calibrated": cal[1],
            "excess_multinomial": mul[0], "ne_multinomial": mul[1], "ne_at_grid_edge": cal[1] in edge or mul[1] in edge,
            "zero_share_replicate_cells": float((reps[present] == 0).mean())}


def signflip(x):
    x = np.asarray(x, float)
    signs = np.array(list(itertools.product((-1, 1), repeat=len(x))), float)
    return float(((signs * x).mean(axis=1) >= x.mean() - 1e-12).mean())


def synth(rng, extra, clones=40, samples=8):
    x0 = rng.dirichlet(np.full(clones, 0.5))
    x0 = np.clip(x0, 1e-6, None)
    x0 = x0 / x0.sum()
    X = drift(x0, 300, samples, rng, extra)
    a, b = -6.0, 1.5
    reps = np.stack([noise_calibrated(X, a, b, np.ones(samples), rng) for _ in range(3)], axis=2)
    return reps, np.full(samples, 150000)


def usable(rows):
    return [r for r in rows if r and "dropped" not in r and not r["ne_at_grid_edge"]]


def gate(rng):
    counts = {"null": {}, "link": {}}
    for label, extra in (("null", 0.0), ("link", 0.8)):
        for _ in range(GATE_COHORTS):
            rows = usable([analyse(*synth(rng, extra, b, t), rng, GATE_CAL, GATE_NULL) for b, t in SHAPES])
            name = "R3_inconclusive"
            if len(rows) >= MIN_FISH:
                name = verdict([r["excess_calibrated"] for r in rows], [r["excess_multinomial"] for r in rows])["label"]
            counts[label][name] = counts[label].get(name, 0) + 1
    rates = {"false_positive": counts["null"].get("R1_link_survives", 0) / GATE_COHORTS,
             "recovery": counts["link"].get("R1_link_survives", 0) / GATE_COHORTS,
             "r2_under_link": counts["link"].get("R2_noise_lead", 0) / GATE_COHORTS,
             "r2_under_null": counts["null"].get("R2_noise_lead", 0) / GATE_COHORTS}
    rates["passed"] = (rates["false_positive"] <= MAX_FALSE_POS and rates["recovery"] >= MIN_RECOVERY
                       and rates["r2_under_link"] <= MAX_R2_UNDER_LINK)
    return rates


def load_train():
    req = urllib.request.Request(URL, headers={"User-Agent": "cellware-seal/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise SystemExit("source table changed: sha256 mismatch")
    rows = csv.reader(io.StringIO(raw.decode("utf-8", "ignore")))
    head = next(rows)
    bc, rd = head.index("barcode"), head.index("reads")
    ids = []
    for row in rows:
        m = next((m for c in row if (m := ID.search(c))), None)
        if m:
            ids.append({"unit": str(int(m.group(1))), "sample": int(m.group(2)), "rep": int(m.group(3)), "bc": row[bc], "raw": row[rd]})
    df = train_only("germline_fish", pd.DataFrame(ids), "unit").copy()
    df["reads"] = pd.to_numeric(df.pop("raw"))
    return df.groupby(["unit", "sample", "rep", "bc"], as_index=False)["reads"].sum()


def fish_arrays(g):
    samples, bcs = sorted(g["sample"].unique()), sorted(g["bc"].unique())
    reps_ids = sorted(g["rep"].unique())
    arr = np.full((len(bcs), len(samples), len(reps_ids)), np.nan)
    si, bi, ri = {s: i for i, s in enumerate(samples)}, {b: i for i, b in enumerate(bcs)}, {r: i for i, r in enumerate(reps_ids)}
    depth = np.zeros(len(samples))
    for (s, r), h in g.groupby(["sample", "rep"]):
        total = h["reads"].sum()
        if total <= 0:
            continue
        arr[:, si[s], ri[r]] = 0
        arr[[bi[b] for b in h["bc"]], si[s], ri[r]] = h["reads"].to_numpy() / total
        depth[si[s]] += total
    keep = depth > 0
    return arr[:, keep, :], depth[keep]


def write(out):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print(json.dumps(out))


def verdict(cal, mul):
    pc, pm = signflip(cal), signflip(mul)
    mc, mm = float(np.mean(cal)), float(np.mean(mul))
    if pc <= ALPHA:
        label = "R1_link_survives"
    elif pm <= ALPHA and mm > 0 and mc <= mm / 2:
        label = "R2_noise_lead"
    else:
        label = "R3_inconclusive"
    sd = float(np.std(cal, ddof=1))
    return {"label": label, "p_calibrated": pc, "p_multinomial": pm, "mean_excess_calibrated": mc,
            "mean_excess_multinomial": mm, "effect_in_fish_sd": mc / sd if sd > 0 else None, "n_fish": len(cal)}


def main():
    rng = np.random.default_rng(SEED)
    checks = gate(rng)
    if not checks["passed"]:
        write({"status": "synthetic_gate_failed", "gate": checks})
        return 0
    df = load_train()
    fish = {}
    for unit, g in df.groupby("unit"):
        if g["sample"].nunique() < MIN_T:
            continue
        reps, depth = fish_arrays(g)
        res = analyse(reps, depth, rng)
        fish[str(unit)] = res if res else {"dropped": "too few clones"}
    used = usable(fish.values())
    out = {"status": "done", "gate": checks, "fish": fish}
    if len(used) >= MIN_FISH:
        out["verdict"] = verdict([r["excess_calibrated"] for r in used], [r["excess_multinomial"] for r in used])
    else:
        out["verdict"] = {"label": "R3_inconclusive", "n_fish": len(used)}
    write(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
