"""G7 (open data only): simulator v2 and its pre-registered predictions. See EXPERIMENT_LOG G7.

v2 = G5 simulator with (1) clone states refitted on the G1a fit shard only (G5 had read sealed
rows, audit), (2) clone-specific memory retention per passage from open data (G5 subclones:
clone 1, clone 3; other clones: G3 pooled rate), (3) a gastruloid-level state noise tau,
calibrated on open clone 1 + 3 pairs (G3 passage-1 files) only if v1 misses their spread.

    uv run --with openpyxl python g7_model.py check     # v1 vs open pair data
    uv run --with openpyxl python g7_model.py predict   # write results_gastruloid/g7_predictions.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.signal import savgol_filter

import g1_gastruloid as g1
import g5_gastruloid as g5

OUT = g1.OUT
RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "gastruloid_raj", "raw_ls2")
OPEN_PAIRS = {1: "clone1+3_pass1_04222024.xlsx", 2: "clone1+3_pass2_04242024.xlsx",
              3: "clone1+3_pass3_4272024.xlsx", 4: "clone1+3_pass4_04292024.xlsx"}  # G3, open
N = 30


def resample(s):
    v = np.array([float(x) for x in str(s).split(",")])
    return v[np.round(np.linspace(0, len(v) - 1, N)).astype(int)]


def pol(v):
    """G1a statistic on one raw line: z-normalise, Savitzky-Golay 5/2, first half minus second."""
    sd = v.std()
    if sd == 0:
        return np.nan
    z = savgol_filter((v - v.mean()) / sd, 5, 2)
    return z[: N // 2].mean() - z[N // 2:].mean()


def open_pairs():
    rows = []
    for p, f in OPEN_PAIRS.items():
        d = pd.read_excel(os.path.join(RAW, f))
        for aid, g in d[d["Condition"] == "EX"].groupby("Annotation ID"):
            r, gr = g[g["Header"] == "Channel 2"]["Values"], g[g["Header"] == "Channel 0"]["Values"]
            if len(r) == 1 and len(gr) == 1:
                rows.append({"passage": p, "P3": pol(resample(r.iloc[0])), "P1": pol(resample(gr.iloc[0]))})
    df = pd.DataFrame(rows).dropna()
    df["D"] = df["P3"] - df["P1"]
    return df


# ---------- simulator v2 ----------

def pair_P(mu_a, mu_b, f_a, tau, rng):
    """50:50-style pair without bulk: per-clone polarity (z-normalised density, G1a statistic).
    Proportion feedback as G5: the higher-state clone gets share F of cells. tau: gastruloid-level
    noise added to each clone's mean state in this gastruloid."""
    m = g5_model()
    ma, mb = mu_a + tau * rng.standard_normal(), mu_b + tau * rng.standard_normal()
    f_ant = f_a if mu_a >= mu_b else 1 - f_a
    F = (1 - m["g"]) * f_ant + m["g"] * m["a_star"]
    na = int(round(g5.N_CELLS * (F if mu_a >= mu_b else 1 - F)))
    s = np.r_[ma + rng.standard_normal(na), mb + rng.standard_normal(g5.N_CELLS - na)]
    lab = np.r_[np.ones(na), np.zeros(g5.N_CELLS - na)]
    return g5.polarity_of(g5.profile(s, lab, 1, rng)), g5.polarity_of(g5.profile(s, lab, 0, rng))


def bulk_P(mu, frac, tau, rng):
    n = int(round(g5.N_CELLS * frac))
    m = mu + tau * rng.standard_normal()
    s = np.r_[m + rng.standard_normal(n), rng.standard_normal(g5.N_CELLS - n)]
    lab = np.r_[np.ones(n), np.zeros(g5.N_CELLS - n)]
    return g5.polarity_of(g5.profile(s, lab, 1, rng))


_M = {}


def g5_model():
    if not _M:
        m = json.load(open(os.path.join(OUT, "g5_model.json")))
        _M.update(g=m["proportion_feedback_g"], a_star=m["a_star"])
    return _M


def fit_mu():
    """Clone states on the G1a fit shard only (polarity() excludes sealed gastruloids)."""
    P = g1.polarity()
    target = P.groupby("clone")["P"].mean().to_dict()
    mu = {int(c.split()[-1]): brentq(lambda m: g5.mean_polarity(m) - t, -6, 6, xtol=1e-3) for c, t in target.items()}
    return mu, {int(c.split()[-1]): float(t) for c, t in target.items()}


def retention():
    """Per-passage retention: G5 subclone p2/p1 for clones 1 and 3 (open), else G3 pooled."""
    mem = json.load(open(os.path.join(OUT, "g5_memory.json")))["ratios"]
    g3 = json.load(open(os.path.join(OUT, "g3_fade.json")))["retention_r"]
    r = {c: g3 for c in (1, 3, 5, 6, 7)}
    r[3] = mem["head"]["p2_over_p1"]
    r[1] = mem["tail"]["p2_over_p1"]
    return r, g3


def sim_pairs(mu_a, mu_b, n, tau, rng, f_a=0.5):
    return np.array([pair_P(mu_a, mu_b, f_a, tau, rng) for _ in range(n)])


def check(seed=71):
    """v1 (tau = 0) vs open 1+3 pairs at passage 1; calibrate tau on SD of D = P3 - P1."""
    df = open_pairs()
    p1 = df[df["passage"] == 1]
    mu, target = fit_mu()
    rng = np.random.default_rng(seed)
    sim0 = sim_pairs(mu[3], mu[1], 400, 0.0, rng)
    obs = {"n": int(len(p1)), "P3_mean": float(p1["P3"].mean()), "P1_mean": float(p1["P1"].mean()),
           "D_mean": float(p1["D"].mean()), "D_sd": float(p1["D"].std()), "share_D_pos": float((p1["D"] > 0).mean())}
    v1 = {"P3_mean": float(sim0[:, 0].mean()), "P1_mean": float(sim0[:, 1].mean()),
          "D_mean": float((sim0[:, 0] - sim0[:, 1]).mean()), "D_sd": float((sim0[:, 0] - sim0[:, 1]).std()),
          "share_D_pos": float(((sim0[:, 0] - sim0[:, 1]) > 0).mean())}

    def sd_gap(tau):
        s = sim_pairs(mu[3], mu[1], 1500, tau, np.random.default_rng(seed + 1))
        return float((s[:, 0] - s[:, 1]).std()) - obs["D_sd"]
    tau = 0.0
    if v1["D_sd"] < obs["D_sd"]:
        tau = brentq(sd_gap, 0.0, 5.0, xtol=1e-2) if sd_gap(5.0) > 0 else 5.0
    sim = sim_pairs(mu[3], mu[1], 1500, tau, np.random.default_rng(seed + 1))
    v2 = {"tau": float(tau), "P3_mean": float(sim[:, 0].mean()), "P1_mean": float(sim[:, 1].mean()),
          "D_mean": float((sim[:, 0] - sim[:, 1]).mean()), "D_sd": float((sim[:, 0] - sim[:, 1]).std()),
          "share_D_pos": float(((sim[:, 0] - sim[:, 1]) > 0).mean())}
    # open memory check: per-clone P over passages in the G3 pair series
    fade = df.groupby("passage")[["P3", "P1"]].mean().round(3).to_dict()
    res = {"mu_fit_shard": {f"clone{c}": round(v, 3) for c, v in mu.items()},
           "polarity_target_fit_shard": {f"clone{c}": round(v, 3) for c, v in target.items()},
           "open_pairs_passage1": obs, "v1_tau0": v1, "v2_tau": v2, "open_pair_series_mean_P": fade}
    json.dump(res, open(os.path.join(OUT, "g7_check.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
    return res


def interval(draw, n, sims=2000, seed=0):
    """95% interval of the mean of n gastruloids under the model (draw(rng) -> one value)."""
    rng = np.random.default_rng(seed)
    pool = np.array([draw(rng) for _ in range(4000)])
    means = [rng.choice(pool, n).mean() for _ in range(sims)]
    return [float(pool.mean()), *np.percentile(means, [2.5, 97.5]).tolist()]


def predict():
    chk = json.load(open(os.path.join(OUT, "g7_check.json")))
    tau = chk["v2_tau"]["tau"]
    mu, _ = fit_mu()
    r, r_pool = retention()
    out = {"tau": tau, "mu": {f"clone{c}": round(v, 3) for c, v in mu.items()},
           "retention_v2": {f"clone{c}": v for c, v in r.items()}, "retention_v1": r_pool}
    # G7a: new pairs at passage 1 (3+7: n 14, red = clone 3; 6+7: n 12, red = clone 6)
    g7a = {}
    for (a, b, n) in ((3, 7, 14), (6, 7, 12)):
        rng = np.random.default_rng(100 + a)
        sims = sim_pairs(mu[a], mu[b], 4000, tau, rng)
        D = sims[:, 0] - sims[:, 1]
        share = []
        for _ in range(2000):
            idx = rng.integers(0, len(D), n)
            share.append(float((D[idx] > 0).mean()))
        dm = [float(rng.choice(D, n).mean()) for _ in range(2000)]
        g7a[f"{a}+{b}"] = {"n": n, "D_red_minus_green_mean": [float(D.mean()), *np.percentile(dm, [2.5, 97.5]).tolist()],
                           "share_red_anterior": [float((D > 0).mean()), *np.percentile(share, [2.5, 97.5]).tolist()],
                           "P_red_mean": float(sims[:, 0].mean()), "P_green_mean": float(sims[:, 1].mean())}
    out["G7a"] = g7a
    # G7b: clone in bulk over passages, sign-adjusted P (toward parent), per fraction mix
    plan = {1: {2: {0.25: 4, 0.5: 12}}, 3: {2: {0.25: 7, 0.5: 14}, 3: {0.25: 25, 0.5: 23}, 4: {0.25: 16, 0.5: 19}}}
    g7b = {}
    for c, passages in plan.items():
        sgn = np.sign(mu[c])
        for p, fr in passages.items():
            n = sum(fr.values())
            res = {}
            for name, rr in (("v2", r[c]), ("v1", r_pool)):
                rng = np.random.default_rng(1000 * c + 10 * p + (name == "v1"))
                vals = []
                for _ in range(3000):  # parameter uncertainty: r drawn from its interval (normal approx)
                    sd = (rr[2] - rr[1]) / 3.92
                    rd = max(0.0, rng.normal(rr[0], sd))
                    f = rng.choice(list(fr.keys()), p=np.array(list(fr.values())) / n)
                    vals.append(sgn * bulk_P(mu[c] * rd ** (p - 1), f, tau, rng))
                vals = np.array(vals)
                ms = [rng.choice(vals, n).mean() for _ in range(2000)]
                res[name] = [float(vals.mean()), *np.percentile(ms, [2.5, 97.5]).tolist()]
            g7b[f"clone{c}_p{p}"] = {"n": n, **res}
    out["G7b"] = g7b
    json.dump(out, open(os.path.join(OUT, "g7_predictions.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "predict"])
    {"check": check, "predict": predict}[ap.parse_args().cmd]()
