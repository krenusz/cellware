"""C3-S1: the gastruloid simulator as a configuration of the core model. See EXPERIMENT_LOG C3-S1.

    uv run --with openpyxl python c3s1_core_gastruloid.py

Core steps used (core_model.py): 1 state + per-condition copy (fate memory; here per gastruloid
rho_g and per passage rho_P), 4 logit map beta * zb (beta = 1) as each cell's anterior score.
Local rules (positions, domain sizes) replace split/growth: rank sorting with cell noise (G5) and
proportion feedback g, a* (G1b). Growth/selection gamma = 0 (not identified here; disclosed).
Data: G1a fit shard (open), G1a sealed shard per-replicate means (spent, ledger), G3 open pairs,
G7a spent summaries. No sealed file is read.
"""

from __future__ import annotations

import csv
import json
import os

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import beta as beta_dist

import core_model as cm
import g1_gastruloid as g1
import g5_gastruloid as g5
import g7_model

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_gastruloid")
N_CELLS, BETA = g5.N_CELLS, 1.0
REPS = 300


def state_copy(z, rho, rng):
    """Core step 1: a condition's copy of the clone state (fate memory rho)."""
    return cm.state_copy(z, rho, rng)


def gastruloid_P(z, frac, rho_g, rng):
    """One clone-in-bulk gastruloid: the clone's cells score beta * zb + cell noise; bulk scores
    cell noise around 0. Returns the clone's polarity (G1a statistic)."""
    zb = state_copy(np.array([z]), rho_g, rng)[0]
    n = int(round(N_CELLS * frac))
    s = np.r_[BETA * zb + rng.standard_normal(n), rng.standard_normal(N_CELLS - n)]
    lab = np.r_[np.ones(n), np.zeros(N_CELLS - n)]
    return g5.polarity_of(g5.profile(s, lab, 1, rng))


def sim_clone(z, rho_g, reps=REPS, seed=0, frac=0.5):
    rng = np.random.default_rng(seed)
    P = np.array([gastruloid_P(z, frac, rho_g, rng) for _ in range(reps)])
    return float(P.mean()), float(P.std(ddof=1))


def pair_D(za, zb_, rho_g, model, rng, ratio_a=0.5):
    """Two-clone gastruloid, D = P_a - P_b; domain sizes by proportion feedback (G5 rule)."""
    a, b = state_copy(np.array([za, zb_]), rho_g, rng)
    ant_is_a = a >= b
    f_ant = ratio_a if ant_is_a else 1 - ratio_a
    F = (1 - model["proportion_feedback_g"]) * f_ant + model["proportion_feedback_g"] * model["a_star"]
    na = int(round(N_CELLS * F))
    hi, lo = (a, b) if ant_is_a else (b, a)
    s = np.r_[BETA * hi + rng.standard_normal(na), BETA * lo + rng.standard_normal(N_CELLS - na)]
    lab = np.r_[np.ones(na), np.zeros(N_CELLS - na)]
    p_hi = g5.polarity_of(g5.profile(s, lab, 1, rng))
    p_lo = g5.polarity_of(g5.profile(s, lab, 0, rng))
    return (p_hi - p_lo) if ant_is_a else (p_lo - p_hi)


def ledger(test):
    with open(os.path.join(HERE, "test_ledger.tsv"), encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["test"] == test:
                return json.loads(r["result"])
    raise KeyError(test)


def fit_states(target, rho_g):
    return {c: brentq(lambda z: sim_clone(z, rho_g, reps=200, seed=1)[0] - t, -8, 8, xtol=1e-3)
            for c, t in target.items()}


def main():
    model = json.load(open(os.path.join(OUT, "g5_model.json")))
    P = g1.polarity()  # fit shard only
    P["c"] = P["clone"].str.split().str[-1].astype(int)
    grp = P.groupby("c")["P"]
    target, sd_obs, n_obs = grp.mean().to_dict(), grp.std().to_dict(), grp.size().to_dict()
    se_obs = {c: sd_obs[c] / np.sqrt(n_obs[c]) for c in target}
    pooled_sd = float(np.sqrt(np.average([sd_obs[c] ** 2 for c in target], weights=[n_obs[c] - 1 for c in target])))
    res = {"fit_shard": {str(c): {"n": int(n_obs[c]), "mean": round(target[c], 3), "sd": round(sd_obs[c], 3)}
                         for c in target}, "pooled_within_clone_sd": round(pooled_sd, 3)}

    # (a) equivalence with G5 at rho_g = 1
    mu5 = {int(k.split()[-1]): v for k, v in model["mu_state"].items()}
    eq = {}
    for c, mu in mu5.items():
        core = sim_clone(mu, 1.0, reps=400, seed=7)[0]
        old = g5.mean_polarity(mu, reps=400, seed=7)
        eq[str(c)] = [round(core, 3), round(old, 3)]
    res["a_equivalence_rho1_core_vs_g5"] = eq
    res["a_pass"] = bool(all(abs(u - v) < 0.05 for u, v in eq.values()))

    # rho_g from pooled within-clone SD (states refit for each rho_g)
    def sd_gap(rho):
        z = fit_states(target, rho)
        sims = [sim_clone(z[c], rho, reps=200, seed=3)[1] for c in target]
        return (float(np.average(sims, weights=[n_obs[c] - 1 for c in target])) - pooled_sd) ** 2
    z1 = fit_states(target, 1.0)
    sim_sd_rho1 = float(np.mean([sim_clone(z1[c], 1.0, reps=200, seed=3)[1] for c in target]))
    if sim_sd_rho1 >= pooled_sd:
        rho_g = 1.0  # cell noise alone already gives at least the observed spread
    else:
        rho_g = float(minimize_scalar(sd_gap, bounds=(0.05, 1.0), method="bounded", options={"xatol": 0.01}).x)
    z = fit_states(target, rho_g)
    res["rho_g"] = round(rho_g, 3)
    res["sim_sd_at_rho1"] = round(sim_sd_rho1, 3)
    res["states_z"] = {str(c): round(v, 3) for c, v in z.items()}

    # (b) #20
    sealed = ledger("gastruloid_polarity_heldout")["cells"]
    sm = {}
    for r in sealed:
        c = int(r["clone"].split()[-1])
        sm.setdefault(c, []).append((r["mean"], r["n"]))
    b = {}
    ok = True
    for c in target:
        m, s = sim_clone(z[c], rho_g, reps=REPS, seed=11)
        fit_ok = abs(m - target[c]) <= 2 * se_obs[c]
        sd_ok = abs(s - sd_obs[c]) <= 0.25 * sd_obs[c]
        vals = sm.get(c, [])
        ns = sum(n for _, n in vals)
        smean = sum(v * n for v, n in vals) / ns if ns else np.nan
        sse = sd_obs[c] / np.sqrt(ns) if ns else np.nan
        sealed_ok = bool(ns and abs(m - smean) <= 2 * sse)
        ok &= fit_ok and sd_ok and sealed_ok
        b[str(c)] = {"sim_mean": round(m, 3), "fit_mean": round(target[c], 3), "fit_2se": round(2 * se_obs[c], 3),
                     "spent_sealed_mean": round(float(smean), 3), "spent_n": ns, "sealed_2se": round(float(2 * sse), 3),
                     "sim_sd": round(s, 3), "obs_sd": round(sd_obs[c], 3),
                     "mean_ok": bool(fit_ok), "sealed_ok": sealed_ok, "sd_ok": bool(sd_ok)}
    res["b_clone_propensity"], res["b_pass"] = b, bool(ok)

    # (c) #26 unseen pairs, states from G1a only
    g7 = ledger("gastruloid_g7_pairs")
    rng = np.random.default_rng(26)
    c_res = {}
    for key, (a_, b_) in (("3+7", (3, 7)), ("6+7", (6, 7))):
        D = np.array([pair_D(z[a_], z[b_], rho_g, model, rng) for _ in range(REPS)])
        share = float((D > 0).mean())
        obs = g7[key]
        k_obs = int(round(obs["share_red_anterior"] * obs["n"]))
        lo, hi = beta_dist.ppf([0.025, 0.975], [k_obs + 0.5] * 2, [obs["n"] - k_obs + 0.5] * 2)
        c_res[key] = {"pred_share_red_anterior": round(share, 3), "obs_share": obs["share_red_anterior"],
                      "obs_ci95": [round(float(lo), 3), round(float(hi), 3)], "pred_D": round(float(D.mean()), 3),
                      "obs_D": [round(v, 3) for v in obs["D_mean"]]}
    c67 = 1 - c_res["6+7"]["pred_share_red_anterior"] >= 0.75
    c37 = c_res["3+7"]["obs_ci95"][0] <= c_res["3+7"]["pred_share_red_anterior"] <= c_res["3+7"]["obs_ci95"][1]
    res["c_pairs"], res["c_pass"] = c_res, bool(c67 and c37)

    # (d) #23 fade: G3 open 1+3 pairs, D by passage; one rho_P
    op = g7_model.open_pairs()
    obsD = {p: (float(g["D"].mean()), float(g["D"].std() / np.sqrt(len(g))), len(g)) for p, g in op.groupby("passage")}

    def predD(rho_p, p, reps=150, seed=5):
        r = np.random.default_rng(seed + p)
        out = []
        for _ in range(reps):
            z3, z1_ = z[3], z[1]
            for _k in range(p - 1):  # one copy per passage event (core memory step)
                z3, z1_ = state_copy(np.array([z3, z1_]), rho_p, r)
            out.append(pair_D(z3, z1_, rho_g, model, r))
        return float(np.mean(out))

    def loss(rho_p):
        return sum(((obsD[p][0] - predD(rho_p, p)) / obsD[p][1]) ** 2 for p in obsD)
    rho_P = float(minimize_scalar(loss, bounds=(0.01, 1.0), method="bounded", options={"xatol": 0.01}).x)
    d_res, ok = {}, True
    for p in sorted(obsD):
        pr = predD(rho_P, p)
        good = abs(pr - obsD[p][0]) <= 2 * obsD[p][1]
        ok &= good
        d_res[f"passage {p}"] = {"obs_D": round(obsD[p][0], 3), "obs_2se": round(2 * obsD[p][1], 3), "n": obsD[p][2],
                                 "pred_D": round(pr, 3), "ok": bool(good)}
    res["d_fade"], res["rho_P"], res["d_pass"] = d_res, round(rho_P, 3), bool(ok)
    res["retention_per_passage_equiv"] = round(float(np.sqrt(rho_P)), 3)
    res["all_pass"] = bool(res["a_pass"] and res["b_pass"] and res["c_pass"] and res["d_pass"])
    json.dump(res, open(os.path.join(OUT, "c3s1_core_gastruloid.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
