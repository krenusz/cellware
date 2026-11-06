"""G5: cell-based 1D gastruloid simulator from clone states plus local rules (chapter 3 model).

Rules (each from a chapter-3 result):
  - clone state: each cell carries an A-P state s ~ N(mu_clone, 1); bulk mu = 0 (#20, G1a)
  - memory: s is lost at passaging events, not per division: mu_clone(t) = mu_res + r^(t-1)
    (mu_clone(1) - mu_res) per passage (G3 r, G5 subclone test)
  - proportions: the gastruloid sizes domains: final anterior-propensity share
    F = (1 - g) f_A + g a* (#21, G1b)
  - positions: cells sort by rank of s + sorting noise (G1a)
  - elongation: descriptive rescue rule (G2 Mgen: clone competences + generic and
    complementary mixing bonuses; mechanism open, candidate chapter 4, #22/#24)

    uv run python g5_gastruloid.py calibrate     # fit mu, g, a*; write results_gastruloid/g5_model.json
    uv run python g5_gastruloid.py demo          # simulate the paper's main conditions
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

import g1_gastruloid as g1
import g2_elongation as g2
import g4_mechanism as g4

OUT = g1.OUT
N_CELLS = 300
N_BINS = 30
SORT_NOISE = 1.0
ANTERIOR = {3, 5, 7}


def profile(states, labels, label, rng):
    """Sort cells along the axis by rank of state + noise (high s = anterior, x = 0), return the
    labelled cells' density per bin, z-normalised as the authors do per gastruloid."""
    order = np.argsort(-(states + SORT_NOISE * rng.standard_normal(len(states))))
    pos = np.empty(len(states))
    pos[order] = np.arange(len(states)) / len(states)
    b = np.minimum((pos * N_BINS).astype(int), N_BINS - 1)
    dens = np.bincount(b[labels == label], minlength=N_BINS).astype(float)
    sd = dens.std()
    return (dens - dens.mean()) / sd if sd > 0 else dens * 0


def polarity_of(z):
    return z[: N_BINS // 2].mean() - z[N_BINS // 2:].mean()


def chimera(mu, frac, rng):
    """One clone at fraction frac with bulk; returns the clone's polarity P (G1a statistic)."""
    n = int(round(N_CELLS * frac))
    s = np.r_[mu + rng.standard_normal(n), rng.standard_normal(N_CELLS - n)]
    lab = np.r_[np.ones(n), np.zeros(N_CELLS - n)]
    return polarity_of(profile(s, lab, 1, rng))


def mean_polarity(mu, frac=0.5, reps=200, seed=0):
    rng = np.random.default_rng(seed)
    return float(np.mean([chimera(mu, frac, rng) for _ in range(reps)]))


def crossing_model(f_ant, g, a_star):
    return 100 * ((1 - g) * f_ant + g * a_star)


def calibrate():
    P = g1.polarity()
    target = P.groupby("clone")["P"].mean().to_dict()
    mu = {}
    for c, t in target.items():
        mu[c] = brentq(lambda m: mean_polarity(m) - t, -6, 6, xtol=1e-3)
    mix = json.load(open(os.path.join(OUT, "g1b_mixing.json")))["ratios"]
    f = np.array([r["f_anterior"] for r in mix])
    x = np.array([r["crossing"] for r in mix])
    A = np.c_[f, np.ones_like(f)]
    slope, icpt = np.linalg.lstsq(A, x / 100, rcond=None)[0]
    g = 1 - slope
    a_star = icpt / g
    g3 = json.load(open(os.path.join(OUT, "g3_fade.json")))
    mem = json.load(open(os.path.join(OUT, "g5_memory.json")))
    d, F, cls = g2.load()
    k, n = d["Elongated"].to_numpy(float), d["Total"].to_numpy(float)
    mg = g4.Mgen(F, cls)
    th = mg.fit(np.arange(len(d)), k, n, np.random.default_rng(0))
    model = {
        "mu_state": {c: round(v, 3) for c, v in mu.items()},
        "polarity_target": {c: round(v, 3) for c, v in target.items()},
        "polarity_model": {c: round(mean_polarity(mu[c], seed=1), 3) for c in mu},
        "proportion_feedback_g": float(g), "a_star": float(a_star),
        "memory_retention_per_passage": g3["retention_r"],
        "memory_rule": "per passage event (G5: p1 subclones keep 1.21 [1.01, 1.50] of parent; per-division <= 0.34 rejected)",
        "memory_by_clone_p2_over_p1": {"clone3": mem["ratios"]["head"]["p2_over_p1"],
                                       "clone1": mem["ratios"]["tail"]["p2_over_p1"]},
        "elongation_rule_Mgen": {"competence_logit": dict(zip(g2.COMP, np.round(th[:len(g2.COMP)], 3).tolist())),
                                 "bonus_comp": float(th[len(g2.COMP)]), "bonus_same": float(th[len(g2.COMP) + 1]),
                                 "bonus_unlabelled": float(th[len(g2.COMP) + 2])},
    }
    json.dump(model, open(os.path.join(OUT, "g5_model.json"), "w"), indent=1)
    print(json.dumps(model, indent=1))


def simulate_pair(model, a, b, ratio_a, passage, rng):
    """Two-clone gastruloid: returns (crossing x*, P(elongated)). a, b clone numbers."""
    r = model["memory_retention_per_passage"][0]
    mu = {c: model["mu_state"].get(f"Clone {c}", 0.0) * r ** (passage - 1) for c in (a, b)}  # clones 2, 4: no G1a state
    ant, post = (a, b) if mu[a] >= mu[b] else (b, a)
    f_ant = ratio_a if ant == a else 1 - ratio_a
    F = (1 - model["proportion_feedback_g"]) * f_ant + model["proportion_feedback_g"] * model["a_star"]
    na = int(round(N_CELLS * F))
    s = np.r_[mu[ant] + rng.standard_normal(na), mu[post] + rng.standard_normal(N_CELLS - na)]
    lab = np.r_[np.ones(na), np.zeros(N_CELLS - na)]
    za, zp = profile(s, lab, 1, rng), profile(s, lab, 0, rng)
    dd = za - zp
    xs = np.linspace(0, 100, N_BINS)
    cross = next((xs[i] + (xs[i + 1] - xs[i]) * dd[i] / (dd[i] - dd[i + 1])
                  for i in range(N_BINS - 1) if dd[i] * dd[i + 1] < 0), np.nan)
    e = model["elongation_rule_Mgen"]
    comp = e["competence_logit"]
    eta = ratio_a * comp[f"c{a}"] + (1 - ratio_a) * comp[f"c{b}"]
    known = {1, 3, 5, 6, 7}
    if a in known and b in known:
        eta += e["bonus_comp"] if ((a in ANTERIOR) != (b in ANTERIOR)) else e["bonus_same"]
    else:
        eta += e["bonus_unlabelled"]
    return float(cross), float(expit(eta))


def demo(reps=200, seed=68):
    model = json.load(open(os.path.join(OUT, "g5_model.json")))
    rng = np.random.default_rng(seed)
    out = {"crossing_vs_ratio_clone1_clone3_passage1": {}, "passage_fade_clone1_clone3_1to1": {},
           "elongation": {}}
    for r13 in (0.8, 2 / 3, 0.5, 1 / 3, 0.2):  # clone 1 share
        xs = [simulate_pair(model, 1, 3, r13, 1, rng)[0] for _ in range(reps)]
        out["crossing_vs_ratio_clone1_clone3_passage1"][f"clone1 {r13:.2f}"] = round(float(np.nanmean(xs)), 1)
    for p in (1, 2, 3, 4):
        P = []
        for _ in range(reps):
            mu1 = model["mu_state"]["Clone 1"] * model["memory_retention_per_passage"][0] ** (p - 1)
            P.append(chimera(mu1, 0.5, rng))
        out["passage_fade_clone1_clone3_1to1"][f"passage {p}: clone 1 polarity in bulk"] = round(float(np.mean(P)), 3)
    comp = model["elongation_rule_Mgen"]["competence_logit"]
    for c in range(1, 8):
        out["elongation"][f"pure clone {c}"] = round(float(expit(comp[f"c{c}"])), 3)
    for a, b in ((1, 3), (1, 6), (3, 7), (4, 6), (5, 6)):
        out["elongation"][f"{a}+{b}"] = round(simulate_pair(model, a, b, 0.5, 1, rng)[1], 3)
    json.dump(out, open(os.path.join(OUT, "g5_demo.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calibrate", "demo"])
    {"calibrate": calibrate, "demo": demo}[ap.parse_args().cmd]()
