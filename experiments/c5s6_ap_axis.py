"""C5-S6: does a clone's position on the passage-1 A-P chromatin axis fade over passages?
(EXPERIMENT_LOG, pre-registered 2026-11-08)

    uv run python c5s6_ap_axis.py          # train (passages 0-1) + power gate

Reuses the C5-S4 summit-bin readout. Sealed passages are read only in test(part), called by test_oracle.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

import c5s4_atac_memory as m4
import partitions

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_gastruloid", "c5s6_train.json")
ANT, POS = ["clone 3", "clone 5", "clone 7"], ["clone 1", "clone 6"]
AP = ANT + POS
G3_R = 0.60
SEED = 57


def samples(passages, allow_sealed=False):
    """[(clone, rep, passage, bins)] for A-P clones, QC-passing only."""
    out = []
    for r in m4.sheet():
        if r["passage"] not in passages or r["clone"] not in AP:
            continue
        if partitions.partition_of(m4.DS, r["passage"]) != "train" and not allow_sealed:
            raise RuntimeError(f"passage {r['passage']} is sealed; read only via test_oracle")
        b = m4._bins(r["v"])
        if b is not None:
            out.append((r["clone"], r["rep"], r["passage"], b))
    return out


def axes(p1):
    """Fold k axis from passage-1 samples with rep != k: (A-bins, P-bins)."""
    res = {}
    for k in sorted({s[1] for s in p1}):
        tr = {s[0]: s[3] for s in p1 if s[1] != k}
        if not all(c in tr for c in AP):
            raise RuntimeError(f"fold {k}: missing clones")
        a = set.intersection(*[tr[c] for c in ANT]) - set.union(*[tr[c] for c in POS])
        p = set.intersection(*[tr[c] for c in POS]) - set.union(*[tr[c] for c in ANT])
        res[k] = (a, p)
    return res


def score(b, ax):
    a, p = ax
    return len(b & a) / len(a) - len(b & p) / len(p)


def scored(smp, ax, cross_fit):
    out = []
    for c, rep, pas, b in smp:
        if cross_fit:
            s = score(b, ax[rep])
        else:
            s = float(np.mean([score(b, x) for x in ax.values()]))
        out.append((c, rep, pas, s))
    return out


def gap(sc, clones=None):
    """Clone means, D, SE(D) with pooled within-clone replicate variance."""
    by = {}
    for c, _, _, s in sc:
        by.setdefault(c, []).append(s)
    cl = [c for c in AP if c in by and (clones is None or c in clones)]
    mean = {c: float(np.mean(by[c])) for c in cl}
    dev = [(v - np.mean(v)) for c in cl for v in [np.array(by[c])] if len(v) >= 2]
    df = sum(len(d) - 1 for d in dev)
    s2 = float(sum((d ** 2).sum() for d in dev) / df) if df > 0 else float("nan")
    a, p = [c for c in ANT if c in cl], [c for c in POS if c in cl]
    D = np.mean([mean[c] for c in a]) - np.mean([mean[c] for c in p])
    var = sum(s2 / len(by[c]) for c in a) / len(a) ** 2 + sum(s2 / len(by[c]) for c in p) / len(p) ** 2
    return {"clone_mean": mean, "n": {c: len(by[c]) for c in cl}, "D": float(D), "se": float(np.sqrt(var)),
            "s_rep": float(np.sqrt(s2)), "df": df}


def ratio_ub(g1, g2, q=1.645):
    if g1["D"] <= 0 or g2["D"] <= 0:
        return None, None, None
    lr = np.log(g2["D"] / g1["D"])
    se = np.sqrt((g1["se"] / g1["D"]) ** 2 + (g2["se"] / g2["D"]) ** 2)
    return float(np.exp(lr)), float(np.exp(lr + q * se)), float(np.exp(lr - q * se))


def power(g1, nrep2, nsim=4000):
    rng = np.random.default_rng(SEED)
    mu = g1["clone_mean"]
    mid = (np.mean([mu[c] for c in ANT]) + np.mean([mu[c] for c in POS])) / 2
    out = {}
    for R in (0.6, 0.83):
        for k in (1.0, 1.5):
            s = g1["s_rep"] * k
            hit = 0
            for _ in range(nsim):
                sc1 = [(c, r, 1, mu[c] + s * rng.standard_normal()) for c in AP for r in (1, 2)]
                sc2 = [(c, r, 2, mid + R * (mu[c] - mid) + s * rng.standard_normal())
                       for c in AP for r in range(nrep2[c])]
                _, ub, _ = ratio_ub(gap(sc1), gap(sc2))
                hit += ub is not None and ub < 1
            out[f"R={R}|s_x{k}"] = round(hit / nsim, 3)
    return out


def train():
    p1 = [s for s in samples({1})]
    p0 = samples({0})
    ax = axes(p1)
    sc1 = scored(p1, ax, cross_fit=True)
    sc0 = scored(p0, ax, cross_fit=False)
    g1 = gap(sc1)
    p0_cl = sorted({s[0] for s in sc0})
    g0m, g1m = gap(sc0, p0_cl), gap(sc1, p0_cl)
    nrep2 = {c: sum(1 for r in m4.sheet() if r["passage"] == 2 and r["clone"] == c) for c in AP}
    pw = power(g1, nrep2)
    res = {"axis_bins": {f"fold{k}": {"A": len(a), "P": len(p)} for k, (a, p) in ax.items()},
           "p1_scores": [(c, r, round(s, 4)) for c, r, _, s in sc1], "p1": g1,
           "p0_described": {"clones": p0_cl, "scores": [(c, round(s, 4)) for c, _, _, s in sc0],
                            "D0": g0m["D"], "D1_same_set": g1m["D"],
                            "ratio_D1_over_D0": g1m["D"] / g0m["D"] if g0m["D"] else None,
                            "note": "single replicates at p0; described only"},
           "nrep_p2_sheet": nrep2, "power": pw,
           "gate": {"rule": "power >= 0.8 at R=0.6 with 1.5 s", "power": pw["R=0.6|s_x1.5"],
                    "pass": bool(pw["R=0.6|s_x1.5"] >= 0.8)}}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "p1_scores"}, indent=1))


def test(part):
    """One look (via test_oracle): R = D(p)/D(1), one-sided 95% upper bound < 1."""
    p = int({v: k for k, v in partitions.PARTITIONED[m4.DS]["fixed"].items()}[part])
    p1 = samples({1})
    ax = axes(p1)
    g1 = gap(scored(p1, ax, cross_fit=True))
    gp = gap(scored(samples({p}, allow_sealed=True), ax, cross_fit=False))
    R, ub, lb = ratio_ub(g1, gp)
    return {"partition": part, "passage": p, "D1": g1, "Dp": gp, "R": R, "R_upper95": ub, "R_lower95": lb,
            "fading": bool(ub is not None and ub < 1), "G3_forecast": G3_R ** (p - 1),
            "G3_inside_90ci": bool(lb is not None and lb <= G3_R ** (p - 1) <= ub)}


W2, W3 = 0.385, 0.615


def contrast(g1, g2, g3):
    L = g1["D"] - (W2 * g2["D"] + W3 * g3["D"])
    se = np.sqrt(g1["se"] ** 2 + W2 ** 2 * g2["se"] ** 2 + W3 ** 2 * g3["se"] ** 2)
    return float(L), float(se), float(L / se)


def _vals(v):
    """C5-S6b R2/R3: {bin: within-sample percentile of signalValue}, all peaks, max per bin."""
    best = {}
    with open(m4._path(v), encoding="utf-8") as f:
        for line in f:
            p = line.split("\t")
            k = (p[0], (int(p[1]) + int(p[9])) // m4.BIN)
            s = float(p[6])
            if s > best.get(k, -1.0):
                best[k] = s
    keys = list(best)
    sig = np.array([best[k] for k in keys])
    pct = (np.argsort(np.argsort(sig, kind="stable"), kind="stable") + 1) / len(sig)
    return dict(zip(keys, pct.tolist()))


def scorer(kind, fold):
    """fold: {clone: sample repr}. Returns sample repr -> score."""
    if kind == "R0":
        a = set.intersection(*[fold[c] for c in ANT]) - set.union(*[fold[c] for c in POS])
        p = set.intersection(*[fold[c] for c in POS]) - set.union(*[fold[c] for c in ANT])
        return lambda b: len(b & a) / len(a) - len(b & p) / len(p)
    if kind == "R1":
        allb = set.union(*fold.values())
        w = {}
        for k in allb:
            d = np.mean([k in fold[c] for c in ANT]) - np.mean([k in fold[c] for c in POS])
            if abs(d) >= 0.5:
                w[k] = d
        tot = sum(abs(x) for x in w.values())
        return lambda b: sum(x for k, x in w.items() if k in b) / tot
    n = {"R2": 2000, "R3": 5000}[kind]
    allb = set().union(*[set(fold[c]) for c in AP])
    w = {k: np.mean([fold[c].get(k, 0.0) for c in ANT]) - np.mean([fold[c].get(k, 0.0) for c in POS]) for k in allb}
    top = sorted(w, key=lambda k: -abs(w[k]))[:n]
    tot = sum(abs(w[k]) for k in top)
    return lambda x: sum(w[k] * x.get(k, 0.0) for k in top) / tot


def readout_search():
    """C5-S6b (train only): cross-fitted p1 D/SE per candidate readout; pooled power for the winner."""
    rows = [r for r in m4.sheet() if r["passage"] == 1 and r["clone"] in AP]
    rows = [r for r in rows if m4._n_peaks(r["v"]) >= m4.TOP]
    reps = {"bins": {r["v"]: m4._bins(r["v"]) for r in rows}, "vals": {r["v"]: _vals(r["v"]) for r in rows}}
    res = {}
    for kind in ("R0", "R1", "R2", "R3"):
        rep = reps["bins"] if kind in ("R0", "R1") else reps["vals"]
        sc = []
        for k in sorted({r["rep"] for r in rows}):
            fold = {r["clone"]: rep[r["v"]] for r in rows if r["rep"] != k}
            f = scorer(kind, fold)
            sc += [(r["clone"], r["rep"], 1, f(rep[r["v"]])) for r in rows if r["rep"] == k]
        g = gap(sc)
        res[kind] = {"D": round(g["D"], 4), "se": round(g["se"], 4), "s_rep": round(g["s_rep"], 4),
                     "D_over_se": round(g["D"] / g["se"], 2), "g": g}
    win = max(res, key=lambda k: res[k]["D_over_se"])
    pw = pooled_power(g1=res[win]["g"], save=False)
    out = {"candidates": {k: {x: y for x, y in v.items() if x != "g"} for k, v in res.items()},
           "clone_means": {k: {c: round(m, 4) for c, m in v["g"]["clone_mean"].items()} for k, v in res.items()},
           "winner": win, "margin_over_R0": round(res[win]["D_over_se"] / res["R0"]["D_over_se"], 2),
           "pooled_power": pw["power"], "gate": pw["gate"],
           "marginal": bool(pw["gate"]["pass"] and pw["gate"]["power"] < 0.85)}
    json.dump(out, open(OUT.replace("_train", "_readout"), "w"), indent=1)
    print(json.dumps(out, indent=1))


def pooled_power(nsim=4000, g1=None, save=True):
    """Amendment (train only): power of L/SE(L) > 1.645 over passages 1-3."""
    rng = np.random.default_rng(SEED + 1)
    if g1 is None:
        g1 = gap(scored(samples({1}), axes(samples({1})), cross_fit=True))
    mu = g1["clone_mean"]
    mid = (np.mean([mu[c] for c in ANT]) + np.mean([mu[c] for c in POS])) / 2
    nrep = {p: {c: sum(1 for r in m4.sheet() if r["passage"] == p and r["clone"] == c) for c in AP} for p in (2, 3)}
    out = {}
    for R in (0.6, 0.83):
        for k in (1.0, 1.5):
            s, hit = g1["s_rep"] * k, 0
            for _ in range(nsim):
                gs = [gap([(c, r, 1, mu[c] + s * rng.standard_normal()) for c in AP for r in (1, 2)])]
                for p in (2, 3):
                    gs.append(gap([(c, r, p, mid + R ** (p - 1) * (mu[c] - mid) + s * rng.standard_normal())
                                   for c in AP for r in range(nrep[p][c])]))
                hit += contrast(*gs)[2] > 1.645
            out[f"R={R}|s_x{k}"] = round(hit / nsim, 3)
    res = {"nrep_sheet": nrep, "power": out,
           "gate": {"rule": "power >= 0.8 at R=0.6 with 1.5 s", "power": out["R=0.6|s_x1.5"],
                    "pass": bool(out["R=0.6|s_x1.5"] >= 0.8)}}
    if save:
        json.dump(res, open(OUT.replace("_train", "_pooled_power"), "w"), indent=1)
        print(json.dumps(res, indent=1))
    return res


def test_pooled():
    """One look at A+B together (via test_oracle, Bence's verbatim OK for B): L/SE(L) > 1.645."""
    p1 = samples({1})
    ax = axes(p1)
    g = [gap(scored(p1, ax, cross_fit=True))] + [gap(scored(samples({p}, allow_sealed=True), ax, cross_fit=False))
                                                 for p in (2, 3)]
    L, se, z = contrast(*g)
    return {"D": [x["D"] for x in g], "SE": [x["se"] for x in g], "L": L, "SE_L": se, "z": z, "fading": bool(z > 1.645)}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if "--pooled-power" in sys.argv:
        pooled_power()
    elif "--readout" in sys.argv:
        readout_search()
    else:
        train()
