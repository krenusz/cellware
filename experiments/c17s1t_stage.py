"""C17-S1T: stage-aware memory model, commitment vs per-division memory on the embryo trees, TRAIN embryos only.

Bence 2026-11-20: "Yes, pre-register", then "1". Serves: C17-S1T. Pre-registered in EXPERIMENT_LOG (C17-S1T).

Pair-level closed form for sister-subclade label agreement a = sum_j f1_j f2_j:
    a(tau, k) = Q + (1 - Q) (1 - eps) [C(tau) + (1 - C(tau)) lambda^(2k)],  C(tau) = logistic(alpha + beta tau)
M_mem (C = 0), M_com (lambda = 0), M_full (both); leave-one-embryo-out held-out weighted squared error.
Loading goes through c14s0_gate (train embryos filtered with partitions.train_only before any other column; trees of
other partitions skipped by key before parsing).

Stages (each runs only if the previous gate passed):
    python -I experiments/c17s1t_stage.py selftest          # offline: pair picker depths, closed form vs simulator, fit
    python -I experiments/c17s1t_stage.py structure         # Actions --petracer: pairs (tau, k, n1, n2), embryo
                                                            #   fractions; readout gate; no agreement computed
    python -I experiments/c17s1t_stage.py smoke <truth>     # Actions, no data: synthetic labels on the real structure
    python -I experiments/c17s1t_stage.py gate              # local: smoke + power gate from the smoke files
    python -I experiments/c17s1t_stage.py fit               # Actions --petracer: the pre-registered train fit
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from partitions import train_only  # noqa: E402,F401  (guard: partitioned reads go through train_only, in c14s0_gate)

RES = HERE / "results_c17"
STRUCT = RES / "c17s1t_structure.json"
GATE = RES / "c17s1t_gate.json"
FIT = RES / "c17s1t_fit.json"
MAX_CLADE, MIN_ARM = 300, 4  # as C17-S1P
MIN_PAIRS_EMB, MIN_EMB = 30, 5  # readout gate (H21)
LAM, EPS = 0.935, 0.12  # C16-S1 half-life 10.3 edges; C17-S1P D_noise / D_rand
N_SETS, STARTS, W_Q = 50, 5, 95
SEED = 20261120
MISSING = {"", "nan", "na", "none", "unknown"}
MODELS = ("mem", "com", "full")
TRUTHS = {  # name: (lambda, commitment slope factor); slope factor 0 = no commitment
    "mem": (LAM, 0.0), "com": (0.0, 1.0), "full": (LAM, 1.0),
    "l085": (0.85, 1.0), "l097": (0.97, 1.0), "half": (LAM, 0.5),
}
LOG: list[str] = []


def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s, flush=True)


def dump(obj, path):
    path.parent.mkdir(exist_ok=True)
    conv = lambda o: o.item() if hasattr(o, "item") else (o.tolist() if hasattr(o, "tolist") else str(o))  # noqa: E731
    path.write_text(json.dumps(obj, indent=1, default=conv), encoding="utf-8")


# ---------------------------------------------------------------- pairs (C17-S1P picker + depths)
def pairs_of(tree, lab):
    """Disjoint sister-subclade pairs, smallest clades first (C17-S1P rule), with depths in edges.

    Returns (clade size, tau = node depth from its root, k = mean depth of the pair's labelled leaves below the node,
    labels arm 1, labels arm 2)."""
    parent, children = {}, collections.defaultdict(list)
    for p, ch in tree["edges"].items():
        for c in ch:
            parent[c] = p
            children[p].append(c)
    nodes = set(parent) | set(tree["edges"])
    roots = [n for n in nodes if n not in parent]
    order, stack, depth = [], list(roots), {r: 0 for r in roots}
    while stack:
        n = stack.pop()
        order.append(n)
        for c in children.get(n, []):
            depth[c] = depth[n] + 1
            stack.append(c)
    lists = {}
    for n in reversed(order):  # children before parents
        if n not in children:
            lists[n] = [n]
        elif all(c in lists for c in children[n]):
            leaves = [x for c in children[n] for x in lists[c]]
            if len(leaves) <= MAX_CLADE:
                lists[n] = leaves
    cand = []
    for n, ch in children.items():
        if n not in lists or len(ch) < 2:
            continue
        a, b = sorted(ch, key=lambda c: -len(lists[c]))[:2]
        xa = [x for x in lists[a] if x in lab]
        xb = [x for x in lists[b] if x in lab]
        if len(xa) >= MIN_ARM and len(xb) >= MIN_ARM:
            k = float(np.mean([depth[x] - depth[n] for x in xa + xb]))
            cand.append((len(lists[n]), str(n), depth[n], k, [lab[x] for x in xa], [lab[x] for x in xb],
                         set(lists[n])))
    cand.sort(key=lambda t: (t[0], t[1]))
    used, out = set(), []
    for size, _, tau, k, la, lb, leaves in cand:
        if leaves & used:
            continue
        used |= leaves
        out.append((size, tau, k, la, lb))
    return out


def load_pairs():
    """Train embryos only. Returns rows (embryo, size, tau, k, labels1, labels2) and per-embryo label counts."""
    import c14s0_gate as g0

    obs, train_embryos, raw = g0.load_obs()
    lab = {c: str(v) for c, v in obs["lineage"].items() if str(v).strip().lower() not in MISSING}
    emb_of = dict(obs["embryo"].items())
    pop = collections.defaultdict(collections.Counter)
    for c, v in lab.items():
        pop[emb_of[c]][v] += 1
    del obs
    rows = []
    for key, tree in g0.iter_train_trees(raw, train_embryos):
        emb = key.rsplit("-C", 1)[0]
        rows += [(emb, *p) for p in pairs_of(tree, lab)]
    del raw
    return train_embryos, rows, pop


# ---------------------------------------------------------------- model
def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def unpack(theta, model):
    if model == "mem":
        return 0.0, 0.0, sig(theta[0]), sig(theta[1]), True
    if model == "com":
        return theta[0], theta[1], 0.0, sig(theta[2]), False
    return theta[0], theta[1], sig(theta[2]), sig(theta[3]), True


def predict(theta, model, z, k, Q):
    """z = standardised tau. Returns expected agreement."""
    al, be, lam, eps, has_mem = unpack(theta, model)
    C = np.zeros_like(z) if model == "mem" else sig(al + be * z)
    m = lam ** (2 * k) if has_mem else np.zeros_like(k)
    return Q + (1 - Q) * (1 - eps) * (C + (1 - C) * m)


def closed_form(lam, al, be, eps, z, k, Q):
    C = sig(al + be * z)
    return Q + (1 - Q) * (1 - eps) * (C + (1 - C) * lam ** (2 * k))


NPAR = {"mem": 2, "com": 3, "full": 4}


def fit(model, z, k, Q, a, w, rng, starts=STARTS):
    def loss(th):
        r = a - predict(th, model, z, k, Q)
        return float(np.sum(w * r * r) / np.sum(w))

    best, vals = None, []
    for _s in range(starts):
        th0 = rng.normal(0, 1.5, NPAR[model])
        if model in ("mem", "full"):
            th0[-2] = rng.uniform(0.5, 4.0)  # lambda start 0.62-0.98
        th0[-1] = rng.uniform(-4, -1)  # eps start 0.02-0.27
        r = minimize(loss, th0, method="L-BFGS-B")
        vals.append(r.fun)
        if best is None or r.fun < best.fun:
            best = r
    return best.x, float(best.fun), float(np.max(vals) - np.min(vals))


def weights(n1, n2):
    w = (n1 * n2).astype(float)
    return np.minimum(w, np.percentile(w, W_Q))


def loeo(emb, z, k, Q, a, w, eligible, rng):
    """Held-out weighted squared error per eligible embryo for each model, and the rule."""
    err, lam_hat = {}, {}
    for e in eligible:
        tr, te = emb != e, emb == e
        err[e] = {}
        for m in MODELS:
            th, _, _ = fit(m, z[tr], k[tr], Q[tr], a[tr], w[tr], rng)
            r = a[te] - predict(th, m, z[te], k[te], Q[te])
            err[e][m] = float(np.sum(w[te] * r * r) / np.sum(w[te]))
            if m == "full":
                lam_hat[e] = float(unpack(th, m)[2])
    wins = [e for e in eligible if err[e]["full"] < err[e]["mem"] and err[e]["full"] < err[e]["com"]]
    return {"err": err, "full_beats_both": wins, "n_eligible": len(eligible),
            "pass": len(wins) >= len(eligible) - 1, "lambda_loeo": lam_hat}


# ---------------------------------------------------------------- synthetic
def simulate(S, lam, al, be, rng):
    """Synthetic arm counts on the real structure; returns per-pair agreement a."""
    eta = 1 - np.sqrt(1 - EPS)  # per-cell label noise, so (1 - eta)^2 = 1 - eps
    C = sig(al + be * S["z"]) if be != 0 else np.zeros(len(S["z"]))
    s = np.sqrt(C + (1 - C) * lam ** (2 * S["k"])) * (1 - eta)
    a = np.empty(len(s))
    for e, pi in S["pi"].items():
        ix = np.flatnonzero(S["emb"] == e)
        if not len(ix):
            continue
        L = rng.choice(len(pi), len(ix), p=pi)
        f = []
        for n in (S["n1"][ix], S["n2"][ix]):
            c = rng.binomial(n, s[ix])
            cnt = rng.multinomial(n - c, pi)
            cnt[np.arange(len(ix)), L] += c
            f.append(cnt / n[:, None])
        a[ix] = np.sum(f[0] * f[1], axis=1)
    return a


def load_struct():
    d = json.loads(STRUCT.read_text(encoding="utf-8"))
    p = d["pairs"]
    emb = np.array(p["embryo"])
    S = {"emb": emb, "tau": np.array(p["tau"], float), "k": np.array(p["k"], float),
         "n1": np.array(p["n1"], int), "n2": np.array(p["n2"], int),
         "pi": {e: np.array(v, float) / np.sum(v) for e, v in d["embryo_label_counts"].items()}}
    S["z"] = (S["tau"] - d["tau_mean"]) / d["tau_sd"]
    S["Q"] = np.array([np.sum(S["pi"][e] ** 2) for e in emb])
    S["eligible"] = d["eligible_embryos"]
    S["d"] = d
    return S


def planted_c(S, slope):
    """logistic(alpha + beta z) going 0.1 -> 0.9 across the 10th-90th percentile of z; slope scales beta at the
    same midpoint."""
    z10, z90 = np.percentile(S["z"], [10, 90])
    lg = np.log(9.0)
    be = 2 * lg / (z90 - z10)
    mid = (z10 + z90) / 2
    be *= slope
    return -be * mid, be


def smoke(truth):
    S = load_struct()
    lam, slope = TRUTHS[truth]
    al, be = planted_c(S, slope) if slope else (0.0, 0.0)
    rng = np.random.default_rng(SEED + sum(map(ord, truth)))
    w = weights(S["n1"], S["n2"])
    sets = []
    zb = np.digitize(S["z"], np.percentile(S["z"], [33.3, 66.7]))
    kb = np.digitize(S["k"], np.percentile(S["k"], [33.3, 66.7]))
    for i in range(N_SETS):
        a = simulate(S, lam, al, be, rng)
        r = loeo(S["emb"], S["z"], S["k"], S["Q"], a, w, S["eligible"], rng)
        th, _, spread = fit("full", S["z"], S["k"], S["Q"], a, w, rng)
        cf = closed_form(lam, al, be, EPS, S["z"], S["k"], S["Q"]) if slope else \
            S["Q"] + (1 - S["Q"]) * (1 - EPS) * lam ** (2 * S["k"])
        bins = [abs(float(np.mean(a[m]) - np.mean(cf[m]))) for zz in range(3) for kk in range(3)
                if (m := (zb == zz) & (kb == kk)).sum() >= 30]
        sets.append({"pass": r["pass"], "wins": len(r["full_beats_both"]), "lambda_hat": float(unpack(th, "full")[2]),
                     "start_spread": spread, "analytic_max_bin_diff": max(bins) if bins else None})
        say(f"{truth} set {i + 1}/{N_SETS}: pass {r['pass']} wins {len(r['full_beats_both'])}/{r['n_eligible']} "
            f"lambda_hat {sets[-1]['lambda_hat']:.3f}")
    res = {"truth": truth, "lambda": lam, "slope": slope, "alpha": al, "beta": be, "eps": EPS, "n_sets": N_SETS,
           "pass_rate": float(np.mean([s["pass"] for s in sets])),
           "recovery_rate": float(np.mean([abs(s["lambda_hat"] - lam) <= 0.05 for s in sets])),
           "lambda_hat_median": float(np.median([s["lambda_hat"] for s in sets])),
           "analytic_max_bin_diff_mean": float(np.mean([s["analytic_max_bin_diff"] for s in sets
                                                         if s["analytic_max_bin_diff"] is not None])),
           "sets": sets}
    dump(res, RES / f"c17s1t_smoke_{truth}.json")
    say(json.dumps({k: v for k, v in res.items() if k != "sets"}))


def gate():
    r = {t: json.loads((RES / f"c17s1t_smoke_{t}.json").read_text(encoding="utf-8")) for t in TRUTHS}
    checks = {
        "mem_false_pass<=0.10": r["mem"]["pass_rate"] <= 0.10,
        "com_false_pass<=0.10": r["com"]["pass_rate"] <= 0.10,
        "full_power>=0.8": r["full"]["pass_rate"] >= 0.8,
        "full_lambda_recovery>=0.8": r["full"]["recovery_rate"] >= 0.8,
        "analytic_bin_diff<=0.01": r["full"]["analytic_max_bin_diff_mean"] <= 0.01,
    }
    out = {"checks": checks, "pass": all(checks.values()),
           "summary": {t: {k: r[t][k] for k in ("pass_rate", "recovery_rate", "lambda_hat_median",
                                                 "analytic_max_bin_diff_mean")} for t in TRUTHS}}
    dump(out, GATE)
    print(json.dumps(out, indent=1))


# ---------------------------------------------------------------- real data stages
def structure():
    train_embryos, rows, pop = load_pairs()
    per = collections.Counter(r[0] for r in rows)
    cats = sorted({x for c in pop.values() for x in c})
    tau = np.array([r[2] for r in rows], float)
    k = np.array([r[3] for r in rows], float)
    emb = np.array([r[0] for r in rows])
    corr = {e: float(np.corrcoef(tau[emb == e], k[emb == e])[0, 1]) for e in per if per[e] >= 3}
    eligible = sorted(e for e in per if per[e] >= MIN_PAIRS_EMB)
    res = {"train_embryos": train_embryos, "n_pairs": len(rows), "pairs_per_embryo": dict(per),
           "eligible_embryos": eligible, "categories": cats,
           "embryo_label_counts": {e: [pop[e][c] for c in cats] for e in sorted(pop)},
           "tau_mean": float(tau.mean()), "tau_sd": float(tau.std()),
           "tau_quantiles": np.percentile(tau, [0, 10, 50, 90, 100]).tolist(),
           "k_quantiles": np.percentile(k, [0, 10, 50, 90, 100]).tolist(),
           "tau_k_corr_within_embryo": corr,
           "pairs": {"embryo": emb.tolist(), "tau": tau.tolist(), "k": k.tolist(),
                     "n1": [len(r[4]) for r in rows], "n2": [len(r[5]) for r in rows]},
           "readout_gate_pass": len(eligible) >= MIN_EMB,
           "note": "structure only: depths, arm sizes and embryo label fractions; no pair agreement computed"}
    dump(res, STRUCT)
    say(json.dumps({k: v for k, v in res.items() if k != "pairs"}, indent=1))


def real_fit():
    g = json.loads(GATE.read_text(encoding="utf-8"))
    if not g.get("pass"):
        raise SystemExit("gate not passed: no fit (pre-registered order)")
    d = json.loads(STRUCT.read_text(encoding="utf-8"))
    train_embryos, rows, pop = load_pairs()
    cats = d["categories"]
    assert len(rows) == d["n_pairs"], "pair set differs from the structure stage"
    ix = {c: i for i, c in enumerate(cats)}
    emb = np.array([r[0] for r in rows])
    tau = np.array([r[2] for r in rows], float)
    k = np.array([r[3] for r in rows], float)
    z = (tau - d["tau_mean"]) / d["tau_sd"]
    pi = {e: np.array([pop[e][c] for c in cats], float) / sum(pop[e].values()) for e in pop}
    Q = np.array([np.sum(pi[e] ** 2) for e in emb])
    n1 = np.array([len(r[4]) for r in rows])
    n2 = np.array([len(r[5]) for r in rows])
    f1 = np.zeros((len(rows), len(cats)))
    f2 = np.zeros_like(f1)
    for i, r in enumerate(rows):
        for x in r[4]:
            f1[i, ix[x]] += 1
        for x in r[5]:
            f2[i, ix[x]] += 1
    a = np.sum((f1 / n1[:, None]) * (f2 / n2[:, None]), axis=1)
    w = weights(n1, n2)
    rng = np.random.default_rng(SEED)
    lo = loeo(emb, z, k, Q, a, w, d["eligible_embryos"], rng)
    full = {}
    for m in MODELS:
        th, loss, spread = fit(m, z, k, Q, a, w, rng)
        al, be, lam, eps, _ = unpack(th, m)
        full[m] = {"loss": loss, "start_spread": spread, "alpha": float(al), "beta_per_sd_tau": float(be),
                   "lambda": float(lam), "eps": float(eps)}
    th = fit("full", z, k, Q, a, w, rng)[0]
    al, be = th[0], th[1]
    tq = np.percentile(tau, [10, 50, 90])
    C_at = {f"tau_{t:.0f}": float(sig(al + be * (t - d["tau_mean"]) / d["tau_sd"])) for t in tq}
    # adequacy: held-out calibration of M_full per tau x k tertile bin (predictions from the LOEO fits are not kept,
    # so the full-data fit is used here and labelled as in-sample)
    zb = np.digitize(z, np.percentile(z, [33.3, 66.7]))
    kb = np.digitize(k, np.percentile(k, [33.3, 66.7]))
    pred = predict(th, "full", z, k, Q)
    calib = {f"z{i}k{j}": {"n": int(m.sum()), "obs": float(np.mean(a[m])), "pred": float(np.mean(pred[m]))}
             for i in range(3) for j in range(3) if (m := (zb == i) & (kb == j)).sum()}
    res = {"verdict": "STAGE-AWARE MODEL SUPPORTED (exploratory)" if lo["pass"] else "NOT SUPPORTED", "loeo": lo,
           "full_data_fits": full, "C_at_tau_quantiles": C_at, "calibration_in_sample": calib,
           "n_pairs": len(rows), "eligible_embryos": d["eligible_embryos"]}
    dump(res, FIT)
    say(json.dumps({k: v for k, v in res.items() if k != "calibration_in_sample"}, indent=1))


# ---------------------------------------------------------------- selftest
def selftest():
    import c17s1p_trees as s1p

    leaves4 = lambda p: [f"{p}{i}" for i in range(4)]  # noqa: E731
    edges = {"r": ["n1", "n2"], "n1": ["x", "y"], "n2": ["u", "v"], "x": leaves4("x"), "y": leaves4("y"),
             "u": leaves4("u"), "v": leaves4("v")}
    lab = {c: ("A" if c[0] in "xu" else "B") for p in "xyuv" for c in leaves4(p)}
    pr = pairs_of({"edges": edges}, lab)
    old = s1p.pairs_of({"edges": edges}, lab)
    assert [(s, la, lb) for s, _, _, la, lb in pr] == old, (pr, old)  # same pairs as C17-S1P
    assert all(t == 1 and kk == 2.0 for _, t, kk, _, _ in pr), pr  # n1/n2 at depth 1, leaves at depth 3
    # closed form vs simulator on a fake structure
    rng = np.random.default_rng(3)
    n, embs = 4000, ["e0", "e1", "e2", "e3", "e4", "e5"]
    pis = {e: rng.dirichlet(np.ones(8)) for e in embs}
    emb = np.array([embs[i % 6] for i in range(n)])
    depth = np.array([6 + 2 * embs.index(e) for e in emb])  # embryos differ in depth, as stages do
    k = rng.integers(1, 8, n).astype(float)
    tau = np.maximum(depth + rng.integers(0, 6, n) - k, 0).astype(float)
    S = {"emb": emb, "z": (tau - tau.mean()) / tau.std(), "k": k, "n1": rng.integers(4, 15, n),
         "n2": rng.integers(4, 15, n), "pi": pis}
    S["Q"] = np.array([np.sum(pis[e] ** 2) for e in emb])
    al, be = planted_c(S, 1.0)
    a = np.mean([simulate(S, LAM, al, be, rng) for _ in range(40)], axis=0)
    cf = closed_form(LAM, al, be, EPS, S["z"], S["k"], S["Q"])
    diff = float(abs(a.mean() - cf.mean()))
    assert diff < 0.005, diff
    a1 = simulate(S, LAM, al, be, rng)
    w = weights(S["n1"], S["n2"])
    th = fit("full", S["z"], S["k"], S["Q"], a1, w, rng)[0]
    lam_hat = float(unpack(th, "full")[2])
    lo = loeo(S["emb"], S["z"], S["k"], S["Q"], a1, w, embs, rng)
    return {"ok": True, "pairs": len(pr), "closed_form_diff": diff, "lambda_hat_toy": lam_hat,
            "toy_rule": {"pass": lo["pass"], "wins": len(lo["full_beats_both"])}}


def main():
    st = sys.argv[1] if len(sys.argv) > 1 else ""
    if st == "selftest":
        print(json.dumps(selftest(), indent=1))
    elif st == "structure":
        structure()
    elif st == "smoke":
        smoke(sys.argv[2])
    elif st == "gate":
        gate()
    elif st == "fit":
        real_fit()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
