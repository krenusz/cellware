"""C17-S1PL-T: the C17-S1T nested test (memory per edge vs commitment) on the dated embryo tree, TRAIN founder clades.

Bence 2026-11-20: "Go with 1". Serves: C17-S1PL. Pre-registered in EXPERIMENT_LOG (C17-S1PL-T).
Model, fit, weights, simulator and planted commitment come unchanged from c17s1t_stage; tree, picker (amendment 2,
split-age-stratified) and founder clades from c17s1pl_tree. Train founders are filtered with partitions.train_only
before any label is read; metadata rows of sealed founders are skipped by row index and never parsed.

    python -I experiments/c17s1pl_fit.py selftest          # offline
    python -I experiments/c17s1pl_fit.py structure         # Actions: train pairs, founder label fractions; no agreement
    python -I experiments/c17s1pl_fit.py smoke <truth>     # Actions: synthetic labels on the real structure
    python -I experiments/c17s1pl_fit.py gate              # local: smoke + power gate
    python -I experiments/c17s1pl_fit.py fit               # Actions, after the gate and Bence's go
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c17s1pl_tree as P  # noqa: E402
import c17s1t_stage as T  # noqa: E402
from partitions import train_only  # noqa: E402

RES = HERE / "results_c17"
STRUCT = RES / "c17s1pl_t_structure.json"
GATE = RES / "c17s1pl_t_gate.json"
FIT = RES / "c17s1pl_t_fit.json"
LABEL = "major_trajectory"
MIN_PAIRS_UNIT, MIN_UNITS, MIN_PAIRS = 30, 20, 1000
POOL_FRAC = 0.01
DATASET = "dtt_founder"


# ---------------------------------------------------------------- loading
def subtree_end(parent):
    """Preorder: the subtree of node n is the index range [n, end[n])."""
    n = len(parent)
    size = np.ones(n, dtype=np.int64)
    for i in range(n - 1, 0, -1):
        if parent[i] >= 0:
            size[parent[i]] += size[i]
    return np.arange(n) + size


def founder_of(parent, is_f):
    f = np.full(len(parent), -1, dtype=np.int64)
    for i in range(len(parent)):
        if is_f[i]:
            f[i] = i
        elif parent[i] >= 0:
            f[i] = f[parent[i]]
    return f


def train_labels(names, leaf_train, path=None):
    """Labels of train leaves only: pass 1 reads the ID column, pass 2 parses only the rows of train leaves."""
    import pandas as pd

    want = {names[i] for i in np.flatnonzero(leaf_train)}
    path = path or (P.RAW / P.META)
    ids = pd.read_csv(path, sep="\t", usecols=[P.ID_COL])[P.ID_COL].astype(str)
    keep = set((np.flatnonzero(ids.isin(want).to_numpy()) + 1).tolist())  # +1: header is file row 0
    df = pd.read_csv(path, sep="\t", usecols=[P.ID_COL, LABEL], skiprows=lambda r: r != 0 and r not in keep)
    lab = df[LABEL].astype(str).str.strip()
    ok = ~lab.str.lower().isin(T.MISSING)
    return dict(zip(df[P.ID_COL].astype(str)[ok], lab[ok], strict=True))


def load_train():
    parent, blen, names, labelled = P.load()
    s = P.structure(parent, blen, labelled)
    fnd = P.founders(parent, s)
    is_f = np.zeros(len(parent), bool)
    is_f[fnd] = True
    f_of = founder_of(parent, is_f)
    import pandas as pd

    pd_units = pd.DataFrame({"unit": [f"F{f}" for f in fnd], "f": fnd})
    train_f = set(train_only(DATASET, pd_units, unit_col="unit")["f"].tolist())
    leaf_train = s["is_leaf"] & np.isin(f_of, list(train_f))
    lab = train_labels(names, leaf_train)
    rows = [r for r in P.pick_pairs(parent, s, P.CAP, True) if f_of[r[0]] in train_f]
    end = subtree_end(parent)
    order, starts, ends = P.children_of(parent)
    # pooled label set over train labelled leaves
    cnt = collections.Counter(lab.values())
    tot = sum(cnt.values())
    keep_cats = sorted(c for c, v in cnt.items() if v >= POOL_FRAC * tot)
    cat = {c: (c if c in keep_cats else "other") for c in cnt}
    cats = keep_cats + (["other"] if any(v == "other" for v in cat.values()) else [])
    ix = {c: i for i, c in enumerate(cats)}

    def arm_counts(node):
        v = np.zeros(len(cats), dtype=np.int64)
        for i in range(node, end[node]):
            if s["is_leaf"][i] and names[i] in lab:
                v[ix[cat[lab[names[i]]]]] += 1
        return v

    out = []
    for nd, _size, tau, _k, t_since, _la, _lb in rows:
        ch = order[starts[nd]:ends[nd]]
        a, b = ch[np.argsort(-s["leaves"][ch], kind="stable")][:2]
        c1, c2 = arm_counts(a), arm_counts(b)
        if c1.sum() < P.MIN_ARM or c2.sum() < P.MIN_ARM:
            continue
        # k over leaves with a non-missing label (as C17-S1T: labelled leaves of both arms)
        dep = [s["depth"][i] for arm in (a, b) for i in range(arm, end[arm]) if s["is_leaf"][i] and names[i] in lab]
        out.append((f"F{f_of[nd]}", tau, float(np.mean(dep) - s["depth"][nd]), t_since, c1, c2))
    pop = collections.defaultdict(lambda: np.zeros(len(cats), dtype=np.int64))
    for i in np.flatnonzero(leaf_train):
        if names[i] in lab:
            pop[f"F{f_of[i]}"][ix[cat[lab[names[i]]]]] += 1
    return out, dict(pop), cats, sorted(f"F{f}" for f in train_f)


# ---------------------------------------------------------------- stages
def structure():
    rows, pop, cats, train_units = load_train()
    unit = np.array([r[0] for r in rows])
    tau = np.array([r[1] for r in rows], float)
    k = np.array([r[2] for r in rows], float)
    per = collections.Counter(unit.tolist())
    eligible = sorted(u for u in per if per[u] >= MIN_PAIRS_UNIT)
    res = {"train_founders": train_units, "n_pairs": len(rows), "pairs_per_founder": dict(per),
           "eligible_embryos": eligible, "categories": cats,
           "embryo_label_counts": {u: pop[u].tolist() for u in sorted(pop)},
           "tau_mean": float(tau.mean()), "tau_sd": float(tau.std()),
           "tau_quantiles": np.percentile(tau, [0, 10, 50, 90, 100]).tolist(),
           "k_quantiles": np.percentile(k, [0, 10, 50, 90, 100]).tolist(),
           "t_since_quantiles": np.percentile([r[3] for r in rows], [0, 10, 50, 90, 100]).tolist(),
           "corr_tau_k": float(np.corrcoef(tau, k)[0, 1]),
           "pairs": {"embryo": unit.tolist(), "tau": tau.tolist(), "k": k.tolist(),
                     "n1": [int(r[4].sum()) for r in rows], "n2": [int(r[5].sum()) for r in rows]},
           "readout_gate_pass": len(eligible) >= MIN_UNITS and len(rows) >= MIN_PAIRS,
           "note": "structure only: depths, arm sizes, founder label fractions (train); no pair agreement computed. "
                   "Key 'embryo' holds the founder clade (c17s1t_stage field names reused)."}
    T.dump(res, STRUCT)
    T.say(json.dumps({k: v for k, v in res.items() if k not in ("pairs", "embryo_label_counts", "pairs_per_founder")},
                     indent=1))


def rule(err, eligible):
    wins = [u for u in eligible if err[u]["full"] < err[u]["mem"] and err[u]["full"] < err[u]["com"]]
    pooled = {m: float(np.mean([err[u][m] for u in eligible])) for m in T.MODELS}
    ok = len(wins) >= 2 * len(eligible) / 3 and pooled["full"] < pooled["mem"] and pooled["full"] < pooled["com"]
    return {"full_beats_both": wins, "n_eligible": len(eligible), "pooled": pooled, "pass": bool(ok)}


def loo(unit, z, k, Q, a, w, eligible, rng):
    err, lam_hat = {}, {}
    for u in eligible:
        tr, te = unit != u, unit == u
        err[u] = {}
        for m in T.MODELS:
            th, _, _ = T.fit(m, z[tr], k[tr], Q[tr], a[tr], w[tr], rng)
            r = a[te] - T.predict(th, m, z[te], k[te], Q[te])
            err[u][m] = float(np.sum(w[te] * r * r) / np.sum(w[te]))
            if m == "full":
                lam_hat[u] = float(T.unpack(th, m)[2])
    return {"err": err, "lambda_loo": lam_hat, **rule(err, eligible)}


def smoke(truth):
    T.STRUCT = STRUCT
    S = T.load_struct()
    lam, slope = T.TRUTHS[truth]
    al, be = T.planted_c(S, slope) if slope else (0.0, 0.0)
    rng = np.random.default_rng(T.SEED + 7 + sum(map(ord, truth)))
    w = T.weights(S["n1"], S["n2"])
    zb = np.digitize(S["z"], np.percentile(S["z"], [33.3, 66.7]))
    kb = np.digitize(S["k"], np.percentile(S["k"], [33.3, 66.7]))
    sets = []
    for i in range(T.N_SETS):
        a = T.simulate(S, lam, al, be, rng)
        r = loo(S["emb"], S["z"], S["k"], S["Q"], a, w, S["eligible"], rng)
        th, _, spread = T.fit("full", S["z"], S["k"], S["Q"], a, w, rng)
        cf = T.closed_form(lam, al, be, T.EPS, S["z"], S["k"], S["Q"]) if slope else \
            S["Q"] + (1 - S["Q"]) * (1 - T.EPS) * lam ** (2 * S["k"])
        bins = [abs(float(np.mean(a[m]) - np.mean(cf[m]))) for zz in range(3) for kk in range(3)
                if (m := (zb == zz) & (kb == kk)).sum() >= 30]
        sets.append({"pass": r["pass"], "wins": len(r["full_beats_both"]), "lambda_hat": float(T.unpack(th, "full")[2]),
                     "start_spread": spread, "analytic_max_bin_diff": max(bins) if bins else None})
        T.say(f"{truth} set {i + 1}/{T.N_SETS}: pass {r['pass']} wins {len(r['full_beats_both'])}/{r['n_eligible']} "
              f"lambda_hat {sets[-1]['lambda_hat']:.3f}")
    res = {"truth": truth, "lambda": lam, "slope": slope, "alpha": al, "beta": be, "eps": T.EPS, "n_sets": T.N_SETS,
           "pass_rate": float(np.mean([s["pass"] for s in sets])),
           "recovery_rate": float(np.mean([abs(s["lambda_hat"] - lam) <= 0.05 for s in sets])),
           "lambda_hat_median": float(np.median([s["lambda_hat"] for s in sets])),
           "analytic_max_bin_diff_mean": float(np.mean([s["analytic_max_bin_diff"] for s in sets
                                                         if s["analytic_max_bin_diff"] is not None])),
           "sets": sets}
    T.dump(res, RES / f"c17s1pl_t_smoke_{truth}.json")


N_ANALYTIC = 200


def bins_of(S):
    zb = np.digitize(S["z"], np.percentile(S["z"], [33.3, 66.7]))
    kb = np.digitize(S["k"], np.percentile(S["k"], [33.3, 66.7]))
    return {f"z{zz}k{kk}": m for zz in range(3) for kk in range(3) if (m := (zb == zz) & (kb == kk)).sum() >= 30}


def analytic():
    """Amendment 1 (Bence: "1"): per-bin bias of the simulator vs the closed form, averaged over sets (full truth)."""
    T.STRUCT = STRUCT
    S = T.load_struct()
    lam, slope = T.TRUTHS["full"]
    al, be = T.planted_c(S, slope)
    rng = np.random.default_rng(T.SEED + 101)
    cf = T.closed_form(lam, al, be, T.EPS, S["z"], S["k"], S["Q"])
    B = bins_of(S)
    d = np.array([[np.mean(a[m]) - np.mean(cf[m]) for m in B.values()]
                  for a in (T.simulate(S, lam, al, be, rng) for _ in range(N_ANALYTIC))])
    bias = d.mean(0)
    res = {"n_sets": N_ANALYTIC, "bins": list(B), "n_per_bin": [int(m.sum()) for m in B.values()],
           "bias_per_bin": bias.tolist(), "se_per_bin": (d.std(0, ddof=1) / np.sqrt(N_ANALYTIC)).tolist(),
           "per_set_sd_per_bin": d.std(0, ddof=1).tolist(), "max_abs_bias": float(np.max(np.abs(bias))),
           "per_set_max_mean": float(np.mean(np.max(np.abs(d), axis=1)))}
    T.dump(res, RES / "c17s1pl_t_analytic.json")
    print(json.dumps(res, indent=1))


def gate():
    r = {t: json.loads((RES / f"c17s1pl_t_smoke_{t}.json").read_text(encoding="utf-8")) for t in T.TRUTHS}
    an = json.loads((RES / "c17s1pl_t_analytic.json").read_text(encoding="utf-8"))
    checks = {
        "mem_false_pass<=0.10": r["mem"]["pass_rate"] <= 0.10,
        "com_false_pass<=0.10": r["com"]["pass_rate"] <= 0.10,
        "full_power>=0.8": r["full"]["pass_rate"] >= 0.8,
        "full_lambda_recovery>=0.8": r["full"]["recovery_rate"] >= 0.8,
        "analytic_bias_over_sets<=0.01": an["max_abs_bias"] <= 0.01,  # amendment 1
    }
    out = {"checks": checks, "pass": all(checks.values()), "amendment": "1 (Bence: \"1\")",
           "analytic_max_abs_bias": an["max_abs_bias"],
           "analytic_per_set_max_mean_as_preregistered": r["full"]["analytic_max_bin_diff_mean"],
           "summary": {t: {k: r[t][k] for k in ("pass_rate", "recovery_rate", "lambda_hat_median",
                                                 "analytic_max_bin_diff_mean")} for t in T.TRUTHS}}
    T.dump(out, GATE)
    print(json.dumps(out, indent=1))


def fit():
    """Train founder clades only; as C17-S1T real_fit, with the 2/3 rule of the C17-S1PL-T pre-registration."""
    if not json.loads(GATE.read_text(encoding="utf-8")).get("pass"):
        raise SystemExit("gate not passed: no fit (pre-registered order)")
    d = json.loads(STRUCT.read_text(encoding="utf-8"))
    rows, pop, cats, _ = load_train()
    assert cats == d["categories"] and len(rows) == d["n_pairs"], "pair set differs from the structure stage"
    unit = np.array([r[0] for r in rows])
    assert unit.tolist() == d["pairs"]["embryo"], "pair order differs from the structure stage"
    tau = np.array([r[1] for r in rows], float)
    k = np.array([r[2] for r in rows], float)
    z = (tau - d["tau_mean"]) / d["tau_sd"]
    pi = {u: v / v.sum() for u, v in pop.items()}
    Q = np.array([np.sum(pi[u] ** 2) for u in unit])
    c1 = np.array([r[4] for r in rows], float)
    c2 = np.array([r[5] for r in rows], float)
    n1, n2 = c1.sum(1), c2.sum(1)
    a = np.sum((c1 / n1[:, None]) * (c2 / n2[:, None]), axis=1)
    w = T.weights(n1, n2)
    rng = np.random.default_rng(T.SEED)
    lo = loo(unit, z, k, Q, a, w, d["eligible_embryos"], rng)
    full = {}
    for m in T.MODELS:
        th, loss, spread = T.fit(m, z, k, Q, a, w, rng)
        al, be, lam, eps, _ = T.unpack(th, m)
        full[m] = {"loss": loss, "start_spread": spread, "alpha": float(al), "beta_per_sd_tau": float(be),
                   "lambda": float(lam), "eps": float(eps)}
    th = T.fit("full", z, k, Q, a, w, rng)[0]
    tq = np.percentile(tau, [10, 50, 90])
    C_at = {f"tau_{t:.0f}": float(T.sig(th[0] + th[1] * (t - d["tau_mean"]) / d["tau_sd"])) for t in tq}
    pred = T.predict(th, "full", z, k, Q)
    calib = {b: {"n": int(m.sum()), "obs": float(np.mean(a[m])), "pred": float(np.mean(pred[m]))}
             for b, m in bins_of({"z": z, "k": k}).items()}
    lam_l = np.array(list(lo["lambda_loo"].values()))
    res = {"verdict": "STAGE-AWARE MODEL SUPPORTED (exploratory)" if lo["pass"] else "NOT SUPPORTED",
           "loo": lo, "lambda_loo_quantiles": np.percentile(lam_l, [0, 10, 50, 90, 100]).tolist(),
           "full_data_fits": full, "C_at_tau_quantiles": C_at, "calibration_in_sample": calib,
           "n_pairs": len(rows), "eligible_founders": d["eligible_embryos"], "label": LABEL,
           "note": "train founder clades of one embryo; lineage-constrained tree (biases toward M_com); "
                   "k in edges is a proxy for divisions; calibration uses the full-data fit (in-sample)."}
    T.dump(res, FIT)
    T.say(json.dumps({k: v for k, v in res.items() if k not in ("calibration_in_sample", "loo")}, indent=1))
    T.say(json.dumps({k: lo[k] for k in ("full_beats_both", "n_eligible", "pooled", "pass")}, indent=1))


def selftest():
    rng = np.random.default_rng(1)
    for _ in range(20):
        n = int(rng.integers(30, 300))
        pool = [f"L{i}:1" for i in range(n)]
        while len(pool) > 1:
            a, b = (pool.pop(int(rng.integers(len(pool)))) for _ in range(2))
            pool.append(f"({a},{b}):1")
        parent, _, names = P.parse_newick(pool[0].rsplit(":", 1)[0] + ";")
        end = subtree_end(parent)
        kids = collections.defaultdict(list)
        for i, p in enumerate(parent):
            if p >= 0:
                kids[p].append(i)
        for nd in rng.choice(len(parent), 10):
            st, seen = [int(nd)], set()
            while st:
                x = st.pop()
                seen.add(x)
                st.extend(kids[x])
            assert seen == set(range(nd, end[nd])), "preorder subtree range"
    err = {"u1": {"mem": 2, "com": 1.1, "full": 1.0}, "u2": {"mem": 2, "com": 1.1, "full": 1.0},
           "u3": {"mem": 2, "com": 0.9, "full": 1.0}}
    assert rule(err, ["u1", "u2", "u3"])["pass"] and not rule(err, ["u3"])["pass"]
    import os
    import tempfile

    tmp = os.path.join(tempfile.mkdtemp(), "m.tsv")
    with open(tmp, "w") as fh:  # sealed rows carry a value that would fail if parsed into the label check
        rows = [f"c{i}\t0\t{'SEALED' if i % 2 else 't' + str(i)}\n" for i in range(10)]
        fh.write(f"{P.ID_COL}\tx\t{LABEL}\n" + "".join(rows))
    got = train_labels([f"c{i}" for i in range(10)], np.array([i % 2 == 0 for i in range(10)]), tmp)
    assert set(got) == {f"c{i}" for i in range(0, 10, 2)} and "SEALED" not in got.values(), got
    print("selftest ok: preorder subtree ranges, rule function, train-only label rows")


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "smoke":
        smoke(sys.argv[2])
    else:
        {"selftest": selftest, "structure": structure, "analytic": analytic, "gate": gate, "fit": fit}[stage]()
