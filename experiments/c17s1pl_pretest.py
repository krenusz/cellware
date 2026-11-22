"""C17-S1PL-P feasibility pretests (Bence 2026-11-21: "can you just do a lot of pretest and see what could even work?").

Serves: C17-S1PL (EXPERIMENT_LOG C17-S1PL-P). Train founder clades only (partitions.train_only inside
c17s1pl_fit.load_train); synthetic labels on the real train structure; no sibling agreement is computed.

    python -I experiments/c17s1pl_pretest.py <clock|edges_celltype> <n_sets>   # Actions
"""

import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c17s1pl_fit as F  # noqa: E402
import c17s1t_stage as T  # noqa: E402
from partitions import train_only  # noqa: E402  (also applied inside F.load_train)

RES = HERE / "results_c17"
DAY_END = 13.5  # every leaf of the dated tree sits at 13.5 days (C17-S1PL-T2 pre-flight)


def struct(label):
    F.LABEL = label
    rows, pop, cats, _ = F.load_train()
    unit = np.array([r[0] for r in rows])
    per = collections.Counter(unit.tolist())
    elig = sorted(u for u in per if per[u] >= F.MIN_PAIRS_UNIT)
    tau = np.array([r[1] for r in rows], float)
    k = np.array([r[2] for r in rows], float)
    date = DAY_END - np.array([r[3] for r in rows], float)
    Q = [float(np.sum((v / v.sum()) ** 2)) for v in pop.values()]
    info = {"label": label, "n_cats": len(cats), "n_pairs": len(rows), "n_eligible": len(elig),
            "readout_gate_pass": len(elig) >= F.MIN_UNITS and len(rows) >= F.MIN_PAIRS,
            "corr_date_k": float(np.corrcoef(date, k)[0, 1]), "corr_tau_k": float(np.corrcoef(tau, k)[0, 1]),
            "Q_median": float(np.median(Q))}
    S = {"emb": unit, "k": k, "tau": tau, "date": date, "n1": np.array([int(r[4].sum()) for r in rows]),
         "n2": np.array([int(r[5].sum()) for r in rows]),
         "pi": {u: v / v.sum() for u, v in pop.items()}, "eligible": elig}
    S["Q"] = np.array([np.sum(S["pi"][u] ** 2) for u in unit])
    return S, info


def power(variant, n_sets, eps=None):
    label, cov = {"clock": ("major_trajectory", "date"), "edges_celltype": ("celltype", "tau"),
                  "edges_major": ("major_trajectory", "tau")}[variant]
    if eps is not None:  # P6: label noise at the level fitted on real data (C17-S1PL-T eps 0.57)
        T.EPS = eps
    S, info = struct(label)
    T.say(json.dumps(info))
    x = S[cov]
    S["z"] = (x - x.mean()) / x.std()
    w = T.weights(S["n1"], S["n2"])
    out = {"variant": variant, "label": label, "commitment_covariate": cov, "eps": T.EPS, "structure": info}
    for truth in ("mem", "com", "full", "half"):
        lam, slope = T.TRUTHS[truth]
        al, be = T.planted_c(S, slope) if slope else (0.0, 0.0)
        rng = np.random.default_rng(T.SEED + 900 + sum(map(ord, truth + variant)))
        res, t0 = [], time.time()
        for _ in range(n_sets):
            a = T.simulate(S, lam, al, be, rng)
            r = F.loo(S["emb"], S["z"], S["k"], S["Q"], a, w, S["eligible"], rng)
            res.append((r["pass"], float(np.median(list(r["lambda_loo"].values())))))
        out[truth] = {"pass_rate": float(np.mean([q[0] for q in res])),
                      "recovery": float(np.mean([abs(q[1] - lam) <= 0.05 for q in res])), "n_sets": n_sets}
        T.say(f"{variant} {truth} {out[truth]} {time.time() - t0:.0f}s")
    T.dump(out, RES / f"c17s1pl_p_power_{variant}{'_eps' + str(eps) if eps is not None else ''}.json")


UNCON = "tree_building/results/3-dated-tree/merged_minB2h_unconstrained.nwk"


def dates():
    """P5: split dates of the train pairs under constrained (age ladder) vs unconstrained dating. Structure only."""
    import hashlib
    import tempfile
    import urllib.request

    import c17s1pl_tree as P
    import pandas as pd

    url = P.BASE.replace("support_data/", "") + UNCON
    raw = urllib.request.urlopen(url, timeout=300).read()
    path = Path(tempfile.mkdtemp()) / "uncon.nwk"
    path.write_bytes(raw)
    parent, blen, names, lab = P.load()
    s = P.structure(parent, blen, lab)
    up, ub, un = P.parse_newick(path.read_text())
    same = len(up) == len(parent) and bool(np.all(up == parent)) and un == names
    out = {"url": url, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "same_topology": same,
           "nodes": [len(parent), len(up)]}

    def leafset_hash(par, nm, key):  # node hash = sum of random 64-bit keys of its leaves (wraps); matches clades
        h = np.zeros(len(par), dtype=np.uint64)
        isl = np.ones(len(par), bool)
        isl[par[par >= 0]] = False
        for i in np.flatnonzero(isl):
            h[i] = key.get(nm[i], 0)
        for i in range(len(par) - 1, 0, -1):
            if par[i] >= 0:
                h[par[i]] += h[i]
        return h, isl

    rk = np.random.default_rng(7)
    leaf = s["is_leaf"]
    key = {names[i]: np.uint64(rk.integers(1, 2**63)) for i in np.flatnonzero(leaf)}
    with np.errstate(over="ignore"):
        hc, _ = leafset_hash(parent, names, key)
        hu, isl_u = leafset_hash(up, un, key)
    out["uncon_leaves"] = int(isl_u.sum())
    out["uncon_leaves_named_in_backbone"] = int(sum(un[i] in key for i in np.flatnonzero(isl_u)))
    us = P.structure(up, ub, np.array([nm in key for nm in un]))
    pos = {int(v): i for i, v in enumerate(hu)}
    if True:
        leaf = s["is_leaf"]
        fnd = P.founders(parent, s)
        is_f = np.zeros(len(parent), bool)
        is_f[fnd] = True
        f_of = F.founder_of(parent, is_f)
        units = pd.DataFrame({"unit": [f"F{f}" for f in fnd], "f": fnd})
        train_f = set(train_only("dtt_founder", units, unit_col="unit")["f"].tolist())
        rows = [r for r in P.pick_pairs(parent, s, P.CAP, True) if f_of[r[0]] in train_f]
        nd_all = np.array([r[0] for r in rows])
        mu = np.array([pos.get(int(hc[n]), -1) for n in nd_all])
        out["pair_nodes_found_in_uncon"] = float(np.mean(mu >= 0))
        nd, k = nd_all[mu >= 0], np.array([r[3] for r in rows], float)[mu >= 0]
        dc, du = s["rdist"][nd], us["rdist"][mu[mu >= 0]]
        out.update({"n_train_pairs_structure": len(rows),
                    "uncon_leaf_rootdist_pct_0_50_100": np.percentile(us["rdist"][isl_u], [0, 50, 100]).tolist(),
                    "corr_dates": float(np.corrcoef(dc, du)[0, 1]),
                    "spearman_dates": float(pd.Series(dc).corr(pd.Series(du), method="spearman")),
                    "abs_shift_days_pct_50_90_99": np.percentile(np.abs(dc - du), [50, 90, 99]).tolist(),
                    "date_pct_10_50_90": {"constrained": np.percentile(dc, [10, 50, 90]).tolist(),
                                          "unconstrained": np.percentile(du, [10, 50, 90]).tolist()},
                    "corr_k": {"constrained": float(np.corrcoef(dc, k)[0, 1]),
                               "unconstrained": float(np.corrcoef(du, k)[0, 1])}})
    T.say(json.dumps(out, indent=1))
    T.dump(out, RES / "c17s1pl_p_dates2.json")


def apower(n_draws=20000):
    """P2: power of a frozen-parameter A test from the spent train per-founder LOO errors (no sealed read)."""
    from scipy import stats

    a = json.loads((RES / "c17s1pl_t_fit.json").read_text(encoding="utf-8"))["loo"]["err"]
    b = json.loads((RES / "c17s1pl_t3_fit.json").read_text(encoding="utf-8"))["loo"]["err"]
    d0 = np.array([a[u]["full"] - b[u]["full"] for u in sorted(a)])
    rng = np.random.default_rng(T.SEED + 2)
    out = {"train": {"n": len(d0), "mean": float(d0.mean()), "sd": float(d0.std(ddof=1)), "wins": int((d0 > 0).sum())}}
    for eff, shift in (("observed", 0.0), ("half", d0.mean() / 2), ("null", d0.mean())):
        d = d0 - shift
        for n in (10, 12, 13, 14, 16, 19, 27):
            x = d[rng.integers(0, len(d), (n_draws, n))]
            m, sd = x.mean(1), x.std(1, ddof=1)
            p = 1 - stats.t.cdf(m / (sd / np.sqrt(n)), n - 1)
            r1 = (p < 0.05) & (m > 0)
            r2 = (m > 0) & ((x > 0).sum(1) >= np.ceil(2 * n / 3))
            out[f"{eff}_n{n}"] = {"R1": float(r1.mean()), "R2": float(r2.mean())}
    T.say(json.dumps(out, indent=1))
    T.dump(out, RES / "c17s1pl_p2_apower.json")


def _pred_split(th, z, k, Q, g, split):
    al, be = th[0], th[1]
    l1 = T.sig(th[2])
    l2 = T.sig(th[3]) if split else l1
    eps = T.sig(th[-1])
    lam = np.where(g, l1, l2)
    C = T.sig(al + be * z)
    return Q + (1 - Q) * (1 - eps) * (C + (1 - C) * lam ** (2 * k))


def _fit_split(z, k, Q, a, w, g, split, rng):
    from scipy.optimize import minimize

    def loss(th):
        r = a - _pred_split(th, z, k, Q, g, split)
        return float(np.sum(w * r * r) / np.sum(w))

    n = 5 if split else 4
    best = None
    for _ in range(T.STARTS):
        th0 = rng.normal(0, 1.5, n)
        th0[2:n - 1] = rng.uniform(0.5, 4.0, n - 3)
        th0[-1] = rng.uniform(-4, -1)
        r = minimize(loss, th0, method="L-BFGS-B")
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def split_power(n_sets, eps):
    """P4 (W5): does memory per edge differ between early and late splits? Synthetic labels, train structure."""
    T.EPS = eps
    S, info = struct("major_trajectory")
    x = S["date"]
    S["z"] = (x - x.mean()) / x.std()
    g = x < np.median(x)
    w = T.weights(S["n1"], S["n2"])
    al, be = -0.018, 1.034  # C17-S1PL-T3 full-data fit (date clock)
    out = {"eps": eps, "structure": info, "early_frac": float(g.mean()),
           "k_pct_early": np.percentile(S["k"][g], [10, 50, 90]).tolist(),
           "k_pct_late": np.percentile(S["k"][~g], [10, 50, 90]).tolist()}
    for truth, (le, ll) in {"null": (0.90, 0.90), "s80_95": (0.80, 0.95), "s85_95": (0.85, 0.95)}.items():
        rng = np.random.default_rng(T.SEED + 400 + sum(map(ord, truth)) + int(eps * 100))
        lam = np.where(g, le, ll)
        res = []
        for _ in range(n_sets):
            a = T.simulate(S, lam, al, be, rng)
            err = {}
            for u in S["eligible"]:
                tr, te = S["emb"] != u, S["emb"] == u
                err[u] = {}
                for name, sp in (("one", False), ("split", True)):
                    th = _fit_split(S["z"][tr], S["k"][tr], S["Q"][tr], a[tr], w[tr], g[tr], sp, rng)
                    r = a[te] - _pred_split(th, S["z"][te], S["k"][te], S["Q"][te], g[te], sp)
                    err[u][name] = float(np.sum(w[te] * r * r) / np.sum(w[te]))
            wins = sum(err[u]["split"] < err[u]["one"] for u in err)
            pooled = {m: float(np.mean([err[u][m] for u in err])) for m in ("one", "split")}
            res.append(wins >= 2 * len(err) / 3 and pooled["split"] < pooled["one"])
        out[truth] = {"pass_rate": float(np.mean(res)), "n_sets": n_sets}
        T.say(f"split eps {eps} {truth} {out[truth]}")
    T.dump(out, RES / f"c17s1r_p4_split_eps{eps}.json")


def split_selftest():
    rng = np.random.default_rng(1)
    n = 4000
    z, k, Q = rng.normal(size=n), rng.uniform(2, 8, n), np.full(n, 0.2)
    g = rng.random(n) < 0.5
    th = np.array([0.0, 1.0, T.logit(0.8) if hasattr(T, "logit") else np.log(0.8 / 0.2),
                   np.log(0.95 / 0.05), np.log(0.4 / 0.6)])
    a = _pred_split(th, z, k, Q, g, True)
    est = _fit_split(z, k, Q, a, np.ones(n), g, True, rng)
    l1, l2 = T.sig(est[2]), T.sig(est[3])
    assert abs(l1 - 0.8) < 0.02 and abs(l2 - 0.95) < 0.02, (l1, l2)
    print("split selftest ok", round(float(l1), 3), round(float(l2), 3))


if __name__ == "__main__":
    if sys.argv[1] == "split":
        split_power(int(sys.argv[2]), float(sys.argv[3]))
    elif sys.argv[1] == "split_selftest":
        split_selftest()
    elif sys.argv[1] == "apower":
        apower()
    elif sys.argv[1] == "dates":
        dates()
    else:
        power(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else None)
