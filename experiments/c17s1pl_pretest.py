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


def power(variant, n_sets):
    label, cov = {"clock": ("major_trajectory", "date"), "edges_celltype": ("celltype", "tau")}[variant]
    S, info = struct(label)
    T.say(json.dumps(info))
    x = S[cov]
    S["z"] = (x - x.mean()) / x.std()
    w = T.weights(S["n1"], S["n2"])
    out = {"variant": variant, "label": label, "commitment_covariate": cov, "structure": info}
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
    T.dump(out, RES / f"c17s1pl_p_power_{variant}.json")


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
    out = {"url": url, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "same_topology": same}
    if same:
        us = P.structure(up, ub, lab)
        leaf = s["is_leaf"]
        fnd = P.founders(parent, s)
        is_f = np.zeros(len(parent), bool)
        is_f[fnd] = True
        f_of = F.founder_of(parent, is_f)
        units = pd.DataFrame({"unit": [f"F{f}" for f in fnd], "f": fnd})
        train_f = set(train_only("dtt_founder", units, unit_col="unit")["f"].tolist())
        rows = [r for r in P.pick_pairs(parent, s, P.CAP, True) if f_of[r[0]] in train_f]
        nd = np.array([r[0] for r in rows])
        k = np.array([r[3] for r in rows], float)
        dc, du = s["rdist"][nd], us["rdist"][nd]
        out.update({"n_train_pairs_structure": len(rows),
                    "uncon_leaf_rootdist_pct_0_50_100": np.percentile(us["rdist"][leaf], [0, 50, 100]).tolist(),
                    "corr_dates": float(np.corrcoef(dc, du)[0, 1]),
                    "spearman_dates": float(pd.Series(dc).corr(pd.Series(du), method="spearman")),
                    "abs_shift_days_pct_50_90_99": np.percentile(np.abs(dc - du), [50, 90, 99]).tolist(),
                    "date_pct_10_50_90": {"constrained": np.percentile(dc, [10, 50, 90]).tolist(),
                                          "unconstrained": np.percentile(du, [10, 50, 90]).tolist()},
                    "corr_k": {"constrained": float(np.corrcoef(dc, k)[0, 1]),
                               "unconstrained": float(np.corrcoef(du, k)[0, 1])}})
    T.say(json.dumps(out, indent=1))
    T.dump(out, RES / "c17s1pl_p_dates.json")


if __name__ == "__main__":
    if sys.argv[1] == "dates":
        dates()
    else:
        power(sys.argv[1], int(sys.argv[2]))
