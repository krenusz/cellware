"""Track B3h: is the sister sharing at EPI-Clone Dynamic CpGs a fate state beyond the clone
record, or record leakage? See the B3h pre-registration.

    uv run python experiments/track_b3h.py
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
from track_b3fg import load
from track_b5_camellia import auc
from track_b_epiclone import top_pcs
from track_b_epiclone_m13 import ERY, MULTI, MYEL

HERE = os.path.dirname(__file__)


def pairs(meth, types, md, rng):
    """B3g's cells and same-group pairs: (clone, group, i, j, sister mask)."""
    clone = md.LARRY.fillna("").to_numpy(dtype=str)
    ctype = md.CellType.astype(str).to_numpy()
    k = np.isin(ctype, list(MULTI)) & (clone != "")
    group = (
        (md.Experiment.astype(str) + "|" + md.ProcessingBatch.astype(str)).to_numpy() + "|" + ctype
    )
    clone, group, m = clone[k], group[k], meth[k]
    key = np.char.add(np.char.add(group.astype(str), "#"), clone)
    u, cnt = np.unique(key, return_counts=True)
    ok = np.isin(key, u[cnt >= 2]) | (rng.random(len(key)) < 0.3)
    clone, group, m = clone[ok], group[ok], m[ok]
    i, j = np.triu_indices(len(clone), 1)
    keep = group[i] == group[j]
    return clone, m, i[keep], j[keep]


def sim(m, cols, i, j):
    c = np.corrcoef(top_pcs(m[:, cols], 20))
    return c[i, j]


def stratified_auc(s_dyn, s_sta, sis, bins=10):
    edges = np.quantile(s_sta, np.linspace(0, 1, bins + 1)[1:-1])
    b = np.digitize(s_sta, edges)
    num = den = 0.0
    for k in range(bins):
        sel = b == k
        p, n = s_dyn[sel & sis], s_dyn[sel & ~sis]
        if len(p) and len(n):
            num += auc(p, n) * len(p)
            den += len(p)
    return num / den, b


def part_a(clone, m, i, j, types, rng):
    sis = clone[i] == clone[j]
    s_sta = sim(m, np.flatnonzero(types == "Static"), i, j)
    dyn = np.flatnonzero(types == "Dynamic")
    s_dyn = sim(m, dyn, i, j)
    a, b = stratified_auc(s_dyn, s_sta, sis)
    nulls = []
    for _ in range(100):
        # permute Dynamic similarity among pairs within each Static decile
        perm = s_dyn.copy()
        for k in range(10):
            sel = np.flatnonzero(b == k)
            perm[sel] = s_dyn[rng.permutation(sel)]
        nulls.append(stratified_auc(perm, s_sta, sis)[0])
    ids = np.unique(clone[i][sis])
    pos = {c: np.flatnonzero(sis & (clone[i] == c)) for c in ids}
    neg = np.flatnonzero(~sis)
    bs = []
    for _ in range(300):
        pick = np.concatenate([pos[c] for c in rng.choice(ids, len(ids))])
        idx = np.r_[pick, rng.choice(neg, min(len(neg), 100_000))]
        bs.append(stratified_auc(s_dyn[idx], s_sta[idx], sis[idx])[0])
    n = int((types == "Static").sum())
    d110 = [
        stratified_auc(sim(m, rng.choice(dyn, n, replace=False), i, j), s_sta, sis)[0]
        for _ in range(20)
    ]
    return {
        "pairs": int(len(sis)),
        "sister_pairs": int(sis.sum()),
        "auc": float(a),
        "null": float(np.mean(nulls)),
        "null_p97_5": float(np.percentile(nulls, 97.5)),
        "ci": np.percentile(bs, [2.5, 97.5]).tolist(),
        "dynamic_110_median": float(np.median(d110)),
        "dynamic_110_range": [float(min(d110)), float(max(d110))],
    }


def ridge_resid_cv(x, z, y, folds, lam=10.0):
    """CV ridge on x after removing (per training fold) its linear prediction from z."""
    pred = np.zeros_like(y)
    for f in range(folds.max() + 1):
        tr, te = folds != f, folds == f
        if z is not None:
            z1 = np.c_[z, np.ones(len(z))]
            w = np.linalg.lstsq(z1[tr], x[tr], rcond=None)[0]
            xr = x - z1 @ w
        else:
            xr = x
        mu, sd = xr[tr].mean(0), xr[tr].std(0) + 1e-9
        a, b = (xr[tr] - mu) / sd, (xr[te] - mu) / sd
        a1, b1 = np.c_[a, np.ones(len(a))], np.c_[b, np.ones(len(b))]
        w = np.linalg.solve(a1.T @ a1 + lam * np.eye(a1.shape[1]), a1.T @ y[tr])
        pred[te] = b1 @ w
    return pred


def fate_design(meth, md):
    clone = md.LARRY.fillna("").to_numpy(dtype=str)
    ctype = md.CellType.astype(str).to_numpy()
    multi = np.isin(ctype, list(MULTI))
    rows = []
    for c in np.unique(clone[clone != ""]):
        cells = clone == c
        my = (cells & np.isin(ctype, list(MYEL))).sum()
        er = (cells & np.isin(ctype, list(ERY))).sum()
        if (cells & multi).sum() >= 1 and my + er >= 5:
            rows.append((c, my / (my + er)))

    def feats(cols):
        pcs = top_pcs(meth[:, cols], 20)
        return np.stack([pcs[(clone == c) & multi].mean(0) for c, _ in rows])

    y = np.array([t[1] for t in rows])
    return {"y": y, "folds": np.random.default_rng(0).integers(0, 5, len(rows)), "feats": feats}


def part_b(meth, types, md, rng):
    d = fate_design(meth, md)
    y, folds, feats = d["y"], d["folds"], d["feats"]
    sta, dyn = feats(types == "Static"), feats(types == "Dynamic")
    preds = {
        "dynamic_given_static": ridge_resid_cv(dyn, sta, y, folds),
        "static_given_dynamic": ridge_resid_cv(sta, dyn, y, folds),
        "both": ridge_resid_cv(np.c_[sta, dyn], None, y, folds),
    }
    idx = rng.integers(0, len(y), (3000, len(y)))
    out = {"clones": len(y)}
    for k, p in preds.items():
        bs = [np.corrcoef(p[t], y[t])[0, 1] for t in idx]
        out[k] = {
            "cv_r": float(np.corrcoef(p, y)[0, 1]),
            "ci": np.percentile(bs, [2.5, 97.5]).tolist(),
        }
    return out


def benchmark(meth, types, md, rng, clone, m, i, j, splits=20):
    """Record-only benchmark: Static half A | half B vs 55 Dynamic CpGs | the same half B."""
    sis = clone[i] == clone[j]
    sta, dyn = np.flatnonzero(types == "Static"), np.flatnonzero(types == "Dynamic")
    fate = fate_design(meth, md)
    res = {"a_static_half": [], "a_dynamic55": [], "b_static_half": [], "b_dynamic55": []}
    for _ in range(splits):
        perm = rng.permutation(sta)
        ha, hb = perm[: len(sta) // 2], perm[len(sta) // 2 :]
        d55 = rng.choice(dyn, len(ha), replace=False)
        s_b = sim(m, hb, i, j)
        res["a_static_half"].append(stratified_auc(sim(m, ha, i, j), s_b, sis, 50)[0])
        res["a_dynamic55"].append(stratified_auc(sim(m, d55, i, j), s_b, sis, 50)[0])
        fb = fate["feats"](hb)
        for name, cols in (("b_static_half", ha), ("b_dynamic55", d55)):
            p = ridge_resid_cv(fate["feats"](cols), fb, fate["y"], fate["folds"])
            res[name].append(float(np.corrcoef(p, fate["y"])[0, 1]))
    return {
        k: {
            "median": float(np.median(v)),
            "p5": float(np.percentile(v, 5)),
            "p95": float(np.percentile(v, 95)),
        }
        for k, v in res.items()
    }


def main():
    warnings.filterwarnings("ignore")
    meth, types, md = load()
    rng = np.random.default_rng(0)
    out = {"b": part_b(meth, types, md, rng)}
    print(json.dumps(out, indent=1), flush=True)
    clone, m, i, j = pairs(meth, types, md, rng)
    out["a"] = part_a(clone, m, i, j, types, rng)
    out["benchmark"] = benchmark(meth, types, md, rng, clone, m, i, j)
    json.dump(out, open(os.path.join(HERE, "results_track_b", "b3h.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
