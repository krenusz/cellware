"""CAND-A-P1: the C14 claim (state beyond type predicts relatives' lineage) on the DNA Typewriter embryo, TRAIN founder
clades only. Pre-registered 2026-11-27 (EXPERIMENT_LOG.md, "## CAND-A-P1"). Serves: C12 CLOSED.

    python experiments/canda1_state.py selftest             # toy tree and synthetic data only (local)
    python experiments/canda1_state.py s0                   # readout + structure; no UMAP column read
    python experiments/canda1_state.py smoke <model> <chunk> # synthetic coordinates on the real readout (Actions)
    python experiments/canda1_state.py combine              # smoke chunk JSONs -> canda1_smoke.json
    python experiments/canda1_state.py train                # real UMAP, train founders (second dispatch)

Train founder clades are chosen with partitions.train_only before any metadata row is parsed; metadata rows of
sealed founders are skipped by row index and never parsed. Outputs: counts, AUC summaries, rejection rates.
"""

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c17s1pl_tree as P  # noqa: E402

OUT = HERE / "results_cand"
DATASET = "dtt_founder"
TYPE_COLS = ("celltype", "celltype_from_which_major_trajectory")
LIN_COL = "major_trajectory"
UMAP = ("UMAP_1", "UMAP_2", "UMAP_3", "UMAP_2d_1", "UMAP_2d_2")
MISSING = {"", "nan", "na", "none", "unknown"}
K_REL, MAX_CLADE, L_SHARE, MIN_INF = 5, 300, 0.20, 3
MIN_CELLS, MIN_CLUST, MIN_CLASS_F = 200, 30, 5
MIN_FOUNDERS, MIN_A_YIELD, A_FOUNDERS, B_FOUNDERS = 20, 10, 19, 18
N_FOLD, FOLD_SALT = 5, "cellware-canda-fold-v1"
Z_CRIT, NULL_MAX, PLANT_MIN = 1.96, 0.08, 0.80
REPS = {"N1": 200, "N2": 200, "N3": 200, "P": 100}
CHUNK = 50
SEED = 20261127


# ---------------------------------------------------------------- loading
def subtree_end(parent):
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


def read_rows(names, leaf_train, cols, path=None):
    """Columns `cols` for train leaves only: pass 1 reads the ID column, pass 2 parses only train leaves' rows."""
    import pandas as pd

    want = {names[i] for i in np.flatnonzero(leaf_train)}
    path = path or (P.RAW / P.META)
    ids = pd.read_csv(path, sep="\t", usecols=[P.ID_COL])[P.ID_COL].astype(str)
    keep = set((np.flatnonzero(ids.isin(want).to_numpy()) + 1).tolist())
    df = pd.read_csv(path, sep="\t", usecols=[P.ID_COL, *cols], skiprows=lambda r: r != 0 and r not in keep)
    df[P.ID_COL] = df[P.ID_COL].astype(str)
    return df.set_index(P.ID_COL)


def load_train():
    from partitions import train_only  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    parent, blen, names, labelled = P.load()
    s = P.structure(parent, blen, labelled)
    fnd = P.founders(parent, s)
    is_f = np.zeros(len(parent), bool)
    is_f[fnd] = True
    f_of = founder_of(parent, is_f)
    units = pd.DataFrame({"unit": [f"F{f}" for f in fnd], "f": fnd})
    train_f = set(train_only(DATASET, units, unit_col="unit")["f"].tolist())
    leaf_train = s["is_leaf"] & np.isin(f_of, list(train_f))
    return {"parent": parent, "s": s, "names": names, "f_of": f_of, "train_f": train_f, "leaf_train": leaf_train,
            "end": subtree_end(parent)}


def clean(series):
    v = series.astype(str).str.strip()
    return v.where(~v.str.lower().isin(MISSING))


# ---------------------------------------------------------------- readout (C14 rule)
def readout(parent, leaves, end, leaf_nodes, tcode, lcode, n_lin):
    """leaf_nodes: preorder indices of labelled train leaves (sorted). Returns per evaluable cell:
    position in leaf_nodes, ancestor, relatives' lineage counts (different-type only)."""
    pos, anc, cnt = [], [], []
    for i, c in enumerate(leaf_nodes):
        tc, a = tcode[i], parent[c]
        while a >= 0 and leaves[a] <= MAX_CLADE:
            lo, hi = np.searchsorted(leaf_nodes, [a, end[a]])
            other = tcode[lo:hi] != tc
            if other.sum() >= K_REL:
                pos.append(i)
                anc.append(a)
                cnt.append(np.bincount(lcode[lo:hi][other], minlength=n_lin))
                break
            a = parent[a]
    return np.array(pos, int), np.array(anc, int), np.array(cnt, int).reshape(-1, n_lin)


def outcomes(pos, anc, cnt, tcode, n_type):
    """Per type: L1, L2 (each >= L_SHARE of different-type relatives); y per cell (ties / < MIN_INF dropped).
    Returns keep mask over evaluable cells, y, and per-type (L1, L2)."""
    y = np.full(len(pos), -1)
    lines = {}
    t_of = tcode[pos]
    for t in range(n_type):
        m = t_of == t
        if not m.any():
            continue
        tot = cnt[m].sum(0)
        if tot.sum() == 0:
            continue
        top = np.argsort(-tot, kind="stable")[:2]
        if len(top) < 2 or tot[top[1]] < L_SHARE * tot.sum() or tot[top[0]] < L_SHARE * tot.sum():
            continue
        lines[t] = (int(top[0]), int(top[1]))
        n1, n2 = cnt[m][:, top[0]], cnt[m][:, top[1]]
        ok = (n1 + n2 >= MIN_INF) & (n1 != n2)
        yy = np.where(ok, (n1 > n2).astype(int), -1)
        y[np.flatnonzero(m)] = yy
    return y >= 0, y, lines


def eligible_types(t_of, y, anc):
    el = {}
    for t in np.unique(t_of):
        m = t_of == t
        n = int(m.sum())
        c1 = len(set(anc[m & (y == 1)].tolist()))
        c0 = len(set(anc[m & (y == 0)].tolist()))
        el[int(t)] = {"cells": n, "clusters_y1": c1, "clusters_y0": c0,
                      "eligible": n >= MIN_CELLS and c1 >= MIN_CLUST and c0 >= MIN_CLUST}
    return el


def usable_founders(t_of, y, fnd, elig, anc):
    """Founder f is usable for type t if f holds >= MIN_CLASS_F distinct clusters (ancestors) in each outcome class
    of t (the outcome is constant per ancestor and type, so clusters, not cells, carry the information)."""
    out = {}
    for f in np.unique(fnd):
        ts = []
        for t in elig:
            if not elig[t]["eligible"]:
                continue
            m = (fnd == f) & (t_of == t)
            if (len(set(anc[m & (y == 1)].tolist())) >= MIN_CLASS_F
                    and len(set(anc[m & (y == 0)].tolist())) >= MIN_CLASS_F):
                ts.append(t)
        if ts:
            out[int(f)] = ts
    return out


def founder_table(t_of, y, fnd, anc, use):
    """Per usable founder (sorted by size, no IDs): cells and clusters per side over its usable types."""
    rows = []
    for f, ts in use.items():
        m = (fnd == f) & np.isin(t_of, ts)
        rows.append({"types": len(ts), "cells_y1": int((m & (y == 1)).sum()), "cells_y0": int((m & (y == 0)).sum()),
                     "clusters_y1": len(set(anc[m & (y == 1)].tolist())),
                     "clusters_y0": len(set(anc[m & (y == 0)].tolist()))})
    return sorted(rows, key=lambda r: -(r["cells_y1"] + r["cells_y0"]))


# ---------------------------------------------------------------- models and statistic
def fold_of(f):
    return int(hashlib.sha256(f"{FOLD_SALT}:F{f}".encode()).hexdigest(), 16) % N_FOLD


def auc(score, y):
    from scipy.stats import rankdata  # noqa: PLC0415

    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    r = rankdata(score)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def _fit_pred(Xtr, ytr, Xte):
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    m = LogisticRegression(C=1.0, max_iter=200)
    m.fit((Xtr - mu) / sd, ytr)
    return m.decision_function((Xte - mu) / sd)


def d_by_founder(D, Z, founders_used=None):
    """Cross-validated d_f per usable founder. D: dict with base (n x p), t (type), y, fnd, use (founder->types).
    Z: n x 5 state columns."""
    base, full = D["base"], np.hstack([D["base"], Z])
    folds = np.array([fold_of(f) for f in D["fnd"]])
    sb, sf = np.full(len(D["y"]), np.nan), np.full(len(D["y"]), np.nan)
    for t in D["types"]:
        mt = D["t"] == t
        for k in range(N_FOLD):
            tr, te = mt & (folds != k), mt & (folds == k)
            if not te.any() or len(np.unique(D["y"][tr])) < 2:
                continue
            sb[te] = _fit_pred(base[tr], D["y"][tr], base[te])
            sf[te] = _fit_pred(full[tr], D["y"][tr], full[te])
    out = {}
    for f, ts in D["use"].items():
        if founders_used is not None and f not in founders_used:
            continue
        num = den = 0.0
        for t in ts:
            m = (D["fnd"] == f) & (D["t"] == t)
            if np.isnan(sb[m]).any():
                continue
            num += m.sum() * (auc(sf[m], D["y"][m]) - auc(sb[m], D["y"][m]))
            den += m.sum()
        if den:
            out[f] = num / den
    return out, sb, sf


def ztest(vals):
    v = np.array(list(vals), float)
    if len(v) < 3 or v.std(ddof=1) == 0:
        return float(v.mean()) if len(v) else float("nan"), 0.0
    return float(v.mean()), float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))))


# ---------------------------------------------------------------- stages
def build(L, with_umap=False):
    s, names, parent = L["s"], L["names"], L["parent"]
    cols = [*TYPE_COLS, LIN_COL] + (list(UMAP) if with_umap else [])
    df = read_rows(names, L["leaf_train"], cols)
    levels = {c: int(clean(df[c]).nunique()) for c in (*TYPE_COLS, LIN_COL)}
    tcol = max(TYPE_COLS, key=lambda c: (levels[c], c == TYPE_COLS[1]))
    typ, lin = clean(df[tcol]), clean(df[LIN_COL])
    ok = typ.notna() & lin.notna()
    if with_umap:
        ok &= df[list(UMAP)].notna().all(1)
    have = set(df.index[ok])
    leaf_nodes = np.array([i for i in np.flatnonzero(L["leaf_train"]) if names[i] in have], int)
    tl = sorted(typ[ok].unique())
    ll = sorted(lin[ok].unique())
    ti, li = {v: k for k, v in enumerate(tl)}, {v: k for k, v in enumerate(ll)}
    ids = [names[i] for i in leaf_nodes]
    tcode = np.array([ti[typ[x]] for x in ids], int)
    lcode = np.array([li[lin[x]] for x in ids], int)
    pos, anc, cnt = readout(parent, s["leaves"], L["end"], leaf_nodes, tcode, lcode, len(ll))
    keep, y, lines = outcomes(pos, anc, cnt, tcode, len(tl))
    pos, anc, y = pos[keep], anc[keep], y[keep]
    t_of = tcode[pos]
    fnd = L["f_of"][leaf_nodes[pos]]
    elig = eligible_types(t_of, y, anc)
    use = usable_founders(t_of, y, fnd, elig, anc)
    types = [t for t in elig if elig[t]["eligible"]]
    m = np.isin(t_of, types)
    own = np.eye(len(ll))[lcode[pos]]
    base = np.column_stack([own, np.log(s["leaves"][anc]), s["depth"][anc], s["rdist"][anc]])
    D = {"base": base[m], "t": t_of[m], "y": y[m], "fnd": fnd[m], "anc": anc[m], "types": types, "use": use}
    other = clean(df[[c for c in TYPE_COLS if c != tcol][0]])
    D["t_other"] = np.array([str(other[ids[p]]) for p in pos[m]], dtype=object)
    if with_umap:
        D["Z"] = df.loc[[ids[p] for p in pos[m]], list(UMAP)].to_numpy(float)
        # CAND-A-P2 (boundary check): the neighbour pool is every evaluable cell, not only eligible types
        D["pool_t"], D["pool_Z"] = t_of, df.loc[[ids[p] for p in pos], list(UMAP)].to_numpy(float)
        D["pool_row"] = np.flatnonzero(m)
        D["pool_all_t"], D["pool_all_Z"] = tcode, df.loc[ids, list(UMAP)].to_numpy(float)  # all labelled train leaves
        D["pool_all_row"] = pos[m]
    info = {"type_column": tcol, "levels": levels, "type_levels_used": len(tl), "lineage_levels": len(ll),
            "labelled_train_leaves": int(len(leaf_nodes)), "evaluable_cells": int(len(y)),
            "types_with_lines": len(lines), "eligible_types": len(types),
            "per_type_by_code": {f"t{t:03d}": v for t, v in sorted(elig.items())},  # codes only, no label strings
            "usable_founder_table": founder_table(t_of, y, fnd, anc, use),
            "train_founders": len(L["train_f"]), "usable_founders": len(use),
            "projected_A_yield": len(use) / len(L["train_f"]) * A_FOUNDERS}
    return D, info


def s0_check(info):
    fails = []
    if info["usable_founders"] < MIN_FOUNDERS:
        fails.append(f"usable train founders {info['usable_founders']} < {MIN_FOUNDERS}")
    if info["projected_A_yield"] < MIN_A_YIELD:
        fails.append(f"projected A yield {info['projected_A_yield']:.1f} < {MIN_A_YIELD}")
    return fails


def write(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")


def stage_s0():
    D, info = build(load_train())
    fails = s0_check(info)
    info.update({"pass": not fails, "fails": fails, "evaluable_cells_in_eligible_types": int(len(D["y"]))})
    write("canda1_s0.json", info)
    print(json.dumps({k: info[k] for k in ("type_column", "levels", "eligible_types", "usable_founders",
                                           "projected_A_yield", "pass", "fails")}))


def synth_Z(D, model, rng):
    n = len(D["y"])
    Z = rng.normal(size=(n, len(UMAP)))
    if model == "N2":
        fs = np.unique(D["fnd"])
        my = {f: D["y"][D["fnd"] == f].mean() for f in fs}
        mv = np.array(list(my.values()))
        off = {f: np.r_[(my[f] - mv.mean()) / (mv.std() or 1.0), rng.normal(size=len(UMAP) - 1)] for f in fs}
        Z += np.array([off[f] for f in D["fnd"]])
    elif model == "N3":
        x = D["base"][:, -3]
        Z[:, 0] = (x - x.mean()) / (x.std() or 1.0) + 0.5 * rng.normal(size=n)
    elif model == "P":
        col = np.zeros(n)
        for t in D["types"]:
            m = D["t"] == t
            yy = D["y"][m].astype(float)
            col[m] = (yy - yy.mean()) / (yy.std() or 1.0)
        Z[:, 0] = 0.5 * col + rng.normal(size=n)
    return Z


_D = {}  # set before the fork so workers inherit it instead of receiving a pickled copy per replicate


def one_rep(args):
    model, seed, n_a = args
    D = _D["D"]
    rng = np.random.default_rng(seed)
    d, *_ = d_by_founder(D, synth_Z(D, model, rng))
    fs = sorted(d)
    sub = rng.choice(fs, size=min(n_a, len(fs)), replace=False)
    T, z = ztest(d[f] for f in sub)
    return {"T": T, "z": z, "n": int(len(sub))}


def stage_smoke(model, chunk):
    D, info = build(load_train())
    fails = s0_check(info)
    name = f"canda1_smoke_{model}_{chunk}.json"
    if fails:
        write(name, {"skipped": "S0 failed", "fails": fails})
        print("S0 failed:", fails)
        return
    n_a = int(round(info["projected_A_yield"]))
    mi = list(REPS).index(model)
    _D["D"] = D
    args = [(model, [SEED, mi, int(chunk), r], n_a) for r in range(CHUNK)]
    if os.name == "posix":
        import multiprocessing as mp  # noqa: PLC0415

        with mp.get_context("fork").Pool(4) as pool:
            reps = pool.map(one_rep, args)
    else:
        reps = [one_rep(a) for a in args]
    write(name, {"model": model, "chunk": int(chunk), "n_a": n_a, "reps": reps})
    print(model, chunk, "reject", np.mean([r["z"] > Z_CRIT for r in reps]))


def stage_combine():
    res = {}
    for model, need in REPS.items():
        reps = []
        for ch in range(need // CHUNK):
            p = OUT / f"canda1_smoke_{model}_{ch}.json"
            if not p.exists():
                raise SystemExit(f"missing {p.name}")
            j = json.loads(p.read_text(encoding="utf-8"))
            if "skipped" in j:
                raise SystemExit(f"{p.name}: {j['skipped']}")
            reps += j["reps"]
        res[model] = {"reject": float(np.mean([r["z"] > Z_CRIT for r in reps])), "reps": len(reps),
                      "mean_T": float(np.mean([r["T"] for r in reps]))}
    fails = [f"null {m} rejects {res[m]['reject']:.3f} > {NULL_MAX}" for m in ("N1", "N2", "N3")
             if res[m]["reject"] > NULL_MAX]
    if res["P"]["reject"] < PLANT_MIN:
        fails.append(f"planted rejects {res['P']['reject']:.3f} < {PLANT_MIN}")
    res.update({"pass": not fails, "fails": fails})
    write("canda1_smoke.json", res)
    print(json.dumps(res))


def stage_train():
    import power  # noqa: PLC0415

    s0 = json.loads((OUT / "canda1_s0.json").read_text(encoding="utf-8"))
    sm = json.loads((OUT / "canda1_smoke.json").read_text(encoding="utf-8"))
    if not (s0.get("pass") and sm.get("pass")):
        raise SystemExit("S0 and smoke must both pass first")
    D, info = build(load_train(), with_umap=True)
    # amendment 1 (report only): the readout scored here must be the one S0 and the smoke certified
    keys = ("evaluable_cells", "eligible_types", "usable_founders", "evaluable_cells_in_eligible_types")
    info["evaluable_cells_in_eligible_types"] = int(len(D["y"]))
    same = {k: (s0[k], info[k]) for k in keys}
    multi = sum(len(set(D["t_other"][D["t"] == t].tolist())) > 1 for t in D["types"])
    if any(a != b for a, b in same.values()):
        write("canda1_train.json", {"stopped": "readout differs from S0 (amendment 1)", "s0_vs_train": same})
        raise SystemExit(f"readout differs from S0: {same}")
    rng = np.random.default_rng(SEED + 1)
    d, sb, sf = d_by_founder(D, D["Z"])
    v = np.array(list(d.values()))
    T, z = ztest(v)
    boot = [rng.choice(v, size=len(v), replace=True).mean() for _ in range(2000)]
    sd = float(v.std(ddof=1))
    nA = int(round(info["projected_A_yield"]))
    nB = int(round(len(D["use"]) / info["train_founders"] * B_FOUNDERS))
    res = {"T": T, "z_train": z, "n_founders": len(v), "positives": int((v > 0).sum()),
           "d_f": {str(p): float(np.percentile(v, p)) for p in (5, 25, 50, 75, 95)},
           "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], "sd_d_f": sd,
           "base_auc_pooled": float(auc(sb[~np.isnan(sb)], D["y"][~np.isnan(sb)])),
           "power_half_T": {"A": power.mean_power(T / 2, sd, {"A": nA}) if T > 0 else 0.0,
                            "A+B": power.mean_power(T / 2, sd, {"AB": nA + nB}) if T > 0 else 0.0},
           "n_projected": {"A": nA, "A+B": nA + nB}, "exploratory": {}, "s0_vs_train": same,
           "eligible_types_with_several_other_label_levels": int(multi)}
    for name, ix in (("umap3d_only", [0, 1, 2]), ("umap2d_only", [3, 4])):
        de, *_ = d_by_founder(D, D["Z"][:, ix])
        res["exploratory"][name] = dict(zip(("T", "z"), ztest(de.values()), strict=True))
    write("canda1_train.json", res)
    print(json.dumps({k: res[k] for k in ("T", "ci95", "n_founders", "positives", "power_half_T")}))


# ---------------------------------------------------------------- selftest
def toy_tree():
    """Root -> 2 founders; each founder a balanced binary tree of 64 leaves. Preorder construction."""
    parent = [-1]
    def grow(p, depth):
        me = len(parent)
        parent.append(p)
        if depth:
            grow(me, depth - 1)
            grow(me, depth - 1)
    grow(0, 6)
    grow(0, 6)
    return np.array(parent)


def selftest():
    r = {}
    parent = toy_tree()
    n = len(parent)
    nchild = np.bincount(parent[parent >= 0], minlength=n)
    leaves = (nchild == 0).astype(int)
    for i in range(n - 1, 0, -1):
        leaves[parent[i]] += leaves[i]
    end = subtree_end(parent)
    leaf_nodes = np.flatnonzero(nchild == 0)
    rng = np.random.default_rng(0)
    # types alternate in blocks of 4 leaves; lineage = block parity of 8
    tcode = (np.arange(len(leaf_nodes)) // 4) % 3
    lcode = (np.arange(len(leaf_nodes)) // 8) % 2
    pos, anc, cnt = readout(parent, leaves, end, leaf_nodes, tcode, lcode, 2)
    r["readout_cells"] = int(len(pos))
    # every chosen ancestor has >= K_REL other-type leaves and <= MAX_CLADE leaves, and is the smallest such
    okk = True
    for p, a in zip(pos, anc, strict=True):
        lo, hi = np.searchsorted(leaf_nodes, [a, end[a]])
        okk &= (tcode[lo:hi] != tcode[p]).sum() >= K_REL and leaves[a] <= MAX_CLADE
        b = parent[leaf_nodes[p]]
        while b != a:
            lo, hi = np.searchsorted(leaf_nodes, [b, end[b]])
            okk &= (tcode[lo:hi] != tcode[p]).sum() < K_REL
            b = parent[b]
    r["readout_rule"] = bool(okk)
    # outcomes / eligibility / usable founders on the toy readout (thresholds lowered for the toy size)
    keep, yy, lines = outcomes(pos, anc, cnt, tcode, 3)
    r["outcome_lines_ok"] = all(a != b for a, b in lines.values())
    r["outcome_y_binary"] = bool(set(yy[keep].tolist()) <= {0, 1})
    global MIN_CELLS, MIN_CLUST, MIN_CLASS_F
    saved = (MIN_CELLS, MIN_CLUST, MIN_CLASS_F)
    MIN_CELLS, MIN_CLUST, MIN_CLASS_F = 2, 1, 1
    pk, ak, yk = pos[keep], anc[keep], yy[keep]
    toy_f = (leaf_nodes[pk] > leaf_nodes[len(leaf_nodes) // 2 - 1]).astype(int)
    el = eligible_types(tcode[pk], yk, ak)
    us = usable_founders(tcode[pk], yk, toy_f, el, ak)
    MIN_CELLS, MIN_CLUST, MIN_CLASS_F = saved
    hand = {}
    for f in (0, 1):
        ts = [t for t in el if el[t]["eligible"]
              and len(set(ak[(toy_f == f) & (tcode[pk] == t) & (yk == 1)].tolist())) >= 1
              and len(set(ak[(toy_f == f) & (tcode[pk] == t) & (yk == 0)].tolist())) >= 1]
        if ts:
            hand[f] = ts
    r["usable_matches_hand_count"] = us == hand
    # statistic on synthetic data: null near 0, planted positive
    nf, per = 30, 120
    fnd = np.repeat(np.arange(nf), per)
    t = np.tile(np.repeat([0, 1], per // 2), nf)
    y = rng.integers(0, 2, size=len(fnd))
    D = {"base": rng.normal(size=(len(fnd), 3)), "t": t, "y": y, "fnd": fnd, "types": [0, 1],
         "use": {f: [0, 1] for f in range(nf)}}
    dn, *_ = d_by_founder(D, synth_Z(D, "N1", rng))
    dp, *_ = d_by_founder(D, synth_Z(D, "P", rng))
    d2, *_ = d_by_founder(D, synth_Z(D, "N2", rng))
    r["null_T"], r["planted_T"], r["offset_T"] = ztest(dn.values())[0], ztest(dp.values())[0], ztest(d2.values())[0]
    r["planted_z"] = ztest(dp.values())[1]
    r["folds_balanced"] = sorted(Counter(fold_of(f) for f in range(93)).values())
    r["pass"] = bool(r["readout_rule"] and r["readout_cells"] > 0 and r["outcome_lines_ok"] and r["outcome_y_binary"]
                     and r["usable_matches_hand_count"] and abs(r["null_T"]) < 0.03
                     and r["planted_T"] > 0.05 and r["planted_z"] > Z_CRIT and r["offset_T"] < 0.03)
    return r


if __name__ == "__main__":
    a = sys.argv[1:] or ["selftest"]
    if a[0] == "selftest":
        print(json.dumps(selftest(), default=float))
    elif a[0] == "s0":
        stage_s0()
    elif a[0] == "smoke":
        stage_smoke(a[1], a[2])
    elif a[0] == "combine":
        stage_combine()
    elif a[0] == "train":
        stage_train()
    else:
        raise SystemExit(f"unknown stage {a[0]}")
