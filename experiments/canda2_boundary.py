"""CAND-A-P2: train-only boundary check for CAND-A-P1 (amendment 3), mirroring C14-S5 (b). Pre-registered 2026-11-27
(EXPERIMENT_LOG.md, "## CAND-A-P2"). Serves: C12 CLOSED.

    python experiments/canda2_boundary.py selftest   # toy data only (local)
    python experiments/canda2_boundary.py run        # train founders, real UMAP (Actions)

Order inside `run` (no real T is computed before each gate passes):
  1. amendment-1 readout identity vs S0; 2. kNN type purity (raw UMAP, k = 15, pool = evaluable cells, self excluded
  by index); 3. usable founders on core cells (>= 20, else INCONCLUSIVE, counts only); 4. planted smoke on synthetic
  columns with the real purity (edge-only -> retained < 0.25; everywhere -> retained >= 0.5; else STOP);
  5. full-cell rescore must reproduce CAND-A-P1's T; 6. primary: retained_core = T_core / T_train; 7. exploratory.
Outputs: results_cand/canda2_boundary.json (codes and counts only).
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import canda1_state as C  # noqa: E402

K_NN, CORE, EDGE = 15, 0.9, 0.6
RETAIN_PASS, RETAIN_BOUNDARY = 0.5, 0.25
PLANT = 1.0
SEED = C.SEED + 2
REPRO_TOL = 1e-6


def purity(Z, t, k=K_NN):
    """Share of each cell's k nearest neighbours (Euclidean, raw columns) with the same type; self excluded by index."""
    from sklearn.neighbors import NearestNeighbors  # noqa: PLC0415

    n = len(t)
    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z)
    drop = idx == np.arange(n)[:, None]
    drop[~drop.any(1), -1] = True  # self not returned (duplicate coordinates): drop the farthest instead
    nb = idx[~drop].reshape(n, k)
    return (t[nb] == t[:, None]).mean(1)


def floor_ok(D, m):
    a1, a0 = D["anc"][m & (D["y"] == 1)], D["anc"][m & (D["y"] == 0)]
    return len(np.unique(a1)) >= C.MIN_CLASS_F and len(np.unique(a0)) >= C.MIN_CLASS_F


def d_on(D, sb, sf, mask, pairs=None):
    """Per-founder cell-weighted delta AUC on cells in `mask`, re-applying the >= MIN_CLASS_F clusters per class floor.
    `pairs` (optional set of (founder, type)) restricts which founder x type pairs count. Returns d, surviving pairs."""
    out, kept = {}, set()
    for f, ts in D["use"].items():
        num = den = 0.0
        for t in ts:
            if pairs is not None and (f, t) not in pairs:
                continue
            m = (D["fnd"] == f) & (D["t"] == t) & mask
            if not floor_ok(D, m) or np.isnan(sb[m]).any():
                continue
            num += m.sum() * (C.auc(sf[m], D["y"][m]) - C.auc(sb[m], D["y"][m]))
            den += m.sum()
            kept.add((f, t))
        if den:
            out[f] = num / den
    return out, kept


def usable_on(D, mask):
    """Founders with >= 1 eligible type meeting the cluster floor on masked cells (no model, no state column)."""
    return sum(any(floor_ok(D, (D["fnd"] == f) & (D["t"] == t) & mask) for t in ts) for f, ts in D["use"].items())


def planted(D, where, rng):
    """Synthetic state columns: column 1 = PLANT x within-type standardised y on cells in `where`, plus noise."""
    Z = rng.normal(size=(len(D["y"]), len(C.UMAP)))
    for t in D["types"]:
        m = D["t"] == t
        yy = D["y"][m].astype(float)
        Z[m, 0] += PLANT * np.where(where[m], (yy - yy.mean()) / (yy.std() or 1.0), 0.0)
    return Z


def retained(D, Z, core):
    d_all, sb, sf = C.d_by_founder(D, Z)
    d_core, _ = d_on(D, sb, sf, core)
    T_all, T_core = C.ztest(d_all.values())[0], C.ztest(d_core.values())[0]
    return T_all, T_core, (T_core / T_all if T_all else float("nan"))


def summary(d, rng):
    v = np.array(list(d.values()), float)
    T, z = C.ztest(v)
    boot = [rng.choice(v, size=len(v), replace=True).mean() for _ in range(2000)] if len(v) else [np.nan]
    return {"T": T, "z": z, "n_founders": int(len(v)), "positives": int((v > 0).sum()),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}


def subset(D, keep, pairs):
    """Rows in `keep`, with the usable map restricted to surviving founder x type pairs (exploratory refit)."""
    n = len(D["y"])
    S = {k: (v[keep] if isinstance(v, np.ndarray) and v.shape[:1] == (n,) else v)
         for k, v in D.items() if not k.startswith("pool_")}
    use = {f: [t for t in ts if (f, t) in pairs] for f, ts in D["use"].items()}
    S["use"] = {f: ts for f, ts in use.items() if ts}
    return S


def stage_run():
    s0 = json.loads((C.OUT / "canda1_s0.json").read_text(encoding="utf-8"))
    tr = json.loads((C.OUT / "canda1_train.json").read_text(encoding="utf-8"))
    if not (s0.get("pass") and "T" in tr):
        raise SystemExit("needs a passing canda1_s0.json and canda1_train.json")
    T_train = float(tr["T"])
    res = {"T_train": T_train, "k": K_NN, "core": CORE, "edge": EDGE}

    def done(outcome, note):
        res.update({"outcome": outcome, "note": note})
        C.write("canda2_boundary.json", res)
        print(json.dumps({k: res.get(k) for k in ("outcome", "note", "core_share", "usable_core", "primary")}))

    # 1. readout identity (amendment 1)
    D, info = C.build(C.load_train(), with_umap=True)
    info["evaluable_cells_in_eligible_types"] = int(len(D["y"]))
    keys = ("evaluable_cells", "eligible_types", "usable_founders", "evaluable_cells_in_eligible_types")
    same = {k: (s0[k], info[k]) for k in keys}
    res["s0_vs_now"] = same
    if any(a != b for a, b in same.values()):
        return done("stopped", "readout differs from S0")
    # 2. purity on the pool of evaluable cells, raw UMAP columns
    pur_pool = purity(D["pool_Z"], D["pool_t"])
    pur = pur_pool[D["pool_row"]]
    core, edge = pur >= CORE, pur < EDGE
    res.update({"pool_cells": int(len(pur_pool)), "purity_median_pool": float(np.median(pur_pool)),
                "core_share": float(core.mean()), "edge_share": float(edge.mean()),
                "core_share_by_type_code": {f"t{t:03d}": float(core[D["t"] == t].mean()) for t in D["types"]}})
    # 3. yield gate on core cells (no model)
    res["usable_core"] = usable_on(D, core)
    if res["usable_core"] < C.MIN_FOUNDERS:
        return done("inconclusive", f"usable founders on core cells {res['usable_core']} < {C.MIN_FOUNDERS}")
    # 4. planted smoke (synthetic columns, real purity)
    rng = np.random.default_rng(SEED)
    sm = {}
    for name, where in (("edge_only", edge), ("everywhere", np.ones_like(edge))):
        T_all, T_core, r = retained(D, planted(D, where, rng), core)
        sm[name] = {"T_all": T_all, "T_core": T_core, "retained_core": r}
    res["smoke"] = sm
    ok_edge = sm["edge_only"]["T_all"] > 0 and sm["edge_only"]["retained_core"] < RETAIN_BOUNDARY
    ok_all = sm["everywhere"]["retained_core"] >= RETAIN_PASS
    if not (ok_edge and ok_all):
        return done("stopped", "planted smoke failed: the check cannot separate edge from core signal")
    # 5. reproduce CAND-A-P1's full-cell T
    d_all, sb, sf = C.d_by_founder(D, D["Z"])
    T_full = C.ztest(d_all.values())[0]
    res["T_full_recomputed"] = T_full
    if abs(T_full - T_train) > REPRO_TOL:
        return done("stopped", f"full-cell T {T_full} does not reproduce the train T {T_train}")
    # 6. primary: rescore on core cells (count gate before any T)
    d_core, pairs = d_on(D, sb, sf, core)
    if len(d_core) < C.MIN_FOUNDERS:
        res["scored_core_founders"] = len(d_core)
        return done("inconclusive", f"scored founders on core cells {len(d_core)} < {C.MIN_FOUNDERS}")
    prim = summary(d_core, np.random.default_rng(SEED + 1))
    prim["retained_core"] = prim["T"] / T_train
    res["primary"] = prim
    r = prim["retained_core"]
    if r >= RETAIN_PASS and prim["T"] > 0:
        outcome = "pass"
    else:
        outcome = "boundary_driven" if r < RETAIN_BOUNDARY else "mixed"
    done(outcome, "pass needs retained_core >= 0.5 with T_core > 0; only pass allows a look proposal")
    # 7. exploratory (no rule; cannot change the outcome); the primary is already written
    try:
        res["exploratory"] = exploratory(D, sb, sf, pur, pairs)
    except Exception as e:  # noqa: BLE001
        res["exploratory"] = {"error": repr(e)}
    done(outcome, res["note"])


def exploratory(D, sb, sf, pur, pairs):
    core, edge = pur >= CORE, pur < EDGE
    ex = {"full_cells_on_surviving_pairs": summary(d_on(D, sb, sf, np.ones_like(core), pairs=pairs)[0],
                                                   np.random.default_rng(SEED + 3)),
          "edge_cells": summary(d_on(D, sb, sf, edge)[0], np.random.default_rng(SEED + 4))}
    for cut in (0.8, 1.0):
        ex[f"core_cut_{cut}"] = summary(d_on(D, sb, sf, pur >= cut)[0], np.random.default_rng(SEED + 5))
    zs = (D["pool_Z"] - D["pool_Z"].mean(0)) / D["pool_Z"].std(0)
    pz = purity(zs, D["pool_t"])[D["pool_row"]]
    ex["core_zscored_umap"] = summary(d_on(D, sb, sf, pz >= CORE)[0], np.random.default_rng(SEED + 6))
    pa = purity(D["pool_all_Z"], D["pool_all_t"])[D["pool_all_row"]]
    ex["all_labelled_pool"] = {"core_share": float((pa >= CORE).mean()),
                               **summary(d_on(D, sb, sf, pa >= CORE)[0], np.random.default_rng(SEED + 8))}
    S = subset(D, core, pairs)
    ex["refit_on_core"] = summary(C.d_by_founder(S, S["Z"])[0], np.random.default_rng(SEED + 7))
    return ex


# ---------------------------------------------------------------- selftest
def selftest():
    r = {}
    rng = np.random.default_rng(1)
    # purity: two separated blobs of two types, plus 20 exact duplicate points
    Z = np.vstack([rng.normal(0, 0.1, (40, 2)), rng.normal(5, 0.1, (40, 2)), np.zeros((20, 2))])
    t = np.r_[np.zeros(40, int), np.ones(40, int), np.zeros(20, int)]
    p = purity(Z, t, k=5)
    r["purity_separated"] = bool(p.min() == 1.0)
    t2 = t.copy()
    t2[0] = 1  # a mislabelled cell inside blob 0 has purity 0 against its label
    r["purity_mislabel"] = float(purity(Z, t2, k=5)[0])
    # statistic pieces on synthetic founders
    nf, per = 30, 160
    fnd = np.repeat(np.arange(nf), per)
    tt = np.tile(np.repeat([0, 1], per // 2), nf)
    y = rng.integers(0, 2, size=len(fnd))
    D = {"base": rng.normal(size=(len(fnd), 3)), "t": tt, "y": y, "fnd": fnd, "anc": np.arange(len(fnd)),
         "types": [0, 1], "use": {f: [0, 1] for f in range(nf)}}
    core = rng.random(len(fnd)) < 0.75
    Zp = planted(D, np.ones(len(fnd), bool), rng)
    d_all, sb, sf = C.d_by_founder(D, Zp)
    d_same, pairs = d_on(D, sb, sf, np.ones(len(fnd), bool))
    r["all_mask_reproduces"] = bool(max(abs(d_all[f] - d_same[f]) for f in d_all) < 1e-12)
    r["retained_everywhere"] = retained(D, Zp, core)[2]
    r["retained_edge_only"] = retained(D, planted(D, ~core, rng), core)[2]
    r["usable_core"] = usable_on(D, core)
    S = subset(D, core, pairs)
    r["refit_rows"] = int(len(S["y"]))
    # exploratory block end to end on toy pools (core flag reused as the toy purity)
    n = len(fnd)
    ta = np.r_[tt, tt[:50]]
    sep = rng.normal(size=(n + 50, 5)) + 10.0 * ta[:, None]  # type-separated toy coordinates: purity 1
    D.update({"Z": Zp, "pool_t": tt, "pool_Z": sep[:n], "pool_row": np.arange(n),
              "pool_all_t": ta, "pool_all_Z": sep, "pool_all_row": np.arange(n)})
    ex = exploratory(D, sb, sf, np.where(core, 1.0, 0.3), pairs)
    r["exploratory_keys"] = sorted(ex)
    r["exploratory_ok"] = len(ex) == 7 and all(np.isfinite(v["T"]) for v in ex.values())
    r["pass"] = bool(r["purity_separated"] and r["purity_mislabel"] == 0.0 and r["all_mask_reproduces"]
                     and r["retained_everywhere"] >= RETAIN_PASS and r["retained_edge_only"] < RETAIN_BOUNDARY
                     and r["usable_core"] == nf and r["refit_rows"] == int(core.sum()) and r["exploratory_ok"])
    return r


if __name__ == "__main__":
    a = sys.argv[1:] or ["selftest"]
    if a[0] == "selftest":
        print(json.dumps(selftest(), default=float))
    elif a[0] == "run":
        stage_run()
    else:
        raise SystemExit(f"unknown stage {a[0]}")
