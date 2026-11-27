"""CAND-A-L1: union look on dtt_founder A + B (CAND-A-P1's look branch, opened by CAND-A-P2 PASS). Pre-registered
2026-11-27 (EXPERIMENT_LOG.md, "## CAND-A-L1"). Serves: C12 CLOSED.

    python experiments/canda3_look.py selftest   # toy tree only (local)
    python experiments/canda3_look.py fit        # TRAIN only (Actions): frozen model, train dry run, selection file
    test_oracle look --test dtt_canda --exp CAND-A-L1 --partition AB   # calls counts_AB / test_AB (needs Bence's OK)

One function, `evaluate`, scores any leaf set under a frozen model file; the train dry run and the sealed test both
use it. Types and lineages are stored as salted sha256 codes, never as label strings.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import canda1_state as C  # noqa: E402
import canda2_boundary as B  # noqa: E402

DS = C.DATASET
EXP, TEST = "CAND-A-L1", "dtt_canda"
HSALT = "cellware-canda-code-v1"
MODEL = C.OUT / "canda3_model.json"
SEL = HERE / "selection" / f"{EXP}.json"
DRY_FOLD, TOL = 0, 1e-6
MIN_USABLE = 20
POWER_SD_SOURCE = "train founder-level delta AUC sd (CAND-A-P1 train stage)"
COLS = [*C.TYPE_COLS, C.LIN_COL, *C.UMAP]


def code(label):
    return hashlib.sha256(f"{HSALT}:{label}".encode()).hexdigest()[:16]


# ---------------------------------------------------------------- readout on any leaf set
def frame(L, leaf_sel, df, tcol, spec=None):
    """CAND-A-P1 readout on the leaves in `leaf_sel`. Types are coded from these leaves' own labels (an unseen label
    is its own type). Without `spec`, derive the frozen spec from these leaves (train). Returns D, spec, info."""
    s, names, parent = L["s"], L["names"], L["parent"]
    typ, lin = C.clean(df[tcol]), C.clean(df[C.LIN_COL])
    ok = typ.notna() & lin.notna() & df[list(C.UMAP)].notna().all(axis=1)
    have = set(df.index[ok])
    leaf_nodes = np.array([i for i in np.flatnonzero(leaf_sel) if names[i] in have], int)
    ids = [names[i] for i in leaf_nodes]
    tl, ll = sorted(typ[ok].unique()), sorted(lin[ok].unique())
    ti, li = {v: k for k, v in enumerate(tl)}, {v: k for k, v in enumerate(ll)}
    tcode = np.array([ti[typ[x]] for x in ids], int)
    lcode = np.array([li[lin[x]] for x in ids], int)
    thash, lhash = [code(v) for v in tl], [code(v) for v in ll]
    pos, anc, cnt = C.readout(parent, s["leaves"], L["end"], leaf_nodes, tcode, lcode, len(ll))
    if spec is None:
        keep, y0, lines = C.outcomes(pos, anc, cnt, tcode, len(tl))
        el = C.eligible_types(tcode[pos][keep], y0[keep], anc[keep])
        spec = {"types": [thash[t] for t in sorted(el) if el[t]["eligible"]],
                "lines": {thash[t]: [lhash[a], lhash[b]] for t, (a, b) in sorted(lines.items())},
                "lineages": lhash, "all_types": thash, "type_column": tcol}
    lidx = {h: k for k, h in enumerate(lhash)}
    tpos = tcode[pos]
    y = np.full(len(pos), -1)
    for tc in np.unique(tpos):
        h = thash[tc]
        if h not in spec["lines"]:
            continue
        m = np.flatnonzero(tpos == tc)
        a, b = spec["lines"][h]
        n1 = cnt[m, lidx[a]] if a in lidx else np.zeros(len(m), int)
        n2 = cnt[m, lidx[b]] if b in lidx else np.zeros(len(m), int)
        y[m] = np.where((n1 + n2 >= C.MIN_INF) & (n1 != n2), (n1 > n2).astype(int), -1)
    ev = y >= 0
    tix = {h: k for k, h in enumerate(spec["types"])}
    t_all = np.array([tix.get(thash[t], -1) for t in tpos], int)
    m = ev & (t_all >= 0)
    sidx = {h: k for k, h in enumerate(spec["lineages"])}
    own = np.zeros((len(pos), len(spec["lineages"])))
    for r, lc in enumerate(lcode[pos]):
        if lhash[lc] in sidx:  # a lineage unseen in train stays all zeros
            own[r, sidx[lhash[lc]]] = 1.0
    nodes = anc
    base = np.column_stack([own, np.log(s["leaves"][nodes]), s["depth"][nodes], s["rdist"][nodes]])
    fnd = L["f_of"][leaf_nodes[pos]]
    Zall = df.loc[[ids[p] for p in pos], list(C.UMAP)].to_numpy(float)
    D = {"base": base[m], "Z": Zall[m], "t": t_all[m], "y": y[m], "fnd": fnd[m], "anc": anc[m],
         "types": list(range(len(spec["types"])))}
    D["use"] = C.usable_founders(D["t"], D["y"], D["fnd"], {k: {"eligible": True} for k in D["types"]}, D["anc"])
    D["pool_t"], D["pool_Z"] = tpos[ev], Zall[ev]
    D["pool_row"] = np.flatnonzero(m[ev])
    info = {"evaluable_cells": int(ev.sum()), "eligible_types": len(set(D["t"].tolist())),
            "usable_founders": len(D["use"]), "evaluable_cells_in_eligible_types": int(m.sum()),
            "type_levels_here": len(tl), "type_levels_unseen_in_train": len(set(thash) - set(spec["all_types"])),
            "lineage_levels_unseen_in_train": sum(h not in sidx for h in lhash)}
    return D, spec, info


# ---------------------------------------------------------------- frozen models
def lr(X, y):
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    m = LogisticRegression(C=1.0, max_iter=200).fit((X - mu) / sd, y)
    return {"mu": mu.tolist(), "sd": sd.tolist(), "coef": m.coef_[0].tolist(), "b": float(m.intercept_[0])}


def dec(p, X):
    return ((X - np.array(p["mu"])) / np.array(p["sd"])) @ np.array(p["coef"]) + p["b"]


def fit_models(D, spec, rows):
    full = np.hstack([D["base"], D["Z"]])
    out = {}
    for k, h in enumerate(spec["types"]):
        m = rows & (D["t"] == k)
        if len(np.unique(D["y"][m])) < 2:
            continue
        out[h] = {"base": lr(D["base"][m], D["y"][m]), "state": lr(full[m], D["y"][m])}
    return {**spec, "models": out}


def evaluate(L, leaf_sel, df, model):
    """Score the leaves in `leaf_sel` under a frozen model. Returns the primary result, D, sb, sf."""
    D, _, info = frame(L, leaf_sel, df, model["type_column"], spec=model)
    n = len(D["y"])
    sb, sf = np.full(n, np.nan), np.full(n, np.nan)
    full = np.hstack([D["base"], D["Z"]])
    for k, h in enumerate(model["types"]):
        mm = model["models"].get(h)
        r = D["t"] == k
        if mm is None or not r.any():
            continue
        sb[r], sf[r] = dec(mm["base"], D["base"][r]), dec(mm["state"], full[r])
    d, _ = B.d_on(D, sb, sf, np.ones(n, bool))
    return {"d": d, "info": info}, D, sb, sf


def primary(d, train_sd):
    v = np.array([d[f] for f in sorted(d)], float)
    T, z = C.ztest(v)
    sd = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
    sk = max(train_sd, sd) if len(v) > 1 else float("nan")
    return {"n_founders": int(len(v)), "T": T, "sd_sample": sd, "z": z, "confirmed": bool(z > C.Z_CRIT),
            "positives": int((v > 0).sum()), "z_known_sd": float(T / (sk / np.sqrt(len(v)))) if len(v) > 1 else 0.0,
            "d_f_sorted": [round(float(x), 4) for x in sorted(v)]}


# ---------------------------------------------------------------- train stage
def stage_fit():
    s0 = json.loads((C.OUT / "canda1_s0.json").read_text(encoding="utf-8"))
    tr = json.loads((C.OUT / "canda1_train.json").read_text(encoding="utf-8"))
    p2 = json.loads((C.OUT / "canda2_boundary.json").read_text(encoding="utf-8"))
    if not (s0.get("pass") and "T" in tr and p2.get("outcome") == "pass"):
        raise SystemExit("needs passing S0, the CAND-A-P1 train result and a CAND-A-P2 pass")
    out = {}

    def stop(why):
        out["stopped"] = why
        C.write("canda3_fit.json", out)
        raise SystemExit(why)

    L = C.load_train()
    df = C.read_rows(L["names"], L["leaf_train"], COLS)
    D, spec, info = frame(L, L["leaf_train"], df, s0["type_column"])
    keys = ("evaluable_cells", "eligible_types", "usable_founders", "evaluable_cells_in_eligible_types")
    out["s0_vs_fit"] = {k: (s0[k], info[k]) for k in keys}
    if any(a != b for a, b in out["s0_vs_fit"].values()):
        stop("readout differs from S0")
    # dry run of the oracle path on train fold DRY_FOLD
    folds = np.array([C.fold_of(f) for f in D["fnd"]])
    dry = json.loads(json.dumps(fit_models(D, spec, folds != DRY_FOLD)))  # through the file format
    held = [f for f in L["train_f"] if C.fold_of(f) == DRY_FOLD]
    sel = L["leaf_train"] & np.isin(L["f_of"], held)
    df0 = C.read_rows(L["names"], sel, COLS)  # read exactly as test_AB reads its rows
    res0, D0, sb0, sf0 = evaluate(L, sel, df0, dry)
    got = res0["d"]
    fin = json.loads(json.dumps(finish(D0, sb0, sf0, res0, tr["sd_d_f"], 1)))  # plain dumps, as test_oracle does
    Dc, _ = C.build(L, with_umap=True)
    ref = {f: v for f, v in C.d_by_founder(Dc, Dc["Z"])[0].items() if C.fold_of(f) == DRY_FOLD}
    diff = max((abs(got[f] - ref[f]) for f in ref), default=float("inf")) if set(got) == set(ref) else float("inf")
    t_ok = abs(fin.get("primary", {}).get("T", np.inf) - float(np.mean(list(ref.values())))) <= TOL
    out["dry_run"] = {"fold": DRY_FOLD, "founders": len(ref), "founders_match": set(got) == set(ref),
                      "max_abs_diff": diff, "finish_T_matches": bool(t_ok),
                      "finish_secondary_ok": "error" not in fin.get("secondary_core", {"error": 1})}
    if not (diff <= TOL and t_ok and out["dry_run"]["finish_secondary_ok"]):
        stop("dry run of the oracle path does not reproduce CAND-A-P1's cross-validated d_f")
    # final model on all train cells
    model = fit_models(D, spec, np.ones(len(D["y"]), bool))
    txt = json.dumps(model, indent=0)
    MODEL.write_text(txt, encoding="utf-8")
    sha = hashlib.sha256(txt.encode("utf-8")).hexdigest()
    el = [v for v in s0["per_type_by_code"].values() if v["eligible"]]
    ex = tr["exploratory"]
    sel_json = {
        "test": TEST, "higher_is_better": True,
        "candidates": {"state_umap5": tr["T"], "umap3d_only": ex["umap3d_only"]["T"],
                       "umap2d_only": ex["umap2d_only"]["T"], "base_only": 0.0},
        "chosen": "state_umap5",
        "note": "candidates = CAND-A-P1 train founder-grouped CV T; primary = mean of A+B founder d_f, sample-sd z > "
                "1.96 one-sided; adequacy >= 20 usable founders; CAND-A-P2 boundary check PASS (retained 0.887)",
        "model_sha256": sha,
        "train": {**info, "models": len(model["models"]), "dry_run": out["dry_run"]},
        "power": {"effect": tr["T"] / 2, "sd": tr["sd_d_f"], "alpha": 0.025, "one_sided": True, "min_power": 0.8,
                  "sources": {"effect": "CAND-A-P1 train T x 0.5", "sd": POWER_SD_SOURCE}},
        "readout_gate": {"pass": True, "sides": {"y1_clusters": sum(v["clusters_y1"] for v in el),
                                                 "y0_clusters": sum(v["clusters_y0"] for v in el)}},
        "exposure": "train labels and UMAP (CAND-A-P1, -P2); source preprint headings; README; A and B never read",
    }
    SEL.parent.mkdir(exist_ok=True)
    SEL.write_text(json.dumps(sel_json, indent=1), encoding="utf-8")
    # the Actions artifact keeps experiments/results_*/ only: an identical copy, moved to selection/ on collection
    (C.OUT / "canda3_selection.json").write_text(json.dumps(sel_json, indent=1), encoding="utf-8")
    out.update({"model_sha256": sha, "models": len(model["models"]), "selection": "written"})
    C.write("canda3_fit.json", out)
    print(json.dumps(out))


# ---------------------------------------------------------------- oracle side (sealed; called only via test_oracle)
def counts_AB(part="AB"):
    """Structure only: founder clades in the union of the sealed letters, as ONE group (power.mean_power)."""
    import partitions  # noqa: PLC0415

    fixed = partitions._fixed_from_file(DS)
    return {part: sum(1 for u in fixed if partitions.partition_of(DS, u) in part)}


def test_AB(part="AB"):
    import pandas as pd  # noqa: PLC0415
    import partitions  # noqa: PLC0415

    mtxt = MODEL.read_text(encoding="utf-8")
    assert hashlib.sha256(mtxt.encode("utf-8")).hexdigest() == json.loads(SEL.read_text())["model_sha256"], \
        "frozen model changed"
    model = json.loads(mtxt)
    train_sd = json.loads(SEL.read_text())["power"]["sd"]
    L = C.load_train()
    fnd = C.P.founders(L["parent"], L["s"])
    units = pd.DataFrame({"unit": [f"F{f}" for f in fnd], "f": fnd})
    sealed_f = set()
    for letter in part:
        sealed_f |= set(partitions.sealed_rows(DS, units, "unit", letter)["f"].tolist())
    leaf_sel = L["s"]["is_leaf"] & np.isin(L["f_of"], list(sealed_f))
    out = {"partition": part, "founders_in_partition": len(sealed_f)}
    try:  # after the sealed read nothing may raise: the oracle must always ledger the spend
        df = C.read_rows(L["names"], leaf_sel, COLS)
        res, D, sb, sf = evaluate(L, leaf_sel, df, model)
        out.update(finish(D, sb, sf, res, train_sd, MIN_USABLE))
    except Exception as e:  # noqa: BLE001
        out.update({"error": repr(e), "confirmed": False})
    return out


def finish(D, sb, sf, res, train_sd, min_usable):
    """Everything after scoring: adequacy, primary, guarded secondary. Shared by the train dry run and test_AB."""
    out = {"readout": res["info"]}
    if len(res["d"]) < min_usable:
        out.update({"stop": f"adequacy: {len(res['d'])} usable founders < {min_usable}", "confirmed": False})
        return out
    out["primary"] = primary(res["d"], train_sd)
    out["confirmed"] = out["primary"]["confirmed"]
    try:  # secondary, reported only: core-cell rescore (CAND-A-P2 rules, pool = this set's evaluable cells)
        pur = B.purity(D["pool_Z"], D["pool_t"])[D["pool_row"]]
        core = pur >= B.CORE
        pc = primary(B.d_on(D, sb, sf, core)[0], train_sd)
        out["secondary_core"] = {"core_share": float(core.mean()), "T_core": pc["T"], "n": pc["n_founders"],
                                 "retained": pc["T"] / out["primary"]["T"] if out["primary"]["T"] else None}
    except Exception as e:  # noqa: BLE001
        out["secondary_core"] = {"error": repr(e)}
    return out


# ---------------------------------------------------------------- selftest
def toy():
    parent = C.toy_tree()
    n = len(parent)
    nchild = np.bincount(parent[parent >= 0], minlength=n)
    is_leaf = nchild == 0
    leaves = is_leaf.astype(int)
    depth = np.zeros(n)
    for i in range(1, n):
        depth[i] = depth[parent[i]] + 1
    for i in range(n - 1, 0, -1):
        leaves[parent[i]] += leaves[i]
    is_f = np.zeros(n, bool)
    is_f[np.flatnonzero(parent == 0)] = True
    s = {"leaves": leaves, "depth": depth, "rdist": depth * 0.5, "is_leaf": is_leaf}
    return {"parent": parent, "s": s, "names": [f"c{i}" for i in range(n)], "f_of": C.founder_of(parent, is_f),
            "end": C.subtree_end(parent)}


def toy_df(L, rng):
    import pandas as pd  # noqa: PLC0415

    lf = np.flatnonzero(L["s"]["is_leaf"])
    k = np.arange(len(lf))
    df = pd.DataFrame({"celltype": [f"T{v}" for v in (k // 4) % 3], "celltype_from_which_major_trajectory": "x",
                       "major_trajectory": [f"L{v}" for v in (k // 8) % 2]}, index=[L["names"][i] for i in lf])
    for c in C.UMAP:
        df[c] = rng.normal(size=len(lf))
    return df


def selftest():
    r = {}
    rng = np.random.default_rng(3)
    L = toy()
    df = toy_df(L, rng)
    sel = L["s"]["is_leaf"]
    saved = (C.MIN_CELLS, C.MIN_CLUST, C.MIN_CLASS_F)
    C.MIN_CELLS, C.MIN_CLUST, C.MIN_CLASS_F = 2, 1, 1
    try:
        D, spec, info = frame(L, sel, df, "celltype")
        D2, _, _ = frame(L, sel, df, "celltype", spec=json.loads(json.dumps(spec)))
        r["spec_roundtrip_same"] = all(np.array_equal(D[k], D2[k]) for k in ("y", "t", "base", "fnd", "anc"))
        r["types"], r["cells"] = len(spec["types"]), int(len(D["y"]))
        model = json.loads(json.dumps(fit_models(D, spec, np.ones(len(D["y"]), bool))))
        h = spec["types"][0]
        m = D["t"] == 0
        ref = C._fit_pred(D["base"][m], D["y"][m], D["base"][m])
        r["decision_matches_fit_pred"] = bool(np.allclose(dec(model["models"][h]["base"], D["base"][m]), ref,
                                                          atol=1e-9))
        res, De, sbe, sfe = evaluate(L, sel, df, model)
        r["evaluate_founders"] = len(res["d"])
        fin = json.loads(json.dumps(finish(De, sbe, sfe, res, 0.09, 1)))
        stp = json.loads(json.dumps(finish(De, sbe, sfe, res, 0.09, 99)))
        r["finish_ok"] = bool("primary" in fin and "secondary_core" in fin and "error" not in fin["secondary_core"]
                              and abs(fin["primary"]["T"] - np.mean(list(res["d"].values()))) < 1e-12
                              and "stop" in stp and stp["confirmed"] is False)
        # unseen labels on the "sealed" side: one type block renamed, one lineage renamed
        df2 = df.copy()
        df2.loc[df2["celltype"] == "T2", "celltype"] = "NEW"
        df2.loc[df2.index[:8], "major_trajectory"] = "LNEW"
        D3, _, info3 = frame(L, sel, df2, "celltype", spec=model)
        r["unseen_type_counted"] = info3["type_levels_unseen_in_train"]
        r["unseen_lineage_counted"] = info3["lineage_levels_unseen_in_train"]
        oh = D3["base"][:, : len(spec["lineages"])].sum(1)
        r["unseen_lineage_zero_onehot"] = bool((oh == 0).any() and (oh <= 1).all())
        r["unseen_type_not_scored"] = bool(len(D3["y"]) < len(D["y"]))
    finally:
        C.MIN_CELLS, C.MIN_CLUST, C.MIN_CLASS_F = saved
    r["counts_one_group"] = list(counts_AB("AB")) == ["AB"] and counts_AB("AB")["AB"] > 0
    r["pass"] = bool(r["spec_roundtrip_same"] and r["types"] >= 1 and r["decision_matches_fit_pred"]
                     and r["evaluate_founders"] >= 1 and r["finish_ok"] and r["unseen_type_counted"] == 1
                     and r["unseen_lineage_counted"] == 1 and r["unseen_lineage_zero_onehot"]
                     and r["unseen_type_not_scored"] and r["counts_one_group"])
    return r


if __name__ == "__main__":
    a = sys.argv[1:] or ["selftest"]
    if a[0] == "selftest":
        print(json.dumps(selftest(), default=float))
    elif a[0] == "fit":
        stage_fit()
    else:
        raise SystemExit(f"unknown stage {a[0]} (the look runs only through test_oracle)")
