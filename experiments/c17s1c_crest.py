"""C17-S1C: CRISPR brain barcoding set (GSE210139), S0 structure check of the single-cell barcode tables.

Bence 2026-11-19: "go" (download the CRISPR brain set). Serves: C17-S1 (C17-S1R: grow the systems).
Pre-registered in EXPERIMENT_LOG (C17-S1C). Structure only (H9): column names and types, row counts, distinct
counts; per-clone cell and allele counts for the gate. Columns whose names suggest a cell type or fate are
listed by name and type only, never tabulated. R files are parsed with `rdata` (no R code runs); run with -I.

    python -I experiments/c17s1c_crest.py schema                   # columns, types, distinct counts
    python -I experiments/c17s1c_crest.py gate CLONE ALLELE SAMPLE  # gate with the named columns
    python -I experiments/c17s1c_crest.py split                    # amendment 2: split clones + gate
    python -I experiments/c17s1c_crest.py seal                     # amendment 2: H24 map by clone
    python -I experiments/c17s1c_crest.py selftest
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "data" / "ext_crest" / "raw"
OUT = HERE / "results_c17"
FILE = "GSE210139_single_cell_CREST_barcode_qc_updated20230422.rds.gz"
FATE_LIKE = re.compile(r"type|fate|cluster|annot|class|ident|lineage|celltype|state", re.I)
MIN_CELLS, MIN_SUB, MIN_SUBS = 8, 4, 2
NAME, SALT = "crest_split_clone", "cellware-crest-split-clone-v1"
MAPS = HERE / "partition_maps"


def load(path):
    import rdata

    obj = rdata.read_rds(path)
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, dict):
        frames = {str(k): v for k, v in obj.items() if isinstance(v, pd.DataFrame)}
        if len(frames) == 1:
            return next(iter(frames.values()))
        # a list of per-sample tables plus merged ones: stack the replicate tables only (no duplicates)
        reps = {k: v for k, v in frames.items() if re.search(r"_rep\d+$", k)}
        return pd.concat([v.assign(sample_key=k) for k, v in reps.items()], ignore_index=True)
    return pd.DataFrame(obj)


def schema(df):
    cols = []
    for c in df.columns:
        s = df[c]
        entry = {"name": str(c), "dtype": str(s.dtype), "n_distinct": int(s.nunique(dropna=True)),
                 "n_missing": int(s.isna().sum())}
        if FATE_LIKE.search(str(c)):
            entry["fate_like"] = "name and type only"
        elif entry["n_distinct"] <= 12 and s.dtype == object or str(s.dtype) == "category":
            if entry["n_distinct"] <= 12:
                entry["tags"] = sorted(map(str, s.dropna().unique()))
        cols.append(entry)
    return {"rows": len(df), "columns": cols}


def gate(df, clone, allele, sample):
    df = df[df[clone].notna()]
    per = []
    for cid, g in df.groupby(clone):
        subs = g[allele].value_counts()
        per.append({"clone": str(cid), "cells": len(g), "subclades_ge4": int((subs >= MIN_SUB).sum()),
                    "samples": int(g[sample].nunique())})
    per = pd.DataFrame(per)
    elig = per[(per["cells"] >= MIN_CELLS) & (per["subclades_ge4"] >= MIN_SUBS)]
    return {"cells_with_clone": int(len(df)), "clones": int(len(per)), "eligible_clones": int(len(elig)),
            "expected_train_eligible": int(0.4 * len(elig)), "gate_pass": bool(0.4 * len(elig) >= 30),
            "clones_ge8_cells": int((per["cells"] >= MIN_CELLS).sum()),
            "clones_spanning_samples": int((per["samples"] > 1).sum()),
            "cells_per_clone_quantiles": per["cells"].quantile([0.5, 0.9, 0.99]).round(1).tolist()}


def informative(p):
    """A barcode pattern that can mark a clone: some edit, and not one deletion spanning every site."""
    if not isinstance(p, str) or p in ("NA", ""):
        return False
    sites = p.split("_")
    if all(s == "NONE" for s in sites):
        return False
    return len(set(sites)) > 1 or sites[0] == "NONE"


def mapped(df):
    """Amendment 1 mapping: clone = informative pattern.v2 seen in one sample only (recurrent alleles are
    homoplasy); subclade = pattern.v1 within the clone (informative, non-missing)."""
    group = df["sample_key"].str.replace(r"_rep\d+$", "", regex=True)
    d = df.assign(group=group)
    d = d[d["pattern.v2"].map(informative)]
    n_samp = d.groupby("pattern.v2")["sample_key"].nunique()
    d = d[d["pattern.v2"].map(n_samp) == 1]
    d = d[d["pattern.v1"].map(informative)]
    return d.assign(clone=d["sample_key"] + "|" + d["pattern.v2"])


def split_clones(df):
    """Amendment 2: clones shared by pandaE11_k and OGN_k (same k) and absent from every other sample."""
    d = df[df["pattern.v2"].map(informative)].copy()
    where = d.groupby("pattern.v2")["sample_key"].agg(lambda s: frozenset(s))
    rows = []
    for k in ("1", "2", "3"):
        pair = frozenset({f"pandaE11_rep{k}", f"OGN_rep{k}"})
        pats = [p for p, s in where.items() if s == pair]
        for p in pats:
            g = d[d["pattern.v2"] == p]
            rows.append({"embryo": k, "pattern": p, "arm1": int((g["sample_key"] == f"pandaE11_rep{k}").sum()),
                         "arm2": int((g["sample_key"] == f"OGN_rep{k}").sum())})
    return pd.DataFrame(rows)


def split_gate(cl, a1=3, a2=4):
    elig = cl[(cl["arm1"] >= a1) & (cl["arm2"] >= a2)]
    return {"shared_clones": int(len(cl)), "eligible_clones": int(len(elig)),
            "eligible_per_embryo": {str(k): int(v) for k, v in elig.groupby("embryo").size().items()},
            "expected_train_eligible": int(0.4 * len(elig)), "gate_pass": bool(0.4 * len(elig) >= 30),
            "arm1_cells_median": float(cl["arm1"].median()) if len(cl) else 0.0,
            "arm2_cells_median": float(cl["arm2"].median()) if len(cl) else 0.0}


def coincidence(df):
    """Informative v2 patterns shared by each matched pair vs by unrelated pandaE11/OGN pairs (structure)."""
    d = df[df["pattern.v2"].map(informative)]
    pats = {k: set(g["pattern.v2"]) for k, g in d.groupby("sample_key")}
    pair = {k: len(pats[f"pandaE11_rep{k}"] & pats[f"OGN_rep{k}"]) for k in "123"}
    other = [len(pats[f"pandaE11_rep{i}"] & pats[f"OGN_rep{j}"]) for i in "123" for j in "123" if i != j]
    return {"matched_pair_shared": pair, "unrelated_pair_shared": other,
            "coincidental_share_est": round(sum(other) / len(other) / (sum(pair.values()) / 3), 3)}


def seal(cl):
    """H24 balanced map by clone (unit = '<embryo>:<v2 pattern>', size = cells in both arms, strata = embryo)."""
    sys.path.insert(0, str(HERE))  # -I drops the script dir; partitions.py is ours, not downloaded
    from partitions import balanced_split  # noqa: PLC0415

    size = {f"{r.embryo}:{r.pattern}": int(r.arm1 + r.arm2) for r in cl.itertuples()}
    strata = {f"{r.embryo}:{r.pattern}": r.embryo for r in cl.itertuples()}
    elig = {f"{r.embryo}:{r.pattern}" for r in cl.itertuples() if r.arm1 >= 3 and r.arm2 >= 4}
    mapping = balanced_split(size, SALT, strata)
    MAPS.mkdir(exist_ok=True)
    with open(MAPS / f"partition_{NAME}.tsv", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# {NAME}: H24 balanced map, salt {SALT}, sealed 2026-11-19 before any statistic\n")
        for u in sorted(mapping):
            fh.write(f"{u}\t{mapping[u]}\n")
    parts = sorted(set(mapping.values()))
    return {"salt": SALT, "units": len(mapping), "totals": {p: {
        "units": sum(v == p for v in mapping.values()),
        "eligible": sum(mapping[u] == p for u in elig),
        "cells": sum(size[u] for u in mapping if mapping[u] == p)} for p in parts},
        "eligible_by_embryo": {k: {p: sum(mapping[u] == p for u in elig if strata[u] == k) for p in parts}
                               for k in "123"}}


def selftest():
    df = pd.DataFrame({"clone": ["a"] * 9 + ["b"] * 3, "allele": ["x"] * 5 + ["y"] * 4 + ["z"] * 3,
                       "sample": ["s1"] * 12, "celltype": ["n"] * 12})
    sc = schema(df)
    ct = next(c for c in sc["columns"] if c["name"] == "celltype")
    assert "tags" not in ct and ct["fate_like"], ct
    g = gate(df, "clone", "allele", "sample")
    assert g["clones"] == 2 and g["eligible_clones"] == 1 and not g["gate_pass"], g
    sp = pd.DataFrame({"pattern.v2": ["1D+5_NONE"] * 7 + ["9D+2_NONE"] * 2 + ["4D+1_NONE"] * 2,
                       "sample_key": ["pandaE11_rep1"] * 3 + ["OGN_rep1"] * 4 + ["pandaE11_rep1", "E15_rep1"]
                       + ["pandaE11_rep2", "OGN_rep2"]})
    cl = split_clones(sp)
    assert len(cl) == 2 and set(cl["pattern"]) == {"1D+5_NONE", "4D+1_NONE"}, cl  # 9D+2 seen in E15: dropped
    assert split_gate(cl)["eligible_clones"] == 1
    return {"ok": True}


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    if stage == "selftest":
        print(json.dumps(selftest()))
        return
    df = load(RAW / FILE)
    if stage == "schema":
        res = schema(df)
    elif stage == "split":  # amendment 2
        cl = split_clones(df)
        res = {"mapping": split_clones.__doc__.strip(), **split_gate(cl), **coincidence(df)}
    elif stage == "seal":  # amendment 2, gate passed: seal before any fate read (H1, H24)
        res = seal(split_clones(df))
    else:  # amendment 1: gate on the mapped clones, per sample group and pooled
        d = mapped(df)
        res = {"mapping": mapped.__doc__.strip(), "pooled": gate(d, "clone", "pattern.v1", "sample_key"),
               "by_group": {g: gate(x, "clone", "pattern.v1", "sample_key") for g, x in d.groupby("group")}}
    OUT.mkdir(exist_ok=True)
    json.dump(res, open(OUT / f"c17s1c_{stage}.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
