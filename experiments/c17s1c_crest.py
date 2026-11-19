"""C17-S1C: CRISPR brain barcoding set (GSE210139), S0 structure check of the single-cell barcode tables.

Bence 2026-11-19: "go" (download the CRISPR brain set). Serves: C17-S1 (C17-S1R: grow the systems).
Pre-registered in EXPERIMENT_LOG (C17-S1C). Structure only (H9): column names and types, row counts, distinct
counts; per-clone cell and allele counts for the gate. Columns whose names suggest a cell type or fate are
listed by name and type only, never tabulated. R files are parsed with `rdata` (no R code runs); run with -I.

    python -I experiments/c17s1c_crest.py schema                   # columns, types, distinct counts
    python -I experiments/c17s1c_crest.py gate CLONE ALLELE SAMPLE  # gate with the named columns
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


def selftest():
    df = pd.DataFrame({"clone": ["a"] * 9 + ["b"] * 3, "allele": ["x"] * 5 + ["y"] * 4 + ["z"] * 3,
                       "sample": ["s1"] * 12, "celltype": ["n"] * 12})
    sc = schema(df)
    ct = next(c for c in sc["columns"] if c["name"] == "celltype")
    assert "tags" not in ct and ct["fate_like"], ct
    g = gate(df, "clone", "allele", "sample")
    assert g["clones"] == 2 and g["eligible_clones"] == 1 and not g["gate_pass"], g
    return {"ok": True}


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    if stage == "selftest":
        print(json.dumps(selftest()))
        return
    df = load(RAW / FILE)
    if stage == "schema":
        res = schema(df)
    else:  # amendment 1: gate on the mapped clones, per sample group and pooled
        d = mapped(df)
        res = {"mapping": mapped.__doc__.strip(), "pooled": gate(d, "clone", "pattern.v1", "sample_key"),
               "by_group": {g: gate(x, "clone", "pattern.v1", "sample_key") for g, x in d.groupby("group")}}
    OUT.mkdir(exist_ok=True)
    json.dump(res, open(OUT / f"c17s1c_{stage}.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
