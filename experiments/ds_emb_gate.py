"""DS-EMB-G: S0 structure gates for embryo leads (structure only, no counts of clones or pairs yet).

Bence 2026-11-25: "2" = gate all six. Pre-registered in EXPERIMENT_LOG (DS-EMB-G).
Lineage tables only (data/ext_zf_scartrace, data/ext_zf_linnaeus); read with -I.
Columns whose names suggest cell type, cluster, tissue, organ or fate are listed by name only.

    python -I experiments/ds_emb_gate.py schema
"""

from __future__ import annotations

import gzip
import io
import json
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUT = HERE / "results_c17"
HELD = re.compile(r"type|fate|cluster|tissue|organ|annot|class|ident|state|leiden|louvain|region|lineage|germ|layer|tsne|umap|clust|location|loc$|bin$",
                  re.I)
EXAMPLE_OK = re.compile(r"barcode|cell|clone|scar|allele|cigar|read|count|umi|fish|larva|animal|embryo|sample|^id$|^unnamed",
                        re.I)
DIRS = ("ext_zf_scartrace", "ext_zf_linnaeus")


def read_table(path):
    name = path.name
    sep = "," if ".csv" in name else "\t"
    raw = gzip.decompress(path.read_bytes())
    return pd.read_csv(io.BytesIO(raw), sep=sep, low_memory=False)


def mask(v):
    """Cell IDs in the scar-clone tables embed free-text tokens; keep only the shape (letters a, digits 9)."""
    return re.sub(r"[A-Za-z]+", "a", re.sub(r"\d+", "9", v))


def cols(df, mask_ids=False):
    out, held = [], []
    for c in df.columns:
        s = df[c]
        e = {"name": str(c), "dtype": str(s.dtype)}
        if HELD.search(str(c)):
            held.append(str(c))
            e["held"] = "name only"
        elif EXAMPLE_OK.search(str(c)) and s.notna().any():
            e["examples"] = [(mask(str(v)) if mask_ids else str(v))[:60] for v in s.dropna().iloc[:3]]
        out.append(e)
    return {"rows": int(len(df)), "columns": out, "held_columns": held}


def schema():
    res = {}
    for d in DIRS:
        for f in sorted((DATA / d / "raw").glob("*.gz")):
            res[f"{d}/{f.name}"] = cols(read_table(f), mask_ids=(d == "ext_zf_scartrace"))
    (OUT / "ds_emb_gate_schema.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


if __name__ == "__main__":
    if sys.argv[1:] == ["schema"]:
        r = schema()
        for k, v in r.items():
            print(k, v["rows"], len(v["columns"]), "HELD:", v["held_columns"])
    else:
        sys.exit(__doc__)
