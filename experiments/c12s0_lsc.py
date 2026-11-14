"""C12-S0: LSC expansion LARRY gate check (Zenodo 18444428). Pre-registered in EXPERIMENT_LOG ("C12-S0").

Serves: C12-S0. Bence: "Yes, download LSC".
    python c12s0_lsc.py seal     # D20 ex vivo barcode co-calling -> clone map (written once, then locked)
    python c12s0_lsc.py counts   # unit counts per partition (structural); outcomes on TRAIN clones only
"""

from __future__ import annotations

import collections
import gzip
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

import partitions

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "data", "ext_lsc_larry")
RAW = os.path.join(D, "raw")
MAP = os.path.join(D, "clone_map_d20.tsv")
OUT = os.path.join(HERE, "results_c12", "c12s0_counts.json")
DS = "lsc_larry"
MIN_UMI, MIN_LINK, MIN_CELLS = 3, 2, 3
LINK_FRAC, AMBIENT = 0.5, 0.02   # amendment 1 (EXPERIMENT_LOG): symmetric co-occurrence, ambient barcodes removed
SAMPLES = ["D20_exvivo", "D20_invivo", "D40_exvivo", "D40_invivo_control", "D40_invivo_treated"]


def _features(s):
    with gzip.open(os.path.join(RAW, f"{s}_features.tsv.gz"), "rt") as f:
        return [line.rstrip("\n").split("\t") for line in f]


def calls(s):
    """{cell index: [called barcode IDs]} for one sample (LARRY Custom features, >= MIN_UMI)."""
    feats = _features(s)
    cust = {i + 1: x[0] for i, x in enumerate(feats) if x[2] == "Custom"}   # 1-based mtx rows
    lo, hi = min(cust), max(cust)
    out = collections.defaultdict(list)
    path = os.path.join(RAW, f"{s}_matrix.mtx.gz")
    reader = pd.read_csv(path, sep=" ", comment="%", header=None, skiprows=3, chunksize=5_000_000,
                         dtype={0: np.int64, 1: np.int64, 2: np.int64})
    for ch in reader:
        r, c, v = ch[0].to_numpy(), ch[1].to_numpy(), ch[2].to_numpy()
        k = (r >= lo) & (r <= hi) & (v >= MIN_UMI)
        for rr, cc in zip(r[k], c[k]):
            if rr in cust:
                out[int(cc)].append(cust[rr])
    return out


def seal():
    assert not os.path.exists(MAP), "clone map already written; never redrawn"
    cc = calls("D20_exvivo")
    n_b = collections.Counter(b for bcs in cc.values() for b in set(bcs))
    ambient = {b for b, n in n_b.items() if n > AMBIENT * len(cc)}               # amendment 1
    cc = {cell: [b for b in bcs if b not in ambient] for cell, bcs in cc.items()}
    cc = {cell: bcs for cell, bcs in cc.items() if bcs}
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs = collections.Counter()
    for bcs in cc.values():
        u = sorted(set(bcs))
        for b in u:
            find(b)
        for i in range(len(u)):
            for j in range(i + 1, len(u)):
                pairs[(u[i], u[j])] += 1
    for (a, b), n in pairs.items():
        if n >= MIN_LINK and n / max(n_b[a], n_b[b]) >= LINK_FRAC:              # amendment 1
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    comp = collections.defaultdict(list)
    for b in parent:
        comp[find(b)].append(b)
    rows = [(b, min(m)) for m in comp.values() for b in m] + [(b, "AMBIENT") for b in ambient]
    pd.DataFrame(rows, columns=["barcode", "clone"]).sort_values("barcode").to_csv(MAP, sep="\t", index=False)
    sizes = collections.Counter(min(len(m), 5) for m in comp.values())
    print(json.dumps({"ambient_barcodes_removed": len(ambient), "cells_with_call": len(cc), "called_barcodes": len(parent), "components": len(comp),
                      "largest_component_barcodes": max(len(m) for m in comp.values()),
                      "components_by_n_barcodes(5=5+)": dict(sorted(sizes.items()))}, indent=1))
    h = hashlib.sha256(open(MAP, "rb").read()).hexdigest()
    with open(os.path.join(D, "SHA256SUMS"), "a") as f:
        f.write(f"{h}  clone_map_d20.tsv\n")
    print("clone map sha256", h)


def assign(s, cmap):
    """cell -> clone key; cells whose called barcodes map to >1 clone are dropped (ambiguous)."""
    out, amb = {}, 0
    for cell, bcs in calls(s).items():
        ks = {cmap.get(b, b) for b in bcs} - {"AMBIENT"}
        if not ks:
            continue
        if len(ks) == 1:
            out[cell] = ks.pop()
        else:
            amb += 1
    return out, amb


def counts():
    cmap = dict(pd.read_csv(MAP, sep="\t").itertuples(index=False, name=None))
    res = {"min_umi": MIN_UMI, "min_link": MIN_LINK, "min_cells": MIN_CELLS}
    size = {}
    for s in SAMPLES:
        a, amb = assign(s, cmap)
        size[s] = collections.Counter(a.values())
        res[f"{s}_cells_assigned"] = len(a)
        res[f"{s}_cells_ambiguous"] = amb
    d20 = [k for k, n in size["D20_exvivo"].items() if n >= MIN_CELLS]
    part = {k: partitions.partition_of(DS, k) for k in d20}
    res["D20_units_ge3_per_partition"] = dict(collections.Counter(part.values()))       # structural
    tr = [k for k in d20 if part[k] == "train"]
    res["train"] = {"n_units": len(tr)}
    for s in SAMPLES[1:]:
        n = np.array([size[s].get(k, 0) for k in tr])
        res["train"][s] = {"present": int((n > 0).sum()), "frac_present": round(float((n > 0).mean()), 3),
                           "ge3": int((n >= 3).sum()),
                           "size_q": [int(x) for x in np.quantile(n, [0.25, 0.5, 0.75, 0.9])] if len(n) else None}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"seal": seal, "counts": counts}[sys.argv[1]]()
