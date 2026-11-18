"""C17-S0c: seal two new systems by clone before any statistic (H1, H24). Bence 2026-11-18: "Yes, pre-register".

Serves: C17-S0. Structure only (H9): unit IDs, cells per unit and design labels; no expression, fate or cluster.

rf2020_clone (GSE134242): unit = "<sample set>:<LARRY barcode set>"; size = barcoded cells; stratum = sample set x
    shared (>= 3 cells in each of two recipients) vs not. Recipient = sample-ID prefix (Sample1: second token).
cortical_clone (GSE284197 clones.h5ad): unit = obs index (asserted unique); size = n_cells;
    stratum = guide target x donor batch.

Writes results_c17/partition_<name>.tsv (unit<TAB>partition) and results_c17/c17s0c_seal.json (counts per partition
and stratum, file sha256). The maps are then committed under experiments/partition_maps/ and registered in partitions.py.

    python -I experiments/c17s0c_seal.py
"""

from __future__ import annotations

import collections
import gzip
import hashlib
import io
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from partitions import balanced_split  # noqa: E402

OUT = os.path.join(HERE, "results_c17")
GEO = "https://ftp.ncbi.nlm.nih.gov/geo/series"
RF = f"{GEO}/GSE134nnn/GSE134242/suppl/GSE134242_ARF2019_"
CORT = f"{GEO}/GSE284nnn/GSE284197/suppl/GSE284197_clones.h5ad"
SHA = {}


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "cellware-seal/1.0"})
    for i in range(6):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                b = r.read()
            SHA[url.rsplit("/", 1)[-1]] = hashlib.sha256(b).hexdigest()
            return b
        except Exception as e:  # noqa: BLE001
            print(f"retry {i + 1} {url}: {e}", flush=True)
            time.sleep(5 * (i + 1))
    raise RuntimeError(url)


def lines(name):
    return gzip.decompress(fetch(RF + name + ".txt.gz")).decode().splitlines()


def recipient(sample_id):
    t = sample_id.split("_")
    return t[1] if t[0] in ("HSC", "Kit", "KIT") and len(t) > 1 else t[0]


def rf2020():
    sets = {"Sample1": "samp_id", "Sample2": "samp_id_flat", "Sample3": "samp_id_flat",
            "Sample4": "samp_id_flat", "Sample5": "samp_id_flat"}
    size, per_rec = collections.Counter(), collections.defaultdict(collections.Counter)
    for s, id_f in sets.items():
        bcs, ids = lines(f"{s}_barcode_list"), lines(f"{s}_{id_f}")
        if len(ids) == len(bcs) + 1:
            ids = ids[1:]
        assert len(ids) == len(bcs), (s, len(ids), len(bcs))
        for b, i in zip(bcs, ids, strict=True):
            b = b.strip().strip("_")
            if b:
                u = f"{s}:{b}"
                size[u] += 1
                per_rec[u][recipient(i)] += 1
    shared = {u: sum(n >= 3 for n in per_rec[u].values()) >= 2 for u in size}
    strata = {u: f"{u.split(':')[0]}|{'shared' if shared[u] else 'single'}" for u in size}
    return dict(size), strata


def cortical():
    import h5py  # noqa: PLC0415

    with h5py.File(io.BytesIO(fetch(CORT)), "r") as h:
        obs = h["obs"]
        idx = obs.attrs.get("_index", "_index")
        index = [x.decode() if isinstance(x, bytes) else str(x) for x in obs[idx][:]]
        assert len(set(index)) == len(index), "obs index not unique: rows are not clones"

        def cat(col):
            g = obs[col]
            cats = [c.decode() if isinstance(c, bytes) else str(c) for c in g["categories"][:]]
            return [cats[k] if k >= 0 else "NA" for k in g["codes"][:]]

        n = obs["n_cells"][:]
        target, batch = cat("Gene_target"), cat("batch")
    size = {u: int(c) for u, c in zip(index, n, strict=True)}
    strata = {u: f"{t}|{b}" for u, t, b in zip(index, target, batch, strict=True)}
    return size, strata


def summarise(mapping, size, strata):
    parts = sorted(set(mapping.values()))
    tot = {p: {"units": sum(1 for u in mapping if mapping[u] == p),
               "cells": sum(size[u] for u in mapping if mapping[u] == p)} for p in parts}
    by_stratum = collections.defaultdict(dict)
    for s in sorted(set(strata.values())):
        for p in parts:
            by_stratum[s][p] = sum(1 for u in mapping if strata[u] == s and mapping[u] == p)
    return {"totals": tot, "units_by_stratum": by_stratum}


def main():
    os.makedirs(OUT, exist_ok=True)
    out = {}
    for name, fn, salt in (("rf2020_clone", rf2020, "cellware-rf2020-clone-v1"),
                           ("cortical_clone", cortical, "cellware-cortical-clone-v1")):
        size, strata = fn()
        mapping = balanced_split(size, salt, strata)
        with open(os.path.join(OUT, f"partition_{name}.tsv"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(f"# {name}: H24 balanced map, salt {salt}, sealed 2026-11-18 before any statistic\n")
            for u in sorted(mapping):
                fh.write(f"{u}\t{mapping[u]}\n")
        out[name] = summarise(mapping, size, strata)
    out["sha256"] = SHA
    with open(os.path.join(OUT, "c17s0c_seal.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("sealed:", {k: v["totals"] for k, v in out.items() if k != "sha256"})


if __name__ == "__main__":
    main()
