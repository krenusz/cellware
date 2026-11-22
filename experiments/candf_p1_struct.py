"""CAND-F-P1: which recipients enter the source's cross-recipient correlation? (structural counts only)

The source figure (Rodriguez-Correa et al. 2026, Fig. 5f) has 7 mice with clone counts {14, 22, 24, 27, 33, 64, 86}
(only that column was read; bias values stay unread). Its methods: cells without a barcode dropped, clones with < 2
cells dropped, clones without >= 1 barcoded HSC dropped. This script counts, per GSE134242 recipient, clones under
those filters using barcode lists and library (sample-ID) labels only: no expression, cluster or annotation file is
read, and no clone is split by partition (a count per recipient, like C17-S0b).

Pre-registered (EXPERIMENT_LOG CAND-F-P1): mapped = some 7 recipients match the published counts exactly under one
filter variant; else 'units unresolved' (CAND-F stops at P1).
"""

import collections
import gzip
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_cand", "candf_p1_struct.json")
RF = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE134nnn/GSE134242/suppl/GSE134242_ARF2019_"
SETS = {"Sample1": "samp_id", "Sample2": "samp_id_flat", "Sample3": "samp_id_flat",
        "Sample4": "samp_id_flat", "Sample5": "samp_id_flat"}
PUBLISHED = sorted([64, 86, 33, 22, 14, 24, 27])


def lines(name):
    req = urllib.request.Request(RF + name + ".txt.gz", headers={"User-Agent": "cellware/1.0"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return gzip.decompress(r.read()).decode().splitlines()


def recipient(sample_id):
    """Mouse id: Sample1 ids are '<compartment>_<mouse>_<rep>', the others '<mouse>_<library>'."""
    t = sample_id.split("_")
    return t[1] if t[0] in ("HSC", "Kit", "KIT") and len(t) > 1 else t[0]


def is_hsc(sample_id):
    return "HSC" in sample_id.upper()


def main():
    per = {}
    for s, id_f in SETS.items():
        bcs, ids = lines(f"{s}_barcode_list"), lines(f"{s}_{id_f}")
        if len(ids) == len(bcs) + 1:
            ids = ids[1:]
        assert len(ids) == len(bcs), (s, len(ids), len(bcs))
        cells = collections.defaultdict(lambda: collections.Counter())  # (rec, clone) -> {hsc, all}
        for b, i in zip(bcs, ids, strict=True):
            b = b.strip().strip("_")
            if not b:
                continue
            c = cells[(recipient(i), b)]
            c["all"] += 1
            c["hsc"] += is_hsc(i)
        recs = sorted({r for r, _ in cells})
        for r in recs:
            cl = [v for (rr, _), v in cells.items() if rr == r]
            per[f"{s}:{r}"] = {
                "clones_any": len(cl),
                "clones_ge2": sum(v["all"] >= 2 for v in cl),
                "clones_ge2_hsc1": sum(v["all"] >= 2 and v["hsc"] >= 1 for v in cl),
                "clones_ge2_hsc2": sum(v["all"] >= 2 and v["hsc"] >= 2 for v in cl),
                "cells": sum(v["all"] for v in cl),
                "hsc_cells": sum(v["hsc"] for v in cl),
            }
    variants = ["clones_any", "clones_ge2", "clones_ge2_hsc1", "clones_ge2_hsc2"]
    match = {}
    for v in variants:
        vals = collections.Counter(d[v] for d in per.values())
        hit = sum(min(vals[x], c) for x, c in collections.Counter(PUBLISHED).items())
        match[v] = {"published_counts_matched": hit, "of": len(PUBLISHED)}
    mapped = any(m["published_counts_matched"] == len(PUBLISHED) for m in match.values())
    res = {"recipients": per, "n_recipients": len(per), "published_counts": PUBLISHED, "match": match,
           "verdict": "mapped" if mapped else "units unresolved"}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    print(json.dumps({"n_recipients": len(per), "match": match, "verdict": res["verdict"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
