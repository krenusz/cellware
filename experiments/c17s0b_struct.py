"""C17-S0b: structure-only checks for two candidate systems (Bence 2026-11-18: "go 1").

Serves: C17-S0. H9: prints columns, tags and counts only; no expression, fate or outcome statistic.

1. GSE134242 (LARRY-barcoded HSC transplants): per sample file set, cells per recipient prefix, clones
   (barcode sets) per recipient, and how many clones are shared across >= 2 recipients (cells >= 1 / >= 3 each).
   Fate labels and expression are not downloaded.
2. GSE284197 (cortical CRISPRi Perturb-seq + lineage): obs column names, dtypes and the number of levels per
   column, read over HTTP range requests (the 4.4 GB screen file is not downloaded); level names only for
   columns that look like guide, target, clone or sample identifiers.

    python -I experiments/c17s0b_struct.py   # writes experiments/results_c17/c17s0b_struct.json
"""

from __future__ import annotations

import collections
import gzip
import io
import json
import os
import re
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_c17")
GEO = "https://ftp.ncbi.nlm.nih.gov/geo/series"
RF = f"{GEO}/GSE134nnn/GSE134242/suppl/GSE134242_ARF2019_"
CORT = f"{GEO}/GSE284nnn/GSE284197/suppl/GSE284197_"
UA = {"User-Agent": "cellware-struct/1.0"}
ID_LIKE = re.compile(r"guide|sgrna|grna|target|perturb|clone|lineage|barcode|sample|batch|donor|individual|lib", re.I)


def fetch(url, rng=None):
    hdr = dict(UA)
    if rng:
        hdr["Range"] = f"bytes={rng[0]}-{rng[1]}"
    for i in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=120) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            print(f"retry {i + 1} {url}: {e}", flush=True)
            time.sleep(5 * (i + 1))
    raise RuntimeError(url)


def lines(name):
    return gzip.decompress(fetch(RF + name + ".txt.gz")).decode().splitlines()


def rf2020():
    out = {}
    sets = {
        "Sample1": ("barcode_list", "samp_id"),
        "Sample2": ("barcode_list", "samp_id_flat"),
        "Sample3": ("barcode_list", "samp_id_flat"),
        "Sample4": ("barcode_list", "samp_id_flat"),
        "Sample5": ("barcode_list", "samp_id_flat"),
    }
    for s, (bc_f, id_f) in sets.items():
        bcs, ids = lines(f"{s}_{bc_f}"), lines(f"{s}_{id_f}")
        if len(ids) == len(bcs) + 1:
            ids = ids[1:]
        if len(ids) != len(bcs):
            out[s] = {"error": f"length mismatch {len(bcs)} vs {len(ids)}"}
            continue
        rec = [i.split("_")[0] for i in ids]
        lib = collections.Counter(ids)
        clone_rec = collections.defaultdict(collections.Counter)
        for b, r in zip(bcs, rec, strict=True):
            b = b.strip().strip("_")
            if b:
                clone_rec[b][r] += 1
        recs = sorted(set(rec))
        shared1 = sum(len(c) >= 2 for c in clone_rec.values())
        shared3 = sum(sum(n >= 3 for n in c.values()) >= 2 for c in clone_rec.values())
        out[s] = {
            "cells": len(ids),
            "cells_with_barcode": sum(bool(b.strip().strip("_")) for b in bcs),
            "libraries": dict(lib),
            "recipient_prefixes": {r: rec.count(r) for r in recs},
            "clones": len(clone_rec),
            "clones_per_recipient": {r: sum(r in c for c in clone_rec.values()) for r in recs},
            "clones_shared_2plus_recipients": shared1,
            "clones_shared_2plus_recipients_3plus_cells_each": shared3,
        }
    return out


class HttpFile(io.RawIOBase):
    """Read-only file object over HTTP range requests, with a block cache (for h5py)."""

    def __init__(self, url, block=1 << 20):
        self.url, self.block, self.pos, self.cache, self.fetched = url, block, 0, {}, 0
        req = urllib.request.Request(url, headers=UA, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def _blk(self, i):
        if i not in self.cache:
            a = i * self.block
            self.cache[i] = fetch(self.url, (a, min(a + self.block, self.size) - 1))
            self.fetched += len(self.cache[i])
        return self.cache[i]

    def readinto(self, b):
        n, out = len(b), bytearray()
        while len(out) < n and self.pos < self.size:
            i, o = divmod(self.pos, self.block)
            chunk = self._blk(i)[o : o + n - len(out)]
            out += chunk
            self.pos += len(chunk)
        b[: len(out)] = out
        return len(out)


def obs_structure(name):
    import h5py  # noqa: PLC0415

    f = HttpFile(CORT + name)
    res = {"file_mb": round(f.size / 1e6), "columns": {}}
    with h5py.File(io.BufferedReader(f, buffer_size=1 << 20), "r") as h:
        obs = h["obs"]
        idx = obs.attrs.get("_index", "_index")
        res["n_obs"] = int(obs[idx].shape[0]) if idx in obs else None
        res["top_level"] = sorted(h.keys())
        for k in obs.keys():
            if k == idx:
                continue
            node = obs[k]
            if isinstance(node, h5py.Group) and "categories" in node:
                cats = node["categories"][:]
                col = {"kind": "categorical", "levels": int(len(cats))}
                if ID_LIKE.search(k) and len(cats) <= 200:
                    col["level_names"] = [c.decode() if isinstance(c, bytes) else str(c) for c in cats]
            else:
                col = {"kind": str(getattr(node, "dtype", type(node).__name__))}
            res["columns"][k] = col
    res["mb_fetched"] = round(f.fetched / 1e6, 1)
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    out = {"GSE134242": rf2020(), "GSE284197": {}}
    for name in ("clones.h5ad", "screen.h5ad"):
        try:
            out["GSE284197"][name] = obs_structure(name)
        except Exception as e:  # noqa: BLE001
            out["GSE284197"][name] = {"error": repr(e)[:300]}
    json.dump(out, open(os.path.join(OUT, "c17s0b_struct.json"), "w", encoding="utf-8"), indent=1)
    print("wrote results_c17/c17s0b_struct.json")


if __name__ == "__main__":
    main()
