"""Track B5: clone identity in methylation vs expression, Camellia-seq sister HSCs.

Runs on GitHub Actions (NCBI blocks the local machine). Fetches per-cell CpG tables
(GSE222328) and plate RNA tables (GSE222246) for the cells with a DARLIN clone ID in
camellia_sample_sheet.tsv.gz (DARLIN_tutorial), builds 100-kb methylation bins and expression
features, and scores how well each readout separates sister from non-sister pairs (same mouse
and tissue). See the B5 pre-registration in EXPERIMENT_LOG.md.

    python experiments/track_b5_camellia.py
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import os
import re
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(__file__)
BIN = 100_000


def get(url, tries=8):
    url = url.replace("ftp://", "https://")
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                return r.read()
        except Exception:  # noqa: BLE001  NCBI throttles: back off and retry
            time.sleep(min(60, 5 * (i + 1)))
    raise RuntimeError(url)


def gsm_files(acc):
    text = get(
        f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={acc}&targ=gsm&form=text&view=brief"
    ).decode()
    out = {}
    for block in text.split("^SAMPLE = ")[1:]:
        title = re.search(r"!Sample_title = (.*)", block).group(1).strip()
        out[title] = re.findall(r"!Sample_supplementary_file_\d+ = (\S+)", block)
    return out


def meth_bins(raw):
    met, tot = {}, {}
    for line in gzip.GzipFile(fileobj=io.BytesIO(raw)).read().decode().splitlines()[1:]:
        c, pos, m, n, _ = line.split("\t")
        key = (c, int(pos) // BIN)
        met[key] = met.get(key, 0) + int(m)
        tot[key] = tot.get(key, 0) + int(m) + int(n)
    return {k: met[k] / tot[k] for k in tot}, tot


def rna_table(raw, barcodes):
    """Plate UMI table -> (cell barcodes, genes, genes x cells); handles either orientation."""
    lines = gzip.GzipFile(fileobj=io.BytesIO(raw)).read().decode().splitlines()
    head = lines[0].split("	")
    names, rows = [], []
    for line in lines[1:]:
        f = line.split("	")
        names.append(f[0])
        rows.append([float(v) for v in f[1:]])
    cols = head[1:] if len(head) == len(rows[0]) + 1 else head
    mat = np.array(rows)
    if len(set(cols) & barcodes) >= len(set(names) & barcodes):
        return cols, names, mat  # genes x cells
    return names, cols, mat.T  # cells x genes -> genes x cells


def pcs(x, k=20):
    x = x - x.mean(0)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    return u[:, :k] * s[:k]


def auc(pos, neg):
    allv = np.r_[pos, neg]
    ranks = allv.argsort().argsort() + 1
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def pair_scores(f, clone, group):
    c = np.corrcoef(f)
    i, j = np.triu_indices(len(f), 1)
    same_group = group[i] == group[j]
    i, j = i[same_group], j[same_group]
    sister = clone[i] == clone[j]
    return c[i, j], sister, i, j


def main():
    sheet = list(
        csv.DictReader(
            gzip.open(os.path.join(HERE, "camellia_sample_sheet.tsv.gz"), "rt"), delimiter="\t"
        )
    )
    cells = [
        r
        for r in sheet
        if r["large_clone_id"] and r["pass_metQC"] == "True" and r["pass_rnaQC"] == "True"
    ]
    meth_titles, rna_titles = gsm_files("GSE222328"), gsm_files("GSE222246")
    by_met = {t.split("HSC_", 1)[-1]: f for t, f in meth_titles.items()}
    by_rna = {t.split("HSC_", 1)[-1].replace("_RNA", ""): f for t, f in rna_titles.items()}
    meth, depth, keep = [], [], []
    for r in cells:
        files = [f for f in by_met.get(r["id_met"], []) if f.endswith("CpG.tsv.gz")]
        if files:
            try:
                rate, tot = meth_bins(get(files[0]))
                meth.append(rate)
                depth.append(tot)
                keep.append(r)
            except RuntimeError:
                pass
    plates = {}
    for r in keep:
        plate = r["id_rna"].rsplit("_RNA_", 1)[0]
        if plate not in plates and by_rna.get(plate):
            wanted = {c["id_rna"].rsplit("_RNA_", 1)[1] for c in keep}
            plates[plate] = rna_table(get(by_rna[plate][0]), wanted)
    rows, mfeat, efeat = [], [], []
    bins = sorted({k for m in meth for k in m})
    cover = np.array([[m.get(b, np.nan) for b in bins] for m in meth])
    ok_bins = np.mean(~np.isnan(cover), 0) >= 0.5
    cover = cover[:, ok_bins]
    kept_bins = [b for b, ok in zip(bins, ok_bins, strict=True) if ok]
    reads = np.array([[d.get(b, 0) for b in kept_bins] for d in depth])
    # plates list genes in different orders / sets: align on the genes common to all plates
    genes = sorted(set.intersection(*(set(v[1]) for v in plates.values())))
    gidx = {}
    for p, v in plates.items():
        pos = {x: k for k, x in enumerate(v[1])}
        gidx[p] = [pos[x] for x in genes]
    rreads = []
    for r, mrow, rd in zip(keep, cover, reads, strict=True):
        plate, bc = r["id_rna"].rsplit("_RNA_", 1)
        if plate not in plates:
            continue
        cols, _, mat = plates[plate]
        if bc not in cols:
            continue
        rows.append(r)
        mfeat.append(mrow)
        rreads.append(rd)
        efeat.append(mat[gidx[plate], cols.index(bc)])
    diag = {
        "cells_with_clone": len(cells),
        "meth_fetched": len(keep),
        "plates": len(plates),
        "matched": len(rows),
        "common_genes": len(genes),
        "example_cols": {p: v[0][:5] for p, v in plates.items()},
    }
    if len(rows) < 20:  # mapping failed: report what was seen and stop
        os.makedirs(os.path.join(HERE, "results_track_b"), exist_ok=True)
        json.dump(
            diag, open(os.path.join(HERE, "results_track_b", "b5_camellia.json"), "w"), indent=1
        )
        print(json.dumps(diag, indent=1))
        return
    m = np.array(mfeat)
    m = np.where(np.isnan(m), np.nanmean(m, 1, keepdims=True), m)
    e = np.array(efeat)
    e = np.log1p(e / np.maximum(e.sum(1, keepdims=True), 1) * 1e4)
    e = e[:, np.argsort(e.var(0))[-2000:]]
    clone = np.array([r["large_clone_id"] for r in rows])
    group = np.array([r["mouse"] + "|" + r["tissue"] for r in rows])
    feats = {"methylation": pcs(m), "expression": pcs(e)}
    rng = np.random.default_rng(0)
    out = {**diag, "cells": len(rows), "clones": int(len(set(clone))), "bins": int(ok_bins.sum())}
    scores = {}
    for name, f in feats.items():
        s, sis, i, j = pair_scores(f, clone, group)
        scores[name] = (s, sis, i, j)
        null = []
        for _ in range(500):  # permute clone labels within mouse|tissue
            perm = clone.copy()
            for gname in set(group):
                idx = np.flatnonzero(group == gname)
                perm[idx] = rng.permutation(perm[idx])
            ps = perm[i] == perm[j]
            if ps.any() and (~ps).any():
                null.append(auc(s[ps], s[~ps]))
        out[name] = {
            "auc": float(auc(s[sis], s[~sis])),
            "sister_pairs": int(sis.sum()),
            "null_95": float(np.percentile(null, 95)),
            "null_mean": float(np.mean(null)),
        }
    clones = np.array(sorted(set(clone)))
    diffs = []
    for _ in range(2000):  # bootstrap over clones: resample clones, keep pairs within chosen clones
        pick = set(rng.choice(clones, len(clones)))
        vals = []
        for name in ("methylation", "expression"):
            s, sis, i, j = scores[name]
            sel = np.isin(clone[i], list(pick)) & np.isin(clone[j], list(pick))
            vals.append(auc(s[sel & sis], s[sel & ~sis]) if (sel & sis).any() else np.nan)
        diffs.append(vals)
    d = np.array(diffs)
    d = d[~np.isnan(d).any(1)]
    out["methylation"]["ci"] = np.percentile(d[:, 0], [2.5, 97.5]).tolist()
    out["expression"]["ci"] = np.percentile(d[:, 1], [2.5, 97.5]).tolist()
    out["difference_ci"] = np.percentile(d[:, 0] - d[:, 1], [2.5, 97.5]).tolist()
    out["b5b"] = b5b(
        np.array(mfeat), np.array(rreads), np.array(efeat), genes, rows, clone, group, rng
    )
    os.makedirs(os.path.join(HERE, "results_track_b"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results_track_b", "b5b_camellia.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


PROLIF = ("Mki67", "Top2a", "Ccnb1", "Cdk1", "Mcm2", "Pcna")


def score_one(f, clone, group, rng, boot=1000, perms=300):
    """AUC, permutation-null 95th pct and clone-bootstrap CI for one feature matrix."""
    s, sis, i, j = pair_scores(f, clone, group)
    null = []
    for _ in range(perms):
        perm = clone.copy()
        for gname in set(group):
            idx = np.flatnonzero(group == gname)
            perm[idx] = rng.permutation(perm[idx])
        ps = perm[i] == perm[j]
        null.append(auc(s[ps], s[~ps]))
    clones = np.array(sorted(set(clone)))
    bs = []
    for _ in range(boot):
        pick = list(set(rng.choice(clones, len(clones))))
        sel = np.isin(clone[i], pick) & np.isin(clone[j], pick)
        if (sel & sis).any():
            bs.append(auc(s[sel & sis], s[sel & ~sis]))
    return {
        "auc": float(auc(s[sis], s[~sis])),
        "null_95": float(np.percentile(null, 95)),
        "ci": np.percentile(bs, [2.5, 97.5]).tolist(),
    }


def b5b(m, reads, e_raw, genes, rows, clone, group, rng):
    """B5b: confound-controlled methylation AUC and the identical-sister measurement ceiling."""
    imp = np.where(np.isnan(m), np.nanmean(m, 1, keepdims=True), m)
    e = np.log1p(e_raw / np.maximum(e_raw.sum(1, keepdims=True), 1) * 1e4)
    present = [genes.index(g) for g in PROLIF if g in genes]
    prolif = e[:, present].mean(1) if present else np.zeros(len(e))
    glob = np.array([float(r["met_rate"]) for r in rows])
    cov = np.log(np.array([float(r["nCG"]) for r in rows]))
    z = np.c_[np.ones(len(m)), glob, cov, prolif]
    resid = imp - z @ np.linalg.lstsq(z, imp, rcond=None)[0]
    out = {
        "proliferation_genes": [g for g in PROLIF if g in genes],
        "controlled": score_one(pcs(resid), clone, group, rng),
    }
    # ceiling: sisters identical (clone-pooled bin rate), each cell keeps its own read depth
    pooled = {}
    for c in set(clone):
        idx = clone == c
        k = np.nansum(np.where(np.isnan(m[idx]), 0, m[idx]) * reads[idx], 0)
        n = reads[idx].sum(0)
        pooled[c] = np.where(n > 0, k / np.maximum(n, 1), np.nan)
    ceil = []
    for _ in range(20):
        sim = np.full_like(m, np.nan)
        for t in range(len(m)):
            p, n = pooled[clone[t]], reads[t]
            ok = (n > 0) & ~np.isnan(p)
            sim[t, ok] = rng.binomial(n[ok].astype(int), p[ok]) / n[ok]
        sim = np.where(np.isnan(sim), np.nanmean(sim, 1, keepdims=True), sim)
        s, sis, _, _ = pair_scores(pcs(sim), clone, group)
        ceil.append(auc(s[sis], s[~sis]))
    out["ceiling_auc"] = float(np.median(ceil))
    out["ceiling_range"] = [float(min(ceil)), float(max(ceil))]
    return out


if __name__ == "__main__":
    main()
