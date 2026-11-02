"""Track B6: where the Camellia clone signal lives (B6a), Meg fate (B6b), leave one mouse out (B6c).

Runs on GitHub Actions (NCBI blocks the local machine). Same cells and (mouse, clone) keys as
B5c. See the B6 pre-registration in EXPERIMENT_LOG.md.

    python experiments/track_b6_camellia.py
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import os

import numpy as np
from track_b5_camellia import auc, get, gsm_files, pcs, rna_table, score_one

HERE = os.path.dirname(__file__)
GTF = "https://ftp.ensembl.org/pub/release-102/gtf/mus_musculus/Mus_musculus.GRCm38.102.gtf.gz"
LINEAGE = (
    "Procr Hlf Mecom Fgd5 Hoxb5 Mllt3 Vwf Pf4 Itga2b Gp9 Gata1 Klf1 Epor Gata2 Car1 Tfrc Cebpa "
    "Cebpe Spi1 Mpo Elane Csf1r Irf8 Ly86 Flt3 Il7r Ebf1 Dntt Rag1 Cd34 Kit Ly6a Mpl Slamf1 Cd48 "
    "Meis1 Ikzf1 Runx1 Gfi1 Gfi1b"
).split()


def genes():
    """Protein-coding genes: name -> (chrom, tss, start, end)."""
    out = {}
    for line in gzip.GzipFile(fileobj=io.BytesIO(get(GTF))).read().decode().splitlines():
        if line.startswith("#"):
            continue
        f = line.split("\t")
        if f[2] != "gene" or 'gene_biotype "protein_coding"' not in f[8]:
            continue
        name = f[8].split('gene_name "')[1].split('"')[0]
        start, end = int(f[3]), int(f[4])
        out[name] = (f[0], start if f[6] == "+" else end, start, end)
    return out


def windows(names, gene_map, half=50_000, step=10_000):
    regs = []
    for g in names:
        if g in gene_map:
            c, tss, _, _ = gene_map[g]
            regs += [(c, s, s + step) for s in range(tss - half, tss + half, step)]
    return regs


def region_rates(raw, regions):
    """Per-region CpG methylation rate for one cell (NaN where uncovered)."""
    by_chr = {}
    for line in gzip.GzipFile(fileobj=io.BytesIO(raw)).read().decode().splitlines()[1:]:
        c, pos, m, n, _ = line.split("	")
        by_chr.setdefault(c, []).append((int(pos), int(m), int(m) + int(n)))
    out = np.full(sum(len(v[0]) for v in regions.values()), np.nan)
    for c, (idx, st, en) in regions.items():
        if c not in by_chr:
            continue
        a = np.array(sorted(by_chr[c]))
        cm, ct = np.r_[0, np.cumsum(a[:, 1])], np.r_[0, np.cumsum(a[:, 2])]
        i, j = np.searchsorted(a[:, 0], st), np.searchsorted(a[:, 0], en)
        tot = ct[j] - ct[i]
        out[idx] = np.where(tot > 0, (cm[j] - cm[i]) / np.maximum(tot, 1), np.nan)
    return out


def by_chrom(regs):
    """[(chrom, start, end)] -> {chrom: (indices, starts, ends)} for vectorised lookup."""
    out = {}
    for k, (c, s, e) in enumerate(regs):
        out.setdefault(c, ([], [], []))
        out[c][0].append(k)
        out[c][1].append(s)
        out[c][2].append(e)
    return {c: tuple(np.array(v) for v in t) for c, t in out.items()}


def feature_auc(x, clone, group, rng):
    ok = np.mean(~np.isnan(x), 0) >= 0.5
    x = x[:, ok]
    if x.shape[1] < 3:
        return {"regions": int(ok.sum()), "auc": None}
    x = np.where(np.isnan(x), np.nanmean(x, 1, keepdims=True), x)
    r = score_one(pcs(x, min(20, x.shape[1] - 1)), clone, group, rng, boot=300, perms=100)
    return {"regions": int(ok.sum()), **r}


def main():
    rng = np.random.default_rng(0)
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
    gmap = genes()
    names = sorted(gmap)
    sets = {"lineage": windows(LINEAGE, gmap)}
    sets["promoters"] = [(c, t - 2000, t + 2000) for c, t, _, _ in gmap.values()]
    sets["gene_bodies"] = [(c, s, e) for c, _, s, e in gmap.values()]
    ends = {}
    for c, _, _, e in gmap.values():
        ends[c] = max(ends.get(c, 0), e)
    sets["bins100k"] = [(c, s, s + 100_000) for c, e in ends.items() for s in range(0, e, 100_000)]
    for k in range(50):
        sets[f"random_{k}"] = windows(list(rng.choice(names, len(LINEAGE), replace=False)), gmap)
    meth_titles, rna_titles = gsm_files("GSE222328"), gsm_files("GSE222246")
    by_met = {t.split("HSC_", 1)[-1]: f for t, f in meth_titles.items()}
    by_rna = {t.split("HSC_", 1)[-1].replace("_RNA", ""): f for t, f in rna_titles.items()}
    flat = [(name, reg) for name, regs in sets.items() for reg in regs]
    lookup = by_chrom([reg for _, reg in flat])
    vals, keep = [], []
    for r in cells:
        files = [f for f in by_met.get(r["id_met"], []) if f.endswith("CpG.tsv.gz")]
        if not files:
            continue
        try:
            vals.append(region_rates(get(files[0]), lookup))
            keep.append(r)
        except RuntimeError:
            pass
    vals = np.array(vals)
    wanted = {c["id_rna"].rsplit("_RNA_", 1)[1] for c in keep}
    plates = {}
    for r in keep:
        plate = r["id_rna"].rsplit("_RNA_", 1)[0]
        if plate not in plates and by_rna.get(plate):
            plates[plate] = rna_table(get(by_rna[plate][0]), wanted)
    common = sorted(set.intersection(*(set(v[1]) for v in plates.values())))
    gidx = {p: [{x: k for k, x in enumerate(v[1])}[x] for x in common] for p, v in plates.items()}
    rows, mrows, erows = [], [], []
    for r, v in zip(keep, vals, strict=True):
        plate, bc = r["id_rna"].rsplit("_RNA_", 1)
        if plate in plates and bc in plates[plate][0]:
            rows.append(r)
            mrows.append(v)
            erows.append(plates[plate][2][gidx[plate], plates[plate][0].index(bc)])
    m, e = np.array(mrows), np.array(erows)
    e = np.log1p(e / np.maximum(e.sum(1, keepdims=True), 1) * 1e4)
    e = e[:, np.argsort(e.var(0))[-2000:]]
    clone = np.array([r["mouse"] + "|" + r["large_clone_id"] for r in rows])
    group = np.array([r["mouse"] + "|" + r["tissue"] for r in rows])
    mouse = np.array([r["mouse"] for r in rows])
    cols = np.array([name for name, _ in flat])
    out = {"cells": len(rows), "clones": int(len(set(clone)))}
    # B6a
    b6a = {
        s: feature_auc(m[:, cols == s], clone, group, rng)
        for s in ("lineage", "promoters", "gene_bodies", "bins100k")
    }
    rand = [feature_auc(m[:, cols == f"random_{k}"], clone, group, rng)["auc"] for k in range(50)]
    rand = [a for a in rand if a is not None]
    b6a["random_sets"] = {
        "n": len(rand),
        "median": float(np.median(rand)),
        "p95": float(np.percentile(rand, 95)),
    }
    out["b6a"] = b6a
    # B6b: Meg vs nonMeg clones, leave-one-clone-out ridge on clone-mean PCs
    lab = {}
    for r in rows:
        if r["Meg_bias"] in ("Meg", "nonMeg"):
            lab.setdefault(r["mouse"] + "|" + r["large_clone_id"], []).append(
                r["Meg_bias"] == "Meg"
            )
    lab = {c: float(np.mean(v)) > 0.5 for c, v in lab.items()}
    b6b = {"clones": len(lab), "meg_clones": int(sum(lab.values()))}
    mfull = m[:, cols == "promoters"]
    mfull = mfull[:, np.mean(~np.isnan(mfull), 0) >= 0.5]
    mfull = np.where(np.isnan(mfull), np.nanmean(mfull, 1, keepdims=True), mfull)
    for name, f in (("methylation", pcs(mfull, 10)), ("expression", pcs(e, 10))):
        cl = sorted(lab)
        x = np.stack([f[clone == c].mean(0) for c in cl])
        y = np.array([lab[c] for c in cl], float)
        pred = []
        for k in range(len(cl)):
            tr = np.arange(len(cl)) != k
            a = np.c_[x[tr], np.ones(tr.sum())]
            w = np.linalg.solve(a.T @ a + 1.0 * np.eye(a.shape[1]), a.T @ y[tr])
            pred.append(np.r_[x[k], 1] @ w)
        pred = np.array(pred)
        b6b[name] = float(auc(pred[y == 1], pred[y == 0])) if 0 < y.sum() < len(y) else None
    out["b6b"] = b6b
    # B6c: leave one mouse out, 100-kb bins as in B5, and expression PCs
    mfull = m[:, cols == "bins100k"]
    mfull = mfull[:, np.mean(~np.isnan(mfull), 0) >= 0.5]
    mfull = np.where(np.isnan(mfull), np.nanmean(mfull, 1, keepdims=True), mfull)
    b6c = {}
    for mo in sorted(set(mouse)):
        k = mouse != mo
        b6c[mo] = {
            "methylation": score_one(pcs(mfull[k]), clone[k], group[k], rng, boot=200, perms=100),
            "expression": score_one(pcs(e[k]), clone[k], group[k], rng, boot=200, perms=100),
        }
    out["b6c"] = b6c
    os.makedirs(os.path.join(HERE, "results_track_b"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results_track_b", "b6_camellia.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
