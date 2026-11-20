"""C17-S1G: S0 structure gates for the neural candidates of C17-S1N2 (structure only, no fate read).

Bence 2026-11-20: "Yes, gate both", "gate all 5 actually". Serves: C17-S1 (C17-S1R: grow the systems; neural class).
Pre-registered in EXPERIMENT_LOG (C17-S1G). Lineage and metadata tables only (downloaded with sha256 into
data/ext_zf_dev, data/ext_zf_adult, data/ext_itracer); read with -I. Columns whose names suggest a cell type or fate
are listed by name and type only, never tabulated.

    python -I experiments/c17s1g_gates.py madm                 # G1: clones from GEO sample titles
    python -I experiments/c17s1g_gates.py schema               # G2-G4: file members, columns, types
    python -I experiments/c17s1g_gates.py gate <zf_dev|zf_adult|itracer>
    python -I experiments/c17s1g_gates.py selftest
"""

from __future__ import annotations

import collections
import gzip
import io
import json
import re
import sys
import tarfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUT = HERE / "results_c17"
FATE_LIKE = re.compile(r"type|fate|cluster|annot|class|ident|lineage_?state|celltype|state|leiden|louvain|region",
                       re.I)
ID_LIKE = re.compile(r"barcode|scar|hmid|cigar|location|umi|cellnum|readname|organoid|^sample$|batch|read", re.I)
MIN_ARM, MIN_TRAIN, TRAIN_SHARE = 4, 30, 0.4
UA = {"User-Agent": "cellware-struct/1.0 (public metadata, research use)"}


# --- shared clade picker (G2, G3): disjoint sister-allele pairs inside edit-defined clades -------------------------


def edits_of(allele):
    """Edits of a full allele string; sites joined by '_' (or ';' / ','); unedited sites ('NONE', '') dropped."""
    parts = re.split(r"[_;,]", str(allele))
    return {f"{i}:{p}" for i, p in enumerate(parts) if p and p.upper() not in ("NONE", "NA", "WT", "UNEDITED")}


def hmid_edits(allele):
    """G2 (amendment 1): GESTALT HMID, sites joined by '-'; NONE / UNKNOWN sites dropped."""
    return {f"{i}:{p}" for i, p in enumerate(str(allele).split("-")) if p and p.upper() not in ("NONE", "UNKNOWN")}


def joined_edits(allele):
    """G3 (amendment 1): full allele = '|'-joined edits."""
    return set(str(allele).split("|")) - {""}


def sister_pairs(cell_allele, min_arm=MIN_ARM, edits=None):
    """cell_allele: {cell: full allele} within one animal. Clade = cells carrying an edit present in >= 2 distinct
    alleles and in < 50% of the animal's cells; arms = its two largest distinct alleles; clades smallest first,
    skipped if they share cells with a clade already taken. Returns [(n_arm1, n_arm2), ...] for all taken pairs."""
    by_allele = collections.defaultdict(list)
    for c, a in cell_allele.items():
        by_allele[a].append(c)
    n = len(cell_allele)
    edit_alleles = collections.defaultdict(set)
    for a in by_allele:
        for e in (edits or edits_of)(a):
            edit_alleles[e].add(a)
    clades = []
    for e, al in edit_alleles.items():
        cells = sum(len(by_allele[a]) for a in al)
        if len(al) >= 2 and cells < 0.5 * n:
            clades.append((cells, e, al))
    clades.sort(key=lambda x: (x[0], x[1]))
    used, pairs = set(), []
    for _, _, al in clades:
        cells = {c for a in al for c in by_allele[a]}
        if cells & used:
            continue
        top = sorted(al, key=lambda a: (-len(by_allele[a]), a))[:2]
        used |= cells
        pairs.append((len(by_allele[top[0]]), len(by_allele[top[1]])))
    return pairs


def summarise(pairs_by_animal, recurrent_share=None):
    elig = {k: sum(1 for a, b in p if a >= MIN_ARM and b >= MIN_ARM) for k, p in pairs_by_animal.items()}
    tot = sum(elig.values())
    return {"pairs_taken": {k: len(p) for k, p in pairs_by_animal.items()}, "eligible_by_animal": elig,
            "eligible": tot, "expected_train_eligible": int(TRAIN_SHARE * tot),
            "gate_pass": bool(TRAIN_SHARE * tot >= MIN_TRAIN), "recurrent_allele_share": recurrent_share}


# --- G1 MADM-CloneSeq: clones from sample titles -------------------------------------------------------------------


def madm():
    out = {}
    for acc in ("GSE245102", "GSE245103"):
        url = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?" + urllib.parse.urlencode(
            {"acc": acc, "targ": "gsm", "form": "text", "view": "brief"})
        text = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120).read().decode("utf-8",
                                                                                                          "replace")
        titles = [t.strip() for t in re.findall(r"^!Sample_title = (.+)$", text, flags=re.M)]
        clones = collections.Counter(m.group(1) for t in titles if (m := re.match(r"([A-Za-z]+\d+)\.\d+$", t)))
        out[acc] = {"samples": len(titles), "clones": len(clones), "clones_ge4_cells": sum(v >= 4 for v in
                    clones.values()), "cells_per_clone_median": float(pd.Series(list(clones.values())).median())}
        time.sleep(2)
    bound = sum(v["clones_ge4_cells"] for v in out.values())
    out["upper_bound_eligible"] = bound
    out["gate_pass"] = bool(bound >= 75)
    return out


# --- schema (G2-G4) -----------------------------------------------------------------------------------------------


def cols(df):
    out = []
    for c in df.columns:
        s = df[c]
        e = {"name": str(c), "dtype": str(s.dtype), "n_distinct": int(s.nunique(dropna=True))}
        if FATE_LIKE.search(str(c)):
            e["fate_like"] = "name and type only"
        elif ID_LIKE.search(str(c)) and s.notna().any():  # examples only for ID and lineage columns
            e["example"] = str(s.dropna().iloc[0])[:60]
        out.append(e)
    return {"rows": len(df), "columns": out}


def read_table(fh, name):
    sep = "," if name.endswith((".csv", ".csv.gz")) else "\t"
    raw = fh.read()
    if name.endswith(".gz"):
        raw = gzip.decompress(raw)
    return pd.read_csv(io.BytesIO(raw), sep=sep, low_memory=False)


def schema():
    res = {}
    d = DATA / "ext_zf_dev" / "raw"
    with tarfile.open(d / "GSE105010_RAW.tar") as tf:
        mem = tf.getmembers()
        res["zf_dev_tar_members"] = [(m.name, m.size) for m in mem]
        first = next(m for m in mem if m.isfile())
        res["zf_dev_tar_first"] = cols(read_table(tf.extractfile(first), first.name))
    g = sorted(d.glob("GSE158142_*GestMaster*"))
    res["zf_dev_gestmaster"] = cols(read_table(open(g[0], "rb"), g[0].name))
    a = sorted((DATA / "ext_zf_adult" / "raw").glob("*_filtered_scars.csv.gz"))
    res["zf_adult_scars"] = cols(read_table(open(a[0], "rb"), a[0].name))
    for f in sorted((DATA / "ext_itracer" / "raw").glob("*.tsv.gz")):
        if f.name.startswith(PERTURB):  # amendment 1: perturbation set excluded
            continue
        res[f"itracer_{f.name}"] = cols(read_table(open(f, "rb"), f.name))
    return res


PERTURB = "E-MTAB-10971"


def gate_zf_dev():
    d = DATA / "ext_zf_dev" / "raw"
    tables = {}
    with tarfile.open(d / "GSE105010_RAW.tar") as tf:
        for m in tf.getmembers():
            if m.name.endswith(".GestMaster.txt.gz"):
                tables[m.name.split("_")[1].split(".")[0]] = read_table(tf.extractfile(m), m.name)
    for f in sorted(d.glob("GSE158142_*.GestMaster.txt.gz")):
        tables[f.name.split("_")[1].split(".")[0]] = read_table(open(f, "rb"), f.name)
    pairs, seen = {}, collections.defaultdict(set)
    for fish, t in tables.items():
        t = t[t["HMID"].notna()]
        ca = dict(zip(t["CellBarcode"].astype(str), t["HMID"].astype(str), strict=True))
        pairs[fish] = sister_pairs(ca, edits=hmid_edits)
        for a in set(ca.values()):
            seen[a].add(fish)
    rec = sum(len(v) > 1 for v in seen.values()) / max(len(seen), 1)
    return {"animals": len(tables), "cells": {k: int(t["HMID"].notna().sum()) for k, t in tables.items()},
            **summarise(pairs, round(rec, 3))}


def gate_zf_adult():
    pairs, seen, called = {}, collections.defaultdict(set), {}
    for rep in ("lin1", "lin2"):
        per_cell = collections.defaultdict(set)
        for f in sorted((DATA / "ext_zf_adult" / "raw").glob(f"*_{rep}_*_filtered_scars.csv.gz")):
            site = f.name.split("_")[2]
            t = pd.read_csv(f, sep="\t")
            t = t[(t["Reads"] >= 2) & t["CIGAR"].astype(str).str.contains(r"[ID]")]
            for cell, loc, cig in zip(t["Barcode"], t["Location"], t["CIGAR"], strict=True):
                per_cell[str(cell)].add(f"{site}:{loc}:{cig}")
        ca = {c: "|".join(sorted(e)) for c, e in per_cell.items() if len({x.split(":")[0] for x in e}) >= 2}
        called[rep] = len(ca)
        pairs[rep] = sister_pairs(ca, edits=joined_edits)
        for a in set(ca.values()):
            seen[a].add(rep)
    rec = sum(len(v) > 1 for v in seen.values()) / max(len(seen), 1)
    return {"animals": 2, "cells_called": called, **summarise(pairs, round(rec, 3))}


def unedited(scar):
    return all(re.fullmatch(r"\d+:\d+M", s) for s in str(scar).split("_"))


def gate_itracer():
    specs = [("E-MTAB-10973_meta.tsv.gz", "orig.ident", "GeneBarcodeMerge", "ScarMerge", None),
             ("E-MTAB-10974_meta_org1-12.tsv.gz", "organoid", "GeneBarcodeMerge", "ScarMerge", "genotype"),
             ("E-MTAB-10974_meta_org_d15.tsv.gz", "sample", "GeneBarcodeMerged", "ScarMerged", None)]
    per_org, design = collections.defaultdict(list), {}
    for fname, org, bc, scar, geno in specs:
        t = read_table(open(DATA / "ext_itracer" / "raw" / fname, "rb"), fname)
        if geno:  # design label: keep the most common level only
            top = t[geno].value_counts().idxmax()
            design[fname] = {"genotype_levels": int(t[geno].nunique()), "kept_share": round(float((t[geno] == top)
                                                                                                    .mean()), 3)}
            t = t[t[geno] == top]
        t = t[t[bc].notna() & t[scar].notna()]
        for (o, b), g in t.groupby([org, bc]):
            al = g[scar][~g[scar].map(unedited)].value_counts()
            if len(al) >= 2:
                per_org[f"{fname.split('_')[0]}|{o}"].append((int(al.iloc[0]), int(al.iloc[1])))
            else:
                per_org[f"{fname.split('_')[0]}|{o}"].append((int(al.iloc[0]) if len(al) else 0, 0))
    return {"design": design, "organoids": len(per_org), **summarise(dict(per_org))}


def selftest():
    toy = {f"c{i}": a for i, a in enumerate(
        ["F_1:X_NONE"] * 30                         # founder edit 0:F in > 50% when combined below
        + ["F_2:Y_3:Z"] * 5 + ["F_2:Y_4:W"] * 4      # clade 1:2:Y with two alleles (5, 4)
        + ["F_5:Q_6:R"] * 6 + ["F_5:Q_7:S"] * 4)}   # clade 1:5:Q with two alleles (6, 4)
    p = sister_pairs(toy)
    assert sorted(p) == [(5, 4), (6, 4)], p
    s = summarise({"a": p})
    assert s["eligible"] == 2 and not s["gate_pass"], s
    assert edits_of("NONE_3D+2_NONE") == {"1:3D+2"}
    assert hmid_edits("45D+140-NONE-UNKNOWN-1I+331+T") == {"0:45D+140", "3:1I+331+T"}
    assert unedited("0:91M_0:91M") and not unedited("0:44M8D47M_0:91M")
    return {"ok": True}


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if stage == "selftest":
        print(json.dumps(selftest()))
        return
    if stage == "gate":
        cand = sys.argv[2]
        res = {"zf_dev": gate_zf_dev, "zf_adult": gate_zf_adult, "itracer": gate_itracer}[cand]()
        stage = f"gate_{cand}"
    else:
        res = madm() if stage == "madm" else schema()
    OUT.mkdir(exist_ok=True)
    json.dump(res, open(OUT / f"c17s1g_{stage}.json", "w", encoding="utf-8"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
