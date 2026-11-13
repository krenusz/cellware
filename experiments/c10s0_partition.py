"""C10-S0: ReSisTrace (GSE223003) lineage partition + PRE-SIDE structural counts only.

Serves: C10-S0 (access and structure scan, H19 step 1). Bence: "Yes, read + partition" (2026-11-13).

Unit: pre-treatment lineage, keyed "<sample>:s<sisters>" when the cell has a sister group, else
"<sample>:c<cell barcode>". The key never uses an outcome column. Post-treatment cells join a lineage
later through `prer_r_group` (an outcome-linked family ID); that join is NOT done here.

main(): reads only cell barcode, `sisters`, and whether `drugSens` is missing (= lineage label detected);
no outcome value; post tables not opened.
train_links() / train_post() (amendments 1-2): read `prer_r_group` presence on TRAIN lineages only
(partitions.train_only); sealed thirds never touched.
Output: per sample x partition, labelled pre cells, lineage units, sister groups (>= 2 cells).
"""
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # run with python -I (downloaded data), so add the script dir explicitly

import pandas as pd  # noqa: E402

import partitions  # noqa: E402

DS = "resistrace"
DATA = os.path.join(HERE, "..", "data", "ext_resistrace", "processed")
OUT = os.path.join(HERE, "results", "c10s0_counts.json")


def pre_units(path):
    sample = re.search(r"ReSisTrace_(\w+?_\d)_pre", path).group(1)
    df = pd.read_csv(path, sep="\t", index_col=0, usecols=lambda c: c in ("", "Unnamed: 0", "sisters", "drugSens"))
    # Authors' Preprocess Rmd: drugSens NA = "No labels are found in this cell"; non-NA = pre-resistant OR
    # pre-sensitive (both labelled). So notna() = label detected, independent of outcome; value never read.
    labelled = df["drugSens"].notna()
    df = df.loc[labelled, ["sisters"]]
    key = [f"{sample}:s{int(s)}" if pd.notna(s) else f"{sample}:c{bc}" for bc, s in zip(df.index, df["sisters"])]
    df = df.assign(unit=key)
    df["part"] = df["unit"].map(lambda u: partitions.partition_of(DS, u))
    return sample, df


def main():
    files = sorted(glob.glob(os.path.join(DATA, "*_pre_treatment_cell_info.tsv.gz")))
    assert len(files) == 8, files
    out = {}
    for f in files:
        sample, df = pre_units(f)
        rows = {}
        for part, g in df.groupby("part"):
            size = g.groupby("unit").size()
            rows[part] = {"labelled_pre_cells": int(len(g)), "lineage_units": int(len(size)),
                          "sister_groups_ge2": int((size >= 2).sum())}
        out[sample] = rows
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1, sort_keys=True)
    print(f"{'sample':16s} {'part':6s} {'cells':>6s} {'units':>6s} {'sis>=2':>6s}")
    for s, rows in out.items():
        for p in ("train", "A", "B", "C"):
            r = rows.get(p, {"labelled_pre_cells": 0, "lineage_units": 0, "sister_groups_ge2": 0})
            print(f"{s:16s} {p:6s} {r['labelled_pre_cells']:6d} {r['lineage_units']:6d} {r['sister_groups_ge2']:6d}")


def train_links():
    """Amendment 1 (2026-11-13): TRAIN ONLY. Lineages with a post-treatment link (any pre cell with a
    prer_r_group ID = perfect-match pre-resistant family) vs without, per sample. H21 side counts on train."""
    out = {}
    for f in sorted(glob.glob(os.path.join(DATA, "*_pre_treatment_cell_info.tsv.gz"))):
        sample = re.search(r"ReSisTrace_(\w+?_\d)_pre", f).group(1)
        df = pd.read_csv(f, sep="\t", index_col=0,
                         usecols=lambda c: c in ("", "Unnamed: 0", "sisters", "drugSens", "prer_r_group"))
        df = df[df["drugSens"].notna()]
        df["unit"] = [f"{sample}:s{int(s)}" if pd.notna(s) else f"{sample}:c{bc}"
                      for bc, s in zip(df.index, df["sisters"])]
        tr = partitions.train_only(DS, df, "unit")
        linked = tr.groupby("unit")["prer_r_group"].apply(lambda x: x.notna().any())
        multi = tr.groupby("unit").size() >= 2
        out[sample] = {"train_units": int(len(linked)), "linked": int(linked.sum()),
                       "unlinked": int((~linked).sum()), "linked_with_ge2_pre_sisters": int((linked & multi).sum())}
    path = os.path.join(HERE, "results", "c10s0_train_links.json")
    json.dump(out, open(path, "w"), indent=1, sort_keys=True)
    for s, r in out.items():
        print(s, r)


def train_post():
    """Amendment 2 (2026-11-13): TRAIN ONLY. Post cells per linked family. Each prer_r_group ID must map to
    exactly one pre lineage key; post cells without an ID are dropped before counting (never read further)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(DATA, "*_pre_treatment_cell_info.tsv.gz"))):
        sample = re.search(r"ReSisTrace_(\w+?_\d)_pre", f).group(1)
        pre = pd.read_csv(f, sep="\t", index_col=0,
                          usecols=lambda c: c in ("", "Unnamed: 0", "sisters", "drugSens", "prer_r_group"))
        pre = pre[pre["drugSens"].notna() & pre["prer_r_group"].notna()]
        pre["unit"] = [f"{sample}:s{int(s)}" if pd.notna(s) else f"{sample}:c{bc}"
                       for bc, s in zip(pre.index, pre["sisters"])]
        fam2unit = pre.groupby("prer_r_group")["unit"].unique()
        assert (fam2unit.map(len) == 1).all(), f"{sample}: a family maps to >1 lineage"
        fam2unit = fam2unit.map(lambda u: u[0])
        train_fams = {fam for fam, u in fam2unit.items() if partitions.partition_of(DS, u) == "train"}
        post_f = f.replace("_pre_treatment_", "_post_treatment_").replace(
            re.search(r"GSM\d+", f).group(0), "GSM*")
        post_f = glob.glob(post_f)[0]
        post = pd.read_csv(post_f, sep="\t", index_col=0, usecols=lambda c: c in ("", "Unnamed: 0", "prer_r_group"))
        post = post[post["prer_r_group"].notna()]
        post = post[post["prer_r_group"].isin(train_fams)]
        n = post.groupby("prer_r_group").size().reindex(sorted(train_fams), fill_value=0)
        out[sample] = {"train_families": int(len(train_fams)), "post_cells": int(n.sum()),
                       "families_ge2_post": int((n >= 2).sum()), "families_ge5_post": int((n >= 5).sum()),
                       "median_post_per_family": float(n.median()) if len(n) else 0.0}
    path = os.path.join(HERE, "results", "c10s0_train_post.json")
    json.dump(out, open(path, "w"), indent=1, sort_keys=True)
    for s, r in out.items():
        print(s, r)


if __name__ == "__main__":
    if sys.argv[1:] == ["train_post"]:
        train_post()
    elif sys.argv[1:] == ["train_links"]:
        train_links()
    else:
        main()
