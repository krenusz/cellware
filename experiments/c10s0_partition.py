"""C10-S0: ReSisTrace (GSE223003) lineage partition + PRE-SIDE structural counts only.

Serves: C10-S0 (access and structure scan, H19 step 1). Bence: "Yes, read + partition" (2026-11-13).

Unit: pre-treatment lineage, keyed "<sample>:s<sisters>" when the cell has a sister group, else
"<sample>:c<cell barcode>". The key never uses an outcome column. Post-treatment cells join a lineage
later through `prer_r_group` (an outcome-linked family ID); that join is NOT done here.

Reads only: cell barcode, `sisters`, and whether `drugSens` is missing (= lineage label detected).
The values of `drugSens` and `prer_r_group` are never loaded, tabulated or printed. Post tables are not opened.
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


if __name__ == "__main__":
    main()
