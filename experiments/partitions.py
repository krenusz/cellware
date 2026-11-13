"""Partitioned hold-out (rules H11-H13, program.md).

Each partitioned data set is split by a salted hash of its biological unit into 5 buckets:
buckets 0-1 = train (40%, the only data analysis code may read), buckets 2, 3, 4 = sealed
partitions A, B, C (20% each). One oracle look spends one partition; the last unspent partition
needs Bence's explicit OK. Partitions are fixed before any statistic and never redrawn.

    from partitions import train_only
    df = train_only("gastruloid_gse291244", df, unit_col="unit")
"""

from __future__ import annotations

import functools
import hashlib

PARTITIONED = {
    # "gastruloid_gse291244" (sealed 2026-11-07 by RT well / 10x lane) is VOID: the paper pools and
    # dissociates gastruloids before RT, so an RT well or lane is not a gastruloid and every gastruloid
    # fell into every partition. No statistic was ever computed on it. Replaced by the entries below.
    "gse291244_founders": {
        "salt": "cellware-gse291244-founders-v2",
        "unit": "founder clone cluster (`well_cluster`, authors' Debris-seq clustering; 28 clusters in "
                "multiome/monoclonal_gas/cell_well_cluster_assignment.rds and multiome/"
                "cell_well_cluster_assignment.rds; gastruloid cells and mESC cells of a cluster go together)",
        "sealed_on": "2026-11-07",
        "prior_exposure": "Regalado/Qiu Cell 2026 Fig. 5 read: published DE (Tet1, Meg3/Rian, Peg10 in "
                          "PSC-biased subclones) and fate-bias categories of the 26 clusters with >= 200 cells",
    },
    "raj_diffassay": {
        "fixed": {"1": "train", "2": "A"},
        "files": ["DiffAssay_Oct212024.csv"],  # 2 replicates only: replicate 1 train, replicate 2 = one sealed partition
        "unit": "replicate (DiffAssay_Oct212024.csv: 9 clones x 4 conditions x 4 times x 2 replicates; "
                "too small for 4 hash partitions)",
        "sealed_on": "2026-11-07",
    },
    "raj_wnt": {
        "salt": "cellware-rajwnt-v1",
        "unit": "Annotation ID (one line scan = one gastruloid), Wnt hi/lo files of Apr/May 2025",
        "files": ["clone1_Wnt_04262025.csv", "clone3_Wnt_04262025.csv"],
        "sealed_on": "2026-11-07",
    },
    "raj_morph_0831": {
        "salt": "cellware-rajmorph0831-v1",
        "buckets": 3,  # mixtures only, thirds A/B/C; the pure plate (bulk_5s2_6a.json) is the train data
        "unit": "Nimbus annotation _id (one polygon = one gastruloid = one well) in "
                "morph_08312026/bulk+5s2_bulk+6a_5s2+6a.json (5s2+6a 27, 5s2+bulk 14, 6a+bulk 29)",
        "files": ["bulk+5s2_bulk+6a_5s2+6a.json"],
        "sealed_on": "2026-11-07",
        "prior_exposure": "tags and counts per tag/well only; pure plate unread until C4-S2a pre-registration",
    },
    "raj_atac": {
        "fixed": {"0": "train", "1": "train", "2": "A", "3": "B"},  # forecasting split by passage; B = last (H13)
        "unit": "passage (atac_peaks/Sorted_V*_rg_nodup_*.narrowPeak; sample sheet atac_meta/align.txt: "
                "V# -> clone, replicate, passage)",
        "files": ["atac_peaks/ passages 2-3 (V33-V64), locked per file in sealed.DATASETS"],
        "sealed_on": "2026-11-07",
        "prior_exposure": "sample sheet, file names and sizes; paper's peak totals (p1 44,540; p3 41,593) and "
                          "passage-1 anterior/posterior shared-peak shares",
    },
    "celltag_lsk": {
        "salt": "cellware-celltag-lsk-v1",
        "unit": "clone (`clone.id` in data/ext_celltag_multi/hsc.rna_atac.r1_2_master_v2.csv, 6,822 clones; "
                "all siblings of a clone, d2.5 RNA/ATAC and d5 r1/r2, go together)",
        "files": ["hsc.rna_atac.r1_2_master_v2.csv"],
        "sealed_on": "2026-11-11",
        "prior_exposure": "column names and structural counts (day, assay, sibling counts) only; fate columns unread; "
                          "Jindal 2024 methods read, results/figures not read",
    },
    "celltag_reprog": {
        "salt": "cellware-celltag-reprog-v1",
        "unit": "clone (`clone_id` in data/ext_celltag_multi/b4.rna_atac.r1_2_master_rev1_annotated.csv, 8,050 clones)",
        "files": ["b4.rna_atac.r1_2_master_rev1_annotated.csv"],
        "sealed_on": "2026-11-11",
        "prior_exposure": "as celltag_lsk",
    },
    "chspctrack": {
        "salt": "cellware-chspctrack-v1",
        "unit": "clone (`clone_id` in data/ext_chspctrack/cells_with_clone_info.h5ad obs, 13,700 clones; "
                "human CD34+ single-cell-derived clones, serial half-well sampling d3-d12, 8 cytokine conditions)",
        "files": ["cells_with_clone_info.h5ad", "metacells.h5ad"],
        "sealed_on": "2026-11-11",
        "prior_exposure": "obs column names, cells per clone per day / well / condition only; cell_type and "
                          "pseudotime / score columns unread; Blood 2026 abstract read (clones diversify "
                          "stochastically, cytokines bias; clonal memory stabilises a few genes per clone)",
    },
    "resistrace": {
        "salt": "cellware-resistrace-v1",
        "unit": "pre-treatment lineage in GSE223003 processed cell_info (data/ext_resistrace/processed): key "
                "'<arm>_<rep>:s<sisters>' for cells with a sister group, else '<arm>_<rep>:c<cell barcode>'; "
                "only cells with a detected label (drugSens not missing). Post-treatment cells inherit the "
                "partition of their pre lineage through prer_r_group (outcome-linked; join only in train or oracle)",
        "files": ["16 *_cell_info.tsv.gz (8 pre, 8 post), sha256 in data/ext_resistrace/SHA256SUMS_processed"],
        "sealed_on": "2026-11-13",
        "prior_exposure": "column names; per-sample cell counts; via a 2026-11-12 column-shift bug, per-sample number "
                          "aggregates: distinct prer_r_group IDs (pre-resistant families; whether singletons were included is unverified), cells carrying prer_r_group (perfect-match pre-resistant cells), family size spread, and post cells carrying prer_r_group (matched post cells); all pooled across partitions (outcome aggregates) "
                          "and two example rows; no per-lineage outcome; Dai 2024 full text read "
                          "(design, label accuracy, subclone A enrichment reported in the paper)",
    },
}
BUCKETS = 5
TRAIN = (0, 1)
SEALED = {"A": 2, "B": 3, "C": 4}


@functools.lru_cache(maxsize=None)
def bucket(dataset, unit):
    salt = PARTITIONED[dataset]["salt"]
    return int(hashlib.sha256(f"{salt}:{unit}".encode()).hexdigest(), 16) % PARTITIONED[dataset].get("buckets", BUCKETS)


def partition_of(dataset, unit):
    fixed = PARTITIONED[dataset].get("fixed")
    if fixed is not None:
        return fixed[str(unit)]
    b = bucket(dataset, unit)
    if PARTITIONED[dataset].get("buckets") == 3:
        return "ABC"[b]
    if b in TRAIN:
        return "train"
    return next(k for k, v in SEALED.items() if v == b)


def train_only(dataset, df, unit_col):
    """Rows of df whose unit is in the train partition. The only allowed way to read a
    partitioned data set outside test_oracle."""
    keep = df[unit_col].map(lambda u: partition_of(dataset, u) == "train")
    return df[keep]


def sealed_rows(dataset, df, unit_col, part):
    """Rows of one sealed partition. Call ONLY from an oracle test function (check_sealed enforces)."""
    assert part in SEALED, part
    keep = df[unit_col].map(lambda u: partition_of(dataset, u) == part)
    return df[keep]


