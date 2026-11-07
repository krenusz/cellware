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
    "gastruloid_gse291244": {
        "salt": "cellware-gse291244-v1",
        "unit": "sci-RNA-seq3 cells: plate + RT barcode well (one gastruloid per RT well, to be "
                "confirmed from the GEO sample sheet BEFORE any statistic; if the true unit is "
                "coarser, re-partition first); 10x cells (tot_cell_annotation): experiment_id",
        "sealed_on": "2026-11-07",
    },
    "raj_wnt": {
        "salt": "cellware-rajwnt-v1",
        "unit": "Annotation ID (one line scan = one gastruloid), Wnt hi/lo files of Apr/May 2025",
        "files": ["clone1_Wnt_04262025.csv", "clone3_Wnt_04262025.csv"],
        "sealed_on": "2026-11-07",
    },
}
BUCKETS = 5
TRAIN = (0, 1)
SEALED = {"A": 2, "B": 3, "C": 4}


@functools.lru_cache(maxsize=None)
def bucket(dataset, unit):
    salt = PARTITIONED[dataset]["salt"]
    return int(hashlib.sha256(f"{salt}:{unit}".encode()).hexdigest(), 16) % BUCKETS


def partition_of(dataset, unit):
    b = bucket(dataset, unit)
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


def gse291244_unit(cell_id):
    """Unit id from a cell name. sci-RNA-seq3: 'P1-01A.LIG-P1-F03_RT-P3-H02' -> 'RT-P3-H02' joined
    with its plate column by the caller; 10x: 'mGASv5_1_<barcode>' -> 'mGASv5_1'."""
    if "_RT-" in cell_id:
        return cell_id.split("_RT-")[1]
    return cell_id.rsplit("_", 1)[0]
