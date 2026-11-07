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


