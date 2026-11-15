"""C13-S0b format check (H9) of the embryo lineage set (data/ext_petracer, Zenodo 19892784).

Prints structure only: CSV headers with dtypes, rows per embryo id, and the HDF5 group/dataset
tree with shapes and dtypes. Never prints a data row, cell type, spatial score, character state
or tree topology. Run with the venv python and -I (downloaded data).
"""
import collections
import re
import sys
from pathlib import Path

import h5py
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "ext_petracer" / "raw"
EMB = re.compile(r"^E\d+\.\d-R\d$")
NAMES = {f"E{s}-R{r}" for s in ["7.5", "8.0", "8.5", "9.0", "9.5"] for r in (1, 2, 3)} | {"E10.0-R1"}


def embryo_column(path, head):
    """Name of a column whose sampled values are all embryo ids (structural id column)."""
    for c in head.columns:
        if re.search(r"embryo|sample|stage", c, re.I) and head[c].astype(str).str.match(EMB).all():
            return c
    for c in head.columns:
        if head[c].dtype == object and head[c].astype(str).str.match(EMB).all():
            return c
    return None


def csv_report(path):
    head = pd.read_csv(path, nrows=2000)
    print(f"\n== {path.name}: {len(head.columns)} columns")
    for c in head.columns:
        print(f"   {c!r}: {head[c].dtype}")
    col = embryo_column(path, head)
    if col is None:
        # embryo id may be a prefix of the first (index) column
        first = head.columns[0]
        pref = head[first].astype(str).str.extract(r"^(E\d+\.\d-R\d)")[0]
        if pref.notna().all():
            counts = collections.Counter()
            for ch in pd.read_csv(path, usecols=[first], chunksize=500_000):
                counts.update(ch[first].astype(str).str.extract(r"^(E\d+\.\d-R\d)")[0].dropna())
            report_counts(path, f"prefix of {first!r}", counts)
            return
        print("   no embryo id column found: STOP rule (adequacy) applies to this file")
        return
    counts = collections.Counter()
    for ch in pd.read_csv(path, usecols=[col], chunksize=500_000):
        counts.update(ch[col].astype(str))
    report_counts(path, repr(col), counts)


def report_counts(path, how, counts):
    total = sum(counts.values())
    unknown = set(counts) - NAMES
    print(f"   embryo id via {how}: {len(counts)} ids, {total} rows; ids outside the 16 names: {sorted(unknown)[:5]}")
    for e in sorted(counts, key=lambda x: (float(x[1:].split('-')[0]), x)):
        print(f"   {e}: {counts[e]}")


def h5_report(path):
    print(f"\n== {path.name}: HDF5 structure (names, shapes, dtypes only)")
    lines = []

    def visit(name, obj):
        depth = name.count("/")
        if depth > 3:
            return
        if isinstance(obj, h5py.Dataset):
            lines.append(f"   {name}: dataset shape={obj.shape} dtype={obj.dtype}")
        else:
            lines.append(f"   {name}/ (group, {len(obj)} members, attrs={sorted(obj.attrs.keys())[:8]})")

    with h5py.File(path, "r") as f:
        print(f"   root attrs: {sorted(f.attrs.keys())[:10]}; top-level: {sorted(f.keys())}")
        f.visititems(visit)
    for ln in lines[:200]:
        print(ln)
    if len(lines) > 200:
        print(f"   ... {len(lines) - 200} more entries")


def main():
    for p in sorted(RAW.glob("*.csv")):
        csv_report(p)
    for p in sorted(RAW.glob("*.h5td")):
        h5_report(p)


if __name__ == "__main__":
    sys.exit(main())
