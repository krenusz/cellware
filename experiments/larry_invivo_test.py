"""E6b held-out test: cross-mouse fate concordance in the Weinreb 2020 in vivo data.

Running this is the single pre-registered look at the in vivo test set (EXPERIMENT_LOG.md, E6b).
"""

import gzip
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from larry_stats import test_stats  # noqa: E402

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "larry_invivo")


def main():
    with gzip.open(os.path.join(DATA, "metadata.txt.gz"), "rt") as f:
        rows = [line.rstrip("\n").split("\t") for line in f][1:]
    with gzip.open(os.path.join(DATA, "clone_matrix.mtx.gz"), "rt") as f:
        m = np.loadtxt(f, comments="%", skiprows=3, dtype=int)  # clones x cells
    by_clone = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    for clone, cell in zip(m[:, 0] - 1, m[:, 1] - 1, strict=True):
        day, kind, mouse = rows[cell][2], rows[cell][4], rows[cell][5]
        if day == "2" or kind in ("Undifferentiated", "Prog"):
            continue
        j = 0 if kind == "Neu" else 1 if kind == "Mono" else 2
        by_clone[clone][mouse][j] += 1
    a, b = {}, {}
    for clone, mice in by_clone.items():
        ranked = sorted(mice.values(), key=lambda v: -(v[0] + v[1]))
        if len(ranked) >= 2:
            a[clone], b[clone] = ranked[0], ranked[1]
    result = test_stats(a, b)
    json.dump(result, open(os.path.join(HERE, "results_larry", "invivo_test.json"), "w"), indent=1)
    print("IN VIVO TEST:", result)


if __name__ == "__main__":
    main()
