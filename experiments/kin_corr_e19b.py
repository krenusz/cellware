"""E19b: E19's kin-correlation ratio in LARRY in vivo (groups = recipient mouse x time point).
See the 2026-11-03 batch pre-registration in EXPERIMENT_LOG.md.

    uv run python experiments/kin_corr_e19b.py
"""

from __future__ import annotations

import gzip
import json
import os

import numpy as np
from kin_corr_e19 import run

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "larry_invivo")


def load():
    """{clone: {(time, mouse): [Neu, Mono]}} for clones seen in >= 2 mice."""
    with gzip.open(os.path.join(DATA, "metadata.txt.gz"), "rt") as f:
        rows = [line.rstrip("\n").split("\t") for line in f][1:]
    with gzip.open(os.path.join(DATA, "clone_matrix.mtx.gz"), "rt") as f:
        m = np.loadtxt(f, comments="%", skiprows=3, dtype=int)
    if m[:, 0].max() < m[:, 1].max():  # make column 0 the cell index
        m = m[:, [1, 0, 2]]
    out = {}
    for cell, clone in zip(m[:, 0] - 1, m[:, 1] - 1, strict=True):
        t, kind, mouse = rows[cell][2], rows[cell][4], rows[cell][5]
        if t == "2" or kind not in ("Neu", "Mono"):
            continue
        k = out.setdefault(int(clone), {}).setdefault((t, mouse), [0, 0])
        k[0 if kind == "Neu" else 1] += 1
    return {c: g for c, g in out.items() if len({w for _, w in g}) >= 2}


def main():
    groups = load()
    out = {
        "clones_in_2plus_mice": len(groups),
        "primary": run(groups, False, np.random.default_rng(0)),
    }
    out["same_time_cross"] = run(groups, True, np.random.default_rng(0))
    json.dump(out, open(os.path.join(HERE, "results_e18", "e19b_invivo_kin.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
