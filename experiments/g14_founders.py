"""G14 pre-flight: fate composition of TRAIN founder clusters (gse291244_founders). Train only.

    uv run --with pyreadr python g14_founders.py
"""

from __future__ import annotations

import os

import pyreadr

from partitions import train_only

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "data", "gastruloid_gse291244", "meta", "sec4")


def train_cells():
    g = pyreadr.read_r(os.path.join(D, "gas_cell_well_cluster.rds"))[None]
    g = train_only("gse291244_founders", g, unit_col="well_cluster")
    pd_ = pyreadr.read_r(os.path.join(D, "gas_obj_pd.rds"))[None][["cell_id", "celltype"]]
    return g.merge(pd_, on="cell_id", how="left")


def main():
    x = train_cells()
    comp = x.groupby(["well_cluster", "celltype"]).size().unstack(fill_value=0)
    share = comp.div(comp.sum(axis=1), axis=0)
    print("train clusters:", len(comp), "cells:", int(comp.values.sum()))
    print(comp.sum(axis=1).to_string())
    print((share.round(2)).to_string())


if __name__ == "__main__":
    main()
