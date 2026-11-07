"""G13: flat-dish growth bias vs head-to-tail preference. DiffAssay replicate 1 only. See EXPERIMENT_LOG G13.

    uv run --with openpyxl python g13_diff.py
"""

from __future__ import annotations

import itertools
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from partitions import train_only

HERE = os.path.dirname(os.path.abspath(__file__))
F = os.path.join(HERE, "..", "data", "gastruloid_raj", "diff_assay", "DiffAssay_Oct212024.csv")
OUT = os.path.join(HERE, "results_gastruloid", "g13_diff.json")
P = {1: -0.922, 3: 0.881, 5: 1.051, 6: -1.132, 7: 0.726}  # G1a fit-shard clone means (g7_model.fit_mu)


def exact_p(x, y):
    obs = abs(spearmanr(x, y).statistic)
    perms = list(itertools.permutations(y))
    return sum(abs(spearmanr(x, p).statistic) >= obs - 1e-12 for p in perms) / len(perms)


def main():
    d = train_only("raj_diffassay", pd.read_csv(F), unit_col="replicate")
    last = d[d["time"] == d["time"].max()].pivot_table(index="clone", columns="condition", values="confluency")
    last["meso_vs_neural"] = np.log(last["mesoderm"] / last["neural"])
    last["endo_vs_neural"] = np.log(last["endoderm"] / last["neural"])
    res = {"time": float(d["time"].max()), "n_rows": int(len(d)),
           "per_clone": last.round(3).reset_index().to_dict("records"), "tests": {}}
    clones = sorted(P)
    pv = [P[c] for c in clones]
    for col in ("meso_vs_neural", "endo_vs_neural", "2i"):
        x = [float(last.loc[f"clone {c}", col]) for c in clones]
        ant = [x[i] for i, c in enumerate(clones) if P[c] > 0]
        post = [x[i] for i, c in enumerate(clones) if P[c] < 0]
        sep = min(ant) > max(post) or max(ant) < min(post)
        res["tests"][col] = {"x": dict(zip(map(str, clones), np.round(x, 3).tolist())),
                             "spearman": round(float(spearmanr(x, pv).statistic), 3), "exact_p": round(exact_p(x, pv), 3),
                             "separates_anterior_from_posterior": bool(sep)}
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
