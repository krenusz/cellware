"""B32: are the 2-cell well-private clones part of the shared clone pool? Presence elsewhere
(vs E48), day-2 presence, and library structure. See the B32 pre-registration.

    uv run python experiments/b32_small_clones.py
"""

from __future__ import annotations

import glob
import gzip
import json
import os
from collections import defaultdict

import e26_selection as e26
import e32_multinomial as e32
import e33_measured as e33
import e34_fullarm as e34
import e48_capsel6 as e48
import numpy as np
import pandas as pd
from clone_state_model import OUT, split
from fit_cytokine import DATA, pooled
from sweep_guard import gated
from test_oracle import shard_of

SIM_CONDS = ("full", "G", "M", "Epo", "SCF")
GROUPS = (("2", 2, 2), ("3-5", 3, 5))
RNG = np.random.default_rng(32)


def presence_elsewhere(cnt):
    p = {c: pooled(cnt[c]) for c in SIM_CONDS}
    size = {c: {k: v[0] + v[1] for k, v in p[c].items()} for c in SIM_CONDS}
    seen = {c: {k for k, s in size[c].items() if s >= 1} for c in SIM_CONDS}
    out = {}
    for src in ("M", "G"):
        others = [c for c in SIM_CONDS if c != src]
        for lab, lo, hi in GROUPS:
            ks = [k for k, s in size[src].items() if lo <= s <= hi]
            out[f"{src}{lab}_n"] = len(ks)
            for c in others:
                out[f"{src}{lab}_in_{c}"] = float(np.mean([k in seen[c] for k in ks])) if ks else np.nan
            out[f"{src}{lab}_in_any"] = (
                float(np.mean([any(k in seen[c] for c in others) for k in ks])) if ks else np.nan
            )
    return out, size


def day2_share(size, n2):
    res = {}
    for src in ("M", "G"):
        grp = {
            lab: np.array([n2.get(int(k), 0) > 0 for k, s in size[src].items() if lo <= s <= hi])
            for lab, lo, hi in GROUPS
        }
        diffs = [
            RNG.choice(grp["2"], len(grp["2"])).mean() - RNG.choice(grp["3-5"], len(grp["3-5"])).mean()
            for _ in range(2000)
        ]
        res[src] = {
            "size2": float(grp["2"].mean()),
            "size3-5": float(grp["3-5"].mean()),
            "n": [len(grp["2"]), len(grp["3-5"])],
            "diff_95": np.percentile(diffs, [2.5, 97.5]).tolist(),
        }
    return res


def library_structure():
    with gzip.open(os.path.join(DATA, "metadata.txt.gz"), "rt") as f:
        rows = [line.rstrip("\n").split("\t") for line in f][1:]
    with gzip.open(os.path.join(DATA, "clone_matrix.mtx.gz"), "rt") as f:
        m = np.loadtxt(f, comments="%", skiprows=3, dtype=int)
    cells = defaultdict(list)  # (cond, day, clone) -> libraries of its N/M cells
    libsize = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if r[2] != "2" and r[3] in ("Neu", "Mo"):
            libsize[(r[4], r[2])][r[0]] += 1
    for clone, cell in zip(m[:, 0] - 1, m[:, 1] - 1):
        r = rows[cell]
        if r[2] == "2" or r[3] not in ("Neu", "Mo") or shard_of(int(clone), 2) == 1:
            continue
        cells[(r[4], r[2], int(clone))].append(r[0])
    res = {}
    for cond in ("M-csf", "G-csf"):
        two, pairs, exp = [], [], []
        for (c, d, k), libs in cells.items():
            if c != cond:
                continue
            ls = libsize[(c, d)]
            tot = sum(ls.values())
            if len(libs) == 2:
                two.append(libs[0] == libs[1])
                exp.append(sum((v / tot) ** 2 for v in ls.values()))
            elif 3 <= len(libs) <= 5:
                i, j = RNG.choice(len(libs), 2, replace=False)
                pairs.append(libs[i] == libs[j])
        res[cond] = {
            "n_two": len(two),
            "same_lib_two": float(np.mean(two)) if two else np.nan,
            "expected_by_size": float(np.mean(exp)) if exp else np.nan,
            "same_lib_pairs_3to5": float(np.mean(pairs)) if pairs else np.nan,
            "n_3to5": len(pairs),
        }
    return res


def main(nsim=200):
    e34.e33.unpack_base = e33.unpack
    cnt = e33.load_counts(e33.CONDS5)
    fit = split(cnt, None)
    co = e32.copresence(fit)
    t = pd.read_csv(os.path.join(OUT, "b14_clone_day2.csv"), index_col=0)
    n2 = {int(k): int(v) for k, v in t["n2"].items()}
    data, size = presence_elsewhere(fit)
    res = [json.load(open(f)) for f in glob.glob(os.path.join(OUT, "e48_s*.json"))]
    best = min((r for r in res if gated(r, co)), key=lambda r: r["loss"])
    clones = e26.universe(cnt)
    meas = e33.measured(clones, np.random.default_rng(77))
    sims = [
        presence_elsewhere(
            split(e48.simulate(clones, best["params"], np.random.default_rng(1000 + s), meas), None)
        )[0]
        for s in range(nsim)
    ]
    comp = {}
    for k, v in data.items():
        x = np.array([s[k] for s in sims], float)
        x = x[np.isfinite(x)]
        lo, hi = np.percentile(x, [2.5, 97.5])
        comp[k] = {"data": v, "sim_mean": float(x.mean()), "sim_95": [float(lo), float(hi)],
                   "inside": bool(lo <= v <= hi)}
    rep = {"e48_seed": best["seed"], "presence": comp, "day2": day2_share(size, n2),
           "library": library_structure()}
    json.dump(rep, open(os.path.join(OUT, "b32_small_clones.json"), "w"), indent=1)
    for k, v in comp.items():
        if k.endswith("_n"):
            print(f"{k:16s} data {v['data']:.0f} (e48 mean {v['sim_mean']:.0f})")
        else:
            flag = "in " if v["inside"] else "OUT"
            print(f"{k:16s} data {v['data']:.3f} | e48 {v['sim_mean']:.3f} "
                  f"[{v['sim_95'][0]:.2f},{v['sim_95'][1]:.2f}] {flag}")
    print(json.dumps({"day2": rep["day2"], "library": rep["library"]}, indent=1))


if __name__ == "__main__":
    main()
