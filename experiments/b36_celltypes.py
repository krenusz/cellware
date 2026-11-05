"""B36: Epo/SCF sharing with G/M clones, counting N+M only vs any cell type, against E48.
See the B36 pre-registration in EXPERIMENT_LOG.md.

    uv run python experiments/b36_celltypes.py
"""

from __future__ import annotations

import glob
import gzip
import json
import os
from collections import Counter, defaultdict

import e26_selection as e26
import e32_multinomial as e32
import e33_measured as e33
import e34_fullarm as e34
import e48_capsel6 as e48
import numpy as np
from clone_state_model import OUT, split
from fit_cytokine import DATA, pooled
from sweep_guard import gated
from test_oracle import shard_of

COND = {"G-csf": "G", "M-csf": "M", "Epo": "Epo", "SCF": "SCF", "full cocktail": "full"}


def any_cell_presence():
    with gzip.open(os.path.join(DATA, "metadata.txt.gz"), "rt") as f:
        rows = [line.rstrip("\n").split("\t") for line in f][1:]
    with gzip.open(os.path.join(DATA, "clone_matrix.mtx.gz"), "rt") as f:
        m = np.loadtxt(f, comments="%", skiprows=3, dtype=int)
    anyc = defaultdict(set)
    types = defaultdict(Counter)  # (cond, clone) -> cell types
    for clone, cell in zip(m[:, 0] - 1, m[:, 1] - 1):
        r = rows[cell]
        if r[2] == "2" or r[4] not in COND or shard_of(int(clone), 2) == 1:
            continue
        c = COND[r[4]]
        anyc[c].add(int(clone))
        types[(c, int(clone))][r[3]] += 1
    return anyc, types


def nm_stats(cnt):
    p = {c: pooled(cnt[c]) for c in ("G", "M", "Epo", "SCF")}
    src = {c: {k for k, v in p[c].items() if v[0] + v[1] >= 2} for c in ("G", "M")}
    seen = {c: {k for k, v in p[c].items() if v[0] + v[1] >= 1} for c in ("Epo", "SCF")}
    return {f"{s}_in_{d}": float(np.mean([k in seen[d] for k in src[s]])) for s in src for d in seen}, src


def main(nsim=200):
    e34.e33.unpack_base = e33.unpack
    cnt = e33.load_counts(e33.CONDS5)
    fit = split(cnt, None)
    co = e32.copresence(fit)
    nm, src = nm_stats(fit)
    anyc, types = any_cell_presence()
    allt = {f"{s}_in_{d}": float(np.mean([int(k) in anyc[d] for k in src[s]])) for s in src for d in ("Epo", "SCF")}
    res = [json.load(open(f)) for f in glob.glob(os.path.join(OUT, "e48_s*.json"))]
    best = min((r for r in res if gated(r, co)), key=lambda r: r["loss"])
    clones = e26.universe(cnt)
    meas = e33.measured(clones, np.random.default_rng(77))
    sims = [nm_stats(split(e48.simulate(clones, best["params"], np.random.default_rng(1000 + s), meas), None))[0]
            for s in range(nsim)]
    out = {}
    for k in nm:
        x = np.array([s[k] for s in sims])
        lo, hi = np.percentile(x, [2.5, 97.5])
        gap = x.mean() - nm[k]
        closed = (allt[k] - nm[k]) / gap if gap > 0 else np.nan
        out[k] = {"data_NM": nm[k], "data_any": allt[k], "e48_mean": float(x.mean()), "e48_95": [float(lo), float(hi)],
                  "any_inside": bool(lo <= allt[k] <= hi), "gap_closed": float(closed),
                  "n_source": len(src[k.split("_")[0]])}
    comp = Counter()
    for s in ("G", "M"):
        for k in src[s]:
            comp.update(types.get(("Epo", int(k)), Counter()))
    rep = {"e48_seed": best["seed"], "sharing": out, "epo_types_in_shared_clones": dict(comp.most_common())}
    for d in ("Epo", "SCF"):
        ins = [out[f"{s}_in_{d}"]["any_inside"] for s in ("G", "M")]
        half = [out[f"{s}_in_{d}"]["gap_closed"] >= 0.5 for s in ("G", "M")]
        rep[f"reading_{d}"] = "counting effect" if all(ins) else "partial" if all(half) else "no"
    json.dump(rep, open(os.path.join(OUT, "b36_celltypes.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
