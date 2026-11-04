"""B25: is the low M-CSF replicate-library agreement explained by sampling alone?
See the B25 pre-registration in EXPERIMENT_LOG.md.

    uv run python experiments/track_b25.py
"""

from __future__ import annotations

import itertools
import json
import os

import numpy as np
from track_b21 import OUT, by_library, pairs, r_rows

NULL = 1000


def rows_for(libs):
    rows = {}
    for a, b in itertools.combinations(libs, 2):
        for k, q in pairs(a, b).items():
            rows.setdefault(k, []).extend(q)
    return rows


def resplit(libs, rng):
    """Shuffle each clone's N/M labels across its libraries, keeping per-library cell counts."""
    out = [dict() for _ in libs]
    for c in set().union(*libs):
        sizes = [sum(lib.get(c, [0, 0])[:2]) for lib in libs]
        n_tot = sum(lib.get(c, [0, 0])[0] for lib in libs)
        labels = np.zeros(sum(sizes), int)
        labels[:n_tot] = 1
        rng.shuffle(labels)
        i = 0
        for j, s in enumerate(sizes):
            if s:
                n = int(labels[i : i + s].sum())
                out[j][c] = [n, s - n]
                i += s
    return out


def main():
    lib = by_library()
    groups = {}
    for (c, d, _name), v in lib.items():
        groups.setdefault((c, d), []).append({k: x[:2] for k, x in v.items()})
    rng = np.random.default_rng(0)
    rep = {}
    for (c, d), libs in sorted(groups.items()):
        if len(libs) < 2:
            continue
        rows = rows_for(libs)
        obs = r_rows(rows.values())
        null = np.array([r_rows(rows_for(resplit(libs, rng)).values()) for _ in range(NULL)])
        null = null[np.isfinite(null)]
        lo, hi = np.percentile(null, [2.5, 97.5]) if len(null) else (np.nan, np.nan)
        p = [q for v in rows.values() for q in v]
        rep[f"{c}{d}"] = {
            "clones": len(rows),
            "observed_r": obs,
            "null_mean": float(null.mean()) if len(null) else None,
            "null_95": [float(lo), float(hi)],
            "below_null": bool(np.isfinite(obs) and obs < lo),
            "mean_Nshare": float(np.mean([a for a, _ in p])) if p else None,
        }
    m6 = rep.get("M6", {})
    rep["reading"] = (
        "extra M-CSF library noise: down-weight"
        if m6.get("below_null")
        else "sampling explains it: no down-weighting"
    )
    json.dump(rep, open(os.path.join(OUT, "b25_mreplicate.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
