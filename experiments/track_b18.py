"""B18: does cross-condition agreement fall with the gap in sampling day?
See the B18 pre-registration in EXPERIMENT_LOG.md. Full, G-CSF and M-CSF only (no Epo/SCF).

    uv run python experiments/track_b18.py
"""

from __future__ import annotations

import json
import os

import numpy as np
from clone_state_model import OUT
from fit_cytokine import CONDS, load_counts
from larry_stats import test_stats

BOOT = 2000


def share(v):
    return v[0] / (v[0] + v[1]) if v[0] + v[1] >= 2 else None


def rows(cnt, pairs):
    """{clone: [(x, y), ...]} over the given ((cond, day), (cond, day)) pairs."""
    out = {}
    for (a, da), (b, db) in pairs:
        for c in set(cnt[a][da]) & set(cnt[b][db]):
            x, y = share(cnt[a][da][c]), share(cnt[b][db][c])
            if x is not None and y is not None:
                out.setdefault(c, []).append((x, y))
    return out


def r_of(rs):
    p = np.array([q for v in rs for q in v])
    return float(np.corrcoef(p[:, 0], p[:, 1])[0, 1]) if len(p) >= 3 else np.nan


def t1(cnt, rng):
    same = rows(cnt, [(("G", 4), ("M", 4)), (("G", 6), ("M", 6))])
    diff = rows(cnt, [(("G", 4), ("M", 6)), (("G", 6), ("M", 4))])
    clones = sorted(set(same) | set(diff))
    obs = r_of(same.values()) - r_of(diff.values())
    bs = []
    for _ in range(BOOT):
        pick = [clones[i] for i in rng.integers(len(clones), size=len(clones))]
        bs.append(
            r_of([same[c] for c in pick if c in same]) - r_of([diff[c] for c in pick if c in diff])
        )
    return {
        "r_same": r_of(same.values()),
        "r_diff": r_of(diff.values()),
        "rows_same": sum(map(len, same.values())),
        "rows_diff": sum(map(len, diff.values())),
        "clones_same": len(same),
        "clones_diff": len(diff),
        "diff": obs,
        "diff_ci": np.nanpercentile(bs, [2.5, 97.5]).tolist(),
    }


def pair(cnt, a, da, b, db):
    s = test_stats(cnt[a][da], cnt[b][db], boot=BOOT)
    return {"r": s["r"], "ci": s["r_ci"], "clones": s["clones"]}


def main():
    cnt = load_counts(CONDS)
    rng = np.random.default_rng(0)
    rep = {"T1_GM": t1(cnt, rng)}
    rep["T1_cells"] = {
        f"{c}{d}": [
            int(sum(v[0] for v in cnt[c][d].values())),
            int(sum(v[1] for v in cnt[c][d].values())),
        ]
        for c in cnt
        for d in (4, 6)
    }
    rep["T1_parts"] = {
        f"G{da}-M{db}": pair(cnt, "G", da, "M", db) for da in (4, 6) for db in (4, 6)
    }
    rep["T2_d4_d6"] = {c: pair(cnt, c, 4, c, 6) for c in ("G", "M", "full")}
    rep["T3"] = {f"full6-{b}{d}": pair(cnt, "full", 6, b, d) for b in ("G", "M") for d in (4, 6)}
    json.dump(rep, open(os.path.join(OUT, "b18_timing.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
