"""B19: which M-CSF clones make the rare neutrophils, in data vs the E38 model?
See the B19 pre-registration in EXPERIMENT_LOG.md. Full, G-CSF and M-CSF only.

    uv run python experiments/track_b19.py
"""

from __future__ import annotations

import json
import os

import e26_selection as e26
import e33_measured as e33
import e37_pool as e37
import numpy as np
from clone_state_model import OUT
from fit_cytokine import CONDS5, load_counts, pooled

GATED = (1, 3, 4, 6, 8, 11, 12, 14, 15)  # E38 seeds passing the gate; 1 is the best


def link(a_cnt, m_cnt):
    """(partner N-share, M-CSF N+) for clones with >= 2 N+M cells in both."""
    out = []
    for c in set(a_cnt) & set(m_cnt):
        a, m = a_cnt[c], m_cnt[c]
        if a[0] + a[1] >= 2 and m[0] + m[1] >= 2:
            out.append((a[0] / (a[0] + a[1]), m[0] > 0))
    return np.array(out, float).reshape(-1, 2)


def auc(x, pos):
    p, n = x[pos], x[~pos]
    if not len(p) or not len(n):
        return np.nan
    return float(((p[:, None] > n) + 0.5 * (p[:, None] == n)).mean())


def stat(arr):
    if len(arr) == 0:
        return {"d": np.nan, "auc": np.nan, "n_pos": 0, "n_neg": 0}
    x, pos = arr[:, 0], arr[:, 1] > 0
    d = x[pos].mean() - x[~pos].mean() if pos.any() and (~pos).any() else np.nan
    return {"d": float(d), "auc": auc(x, pos), "n_pos": int(pos.sum()), "n_neg": int((~pos).sum())}


def boot_ci(arr, rng, n=2000):
    ds = [stat(arr[rng.integers(len(arr), size=len(arr))])["d"] for _ in range(n)]
    return np.nanpercentile(ds, [2.5, 97.5]).tolist()


def sim_stats(prm, clones, meas, seeds):
    out = {"full": [], "G": []}
    for s in seeds:
        sim = e37.simulate(clones, prm, np.random.default_rng(5000 + s), meas)
        pm = pooled(sim["M"])
        for a in out:
            out[a].append(stat(link(pooled(sim[a]), pm))["d"])
    return {a: np.array(v, float) for a, v in out.items()}


def main():
    cnt = load_counts(CONDS5)
    clones = e26.universe(cnt)
    meas = e33.measured(clones, np.random.default_rng(77))
    rng = np.random.default_rng(0)
    pm = pooled(cnt["M"])
    rep = {}
    for a, key in (("full", "T1_full_M"), ("G", "T2_G_M")):
        arr = link(pooled(cnt[a]), pm)
        rep[key] = {"data": {**stat(arr), "d_ci": boot_ci(arr, rng)}}
    load = lambda s: json.load(open(os.path.join(OUT, f"e38_s{s}.json")))["params"]  # noqa: E731
    best = sim_stats(load(GATED[0]), clones, meas, range(200))
    others = {s: sim_stats(load(s), clones, meas, range(50)) for s in GATED[1:]}
    for a, key in (("full", "T1_full_M"), ("G", "T2_G_M")):
        v = best[a][np.isfinite(best[a])]
        d = rep[key]["data"]["d"]
        rep[key]["model_e38_s1"] = {
            "mean": float(v.mean()),
            "p95": np.percentile(v, [2.5, 97.5]).tolist(),
            "data_pct": float((v < d).mean()),
        }
        rep[key]["other_seed_means"] = {s: float(np.nanmean(o[a])) for s, o in others.items()}
    json.dump(rep, open(os.path.join(OUT, "b19_rare_neutrophils.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
