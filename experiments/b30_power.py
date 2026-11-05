"""B30: can the held-out Epo/SCF shard-1 pooled-pairs test tell the candidate models apart?
Simulated predictions only; no held-out outcome is read. See the B30 pre-registration.

    uv run python experiments/b30_power.py
"""

from __future__ import annotations

import glob
import json
import os
from itertools import combinations

import e26_selection as e26
import e32_multinomial as e32
import e33_measured as e33
import e34_fullarm as e34
import e37_pool as e37
import e46_selexp as e46
import e47_capsel as e47
import e48_capsel6 as e48
import numpy as np
from clone_state_model import OUT, split
from fit_cytokine import pooled_pairs_r
from sweep_guard import gated

MODS = {"e38": e37, "e46": e46, "e47": e47, "e48": e48}
PAIRS_FOR_RULE = ("e38", "e46", "e48")
E14_HALF = 0.37


def predict(mod, prm, clones, meas, nsim):
    out = []
    for s in range(nsim):
        sim = mod.simulate(clones, prm, np.random.default_rng(3000 + s), meas)
        res = pooled_pairs_r(split(sim, 1), boot=1)
        out.append((res["r"], res["clones"]))
    return out


def main(nsim=50):
    e34.e33.unpack_base = e33.unpack
    cnt = e33.load_counts(e33.CONDS5)
    co = e32.copresence(split(cnt, None))
    clones = e26.universe(cnt)
    meas = e33.measured(clones, np.random.default_rng(77))
    rep = {"models": {}}
    for name, mod in MODS.items():
        res = [json.load(open(f)) for f in glob.glob(os.path.join(OUT, f"{name}_s*.json"))]
        g = sorted((r for r in res if gated(r, co)), key=lambda r: r["seed"])
        draws = [d for r in g for d in predict(mod, r["params"], clones, meas, nsim)]
        r = np.array([d[0] for d in draws])
        n = np.array([d[1] for d in draws])
        rep["models"][name] = {
            "gated": len(g),
            "mean": float(r.mean()),
            "q95": np.percentile(r, [2.5, 97.5]).tolist(),
            "half_width": float(1.96 * r.std()),
            "median_clones": float(np.median(n)),
            "sentinel_share": float((n < 10).mean()),
        }
    rep["pairs"] = {}
    for a, b in combinations(PAIRS_FOR_RULE, 2):
        A, B = rep["models"][a], rep["models"][b]
        diff = abs(A["mean"] - B["mean"])
        need = max(E14_HALF, A["half_width"], B["half_width"])
        rep["pairs"][f"{a}-{b}"] = {"diff": diff, "needed": need, "discriminates": bool(diff > need)}
    rep["look_informative"] = any(p["discriminates"] for p in rep["pairs"].values())
    json.dump(rep, open(os.path.join(OUT, "b30_power.json"), "w"), indent=1)
    print(json.dumps({"pairs": rep["pairs"], "look_informative": rep["look_informative"]}, indent=1))


if __name__ == "__main__":
    main()
