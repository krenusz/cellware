"""B33: can selection-sensitive statistics (B31) on shard-1 clones tell the models apart?
Simulated predictions only; no observed shard-1 value is computed. See the B33 pre-registration.

    uv run python experiments/b33_power.py
"""

from __future__ import annotations

import glob
import json
import os
from itertools import combinations

import b31_copresence as b31
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
from sweep_guard import gated

MODS = {"e38": e37, "e46": e46, "e47": e47, "e48": e48}
RULE_MODELS = ("e38", "e46", "e48")
STATS = {  # statistic -> its denominator key in b31.stats
    "a_lift_ge2": "a_den_ge2",
    "a_lift_ge3": "a_den_ge3",
    "b_PG_given_M6+": "b_den_M6+",
    "b_PM_given_G6+": "b_den_G6+",
    "b_PG_given_M2": "b_den_M2",
}


def main(nsim=20):
    e34.e33.unpack_base = e33.unpack
    cnt = e33.load_counts(e33.CONDS5)
    co = e32.copresence(split(cnt, None))
    clones = e26.universe(cnt)
    meas = e33.measured(clones, np.random.default_rng(77))
    rep = {"models": {}}
    for name, mod in MODS.items():
        res = [json.load(open(f)) for f in glob.glob(os.path.join(OUT, f"{name}_s*.json"))]
        g = sorted((r for r in res if gated(r, co)), key=lambda r: r["seed"])
        draws = [
            b31.stats(split(mod.simulate(clones, r["params"], np.random.default_rng(4000 + s), meas), 1), {})
            for r in g
            for s in range(nsim)
        ]
        m = {"gated": len(g)}
        for k, den in STATS.items():
            x = np.array([d[k] for d in draws], float)
            x = x[np.isfinite(x)]
            m[k] = {
                "mean": float(x.mean()),
                "q95": np.percentile(x, [2.5, 97.5]).tolist(),
                "half_width": float(1.96 * x.std()),
                "median_den": float(np.median([d[den] for d in draws])),
            }
        rep["models"][name] = m
        print(name, len(g), {k: round(v["mean"], 3) for k, v in m.items() if k != "gated"}, flush=True)
    rep["pairs"] = {}
    for k in STATS:
        for a, b in combinations(RULE_MODELS, 2):
            A, B = rep["models"][a][k], rep["models"][b][k]
            diff = abs(A["mean"] - B["mean"])
            need = max(A["half_width"], B["half_width"])
            enough = min(A["median_den"], B["median_den"]) >= 30
            rep["pairs"][f"{k} {a}-{b}"] = {
                "diff": diff, "needed": need, "enough_clones": bool(enough),
                "discriminates": bool(enough and diff > need),
            }
    rep["any_discriminates"] = any(p["discriminates"] for p in rep["pairs"].values())
    json.dump(rep, open(os.path.join(OUT, "b33_power.json"), "w"), indent=1)
    for k, p in rep["pairs"].items():
        print(f"{k:30s} diff {p['diff']:.3f} need {p['needed']:.3f} n_ok {p['enough_clones']} -> {p['discriminates']}")
    print("any_discriminates", rep["any_discriminates"])


if __name__ == "__main__":
    main()
