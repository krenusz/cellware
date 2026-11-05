"""Standard readings for any fitted model family (replaces the per-experiment analyze_eNN wrappers).

Runs, for the family's gated seeds:
  1. the E46-style analysis (gate, primary on stored stats, B27, B24/B26 statistics, instruction
     shifts vs E38) -> results_e18/<name>_analysis.json
  2. the B29 fit-set re-score (primary, stability, CRPS) -> results_e18/rescore_<name>.json
  3. for the best gated fit, the presence diagnostics (B31 lift and cross-presence, B32 presence
     elsewhere, B34 privacy by size; 200 simulations) -> results_e18/<name>_presence.json

    uv run python experiments/analyze_model.py e48 e48_capsel6
"""

from __future__ import annotations

import glob
import importlib
import json
import os
import sys

import analyze_e46
import b29_rescore
import b31_copresence as b31
import b32_small_clones as b32
import b34_private as b34
import e26_selection as e26
import e32_multinomial as e32
import e33_measured as e33
import e34_fullarm as e34
import numpy as np
from clone_state_model import OUT, split
from sweep_guard import gated

PRESENCE_KEYS = ("a_lift_ge2", "a_lift_ge3", "b_PG_given_M2", "b_PG_given_M6+", "M2_in_any",
                 "G_3-5", "G_6-15", "M_2", "M_3-5")


def collect(o):
    return {**b31.stats(o, {}), **b32.presence_elsewhere(o)[0], **b34.private_by_size(o)[0]}


def presence(name, mod, nsim=200):
    e34.e33.unpack_base = e33.unpack
    cnt = e33.load_counts(e33.CONDS5)
    fit = split(cnt, None)
    co = e32.copresence(fit)
    res = [json.load(open(f)) for f in glob.glob(os.path.join(OUT, f"{name}_s*.json"))]
    g = [r for r in res if gated(r, co)]
    if not g:
        print(f"{name}: no gated fit, presence diagnostics skipped")
        return None
    best = min(g, key=lambda r: r["loss"])
    clones = e26.universe(cnt)
    meas = e33.measured(clones, np.random.default_rng(77))
    data = collect(fit)
    sims = [collect(split(mod.simulate(clones, best["params"], np.random.default_rng(1000 + s), meas), None))
            for s in range(nsim)]
    out = {}
    for k in PRESENCE_KEYS:
        x = np.array([s[k] for s in sims], float)
        x = x[np.isfinite(x)]
        lo, hi = np.percentile(x, [2.5, 97.5])
        out[k] = {"data": data[k], "sim_mean": float(x.mean()), "sim_95": [float(lo), float(hi)],
                  "inside": bool(lo <= data[k] <= hi)}
    rep = {"best_gated": best["seed"], "gated": len(g), "presence": out}
    json.dump(rep, open(os.path.join(OUT, f"{name}_presence.json"), "w"), indent=1)
    for k, v in out.items():
        print(f"{k:16s} data {v['data']:.3f} | {name} {v['sim_mean']:.3f} "
              f"[{v['sim_95'][0]:.2f},{v['sim_95'][1]:.2f}] {'in ' if v['inside'] else 'OUT'}")
    return rep


def main(name, module):
    mod = importlib.import_module(module)
    analyze_e46.main(name=name, sim_mod=mod)
    b29_rescore.main(names=(name,), extra={name: mod})
    presence(name, mod)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
