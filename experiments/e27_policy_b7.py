"""E27: do the trained learned-rule policies (E11, E12, r_e16) reproduce B7?
See the 2026-11-03 (5) batch pre-registration in EXPERIMENT_LOG.md.

    uv run python experiments/e27_policy_b7.py e11|e12|r_e16 [n_sims]
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import train_clone_e11 as e11
from clone_state_model import OUT
from e25_clone_types import groups, presence_stats

SRC = os.path.join(os.path.dirname(__file__), "results_clone_e11")
DATA = {"G_mean_both_minus_Gonly": -0.3728378396099915, "M_both_minus_Monly": 0.0836171870376283}


def main(family, n_sims=20):
    torch.set_num_threads(2)
    rep = {"family": family, "policies": {}}
    for memory in ("inherit", "resample"):
        for seed in range(5):
            name = f"{family}_{memory}_seed{seed}"
            cfg = json.load(open(os.path.join(SRC, name + ".json")))
            sd = torch.load(os.path.join(SRC, name + ".pt"))
            net = e11.Policy(trait="trait_scale" in sd, rank="rank_scale" in sd)
            net.load_state_dict(sd)
            clones = cfg["config"].get("clones") or 1500
            stats, sizes = [], []
            for s in range(100, 100 + n_sims):
                cnt = e11.simulate(net, cfg["landscape"], memory, clones=clones, seed=s)
                stats.append(presence_stats(cnt))
                _, _, both, g_only, m_only = groups(cnt)
                sizes.append([len(both), len(g_only), len(m_only)])
            r = {"n_both_G_M": np.mean(sizes, 0).tolist()}
            inside = True
            for k, d in DATA.items():
                v = np.array([x[k] for x in stats], float)
                v = v[np.isfinite(v)]
                lo, hi = np.percentile(v, [2.5, 97.5]) if len(v) else (np.nan, np.nan)
                ok = bool(lo <= d <= hi)
                inside &= ok
                r[k] = {
                    "data": d,
                    "sim_mean": float(v.mean()) if len(v) else None,
                    "sim_95": [float(lo), float(hi)],
                    "inside": ok,
                }
            r["both_inside"] = inside
            rep["policies"][name] = r
            print(name, json.dumps(r), flush=True)
    pol = rep["policies"].values()
    rep["n_both_inside"] = sum(p["both_inside"] for p in pol)
    rep["family_mean_G"] = float(np.mean([p["G_mean_both_minus_Gonly"]["sim_mean"] for p in pol]))
    rep["family_mean_M"] = float(np.mean([p["M_both_minus_Monly"]["sim_mean"] for p in pol]))
    rep["reproduces_B7"] = rep["n_both_inside"] >= 6
    json.dump(rep, open(os.path.join(OUT, f"e27_{family}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "policies"}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20)
