"""Audit (2026-11-04): re-score every saved E33-E41b fit's B7 checks on the fit set only.

The stored B7 data values (b7_selection.json) and the simulated B7 intervals were both computed on
all clones, including the held-out shard-1 clones. This recomputes both on the fit set
(split(..., None)) with the same 200 simulation seeds and reports which verdicts change.

    uv run python experiments/audit_b7_fitset.py
"""

from __future__ import annotations

import glob
import json
import os
import re

import e26_selection as e26
import e32_multinomial as e32
import e33_measured as e33
import e34_fullarm as e34
import e37_pool as e37
import e39_mclone as e39
import e41_memory as e41
import numpy as np
from clone_state_model import OUT, split
from e25_clone_types import presence_stats
from sweep_guard import gated

SIM = {"e33": e33.simulate, "e34": e34.simulate, "e35": e33.simulate, "e36s": e34.simulate}
SIM |= {"e37": e37.simulate, "e38": e37.simulate, "e40": e37.simulate, "e39": e39.simulate}
SIM |= {"e41": e41.simulate, "e41b": e41.simulate}
KEYS = ("G_mean_both_minus_Gonly", "M_both_minus_Monly")


def family(name):
    base = re.sub(r"_s\d+$", "", name)
    return "e40" if base.startswith("e40g") else base


def main(nsim=200):
    cnt = e33.load_counts(e33.CONDS5)
    fit = split(cnt, None)
    data = presence_stats(fit)
    co = e32.copresence(fit)
    clones = e26.universe(cnt)
    meas = e33.measured(clones, np.random.default_rng(77))
    rows = []
    files = sorted(glob.glob(os.path.join(OUT, "e3[3-9]*_s*.json")))
    files += sorted(glob.glob(os.path.join(OUT, "e4[01]*_s*.json")))
    for f in files:
        name = os.path.basename(f)[:-5]
        fam = family(name)
        if fam not in SIM:
            continue
        rep = json.load(open(f))
        sims = [
            presence_stats(
                split(SIM[fam](clones, rep["params"], np.random.default_rng(1000 + s), meas), None)
            )
            for s in range(nsim)
        ]
        row = {
            "name": name,
            "family": fam,
            "gated": bool(gated(rep, co)),
            "adequate": rep["adequate"],
        }
        for k in KEYS:
            v = np.array([s[k] for s in sims], float)
            v = v[np.isfinite(v)]
            lo, hi = np.percentile(v, [2.5, 97.5])
            row[f"{k}_old"] = bool(rep[k]["inside"])
            row[f"{k}_new"] = bool(lo <= data[k] <= hi)
            row[f"{k}_sim"] = [float(v.mean()), float(lo), float(hi)]
        gm = rep["G-M"]["p_two_sided"] >= 0.05
        row["passes_old"] = bool(rep["passes"])
        row["passes_new"] = bool(
            rep["adequate"] and gm and row[f"{KEYS[0]}_new"] and row[f"{KEYS[1]}_new"]
        )
        rows.append(row)
        changed = [k for k in KEYS if row[f"{k}_old"] != row[f"{k}_new"]]
        print(name, "gated" if row["gated"] else "-", "changed:", changed or "none", flush=True)
    summary = {}
    for fam in sorted({r["family"] for r in rows}):
        rs = [r for r in rows if r["family"] == fam]
        g = [r for r in rs if r["gated"]]
        summary[fam] = {
            "fits": len(rs),
            "gated": len(g),
            "B7G_pass_gated_old_new": [
                sum(r[f"{KEYS[0]}_old"] for r in g),
                sum(r[f"{KEYS[0]}_new"] for r in g),
            ],
            "B7M_pass_gated_old_new": [
                sum(r[f"{KEYS[1]}_old"] for r in g),
                sum(r[f"{KEYS[1]}_new"] for r in g),
            ],
            "passes_old_new": [sum(r["passes_old"] for r in rs), sum(r["passes_new"] for r in rs)],
            "verdict_changes": [
                r["name"] for r in rs if any(r[f"{k}_old"] != r[f"{k}_new"] for k in KEYS)
            ],
        }
    out = {"data_fitset": data, "summary": summary, "rows": rows}
    json.dump(out, open(os.path.join(OUT, "audit_b7_fitset.json"), "w"), indent=1)
    print(json.dumps({"data_fitset": data, "summary": summary}, indent=1))


if __name__ == "__main__":
    main()
