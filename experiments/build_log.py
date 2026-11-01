"""Rebuild experiments/results.tsv: one row per training run, autoresearch-style.

Each row is a finished run with its study, the commit that produced it, the primary metric
(mean over the last 30 iterations) and a status. Re-run after any new study.
"""

import csv
import glob
import json
import os
from statistics import mean

TAIL = 30
# study id -> (results dir, commit that produced the results, primary metric)
STUDIES = {
    "E1_dish": ("results", "e85d3b6", "dish_total_return"),
    "E2_dish_hard": ("results_hard", "8380ed5", "dish_total_return"),
    "E3_tissue_global": ("results_tissue", "4fd1901", "score"),
    "E4_tissue_spatial": ("results_tissue_spatial", "7263c29", "score"),
    "E5_tissue_stable": ("results_tissue_stable", "c1acd59", "score"),
    "E7_cue": ("results_cue", "607e9ff", "score"),
    "E8_es": ("results_es", "57862e9", "score"),
}
SECONDARY = [
    "size", "prop_stem", "prop_a", "prop_b", "pattern_accuracy", "sister_concordance",
    "segregation", "return_cv_across_agents", "policy_js_divergence", "territory_overlap",
    "coop_payouts_per_dish", "cue_accuracy",
]


def tail_mean(history, key):
    vals = [h[key] for h in history[-TAIL:] if h.get(key) is not None]
    return mean(vals) if vals else None


def status(study, history):
    if study.startswith("E1") or study.startswith("E2"):
        return "ok"
    late = [h["size"] for h in history[-100:]]
    if min(late) <= 1.5:
        return "collapsed"
    return "ok"


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    rows = []
    for study, (folder, commit, primary) in STUDIES.items():
        for path in sorted(glob.glob(os.path.join(here, folder, "*.json"))):
            run = json.load(open(path))
            cfg, hist = run["config"], run["history"]
            name = os.path.basename(path)[:-5]
            condition = name.rsplit("_seed", 1)[0]
            row = {
                "study": study, "commit": commit, "run": name, "condition": condition,
                "seed": cfg["seed"], "iterations": len(hist), "primary_metric": primary,
                "primary": tail_mean(hist, primary), "min_size_last100": None,
                "status": status(study, hist),
            }
            if "size" in hist[0]:
                row["min_size_last100"] = min(h["size"] for h in hist[-100:])
            for k in SECONDARY:
                row[k] = tail_mean(hist, k)
            rows.append(row)
    fields = list(rows[0].keys())
    with open(os.path.join(here, "results.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, fields, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else ("" if v is None else v))
                        for k, v in r.items()})
    print(f"{len(rows)} runs -> results.tsv")
    for study in STUDIES:
        sub = [r for r in rows if r["study"] == study]
        bad = sum(r["status"] == "collapsed" for r in sub)
        print(study, len(sub), "runs,", bad, "collapsed")


if __name__ == "__main__":
    main()
