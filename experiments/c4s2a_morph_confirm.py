"""C4-S2a: mixing rescue on the Aug 2026 plates (5s2, 6a). See EXPERIMENT_LOG C4-S2a.

    uv run python c4s2a_morph_confirm.py        # train stats, selection file, power table (counts only)

The mixture plate is read only through `test(part)` (called by test_oracle) and `counts(part)` (counts only).
Y = 1 - Nimbus Circularity (higher = more elongated).
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import norm

import partitions

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "..", "data", "gastruloid_raj", "morph_08312026")
PURE = os.path.join(DIR, "bulk_5s2_6a.json")
MIX = os.path.join(DIR, "bulk+5s2_bulk+6a_5s2+6a.json")
DATASET = "raj_morph_0831"
EFFECT_SD = 0.81  # Mgen delta_comp 1.47 logit / 1.81 (logistic SD); pre-registered
SEL = os.path.join(HERE, "selection", "C4-S2a.json")


def _load(path, values=True):
    d = json.load(open(path, encoding="utf-8"))
    rows = []
    for a in d["annotations"]:
        r = {"id": a["_id"], "tag": a["tags"][0], "xy": a["location"]["XY"]}
        if values:
            pv = d["annotationPropertyValues"].get(a["_id"], {})
            m = next(iter(pv.values()), {}) if pv else {}
            r["Y"] = 1 - float(m["Circularity"]) if "Circularity" in m else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def train():
    p = _load(PURE)
    g = p.groupby("tag")["Y"]
    sd = float(np.sqrt(np.average(g.var(ddof=1), weights=g.size() - 1)))
    return p, {t: float(v) for t, v in g.mean().items()}, sd


def counts(part):
    """Units per contrast group (counts only, H9/H14)."""
    m = _load(MIX, values=False)
    m = m[m["id"].map(lambda u: partitions.partition_of(DATASET, u) in part)]
    pure = _load(PURE, values=False)["tag"].value_counts()
    return {"mix": int((m["tag"] == "5s2+6a").sum()), "m5s2": int(pure["5s2"]), "m6a": int(pure["6a"])}


def test(part):
    """Sealed reader: one partition (or a pre-registered union such as 'AB'). Call only from test_oracle."""
    p, mu, sd = train()
    m = _load(MIX)
    m = m[m["id"].map(lambda u: partitions.partition_of(DATASET, u) in part)]
    y = {t: g["Y"].to_numpy() for t, g in m.groupby("tag")}
    py = {t: g["Y"].to_numpy() for t, g in p.groupby("tag")}
    rng = np.random.default_rng(4202)

    def boot(f, B=4000):
        return np.array([f({t: rng.choice(v, len(v)) for t, v in y.items()},
                           {t: rng.choice(v, len(v)) for t, v in py.items()}) for _ in range(B)])

    def t2(yy, pp):
        return yy["5s2+6a"].mean() - (pp["5s2"].mean() + pp["6a"].mean()) / 2

    def n3(yy, pp):
        return yy["5s2+6a"].mean() - max(pp["5s2"].mean(), pp["6a"].mean())

    def bulkmix(yy, pp, c):
        return yy[f"{c}+bulk"].mean() - (pp[c].mean() + pp["bulk"].mean()) / 2

    def t1(yy, pp):
        mix = yy["5s2+6a"].mean() - (yy["5s2+bulk"].mean() + yy["6a+bulk"].mean()) / 2
        m0 = ((pp["5s2"].mean() + pp["6a"].mean()) / 2 - pp["bulk"].mean()) / 2
        return mix - m0

    est = t2(y, py)
    b = boot(t2)
    lo = float(np.percentile(b, 5))
    res = {"partition": part, "n_mix": {t: int(len(v)) for t, v in y.items()},
           "train_means": {t: round(v, 4) for t, v in mu.items()}, "train_sd": round(sd, 4),
           "T2": round(float(est), 4), "T2_lower95_one_sided": round(lo, 4),
           "T2_sd_units": round(float(est / sd), 3), "predicted_sd_units": EFFECT_SD,
           "primary": "pass" if lo > 0 else "fail"}
    sec = {"N3_better_member": n3}
    if "5s2+bulk" in y:
        sec["bulk+5s2_vs_members"] = lambda yy, pp: bulkmix(yy, pp, "5s2")
    if "6a+bulk" in y:
        sec["bulk+6a_vs_members"] = lambda yy, pp: bulkmix(yy, pp, "6a")
    if "5s2+bulk" in y and "6a+bulk" in y:
        sec["T1_sorting_within_plate"] = t1
    for k, f in sec.items():
        bb = boot(f, 2000)
        res[k] = [round(float(f(y, py)), 4)] + [round(float(v), 4) for v in np.percentile(bb, [2.5, 97.5])]
    return res


def main():
    p, mu, sd = train()
    sel = {"test": "gastruloid_c4s2a_rescue", "experiment": "C4-S2a",
           "candidates": {"Mgen": -1248.63, "S": -1248.9, "M0": -1282.84}, "higher_is_better": True,
           "chosen": "Mgen",
           "note": "train-CV log-lik over 61 held-out compositions (C3-S2, G2 data); Mgen and S both predict a "
                   "complementary-pair rescue (1.47 / 1.54 logit); M0 predicts none",
           "power": {"sd": round(sd, 4), "effect": round(EFFECT_SD * sd, 4), "alpha": 0.05, "one_sided": True,
                     "min_power": 0.8, "weights": {"mix": 1.0, "m5s2": 0.5, "m6a": 0.5}},
           "train": {"means": {t: round(v, 4) for t, v in mu.items()}, "n": p["tag"].value_counts().to_dict()}}
    os.makedirs(os.path.dirname(SEL), exist_ok=True)
    json.dump(sel, open(SEL, "w", encoding="utf-8"), indent=1)
    w = sel["power"]["weights"]
    table = {}
    for part in ("A", "B", "C", "AB", "ABC"):
        n = counts(part)
        se = sd * np.sqrt(sum(w[g] ** 2 / v for g, v in n.items()))
        table[part] = {"counts": n, "power": round(float(1 - norm.cdf(norm.ppf(0.95) - EFFECT_SD * sd / se)), 3)}
    print(json.dumps({"train_n": sel["train"]["n"], "train_sd": round(sd, 4), "power": table}, indent=1))


if __name__ == "__main__":
    main()
