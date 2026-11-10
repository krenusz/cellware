"""C6-S8d: model prediction + power for the seeding-size effect (100 vs 300 cells) on moment_elong. Synthetic only.
See EXPERIMENT_LOG C6-S8d. Chunked on GitHub Actions (--part TAG), merged locally (no args). Reads no atlas value.

    python c6s8d_size.py --plan | --part TAG | (no args = merge)
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.stats import norm

import c6s2_space as c2
import c6s3_break as b
import c6s7l_axis as L
import c6s8a_power as A

OUT = c2.OUT
L.PARTS = os.path.join(OUT, "c6s8d_parts")
SIZES = (300, 100)
N_SIM = 64
CH = 8
N_DATA = {100: 355, 300: 174}


def job(a):
    p, n, seed = a
    rng = np.random.default_rng(seed)
    s, c, lab = b.gastruloid([("bulk", 1.0, None, 1.0)], n, rng)
    x, P, sh = L.simulate(s, c, p, rng)
    return {"moment_elong": sh["moment_elong"], "one_minus_circ": sh["one_minus_circ"], "align": L.align(x, P),
            "p_frac": float(np.mean(P)), "multipolar": bool(L.multipolar(x, P)), "n": int(len(x))}


def collect():
    raw = {}
    for ai, arm in enumerate(("a", "c")):
        for ni, n in enumerate(SIZES):
            tag = f"Z_{arm}_N{n}"
            raw[tag] = L.gather(job, tag, [(A.ARMS[arm], n, 131000 + 1000 * ai + 300 * ni + j) for j in range(N_SIM)],
                                CH)
    return raw


def main():
    raw = collect()
    rng = np.random.default_rng(7)
    res = {"n_data": N_DATA, "by_arm": {}}
    for arm in ("a", "c"):
        e = {n: np.array([r["moment_elong"] for r in raw[f"Z_{arm}_N{n}"]]) for n in SIZES}
        d = float(e[100].mean() - e[300].mean())
        sd = float(np.sqrt((e[100].var(ddof=1) + e[300].var(ddof=1)) / 2))
        se = sd * np.sqrt(1 / N_DATA[100] + 1 / N_DATA[300])
        power = float(norm.sf(norm.ppf(0.95) - abs(d) / se))
        boot = [rng.choice(e[100], len(e[100])).mean() - rng.choice(e[300], len(e[300])).mean() for _ in range(4000)]
        summ = {str(n): {k: float(np.mean([r[k] for r in raw[f"Z_{arm}_N{n}"]]))
                         for k in ("moment_elong", "one_minus_circ", "align", "p_frac", "multipolar")} for n in SIZES}
        res["by_arm"][arm] = {"delta_100_minus_300": d, "sd_pooled": sd, "cohen_d": d / sd, "power": power,
                              "delta_90ci": [float(np.percentile(boot, 5)), float(np.percentile(boot, 95))],
                              "sign": int(np.sign(d)), "means": summ}
    ba = res["by_arm"]
    res["gate"] = {"same_sign": ba["a"]["sign"] == ba["c"]["sign"] != 0,
                   "min_power": min(v["power"] for v in ba.values())}
    res["gate"]["proposable"] = bool(res["gate"]["same_sign"] and res["gate"]["min_power"] >= 0.8)
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(OUT, "c6s8d_size.json"), "w"), indent=1)


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--plan" in a:
        L.MODE["plan"] = []
        collect()
        print("\n".join(L.MODE["plan"]))
    elif "--part" in a:
        L.MODE["only"] = a[a.index("--part") + 1]
        try:
            collect()
        except L._Done:
            pass
    else:
        main()
