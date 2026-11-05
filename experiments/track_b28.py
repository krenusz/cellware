"""B28: barcode detection by cell type, and a design check of the day-2 split.
See the B28 pre-registration in EXPERIMENT_LOG.md.

    uv run python experiments/track_b28.py
"""

from __future__ import annotations

import glob
import gzip
import json
import os

import numpy as np
from fit_cytokine import DATA

HERE = os.path.dirname(__file__)
MYE = {"full cocktail": "full", "G-csf": "G", "M-csf": "M"}
RNG = np.random.default_rng(28)


def load():
    with gzip.open(os.path.join(DATA, "metadata.txt.gz"), "rt") as f:
        rows = [line.rstrip("\n").split("\t") for line in f][1:]
    with gzip.open(os.path.join(DATA, "clone_matrix.mtx.gz"), "rt") as f:
        m = np.loadtxt(f, comments="%", skiprows=3, dtype=int)
    clone = np.full(len(rows), -1)
    clone[m[:, 1] - 1] = m[:, 0] - 1
    return rows, clone


def part_a(rows, clone, nboot=2000):
    lib = np.array([r[0] for r in rows])
    day = np.array([r[2] for r in rows])
    typ = np.array([r[3] for r in rows])
    cond = np.array([MYE.get(r[4], "") for r in rows])
    has = clone >= 0
    keep = (day != "2") & (cond != "")
    out = {}
    for c in ("pooled", "full", "G", "M"):
        sel = keep & ((cond == c) if c != "pooled" else True)
        libs = np.unique(lib[sel])
        res = {"libraries": len(libs)}
        for t in np.unique(typ[sel]):
            s = sel & (typ == t)
            res[t] = {"n": int(s.sum()), "detected": float(has[s].mean())}
        neu, mo = sel & (typ == "Neu"), sel & (typ == "Mo")
        n_neu, n_mo = int(neu.sum()), int(mo.sum())
        ratio = has[neu].mean() / has[mo].mean() if n_neu and n_mo else np.nan
        # bootstrap over libraries
        per = {lb: (has[neu & (lib == lb)].sum(), (neu & (lib == lb)).sum(),
                    has[mo & (lib == lb)].sum(), (mo & (lib == lb)).sum()) for lb in libs}
        boots = []
        for _ in range(nboot):
            pick = RNG.choice(libs, len(libs))
            a = np.array([per[p] for p in pick]).sum(0)
            if a[1] and a[3] and a[2]:
                boots.append((a[0] / a[1]) / (a[2] / a[3]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        res["ratio_Neu_Mo"] = {"value": float(ratio), "95": [float(lo), float(hi)],
                               "n_Neu": n_neu, "n_Mo": n_mo}
        out[c] = res
    p = out["pooled"]["ratio_Neu_Mo"]
    if p["n_Neu"] < 300 or p["n_Mo"] < 300:
        reading = "unresolved (too few cells)"
    elif p["95"][1] < 0.8 or p["95"][0] > 1.25:
        reading = "detection bias"
    elif p["95"][0] >= 0.8 and p["95"][1] <= 1.25:
        reading = "no material bias"
    else:
        reading = "unresolved"
    out["reading"] = reading
    return out


def solve_d(share, k0, ns=200_000):
    """Per-cell day-2 capture d such that P(n2 > 0) = share, K ~ 1 + Poisson(k0 - 1),
    each cell sequenced with p 0.25 and captured with d."""
    k = 1 + RNG.poisson(k0 - 1, ns)
    lo, hi = 0.0, 1.0
    for _ in range(50):
        d = (lo + hi) / 2
        p = 1 - np.mean((1 - 0.25 * d) ** k)
        lo, hi = (d, hi) if p < share else (lo, d)
    return d, float(1 - np.mean((1 - 0.25) ** k))  # also the ceiling at d = 1


def part_b(rows, clone):
    day = np.array([r[2] for r in rows])
    cond = np.array([MYE.get(r[4], "") for r in rows])
    has = clone >= 0
    d2 = has & (day == "2")
    n2 = np.bincount(clone[d2], minlength=clone.max() + 1)
    later = np.unique(clone[has & (day != "2") & (cond != "")])
    share = float((n2[later] > 0).mean())
    pos = n2[n2 > 0]
    import e32_multinomial as e32
    import e33_measured as e33
    from clone_state_model import split
    from sweep_guard import gated

    co = e32.copresence(split(e33.load_counts(e33.CONDS5), None))
    nus = []
    for f in glob.glob(os.path.join(HERE, "results_e18", "e38_s*.json")):
        d = json.load(open(f))
        if gated(d, co):
            nus.append(d["params"]["nu"])
    nu = float(np.median(nus)) if nus else np.nan
    sol = {}
    for k0 in (4, 8, 16):
        dd, ceil = solve_d(share, k0)
        sol[str(k0)] = {"d": dd, "share_ceiling_at_d1": ceil,
                        "implied_survival": nu / (0.1875 * k0) if nus else None}
    days = {}
    for r in rows:
        days.setdefault(r[4], set()).add(r[2])
    return {
        "day2_barcoded_cells": int(d2.sum()),
        "day2_cells": int((day == "2").sum()),
        "day2_clones": int((n2 > 0).sum()),
        "n2_hist": {str(i): int((pos == i).sum()) for i in range(1, 6)} | {"6+": int((pos >= 6).sum())},
        "n2_mean_given_seen": float(pos.mean()),
        "later_myeloid_clones": int(len(later)),
        "share_later_seen_day2": share,
        "e38_gated_nu_median": nu,
        "e38_gated_n": len(nus),
        "by_K0": sol,
        "conditions_days": {k: sorted(v) for k, v in days.items()},
    }


def main():
    rows, clone = load()
    rep = {"A": part_a(rows, clone), "B": part_b(rows, clone)}
    out = os.path.join(HERE, "results_larry", "b28_detection.json")
    json.dump(rep, open(out, "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
