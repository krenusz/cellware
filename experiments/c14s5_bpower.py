"""C14-S5 (a): simulated power of the frozen C14-S2 test on partition B (pre-registered in EXPERIMENT_LOG).

Inputs are committed numbers only: train per-embryo LOEO deltas (results_c14/c14s3_region.json: d = P over base,
dR = the same with region in both models) and the spent A per-embryo deltas (C14-S2 look, WRITEUP_C14.md). B's
embryos are not read; only their public stage labels from partitions.py.

Run: python -I experiments/c14s5_bpower.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from power import sim_power  # noqa: E402

OUT = HERE / "results_c14"
SD_FLOOR = 0.0374  # frozen C14-S2 rule: sd = max(train embryo sd, look's own sd)
B_STAGES = [7.5, 7.5, 9.0]  # E7.5-R2, E7.5-R3, E9.0-R3 (public labels, partitions.py)
A_SPENT = {"E8.0-R2": -0.006, "E8.5-R1": 0.043, "E9.0-R2": 0.115, "E9.5-R2": 0.137, "E9.5-R3": 0.131}  # C14-S2, 88f4654
N_SIM = 20000
SEED = 20261117


def stage(name):
    return float(name.split("-")[0][1:])


def frozen_z(vals):
    v = np.asarray(vals, float)
    n = len(v)
    if n == 0:
        return 0.0
    sd = max(SD_FLOOR, float(v.std(ddof=1))) if n >= 2 else SD_FLOOR
    return float(v.mean() / (sd / np.sqrt(n)))


def simulate(points, keep_e75, rng):
    st = np.array([stage(k) for k in points])
    y = np.array(list(points.values()))
    zs = []
    for _ in range(N_SIM):
        i = rng.integers(0, len(y), len(y))  # bootstrap embryos: parameter uncertainty
        if len(np.unique(st[i])) < 2:
            continue
        b, a = np.polyfit(st[i], y[i], 1)
        res = y[i] - (a + b * st[i])
        sd = float(np.sqrt((res ** 2).sum() / max(len(y) - 2, 1)))
        stages = [s for s in B_STAGES if keep_e75 or s > 7.5]
        draw = [a + b * s + rng.normal(0, sd) for s in stages]
        zs.append(frozen_z(draw))
    return np.array(zs)


def main():
    r = json.loads((OUT / "c14s3_region.json").read_text(encoding="utf-8"))
    sets = {"d (train + spent A)": {**r["d"], **A_SPENT}, "dR region-adjusted (train only)": dict(r["dR"])}
    rng = np.random.default_rng(SEED)
    out, lines = {}, []
    for label, pts in sets.items():
        st = np.array([stage(k) for k in pts])
        b, a = np.polyfit(st, np.array(list(pts.values())), 1)
        for scen, keep in [("S-drop (E7.5 gives no readout)", False), ("S-keep (E7.5 at extrapolated effect)", True)]:
            zs = simulate(pts, keep, rng)
            p = sim_power(zs, 1.0)
            out[f"{label} | {scen}"] = dict(power=round(p, 3), median_z=round(float(np.median(zs)), 2), n_sim=len(zs),
                                            fit_intercept=round(float(a), 4), fit_slope_per_day=round(float(b), 4),
                                            effect_at_E7_5=round(float(a + b * 7.5), 4), effect_at_E9_0=round(float(a + b * 9.0), 4),
                                            n_embryos_in_fit=len(pts))
            lines.append(f"{label} | {scen}: power {p:.3f}, median z {np.median(zs):.2f}; "
                         f"fit effect E7.5 {a + b * 7.5:+.3f}, E9.0 {a + b * 9.0:+.3f}")
    concl = out["dR region-adjusted (train only) | S-drop (E7.5 gives no readout)"]["power"]
    verdict = "LOOKABLE" if concl >= 0.8 else "UNDERPOWERED"
    lines.append(f"decision (dR, S-drop): power {concl:.3f} -> B {verdict}")
    print("\n".join(lines))
    res = dict(verdict=verdict, rule="frozen C14-S2 z test; sd floor 0.0374; z > 1.96", b_stages=B_STAGES,
               inputs="train LOEO d/dR (c14s3_region.json) + spent A d (C14-S2); B not read", results=out)
    (OUT / "c14s5_bpower.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
