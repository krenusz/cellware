"""E44 readings, as pre-registered: gate, primary, stability, class use, and the B24 statistics
(fit set) for the best gated fit.

    uv run python experiments/analyze_e43.py
"""

from __future__ import annotations

import glob
import json
import os

import e26_selection as e26
import e32_multinomial as e32
import e33_measured as e33
import e44_myeloid_class as e43
import numpy as np
import track_b24 as b24
import track_b26 as b26
from clone_state_model import OUT, split
from sweep_guard import gated

PRIMARY = ("G-M", "full-G", "full-M")
B7 = ("G_mean_both_minus_Gonly", "M_both_minus_Monly")


def main(nsim=200):
    cnt = e33.load_counts(e33.CONDS5)
    fit = split(cnt, None)
    co = e32.copresence(fit)
    reps = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, "e44_s*.json")))]
    rows = []
    for r in reps:
        comp = {k: r[k]["p_two_sided"] >= 0.05 for k in PRIMARY}
        comp |= {k: r[k]["inside"] for k in B7}
        rows.append(
            {
                "seed": r["seed"],
                "loss": r["loss"],
                "gens": r["generations"],
                "gated": bool(gated(r, co)),
                "pi_M": r["params"]["pi_M"],
                "g_up": r["params"]["g_up"],
                "s_G": r["params"]["s"]["G"],
                "comp": comp,
                "primary": all(comp.values()),
            }
        )
    g = [x for x in rows if x["gated"]]
    out = {"fits": len(rows), "gated": len(g), "rows": rows}
    if g:
        agree = {k: sum(x["comp"][k] for x in g) for k in (*PRIMARY, *B7)}
        out["component_pass_in_gated"] = agree
        out["primary_pass_gated"] = sum(x["primary"] for x in g)
        out["stable"] = bool(
            len(g) >= 9 and all(max(v, len(g) - v) >= 2 * len(g) / 3 for v in agree.values())
        )
        out["stable_and_passing"] = bool(
            out["stable"] and all(v >= 2 * len(g) / 3 for v in agree.values())
        )
        out["median_pi_M"] = float(np.median([x["pi_M"] for x in g]))
        out["median_g_up"] = float(np.median([x["g_up"] for x in g]))
        out["median_s_G"] = float(np.median([x["s_G"] for x in g]))
        out["class_used"] = (
            bool(out["median_pi_M"] > 0.02 and out["median_g_up"] > 0.5) if len(g) >= 5 else None
        )
        best = min(g, key=lambda x: x["loss"])
        prm = next(r for r in reps if r["seed"] == best["seed"])["params"]
        clones = e26.universe(cnt)
        meas = e33.measured(clones, np.random.default_rng(77))
        data = b24.stats(fit)
        sims = [
            b24.stats(split(e43.simulate(clones, prm, np.random.default_rng(1000 + s), meas), None))
            for s in range(nsim)
        ]
        b = {}
        for k in ("n_shared", "B7G", "D2_2", "D3_frac"):
            v = np.array([s[k] for s in sims], float)
            v = v[np.isfinite(v)]
            lo, hi = np.percentile(v, [2.5, 97.5])
            b[k] = {"data": data[k], "sim_mean": float(v.mean()), "sim_95": [float(lo), float(hi)]}
            b[k]["inside"] = bool(lo <= data[k] <= hi)
        d26 = b26.stats(fit)
        s26 = [
            b26.stats(split(e43.simulate(clones, prm, np.random.default_rng(1000 + s), meas), None))
            for s in range(nsim)
        ]
        v = np.array([x["C1_16+_nA"] for x in s26], float)
        lo, hi = np.percentile(v, [2.5, 97.5])
        b["B26_16plus_Mpure"] = {
            "data": d26["C1_16+_nA"],
            "sim_mean": float(v.mean()),
            "sim_95": [float(lo), float(hi)],
            "inside": bool(lo <= d26["C1_16+_nA"] <= hi),
        }
        out["best_gated"] = best["seed"]
        out["B24_best"] = b
        out["explains_diagnostics"] = all(
            b[k]["inside"] for k in ("n_shared", "D2_2", "D3_frac", "B26_16plus_Mpure")
        )
    json.dump(out, open(os.path.join(OUT, "e44_analysis.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    for x in rows:
        flags = "".join("1" if v else "0" for v in x["comp"].values())
        print(
            f"s{x['seed']:>2} gate {int(x['gated'])} loss {x['loss']:7.1f} gen {x['gens']} "
            f"pi {x['pi_M']:.3f} g_up {x['g_up']:.2f} s_G {x['s_G']:.2f} "
            f"GM/fG/fM/B7G/B7M {flags}"
        )


if __name__ == "__main__":
    main()
