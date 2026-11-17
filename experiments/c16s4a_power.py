"""C16-S4a: H17 power for a one-look confirmation of C16 claims on the embryo set's partition B (pre-registered
in EXPERIMENT_LOG). Arithmetic on committed train results only; reads no data set.

Run: python -I experiments/c16s4a_power.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import power  # noqa: E402

R = HERE / "results_c16"
ALPHA = 0.025 / 3
DRAWS = 100_000
SEED = 20261122
STAGE_MATCHED = ["E8.5-R2", "E8.5-R3", "E9.0-R1", "E9.5-R1"]


def load():
    s1 = json.loads((R / "c16s1_time.json").read_text(encoding="utf-8"))
    ci = json.loads((R / "c16s3c_i.json").read_text(encoding="utf-8"))
    a = json.loads((R / "c16s3a_position.json").read_text(encoding="utf-8"))
    return {
        "K1 hidden state beyond depth-varying rates (S1)": (
            s1["primary"], float(np.std(list(s1["smoke"]["no_memory"].values()), ddof=1))),
        "K2 family bias beyond within-embryo position (S3c-i)": (
            {e: r["delta"] for e, r in ci["per_embryo"].items() if r.get("in_rule")}, ci["smoke"]["limit"] * np.sqrt(6) / 2),
        "K3 readable in expression beyond axial identity (S3a)": (
            {e: r["delta"] for e, r in a["per_embryo"].items() if r.get("in_rule")}, a["smoke"]["limit"] * np.sqrt(6) / 2),
    }


def predictive(vals, rng):
    v = np.asarray(vals, float)
    n = len(v)
    return v.mean() + v.std(ddof=1) * np.sqrt(1 + 1 / n) * rng.standard_t(n - 1, DRAWS)


def main():
    rng = np.random.default_rng(SEED)
    out, lines = {}, []
    crit_z = power.z_crit(ALPHA, True)
    for name, (per, null_sd) in load().items():
        row = dict(train=per, null_sd=null_sd, critical=crit_z * null_sd)
        for scen, keys in (("all 6", list(per)), ("stage-matched E8.5-E9.5", [e for e in STAGE_MATCHED if e in per])):
            vals = [per[e] for e in keys]
            p = power.sim_power(predictive(vals, rng), null_sd, alpha=ALPHA, one_sided=True)
            row[scen] = dict(n=len(vals), mean=float(np.mean(vals)), sd=float(np.std(vals, ddof=1)), power=p)
        row["passes"] = all(row[s]["power"] >= 0.8 for s in ("all 6", "stage-matched E8.5-E9.5"))
        out[name] = row
        lines.append(f"{name}: train " + ", ".join(f"{e} {v:+.4f}" for e, v in per.items()))
        lines.append(f"  null sd {null_sd:.4f}, critical value {row['critical']:.4f} (z {crit_z:.2f}, alpha {ALPHA:.4f} one-sided)")
        for s in ("all 6", "stage-matched E8.5-E9.5"):
            r = row[s]
            lines.append(f"  {s}: mean {r['mean']:+.4f}, sd {r['sd']:.4f}, n {r['n']} -> power {r['power']:.3f}")
        lines.append(f"  -> {'ENTERS the B look' if row['passes'] else 'does NOT enter (power < 0.8)'}")
    passing = [k for k, v in out.items() if v["passes"]]
    lines.append(f"claims passing the gate: {len(passing)}/3" + (f" ({', '.join(k.split()[0] for k in passing)})" if passing else ""))
    print("\n".join(lines))
    (R / "c16s4a_power.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    (R / "c16s4a_power.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
