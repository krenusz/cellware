"""C15-S0: H17 simulated power of the planned per-host test at each sealed partition's structural counts
(pre-registered in EXPERIMENT_LOG). Input: results_c15/c15s0_struct.json only (structure, no outcome).

Run: python -I experiments/c15s0_power.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from power import sim_power  # noqa: E402

OUT = HERE / "results_c15"
SD = 0.0374  # C14 train embryo sd of LOEO delta AUC (C14-S1b)
REF_CELLS = 163305 / 6  # C14 train readout cells per embryo (C14-S5b)
EFFECTS = {"shrunk 0.0556": 0.0556, "half 0.028": 0.028}
MIN_LEAVES = 200
N_SIM = 20000
SEED = 20261117


def host_sds(leaves, case):
    if case == "optimistic":
        return np.full(len(leaves), SD)
    return np.maximum(SD, SD * np.sqrt(REF_CELLS / np.maximum(np.asarray(leaves, float), 1)))


def power(leaves, effect, case, rng):
    leaves = [x for x in leaves if x >= MIN_LEAVES]
    n = len(leaves)
    if n < 2:
        return 0.0, n
    sds = host_sds(leaves, case)
    draws = effect + rng.normal(0, 1, (N_SIM, n)) * sds
    own = draws.std(1, ddof=1)
    z = draws.mean(1) / (np.maximum(SD, own) / np.sqrt(n))
    return sim_power(z, 1.0), n


def main():
    s = json.loads((OUT / "c15s0_struct.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(SEED)
    grid, lines = {}, []
    for p in ["A", "B", "C"]:
        lv = s["partitions"][p]["leaves_per_host"]
        for e_name, e in EFFECTS.items():
            for case in ["optimistic", "pessimistic"]:
                pw, n = power(lv, e, case, rng)
                grid[f"{p} | {e_name} | {case}"] = dict(power=round(pw, 3), hosts_used=n)
                lines.append(f"{p} ({n} hosts >= {MIN_LEAVES} leaves) | effect {e_name} | sd {case}: power {pw:.3f}")
    ok = [p for p in "ABC" if grid[f"{p} | shrunk 0.0556 | pessimistic"]["power"] >= 0.8]
    verdict = f"PASS (powered partitions: {', '.join(ok)})" if ok else "FAIL (no sealed partition >= 0.8 at the shrunk effect, pessimistic sd)"
    lines.append(f"H19 power gate: {verdict}")
    print("\n".join(lines))
    (OUT / "c15s0_power.json").write_text(json.dumps(dict(verdict=verdict, grid=grid, sd=SD, ref_cells=REF_CELLS,
                                                          min_leaves=MIN_LEAVES), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
