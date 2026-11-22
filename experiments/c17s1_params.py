"""C17-S1 parameter table: the general model's parameters per system (Bence 2026-11-19: "1").

Serves: C17-S1 (C17-S1R: the general model is built system by system). Used estimates only, copied from
EXPERIMENT_LOG.md; each row names a phrase that must appear in the log or the cited script (smoke against
transcription errors). No data are read and nothing is simulated.

Parameters: rho_div (memory retained per division), rho_event (per split/passage event), tau (instruction
shift, logit units), gamma (selection, log growth per unit state), heritable_share (cross-condition sharing),
split_predictability (state-to-fate, excess kernel predictability; not on the heritable-share scale).

    uv run python experiments/c17s1_params.py   # writes results_c17/c17s1_params.json
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "EXPERIMENT_LOG.md")
OUT = os.path.join(HERE, "results_c17", "c17s1_params.json")
BLOOD, PLURI, NEURO = "committed blood progenitor", "pluripotent", "neural progenitor"

# system, class, parameter, point, lo, hi, identified, source phrase (log or c3s1 script)
ROWS = [
    ("LARRY cytokine wells", BLOOD, "heritable_share", 0.88, 0.82, 0.92, True, "0.88 [0.82, 0.92]"),
    ("LARRY cytokine wells", BLOOD, "rho_div", 0.964, 0.927, 0.997, True, "rho_div 0.964 [0.927,"),
    ("LARRY cytokine wells", BLOOD, "tau_G", 0.52, 0.38, 0.70, True, "| instruction shift G | 0.52 | 0.38-0.70"),
    ("LARRY cytokine wells", BLOOD, "tau_M", 0.70, 0.40, 0.92, True, "| instruction shift M | 0.70 | 0.40-0.92"),
    ("LARRY cytokine wells", BLOOD, "gamma", 0.560, 0.482, 0.568, True, "| gamma | 0.560 | 0.482-0.568"),
    ("SIS-seq / SIS-skew", BLOOD, "heritable_share", 0.71, 0.59, 0.78, True, "SIS-seq heritable share 0.71 [0.59, 0.78]"),
    ("SIS-seq / SIS-skew", BLOOD, "gamma_cDC2", 0.34, -0.28, 0.89, False, "gamma_cDC2 0.34 [-0.28,"),
    ("SIS-seq / SIS-skew", BLOOD, "gamma_pDC", 0.10, -0.54, 0.78, False, "gamma_pDC 0.10 [-0.54, 0.78]"),
    ("SIS-seq / SIS-skew", BLOOD, "tau_cDC2", -1.18, -2.00, 0.71, False, "tau + label difference: cDC2 -1.18"),
    # C7-S2 pre-registered primary (>= 4 cells per well), train only; added 2026-11-19 (Bence: "go 1 2")
    ("CellTag-multi LSK culture", BLOOD, "heritable_share", 0.66, 0.58, 0.75, True, "H 0.66 [0.58, 0.75]"),
    ("Raj gastruloids", PLURI, "rho_div", 1.00, 1.00, 1.00, True, "no per-division shrink at all is allowed"),
    # C17-S1P, train embryos only, pooled over stages (rises E7.5 -0.01 to E9.5 0.89); shared-niche upper bound
    ("PEtracer embryo trees (train)", PLURI, "heritable_share", 0.81, 0.53, 0.87, True, "H 0.81 [0.53, 0.87]"),
    # C17-S1C-S1 + amendment 1, train clones only; state-to-fate predictability (not a heritable share)
    ("CRISPR brain split clones (train)", NEURO, "split_predictability", 0.22, 0.17, 0.25, True,
     "P_excess 0.218, 95% CI [0.166, 0.247]"),
    ("Raj gastruloids", PLURI, "rho_event_passage", 0.46, 0.23, 0.68, True, "rho_P = 0.21 per passage (retention 0.46"),
    ("Raj gastruloids", PLURI, "rho_context_gastruloid", 0.06, None, None, True, "rho_g = 0.06"),
    ("Raj gastruloids", PLURI, "gamma", None, None, None, False, "Growth/selection gamma = 0 (not identified here"),
    # C17-S1PL-T/T3/T4 + P3, train founder clades of one embryo; added 2026-11-21 (Bence: "1").
    # Memory per division edge: stable estimate, but the nested test is NOT SUPPORTED in all three powered variants
    # (adds little beyond commitment), so not counted as identified; lo/hi = leave-one-out range, not a CI.
    ("DNA Typewriter embryo tree (train)", PLURI, "rho_div_edge", 0.899, 0.892, 0.904, False,
     "lambda 0.899 on all data, leave-one-out 0.892-0.904"),
    # Commitment by split date (T3 clock): commitment-only beats memory-only pooled in T, T3 and T4.
    ("DNA Typewriter embryo tree (train)", PLURI, "commitment_C_at_9.4_days", 0.48, None, None, True,
     "Commitment C rises 0.21 -> 0.48 -> 0.81"),
]
COLS = ["system", "cell_class", "parameter", "point", "lo", "hi", "identified", "source"]

# Scale of each row (RETRO_2026-11-20 Went wrong 1; adopted 2026-11-22, Bence: "go 1 2 3"). A cross-system law test
# compares rows of ONE scale only and counts systems per class per scale.
MEM_DIV, MEM_EVENT, HERIT, COMMIT, POSITION, INSTR, SELECT = (
    "memory_per_division", "memory_per_event", "heritable_share", "commitment_dominated", "position_dominated",
    "instruction", "selection")
SCALE = {
    ("LARRY cytokine wells", "heritable_share"): HERIT,
    ("LARRY cytokine wells", "rho_div"): MEM_DIV,
    ("LARRY cytokine wells", "tau_G"): INSTR,
    ("LARRY cytokine wells", "tau_M"): INSTR,
    ("LARRY cytokine wells", "gamma"): SELECT,
    ("SIS-seq / SIS-skew", "heritable_share"): HERIT,
    ("SIS-seq / SIS-skew", "gamma_cDC2"): SELECT,
    ("SIS-seq / SIS-skew", "gamma_pDC"): SELECT,
    ("SIS-seq / SIS-skew", "tau_cDC2"): INSTR,
    ("CellTag-multi LSK culture", "heritable_share"): HERIT,
    ("Raj gastruloids", "rho_div"): MEM_DIV,
    # C17-S1T: the embryo-tree H is mostly commitment, so it is not on the heritable-share scale of the blood rows
    ("PEtracer embryo trees (train)", "heritable_share"): COMMIT,
    # C17-S1C-S1: likely regional identity kept by descent
    ("CRISPR brain split clones (train)", "split_predictability"): POSITION,
    ("Raj gastruloids", "rho_event_passage"): MEM_EVENT,
    ("Raj gastruloids", "rho_context_gastruloid"): MEM_EVENT,
    ("Raj gastruloids", "gamma"): SELECT,
    ("DNA Typewriter embryo tree (train)", "rho_div_edge"): MEM_DIV,
    ("DNA Typewriter embryo tree (train)", "commitment_C_at_9.4_days"): COMMIT,
}


def main():
    text = open(LOG, encoding="utf-8").read()
    text += open(os.path.join(HERE, "c3s1_core_gastruloid.py"), encoding="utf-8").read()
    text = " ".join(text.split())
    missing = [r[-1] for r in ROWS if " ".join(r[-1].split()) not in text]
    assert not missing, f"source phrases not found: {missing}"
    assert set(SCALE) == {(r[0], r[2]) for r in ROWS} and len(SCALE) == len(ROWS), "every row needs one scale"
    table = [{**dict(zip(COLS, r, strict=True)), "scale": SCALE[(r[0], r[2])]} for r in ROWS]
    by_class, by_scale = {}, {}
    for r in table:
        if r["identified"]:
            by_class.setdefault(r["cell_class"], set()).add(r["system"])
            by_scale.setdefault(r["scale"], {}).setdefault(r["cell_class"], set()).add(r["system"])
    summary = {c: sorted(s) for c, s in by_class.items()}
    per_scale = {sc: {c: sorted(s) for c, s in d.items()} for sc, d in sorted(by_scale.items())}
    gate = {sc: {c: len(s) >= 3 for c, s in d.items()} for sc, d in per_scale.items()}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"rows": table, "systems_with_identified_parameters_by_class": summary,
                   "systems_by_scale_and_class": per_scale,
                   "law_test_gate_ge_3_systems_per_class_per_scale": gate,
                   "note": "the law-test gate counts systems per class within one scale (2026-11-22); the class-level "
                           "list is context only"}, fh, indent=1)
    print(json.dumps({"by_scale": per_scale, "gate": gate}, indent=1))


if __name__ == "__main__":
    main()
