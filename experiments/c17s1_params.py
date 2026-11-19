"""C17-S1 parameter table: the general model's parameters per system (Bence 2026-11-19: "1").

Serves: C17-S1 (C17-S1R: the general model is built system by system). Used estimates only, copied from
EXPERIMENT_LOG.md; each row names a phrase that must appear in the log or the cited script (smoke against
transcription errors). No data are read and nothing is simulated.

Parameters: rho_div (memory retained per division), rho_event (per split/passage event), tau (instruction
shift, logit units), gamma (selection, log growth per unit state), heritable_share (cross-condition sharing).

    uv run python experiments/c17s1_params.py   # writes results_c17/c17s1_params.json
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "EXPERIMENT_LOG.md")
OUT = os.path.join(HERE, "results_c17", "c17s1_params.json")
BLOOD, PLURI = "committed blood progenitor", "pluripotent"

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
    ("Raj gastruloids", PLURI, "rho_div", 1.00, 1.00, 1.00, True, "no per-division shrink at all is allowed"),
    ("Raj gastruloids", PLURI, "rho_event_passage", 0.46, 0.23, 0.68, True, "rho_P = 0.21 per passage (retention 0.46"),
    ("Raj gastruloids", PLURI, "rho_context_gastruloid", 0.06, None, None, True, "rho_g = 0.06"),
    ("Raj gastruloids", PLURI, "gamma", None, None, None, False, "Growth/selection gamma = 0 (not identified here"),
]
COLS = ["system", "cell_class", "parameter", "point", "lo", "hi", "identified", "source"]


def main():
    text = open(LOG, encoding="utf-8").read()
    text += open(os.path.join(HERE, "c3s1_core_gastruloid.py"), encoding="utf-8").read()
    text = " ".join(text.split())
    missing = [r[-1] for r in ROWS if " ".join(r[-1].split()) not in text]
    assert not missing, f"source phrases not found: {missing}"
    table = [dict(zip(COLS, r, strict=True)) for r in ROWS]
    by_class = {}
    for r in table:
        if r["identified"]:
            by_class.setdefault(r["cell_class"], set()).add(r["system"])
    summary = {c: sorted(s) for c, s in by_class.items()}
    gate = {c: len(s) >= 3 for c, s in summary.items()}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"rows": table, "systems_with_identified_parameters_by_class": summary,
                   "law_test_gate_ge_3_systems_per_class": gate}, fh, indent=1)
    print(json.dumps({"by_class": summary, "gate": gate}, indent=1))


if __name__ == "__main__":
    main()
