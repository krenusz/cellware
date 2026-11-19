"""C17-S1M: MADM organoid clones (GSE327468). Stage `struct`: S0 structure gate before sealing.

Bence 2026-11-19: "Go MADM". Serves: C17-S1 (C17-S1R: grow the systems). Pre-registered in EXPERIMENT_LOG (C17-S1M).

Structure only (H9): per-cell tags `cloneid`, `color` (G / R / NA = the two MADM sister sublineages) and `plate` from
the GEO sample records; cells per clone per colour. The authors' `clone type` tag (an outcome class) is not
tabulated. No expression or fate is read. The log prints only totals.

    python experiments/c17s1m_madm.py struct     # writes results_c17/c17s1m_struct.json (Actions)
    python experiments/c17s1m_madm.py selftest   # offline check of the parser and the gate
"""

from __future__ import annotations

import collections
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_c17", "c17s1m_struct.json")
GSE = "GSE327468"
UA = {"User-Agent": "cellware-struct/1.0 (public metadata, research use)"}
MIN_ARM = 2  # cells per colour (C7-S2a: smallest validated sister-arm size)
TRAIN_SHARE, MIN_TRAIN = 0.4, 30  # H11 split, H21 readout gate


def soft(acc, targ):
    url = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?" + urllib.parse.urlencode(
        {"acc": acc, "targ": targ, "form": "text", "view": "brief"})
    for i in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90) as r:
                return r.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            time.sleep(3 * (i + 1))
    raise RuntimeError("GEO fetch failed")


def cells(text):
    """One dict per sample: gsm, cloneid, color, plate (structure tags only)."""
    out = []
    for block in re.split(r"^\^SAMPLE = ", text, flags=re.M)[1:]:
        gsm = block.split("\n", 1)[0].strip()
        tags = {}
        for ln in block.splitlines():
            if ln.startswith("!Sample_characteristics_ch1 = "):
                k, _, v = ln.split(" = ", 1)[1].partition(":")
                k = k.strip().lower()
                if k in ("cloneid", "color", "plate"):
                    tags[k] = v.strip()
        out.append({"gsm": gsm, **tags})
    return out


def gate(rows):
    per = collections.defaultdict(lambda: collections.Counter())
    plate = {}
    for r in rows:
        cid = r.get("cloneid")
        if not cid or cid.upper() == "NA":
            continue
        per[cid][(r.get("color") or "NA").upper()] += 1
        plate.setdefault(cid, set()).add(r.get("plate", "NA"))
    eligible = [c for c, k in per.items() if k["G"] >= MIN_ARM and k["R"] >= MIN_ARM]
    both = [c for c, k in per.items() if k["G"] >= 1 and k["R"] >= 1]
    n_train = int(TRAIN_SHARE * len(eligible))
    return {
        "cells": len(rows),
        "cells_with_clone": sum(sum(k.values()) for k in per.values()),
        "clones": len(per),
        "clones_with_both_colours": len(both),
        "eligible_clones_ge2_each": len(eligible),
        "expected_train_eligible": n_train,
        "gate_pass": n_train >= MIN_TRAIN,
        "arm_size_table": dict(sorted(collections.Counter(
            f"G{min(k['G'], 5)}_R{min(k['R'], 5)}" for k in per.values()).items())),
        "clones_on_several_plates": sum(len(p) > 1 for p in plate.values()),
        "per_clone": {c: {"G": k["G"], "R": k["R"], "NA": k["NA"], "plates": sorted(plate[c])}
                      for c, k in sorted(per.items())},
    }


def selftest():
    txt = ""
    for i, (cid, col) in enumerate([("c1", "G"), ("c1", "G"), ("c1", "R"), ("c1", "R"), ("c2", "G"), ("c2", "R"),
                                    ("NA", "NA")]):
        txt += (f"^SAMPLE = GSM{i}\n!Sample_title = cell {i}\n!Sample_characteristics_ch1 = cloneid: {cid}\n"
                f"!Sample_characteristics_ch1 = color: {col}\n!Sample_characteristics_ch1 = plate: 1\n"
                f"!Sample_characteristics_ch1 = clone type: N\n")
    rows = cells(txt)
    assert len(rows) == 7 and "clone type" not in rows[0], rows[0]
    g = gate(rows)
    assert g["clones"] == 2 and g["eligible_clones_ge2_each"] == 1 and not g["gate_pass"], g
    assert g["clones_with_both_colours"] == 2
    return {"ok": True}


def main():
    if sys.argv[1:2] == ["selftest"]:
        print(json.dumps(selftest()))
        return
    rows = cells(soft(GSE, "gsm"))
    res = gate(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"gse": GSE, "rule": f">= {MIN_ARM} cells per colour; pass if {TRAIN_SHARE} x eligible >= {MIN_TRAIN}",
               **res}, open(OUT, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_clone"}, indent=1))


if __name__ == "__main__":
    main()
