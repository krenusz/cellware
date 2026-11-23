"""CAND-I-P1: structure of the public germline barcode tables (pre-registered in EXPERIMENT_LOG, 2026-11-23).

Reads the header and sample identifiers of one table at a pinned commit. Writes column names and counts of fish,
time points and technical replicates. No barcode frequency, clone size or drift value is computed or written.
"""

import csv
import hashlib
import io
import json
import os
import re
import sys
import urllib.request
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_cand", "candi_p1_struct.json")
COMMIT = "99b327ca17bfd092063dbf911e088f993e036172"
URL = f"https://raw.githubusercontent.com/Gagnon-lab/weber-shrader/{COMMIT}/experiments/full_pipeline/final_data.csv"
ID = re.compile(r"fish(\d+)_s(\d+)_rep(\d+)", re.I)


def main():
    req = urllib.request.Request(URL, headers={"User-Agent": "cellware-seal/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read()
    rows = csv.reader(io.StringIO(raw.decode("utf-8", "ignore")))
    header = next(rows)
    ids, n = set(), 0
    for row in rows:
        n += 1
        for cell in row:
            m = ID.search(cell)
            if m:
                ids.add(tuple(int(x) for x in m.groups()))
                break
    ids |= {tuple(int(x) for x in m.groups()) for c in header if (m := ID.search(c))}
    fish = defaultdict(lambda: defaultdict(set))
    for f, s, rep in ids:
        fish[f][s].add(rep)
    tp = sorted(len(v) for v in fish.values())
    reps = [len(r) for v in fish.values() for r in v.values()]
    out = {"commit": COMMIT, "sha256": hashlib.sha256(raw).hexdigest(), "columns": header[:40], "n_columns": len(header),
           "n_rows": n, "n_fish": len(fish), "timepoints_per_fish": tp, "fish_with_ge6_timepoints": sum(x >= 6 for x in tp),
           "samples": len(reps), "samples_with_3_reps": sum(x == 3 for x in reps),
           "samples_with_ge2_reps": sum(x >= 2 for x in reps)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print(json.dumps({k: v for k, v in out.items() if k != "columns"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
