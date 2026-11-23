"""CAND-I-T0: structure of the train fish only (pre-registered in EXPERIMENT_LOG, 2026-11-23; Bence: "Yes, train look").

Counts distinct barcodes, sample indices and replicates per eligible train fish, and barcodes per sample replicate.
Non-train rows are removed by partitions.train_only first. The read and proportion columns are never parsed.
"""

import csv
import hashlib
import io
import json
import os
import re
import sys
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from partitions import train_only  # noqa: E402

OUT = os.path.join(HERE, "results_cand", "candi_t0_struct.json")
COMMIT = "99b327ca17bfd092063dbf911e088f993e036172"
SHA256 = "6985096f252b99bd2e53e98daef23089a40bdc52345db4eaf6095105fc08de80"
URL = f"https://raw.githubusercontent.com/Gagnon-lab/weber-shrader/{COMMIT}/experiments/full_pipeline/final_data.csv"
ID = re.compile(r"fish(\d+)_s(\d+)_rep(\d+)", re.I)


def main():
    req = urllib.request.Request(URL, headers={"User-Agent": "cellware-seal/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise SystemExit("source table changed: sha256 mismatch")
    rows = csv.reader(io.StringIO(raw.decode("utf-8", "ignore")))
    bc = next(rows).index("barcode")
    ids = []
    for row in rows:
        m = next((m for c in row if (m := ID.search(c))), None)
        if m:
            ids.append({"unit": str(int(m.group(1))), "sample": int(m.group(2)), "rep": int(m.group(3)), "bc": row[bc]})
    df = train_only("germline_fish", pd.DataFrame(ids), "unit")
    fish = {}
    for unit, g in df.groupby("unit"):
        if g["sample"].nunique() < 6:
            continue
        per_rep = g.groupby(["sample", "rep"])["bc"].nunique()
        fish[unit] = {"barcodes": int(g["bc"].nunique()), "samples": int(g["sample"].nunique()),
                      "sample_replicates": int(len(per_rep)), "barcodes_per_replicate_median": float(per_rep.median()),
                      "barcodes_per_replicate_min": int(per_rep.min()), "barcodes_per_replicate_max": int(per_rep.max())}
    out = {"commit": COMMIT, "n_eligible_train_fish": len(fish), "fish": fish}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
