"""CAND-I-P1 seal: H24 balanced partition of the germline barcode set by fish (pre-registered in EXPERIMENT_LOG).

Bence 2026-11-23: "Yes, seal CAND-I". Downloads one table at the pinned commit, checks its sha256 and parses sample
identifiers only. Unit = numbered fish. Size = number of distinct sample indices. Strata = at least 6 sample indices or
fewer. Reads, proportions, barcodes and rows per fish are never used. The three control fish are not numbered, are
outside the map and are never read.
"""

import csv
import hashlib
import io
import json
import os
import re
import sys
import urllib.request
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from partitions import balanced_split  # noqa: E402

OUT = os.path.join(HERE, "results_cand", "candi_seal.json")
COMMIT = "99b327ca17bfd092063dbf911e088f993e036172"
SHA256 = "6985096f252b99bd2e53e98daef23089a40bdc52345db4eaf6095105fc08de80"
URL = f"https://raw.githubusercontent.com/Gagnon-lab/weber-shrader/{COMMIT}/experiments/full_pipeline/final_data.csv"
SALT = "cellware-germline-fish-v1"
ID = re.compile(r"fish(\d+)_s(\d+)_rep(\d+)", re.I)
MIN_TIMEPOINTS = 6


def seal(sample_ids):
    per_fish = defaultdict(set)
    for fish, sample, _rep in sample_ids:
        per_fish[str(fish)].add(sample)
    sizes = {f: len(s) for f, s in per_fish.items()}
    strata = {f: "a_ge6" if n >= MIN_TIMEPOINTS else "b_short" for f, n in sizes.items()}
    return balanced_split(sizes, SALT, strata), sizes, strata


def main():
    req = urllib.request.Request(URL, headers={"User-Agent": "cellware-seal/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise SystemExit("source table changed: sha256 mismatch")
    ids = set()
    for row in csv.reader(io.StringIO(raw.decode("utf-8", "ignore"))):
        for cell in row:
            m = ID.search(cell)
            if m:
                ids.add(tuple(int(x) for x in m.groups()))
                break
    mapping, sizes, strata = seal(ids)
    if len(mapping) != 24:
        raise SystemExit(f"expected 24 numbered fish, got {len(mapping)}")
    eligible = Counter(mapping[f] for f in mapping if strata[f] == "a_ge6")
    sample_indices = Counter()
    for f, part in mapping.items():
        sample_indices[part] += sizes[f]
    out = {"commit": COMMIT, "sha256": SHA256, "salt": SALT, "map": dict(sorted(mapping.items(), key=lambda x: int(x[0]))),
           "fish_per_partition": dict(Counter(mapping.values())), "eligible_fish_per_partition": dict(eligible),
           "sample_indices_per_partition": dict(sample_indices)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print(json.dumps({k: v for k, v in out.items() if k != "map"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
