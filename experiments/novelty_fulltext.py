"""Full-text unit scan of the citation-scan papers (2026-11-23; Bence: "Yes, broader scan").

Input is the list of open full-text ids from the citation scan, whose records already passed the neutral-wording filter
on title and abstract. Each full text is used for two things only: accession ids and stated unit counts. The text itself
is not saved. No data file is opened and no statistic is computed.
"""

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from novelty_units import full_text_facts  # noqa: E402

IDS = os.path.join(HERE, "novelty_fulltext_ids_2026-11-23.txt")
OUT = os.path.join(HERE, "results_lit", "novelty_fulltext_2026-11-23.json")


def main():
    with open(IDS, encoding="utf-8") as fh:
        ids = [line.strip() for line in fh if line.strip()]
    records = {}
    for pmcid in ids:
        records[pmcid] = full_text_facts(pmcid)
        time.sleep(0.1)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"n_ids": len(ids), "records": records}, fh, indent=1, sort_keys=True)
    got = [r for r in records.values() if r]
    print(json.dumps({"ids": len(ids), "full_text": len(got), "with_units": sum(bool(r["unit_counts"]) for r in got),
                      "with_accession": sum(bool(r["accessions"]) for r in got),
                      "units_and_accession": sum(bool(r["unit_counts"] and r["accessions"]) for r in got)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
