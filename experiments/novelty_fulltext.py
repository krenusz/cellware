"""Full-text unit scan of the citation-scan papers (2026-11-23; Bence: "Yes, broader scan").

Input is the list of open full-text ids from the citation scan, whose records already passed the neutral-wording filter
on title and abstract. Each full text is used for two things only: accession ids and stated unit counts. The text itself
is not saved. No data file is opened and no statistic is computed.

Run 45275946669 hung until the 6-hour job limit with no output, so six papers are fetched at once, the whole scan has a
90 min budget, and the output is saved every 20 papers.
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FutureTimeout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from novelty_units import full_text_facts  # noqa: E402

IDS = os.path.join(HERE, "novelty_fulltext_ids_2026-11-23.txt")
OUT = os.path.join(HERE, "results_lit", "novelty_fulltext_2026-11-23.json")
BUDGET = 90 * 60


def save(ids, records, status):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"n_ids": len(ids), "status": status, "records": records}, fh, indent=1, sort_keys=True)


def main():
    with open(IDS, encoding="utf-8") as fh:
        ids = [line.strip() for line in fh if line.strip()]
    records, start = {}, time.time()
    pool = ThreadPoolExecutor(max_workers=6)
    futures = {pool.submit(full_text_facts, pmcid): pmcid for pmcid in ids}
    try:
        for i, f in enumerate(as_completed(futures, timeout=BUDGET), 1):
            records[futures[f]] = f.result()
            if i % 20 == 0:
                save(ids, records, "partial")
                print(f"{i}/{len(ids)} done, {time.time() - start:.0f}s", flush=True)
    except FutureTimeout:
        pass
    for pmcid in ids:
        records.setdefault(pmcid, "timeout")
    save(ids, records, "done" if "timeout" not in records.values() else "budget")
    got = [r for r in records.values() if isinstance(r, dict)]
    print(json.dumps({"ids": len(ids), "timeouts": sum(r == "timeout" for r in records.values()),
                      "full_text": len(got), "with_units": sum(bool(r["unit_counts"]) for r in got),
                      "with_accession": sum(bool(r["accessions"]) for r in got),
                      "units_and_accession": sum(bool(r["unit_counts"] and r["accessions"]) for r in got)}))
    sys.stdout.flush()
    os._exit(0)  # abandon any hung fetch thread


if __name__ == "__main__":
    sys.exit(main())
