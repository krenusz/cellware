"""Paper half of the morphology-table scan, rerun with fixed queries (2026-11-24; Bence: "1").

morph_scan.py's paper half had weak recall: its query expansion swallowed closing brackets, and wildcards inside quoted
phrases match nothing, so three queries returned 0 hits and the recall paper was missed. This rerun uses a bracket-safe
expansion, no wildcards inside phrases, cursor paging up to 500 records per query, and a recall gate: the run exits 1 if
the known paper is missing. Abstracts from Europe PMC; from open full text only stated unit counts and accessions. A
record whose title or abstract hits the neutral-wording filter is dropped whole. No data file is opened and no
statistic is computed.
"""

import json
import os
import re
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FutureTimeout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lit_gap import CELL_LINE  # noqa: E402
from morph_scan import BASE, facts  # noqa: E402
from novelty_scan import API, FILTER, get  # noqa: E402

OUT = os.path.join(HERE, "results_lit", "morph_papers_2026-11-24.json")
QUERIES = [
    "TITLE_ABS:gastruloid*",
    "(TITLE_ABS:\"embryoid body\" OR TITLE_ABS:\"embryoid bodies\" OR TITLE_ABS:embryoid*) AND "
    "(TITLE_ABS:imaging OR TITLE_ABS:variability OR TITLE_ABS:screen*)",
    "TITLE_ABS:organoid* AND TITLE_ABS:reproducib* AND TITLE_ABS:morpholog*",
    "TITLE_ABS:organoid* AND TITLE_ABS:variability AND TITLE_ABS:morpholog*",
    "TITLE_ABS:organoid* AND TITLE_ABS:\"high-content\" AND TITLE_ABS:imaging",
    "TITLE_ABS:organoid* AND TITLE_ABS:heterogeneity AND TITLE_ABS:\"live imaging\"",
    "TITLE_ABS:\"stem cell\" AND TITLE_ABS:\"symmetry breaking\" AND TITLE_ABS:imaging",
    "TITLE_ABS:organoid* AND TITLE_ABS:\"high-throughput\" AND TITLE_ABS:imaging AND TITLE_ABS:phenotyp*",
    "(TITLE_ABS:somitoid* OR TITLE_ABS:segmentoid* OR TITLE_ABS:blastoid*) AND "
    "(TITLE_ABS:imaging OR TITLE_ABS:variability OR TITLE_ABS:efficiency)",
]
RECALL = ["10.1016/j.stem.2023.04.018"]
MAX_PER_QUERY, BUDGET = 500, 60 * 60


def expand(q):
    return re.sub(r'TITLE_ABS:("[^"]+"|[^\s()]+)', r"(TITLE:\1 OR ABSTRACT:\1)", q)


def search(q):
    rows, cursor, hit = [], "*", 0
    while len(rows) < MAX_PER_QUERY:
        params = {"query": expand(q) + BASE, "format": "json", "pageSize": 100, "resultType": "core", "cursorMark": cursor}
        r = get(API + "search?" + urllib.parse.urlencode(params))
        if not r:
            raise RuntimeError(f"query failed: {q}")
        hit = r.get("hitCount", 0)
        got = r.get("resultList", {}).get("result", [])
        rows += got
        nxt = r.get("nextCursorMark")
        if not got or not nxt or nxt == cursor:
            break
        cursor = nxt
        time.sleep(0.3)
    return hit, rows


def save(out, status):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({**out, "status": status}, fh, indent=1, sort_keys=True)


def main():
    recs, dropped, hits = {}, 0, {}
    for q in QUERIES:
        hits[q], rows = search(q)
        for w in rows:
            title, abstract = w.get("title") or "", w.get("abstractText") or ""
            key = (w.get("doi") or w.get("id") or "").lower()
            if FILTER.search(title) or FILTER.search(abstract):
                dropped += 1
                recs.pop(key, None)
                continue
            recs[key] = {"title": CELL_LINE.sub("[line]", title), "year": w.get("pubYear"), "pmcid": w.get("pmcid"),
                         "abstract": CELL_LINE.sub("[line]", abstract[:900])}
    missed = [d for d in RECALL if d not in recs]
    out = {"queries": hits, "dropped_by_filter": dropped, "recall_missed": missed, "records": recs}
    print(json.dumps({"kept": len(recs), "dropped": dropped, "hits": hits, "recall_missed": missed}), flush=True)
    if missed:
        save(out, "recall_gate_failed")
        sys.stdout.flush()
        os._exit(1)
    todo = [k for k, r in recs.items() if r["pmcid"]]
    pool = ThreadPoolExecutor(max_workers=6)
    futures = {pool.submit(facts, recs[k]["pmcid"]): k for k in todo}
    try:
        for i, f in enumerate(as_completed(futures, timeout=BUDGET), 1):
            recs[futures[f]]["facts"] = f.result()
            if i % 50 == 0:
                save(out, "partial")
                print(f"{i}/{len(todo)} full texts", flush=True)
    except FutureTimeout:
        pass
    timeouts = sum("facts" not in recs[k] for k in todo)
    save(out, "done" if not timeouts else "budget")
    got = [r["facts"] for r in recs.values() if r.get("facts")]
    big = [r for r in got if any(v >= 50 for u, v in r["unit_counts"].items() if u in ("organoids", "gastruloids", "embryos"))]
    print(json.dumps({"with_pmcid": len(todo), "timeouts": timeouts, "full_text": len(got), "units_ge50": len(big),
                      "units_ge50_and_accession": sum(bool(r["accessions"]) for r in big)}))
    sys.stdout.flush()
    os._exit(0)  # abandon any hung fetch thread


if __name__ == "__main__":
    sys.exit(main())
