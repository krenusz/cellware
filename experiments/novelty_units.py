"""Candidate scan for designs with many independent units (2026-11-23; Bence: "search for more cands in the mt").

Metadata and abstracts from Europe PMC. A record whose title or abstract hits the neutral-wording filter is dropped
whole. For the rest, the open full text is used for two things only: accession ids, and stated counts of animals,
donors, embryos, organoids or gastruloids. No data file is opened and no statistic is computed.
"""

import json
import os
import re
import sys
import time
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lit_gap import CELL_LINE  # noqa: E402
from novelty_scan import ACC, API, BASE, CLAIM, FILTER, expand, get  # noqa: E402

OUT = os.path.join(HERE, "results_lit", "novelty_units_2026-11-23.json")
QUERIES = [
    'TITLE_ABS:"clone size" AND TITLE_ABS:"lineage tracing" AND TITLE_ABS:longitudinal',
    'TITLE_ABS:barcod* AND TITLE_ABS:"clonal dynamics"',
    'TITLE_ABS:"fluctuation test" AND TITLE_ABS:heritable',
    'TITLE_ABS:"fluctuation analysis" AND TITLE_ABS:"single-cell"',
    'TITLE_ABS:barcod* AND TITLE_ABS:organoid AND TITLE_ABS:clonal',
    'TITLE_ABS:gastruloid AND TITLE_ABS:variability',
    'TITLE_ABS:gastruloid AND TITLE_ABS:clonal',
    'TITLE_ABS:"neutral drift" AND TITLE_ABS:clone*',
    'TITLE_ABS:"clonal competition" AND TITLE_ABS:barcod*',
    'TITLE_ABS:"stem cell" AND TITLE_ABS:barcod* AND TITLE_ABS:"time series"',
    'TITLE_ABS:"clone size" AND TITLE_ABS:neutral AND TITLE_ABS:"stem cell"',
    'TITLE_ABS:zebrafish AND TITLE_ABS:barcod* AND TITLE_ABS:clonal',
]
UNITS = re.compile(r"\b(?:n\s*=\s*)?(\d{2,4})\s+(?:\w+\s+){0,2}?(mice|donors|animals|fish|embryos|organoids|gastruloids)\b", re.I)
MIN_UNITS, MAX_UNITS = 20, 5000


def full_text_facts(pmcid):
    text = get(f"{API}{pmcid}/fullTextXML", as_json=False) if pmcid else None
    if not text:
        return None
    plain = re.sub(r"<[^>]+>", " ", text)
    counts = {}
    for m in UNITS.finditer(plain):
        n = int(m.group(1))
        if MIN_UNITS <= n <= MAX_UNITS:
            unit = m.group(2).lower()
            counts[unit] = max(counts.get(unit, 0), n)
    acc = sorted({(m.group(1) or "zenodo." + m.group(2)) for m in ACC.finditer(text)})
    return {"unit_counts": counts, "accessions": acc}


def main():
    kept, dropped, hits = {}, 0, {}
    for q in QUERIES:
        params = {"query": expand(q) + BASE, "format": "json", "pageSize": 100, "resultType": "core", "sort": "CITED desc"}
        r = get(API + "search?" + urllib.parse.urlencode(params))
        if not r:
            raise RuntimeError(f"query failed: {q}")
        hits[q] = r.get("hitCount", 0)
        for w in r.get("resultList", {}).get("result", []):
            title, abstract = w.get("title") or "", w.get("abstractText") or ""
            if FILTER.search(title) or FILTER.search(abstract):
                dropped += 1
                continue
            key = (w.get("doi") or w.get("id") or "").lower()
            kept[key] = {"title": CELL_LINE.sub("[line]", title), "year": w.get("pubYear"), "cites": w.get("citedByCount", 0),
                         "pmcid": w.get("pmcid"), "claim_words": len({m.lower() for m in CLAIM.findall(abstract)}),
                         "abstract": CELL_LINE.sub("[line]", abstract)}
        time.sleep(0.3)
    for rec in kept.values():
        rec["facts"] = full_text_facts(rec["pmcid"])
        time.sleep(0.1)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"queries": hits, "dropped_by_filter": dropped, "records": kept}, fh, indent=1, sort_keys=True)
    lead = sum(bool(r["facts"] and r["facts"]["unit_counts"] and r["facts"]["accessions"]) for r in kept.values())
    print(json.dumps({"kept": len(kept), "dropped_by_filter": dropped, "open_full_text": sum(bool(r["facts"]) for r in kept.values()),
                      "units_and_accession": lead}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
