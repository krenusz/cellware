"""Novelty-first scan for correction-type gaps (2026-11-23; Bence: "go 1 and list the still open candidates").

Metadata and abstracts of recent open-access clonal and lineage papers from Europe PMC. A record whose title or
abstract hits the neutral-wording filter of lit_gap.py is dropped whole, before anything is kept. For the records
left, the open full text is used for one thing only: the list of GEO, Zenodo and ArrayExpress accessions it names.
No data file is opened and no statistic is computed.
"""

import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lit_gap import CELL_LINE, SENSITIVE  # noqa: E402

OUT = os.path.join(HERE, "results_lit", "novelty_scan_2026-11-23b.json")
API = "https://www.ebi.ac.uk/europepmc/webservices/rest/"
UA = {"User-Agent": "cellware-novelty/1.0 (public metadata, research use)"}
BASE = " AND FIRST_PDATE:[2023-01-01 TO 2026-11-23]"
QUERIES = [
    'TITLE_ABS:"lineage barcoding" AND TITLE_ABS:clonal',
    'TITLE_ABS:"lineage tracing" AND TITLE_ABS:"single-cell" AND TITLE_ABS:"clone size"',
    'TITLE_ABS:clonal AND TITLE_ABS:"fate bias" AND TITLE_ABS:barcod*',
    'TITLE_ABS:"clonal expansion" AND TITLE_ABS:"lineage tracing" AND TITLE_ABS:fate',
    'TITLE_ABS:"Perturb-seq" AND TITLE_ABS:lineage',
    'TITLE_ABS:CRISPR AND TITLE_ABS:"lineage tracing" AND TITLE_ABS:clonal AND TITLE_ABS:differentiation',
    'TITLE_ABS:clonal AND TITLE_ABS:heritable AND TITLE_ABS:transcriptional AND TITLE_ABS:memory',
    'TITLE_ABS:"sister cells" AND TITLE_ABS:fate AND TITLE_ABS:"single-cell"',
    'TITLE_ABS:LARRY AND TITLE_ABS:barcod*',
    'TITLE_ABS:CellTag',
    'TITLE_ABS:clonal AND TITLE_ABS:organoid AND TITLE_ABS:barcod*',
    'TITLE_ABS:"clonal dominance" AND TITLE_ABS:lineage',
]
CLAIM = re.compile(r"clone size|expansion|correlat|bias|instruct|select|heritab|memory|predict|dominan|sampling", re.I)
ACC = re.compile(r"\b(GSE\d{5,7}|E-MTAB-\d{3,6})\b|zenodo\.(\d{6,9})", re.I)
FILTER = re.compile(SENSITIVE, re.I)
# Recall check: the first scan (TITLE_ABS, open access only) returned 31 hits; these known papers must come back.
RECALL = ["10.1038/s41556-026-01958-0", "10.1038/s41587-023-01931-4"]


def expand(q):
    return re.sub(r'TITLE_ABS:("[^"]+"|\S+)', r"(TITLE: OR ABSTRACT:)", q)


def get(url, as_json=True):
    req = urllib.request.Request(url, headers=UA)
    for i in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r) if as_json else r.read().decode("utf-8", "ignore")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(3 * (i + 1))
        except Exception:  # noqa: BLE001
            time.sleep(3 * (i + 1))
    return None


def accessions(pmcid):
    text = get(f"{API}{pmcid}/fullTextXML", as_json=False) if pmcid else None
    if not text:
        return None
    found = {(m.group(1) or "zenodo." + m.group(2)) for m in ACC.finditer(text)}
    return sorted(x.upper() if not x.startswith("zenodo") else x for x in found)


def main():
    kept, dropped, counts = {}, 0, {}
    for q in QUERIES:
        params = {"query": expand(q) + BASE, "format": "json", "pageSize": 100, "resultType": "core", "sort": "CITED desc"}
        r = get(API + "search?" + urllib.parse.urlencode(params))
        if not r:
            raise RuntimeError(f"query failed: {q}")
        counts[q] = r.get("hitCount", 0)
        for w in r.get("resultList", {}).get("result", []):
            title, abstract = w.get("title") or "", w.get("abstractText") or ""
            if FILTER.search(title) or FILTER.search(abstract):
                dropped += 1
                continue
            key = (w.get("doi") or w.get("id") or "").lower()
            kept[key] = {"title": CELL_LINE.sub("[line]", title), "year": w.get("pubYear"),
                         "cites": w.get("citedByCount", 0), "doi": w.get("doi"), "pmcid": w.get("pmcid"),
                         "source": w.get("source"), "claim_words": len({m.lower() for m in CLAIM.findall(abstract)}),
                         "abstract": CELL_LINE.sub("[line]", abstract)}
        time.sleep(0.3)
    for rec in kept.values():
        rec["accessions"] = accessions(rec["pmcid"])
        time.sleep(0.1)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"recall_missed": [d for d in RECALL if d not in kept], "queries": counts, "dropped_by_filter": dropped, "records": kept}, fh, indent=1, sort_keys=True)
    missed = [d for d in RECALL if d not in kept]
    with_data = sum(bool(r["accessions"]) for r in kept.values())
    print(json.dumps({"kept": len(kept), "dropped_by_filter": dropped, "with_accession": with_data, "recall_missed": missed}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
