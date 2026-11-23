"""Novelty-first scan, citation route (2026-11-23; Bence: "go 1 and list the still open candidates").

Keyword search returned too few papers (novelty_scan.py, 30 kept, one known paper missed). This pulls works since
2023 that cite the main clonal barcoding and lineage recording methods (OpenAlex), drops any record whose title or
abstract hits the neutral-wording filter, and lists GEO, Zenodo and ArrayExpress accessions from open full text
(Europe PMC). Metadata, abstracts and accession ids only. No data file is opened and no statistic is computed.
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
from novelty_scan import API as EPMC  # noqa: E402
from novelty_scan import CLAIM, FILTER, RECALL, accessions, get  # noqa: E402

OUT = os.path.join(HERE, "results_lit", "novelty_citers_2026-11-23.json")
OA = "https://api.openalex.org/works"
METHODS = {
    "LARRY": "10.1126/science.aaw3381",
    "CellTagging": "10.1038/s41586-018-0744-4",
    "CoSpar": "10.1038/s41587-022-01209-1",
    "CARLIN": "10.1016/j.cell.2020.04.048",
    "Cassiopeia": "10.1186/s13059-020-02000-8",
    "Embryo recorder": "10.1038/s41586-019-1184-5",
    "GESTALT": "10.1126/science.aaf7907",
    "Polylox": "10.1038/nature23653",
    "SIS-seq": "10.1016/j.immuni.2021.03.012",
    "CellTag-multi": "10.1038/s41587-023-01931-4",
}
MAX_PER_METHOD = 1000


def abstract(inv):
    if not inv:
        return ""
    pos = sorted((i, w) for w, idx in inv.items() for i in idx)
    return " ".join(w for _, w in pos)


def citers(work_id):
    # plain page paging: cursor paging returns 401 without a key (run 45186597216, as in lit_citers.py)
    page, out = 1, []
    while len(out) < MAX_PER_METHOD:
        params = {"filter": f"cites:{work_id},from_publication_date:2023-01-01", "per-page": 100, "page": page,
                  "select": "id,doi,display_name,publication_year,cited_by_count,abstract_inverted_index,ids"}
        r = get(f"{OA}?" + urllib.parse.urlencode(params))
        if not r:
            raise RuntimeError(f"citers failed: {work_id} page {page}")
        rows = r.get("results", [])
        out += rows
        if len(rows) < 100:
            break
        page += 1
        time.sleep(0.2)
    return out


def main():
    kept, dropped_keys, resolved = {}, set(), {}
    for name, doi in METHODS.items():
        w = get(f"{OA}/https://doi.org/{doi}?select=id")
        if not w:
            resolved[name] = None
            continue
        rows = citers(w["id"].rsplit("/", 1)[-1])
        resolved[name] = len(rows)
        for x in rows:
            key = (x.get("doi") or x["id"]).replace("https://doi.org/", "").lower()
            title, abs_text = x.get("display_name") or "", abstract(x.get("abstract_inverted_index"))
            if FILTER.search(title) or FILTER.search(abs_text):
                dropped_keys.add(key)
                continue
            rec = kept.setdefault(key, {"title": CELL_LINE.sub("[line]", title), "year": x.get("publication_year"),
                                        "cites": x.get("cited_by_count") or 0, "methods": [],
                                        "pmcid": (x.get("ids") or {}).get("pmcid", "").rsplit("/", 1)[-1] or None,
                                        "claim_words": len({m.lower() for m in CLAIM.findall(abs_text)}),
                                        "abstract": CELL_LINE.sub("[line]", abs_text)})
            rec["methods"].append(name)
    for rec in kept.values():
        rec["accessions"] = None
        if rec["claim_words"] >= 2 and rec["pmcid"]:
            pmc = rec["pmcid"] if rec["pmcid"].upper().startswith("PMC") else "PMC" + rec["pmcid"]
            rec["accessions"] = accessions(pmc)
            time.sleep(0.1)
    recall = {d: "kept" if d in kept else "dropped_by_filter" if d in dropped_keys else "absent" for d in RECALL}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"source_api": EPMC, "citers_per_method": resolved, "dropped_by_filter": len(dropped_keys),
                   "recall": recall, "records": kept}, fh, indent=1, sort_keys=True)
    print(json.dumps({"kept": len(kept), "dropped_by_filter": len(dropped_keys), "per_method": resolved,
                      "recall": recall, "with_accession": sum(bool(r["accessions"]) for r in kept.values())}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
