"""Data and design gate for the shortlist of the citation scan (2026-11-23; Bence: "go 1").

Open full text through Europe PMC only. Keeps sentences about data availability, the number of animals, and how
a null or sampling model was built. Sentences with a test statistic, or hitting the neutral-wording filter, are
dropped. No data file is opened and no statistic is computed.
"""

import json
import os
import re
import sys
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from novelty_scan import ACC, API, FILTER, get  # noqa: E402

OUT = os.path.join(HERE, "results_lit", "novelty_datagate_2026-11-23.json")
DOIS = ["10.7554/elife.104537", "10.1101/2024.11.22.624575", "10.64898/2026.06.08.730911",
        "10.1101/2025.09.02.672339", "10.1073/pnas.2505510122"]
KEEP = re.compile(r"data availab|availability|deposited|accession|GEO|SRA|zenodo|github|upon request|on request|"
                  r"[0-9]+ (mice|fish|animals|males)|neutral model|null model|sampling|permut|technical replicate|"
                  r"detection limit|clone size", re.I)
STAT = re.compile(r"\b[pP] ?[<=]|\brho\b|\bR2\b|\br ?=|fold|[0-9]+(\.[0-9]+)? ?%")


def full_text(rec):
    if rec.get("pmcid"):
        text = get(f"{API}{rec['pmcid']}/fullTextXML", as_json=False)
        if text:
            return text, "pmc"
    if rec.get("source") == "PPR":
        text = get(f"{API}PPR/{rec['id']}/fullTextXML", as_json=False)
        if text:
            return text, "preprint"
    return None, None


def main():
    out = {}
    for doi in DOIS:
        q = urllib.parse.urlencode({"query": f'DOI:"{doi}"', "format": "json", "resultType": "lite"})
        hits = (get(f"{API}search?{q}") or {}).get("resultList", {}).get("result", [])
        if not hits:
            out[doi] = {"status": "no record"}
            continue
        text, route = full_text(hits[0])
        if not text:
            out[doi] = {"status": "no open full text", "source": hits[0].get("source")}
            continue
        plain = re.sub(r"<[^>]+>", " ", text)
        sents = [" ".join(s.split()) for s in re.split(r"(?<=[.!?])\s+", plain)]
        keep = [s[:320] for s in sents if KEEP.search(s) and not STAT.search(s) and not FILTER.search(s)]
        acc = sorted({m.group(0) for m in ACC.finditer(plain)})
        out[doi] = {"status": "read", "route": route, "accessions": acc, "sentences": keep[:45]}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print(json.dumps({k: v["status"] for k, v in out.items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
