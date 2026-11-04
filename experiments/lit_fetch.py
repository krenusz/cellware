"""Download the open-access full texts of the top unread gap-map papers (legal OA copies only).

    uv run python experiments/lit_fetch.py [N]      # top N unread from the latest gap_*.json

Sources, in order: Europe PMC PDF / XML, then the NCBI PMC article page (when the paper has a
PMCID), every OpenAlex OA location
(pdf_url, then landing page if it serves a PDF). Saves papers/lit/<doi>.pdf, keeps only files that
start with %PDF, and writes experiments/lit/fetch_<date>.tsv with what worked and what did not.
"""

from __future__ import annotations

import datetime
import glob
import json
import os
import sys
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "papers", "lit")
UA = {
    "User-Agent": "cellware-lit-fetch/1.0 (open-access research use)",
    "Accept": "application/pdf,*/*",
}


def fetch(url, timeout=60):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def jget(url):
    return json.loads(fetch(url))


def pmcid(doi):
    q = urllib.parse.urlencode({"query": f'DOI:"{doi}"', "format": "json", "resultType": "lite"})
    try:
        res = jget(f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?{q}")
        for r in res.get("resultList", {}).get("result", []):
            if r.get("pmcid"):
                return r["pmcid"]
    except Exception:
        pass
    return None


def candidates(doi):
    urls = []
    pid = pmcid(doi)
    if pid:
        urls.append(f"https://europepmc.org/articles/{pid}?pdf=render")
        urls.append(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pid}/fullTextXML")
        urls.append(f"https://pmc.ncbi.nlm.nih.gov/articles/{pid}/")
    try:
        w = jget(f"https://api.openalex.org/works/doi:{doi}?select=locations,best_oa_location")
        locs = [w.get("best_oa_location") or {}] + (w.get("locations") or [])
        for loc in locs:
            if loc and loc.get("is_oa"):
                for k in ("pdf_url", "landing_page_url"):
                    if loc.get(k) and loc[k] not in urls:
                        urls.append(loc[k])
    except Exception:
        pass
    return urls


def main(n=40):
    os.makedirs(OUT, exist_ok=True)
    rows = json.load(open(sorted(glob.glob(os.path.join(HERE, "lit", "gap_*.json")))[-1]))
    unread = [r for r in rows if not r["read"]][:n]
    log = []
    for i, r in enumerate(unread, 1):
        doi = r["doi"]
        if not doi:
            log.append((i, "", "no-doi", "", r["title"]))
            continue
        base = os.path.join(OUT, doi.replace("/", "_"))
        if (
            os.path.exists(base + ".pdf")
            or os.path.exists(base + ".xml")
            or os.path.exists(base + ".html")
        ):
            log.append((i, doi, "have", "", r["title"]))
            continue
        if r["status"] != "open":
            log.append((i, doi, "paywalled", r["pub"], r["title"]))
            continue
        got = None
        for url in candidates(doi):
            try:
                data = fetch(url)
            except Exception as e:
                last = f"{type(e).__name__}"
                continue
            if data[:4] == b"%PDF":
                open(base + ".pdf", "wb").write(data)
                got = ("pdf", url)
                break
            if url.endswith("fullTextXML") and b"<article" in data[:5000]:
                open(base + ".xml", "wb").write(data)
                got = ("xml", url)
                # keep looking for a PDF; XML is the fallback
            if (
                ("pmc.ncbi.nlm.nih.gov/articles/" in url or "/pmc/articles/" in url)
                and not got
                and b"<html" in data[:2000]
                and len(data) > 60000
                and b"Recaptcha" not in data
            ):
                open(base + ".html", "wb").write(data)
                got = ("html", url)
            last = "not-pdf"
            time.sleep(0.5)
        if got:
            log.append((i, doi, got[0], got[1], r["title"]))
        else:
            log.append((i, doi, "failed", locals().get("last", ""), r["title"]))
        print(i, log[-1][2], doi, flush=True)
    date = datetime.date.today().isoformat()
    with open(os.path.join(HERE, "lit", f"fetch_{date}.tsv"), "w", encoding="utf-8") as f:
        f.write("rank\tdoi\tresult\tsource\ttitle\n")
        for row in log:
            f.write("\t".join(str(x) for x in row) + "\n")
    from collections import Counter

    print(Counter(x[2] for x in log))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
