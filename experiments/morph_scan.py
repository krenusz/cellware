"""Scan for public per-organoid or per-gastruloid measurement tables (2026-11-24; Bence: "Yes, morphology tables").

Two metadata sources. Zenodo: record titles, descriptions and file names and sizes. Europe PMC: abstracts, and from
open full text only stated unit counts and accessions, including image archives. A record whose title or description
hits the neutral-wording filter is dropped whole. No data file is opened and no statistic is computed.
"""

import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lit_gap import CELL_LINE  # noqa: E402
from novelty_scan import API, FILTER, expand, get  # noqa: E402
from novelty_units import UNITS  # noqa: E402

OUT = os.path.join(HERE, "results_lit", "morph_scan_2026-11-24.json")
ZENODO = "https://zenodo.org/api/records?"
Z_QUERIES = ["gastruloid", "gastruloids morphology", "organoid morphology", "organoid segmentation",
             "organoid imaging dataset", "embryoid body imaging", "organoid size variability",
             "organoid high-content screen", "somitogenesis gastruloid", "organoid shape"]
P_QUERIES = [
    'TITLE_ABS:gastruloid* AND (TITLE_ABS:morpholog* OR TITLE_ABS:imaging OR TITLE_ABS:shape)',
    'TITLE_ABS:organoid* AND TITLE_ABS:variability AND TITLE_ABS:morpholog*',
    'TITLE_ABS:organoid* AND TITLE_ABS:"high-content" AND TITLE_ABS:imaging',
    'TITLE_ABS:organoid* AND TITLE_ABS:heterogeneity AND TITLE_ABS:"live imaging"',
    'TITLE_ABS:"embryoid bod*" AND TITLE_ABS:imaging AND TITLE_ABS:variability',
    'TITLE_ABS:organoid* AND TITLE_ABS:"reproducib*" AND TITLE_ABS:morpholog*',
    'TITLE_ABS:"stem cell" AND TITLE_ABS:"symmetry breaking" AND TITLE_ABS:imaging',
]
BASE = " AND FIRST_PDATE:[2019-01-01 TO 2026-11-24]"
ACC = re.compile(r"\b(S-BIAD\d+|S-BSST\d+|idr\d{4}|GSE\d{5,7}|E-MTAB-\d{3,6})\b|zenodo\.(\d{6,9})|figshare\.(\d{6,9})", re.I)
TABLE = re.compile(r"\.(csv|tsv|xlsx?|parquet|h5ad|txt)(\.gz)?$", re.I)
RECALL = ["10.1016/j.stem.2023.04.018"]


def zget(params):
    req = urllib.request.Request(ZENODO + urllib.parse.urlencode(params), headers={"User-Agent": "cellware-novelty/1.0"})
    for i in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except Exception:  # noqa: BLE001
            time.sleep(10 * (i + 1))
    return None


def zenodo():
    out, dropped = {}, 0
    for q in Z_QUERIES:
        hits = []
        for page in range(1, 5):  # Zenodo allows 25 per page without a key
            r = zget({"q": q, "size": 25, "page": page, "type": "dataset"})
            if r is None:
                raise RuntimeError(f"zenodo query failed: {q} page {page}")
            got = r.get("hits", {}).get("hits", [])
            hits += got
            if len(got) < 25:
                break
            time.sleep(1)
        for h in hits:
            md = h.get("metadata", {})
            title, desc = md.get("title") or "", re.sub(r"<[^>]+>", " ", md.get("description") or "")
            if FILTER.search(title) or FILTER.search(desc):
                dropped += 1
                continue
            files = [(f.get("key"), f.get("size")) for f in h.get("files", [])]
            out[str(h["id"])] = {"title": CELL_LINE.sub("[line]", title), "year": (md.get("publication_date") or "")[:4],
                                 "n_files": len(files), "tables": [f for f in files if f[0] and TABLE.search(f[0])][:20],
                                 "total_mb": round(sum(f[1] or 0 for f in files) / 1e6, 1),
                                 "unit_words": sorted({m.group(2).lower() for m in UNITS.finditer(desc)}),
                                 "description": CELL_LINE.sub("[line]", desc[:600])}
    return out, dropped


def facts(pmcid):
    text = get(f"{API}{pmcid}/fullTextXML", as_json=False) if pmcid else None
    if not text:
        return None
    plain = re.sub(r"<[^>]+>", " ", text)
    counts = {}
    for m in UNITS.finditer(plain):
        n, unit = int(m.group(1)), m.group(2).lower()
        if 20 <= n <= 5000:
            counts[unit] = max(counts.get(unit, 0), n)
    acc = sorted({m.group(1) or ("zenodo." + m.group(2) if m.group(2) else "figshare." + m.group(3))
                  for m in ACC.finditer(text)})
    return {"unit_counts": counts, "accessions": acc}


def pmc():
    out, dropped, hits = {}, 0, {}
    for q in P_QUERIES:
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
            out[key] = {"title": CELL_LINE.sub("[line]", title), "year": w.get("pubYear"), "pmcid": w.get("pmcid"),
                        "abstract": CELL_LINE.sub("[line]", abstract[:900])}
        time.sleep(0.3)
    for rec in out.values():
        rec["facts"] = facts(rec["pmcid"])
        time.sleep(0.1)
    return out, dropped, hits


def main():
    z, zdrop = zenodo()
    p, pdrop, hits = pmc()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"zenodo": z, "zenodo_dropped": zdrop, "pmc": p, "pmc_dropped": pdrop, "pmc_hits": hits,
                   "recall_missed": [d for d in RECALL if d not in p]}, fh, indent=1, sort_keys=True)
    zlead = sum(bool(r["tables"]) for r in z.values())
    plead = sum(bool(r["facts"] and any(v >= 50 for k, v in r["facts"]["unit_counts"].items()
                                        if k in ("organoids", "gastruloids", "embryos")) and r["facts"]["accessions"])
                for r in p.values())
    print(json.dumps({"zenodo_kept": len(z), "zenodo_dropped": zdrop, "zenodo_with_tables": zlead, "pmc_kept": len(p),
                      "pmc_dropped": pdrop, "pmc_full_text": sum(bool(r["facts"]) for r in p.values()),
                      "pmc_leads": plead, "recall_missed": [d for d in RECALL if d not in p]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
