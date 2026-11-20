"""C17-S1N2: exhaustive neural search pass (any organism, three repositories, topic and method-name queries).

Bence 2026-11-20: "Two more neural systems if it doesnt really take us to a drift and after focus on the
Stage-aware memory model"; "you cannot miss anything: The earlier sweep missed neural-specific data - do it more
robust". Serves: C17-S1 (C17-S1R: grow the systems; neural class).
Pre-registered in EXPERIMENT_LOG (C17-S1N2). Metadata only: no data file is downloaded.

1. Retrieve: GEO (any organism, since 2012) with four queries (neural topic, lineage method names, sister/split,
   serial/organoid); BioStudies/ArrayExpress and Zenodo with neural lineage phrasings, all pages up to a limit.
2. Pre-filter: neural wording AND a clone/lineage/tree/sister term in title or summary.
3. Recall gate: every neural set already known to the project must be retrieved and pass the pre-filter; a miss
   marks the pass INVALID (widen the queries before using any verdict).
4. Check every pre-filtered GEO series with c17s1n_struct.design (series + sample SOFT text), every BioStudies hit
   with its study JSON; Zenodo by title + description only. No cap on the number checked.
5. Verdict (stricter than C17-S1N): sister split, OR serial sampling at >= 2 timepoints, OR a tree term together
   with a clone/barcode term. Titles pass lit_gap.neutral before saving; excluded ones are counted only.

    python -I experiments/c17s1n2_neural.py            # writes results_c17/c17s1n2_neural.json (Actions)
    python -I experiments/c17s1n2_neural.py selftest
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import c17s1n_struct as ns  # noqa: E402
from c17_data_sweep import EUTILS, get  # noqa: E402
from lit_gap import neutral  # noqa: E402

OUT = os.path.join(HERE, "results_c17", "c17s1n2_neural.json")
NEURAL = re.compile(r"brain|cortex|cortical|neur|retina|retinal|spinal|glia|hippocamp|cerebell|forebrain|midbrain|"
                    r"hindbrain|thalam|striat|olfactory|cochlea|cerebral|telenceph|diencephal|ganglion|"
                    r"astrocyt|oligodendro|radial glia", re.I)
LINEAGE = re.compile(ns.CLONE.pattern + "|" + ns.TREE.pattern + "|" + ns.SISTER.pattern + r"|lineage|fate map",
                     re.I)
NEURAL_Q = ('(brain OR cortex OR cortical OR neural OR neuronal OR neuron OR retina OR retinal OR "spinal cord" OR '
            'glia OR glial OR hippocampus OR cerebellum OR forebrain OR midbrain OR hindbrain OR thalamus OR '
            'striatum OR "neural crest" OR "neural progenitor" OR "cerebral organoid" OR "brain organoid")')
METHODS = ('(GESTALT OR scGESTALT OR LINNAEUS OR iTracer OR STICR OR TrackerSeq OR TREX OR CloneSeq OR MADM OR '
           'CellTag OR LARRY OR CARLIN OR DARLIN OR MEMOIR OR intMEMOIR OR PEtracer OR CREST OR Polylox OR '
           'Confetti OR "lineage recorder" OR "lineage recording" OR "CRISPR barcode" OR "CRISPR barcoding" OR '
           '"Cas9 barcode" OR "evolving barcode" OR "lineage barcode" OR "lineage barcoding" OR "clonal barcoding" '
           'OR "viral barcode" OR "lentiviral barcode" OR "scar" OR "lineage tree")')
SC = '("single-cell" OR "single cell" OR scRNA-seq OR "clonal analysis" OR clonal)'
BASE = '("2012"[PDAT] : "3000"[PDAT]) AND gse[ETYP]'
GEO_QUERIES = {
    "neural_lineage": f'{NEURAL_Q} AND ("lineage tracing" OR "lineage barcoding" OR "lineage recording" OR '
                      f'"clonal analysis" OR "clonal tracing" OR barcod* OR "lineage tree" OR "fate mapping") '
                      f'AND {SC} AND {BASE}',
    "method_names": f"{METHODS} AND {BASE}",
    "sister_split": f'{NEURAL_Q} AND (sister OR "sister cells" OR daughter OR "clone splitting" OR "split clones" '
                    f'OR twin OR "two-color" OR "two-colour") AND {SC} AND {BASE}',
    "serial_organoid": f'{NEURAL_Q} AND (barcod* OR "lineage tracing" OR "lineage recording") AND (organoid OR '
                       f'"time course" OR longitudinal OR timepoints OR "time points") AND {BASE}',
}
TEXT_QUERIES = ["lineage tracing brain single cell", "lineage recording organoid", "CRISPR lineage zebrafish brain",
                "lineage barcoding neural", "clonal analysis cortex single-cell", "neural crest lineage barcoding",
                "retina lineage tracing single cell", "scar lineage tree neural", "sister cells neural progenitor",
                "spinal cord lineage barcoding", "cerebral organoid barcoding", "brain clonal barcoding"]
# Neural sets already known to the project: the recall gate (C17-S1N checked these; GSE210139 = CRISPR brain set)
KNOWN_GEO = ["GSE210139", "GSE327468", "GSE300285", "GSE300287", "GSE300289", "GSE324903", "GSE324905"]
KNOWN_BIOSTUDIES = ["E-MTAB-10971"]
DONE = set(ns.NEURAL) | set(ns.SUPER)


def strict(flags, multi_t):
    """Pre-registered rule: sister split, OR serial sampling at >= 2 timepoints, OR tree term + clone term."""
    if flags["sister_split"]:
        return True
    if multi_t and flags["serial_sampling"]:
        return True
    return bool(flags["trees"]) and bool(flags["clone_barcodes"])


def text_flags(text):
    return {k: sorted({m.group(0).lower() for m in r.finditer(text)})
            for k, r in (("sister_split", ns.SISTER), ("serial_sampling", ns.SERIAL), ("trees", ns.TREE),
                         ("clone_barcodes", ns.CLONE))}


def prefilter(title, summary):
    t = f"{title} {summary}"
    return bool(NEURAL.search(t)) and bool(LINEAGE.search(t))


def geo_query(name, q, limit=10000):
    r = get(EUTILS + "esearch.fcgi?" + urllib.parse.urlencode(
        {"db": "gds", "term": q, "retmax": limit, "retmode": "json", "tool": "cellware-sweep"}))["esearchresult"]
    ids, recs = r.get("idlist", []), {}
    for i in range(0, len(ids), 100):
        time.sleep(0.6)
        res = get(EUTILS + "esummary.fcgi?" + urllib.parse.urlencode(
            {"db": "gds", "id": ",".join(ids[i : i + 100]), "retmode": "json", "tool": "cellware-sweep"}))["result"]
        for uid in res.get("uids", []):
            d = res[uid]
            acc = d.get("accession", "")
            if acc.startswith("GSE"):
                recs[acc] = {"title": d.get("title", ""), "summary": d.get("summary", ""),
                             "taxon": d.get("taxon"), "year": (d.get("pdat") or "")[:4]}
    return int(r.get("count", 0)), recs


def biostudies_query(q, pages=3):
    out = {}
    for p in range(1, pages + 1):
        r = get("https://www.ebi.ac.uk/biostudies/api/v1/arrayexpress/search?" + urllib.parse.urlencode(
            {"query": q, "pageSize": 100, "page": p}))
        hits = r.get("hits", [])
        for h in hits:
            out[h.get("accession")] = {"title": h.get("title", ""), "summary": h.get("content", "") or ""}
        if len(hits) < 100:
            break
        time.sleep(0.5)
    return out


def zenodo_query(q, pages=8):
    """Amendment 1: Zenodo refuses size 100 with a type filter (HTTP 400); 25 per page, type filtered here."""
    out = {}
    for p in range(1, pages + 1):
        r = get("https://zenodo.org/api/records?" + urllib.parse.urlencode({"q": q, "size": 25, "page": p}))
        hits = r.get("hits", {}).get("hits", [])
        for h in hits:
            m = h.get("metadata", {})
            if (m.get("resource_type") or {}).get("type") not in (None, "dataset"):
                continue
            out[f"zenodo:{h.get('id')}"] = {"title": m.get("title", ""),
                                            "summary": re.sub(r"<[^>]+>", " ", m.get("description") or "")}
        if len(hits) < 25:
            break
        time.sleep(0.5)
    return out


def safe_title(t):
    return t if neutral({"title": t}) else None


def run():
    res = {"queries": {}, "errors": [], "geo": {}, "biostudies": {}, "zenodo": {}, "excluded_titles": 0}
    geo = {}
    for name, q in GEO_QUERIES.items():
        try:
            n, recs = geo_query(name, q)
            res["queries"][name] = {"count": n, "retrieved": len(recs), "complete": len(recs) >= n * 0.98}
            geo.update(recs)
        except Exception as e:  # noqa: BLE001
            res["errors"].append(f"geo {name}: {str(e)[:100]}")
    bio, zen = {}, {}
    for q in TEXT_QUERIES:
        for store, fn, src in ((bio, biostudies_query, "biostudies"), (zen, zenodo_query, "zenodo")):
            try:
                store.update(fn(q))
            except Exception as e:  # noqa: BLE001
                res["errors"].append(f"{src} {q}: {str(e)[:100]}")
    pre_geo = {a: r for a, r in geo.items() if prefilter(r["title"], r["summary"])}
    pre_bio = {a: r for a, r in bio.items() if prefilter(r["title"], r["summary"])}
    pre_zen = {a: r for a, r in zen.items() if prefilter(r["title"], r["summary"])}
    missed = [a for a in KNOWN_GEO if a not in pre_geo] + [a for a in KNOWN_BIOSTUDIES if a not in pre_bio]
    res["recall"] = {"known": KNOWN_GEO + KNOWN_BIOSTUDIES, "missed": missed, "valid": not missed,
                     "missed_but_retrieved": [a for a in missed if a in geo or a in bio]}
    res["counts"] = {"geo_retrieved": len(geo), "geo_prefiltered": len(pre_geo), "biostudies_retrieved": len(bio),
                     "biostudies_prefiltered": len(pre_bio), "zenodo_retrieved": len(zen),
                     "zenodo_prefiltered": len(pre_zen)}
    for acc, r in sorted(pre_geo.items()):
        if acc in DONE:
            continue
        title = safe_title(r["title"])
        if title is None:
            res["excluded_titles"] += 1
            continue
        try:
            rec = ns.design(ns.soft(acc), ns.samples(ns.soft(acc, "gsm")))
        except Exception as e:  # noqa: BLE001
            res["geo"][acc] = {"title": title, "error": str(e)[:100]}
            continue
        multi_t = len(rec["timepoints"]) >= 2
        res["geo"][acc] = {"title": title, "taxon": r["taxon"], "year": r["year"], "n_samples": rec["n_samples"],
                           "flags": rec["flags"], "timepoints": rec["timepoints"][:12],
                           "verdict": "memory-capable" if strict(rec["flags"], multi_t) else "not by the rule"}
        time.sleep(0.4)
    for acc, r in sorted(pre_bio.items()):
        title = safe_title(r["title"])
        if title is None:
            res["excluded_titles"] += 1
            continue
        try:
            text = json.dumps(get(f"https://www.ebi.ac.uk/biostudies/api/v1/studies/{acc}"))
        except Exception as e:  # noqa: BLE001
            res["biostudies"][acc] = {"title": title, "error": str(e)[:100]}
            continue
        f = text_flags(text)
        multi_t = len({m.group(1) for m in ns.TIMEPOINT.finditer(text)}) >= 2
        res["biostudies"][acc] = {"title": title, "flags": f,
                                  "verdict": "memory-capable" if strict(f, multi_t) else "not by the rule"}
        time.sleep(0.4)
    for acc, r in sorted(pre_zen.items()):
        title = safe_title(r["title"])
        if title is None:
            res["excluded_titles"] += 1
            continue
        f = text_flags(f"{r['title']} {r['summary']}")
        res["zenodo"][acc] = {"title": title, "flags": f,
                              "verdict": "check: description suggests memory design" if strict(f, False)
                              else "not by the rule"}
    res["summary"] = {
        "recall_valid": res["recall"]["valid"], "recall_missed": missed, "counts": res["counts"],
        "memory_capable_new": {a: v["title"] for src in ("geo", "biostudies") for a, v in res[src].items()
                               if v.get("verdict") == "memory-capable"},
        "zenodo_to_check": {a: v["title"] for a, v in res["zenodo"].items() if v["verdict"].startswith("check")},
        "errors": len(res["errors"]), "excluded_titles": res["excluded_titles"],
    }
    return res


def selftest():
    glial = text_flags("Single-cell profiling of the glial scar after spinal cord injury")
    array = text_flags("CRISPR lineage recording with scar arrays and clonal barcodes in zebrafish brain")
    assert not strict(glial, False), glial
    assert strict(array, False), array
    assert strict(text_flags("sister cells split into two wells"), False)
    assert prefilter("Lineage recording in human cerebral organoids", "")
    assert prefilter("Clonal analysis of retinal progenitors", "barcoded")
    assert not prefilter("Hematopoietic clones barcoded", "LARRY")
    assert not prefilter("Neuronal activity atlas", "single-cell RNA-seq")
    return {"ok": True}


def main():
    res = selftest() if sys.argv[1:2] == ["selftest"] else run()
    if "summary" in res:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1, default=str)
        res = res["summary"]
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
