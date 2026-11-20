"""C17-S1D: data sweep for new systems of the general clone model, and the watch list.

Bence 2026-11-19: "do a retro citation map mand swep for data also".
Serves: C17-S1 (C17-S1R: grow the systems; the parameter table has committed blood 2, pluripotent 1).

Metadata only (H9): public catalogue records (accession, title, year, sample count, organism) and keyword flags.
No data file is downloaded. Records whose title carries disease-domain terms, or whose summary carries
tumour/leukaemia/patient/drug-resistance terms, are dropped (standing exclusion); cell-line names in titles are
masked. Summaries are used for flags only and are not stored.

Sources: NCBI GEO (E-utilities, db=gds), Zenodo (records API), EBI BioStudies (ArrayExpress collection).

    python experiments/c17_data_sweep.py            # writes results_c17/c17_data_sweep.json
    python experiments/c17_data_sweep.py selftest   # offline check of flags, masking, domain drop and scoring
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_c17", "c17_data_sweep.json")
UA = {"User-Agent": "cellware-data-sweep/1.0 (public metadata, research use)"}
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

# Further disease, pathogen and drug names (retro 2026-11-19b; same list in lit_gap.py).
DISEASE_EXTRA = (
    r"|anemi|anaemi|aplastic|thalass|sickle|hemophil|haemophil|penia\b|immunodeficien|dystroph|scleros|sclerotic"
    r"|fibrot|arthritis|lupus|asthma|allerg|obes(e|ity)|hypertens|ischemi|ischaemi|stroke|infarct|aneurysm"
    r"|embolism|thrombos[ie]s|thrombotic|hemorrhag|haemorrhag|cirrhos|hepatitis|colitis|crohn|inflammatory bowel"
    r"|ulcer|psoria|dementia|neurodegenerat|macular|epilep|autis|schizophren|deafness|opath(y|ies|ic)\b"
    r"|graft.versus.host|gvhd|\bdisorders?\b|\blesions?\b|pneumonia|eclamp|spina bifida|malnut|infertil"
    r"|congenital|pathogen|malaria|tubercul|influenza|zika|ebola|herpes|cytomegalo|septic|blastoma|adenom"
    r"|hepatom|teratom|mesotheliom|astrocytom|myeloprolif|lymphoprolif|\baml\b|\bcml\b|\bcll\b|metaplas"
    r"|hyperplas|atroph|cystic|trauma|injur|vaccin|antibiotic|pharmac|fluorouracil|\b5-?fu\b"
    r"|\w+(nib|mab|ciclib|parib)\b|cholesta|ichthyo|\bdeficiency\b|\btoxicity\b|mutations? in a|cardiovascular|risk factor|immuniz"
    r"|glioma|(gli|mening|ependym|astrocyt|sarc|neur|aden)oma\b|alcohol|addict|vulnerab|injur|lesion|ischem|stroke|epilep|seizure|autis|schizo|depress|congenit|keratos|\bflu\b|h5n1|pregnan|placent|inflammat|cell-free dna"
)
TITLE_DROP = re.compile(r"cancer|tumou?r|leuk|lymphom|myelom|carcinom|melanom|malignan|oncog|metasta|drug|chemo|"
                        r"therap|patient|disease|clinical|glioma|sarcoma|neoplas|syndrom|dysplas|fibros|diabet|"
                        r"alzheimer|parkinson|sepsis|autoimmun|infection|\bhiv\b|covid|sars|defect" + DISEASE_EXTRA,
                        re.I)
SUMMARY_DROP = re.compile(r"cancer|tumou?r|leuk|lymphom|myelom|carcinom|melanom|malignan|metasta|patient|glioma|"
                          r"sarcoma|neoplas|drug.resist|chemoresist", re.I)
CELL_LINE = re.compile(r"\b(hela|hek ?293[a-z]*|k562|mcf-?7|a549|hct-?116|jurkat|u2os|thp-?1|hl-?60|nb4|molm-?\d+|"
                       r"mv4-?11|kasumi-?\d*|wtc-?11|hues\d+|kolf2?[\w.-]*|(?<!histone )h[179](?![.]\d)|"
                       r"e14(tg2a)?(?![.]\d)|v6\.5|cgr8)\b", re.I)
FLAGS = {
    "lineage": r"barcod|lineage[ -]trac|lineage record|clonal (trac|analys|barcod|lineage)|celltag|\blarry\b|"
               r"polylox|carlin|darlin|lineage tree|cellular barcod|clone[ -]tracing",
    "split_or_memory": r"sister|split|twin|fluctuation|passag|serial|heritab|memory",
    "pluripotent_or_organoid": r"pluripoten|embryonic stem|\b[hm]?es ?cells?\b|\b[hm]?escs?\b|\b[hm]?i?pscs?\b|"
                               r"embryoid|gastruloid|epiblast|"
                               r"blastoid|organoid|embryo",
    "blood": r"hematopoie|haematopoie|progenitor|\bhspc|\bhsc",
    "single_cell": r"single[ -]cell|scrna|sc-?rna-?seq|10x",
}
WEIGHTS = {"lineage": 2, "split_or_memory": 2, "pluripotent_or_organoid": 2, "blood": 1, "single_cell": 1}

GEO_BASE = '("2019"[PDAT] : "3000"[PDAT]) AND gse[ETYP] AND ("Homo sapiens"[ORGN] OR "Mus musculus"[ORGN])'
GEO_QUERIES = {
    "pluripotent_clones": '(barcod* OR "lineage tracing" OR "lineage barcoding" OR "clonal tracing" OR CellTag OR '
                          'LARRY) AND ("embryonic stem" OR pluripotent OR iPSC OR embryoid OR gastruloid OR epiblast '
                          'OR blastoid)',
    "sister_split": '("sister cells" OR "sister cell" OR "sister clones" OR "clone splitting" OR "split clones" OR '
                    '"SIS-seq" OR fluctuation) AND (barcod* OR clonal OR "lineage tracing")',
    "clonal_memory": '(heritable OR "clonal memory" OR "transcriptional memory" OR "epigenetic memory" OR passaging) '
                     'AND (barcod* OR "lineage tracing") AND ("single-cell" OR "single cell")',
    "organoid_clones": 'organoid AND (barcod* OR "lineage tracing" OR "clonal analysis") AND ("single-cell" OR '
                       '"single cell")',
    "progenitor_clones": '(hematopoietic OR haematopoietic OR progenitor) AND (barcod* OR "lineage tracing") AND '
                         '("single-cell" OR "single cell")',
}
ZENODO_QUERIES = {
    "pluripotent_clones": '"lineage tracing" AND (pluripotent OR "embryonic stem" OR gastruloid OR embryoid OR organoid)',
    "sister_split": '("sister cells" OR "sister clones") AND (lineage OR clonal)',
    "clonal_barcoding": '(barcoding OR "lineage barcodes") AND clonal AND single-cell',
}
BIOSTUDIES_QUERIES = {
    "pluripotent_clones": "lineage tracing barcoding pluripotent",
    "sister_split": "sister cells clonal single-cell",
    "organoid_clones": "organoid lineage tracing barcoding",
}
WATCH_GEO = ["GSE324482", "GSE342607"]
WATCH_BIOSTUDIES = ["E-MTAB-17351"]
WATCH_TITLE_SEARCH = {
    "hESC endoderm sister wells": '(endoderm) AND (clonal OR barcod* OR "lineage tracing") AND '
                                  '(sister OR memory OR heritable) AND ("2026"[PDAT] : "3000"[PDAT])',
    "DeepTrack": 'DeepTrack AND ("2025"[PDAT] : "3000"[PDAT])',
}


def get(url, as_json=True, tries=5):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90) as r:
                b = r.read()
            return json.loads(b) if as_json else b.decode()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            print(f"retry {i + 1} {url[:90]}: {e}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"retry {i + 1} {url[:90]}: {e}", flush=True)
        time.sleep(4 * (i + 1))
    raise RuntimeError(url)


def esearch(term, retmax=300):
    q = {"db": "gds", "term": term, "retmax": retmax, "retmode": "json", "tool": "cellware-sweep"}
    r = get(EUTILS + "esearch.fcgi?" + urllib.parse.urlencode(q))["esearchresult"]
    return int(r.get("count", 0)), r.get("idlist", [])


def geo_records(name, q):
    count, ids = esearch(f"({q}) AND {GEO_BASE}")
    recs = []
    for i in range(0, len(ids), 100):
        time.sleep(0.6)
        q2 = {"db": "gds", "id": ",".join(ids[i : i + 100]), "retmode": "json", "tool": "cellware-sweep"}
        res = get(EUTILS + "esummary.fcgi?" + urllib.parse.urlencode(q2))["result"]
        for uid in res.get("uids", []):
            d = res[uid]
            recs.append({"source": "GEO", "accession": d.get("accession"), "title": d.get("title", ""),
                         "summary": d.get("summary", ""), "year": (d.get("pdat") or "")[:4],
                         "n_samples": d.get("n_samples"), "organism": d.get("taxon"), "query": name})
    return count, recs


def zenodo_records(name, q):
    r = get("https://zenodo.org/api/records?" + urllib.parse.urlencode({"q": q, "size": 25, "sort": "mostrecent"}))
    out = []
    for h in r.get("hits", {}).get("hits", []):
        m = h.get("metadata", {})
        if (m.get("resource_type") or {}).get("type") not in (None, "dataset"):
            continue
        out.append({"source": "Zenodo", "accession": f"zenodo:{h.get('id')}", "title": m.get("title", ""),
                    "summary": re.sub(r"<[^>]+>", " ", m.get("description") or ""),
                    "year": (m.get("publication_date") or "")[:4], "n_samples": None, "organism": None,
                    "query": name})
    return r.get("hits", {}).get("total"), out


def biostudies_records(name, q):
    url = "https://www.ebi.ac.uk/biostudies/api/v1/arrayexpress/search?" + urllib.parse.urlencode(
        {"query": q, "pageSize": 100})
    r = get(url)
    out = []
    for h in r.get("hits", []):
        out.append({"source": "BioStudies", "accession": h.get("accession"), "title": h.get("title", ""),
                    "summary": h.get("content", "") or "", "year": str(h.get("release_date", ""))[:4],
                    "n_samples": None, "organism": None, "query": name})
    return r.get("totalHits"), out


# Study families already in the project (H5): GEO sub-series of these studies are not new data.
KNOWN_STEMS = {
    "comprehensive lineage tracing maps the landscape of cell fate decisions in mouse embryogenesis":
        "PEtracer embryo set (sealed)",
    "single-cell lineage capture across genomic modalities with celltag-multi reveals fate-specific gene "
    "regulatory changes": "CellTag-multi LSK and reprogramming sets (sealed)",
    "a mouse model with high clonal barcode diversity for joint lineage, transcriptomic, and epigenomic profiling "
    "in single cells": "darlin (used, pre-protocol)",
    "in situ fate mapping identifies the neonatal origin of platelet-biased hematopoietic stem cells":
        "GSE299000 family (C17 inventory)",
    "heterogeneity in hematopoietic stem cell enables adaptation to the aged microenvironment": "GSE326529 family",
}


def stem(title):
    """Study title without a trailing sub-series tag such as ' [scRNA-seq]' or ' (RNA-seq)'."""
    return re.sub(r"\s*[\[(][^\])]*[\])]\s*\.?$", "", title or "").strip().lower()


def known_accessions():
    text = ""
    files = [os.path.join(HERE, "EXPERIMENT_LOG.md"), os.path.join(HERE, "partitions.py"),
             os.path.join(HERE, "FOCUS.json")] + glob.glob(os.path.join(HERE, "lit", "*.md"))
    for f in files:
        if os.path.exists(f):
            text += open(f, encoding="utf-8", errors="replace").read()
    acc = set(re.findall(r"\b(GSE\d{4,7}|E-[A-Z]{4}-\d+)\b", text))
    acc |= {f"zenodo:{z}" for z in re.findall(r"[Zz]enodo(?:\.org/records?/|\s+)(\d{6,9})", text)}
    return acc


def flags_of(text):
    return [k for k, p in FLAGS.items() if re.search(p, text, re.I)]


def process(raw, known):
    """Drop excluded-domain records, mask cell-line names, flag, score and deduplicate (by accession)."""
    out, dropped = {}, 0
    for r in raw:
        acc = r.get("accession")
        if not acc:
            continue
        if TITLE_DROP.search(r.get("title") or "") or SUMMARY_DROP.search(r.get("summary") or ""):
            dropped += 1
            continue
        fl = flags_of(f"{r.get('title', '')} {r.get('summary', '')}")
        if "lineage" not in fl:
            continue
        rec = out.get(acc)
        if rec is None:
            rec = {"source": r["source"], "accession": acc, "title": CELL_LINE.sub("[line]", r.get("title") or ""),
                   "year": r.get("year"), "n_samples": r.get("n_samples"), "organism": r.get("organism"),
                   "flags": fl, "score": sum(WEIGHTS[f] for f in fl), "known": acc in known, "queries": []}
            rec["pluripotent_candidate"] = all(f in fl for f in ("pluripotent_or_organoid", "split_or_memory"))
            out[acc] = rec
        if r["query"] not in rec["queries"]:
            rec["queries"].append(r["query"])
    fam = {}
    for rec in out.values():
        rec["family"] = stem(rec["title"])
        fam.setdefault(rec["family"], []).append(rec)
    for f, members in fam.items():
        tag = KNOWN_STEMS.get(f)
        if tag or any(m["known"] for m in members):
            for m in members:
                m["known"] = True
                m["known_as"] = tag or "family of a known accession"
    recs = sorted(out.values(), key=lambda x: (-x["score"], -len(x["queries"]), x["accession"]))
    return recs, dropped


def watch():
    res = {}
    for acc in WATCH_GEO:
        try:
            n, _ = esearch(f"{acc}[ACCN] AND gse[ETYP]", retmax=1)
            res[acc] = "public" if n > 0 else "not public"
        except Exception as e:  # noqa: BLE001
            res[acc] = f"error: {str(e)[:60]}"
        time.sleep(0.6)
    for acc in WATCH_BIOSTUDIES:
        try:
            get(f"https://www.ebi.ac.uk/biostudies/api/v1/studies/{acc}", tries=2)
            res[acc] = "public"
        except urllib.error.HTTPError as e:
            res[acc] = "not public" if e.code == 404 else f"http {e.code}"
        except Exception as e:  # noqa: BLE001
            res[acc] = f"error: {str(e)[:60]}"
    for name, term in WATCH_TITLE_SEARCH.items():
        try:
            n, ids = esearch(f"({term}) AND gse[ETYP]", retmax=20)
            titles = []
            if ids:
                q2 = {"db": "gds", "id": ",".join(ids), "retmode": "json", "tool": "cellware-sweep"}
                s = get(EUTILS + "esummary.fcgi?" + urllib.parse.urlencode(q2))["result"]
                for uid in s.get("uids", []):
                    t = s[uid].get("title", "")
                    if not TITLE_DROP.search(t):
                        titles.append(f"{s[uid].get('accession')}: {CELL_LINE.sub('[line]', t)}")
            res[name] = {"hits": n, "titles": titles[:20]}
        except Exception as e:  # noqa: BLE001
            res[name] = f"error: {str(e)[:60]}"
        time.sleep(0.6)
    return res


def sweep():
    raw, counts, errors = [], {}, []
    for src, queries, fn in (("GEO", GEO_QUERIES, geo_records), ("Zenodo", ZENODO_QUERIES, zenodo_records),
                             ("BioStudies", BIOSTUDIES_QUERIES, biostudies_records)):
        for name, q in queries.items():
            try:
                total, recs = fn(name, q)
                counts[f"{src}:{name}"] = {"total": total, "fetched": len(recs)}
                raw += recs
            except Exception as e:  # noqa: BLE001
                errors.append(f"{src}:{name}: {str(e)[:120]}")
            time.sleep(1.0)
    recs, dropped = process(raw, known_accessions())
    by = {
        "records_with_lineage_flag": len(recs),
        "new_records": sum(not r["known"] for r in recs),
        "new_study_families": len({r["family"] for r in recs if not r["known"]}),
        "pluripotent_candidates_new": sum(r["pluripotent_candidate"] and not r["known"] for r in recs),
        "blood_split_or_memory_new": sum(("blood" in r["flags"]) and ("split_or_memory" in r["flags"])
                                         and not r["known"] for r in recs),
        "dropped_excluded_domain": dropped,
    }
    return {"queries": counts, "errors": errors, "summary": by, "watch": watch(), "records": recs[:400]}


def selftest():
    known = {"GSE1"}
    raw = [
        {"source": "GEO", "accession": "GSE1", "title": "Barcoding of H9 hESC sister clones across passages",
         "summary": "single-cell lineage tracing", "query": "q"},
        {"source": "GEO", "accession": "GSE2", "title": "Lineage tracing in a tumour model", "summary": "",
         "query": "q"},
        {"source": "GEO", "accession": "GSE3", "title": "Gastruloid single-cell atlas", "summary": "no clones",
         "query": "q"},
        {"source": "GEO", "accession": "GSE4", "title": "Clonal barcoding of HSPC with serial transplant",
         "summary": "scRNA-seq of progenitors; cells derived from patient samples", "query": "q"},
        {"source": "Zenodo", "accession": "zenodo:5", "title": "LARRY barcoding of organoid split wells",
         "summary": "single-cell", "query": "q2"},
        {"source": "GEO", "accession": "GSE6", "title": "Barcoding of H9 hESC sister clones across passages [ATAC]",
         "summary": "single-cell lineage tracing", "query": "q"},
        {"source": "GEO", "accession": "GSE7", "title": "Comprehensive Lineage Tracing Maps the Landscape of Cell "
         "Fate Decisions in Mouse Embryogenesis [DevMap_mouse_E7.5-R1]", "summary": "lineage tracing", "query": "q"},
    ]
    recs, dropped = process(raw, known)
    acc = [r["accession"] for r in recs]
    assert acc == ["GSE1", "GSE6", "zenodo:5", "GSE7"], acc
    assert dropped == 2, dropped
    by = {r["accession"]: r for r in recs}
    assert by["GSE6"]["known"], "sub-series of a known study must be known"
    assert by["GSE7"]["known"] and by["GSE7"]["known_as"] == "PEtracer embryo set (sealed)", by["GSE7"]
    assert recs[0]["title"] == "Barcoding of [line] hESC sister clones across passages", recs[0]["title"]
    assert recs[0]["known"] and recs[0]["pluripotent_candidate"]
    assert not by["zenodo:5"]["known"] and by["zenodo:5"]["pluripotent_candidate"]
    # extended neutral-wording filter (retro 2026-11-19b): drops disease names, keeps core biology wording
    for t in ("Sclerotome and blastomere fate", "E14.5 lentiviral barcoding of clones",
              "Inducible Cre lineage tracing after transplantation",
              "Platelet-biased HSCs and thrombopoiesis in aging knockout mice",
              "Astrocyte and trophectoderm lineages", "Genotoxic stress in neuromesodermal progenitors"):
        assert not TITLE_DROP.search(t), t
    for t in ("Neutropenia after stroke", "Autism risk genes in organoids", "Glioblastoma clones",
              "Spina bifida", "Acute kidney injury", "Teratoma assay", "Xatinib response", "Xumab", "Xoparib"):
        assert TITLE_DROP.search(t), t
    masked = CELL_LINE.sub("[line]", "E14.5 embryo, histone H1 and H9 hESC")
    assert masked == "E14.5 embryo, histone H1 and [line] hESC", masked
    return {"ok": True, "kept": acc, "dropped": dropped}


def main():
    res = selftest() if (len(sys.argv) > 1 and sys.argv[1] == "selftest") else sweep()
    if "records" in res:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "records"}, indent=1)[:4000])


if __name__ == "__main__":
    main()
