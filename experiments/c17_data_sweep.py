"""C17-S1D: data sweep for new systems of the general clone model, and the watch list.

Bence 2026-11-19: "do a retro citation map mand swep for data also".
Serves: C17-S1 (C17-S1R: grow the systems; the parameter table has committed blood 2, pluripotent 1).

Metadata only (H9): public catalogue records (accession, title, year, sample count, organism) and keyword flags.
No data file is downloaded. Records whose title carries condition-domain terms, or whose summary carries
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

# Further condition, agent and compound names (retro 2026-11-19b; same list in lit_gap.py).
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
    # retro 2026-11-20: same additions as lit_gap.SENSITIVE
    r"|infect|\bs?hiv\b|\bsiv\b|\w+(arabine|rubicin|platin|taxel|mycin)\b|resistan(ce|t) to|drug.?resist"
    r"|relaps|prognos|tremor|\bhsct\b|\bcar[- ]?t\b"
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
               r"polylox|carlin|darlin|lineage tree|cellular barcod|clone[ -]tracing|"
               # embryo scope (DS-EMB-2026-11-25): recorders, natural marks and imaging lineages
               r"gestalt|linnaeus|scartrace|tracer-?seq|lineage[ -]resolved|cell lineage|lineage reconstr|"
               r"somatic (mutation|variant)|mitochondrial (variant|mutation)|phylogen|photoconver|zebrabow|"
               r"brainbow|confetti",
    "embryo": r"embryo|tadpole|larva|gastrula|blastula|neurula|fetal|foetal|somite|neural crest|blastoid|"
              r"gastruloid|primitive streak",
    "tree_or_natural": r"lineage tree|phylogen|gestalt|\bscars?\b|scartrace|lineage[ -]resolved|invariant|somatic (mutation|variant)|"
                       r"mitochondrial (variant|mutation)",
    "imaging": r"light[ -]sheet|live imaging|time[ -]lapse|cell tracking|photoconver|zebrabow|brainbow|confetti",
    "split_or_memory": r"sister|split|twin|fluctuation|passag|serial|heritab|memory",
    "pluripotent_or_organoid": r"pluripoten|embryonic stem|\b[hm]?es ?cells?\b|\b[hm]?escs?\b|\b[hm]?i?pscs?\b|"
                               r"embryoid|gastruloid|epiblast|"
                               r"blastoid|organoid|embryo",
    "blood": r"hematopoie|haematopoie|progenitor|\bhspc|\bhsc",
    "single_cell": r"single[ -]cell|scrna|sc-?rna-?seq|10x",
}
WEIGHTS = {"lineage": 2, "split_or_memory": 2, "pluripotent_or_organoid": 2, "blood": 1, "single_cell": 1,
           "embryo": 2, "tree_or_natural": 2, "imaging": 1}

# Embryo scope (DS-EMB-2026-11-25; Bence: "target mostly embrios that could have human aspects", "wilden the
# search"): vertebrate embryos ranked by closeness to human development, new lineage methods, new repositories.
MAMMALS = ["Macaca fascicularis", "Macaca mulatta", "Callithrix jacchus", "Sus scrofa", "Bos taurus", "Ovis aries",
           "Oryctolagus cuniculus", "Rattus norvegicus"]
OTHER_VERT = ["Gallus gallus", "Xenopus laevis", "Xenopus tropicalis", "Danio rerio", "Oryzias latipes",
              "Ambystoma mexicanum"]
HUMAN_MOUSE = ["Homo sapiens", "Mus musculus"]
RANK = {**{o: 1 for o in HUMAN_MOUSE}, **{o: 2 for o in MAMMALS}, **{o: 3 for o in OTHER_VERT}}
LIN_TERMS = ('(barcod* OR "lineage tracing" OR "lineage recording" OR "lineage tree" OR "cell lineage" OR '
             '"lineage-resolved" OR GESTALT OR scGESTALT OR LINNAEUS OR ScarTrace OR TracerSeq OR "clonal analysis")')
EMB_TERMS = ('(embryo* OR tadpole* OR larva* OR gastrula* OR blastula* OR neurula* OR fetal OR foetal OR somite* OR '
             '"neural crest")')
GEO_QUERIES_EMBRYO = {  # name: (query, organisms)
    "embryo_lineage": (f"{LIN_TERMS} AND {EMB_TERMS}", MAMMALS + OTHER_VERT),
    "embryo_model_lineage": (f'{LIN_TERMS} AND (blastoid* OR gastruloid* OR somitoid* OR "embryo model" OR '
                             '"embryo models" OR "synthetic embryo" OR "primitive streak" OR amnion)',
                             HUMAN_MOUSE + MAMMALS),
    "natural_marks": ('("somatic mutation" OR "somatic mutations" OR "somatic variants" OR "mitochondrial variants" OR '
                      '"mitochondrial mutations" OR mtscATAC OR MAESTER OR phylogeny OR phylogenetic) AND '
                      '(lineage OR clonal OR clones) AND (embryo* OR fetal OR foetal OR development*)',
                      HUMAN_MOUSE + MAMMALS + OTHER_VERT),
    "imaging_lineage": ('("light-sheet" OR "light sheet" OR "live imaging" OR "cell tracking" OR "time-lapse" OR '
                        'photoconversion OR photoconvertible OR Zebrabow OR Brainbow OR Confetti) AND (lineage OR clonal) '
                        f'AND {EMB_TERMS}', HUMAN_MOUSE + MAMMALS + OTHER_VERT),
}
ZENODO_QUERIES_EMBRYO = {
    "embryo_lineage": '("lineage tracing" OR "cell lineage" OR "lineage tree" OR barcoding) AND (embryo OR tadpole OR '
                      'zebrafish OR Xenopus OR chick OR macaque OR marmoset OR pig OR rabbit OR blastoid OR gastruloid)',
    "natural_marks": '("somatic mutations" OR "mitochondrial variants" OR phylogeny) AND lineage AND (embryo OR fetal)',
    "imaging_lineage": '("cell tracking" OR "light-sheet" OR "live imaging") AND lineage AND embryo',
}
EBI_QUERIES_EMBRYO = {  # name: (collection, query)
    "embryo_lineage": ("arrayexpress", '"lineage tracing" AND embryo'),
    "vertebrate_lineage": ("arrayexpress", '"cell lineage" AND (zebrafish OR Xenopus OR chick OR macaque OR pig)'),
    "natural_marks": ("arrayexpress", '"somatic mutations" AND lineage AND (fetal OR embryo)'),
    "imaging_lineage": ("BioImages", 'lineage AND embryo AND (tracking OR "light-sheet" OR "live imaging")'),
    "imaging_cell_lineage": ("BioImages", '"cell lineage" AND embryo'),
}
TEXT_QUERIES_EMBRYO = {  # figshare, Dryad and the Single Cell Portal take plain text
    "embryo_lineage": '"lineage tracing" embryo',
    "cell_lineage_embryo": '"cell lineage" embryo',
    "natural_marks": '"somatic mutations" lineage embryo',
    "imaging_lineage": '"cell tracking" lineage embryo',
}
RECALL_EMBRYO = ["GSE105010"]  # zebrafish embryo scar-array trees (C17-S1G): must come back, else incomplete
SCOPE = "default"  # "embryo" with --scope embryo

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
    # frozen 2026-11-25 (DS-2026-11-25): the first author of the sister-well preprint (C5-S6c), GEO contributor field
    "hESC sister wells, author": 'Linneberg-Agerholm[AUTH] AND ("2026"[PDAT] : "3000"[PDAT])',
    "DeepTrack": 'DeepTrack AND ("2025"[PDAT] : "3000"[PDAT])',
}
# preprints whose data would reopen a parked chapter: a journal version (Crossref relation) is the release signal
WATCH_PREPRINT = {"hESC sister wells preprint": "10.64898/2026.09.02.748863"}
SINCE = None  # "YYYY/MM/DD": restrict the sweep to records released on or after this date (--since)


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


def geo_base():
    if not SINCE:
        return GEO_BASE
    return GEO_BASE.replace('"2019"[PDAT]', f'"{SINCE}"[PDAT]')


def geo_records(name, q):
    count, ids = esearch(f"({q}) AND {geo_base()}", retmax=300 if not SINCE else 2000)
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
    if SINCE:
        q = f"({q}) AND publication_date:[{SINCE.replace('/', '-')} TO *]"
    r = get("https://zenodo.org/api/records?" + urllib.parse.urlencode(
        {"q": q, "size": 25, "sort": "mostrecent"}))  # 25 is the anonymous maximum; the gate catches more
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
    p = {"query": q, "pageSize": 100}
    if SINCE:
        p.update({"sortBy": "release_date", "sortOrder": "descending"})
    url = "https://www.ebi.ac.uk/biostudies/api/v1/arrayexpress/search?" + urllib.parse.urlencode(p)
    r = get(url)
    out, oldest = [], None
    for h in r.get("hits", []):
        rel = str(h.get("release_date", ""))[:10]
        oldest = rel if oldest is None or rel < oldest else oldest
        if SINCE and rel < SINCE.replace("/", "-"):
            continue
        out.append({"source": "BioStudies", "accession": h.get("accession"), "title": h.get("title", ""),
                    "summary": h.get("content", "") or "", "year": rel[:4],
                    "n_samples": None, "organism": None, "query": name})
    if SINCE:
        # sorted newest first: the window is covered once the page reaches a release before SINCE
        covered = oldest is not None and oldest < SINCE.replace("/", "-")
        return (len(out) if covered else r.get("totalHits")), out
    return r.get("totalHits"), out


def geo_records_org(name, q, orgs):
    base = '("2015"[PDAT] : "3000"[PDAT]) AND gse[ETYP] AND (' + " OR ".join(f'"{o}"[ORGN]' for o in orgs) + ")"
    count, ids = esearch(f"({q}) AND {base}", retmax=5000)
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


def zenodo_paged(name, q, pages=24):
    out, total = [], None
    for page in range(1, pages + 1):
        r = get("https://zenodo.org/api/records?" + urllib.parse.urlencode(
            {"q": q, "size": 25, "page": page, "sort": "mostrecent"}))
        hits = r.get("hits", {}).get("hits", [])
        total = r.get("hits", {}).get("total")
        for h in hits:
            m = h.get("metadata", {})
            out.append({"source": "Zenodo", "accession": f"zenodo:{h.get('id')}", "title": m.get("title", ""),
                        "summary": re.sub(r"<[^>]+>", " ", m.get("description") or ""),
                        "year": (m.get("publication_date") or "")[:4], "n_samples": None, "organism": None,
                        "query": name, "_type": (m.get("resource_type") or {}).get("type")})
        if len(hits) < 25 or page * 25 >= (total or 0):
            break
        time.sleep(1.0)
    seen = len(out)  # every hit is counted as fetched; non-dataset records are dropped after counting
    return total, [o for o in out if o.pop("_type") in (None, "dataset")], seen


def ebi_paged(name, collection, q, pages=5):
    out, total = [], None
    for page in range(1, pages + 1):
        r = get(f"https://www.ebi.ac.uk/biostudies/api/v1/{collection}/search?" + urllib.parse.urlencode(
            {"query": q, "pageSize": 100, "page": page}))
        hits = r.get("hits", [])
        total = r.get("totalHits")
        for h in hits:
            out.append({"source": f"BioStudies:{collection}", "accession": h.get("accession"),
                        "title": h.get("title", ""), "summary": h.get("content", "") or "",
                        "year": str(h.get("release_date", ""))[:4], "n_samples": None, "organism": None,
                        "query": name})
        if len(hits) < 100 or page * 100 >= (total or 0):
            break
        time.sleep(1.0)
    return total, out


def figshare_records(name, q, pages=5):
    out, done = [], False
    for page in range(1, pages + 1):
        req = urllib.request.Request("https://api.figshare.com/v2/articles/search", headers={
            **UA, "Content-Type": "application/json"}, data=json.dumps(
            {"search_for": q, "item_type": 3, "page_size": 100, "page": page}).encode())
        with urllib.request.urlopen(req, timeout=90) as resp:
            hits = json.loads(resp.read())
        for h in hits:
            out.append({"source": "figshare", "accession": f"figshare:{h.get('id')}", "title": h.get("title", ""),
                        "summary": "", "year": (h.get("published_date") or "")[:4], "n_samples": None,
                        "organism": None, "query": name})
        if len(hits) < 100:
            done = True
            break
        time.sleep(1.0)
    return (len(out) if done else None), out


def dryad_records(name, q, pages=5):
    out, total = [], None
    for page in range(1, pages + 1):
        r = get("https://datadryad.org/api/v2/search?" + urllib.parse.urlencode({"q": q, "per_page": 100,
                                                                                  "page": page}))
        total = r.get("total")
        hits = (r.get("_embedded") or {}).get("stash:datasets", [])
        for h in hits:
            out.append({"source": "Dryad", "accession": f"dryad:{h.get('identifier')}", "title": h.get("title", ""),
                        "summary": h.get("abstract") or "", "year": str(h.get("publicationDate") or "")[:4],
                        "n_samples": None, "organism": None, "query": name})
        if len(hits) < 100 or page * 100 >= (total or 0):
            break
        time.sleep(1.0)
    return total, out


def scp_records(name, q, pages=30):
    out, total, last = [], None, False
    for page in range(1, pages + 1):
        url = "https://singlecell.broadinstitute.org/single_cell/api/v1/search?" + urllib.parse.urlencode(
            {"type": "study", "terms": q, "page": page})
        with urllib.request.urlopen(urllib.request.Request(url, headers={**UA, "Accept": "application/json"}),
                                    timeout=90) as resp:
            r = json.loads(resp.read())
        total = r.get("total_studies")
        for h in r.get("studies", []):
            out.append({"source": "SCP", "accession": f"scp:{h.get('accession')}", "title": h.get("name", ""),
                        "summary": h.get("description") or "", "year": None, "n_samples": h.get("cell_count"),
                        "organism": None, "query": name})
        if page >= (r.get("total_pages") or 0):
            last = True
            break
        time.sleep(0.8)
    return (total if last else (total or 0) + 1), out


def cellxgene_records(name):
    r = get("https://api.cellxgene.cziscience.com/curation/v1/collections")
    out = [{"source": "CELLxGENE", "accession": f"cxg:{c.get('collection_id')}", "title": c.get("name", ""),
            "summary": c.get("description") or "", "year": (c.get("published_at") or "")[:4], "n_samples": None,
            "organism": None, "query": name} for c in r]
    return len(out), out


def rank_of(rec):
    """Closeness to human development: 1 human/mouse (embryo models), 2 other mammals, 3 other vertebrates."""
    if rec.get("organism"):
        ranks = [RANK[o] for o in RANK if o.lower() in str(rec["organism"]).lower()]
        if ranks:
            return min(ranks)
    t = (rec.get("title") or "").lower()
    if re.search(r"human|blastoid|gastruloid|embryo model", t):
        return 1
    if re.search(r"macaque|monkey|primate|marmoset|pig|porcine|bovine|cattle|sheep|ovine|rabbit|\brat\b", t):
        return 2
    if re.search(r"zebrafish|xenopus|tadpole|frog|chick|avian|medaka|axolotl", t):
        return 3
    return None


def sweep_embryo():
    raw, counts, errors = [], {}, []

    def run(key, fn):
        try:
            res = fn()
            total, recs = res[0], res[1]
            fetched = res[2] if len(res) > 2 else len(recs)
            counts[key] = {"total": total, "fetched": fetched}
            raw.extend(recs)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{key}: {str(e)[:120]}")
        time.sleep(1.0)

    for n, (q, orgs) in GEO_QUERIES_EMBRYO.items():
        run(f"GEO:{n}", lambda q=q, orgs=orgs, n=n: geo_records_org(n, q, orgs))
    for n, q in ZENODO_QUERIES_EMBRYO.items():
        run(f"Zenodo:{n}", lambda q=q, n=n: zenodo_paged(n, q))
    for n, (coll, q) in EBI_QUERIES_EMBRYO.items():
        run(f"{coll}:{n}", lambda coll=coll, q=q, n=n: ebi_paged(n, coll, q))
    for n, q in TEXT_QUERIES_EMBRYO.items():
        run(f"figshare:{n}", lambda q=q, n=n: figshare_records(n, q))
        run(f"Dryad:{n}", lambda q=q, n=n: dryad_records(n, q))
        run(f"SCP:{n}", lambda q=q, n=n: scp_records(n, q))
    run("CELLxGENE:all_collections", lambda: cellxgene_records("all_collections"))
    recs, dropped = process(raw, known_accessions())
    for r in recs:
        r["rank"] = rank_of(r)
        r["embryo_candidate"] = (not r["known"] and "embryo" in r["flags"] and any(
            f in r["flags"] for f in ("split_or_memory", "tree_or_natural", "imaging")))
    recall = {a: any(r["accession"] == a for r in recs) for a in RECALL_EMBRYO}
    # figshare search is full text (> 1,000 hits for any phrase): a capped screen of 500 per query, outside the gate
    for k in counts:
        counts[k]["capped_screen"] = k.startswith("figshare:")
    complete = not errors and all(c["capped_screen"] or (c["total"] is not None and c["fetched"] >= c["total"])
                                  for c in counts.values())
    by_rank = {str(k): sum(r["embryo_candidate"] and r["rank"] == k for r in recs) for k in (1, 2, 3, None)}
    by = {"records_with_lineage_flag": len(recs), "new_records": sum(not r["known"] for r in recs),
          "new_study_families": len({r["family"] for r in recs if not r["known"]}),
          "embryo_candidates_new": sum(r["embryo_candidate"] for r in recs), "embryo_candidates_by_rank": by_rank,
          "dropped_excluded_domain": dropped}
    recs.sort(key=lambda x: (not x["embryo_candidate"], x["rank"] or 9, -x["score"], x["accession"]))
    return {"scope": "embryo", "complete": complete, "recall": recall, "queries": counts, "errors": errors,
            "summary": by, "records": recs[:3000]}


def crossref_relation(doi):
    """Journal version of a preprint, from Crossref metadata only (no text)."""
    m = get("https://api.crossref.org/works/" + urllib.parse.quote(doi))["message"]
    rel = m.get("relation") or {}
    return {"is_preprint_of": len(rel.get("is-preprint-of", [])), "has_version": len(rel.get("has-version", []))}


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
    for name, doi in WATCH_PREPRINT.items():
        try:
            res[name] = crossref_relation(doi)
        except Exception as e:  # noqa: BLE001
            res[name] = f"error: {str(e)[:60]}"
    return res


def counts_only(res):
    """What goes to the public run log: numbers and verdicts, never titles."""
    w = {k: ({"hits": v["hits"], "kept_titles": len(v["titles"])} if isinstance(v, dict) and "titles" in v else v)
         for k, v in res.get("watch", {}).items()}
    return {k: v for k, v in res.items() if k not in ("records", "watch")} | {"watch": w}


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
    # validity gate (DS-2026-11-25): a dated sweep is exhaustive only if every query fetched all its hits
    complete = not errors and all(c["total"] is not None and c["fetched"] >= c["total"] for c in counts.values())
    return {"since": SINCE, "complete": complete, "queries": counts, "errors": errors, "summary": by,
            "watch": watch(), "records": recs[:400]}


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
    # extended neutral-wording filter (retro 2026-11-19b): drops condition names, keeps core biology wording
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
    # dated mode: GEO release window and a titles-free public log
    global SINCE
    SINCE = "2026/11/12"
    assert '"2026/11/12"[PDAT] : "3000"[PDAT]' in geo_base(), geo_base()
    SINCE = None
    assert geo_base() == GEO_BASE
    shown = counts_only({"records": [1], "summary": {}, "watch": {"w": {"hits": 2, "titles": ["GSE9: secret"]},
                                                                  "p": {"is_preprint_of": 0, "has_version": 0}}})
    assert "records" not in shown and "secret" not in json.dumps(shown), shown
    assert shown["watch"]["w"] == {"hits": 2, "kept_titles": 1}, shown
    # embryo scope (DS-EMB-2026-11-25): new method words, embryo flag, human-closeness rank
    for t, want in (("scGESTALT lineage trees of the zebrafish embryo", {"lineage", "embryo", "tree_or_natural"}),
                    ("Somatic mutations reconstruct fetal cell phylogenies", {"lineage", "embryo", "tree_or_natural"}),
                    ("Light-sheet cell tracking and cell lineage of the chick gastrula", {"lineage", "embryo",
                                                                                         "imaging"}),
                    ("Scarce cells in adult liver", set())):
        got = set(flags_of(t)) & {"lineage", "embryo", "tree_or_natural", "imaging"}
        assert got == want, (t, got)
    assert rank_of({"organism": "Macaca fascicularis", "title": ""}) == 2
    assert rank_of({"organism": "Danio rerio", "title": ""}) == 3
    assert rank_of({"organism": None, "title": "Human blastoid lineage tracing"}) == 1
    assert rank_of({"organism": None, "title": "Xenopus tadpole cell lineage"}) == 3
    assert rank_of({"organism": None, "title": "Lineage of something"}) is None
    return {"ok": True, "kept": acc, "dropped": dropped}


def main():
    global SINCE, OUT
    argv = sys.argv[1:]
    if "--since" in argv:
        SINCE = argv[argv.index("--since") + 1]
        assert re.fullmatch(r"\d{4}/\d{2}/\d{2}", SINCE), SINCE
    if "--out" in argv:
        OUT = os.path.abspath(argv[argv.index("--out") + 1])
    embryo = "--scope" in argv and argv[argv.index("--scope") + 1] == "embryo"
    res = selftest() if (argv and argv[0] == "selftest") else (sweep_embryo() if embryo else sweep())
    if "records" in res:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
    print(json.dumps(counts_only(res), indent=1)[:4000])


if __name__ == "__main__":
    main()
