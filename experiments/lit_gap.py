"""Literature gap map: the citation graph around our core papers, ranked by relevance to the
project's open questions, marked read / unread and by open-access status.

Standing retro step (2026-11-04): run at every retro; the report says which unread papers matter,
which are free, and which publishers the paywalled ones sit behind.

    uv run python experiments/lit_gap.py            # writes experiments/lit/gap_<date>.md

Uses the free OpenAlex API (titles, abstracts, open-access status). Nothing is downloaded but
metadata.
"""

from __future__ import annotations

import datetime
import glob
import json
import os
import re
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(__file__)
LIT = os.path.join(HERE, "lit")
API = "https://api.openalex.org/works"
FIELDS = (
    "id,doi,display_name,publication_year,cited_by_count,open_access,primary_location,"
    "abstract_inverted_index,type"
)

# Core papers whose neighbourhood we map (references and citers). Extend as the project grows.
SENSITIVE = r"cancer|tumou?r|leuk|lymphom|myelom|carcinom|melanom|malignan|oncog|metasta|drug|chemo|therap|patient|disease|clinical|syndrom|dysplas|fibros|diabet|alzheimer|parkinson|sepsis|autoimmun|infection|\bhiv\b|covid|sars|defect"
# Further disease, pathogen and drug names (retro 2026-11-19b; same list in c17_data_sweep.py). Lower case only:
# titles are lower-cased before matching.
SENSITIVE += (
    r"|anemi|anaemi|aplastic|thalass|sickle|hemophil|haemophil|penia\b|immunodeficien|dystroph|scleros|sclerotic"
    r"|fibrot|arthritis|lupus|asthma|allerg|obes(e|ity)|hypertens|ischemi|ischaemi|stroke|infarct|aneurysm"
    r"|embolism|thrombos[ie]s|thrombotic|hemorrhag|haemorrhag|cirrhos|hepatitis|colitis|crohn|inflammatory bowel"
    r"|ulcer|psoria|dementia|neurodegenerat|macular|epilep|autis|schizophren|deafness|opath(y|ies|ic)\b"
    r"|graft.versus.host|gvhd|\bdisorders?\b|\blesions?\b|pneumonia|eclamp|spina bifida|malnut|infertil"
    r"|congenital|pathogen|malaria|tubercul|influenza|zika|ebola|herpes|cytomegalo|septic|blastoma|adenom"
    r"|hepatom|teratom|mesotheliom|astrocytom|myeloprolif|lymphoprolif|\baml\b|\bcml\b|\bcll\b|metaplas"
    r"|hyperplas|atroph|cystic|trauma|injur|vaccin|antibiotic|pharmac|fluorouracil|\b5-?fu\b"
    r"|\w+(nib|mab|ciclib|parib)\b|cholesta|ichthyo|\bdeficiency\b|\btoxicity\b|mutations? in a|cardiovascular|risk factor|immuniz"
    r"|glioma|(gli|mening|ependym|astrocyt|sarc|neur|aden)oma\b|alcohol|addict|vulnerab|injur|lesion|ischem|stroke|epilep|seizure|autis|schizo|depress"
)
# Cell-line names are masked in saved titles (same pattern as c17_data_sweep.py).
CELL_LINE = re.compile(
    r"\b(hela|hek ?293[a-z]*|k562|mcf-?7|a549|hct-?116|jurkat|u2os|thp-?1|hl-?60|nb4|molm-?\d+|"
    r"mv4-?11|kasumi-?\d*|wtc-?11|hues\d+|kolf2?[\w.-]*|(?<!histone )h[179](?![.]\d)|"
    r"e14(tg2a)?(?![.]\d)|v6\.5|cgr8)\b",
    re.I,
)


def neutral(row):
    """True if a map row's title, venue, link and DOI carry no sensitive term (standing rule; the md filter
    already checks the whole table line)."""
    text = " ".join(str(row.get(k) or "") for k in ("title", "pub", "url", "doi")).lower()
    return not re.search(SENSITIVE, text)

SEEDS = {
    "Weinreb 2020 (LARRY)": "10.1126/science.aaw3381",
    "Dai 2024 (ReSisTrace)": "10.1038/s41467-024-45478-7",
    "Goyal 2023 (FateMap)": "10.1038/s41586-023-06342-8",
    "Rieger 2009 (cytokines instruct)": "10.1126/science.1171461",
    "Mossadegh-Keller 2013 (M-CSF instructs)": "10.1038/nature12026",
    "Olsson 2016 (mixed-lineage states)": "10.1038/nature19348",
    "Buggenthin 2017 (prospective fate)": "10.1038/nmeth.4182",
    "Bonham-Carter 2024 (proliferation bias)": "10.1093/bioinformatics/btae483",
    "Endele 2017 (CSF-1 instructs)": "10.1182/blood-2016-05-714329",
    "Sierra 2010 (Gfi1 gates G-CSF)": "10.1182/blood-2009-10-246967",
    "LCL 2024 (lineage contrastive)": "10.1101/2024.10.28.620670",
    "Lin/Naik 2024 (restricted progenitors, small clones)": "10.1101/2024.03.28.587126",
    # sister-split designs (2026-11-18, Bence: "go 1 2"): sisters of one clone sent to different wells/conditions
    "Tian 2021 (SIS-seq/SIS-skew)": "10.1016/j.immuni.2021.03.012",
    "Shaffer 2020 (memory sequencing)": "10.1016/j.cell.2020.07.003",
    "cHSPCTrack 2026 (human clones x cytokines)": "10.1182/blood.2026033834",
    "hESC clonal memory 2026 (sister-well endoderm)": "10.64898/2026.09.02.748863",
    "Regalado/Qiu 2026 (gastruloid fluctuation test)": "10.1016/j.cell.2026.09.057",
    "Grover 2014 (Epo guides MPPs to erythroid)": "10.1084/jem.20131189",
    "Tian 2021 (SIS-seq/SIS-skew, Bcor)": "10.1016/j.immuni.2021.03.012",
    "Naik 2013 (heritable lineage imprinting)": "10.1038/nature12013",
    "Lin 2021 (Flt3L clonal tuning)": "10.1038/s41556-021-00636-7",
    "Rodrigues 2018 (pDC lymphoid origin)": "10.1038/s41590-018-0136-9",
    "Dress 2019 (pDC from Ly6D+ lymphoid progenitors)": "10.1038/s41590-019-0420-3",
    # chapter 3 (gastruloids), added 2026-11-07
    "Ayyappan 2025 (gastruloid clone division of labour)": "10.1101/2025.07.12.664536",
    "Regalado 2025 (monoclonal gastruloids, GSE291244)": "10.1101/2025.05.23.655664",
    "Oriola 2024 (gastruloid symmetry breaking)": "10.1101/2024.12.16.628776",
    "Beccari 2018 (gastruloid multi-axial)": "10.1038/s41586-018-0578-0",
    "van den Brink 2014 (gastruloid symmetry breaking)": "10.1242/dev.113001",
    "Shaffer 2020 (MemorySeq, heritable expression)": "10.1016/j.cell.2020.07.003",
    # chapter 4 (monoclonal gastruloids, founder memory), added 2026-11-07b
    "Regalado/Qiu 2026 Cell (fluctuation test, gastruloids)": "10.1016/j.cell.2026.09.057",
    "Choi 2022 (DNA Typewriter)": "10.1038/s41586-022-04922-8",
    "Emert 2021 (Rewind, clonal memory of rare states)": "10.1038/s41587-021-00837-3",
    # chapter 4 (mixing rescue of elongation), added 2026-11-07c
    "Chimeric 3D gastruloids 2022 (Bra-/- + WT)": "10.1242/dev.200812",
    "Braccioli 2025 Dev Cell (cross-lineage dependencies)": "10.1016/j.devcel.2025.02.013",
    "Triandafillou 2025 (gastruloid spatial cell-type map)": "10.1101/2025.07.14.664617",
    # chapter 5 (clone memory carriers), added 2026-11-08
    "Linneberg-Agerholm 2026 (hESC clonal memory, sister wells, multiome)": "10.64898/2026.09.02.748863",
    "Mold 2024 (clonally heritable expression, RNA + ATAC)": "10.1016/j.cels.2024.01.004",
    # chapter 6 (cells into space: symmetry breaking, sorting, elongation), added 2026-11-09
    "Mori 2008 (wave-pinning polarity)": "10.1529/biophysj.107.120824",
    "Graner & Glazier 1992 (Potts cell sorting)": "10.1103/PhysRevLett.69.2013",
    "Mongera 2018 (fluid-to-solid jamming, axis elongation)": "10.1038/s41586-018-0479-2",
    "McNamara 2024 (recording morphogen signals, gastruloid symmetry breaking)": "10.1038/s41556-024-01521-9",
    "Hashmi 2022 (collective movement, endoderm-like region in gastruloids)": "10.7554/eLife.59371",
    "Turner 2017 (gastruloid A-P polarity and elongation)": "10.1242/dev.150391",
    # chapter 6 models adopted in C6-S7 (sorting, long-range pull, crawling, outer start), added 2026-11-10
    "Guan/Hormoz 2025 (agent-based gastruloid symmetry breaking)": "10.1101/2025.09.27.678924",
    "de Jong 2024 (crawling + differential adhesion, gastruloid shapes)": "10.1371/journal.pcbi.1011825",
    "Suppinger 2023 (gastruloid Sox2/Bra spatial onset)": "10.1016/j.stem.2023.04.018",
    "Serrano Najera 2025 (gastruloids on laminin, pole by rearrangement)": "10.1101/2025.08.04.668470",
    "Trani-Bustos 2025 (surface mechanics, proliferation-driven elongation)": "10.1101/2025.10.27.684710",
    # retro 2026-11-10b: next chapter leans to cell differentiation (Bence)
    "Pijuan-Sala 2019 (mouse gastrulation single-cell atlas)": "10.1038/s41586-019-0933-9",
    "Wagner 2018 (zebrafish single-cell lineage landscape)": "10.1126/science.aar4362",
    "Briggs 2018 (Xenopus single-cell differentiation)": "10.1126/science.aar5780",
    # chapter 7 (where is clone fate carried: chromatin vs RNA), added at the C7-S3 retro 2026-11-11
    "Jindal 2024 (CellTag-multi, RNA + ATAC clonal fate)": "10.1038/s41587-023-01931-4",
    "Biddy 2018 (CellTag reprogramming lineage)": "10.1038/s41586-018-0744-4",
    "Ma 2020 (SHARE-seq chromatin potential)": "10.1016/j.cell.2020.09.056",
    # chapter 14 (state beyond type predicts clade fate in embryos), added at C14 close 2026-11-16
    "PEtracer 2025 (prime-editing lineage recorder)": "10.1126/science.adx3800",
    "Chan 2019 (mouse embryogenesis recording)": "10.1038/s41586-019-1184-5",
    "Kalhor 2018 (whole-mouse developmental barcoding)": "10.1126/science.aat9804",
    "Bowling 2020 (CARLIN lineage mouse)": "10.1016/j.cell.2020.04.048",
    "Minkina 2022 (lineage-tethered state heterogeneity)": "10.1101/2022.05.12.491602",
    "Eisele 2024 (GEMLI, memory-gene lineages)": "10.1038/s41467-024-47158-y",
    "Quantitative fate mapping 2022": "10.1016/j.cell.2022.10.028",
    "Nat Genet 2026 (spatiotemporal lineage)": "10.1038/s41588-026-02739-z",
    # chapter 16 (hidden inherited state, paraxial vs lateral plate on embryo trees), added at C16 close 2026-11-17
    "Lawson 1991 (epiblast clonal fate map)": "10.1242/dev.113.3.891",
    "Kinder 1999 (order of mesoderm allocation)": "10.1242/dev.126.21.4691",
    "Tzouanacou 2009 (lineage segregations by clonal analysis)": "10.1016/j.devcel.2009.08.002",
    "Wymeersch 2016 (position-dependent streak progenitor plasticity)": "10.7554/eLife.10042",
    "Dubrulle 2004 (fgf8 mRNA decay gradient)": "10.1038/nature02216",
    "Deschamps 2017 (Hox clock, axial timing)": "10.1101/gad.303123.117",
    "Hormoz 2016 (state transitions from lineage trees)": "10.1016/j.cels.2016.10.015",
    "Beaulieu 2013 (hidden-rate trait model)": "10.1093/sysbio/syt034",
    "Colgan 2026 Cell (embryo data source; titles and abstracts only)": "10.1016/j.cell.2026.09.050",
    # chapter 17 (one clone model across systems; split-recipient transplants), added at retro 2026-11-19b
    "Rodriguez-Fraticelli 2020 (LARRY HSC transplants, split recipients; sealed set source)": "10.1038/s41586-020-2503-6",
    "Rodriguez-Correa 2026 (kinetics model, extrinsic lineage bias)": "10.1038/s41556-026-01958-0",
    "Ding 2026 (cortical TF CRISPRi + lineage; sealed set source)": "10.1038/s41586-025-09997-7",
    "Muller-Sieburg 2002 (daughter HSCs of a clone share lineage contribution)": "10.1182/blood.v100.4.1302.h81602001302_1302_1309",
    "Dykstra 2007 (distinct HSC programs propagate on serial transplant)": "10.1016/j.stem.2007.05.015",
    "Yu 2016 (epigenetic memory, clone-autonomous HSC behaviour)": "10.1016/j.cell.2016.10.045",
    "Klein & Simons 2011 (universal patterns of stem cell fate)": "10.1242/dev.060103",
    "Wang 2022 (CoSpar, fate bias from lineage + state)": "10.1038/s41587-022-01209-1",
    "Busch 2015 (unperturbed haematopoiesis from stem cells)": "10.1038/nature14242",
    "Carrelha 2018 (lineage-restricted fates of multipotent HSCs)": "10.1038/nature25455",
    "Pei 2017 (Polylox, HSC fates in vivo)": "10.1038/nature23653",
}

# Open questions, as keyword groups; a paper scores on how many groups its title/abstract hits.
TOPICS = {
    "fate_carrier_priming": [r"lineage priming", r"chromatin potential", r"multiome", r"share-seq", r"primed.{0,30}(chromatin|enhancer|state)", r"predict.{0,40}fate"],
    "differentiation_commitment": [r"commitment", r"fate restrict", r"fate bias", r"differentiation trajector", r"lineage tracing", r"pseudotime"],
    "mixing_rescue": [r"chimer", r"non-autonomous", r"gastruloid.{0,40}(rescue|mix)", r"elongat", r"mixed population"],
    "clonal_dominance": [r"clonal dominance", r"cell competition", r"chimer", r"take ?over", r"winner"],
    "chromatin_memory": [r"chromatin accessib", r"atac", r"epigenetic memory", r"enhancer.{0,30}prim", r"clonal.{0,20}(signature|fingerprint)"],
    "imprinting_potency": [r"imprint", r"dlk1", r"meg3", r"tet1", r"methylation"],
    "instruction_vs_selection": [
        r"instruct",
        r"selecti",
        r"permissive",
        r"lineage choice",
        r"lineage commitment",
        r"fate decision",
        r"fate choice",
    ],
    "csf_signalling": [
        r"g-csf",
        r"m-csf",
        r"csf1r",
        r"csf3r",
        r"colony-stimulating",
        r"colony stimulating",
        r"cytokine",
    ],
    "neutrophil_monocyte": [
        r"neutrophil",
        r"monocyte",
        r"granulocyt",
        r"\bgmp",
        r"cebpa",
        r"c/ebp",
        r"pu\.1",
        r"irf8",
        r"gfi1",
        r"myeloid",
    ],
    "clonal_tracing": [
        r"lineage tracing",
        r"barcod",
        r"clonal",
        r"\blarry\b",
        r"cospar",
        r"sister",
        r"single-cell lineage",
    ],
    "fate_memory": [r"memory", r"heritab", r"fluctuation", r"transient", r"inherit", r"passag",
                    r"epigenetic", r"clonal variab"],
    "sorting_patterning": [r"cell sorting", r"segregat", r"differential adhesion", r"anterior.posterior",
                           r"axial", r"symmetry breaking", r"pattern", r"convergent extension", r"intercalat", r"planar cell polarity"],
    "imaging_ssl": [
        r"time-lapse",
        r"timelapse",
        r"live imaging",
        r"self-supervised",
        r"contrastive",
        r"dino",
        r"world model",
    ],
    "dendritic_cells": [r"dendritic", r"pdc", r"cdc1", r"cdc2", r"flt3"],
    "genetic_perturbation": [r"crispr", r"knockout", r"perturb", r"bcor", r"deletion"],
    "spatial_mechanism": [r"chemota", r"wave.pinning", r"potts", r"vertex model", r"agent-based", r"jamming",
                          r"convergent extension", r"collective (cell )?migration", r"reaction.diffusion", r"tissue mechanic"],
    "morphogenesis_long_term": [r"gastruloid", r"embryo", r"morphogen", r"organoid", r"self-organi"],
    # C16 close 2026-11-17: timing and place of streak exit, axial identity clocks
    "axial_timing": [r"primitive streak", r"paraxial", r"lateral plate", r"fgf8", r"retinoic",
                     r"axial (elongation|identity|progenitor)", r"\bhox", r"neuromesodermal", r"fate restriction"],
    # C14 close 2026-11-16: in vivo lineage + state data with many independent units (fresh hold-outs)
    "in_vivo_lineage_state": [r"lineage recorder", r"molecular recording", r"phylogen", r"cassiopeia",
                              r"in vivo lineage", r"whole.organism lineage", r"lineage tree", r"prime editing"],
    "sister_split": [r"sister (cells?|wells?|clones?)", r"split.{0,30}(wells|conditions|sisters|across)",
                     r"twin", r"fluctuation test", r"same clone.{0,40}(different|two|both) (conditions|wells|environments)",
                     r"clonal (split|splitting)", r"barcod.{0,60}(divided|split)"],
    "replicate_units": [r"embryos", r"replicate", r"independent (mice|animals|embryos|individuals|donors)",
                        r"geo accession", r"data availability", r"atlas"],
    # C17 retro 2026-11-19b: cross-system clone laws and transplant clone behaviour
    "cross_system_model": [r"universal", r"across (tissues|systems|organs|species)", r"general (model|law|principle)",
                           r"clone.size distribution", r"scaling", r"meta-analys", r"hierarchical model"],
    "transplant_clonal": [r"transplant", r"engraft", r"repopulat", r"(myeloid|lymphoid|lineage).bias",
                          r"clonal (behaviou?r|output|heterogeneity)", r"daughter (cells|hscs?)"],
    "inference_identifiability": [
        r"identifiab",
        r"inference",
        r"bayesian",
        r"branching process",
        r"mathematical model",
        r"stochastic model",
    ],
}


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "cellware-lit-gap"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception:
            if attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))


def works(filt, pages=3):
    out = []
    for page in range(1, pages + 1):
        q = urllib.parse.urlencode(
            {
                "filter": filt,
                "per-page": 200,
                "page": page,
                "select": FIELDS,
                "sort": "cited_by_count:desc",
            }
        )
        res = get(f"{API}?{q}")["results"]
        out += res
        if len(res) < 200:
            break
    return out


def abstract(w):
    inv = w.get("abstract_inverted_index") or {}
    pos = sorted((p, t) for t, ps in inv.items() for p in ps)
    return " ".join(t for _, t in pos)


def score(w):
    text = f"{w.get('display_name') or ''} {abstract(w)}".lower()
    hits = {k: sum(bool(re.search(p, text)) for p in ps) for k, ps in TOPICS.items()}
    groups = [k for k, v in hits.items() if v]
    return sum(min(v, 3) for v in hits.values()) + 2 * len(groups), groups


def read_dois():
    """DOIs we have read: every DOI or doi.org link in the project's literature notes, plus
    experiments/lit/read.txt (one DOI per line) for papers read without a written note."""
    found = set()
    for f in glob.glob(os.path.join(HERE, "LITERATURE_SWEEP_*.md")) + glob.glob(
        os.path.join(HERE, "WRITEUP_*.md")
    ):
        found |= {
            d.lower().rstrip(".,)")
            for d in re.findall(r"10\.\d{4,9}/[^\s\])>\"']+", open(f, encoding="utf-8").read())
        }
    extra = os.path.join(LIT, "read.txt")
    if os.path.exists(extra):
        found |= {line.strip().lower() for line in open(extra, encoding="utf-8") if line.strip()}
    return found


def access(w):
    oa = w.get("open_access") or {}
    src = (w.get("primary_location") or {}).get("source") or {}
    pub = src.get("host_organization_name") or src.get("display_name") or "unknown"
    return ("open" if oa.get("is_oa") else "paywalled"), oa.get("oa_url"), pub


def main(top=40, tag=""):
    os.makedirs(LIT, exist_ok=True)
    pool, seed_ids, unresolved = {}, {}, []
    for name, doi in SEEDS.items():
        try:
            w = get(f"{API}/doi:{doi}?select=id,referenced_works,display_name")
        except Exception as e:  # noqa: BLE001  (a wrong or unindexed DOI must not stop the map)
            unresolved.append(f"{name} ({doi}): {str(e)[:60]}")
            continue
        sid = w["id"].rsplit("/", 1)[-1]
        seed_ids[sid] = name
        refs = [r.rsplit("/", 1)[-1] for r in w.get("referenced_works", [])]
        try:
            for i in range(0, len(refs), 50):
                for x in works("openalex_id:" + "|".join(refs[i : i + 50]), pages=1):
                    pool.setdefault(x["id"], (x, set()))[1].add(f"cited by {name}")
            for x in works(f"cites:{sid}"):
                pool.setdefault(x["id"], (x, set()))[1].add(f"cites {name}")
        except Exception as e:  # noqa: BLE001
            unresolved.append(f"{name} ({doi}) partial: {str(e)[:60]}")
    print(f"seeds resolved {len(seed_ids)} of {len(SEEDS)}; unresolved or partial: {unresolved}", flush=True)
    read = read_dois()
    rows = []
    for wid, (w, via) in pool.items():
        if wid.rsplit("/", 1)[-1] in seed_ids:
            continue
        s, groups = score(w)
        if len(groups) < 2:
            continue
        doi = (w.get("doi") or "").replace("https://doi.org/", "").lower()
        status, url, pub = access(w)
        rows.append(
            dict(
                score=s + 2 * (len(via) - 1),
                groups=groups,
                via=sorted(via),
                doi=doi,
                title=CELL_LINE.sub("[line]", w.get("display_name") or ""),
                year=w.get("publication_year"),
                cites=w.get("cited_by_count", 0),
                status=status,
                url=url,
                pub=pub,
                read=doi in read,
            )
        )
    rows.sort(key=lambda r: (-r["score"], -r["cites"]))
    unread = [r for r in rows if not r["read"]][:top]
    date = datetime.date.today().isoformat()
    pay = {}
    for r in unread:
        if r["status"] == "paywalled":
            pay[r["pub"]] = pay.get(r["pub"], 0) + 1
    lines = [
        f"# Literature gap map, {date}",
        "",
        f"Citation neighbourhood of {len(seed_ids)} of {len(SEEDS)} core papers (unresolved: {len(unresolved)}): "
        f"{len(pool)} works, {len(rows)} "
        f"relevant to at least two open questions, {sum(r['read'] for r in rows)} already read.",
        "",
        f"**Top {len(unread)} unread:** {sum(r['status'] == 'open' for r in unread)} open access, "
        f"{sum(r['status'] == 'paywalled' for r in unread)} paywalled.",
        "",
        "Paywalled, by publisher: "
        + (", ".join(f"{p} {n}" for p, n in sorted(pay.items(), key=lambda t: -t[1])) or "none"),
        "",
        "| # | score | year | title | topics | link | access | via |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(unread, 1):
        link = f"[doi](https://doi.org/{r['doi']})" if r["doi"] else ""
        acc = (
            f"[open]({r['url']})"
            if r["status"] == "open" and r["url"]
            else ("open" if r["status"] == "open" else f"paywalled ({r['pub']})")
        )
        title = (r["title"] or "").replace("|", "/")
        lines.append(
            f"| {i} | {r['score']} | {r['year']} | {title} | {', '.join(r['groups'])} | {link} "
            f"| {acc} | {'; '.join(r['via'])} |"
        )
    # neutral output (standing rule): drop rows whose title, venue, link or DOI carries a sensitive term
    rows = [r for r in rows if neutral(r)]
    lines = [ln for ln in lines if not (ln.startswith("| ") and re.search(SENSITIVE, ln.lower()))]
    stem = f"gap_{date}" + (f"_{tag}" if tag else "")
    path = os.path.join(LIT, f"{stem}.md")
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    json.dump(rows, open(os.path.join(LIT, f"{stem}.json"), "w"), indent=1)
    print("\n".join(lines[:8]))
    print(f"wrote {path}")


if __name__ == "__main__":
    import sys

    main(tag=sys.argv[1] if len(sys.argv) > 1 else "")  # optional tag: a second map on the same day
