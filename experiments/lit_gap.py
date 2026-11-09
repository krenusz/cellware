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
SEEDS = {
    "Weinreb 2020 (LARRY)": "10.1126/science.aaw3381",
    "Rieger 2009 (cytokines instruct)": "10.1126/science.1171461",
    "Mossadegh-Keller 2013 (M-CSF instructs)": "10.1038/nature12026",
    "Olsson 2016 (mixed-lineage states)": "10.1038/nature19348",
    "Buggenthin 2017 (prospective fate)": "10.1038/nmeth.4182",
    "Bonham-Carter 2024 (proliferation bias)": "10.1093/bioinformatics/btae483",
    "Endele 2017 (CSF-1 instructs)": "10.1182/blood-2016-05-714329",
    "Sierra 2010 (Gfi1 gates G-CSF)": "10.1182/blood-2009-10-246967",
    "LCL 2024 (lineage contrastive)": "10.1101/2024.10.28.620670",
    "Lin/Naik 2024 (restricted progenitors, small clones)": "10.1101/2024.03.28.587126",
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
}

# Open questions, as keyword groups; a paper scores on how many groups its title/abstract hits.
TOPICS = {
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
                           r"axial", r"symmetry breaking", r"pattern"],
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
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


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


def main(top=40):
    os.makedirs(LIT, exist_ok=True)
    pool, seed_ids = {}, {}
    for name, doi in SEEDS.items():
        w = get(f"{API}/doi:{doi}?select=id,referenced_works,display_name")
        sid = w["id"].rsplit("/", 1)[-1]
        seed_ids[sid] = name
        refs = [r.rsplit("/", 1)[-1] for r in w.get("referenced_works", [])]
        for i in range(0, len(refs), 50):
            for x in works("openalex_id:" + "|".join(refs[i : i + 50]), pages=1):
                pool.setdefault(x["id"], (x, set()))[1].add(f"cited by {name}")
        for x in works(f"cites:{sid}"):
            pool.setdefault(x["id"], (x, set()))[1].add(f"cites {name}")
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
                title=w.get("display_name"),
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
        f"Citation neighbourhood of {len(SEEDS)} core papers: {len(pool)} works, {len(rows)} "
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
    path = os.path.join(LIT, f"gap_{date}.md")
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    json.dump(rows, open(os.path.join(LIT, f"gap_{date}.json"), "w"), indent=1)
    print("\n".join(lines[:8]))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
