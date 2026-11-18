"""Citer screen for the correction notes (Bence 2026-11-18: "write them up" + "expand the citation map").

Pulls every work citing the target papers (OpenAlex, titles + abstracts), screens them for the specific claims
the notes make, and expands one step out: works citing both a target and a close neighbour. Metadata only.

    python -I experiments/lit_citers.py          # writes experiments/results_lit/citers_<date>.json (and .md)
"""

from __future__ import annotations

import datetime
import json
import os
import re
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
LIT = os.path.join(HERE, "results_lit")  # results_* so the Actions artifact returns it
API = "https://api.openalex.org/works"
TARGETS = {
    "LARRY (Weinreb 2020)": "10.1126/science.aaw3381",
    "SIS-seq/SIS-skew (Tian 2021)": "10.1016/j.immuni.2021.03.012",
}
NEIGHBOURS = {  # works whose co-citation with a target marks a likely reanalysis or debate paper
    "Rieger 2009 (cytokines instruct)": "10.1126/science.1171461",
    "Mossadegh-Keller 2013 (M-CSF instructs)": "10.1038/nature12026",
    "Naik 2013 (lineage imprinting)": "10.1038/nature12013",
    "Lin 2021 (Flt3L clonal tuning)": "10.1038/s41556-021-00636-7",
    "Brown 2020 review (cytokines instruct)": "10.3390/ijms21072362",
}
CLAIMS = {
    "instruction_vs_selection": r"(instruct\w*).{0,80}(select\w*)|(select\w*).{0,80}(instruct\w*)",
    "cytokine_GM": r"g-csf|m-csf|csf-?1|csf-?3|colony.stimulating",
    "identifiability": r"identifiab|non-?identif|cannot (be )?distinguish|indistinguishable",
    "reanalysis": r"re-?analy[sz]",
    "private_or_well_split": r"private clone|well-specific|split (?:into|across) wells|sister wells|sister cells",
    "bcor": r"\bbcor\b",
    "sis_skew": r"sis-?seq|sis-?skew|siblings? in silico|split.{0,20}sibling",
    "editing_efficiency": r"editing efficien|indel|knockout efficien",
}


def get(url):
    for i in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "cellware-lit/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception:  # noqa: BLE001 - retry on rate limits and timeouts
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"failed: {url}")


def wid(doi):
    return get(f"{API}/doi:{doi}?select=id")["id"].rsplit("/", 1)[-1]


def abstract(w):
    inv = w.get("abstract_inverted_index") or {}
    pos = sorted((p, t) for t, ps in inv.items() for p in ps)
    return " ".join(t for _, t in pos)


def citers(work_id):
    out, cursor = [], "*"
    sel = "id,doi,display_name,publication_year,publication_date,abstract_inverted_index,type,referenced_works"
    while cursor:
        q = urllib.parse.urlencode({"filter": f"cites:{work_id}", "per-page": 100, "cursor": cursor, "select": sel})
        d = get(f"{API}?{q}")
        out += d["results"]
        cursor = d["meta"].get("next_cursor")
        time.sleep(0.2)
    return out


def main():
    os.makedirs(LIT, exist_ok=True)
    nb_ids = {wid(doi): name for name, doi in NEIGHBOURS.items()}
    rows = []
    for name, doi in TARGETS.items():
        for w in citers(wid(doi)):
            text = f"{w.get('display_name') or ''} {abstract(w)}".lower()
            hits = [k for k, rx in CLAIMS.items() if re.search(rx, text)]
            refs = {r.rsplit("/", 1)[-1] for r in w.get("referenced_works") or []}
            co = sorted(nb_ids[i] for i in refs & set(nb_ids))
            rows.append(dict(target=name, title=w.get("display_name"), year=w.get("publication_year"),
                             date=w.get("publication_date"), doi=(w.get("doi") or "").replace("https://doi.org/", ""),
                             type=w.get("type"), has_abstract=bool(w.get("abstract_inverted_index")),
                             claims=hits, cocites=co))
    date = datetime.date.today().isoformat()
    json.dump(rows, open(os.path.join(LIT, f"citers_{date}.json"), "w", encoding="utf-8"), indent=1)
    lines = [f"# Citer screen, {date}", ""]
    for name in TARGETS:
        rs = [r for r in rows if r["target"] == name]
        lines.append(f"## {name}: {len(rs)} citers, {sum(r['has_abstract'] for r in rs)} with abstracts")
        flag = [r for r in rs if ({"instruction_vs_selection", "identifiability", "reanalysis", "bcor", "sis_skew",
                                   "private_or_well_split"} & set(r["claims"])) and (r["claims"] or r["cocites"])]
        flag.sort(key=lambda r: (-len(r["claims"]) - len(r["cocites"]), -(r["year"] or 0)))
        lines.append("")
        lines.append("| year | title | claims hit | co-cites | doi |")
        lines.append("|---|---|---|---|---|")
        for r in flag[:40]:
            t = (r["title"] or "").replace("|", "/")[:110]
            lines.append(f"| {r['year']} | {t} | {', '.join(r['claims'])} | {', '.join(r['cocites'])} | {r['doi']} |")
        lines.append("")
    open(os.path.join(LIT, f"citers_{date}.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines[:3]))
    print(f"{len(rows)} rows")


if __name__ == "__main__":
    main()
