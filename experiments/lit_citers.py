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
SENSITIVE = r"cancer|tumou?r|leuk|lymphom|myelom|carcinom|melanom|malignan|oncog|metasta|drug|chemo|therap|patient|disease|clinical"
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
    err = None
    for i in range(8):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "cellware-lit/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001 - retry on rate limits and timeouts
            err = e
            hdr = getattr(e, "headers", None)
            wait = max(int((hdr.get("Retry-After") if hdr else 0) or 0), 5 * 2**i)
            print(f"retry {i + 1}: {type(e).__name__} {getattr(e, 'code', '')} wait {min(wait, 300)}s", flush=True)
            time.sleep(min(wait, 300))
    raise RuntimeError(f"failed: {url}: {err}")


def wid(doi):
    return get(f"{API}/doi:{doi}?select=id")["id"].rsplit("/", 1)[-1]


def abstract(w):
    inv = w.get("abstract_inverted_index") or {}
    pos = sorted((p, t) for t, ps in inv.items() for p in ps)
    return " ".join(t for _, t in pos)


def citers(work_id):
    """Plain page paging (cursor paging returned 401 without a key, run 44305237451)."""
    out, page = [], 1
    sel = "id,doi,display_name,publication_year,publication_date,abstract_inverted_index,type"
    while True:
        q = urllib.parse.urlencode({"filter": f"cites:{work_id}", "per-page": 100, "page": page, "select": sel})
        d = get(f"{API}?{q}")
        out += d["results"]
        if not d["results"] or len(out) >= d["meta"]["count"] or page >= 100:
            return out
        page += 1
        time.sleep(0.5)


def cociting_ids(target_id, nb_id):
    """IDs of works citing both a target and a neighbour (one small filtered query set)."""
    ids, page = set(), 1
    while True:
        q = urllib.parse.urlencode({"filter": f"cites:{target_id},cites:{nb_id}", "per-page": 100, "page": page, "select": "id"})
        d = get(f"{API}?{q}")
        ids |= {r["id"].rsplit("/", 1)[-1] for r in d["results"]}
        if not d["results"] or len(ids) >= d["meta"]["count"] or page >= 20:
            return ids
        page += 1
        time.sleep(0.5)


def main():
    os.makedirs(LIT, exist_ok=True)
    nb_ids = {wid(doi): name for name, doi in NEIGHBOURS.items()}
    rows = []
    for name, doi in TARGETS.items():
        tid = wid(doi)
        co_map = {}
        for nid, nname in nb_ids.items():
            for i in cociting_ids(tid, nid):
                co_map.setdefault(i, []).append(nname)
        for w in citers(tid):
            text = f"{w.get('display_name') or ''} {abstract(w)}".lower()
            hits = [k for k, rx in CLAIMS.items() if re.search(rx, text)]
            co = sorted(co_map.get(w["id"].rsplit("/", 1)[-1], []))
            rows.append(dict(target=name, title=w.get("display_name"), year=w.get("publication_year"),
                             date=w.get("publication_date"), doi=(w.get("doi") or "").replace("https://doi.org/", ""),
                             type=w.get("type"), has_abstract=bool(w.get("abstract_inverted_index")),
                             claims=hits, cocites=co))
    date = datetime.date.today().isoformat()
    # Neutral output only (standing rule: no disease/drug terms in saved files or public artifacts):
    # counts for every citer, rows only for flagged works whose title passes the term screen.
    counts = {t: dict(citers=sum(r["target"] == t for r in rows),
                      abstracts=sum(r["has_abstract"] for r in rows if r["target"] == t),
                      withheld=sum(bool(re.search(SENSITIVE, (r["title"] or "").lower())) for r in rows if r["target"] == t))
              for t in TARGETS}
    rows = [r for r in rows if (r["claims"] or r["cocites"]) and not re.search(SENSITIVE, (r["title"] or "").lower())]
    json.dump({"counts": counts, "rows": rows}, open(os.path.join(LIT, f"citers_{date}.json"), "w", encoding="utf-8"), indent=1)
    lines = [f"# Citer screen, {date}", ""]
    for name in TARGETS:
        rs = [r for r in rows if r["target"] == name]
        lines.append(f"## {name}: {counts[name]['citers']} citers, {counts[name]['abstracts']} with abstracts, {counts[name]['withheld']} titles withheld by the term screen")
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
