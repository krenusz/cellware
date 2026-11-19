"""H25 classic search (amendment 2026-11-19, RETRO_2026-11-19b): the two searches that would have caught the
C17 split-test miss, run for one novelty finding.

1. A no-date, citation-sorted title+abstract search (OpenAlex) for each phrasing given. At least one phrasing
   must use the field's own words (taken from the source paper's title/abstract or the classic reviews), not
   the project's jargon: in the backtest our wording ("stem-cell clone lineage bias agrees hosts split") found
   none of the three classics, the field's wording ("hematopoietic stem cell clonal heterogeneity") found all
   three in the top 25 (Dykstra 2007 in the top 10).
2. The source paper's reference list ranked by citations (bibliography metadata only; fine for sealed sets:
   reference titles carry no results). Backtest: Dykstra 2007 ranked 18 of 76 (2020 source) and 11 of 48
   (2026 kinetics paper).

Titles pass the neutral-wording filter of lit_gap.py before printing or saving. Metadata only; no key, no account.

    python -I experiments/novelty_search.py <step> --source <doi> --q "<field words>" [--q "<our words>"] [--top 25]
    python -I experiments/novelty_search.py backtest      # the C17 split-test miss, must find the classics

Writes results_lit/novelty_<step>.json; the novelty file's `Classic search:` line cites it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lit_gap import CELL_LINE, SENSITIVE  # noqa: E402

import re  # noqa: E402

API = "https://api.openalex.org/works"
UA = {"User-Agent": "cellware-novelty/1.0 (public metadata, research use)"}
BACKTEST = {
    "source": "10.1038/s41586-020-2503-6",
    "queries": ["stem-cell clone lineage bias agrees hosts split size", "hematopoietic stem cell clonal heterogeneity"],
    "targets": {"10.1182/blood.v100.4.1302": "Muller-Sieburg 2002", "10.1016/j.stem.2007.05.015": "Dykstra 2007",
                "10.1016/j.cell.2016.10.045": "Yu 2016"},
}


def get(url):
    for i in range(5):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40))
        except Exception:  # noqa: BLE001
            time.sleep(4 * (i + 1))
    return {}


def row(w):
    title = w.get("display_name") or ""
    if re.search(SENSITIVE, title.lower()):
        return None
    return {"title": CELL_LINE.sub("[line]", title), "year": w.get("publication_year"),
            "cites": w.get("cited_by_count") or 0, "doi": (w.get("doi") or "").replace("https://doi.org/", "").lower()}


def concept(q, top):
    params = {"filter": f"title_and_abstract.search:{q}", "sort": "cited_by_count:desc", "per_page": top,
              "select": "doi,display_name,publication_year,cited_by_count"}
    r = get(f"{API}?" + urllib.parse.urlencode(params))
    rows = [x for x in (row(w) for w in r.get("results", [])) if x]
    return {"query": q, "matches": r.get("meta", {}).get("count"), "top": rows}


def references(doi, top):
    w = get(f"{API}/doi:{doi}?select=id,referenced_works")
    ids = [x.rsplit("/", 1)[-1] for x in w.get("referenced_works", [])]
    rows = []
    for i in range(0, len(ids), 50):
        r = get(f"{API}?" + urllib.parse.urlencode({"filter": "openalex:" + "|".join(ids[i:i + 50]), "per_page": 50,
                                                     "select": "doi,display_name,publication_year,cited_by_count"}))
        rows += [x for x in (row(w) for w in r.get("results", [])) if x]
        time.sleep(1)
    rows.sort(key=lambda x: -x["cites"])
    return {"source": doi, "n_references": len(ids), "top": rows[:top]}


def run(step, source, queries, top=25, ref_top=20):
    out = {"step": step, "date": time.strftime("%Y-%m-%d"), "concept": [], "references": None}
    for q in queries:
        out["concept"].append(concept(q, top))
        time.sleep(1)
    if source:
        out["references"] = references(source, ref_top)
    return out


def hits(res, targets):
    found = {}
    for c in res["concept"]:
        for i, x in enumerate(c["top"], 1):
            for t, name in targets.items():
                if x["doi"].startswith(t):
                    found.setdefault(name, []).append(f"concept '{c['query']}' rank {i}")
    for i, x in enumerate((res["references"] or {}).get("top", []), 1):
        for t, name in targets.items():
            if x["doi"].startswith(t):
                found.setdefault(name, []).append(f"source references rank {i}")
    return found


def main():
    if sys.argv[1:2] == ["backtest"]:
        res = run("backtest", BACKTEST["source"], BACKTEST["queries"])
        found = hits(res, BACKTEST["targets"])
        print(json.dumps(found, indent=1))
        assert found, "classic search missed every classic"
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("step")
    ap.add_argument("--source", default="")
    ap.add_argument("--q", action="append", required=True)
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()
    res = run(a.step, a.source, a.q, a.top)
    path = os.path.join(HERE, "results_lit", f"novelty_{a.step}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(res, open(path, "w", encoding="utf-8"), indent=1)
    for c in res["concept"]:
        print(f"\n## '{c['query']}' ({c['matches']} matches, by citations)")
        for i, x in enumerate(c["top"], 1):
            print(f"{i:>2}. {x['year']} {x['cites']:>6}  {x['title'][:110]}  {x['doi']}")
    if res["references"]:
        r = res["references"]
        print(f"\n## source references ({r['n_references']}), top by citations")
        for i, x in enumerate(r["top"], 1):
            print(f"{i:>2}. {x['year']} {x['cites']:>6}  {x['title'][:110]}  {x['doi']}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
