"""C17-S1N: structure-only check of the neural progenitor clone sets, and the sub-series of GSE342607.

Bence 2026-11-19: "neural prog class!! good", "2607 can be still useful", "go 1 2".
Serves: C17-S1 (C17-S1R: grow the systems). Pre-registered in EXPERIMENT_LOG (C17-S1N) with the decision rule.

Metadata only (H9): GEO SOFT text (series and samples) reduced to design flags, distinct-value counts of sample
characteristics, library strategies and supplementary file names and sizes. No data file is downloaded; no
expression, clone composition or fate is read. Free text (summaries, design paragraphs) is never saved or printed;
only the matched design terms are. The DevMap sub-series of GSE342607 are the sealed PEtracer embryo set: only
accession, sample count and the word "sealed" are kept for them.

    python experiments/c17s1n_struct.py             # writes results_c17/c17s1n_struct.json (Actions)
    python experiments/c17s1n_struct.py selftest    # offline check of the parser and the decision rule
"""

from __future__ import annotations

import collections
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lit_gap import CELL_LINE, SENSITIVE  # noqa: E402

OUT = os.path.join(HERE, "results_c17", "c17s1n_struct.json")
UA = {"User-Agent": "cellware-struct/1.0 (public metadata, research use)"}
ACC = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?"
NEURAL = ["GSE300285", "GSE300287", "GSE300289", "GSE324903", "GSE324905", "GSE327468", "GSE210139"]
SUPER = ["GSE342607"]
SEALED_SUB = re.compile(r"devmap", re.I)

SISTER = re.compile(r"\bMADM\b|two[- ]colou?r|sister|split[- ]wells?|daughter cells|hemi-?clone|twin spot|"
                    r"green and red|red and green|GFP\W{1,3}tdTomato|tdTomato\W{1,3}GFP", re.I)
SERIAL = re.compile(r"\bserial\b|longitudinal|same (?:animals?|clones?|organoids?)|re-?sampl|half[- ]well|"
                    r"repeated sampling", re.I)
TREE = re.compile(r"CRISPR[- ](?:based )?(?:barcod|lineage|scar|array|record)|\bscars?\b|phylogen|lineage trees?|"
                  r"evolving barcode|cumulative edit|lineage recorder|GESTALT|DARLIN|PEtracer", re.I)
CLONE = re.compile(r"barcod|\bclon|TrackerSeq|piggyBac|lineage tracing|CloneSeq", re.I)
TIMEPOINT = re.compile(r"\b(E\d{1,2}(?:\.\d)?|P\d{1,2}|(?:day|d|week|wk) ?\d{1,3}|\d{1,3} ?(?:days?|weeks?|DIV))\b",
                       re.I)


def get(url, head=False):
    for i in range(5):
        try:
            req = urllib.request.Request(url, headers=UA, method="HEAD" if head else "GET")
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.headers.get("Content-Length") if head else r.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            time.sleep(3 * (i + 1))
    return None


def soft(acc, targ="self"):
    return get(ACC + urllib.parse.urlencode({"acc": acc, "targ": targ, "form": "text", "view": "brief"})) or ""


def fields(text, prefix):
    out = collections.defaultdict(list)
    for ln in text.splitlines():
        if ln.startswith(prefix) and " = " in ln:
            k, v = ln.split(" = ", 1)
            out[k[len(prefix):]].append(v.strip())
    return out


def neutral(v):
    return None if re.search(SENSITIVE, v.lower()) else CELL_LINE.sub("[line]", v)


def samples(text):
    """Per-sample blocks of a targ=gsm brief SOFT text."""
    blocks = re.split(r"^\^SAMPLE = ", text, flags=re.M)[1:]
    return [fields("!" + b if not b.startswith("!") else b, "!Sample_") for b in blocks]


def design(series_text, sample_blocks):
    s = fields(series_text, "!Series_")
    free = " ".join(s.get("title", []) + s.get("summary", []) + s.get("overall_design", []))
    stitles = [t for b in sample_blocks for t in b.get("title", [])]
    chars = collections.defaultdict(collections.Counter)
    for b in sample_blocks:
        for c in b.get("characteristics_ch1", []):
            k, _, v = c.partition(":")
            chars[k.strip().lower()][v.strip()] += 1
    allt = " ".join([free, *stitles, *(v for c in chars.values() for v in c)])
    tps = sorted({m.group(1) for m in TIMEPOINT.finditer(" ".join([*stitles, *(v for c in chars.values() for v in c)]))})
    flags = {
        "sister_split": sorted({m.group(0).lower() for m in SISTER.finditer(allt)}),
        "serial_sampling": sorted({m.group(0).lower() for m in SERIAL.finditer(allt)}),
        "trees": sorted({m.group(0).lower() for m in TREE.finditer(allt)}),
        "clone_barcodes": sorted({m.group(0).lower() for m in CLONE.finditer(allt)}),
    }
    char_out = {}
    for k, c in chars.items():
        vals = [x for x in (neutral(v) for v in c) if x]
        char_out[k] = {"n_distinct": len(c), "values": sorted(vals)[:12] if len(c) <= 12 else None}
    multi_t = len(tps) >= 2
    capable = bool(flags["sister_split"]) or bool(flags["trees"]) or (multi_t and bool(flags["serial_sampling"]))
    if capable:
        verdict = "memory-capable"
    elif multi_t and flags["clone_barcodes"]:
        verdict = "check: several timepoints, same clones not shown"
    else:
        verdict = "not memory-capable (clone composition only)" if flags["clone_barcodes"] else "no clone design"
    return {
        "title": neutral(" ".join(s.get("title", []))) or "[excluded wording]",
        "n_samples": len(s.get("sample_id", [])),
        "taxon": s.get("platform_taxid", []),
        "relations": s.get("relation", []),
        "library_strategy": dict(collections.Counter(v for b in sample_blocks for v in b.get("library_strategy", []))),
        "timepoints": tps,
        "flags": flags,
        "characteristics": char_out,
        "verdict": verdict,
        "supplementary": s.get("supplementary_file", []),
    }


def sizes(urls):
    out = []
    for u in urls:
        https = u.replace("ftp://ftp.ncbi.nlm.nih.gov", "https://ftp.ncbi.nlm.nih.gov")
        n = get(https, head=True)
        out.append({"file": u.rsplit("/", 1)[-1], "bytes": int(n) if n and n.isdigit() else None})
    return out


def check(acc):
    st = soft(acc)
    rec = design(st, samples(soft(acc, "gsm")))
    rec["accession"] = acc
    rec["supplementary"] = sizes(rec["supplementary"])
    return rec


def sweep():
    res = {"neural": {}, "superseries": {}}
    seen = set()
    queue = list(NEURAL)
    while queue:
        acc = queue.pop(0)
        if acc in seen:
            continue
        seen.add(acc)
        rec = check(acc)
        res["neural"][acc] = rec
        for rel in rec["relations"]:
            m = re.search(r"SuperSeries of:\s*(GSE\d+)|SubSeries of:\s*(GSE\d+)", rel)
            if m and (m.group(1) or m.group(2)) not in seen and len(seen) < 20:
                queue.append(m.group(1) or m.group(2))
        time.sleep(0.5)
    for acc in SUPER:
        st = soft(acc)
        subs = [re.search(r"(GSE\d+)", r).group(1) for r in fields(st, "!Series_")["relation"]
                if "SuperSeries of" in r]
        out = {}
        for sub in subs:
            sst = soft(sub)
            f = fields(sst, "!Series_")
            title = " ".join(f.get("title", []))
            if SEALED_SUB.search(title):
                out[sub] = {"sealed": "PEtracer embryo set (sealed)", "n_samples": len(f.get("sample_id", []))}
            else:
                rec = check(sub)
                rec["accession"] = sub
                out[sub] = rec
            time.sleep(0.5)
        res["superseries"][acc] = {"n_subseries": len(subs), "subseries": out}
    caps = [a for a, r in res["neural"].items() if r["verdict"] == "memory-capable"]
    res["summary"] = {
        "neural_series_checked": len(res["neural"]),
        "memory_capable": caps,
        "verdicts": {a: r["verdict"] for a, r in res["neural"].items()},
        "gse342607_unsealed_subseries": {a: r.get("verdict") for a, r in
                                         res["superseries"].get("GSE342607", {}).get("subseries", {}).items()
                                         if "sealed" not in r},
    }
    return res


def selftest():
    madm = ("!Series_title = Organoid clones by MADM\n!Series_overall_design = sparse MADM labelling gives green and "
            "red sister sublineages per clone\n!Series_sample_id = GSM1\n!Series_sample_id = GSM2\n")
    madm_s = ("^SAMPLE = GSM1\n!Sample_title = organoid day 30 rep1\n!Sample_characteristics_ch1 = time: day 30\n"
              "^SAMPLE = GSM2\n!Sample_title = organoid day 60 rep1\n!Sample_characteristics_ch1 = time: day 60\n")
    r = design(madm, samples(madm_s))
    assert r["verdict"] == "memory-capable" and "madm" in r["flags"]["sister_split"], r
    assert r["timepoints"] == ["day 30", "day 60"], r["timepoints"]
    tracker = ("!Series_title = Thalamus lineage by TrackerSeq barcodes\n!Series_overall_design = piggyBac barcodes "
               "at E10.5, collected at P7\n!Series_sample_id = GSM3\n")
    tr_s = "^SAMPLE = GSM3\n!Sample_title = P7 thalamus\n!Sample_characteristics_ch1 = age: P7\n"
    r = design(tracker, samples(tr_s))
    assert r["verdict"] == "not memory-capable (clone composition only)", r
    tree = "!Series_title = CRISPR-based barcoding of brain lineages\n!Series_sample_id = GSM4\n"
    r = design(tree, samples("^SAMPLE = GSM4\n!Sample_title = E15.5 cortex\n"))
    assert r["verdict"] == "memory-capable" and r["flags"]["trees"], r
    crispri = "!Series_title = TrackerSeq clones with CRISPRi knockdown\n!Series_sample_id = GSM7\n"
    r = design(crispri, samples("^SAMPLE = GSM7\n!Sample_title = P7 thalamus\n"))
    assert not r["flags"]["trees"] and r["verdict"].startswith("not memory-capable"), r
    multi = "!Series_title = Clone barcoding across stages\n!Series_sample_id = GSM5\n"
    r = design(multi, samples("^SAMPLE = GSM5\n!Sample_title = E12.5\n^SAMPLE = GSM6\n!Sample_title = E14.5\n"))
    assert r["verdict"].startswith("check"), r
    assert neutral("H9 cells") == "[line] cells" and neutral("a tumour sample") is None
    return {"ok": True}


def main():
    res = selftest() if sys.argv[1:2] == ["selftest"] else sweep()
    if "neural" in res:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1, default=str)
        res = res["summary"]
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
