"""Coverage check: is every tried experiment and every candidate recorded, and does a new candidate repeat old work?

Bence 2026-11-22: "make a deterministic check for all the retros to check already tryed exps and candidate ones
to stay aligned with coverage". Every retro runs it and pastes its block (scope guard, H23).

It checks ALIGNMENT WITH RECORDED COVERAGE only (registry COVERAGE.tsv against EXPERIMENT_LOG headings, the oracle
ledger, lit/novelty files and the sealed partitions). It is NOT a novelty gate: a candidate it passes can still be
published; the H25 novelty file decides that.

Errors (exit 1):
  E1 a C*/CAND-* heading in EXPERIMENT_LOG maps to no registry row, or to more than one
  E2 a data set with a spent oracle look has no registry row; a 'confirmed' row on sealed data has no look
  E3 a lit/novelty file is missing from the registry (or listed twice), or its verdict is softer than the row's
  E4 an open row marked published lacks "Published but proceeding: Bence:" in its novelty file
  E5 an open row shares a data set and question family with a closed row and does not name it in `related`
  E6 an open row shares a data set and question family with a dont_repropose row (always an error)
  E7 unknown status / family / genre / novelty value or data set key; duplicate row id
  E8 sensitive wording in the registry (lit_gap.SENSITIVE)

Deterministic: offline, no clock, sorted output, line endings normalised before hashing.
    python -I experiments/coverage_check.py            # report + exit code
    python -I experiments/coverage_check.py --line     # one line for a retro: 'Coverage check: <hash> OK|FAIL n'
"""

from __future__ import annotations

import hashlib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lit_gap import SENSITIVE  # noqa: E402
from partitions import PARTITIONED  # noqa: E402

REGISTRY = os.path.join(HERE, "COVERAGE.tsv")
LOG = os.path.join(HERE, "EXPERIMENT_LOG.md")
LEDGER = os.path.join(HERE, "test_ledger.tsv")
NOVELTY = os.path.join(HERE, "lit", "novelty")
COLS = ["id", "steps", "dataset", "family", "genre", "status", "novelty", "related", "nfiles", "ref", "question"]
STATUS = {"confirmed", "negative", "withdrawn", "underpowered", "exploratory", "published_closed", "failed_gate",
          "dont_repropose", "open", "parked"}
OPEN = {"open"}
NOT_CLOSED = {"open", "parked"}
FAMILY = {"instr_select", "ko_fate_identifiable", "state_predicts_fate", "fate_heritable", "memory_decay",
          "mixing_rescue", "spatial_model", "chromatin_vs_rna", "lineage_position", "cross_system_law",
          "sampling_artefact", "data_gate", "signal_perturbation"}
GENRE = {"correction", "replication", "model", "gate"}
SEVERITY = {"unchecked": 0, "novel": 1, "not_found": 1, "partly": 2, "published": 3}
EXTERNAL = {"external", "larry_cytokine", "sis_skew"}  # data sets outside partitions.PARTITIONED
HEAD_RE = re.compile(r"^## (C\d+[^\s:(]*|CAND-[A-Z]\b)", re.M)
VERDICT_RE = re.compile(r"^\s*-?\s*Verdict:\s*(published|partly published|not found)\b", re.I | re.M)


def norm(text):
    return text.replace("\r\n", "\n")


def read(path):
    with open(path, encoding="utf-8") as fh:
        return norm(fh.read())


def parse_registry(text):
    lines = [x for x in norm(text).split("\n") if x.strip()]
    head = lines[0].split("\t")
    if head != COLS:
        raise SystemExit(f"COVERAGE.tsv header must be {COLS}")
    rows = []
    for x in lines[1:]:
        parts = x.split("\t")
        rows.append(dict(zip(COLS, parts + [""] * (len(COLS) - len(parts)))))
    return rows


def headings(log_text):
    return sorted({m.group(1).rstrip(".,") for m in HEAD_RE.finditer(log_text)})


def spent_looks(ledger_text):
    """{data set: set of spent partitions} from the oracle ledger's shard column ('ds:A', 'ds:AB')."""
    return {ds: parts for ds, (parts, _) in ledger_looks(ledger_text).items()}


def ledger_looks(ledger_text):
    """{data set: (spent partitions, experiment ids)} from the oracle ledger."""
    out = {}
    for line in norm(ledger_text).split("\n")[1:]:
        cols = line.split("\t")
        if len(cols) < 4 or ":" not in cols[3]:
            continue
        ds, parts = cols[3].split(":", 1)
        sp, ex = out.setdefault(ds, (set(), set()))
        sp.update(p for p in parts if p in "ABC")
        ex.add(cols[1])
    return out


def file_severity(text):
    v = [x.lower() for x in VERDICT_RE.findall(text)]
    sev = {"published": 3, "partly published": 2, "not found": 1}
    return max((sev[x] for x in v), default=0)


def datasets(row):
    return {d for d in row["dataset"].split(";") if d}


def check(rows, heads, spent, novelty_texts, partitioned, look_exps=None):
    """All errors as sorted strings. novelty_texts: {file stem: text}; look_exps: {data set: experiment ids}."""
    errs = []
    ids = [r["id"] for r in rows]
    for i in sorted({x for x in ids if ids.count(x) > 1}):
        errs.append(f"E7 duplicate row id {i}")
    known = set(partitioned) | EXTERNAL
    for r in rows:
        for k, vocab in (("status", STATUS), ("family", FAMILY), ("genre", GENRE), ("novelty", SEVERITY)):
            if r[k] not in vocab:
                errs.append(f"E7 {r['id']}: unknown {k} '{r[k]}'")
        for d in sorted(datasets(r) - known):
            errs.append(f"E7 {r['id']}: unknown data set '{d}'")
    # E1 log coverage
    for h in heads:
        hit = [r["id"] for r in rows if r["steps"] and re.fullmatch(r["steps"], h)]
        if len(hit) != 1:
            errs.append(f"E1 log heading {h} maps to {len(hit)} rows {hit}")
    # E2 ledger
    for ds in sorted(spent):
        exps = (look_exps or {}).get(ds, set())
        by_step = any(r["steps"] and re.fullmatch(r["steps"], e) for r in rows for e in exps)
        if not any(ds in datasets(r) for r in rows) and not by_step:
            errs.append(f"E2 data set {ds} has spent looks but no registry row (by data set or experiment id)")
    for r in rows:
        sealed = datasets(r) & set(partitioned)
        if r["status"] == "confirmed" and sealed and not any(spent.get(d) for d in sealed):
            errs.append(f"E2 {r['id']}: confirmed on sealed data but no oracle look on {sorted(sealed)}")
    # E3 novelty files
    listed = {}
    for r in rows:
        for stem in [x for x in r["nfiles"].split(";") if x]:
            listed.setdefault(stem, []).append(r["id"])
    for stem in sorted(novelty_texts):
        owners = listed.get(stem, [])
        if len(owners) != 1:
            errs.append(f"E3 lit/novelty/{stem}.md is listed in {len(owners)} rows {owners}")
    for stem, owners in sorted(listed.items()):
        if stem not in novelty_texts:
            errs.append(f"E3 {owners}: lit/novelty/{stem}.md does not exist")
            continue
        for oid in owners:
            row = next(r for r in rows if r["id"] == oid)
            if file_severity(novelty_texts[stem]) < SEVERITY[row["novelty"]]:
                errs.append(f"E3 {oid}: lit/novelty/{stem}.md verdict is softer than the registry ('{row['novelty']}')")
    # E4 published but open
    for r in rows:
        if r["status"] in OPEN and r["novelty"] == "published":
            texts = [novelty_texts.get(s, "") for s in r["nfiles"].split(";") if s]
            if not any(re.search(r"Published but proceeding:.*Bence:", t) for t in texts):
                errs.append(f"E4 {r['id']}: open but published, no 'Published but proceeding: Bence:' line")
    # E5 / E6 near-repeats
    for r in rows:
        if r["status"] not in OPEN:
            continue
        mine = datasets(r) - {"external"}
        for o in rows:
            if o is r or o["family"] != r["family"] or not (mine & datasets(o)):
                continue
            if o["status"] == "dont_repropose":
                errs.append(f"E6 {r['id']}: repeats dont_repropose row {o['id']} ({o['question']})")
            elif o["status"] not in NOT_CLOSED and o["id"] not in r["related"]:
                errs.append(f"E5 {r['id']}: same data set and family as closed {o['id']}; name it in 'related'")
    # E8 wording
    sens = re.compile(SENSITIVE, re.I)
    for r in rows:
        for k in [c for c in COLS if c != "dataset"]:  # data set keys are fixed identifiers
            m = sens.search(r[k])
            if m:
                errs.append(f"E8 {r['id']}: sensitive wording in {k}")
    return sorted(set(errs))


def sealed_table(rows, spent, partitioned):
    out = ["| data set | spent | unspent | rows |", "|---|---|---|---|"]
    for ds in sorted(partitioned):
        sp = "".join(sorted(spent.get(ds, set())))
        un = "".join(p for p in "ABC" if p not in sp)
        rs = ", ".join(f"{r['id']} ({r['status']})" for r in rows if ds in datasets(r)) or "NONE"
        out.append(f"| {ds} | {sp or '-'} | {un or '-'} | {rs} |")
    return out


def inputs():
    files = [REGISTRY, LOG, LEDGER] + sorted(
        os.path.join(NOVELTY, f) for f in os.listdir(NOVELTY) if f.endswith(".md") and f != "TEMPLATE.md")
    h = hashlib.sha256()
    for p in files:
        h.update(os.path.relpath(p, HERE).replace("\\", "/").encode() + b"\0" + read(p).encode("utf-8") + b"\0")
    return h.hexdigest()[:12]


def run():
    rows = parse_registry(read(REGISTRY))
    heads = headings(read(LOG))
    looks = ledger_looks(read(LEDGER))
    spent = {d: p for d, (p, _) in looks.items()}
    nov = {f[:-3]: read(os.path.join(NOVELTY, f)) for f in sorted(os.listdir(NOVELTY))
           if f.endswith(".md") and f != "TEMPLATE.md"}
    errs = check(rows, heads, spent, nov, PARTITIONED, {d: e for d, (_, e) in looks.items()})
    return rows, heads, spent, errs


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    rows, heads, spent, errs = run()
    line = f"Coverage check: {inputs()} {'OK' if not errs else f'FAIL {len(errs)}'}"
    if "--line" in sys.argv:
        print(line)
        sys.exit(1 if errs else 0)
    by = {}
    for r in rows:
        by.setdefault(r["status"], []).append(r["id"])
    print(f"## {line}")
    print(f"Registry rows {len(rows)}; log headings mapped {len(heads)}.")
    for s in sorted(by):
        print(f"- {s}: {', '.join(sorted(by[s]))}")
    print("\n### Sealed data")
    print("\n".join(sealed_table(rows, spent, PARTITIONED)))
    print("\n### Errors" if errs else "\n### Errors: none")
    for e in errs:
        print(f"- {e}")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
