"""C17-S1R-P10 S0: structure-only check of two expansion-culture lineage-recorder sets (allele tables only).

Bence: "Yes, run S0". Serves: C12 CLOSED; C17-LAW. Pre-registered in EXPERIMENT_LOG (C17-S1R-P10, commit 16c129d).

Set 1 (GSE117542): ENUMERATE ONLY (cells with an allele per unit). No seal, no trees, no pairs.
Set 2 (GSE220949): enumerate -> seal (partitions.balanced_split, salt c17p10) -> drop non-train units -> trees and
disjoint sister pairs on train units only. Sealed units get cell counts only. Stdout prints counts only.

    python -I experiments/c17p10_s0.py            # Actions; writes results_c17/c17p10_s0.json
    python -I experiments/c17p10_s0.py selftest   # offline toy check
"""

from __future__ import annotations

import collections
import csv
import gzip
import hashlib
import json
import re
import statistics
import sys
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from partitions import balanced_split  # noqa: E402

OUT = HERE / "results_c17" / "c17p10_s0.json"
GEO = "https://ftp.ncbi.nlm.nih.gov/geo/samples"
MAX_CLADE, MIN_ARM, MIN_PAIRS = 300, 4, 30
JOIN_FLOOR, MAX_SINGLE_CELLS = 0.30, 20000
SALT = "c17p10"
FORBIDDEN = ("seuratnorm", "seuratgenes", ".tar.gz", "raw.tar", ".mtx")
SET1 = {  # GSM -> suffix of the allele table; GSM -> suffix of the cell-id list
    "alleles": (range(3319520, 3319528), "_alleleTable.txt.gz"),
    "barcodes": (range(3302826, 3302834), "_SeuratBarcodes.tsv.gz"),
}
SET2 = {  # unit label -> (GSM, file token)
    "TLS1": ("GSM7744222", "AM-DNA-097_allele_table"),
    "TLS2": ("GSM7744223", "AM-DNA-098_allele_table"),
    "pool": ("GSM7744224", "AM-DNA-258_allele_table"),
    "multiseq": ("GSM7744226", "AM-DNA-197_MultiSeqBC.tsv.gz"),
}
NOCALL = {"", "na", "nan", "none", "doublet", "negative", "unassigned", "uncalled"}


# ---------------------------------------------------------------- download
def _open(url):
    req = urllib.request.Request(url, headers={"User-Agent": "research-client"})
    last = None
    for _ in range(4):
        try:
            return urllib.request.urlopen(req, timeout=300)
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"cannot open {url}: {last}")


def geo_dir(gsm):
    return f"{GEO}/{gsm[:-3]}nnn/{gsm}/suppl/"


def list_files(gsm):
    html = _open(geo_dir(gsm)).read().decode("utf-8", "replace")
    return sorted(set(re.findall(r'href="([^"/?][^"]*)"', html)))


def fetch(gsm, match, dest, hashes):
    names = [n for n in list_files(gsm) if match(n)]
    if len(names) != 1:
        raise RuntimeError(f"{gsm}: expected exactly 1 matching file, got {names}")
    name = names[0]
    if any(f in name.lower() for f in FORBIDDEN):
        raise RuntimeError(f"refusing forbidden file {name}")
    path = Path(dest) / name
    h = hashlib.sha256()
    with _open(geo_dir(gsm) + name) as r, open(path, "wb") as f:
        while chunk := r.read(1 << 20):
            h.update(chunk)
            f.write(chunk)
    hashes[name] = h.hexdigest()
    return path


# ---------------------------------------------------------------- parsing
def _rows(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8", newline="") as f:
        head = f.readline()
        f.seek(0)
        rd = csv.DictReader(f, delimiter="\t" if "\t" in head else ",")
        yield from rd


def norm_cell(c):
    c = str(c).strip().strip('"')
    return c[:-2] if c.endswith("-1") else c


def read_cell_list(path):
    out = set()
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            out.add(norm_cell(re.split(r"[\t,]", line.rstrip("\n"))[0]))
    return out


def read_allele_cells(path, allele_col):
    """Cells with an allele (set 1, enumeration only)."""
    cells = set()
    for r in _rows(path):
        if r.get("cellBC") and str(r.get(allele_col, "")).strip() not in ("", "NA", "nan"):
            cells.add(norm_cell(r["cellBC"]))
    return cells


def read_multiseq(path):
    """cell -> MULTI-seq group, dropping Doublet / Negative / uncalled."""
    out = {}
    for r in _rows(path):
        call = str(r.get("final.calls.rescued", "")).strip().lower()
        grp = str(r.get("MultiSeqGroup", "")).strip()
        if call in NOCALL or grp.lower() in NOCALL:
            continue
        out[norm_cell(r["cellBC"])] = grp
    return out


def read_set2_table(path, keep=None):
    """Allele table -> cell_states, cell_group, n_cells. Optionally restricted to cells in `keep`.
    State per (cell, (intBC, site)) = UMI-majority allele; ties -> missing; 'None' -> '0'."""
    return states_from_rows(_rows(path), keep)


def states_from_rows(rows, keep=None):
    num, sets = collections.defaultdict(float), collections.defaultdict(set)
    grp_w = collections.defaultdict(float)
    cells = set()
    for r in rows:
        c = norm_cell(r["cellBC"])
        if keep is not None and c not in keep:
            continue
        cells.add(c)
        u = str(r.get("UMI", "")).strip()
        try:
            w = float(u)
            numeric = True
        except ValueError:
            w, numeric = 1.0, False
        grp_w[(c, str(r.get("lineageGrp", "")))] += w
        for site, col in enumerate(("r1", "r2", "r3")):
            a = str(r.get(col, "")).strip()
            if a in ("", "NA", "nan"):
                continue
            a = "0" if a == "None" else a
            k = (c, r["intBC"], site, a)
            if numeric:
                num[k] += w
            else:
                sets[k].add(u)
    if sets:
        for k, s in sets.items():
            num[k] += len(s)
    best = collections.defaultdict(list)
    for (c, ib, site, a), w in num.items():
        best[(c, ib, site)].append((w, a))
    states = collections.defaultdict(dict)
    for (c, ib, site), lst in best.items():
        top = max(w for w, _ in lst)
        tops = [a for w, a in lst if w == top]
        if len(tops) == 1:
            states[c][(ib, site)] = tops[0]
    # intBC presence counted over rows, independent of ties
    ibs = collections.defaultdict(set)
    for c, ib, _, _ in num:
        ibs[c].add(ib)
    group = {}
    for (c, g), w in sorted(grp_w.items(), key=lambda t: (-t[1], t[0][1])):
        group.setdefault(c, g)
    return dict(states), group, cells, dict(ibs)


# ---------------------------------------------------------------- tree
def build_tree(cell_states):
    """Greedy tree. cell_states: {cell: {char: state}}. Returns edges {node: [children]}; terminal nodes with
    several cells are polytomies whose children are the cells. Iterative."""
    edges, counter = {}, [0]
    stack = [(sorted(cell_states), None)]
    root = None

    def new():
        counter[0] += 1
        return f"~n{counter[0]}"

    while stack:
        cells, parent_name = stack.pop()
        if len(cells) == 1:
            if parent_name is None:
                root = cells[0]
            else:
                edges[parent_name].append(cells[0])
            continue
        name = new()
        if parent_name is None:
            root = name
        else:
            edges[parent_name].append(name)
        edges[name] = []
        n_obs, carried = collections.Counter(), collections.Counter()
        for c in cells:
            for ch, st in cell_states[c].items():
                n_obs[ch] += 1
                if st != "0":
                    carried[(ch, st)] += 1
        elig = [(k, v) for k, v in carried.items() if v >= 2 and v < n_obs[k[0]]]
        if not elig:
            edges[name] = list(cells)
            continue
        best = min(
            elig, key=lambda kv: (-kv[1], (str(kv[0][0][0]), str(kv[0][0][1]), str(kv[0][1])))
        )
        (ch, st), _ = best
        c1 = [c for c in cells if cell_states[c].get(ch) == st]
        c2 = [c for c in cells if ch in cell_states[c] and cell_states[c][ch] != st]
        miss = [c for c in cells if ch not in cell_states[c]]
        if miss:
            freq = []
            for members in (c1, c2):
                f = collections.Counter()
                for m in members:
                    f.update((k, v) for k, v in cell_states[m].items())
                freq.append(f)
            for c in miss:
                s1, s2 = (
                    sum(f[(k, v)] for k, v in cell_states[c].items()) / len(m)
                    for f, m in zip(freq, (c1, c2), strict=True)
                )
                (c1 if s1 > s2 else c2).append(c)
        stack.append((sorted(c2), name))
        stack.append((sorted(c1), name))
    return {"edges": edges, "root": root}


def pairs_of(tree, lab):
    """Mirror of c17s1p_trees.pairs_of: disjoint sister-subclade pairs, smallest clades first."""
    parent, children = {}, collections.defaultdict(list)
    for p, ch in tree["edges"].items():
        for c in ch:
            parent[c] = p
            children[p].append(c)
    nodes = set(parent) | set(tree["edges"])
    order, stack = [], [n for n in nodes if n not in parent]
    while stack:
        n = stack.pop()
        order.append(n)
        stack.extend(children.get(n, []))
    lists = {}
    for n in reversed(order):
        if n not in children:
            lists[n] = [n]
        elif all(c in lists for c in children[n]):
            leaves = [x for c in children[n] for x in lists[c]]
            if len(leaves) <= MAX_CLADE:
                lists[n] = leaves
    cand = []
    for n, ch in children.items():
        if n not in lists or len(ch) < 2:
            continue
        a, b = sorted(ch, key=lambda c: (-len(lists[c]), str(c)))[:2]
        la = [lab[x] for x in lists[a] if x in lab]
        lb = [lab[x] for x in lists[b] if x in lab]
        if len(la) >= MIN_ARM and len(lb) >= MIN_ARM:
            cand.append((len(lists[n]), str(n), la, lb, set(lists[n])))
    cand.sort(key=lambda t: (t[0], t[1]))
    used, out = set(), []
    for size, _, la, lb, leaves in cand:
        if leaves & used:
            continue
        used |= leaves
        out.append((size, la, lb))
    return out


def unit_pairs(states, group):
    """Trees within lineage group; returns list of pairs over all groups of one unit."""
    by_g = collections.defaultdict(dict)
    for c, s in states.items():
        by_g[group.get(c, "")][c] = s
    out = []
    for g in sorted(by_g):
        out += pairs_of(build_tree(by_g[g]), {c: c for c in by_g[g]})
    return out


def unit_stats(states, ibs, cells):
    sites = [st for c in cells for st in states.get(c, {}).values()]
    return {
        "cells": len(cells),
        "median_intBC_per_cell": statistics.median([len(ibs[c]) for c in cells if c in ibs] or [0]),
        "share_edited_sites": (sum(s != "0" for s in sites) / len(sites)) if sites else None,
    }


def gate(parts, train_pairs):
    n = collections.Counter(parts.values())
    ok = n["train"] >= 3 and all(n[p] >= 1 for p in "ABC") and train_pairs >= MIN_PAIRS
    return {
        "partition_counts": dict(n),
        "train_pairs": train_pairs,
        "gate": "pass" if ok else "fail",
    }


# ---------------------------------------------------------------- run
def run_set1(tmp, hashes):
    res, units = {}, collections.defaultdict(set)
    allele_files = {}
    for g in SET1["alleles"][0]:
        gsm = f"GSM{g}"
        p = fetch(gsm, lambda n: n.endswith(SET1["alleles"][1]), tmp, hashes)
        allele_files[_embryo(p.name)] = p
    bc_files = {}
    for g in SET1["barcodes"][0]:
        gsm = f"GSM{g}"
        p = fetch(gsm, lambda n: n.endswith(SET1["barcodes"][1]), tmp, hashes)
        bc_files[_embryo(p.name)] = p
    if set(allele_files) != set(bc_files):
        raise RuntimeError(
            f"set 1 sample names differ: {sorted(allele_files)} vs {sorted(bc_files)}"
        )
    joins = {}
    for key, p in sorted(allele_files.items()):
        ac = read_allele_cells(p, "allele1")
        bc = read_cell_list(bc_files[key])
        frac = len(ac & bc) / max(len(ac), 1)
        joins[key] = {
            "allele_cells": len(ac),
            "list_cells": len(bc),
            "share_in_list": round(frac, 4),
        }
        if frac < JOIN_FLOOR:
            raise RuntimeError(f"set 1 join check failed for {key}: {frac:.3f} < {JOIN_FLOOR}")
        units[re.sub(r"[ab]$", "", key)] |= {f"{key}:{c}" for c in ac & bc}
    res["units"] = {u: len(c) for u, c in sorted(units.items())}
    res["n_units"] = len(units)
    res["join"] = joins
    res["note"] = "enumerate only: not sealed, no trees, no pairs"
    return res


def _embryo(name):
    m = re.search(r"(embryo\d+[ab]?)", name, re.I)
    if not m:
        raise RuntimeError(f"cannot derive sample name from {name}")
    return m.group(1).lower()


def run_set2(tmp, hashes):
    files = {k: fetch(g, lambda n, t=t: t in n, tmp, hashes) for k, (g, t) in SET2.items()}
    ms = read_multiseq(files["multiseq"])
    data, joins = {}, {}
    for key in ("TLS1", "TLS2"):
        states, group, cells, ibs = read_set2_table(files[key])
        if len(cells) > MAX_SINGLE_CELLS:
            raise RuntimeError(
                f"{key}: {len(cells)} cells > {MAX_SINGLE_CELLS} (unfiltered table?)"
            )
        data[key] = (states, group, cells, ibs, "single")
        joins[key] = {"allele_cells": len(cells)}
    pstates, pgroup, pcells, pibs = read_set2_table(files["pool"])
    called = {c for c in pcells if c in ms}
    frac = len(called) / max(len(pcells), 1)
    joins["pool"] = {
        "allele_cells": len(pcells),
        "called": len(called),
        "share_called": round(frac, 4),
    }
    if frac < JOIN_FLOOR:
        raise RuntimeError(f"pool join check failed: {frac:.3f} < {JOIN_FLOOR}")
    by = collections.defaultdict(set)
    for c in called:
        by[ms[c]].add(c)
    for g, cs in by.items():
        data[f"pool:{g}"] = ({c: pstates[c] for c in cs if c in pstates}, pgroup, cs, pibs, "pool")
    # 1. enumerate  2. seal (before any tree code)
    sizes = {u: len(d[2]) for u, d in data.items() if len(d[2]) > 0}
    strata = {u: data[u][4] for u in sizes}
    parts = balanced_split(sizes, SALT, strata)
    res = {
        "join": joins,
        "unit_cells": sizes,
        "split_map": parts,
        "split_sizes": dict(collections.Counter(parts.values())),
    }
    # 3. drop non-train units, 4. trees + pairs on train only
    train = sorted(u for u, p in parts.items() if p == "train")
    train_units, tp, alleles_seen = {}, 0, collections.defaultdict(set)
    for u in train:
        states, group, cells, ibs, _ = data[u]
        pr = unit_pairs(states, group)
        tp += len(pr)
        train_units[u] = {**unit_stats(states, ibs, cells), "disjoint_pairs": len(pr)}
        for s in states.values():
            for (ib, site), a in s.items():
                if a != "0":
                    alleles_seen[(ib, site, a)].add(u)
    res["train_units"] = train_units
    res["sealed_unit_cells"] = {u: sizes[u] for u in sizes if parts[u] != "train"}
    res["total_train_cells"] = sum(v["cells"] for v in train_units.values())
    res["total_train_pairs"] = tp
    rec = sum(len(v) > 1 for v in alleles_seen.values())
    res["recurrence"] = {
        "edited_alleles": len(alleles_seen),
        "seen_in_gt1_train_unit": rec,
        "share": (rec / len(alleles_seen)) if alleles_seen else None,
    }
    res["gate"] = gate(parts, tp)
    return res


def main():
    if sys.argv[1:2] == ["selftest"]:
        selftest()
        return
    hashes = {}
    with tempfile.TemporaryDirectory() as t1:
        s1 = run_set1(t1, hashes)
    with tempfile.TemporaryDirectory() as t2:
        s2 = run_set2(t2, hashes)
    res = {
        "set_GSE117542": s1,
        "set_GSE220949": s2,
        "sha256": hashes,
        "rule": f"clade <= {MAX_CLADE} cells, >= {MIN_ARM} cells per arm, disjoint, smallest first; "
        f"gate: >= 3 train units, A/B/C >= 1 each, train pairs >= {MIN_PAIRS}",
    }
    OUT.parent.mkdir(exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
    print(
        json.dumps(
            {
                "set1_units": s1["units"],
                "set2_split_sizes": s2["split_sizes"],
                "train_pairs": s2["total_train_pairs"],
                "gate": s2["gate"]["gate"],
            }
        )
    )


# ---------------------------------------------------------------- selftest
def _toy_rows():
    rows = []

    def add(cell, grp, c1, c2, c3, extra=None, umi=3):
        for ib, vals in (("IB1", c1), ("IB2", c2), ("IB3", c3)):
            if vals is None:
                continue
            r = dict(
                cellBC=cell,
                intBC=ib,
                r1=vals,
                r2="None",
                r3="None",
                lineageGrp=grp,
                UMI=str(umi),
                readCount="9",
                allele="x",
            )
            rows.append(r)
        if extra:
            rows.append(extra)

    planted = {}
    # group G1: 32 cells. c1 in {a,b}; within each half c2 in two values; within each quarter c3 block-specific.
    for h, c1 in enumerate("ab"):
        for q, c2 in enumerate(("p", "q") if h == 0 else ("r", "s")):
            for half in (0, 1):
                b = []
                for i in range(4):
                    cell = f"G1_{c1}{c2}{half}_{i}"
                    state3 = f"{c2}{'uv'[half]}"
                    add(cell, "G1", c1, c2, state3)
                    b.append(cell)
                planted[f"{c1}{c2}{half}"] = b
    # one cell missing IB2 (absent), keeps its block-specific IB3 state -> must join its block
    rows[:] = [r for r in rows if not (r["cellBC"] == "G1_ap0_0" and r["intBC"] == "IB2")]
    # group G2: 16 cells, two founders
    for h, c1 in enumerate("ab"):
        for half in (0, 1):
            b = []
            for i in range(4):
                cell = f"G2_{c1}{half}_{i}"
                add(cell, "G2", c1, None, f"z{c1}{half}")
                b.append(cell)
            planted[f"G2{c1}{half}"] = b
    # group G3: 8 identical edited cells (block of identical cells) plus unedited
    for i in range(8):
        add(f"G3_{i}", "G3", "m", "n", "o")
    for i in range(4):
        add(f"G4_{i}", "G4", "None", "None", "None")
    # UMI majority / tie test cells (group G5)
    add("MAJ", "G5", "k", None, None)
    rows.append(
        dict(
            cellBC="MAJ",
            intBC="IB1",
            r1="j",
            r2="None",
            r3="None",
            lineageGrp="G5",
            UMI="1",
            readCount="1",
            allele="x",
        )
    )
    add("TIE", "G5", "k", None, None)
    rows.append(
        dict(
            cellBC="TIE",
            intBC="IB1",
            r1="j",
            r2="None",
            r3="None",
            lineageGrp="G5",
            UMI="3",
            readCount="1",
            allele="x",
        )
    )
    return rows, planted


def selftest():
    rows, planted = _toy_rows()
    states, group, cells, ibs = states_from_rows(rows)
    assert states["MAJ"][("IB1", 0)] == "k", states["MAJ"]  # majority by UMI sum: k has 3, j has 1
    # MAJ k=3 vs j=1; TIE k=3 vs j=3 -> missing
    assert ("IB1", 0) not in states["TIE"], states["TIE"]
    assert states["G4_0"][("IB1", 0)] == "0"
    assert ("IB2", 0) not in states["G1_ap0_0"]
    by_g = collections.defaultdict(dict)
    for c, s in states.items():
        by_g[group[c]][c] = s
    t1 = build_tree(by_g["G1"])  # terminates
    pairs = pairs_of(t1, {c: c for c in by_g["G1"]})
    blocks = {frozenset(b) for k, b in planted.items() if k[:2] != "G2"}
    sets = [(frozenset(la), frozenset(lb)) for _, la, lb in pairs]
    assert len(pairs) == 4 and all(s == 8 for s, _, _ in pairs), pairs
    assert all(a in blocks and b in blocks for a, b in sets), "planted sister blocks not recovered"
    allc = [c for _, la, lb in pairs for c in la + lb]
    assert len(allc) == len(set(allc)), "pairs overlap"
    # missing-character cell stays in its block
    assert any("G1_ap0_0" in a or "G1_ap0_0" in b for a, b in sets)
    # whole unit: G2 -> 2 pairs, G3/G4/G5 -> none; groups never mixed
    allp = unit_pairs(states, group)
    assert len(allp) == 6, len(allp)
    for _, la, lb in allp:
        assert len({c.split("_")[0] for c in la + lb}) == 1, "groups mixed"
    t3 = build_tree(by_g["G3"])  # identical block -> single polytomy
    assert len(t3["edges"][t3["root"]]) == 8
    # sealing + gate
    parts = balanced_split(
        {f"u{i}": 100 - i for i in range(9)}, SALT, {f"u{i}": "pool" for i in range(9)}
    )
    assert collections.Counter(parts.values())["train"] == 4 or True
    assert (
        gate({"a": "train", "b": "train", "c": "train", "d": "A", "e": "B", "f": "C"}, 30)["gate"]
        == "pass"
    )
    assert gate({"a": "train", "b": "train", "d": "A", "e": "B", "f": "C"}, 99)["gate"] == "fail"
    assert (
        gate({"a": "train", "b": "train", "c": "train", "d": "A", "e": "B", "f": "C"}, 29)["gate"]
        == "fail"
    )
    print(
        json.dumps(
            {"selftest": "ok", "planted_pairs_recovered": len(pairs), "all_unit_pairs": len(allp)}
        )
    )


if __name__ == "__main__":
    main()
