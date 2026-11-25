"""DS-EMB-G: S0 structure gates for embryo leads (structure only, no counts of clones or pairs yet).

Bence 2026-11-25: "2" = gate all six. Pre-registered in EXPERIMENT_LOG (DS-EMB-G).
Lineage tables only (data/ext_zf_scartrace, data/ext_zf_linnaeus); read with -I.
Columns whose names suggest cell type, cluster, tissue, organ or fate are listed by name only.

    python -I experiments/ds_emb_gate.py schema | selftest | gate
"""

from __future__ import annotations

import gzip
import io
import json
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUT = HERE / "results_c17"
HELD = re.compile(r"type|fate|cluster|tissue|organ|annot|class|ident|state|leiden|louvain|region|lineage|germ|layer|tsne|umap|clust|location|loc$|bin$",
                  re.I)
EXAMPLE_OK = re.compile(r"barcode|cell|clone|scar|allele|cigar|read|count|umi|fish|larva|animal|embryo|sample|^id$|^unnamed",
                        re.I)
DIRS = ("ext_zf_scartrace", "ext_zf_linnaeus")


def read_table(path):
    name = path.name
    sep = "," if ".csv" in name else "\t"
    raw = gzip.decompress(path.read_bytes())
    return pd.read_csv(io.BytesIO(raw), sep=sep, low_memory=False)


def mask(v):
    """Cell IDs in the scar-clone tables embed free-text tokens; keep only the shape (letters a, digits 9)."""
    return re.sub(r"[A-Za-z]+", "a", re.sub(r"\d+", "9", v))


def cols(df, mask_ids=False):
    out, held = [], []
    for c in df.columns:
        s = df[c]
        e = {"name": str(c), "dtype": str(s.dtype)}
        if HELD.search(str(c)):
            held.append(str(c))
            e["held"] = "name only"
        elif EXAMPLE_OK.search(str(c)) and s.notna().any():
            e["examples"] = [(mask(str(v)) if mask_ids else str(v))[:60] for v in s.dropna().iloc[:3]]
        out.append(e)
    return {"rows": int(len(df)), "columns": out, "held_columns": held}


def schema():
    res = {}
    for d in DIRS:
        for f in sorted((DATA / d / "raw").glob("*.gz")):
            res[f"{d}/{f.name}"] = cols(read_table(f), mask_ids=(d == "ext_zf_scartrace"))
    (OUT / "ds_emb_gate_schema.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


# --- gate counts (amendment 1 of DS-EMB-G; counts only, no IDs, positions or intensities) --------------------------

import collections  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402

sys.path.insert(0, str(HERE))
from c17s1g_gates import joined_edits, sister_pairs, summarise  # noqa: E402

MIN_TRACK = 5
PURE_MATCH = re.compile(r"^\d+M$")
E2_HELD = {"hclust", "oclust", "organ"}
E4_ANIMALS = ("A5", "A6", "A7", "Z2", "Z4", "Z5")


def tree_e2():
    pairs, seen, cells, alleles = {}, collections.defaultdict(set), {}, {}
    for f in sorted((DATA / "ext_zf_scartrace" / "raw").glob("*_scarclones.txt.gz")):
        fish = f.name.split("_")[1]
        df = read_table(f)
        idc = df.columns[0]
        scar = [c for c in df.columns[1:] if c not in E2_HELD and not PURE_MATCH.match(str(c))]
        m = df[scar].fillna(0).to_numpy() > 0
        ca = {}
        for cell, row in zip(df[idc].astype(str), m, strict=True):
            if row.any():
                ca[cell] = "|".join(sorted(c for c, k in zip(scar, row, strict=True) if k))
        pairs[fish] = sister_pairs(ca, edits=joined_edits)
        cells[fish], alleles[fish] = len(ca), len(set(ca.values()))
        for a in set(ca.values()):
            seen[a].add(fish)
    return pairs, seen, cells, alleles


def tree_e4():
    pairs, seen, cells, alleles = {}, collections.defaultdict(set), {}, {}
    for f in sorted((DATA / "ext_zf_linnaeus" / "raw").glob("*_scars_compared.csv.gz")):
        larva = f.name.split("_")[1]
        if larva not in E4_ANIMALS:
            continue
        df = read_table(f)
        df = df[df["CIGAR"].astype(str).str.contains("[ID]", regex=True)]
        idc = "Cell" if "Cell" in df.columns else "Barcode"
        ca = {c: "|".join(sorted(set(g.astype(str)))) for c, g in df.groupby(df[idc].astype(str))["Scar"]}
        pairs[larva] = sister_pairs(ca, edits=joined_edits)
        cells[larva], alleles[larva] = len(ca), len(set(ca.values()))
        for a in set(ca.values()):
            seen[a].add(larva)
    return pairs, seen, cells, alleles


def tree_lead(fn):
    pairs, seen, cells, alleles = fn()
    rec = sum(len(v) > 1 for v in seen.values()) / max(len(seen), 1)
    out = summarise(pairs, round(rec, 3))
    out["cells_with_edited_allele"], out["distinct_full_alleles"] = cells, alleles
    return out


def track_gate(length, parent, leaf_ok, min_len=MIN_TRACK):
    """length/parent/leaf_ok: dict track -> time points / parent track (or None) / last node positioned.
    Division = track with exactly 2 child tracks. Eligible: mother has a parent track, each daughter spans >= min_len
    time points, all end nodes in both arms (daughter plus descendants) positioned."""
    kids = collections.defaultdict(list)
    for t, p in parent.items():
        if p is not None and p in length and p != t:
            kids[p].append(t)
    order = []
    stack = [t for t in length if parent.get(t) is None or parent[t] not in length or parent[t] == t]
    while stack:
        t = stack.pop()
        order.append(t)
        stack.extend(kids.get(t, ()))
    arm_ok = {}
    for t in reversed(order):
        ks = kids.get(t, ())
        arm_ok[t] = leaf_ok[t] if not ks else all(arm_ok[k] for k in ks)
    divs = [t for t in length if len(kids.get(t, ())) == 2]
    elig = 0
    for m in divs:
        d1, d2 = kids[m]
        has_parent = parent.get(m) in length and parent[m] != m
        if has_parent and length[d1] >= min_len and length[d2] >= min_len and arm_ok[d1] and arm_ok[d2]:
            elig += 1
    return {"tracks": len(length), "tracks_reached_from_roots": len(order), "divisions": len(divs), "eligible": elig,
            "parents_with_gt2_children": sum(len(v) > 2 for v in kids.values())}


def spot_gate(frame, edges, min_arm=MIN_TRACK):
    """frame: dict spot -> frame; edges: [(a, b)], oriented forward in frame here (same-frame edges ignored).
    Division = spot with 2 forward edges. Eligible: >= 1 earlier division on the path back to the root, and each
    daughter arm >= min_arm spots (primary: arm = daughter plus all descendants; also reported: branch only,
    the unbranched chain from the daughter)."""
    out, indeg, same = collections.defaultdict(list), collections.Counter(), 0
    for a, b in edges:
        if a not in frame or b not in frame:
            continue
        if frame[a] > frame[b]:
            a, b = b, a
        if frame[a] == frame[b]:
            same += 1
            continue
        out[a].append(b)
        indeg[b] += 1
    nodes = sorted(frame, key=lambda s: frame[s])
    dd = collections.defaultdict(int)
    for n in nodes:
        for c in out.get(n, ()):
            dd[c] = max(dd[c], dd[n] + (1 if len(out[n]) == 2 else 0))
    sub, branch = {}, {}
    for n in reversed(nodes):
        ks = out.get(n, ())
        sub[n] = 1 + sum(sub[k] for k in ks)
        branch[n] = 1 + (branch[ks[0]] if len(ks) == 1 else 0)
    divs = [n for n in nodes if len(out.get(n, ())) == 2]
    e_sub = sum(1 for n in divs if dd[n] >= 1 and all(sub[k] >= min_arm for k in out[n]))
    e_br = sum(1 for n in divs if dd[n] >= 1 and all(branch[k] >= min_arm for k in out[n]))
    return {"spots": len(frame), "spots_with_edges": len({s for e in edges for s in e if s in frame}),
            "divisions": len(divs), "eligible": e_sub, "eligible_branch_only": e_br,
            "same_frame_edges_ignored": same, "spots_indegree_gt1": sum(v > 1 for v in indeg.values()),
            "spots_outdegree_gt2": sum(len(v) > 2 for v in out.values())}


def e3():
    res = {}
    for f in sorted((DATA / "ext_zebrahub_tracks" / "raw").glob("ZSNS*_tracks.csv")):
        df = pd.read_csv(f, usecols=["track_id", "t", "x", "y", "z", "parent_track_id"])
        df = df.sort_values(["track_id", "t"])
        g = df.groupby("track_id")
        length = g.size().to_dict()
        par = g["parent_track_id"].first()
        parent = {t: (None if pd.isna(p) or p < 0 else int(p)) for t, p in par.items()}
        last = g[["x", "y", "z"]].last().notna().all(axis=1).to_dict()
        r = track_gate(length, parent, last)
        r["rows"] = int(len(df))
        r["max_distinct_parent_per_track"] = int(g["parent_track_id"].nunique().max())
        r["share_tracks_without_parent_track"] = round(sum(p is None or p not in length for p in parent.values())
                                                       / len(parent), 3)
        res[f.name.split("_")[0]] = r
    return res


def e5():
    res = {}
    for f in sorted((DATA / "ext_mouse_gast_tracks" / "raw").glob("Movie*-mamut.xml")):
        frame, edges = {}, []
        for _, el in ET.iterparse(f, events=("end",)):
            if el.tag == "Spot":
                frame[el.get("ID")] = int(el.get("FRAME"))
                el.clear()
            elif el.tag == "Edge":
                edges.append((el.get("SPOT_SOURCE_ID"), el.get("SPOT_TARGET_ID")))
                el.clear()
        res[f.name.split("-")[0]] = spot_gate(frame, edges)
    return res


def finish(per):
    tot = sum(v["eligible"] for v in per.values())
    return {"per_unit": per, "eligible_total": tot, "expected_train_eligible": 0.4 * tot, "gate_pass": bool(0.4 * tot >= 30)}


def gate():
    out = {"E2": tree_lead(tree_e2), "E4": tree_lead(tree_e4), "E3": finish(e3()), "E5": finish(e5())}
    (OUT / "ds_emb_gate_counts.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def selftest():
    a1, a2, a3 = "F|A|B", "F|A|C", "F|A|B|D"
    a4, a5, a6 = "F|E|G", "F|E|H", "F|H|G"
    ca, k = {}, 0
    for a, n in ((a1, 5), (a2, 5), (a3, 4), ("F|Z", 20), (a4, 4), (a5, 4), (a6, 3)):
        for _ in range(n):
            ca[f"c{k}"] = a
            k += 1
    s = summarise({"x": sister_pairs(ca, edits=joined_edits)})
    assert s["pairs_taken"] == {"x": 2} and s["eligible"] == 1, s
    # tracks: R root division (no parent), A -> C + short D (depth 1, short), B -> E + F (depth 1, valid)
    length = {"R": 6, "A": 6, "B": 6, "C": 6, "D": 3, "E": 6, "F": 6}
    parent = {"R": None, "A": "R", "B": "R", "C": "A", "D": "A", "E": "B", "F": "B"}
    r = track_gate(length, parent, dict.fromkeys(length, True))
    assert (r["divisions"], r["eligible"]) == (3, 1), r
    frame, edges = {}, []

    def chain(name, start, n, prev):
        for i in range(n):
            frame[f"{name}{i}"] = start + i
            if i:
                edges.append((f"{name}{i - 1}", f"{name}{i}"))
        if prev:
            edges.append((prev, f"{name}0"))
        return f"{name}{n - 1}"

    r_end = chain("R", 0, 6, None)
    a_end = chain("A", 6, 6, r_end)
    b_end = chain("B", 6, 6, r_end)
    chain("C", 12, 6, a_end)
    chain("D", 12, 3, a_end)
    chain("E", 12, 6, b_end)
    chain("F", 12, 6, b_end)
    edges = [(b, a) if i % 2 else (a, b) for i, (a, b) in enumerate(edges)]  # orientation must not matter
    r = spot_gate(frame, edges)
    assert (r["divisions"], r["eligible"]) == (3, 1), r
    print("selftest OK")


if __name__ == "__main__":
    arg = sys.argv[1:]
    if arg == ["schema"]:
        for k, v in schema().items():
            print(k, v["rows"], len(v["columns"]), "HELD:", v["held_columns"])
    elif arg == ["selftest"]:
        selftest()
    elif arg == ["gate"]:
        print(json.dumps(gate(), indent=1))
    else:
        sys.exit(__doc__)
