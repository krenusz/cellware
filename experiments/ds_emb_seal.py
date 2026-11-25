"""DS-EMB-S: seal E5 (mouse gastrulation manual tracks, S-BIAD1682) by track tree (pre-registered 2026-11-25).

Reads spot ID, FRAME and edges only (never intensity, position or radius). Unit = connected component of the spot
graph, named 'M<movie>:<earliest spot ID>'. Size = eligible depth >= 1 sister pairs (DS-EMB-G rule: a division with
an earlier division on its path, each daughter branch >= 5 spots). Movie5 duplicates Movie3 and is excluded.

    python experiments/ds_emb_seal.py selftest
    python experiments/ds_emb_seal.py seal      # writes partition_maps/partition_mouse_gast_tree.tsv + counts
"""

import json
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "data", "ext_mouse_gast_tracks", "raw")
MOVIES = [1, 2, 3, 4]  # Movie5 = same tracking as Movie3, excluded (DS-EMB-G result)
SALT = "cellware-mouse-gast-tree-v1"
MAP = os.path.join(HERE, "partition_maps", "partition_mouse_gast_tree.tsv")
OUT = os.path.join(HERE, "results_c17", "ds_emb_seal_counts.json")
MIN_BRANCH = 5


def load(path):
    frame, edges = {}, []
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag == "Spot":
            frame[el.get("ID")] = int(float(el.get("FRAME")))
            el.clear()
        elif el.tag == "Edge":
            edges.append((el.get("SPOT_SOURCE_ID"), el.get("SPOT_TARGET_ID")))
            el.clear()
    return frame, edges


def trees(frame, edges, movie):
    """Return {unit: eligible pairs} over connected components."""
    kids, adj = defaultdict(list), defaultdict(set)
    has_parent = set()
    for a, b in edges:
        if a not in frame or b not in frame:
            continue
        s, t = (a, b) if frame[a] < frame[b] else (b, a)
        kids[s].append(t)
        has_parent.add(t)
        adj[a].add(b)
        adj[b].add(a)

    def branch_len(n):
        c = 0
        while True:
            c += 1
            if len(kids[n]) != 1:
                return c
            n = kids[n][0]

    # eligible divisions: walk forward from every root, counting divisions above
    eligible = set()
    for r in (s for s in frame if s not in has_parent):
        stack = [(r, 0)]
        while stack:
            n, above = stack.pop()
            ch = kids[n]
            is_div = len(ch) == 2
            if is_div and above >= 1 and all(branch_len(c) >= MIN_BRANCH for c in ch):
                eligible.add(n)
            for c in ch:
                stack.append((c, above + is_div))
    seen, out = set(), {}
    for s in frame:
        if s in seen:
            continue
        comp, stack = [], [s]
        seen.add(s)
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        root = min(comp, key=lambda x: (frame[x], int(x)))
        out[f"M{movie}:{root}"] = sum(x in eligible for x in comp)
    return out


def selftest():
    # two roots merging into one component (as Movie2), plus a separate tree with a depth-1 division
    frame = {str(i): f for i, f in enumerate([0, 0, 1, 2, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])}
    edges = [("0", "2"), ("1", "2"), ("2", "3"),  # merge component, rooted at 0
             ("4", "5"), ("5", "6"), ("5", "7"),  # root division at 5 (depth 0)
             ("6", "8"), ("8", "9"), ("8", "10")]  # depth-1 division at 8, short branches
    t = trees(frame, edges, 9)
    assert set(t) == {"M9:0", "M9:4"} | {f"M9:{i}" for i in range(11, 18)}, t  # merge -> one unit named by spot 0
    assert t["M9:0"] == 0 and t["M9:4"] == 0, t
    # long branches make the depth-1 division eligible
    frame2 = {str(i): i for i in range(30)}
    e2 = [("0", "1"), ("1", "2"), ("1", "3")]  # root division at 1
    chain = lambda a, ids: list(zip([a] + ids[:-1], ids))  # noqa: E731
    e2 += chain("2", ["4"]) + [("4", "5"), ("4", "6")]  # depth-1 division at 4
    e2 += chain("5", ["7", "8", "9", "10"]) + chain("6", ["11", "12", "13", "14"])
    e2 += chain("3", ["15"])
    t2 = trees(frame2, e2, 7)
    assert t2["M7:0"] == 1, t2
    return {"ok": True}


def seal():
    sys.path.insert(0, HERE)
    import partitions  # noqa: PLC0415

    sizes, strata = {}, {}
    for m in MOVIES:
        t = trees(*load(os.path.join(RAW, f"Movie{m}-mamut.xml")), m)
        sizes.update(t)
        strata.update({u: f"M{m}" for u in t})
    mp = partitions.balanced_split(sizes, SALT, strata)
    os.makedirs(os.path.dirname(MAP), exist_ok=True)
    with open(MAP, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# DS-EMB-S 2026-11-25: E5 track trees (connected components), balanced_split salt {SALT}, "
                 "strata movie, size eligible pairs\n")
        for u in sorted(mp):
            fh.write(f"{u}\t{mp[u]}\n")
    counts = {p: {"trees": sum(v == p for v in mp.values()), "eligible": sum(sizes[u] for u in mp if mp[u] == p)}
              for p in ("train", "A", "B", "C")}
    by_movie = {f"M{m}": dict(Counter(mp[u] for u in mp if strata[u] == f"M{m}")) for m in MOVIES}
    res = {"units": len(mp), "eligible_total": sum(sizes.values()), "by_partition": counts, "trees_by_movie": by_movie}
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    return res


if __name__ == "__main__":
    arg = sys.argv[1:] or ["selftest"]
    print(json.dumps(selftest() if arg[0] == "selftest" else seal(), indent=1))
