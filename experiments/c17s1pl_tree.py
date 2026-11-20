"""C17-S1PL: dated whole-embryo tree (DNA Typewriter mouse, one E13.5 embryo), S0 structure gate and sealing.

Bence 2026-11-20: "1". Serves: C17-S1PL. Pre-registered in EXPERIMENT_LOG (C17-S1PL). Structure only: tree topology,
branch lengths, leaf IDs and the metadata ID join. No cell-type column is read or tabulated before sealing.

    python -I experiments/c17s1pl_tree.py selftest   # offline: Newick parser + count-based picker vs c17s1t_stage
    python -I experiments/c17s1pl_tree.py fetch      # download by release tag, sha256 check, SHA256SUMS
    python -I experiments/c17s1pl_tree.py schema     # metadata column names and dtypes; examples for ID columns only
    python -I experiments/c17s1pl_tree.py gate       # G1-G4 -> results_c17/c17s1pl_gate.json (Actions)
    python -I experiments/c17s1pl_tree.py seal       # founder-clade balanced split -> partition_maps (after a pass)
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

RAW = HERE.parent / "data" / "ext_dtt_mouse" / "raw"
OUT = HERE / "results_c17" / "c17s1pl_gate2.json"  # amendment 2 (first gate: c17s1pl_gate.json)
MAP = HERE / "partition_maps" / "partition_dtt_founder.tsv"
TAG = "v1.1.0"
BASE = f"https://raw.githubusercontent.com/shendurelab/dtt-mouse/{TAG}/support_data/"
FILES = {  # name -> expected sha256 (filled from the first verified download, then committed)
    "merged_minB2h_lineage_constrained.nwk": "dcc12fcdc0da7defe52bb01deeec75d397b38cb6408a1f93df1b5d507feaaa79",
    "cell_metadata.v8.txt.gz": "208dc0a0bb52796c8b77d4d1c6859d4378fb9ef3b0a0598f2bef34010a794058",
}
TREE, META = "merged_minB2h_lineage_constrained.nwk", "cell_metadata.v8.txt.gz"
ID_COL = "cell_id"  # metadata column whose values are the tree leaf names: set by the mapping amendment after `schema`
CAP, CAP_C17S1P, MIN_ARM, FOUNDER_MAX = 3000, 300, 4, 20000
GATE = {"G1_min_pairs": 75, "G2_min_k_range": 2.4, "G3_min_founders": 50, "G4_max_abs_corr": 0.8}
SALT = "cellware-dtt-founder-v1"
PICK_SALT, N_BINS = "cellware-dtt-pick-v1", 5
ID_LIKE = re.compile(r"cell|barcode|\bid\b|_id|name|leaf|node", re.I)
TYPE_LIKE = re.compile(r"type|cluster|annot|lineage|fate|state|tissue|organ|germ|class|label|celltype|region|subtype", re.I)


# ---------------------------------------------------------------- tree
def parse_newick(text):
    """Iterative Newick parser. Returns parent (int array, -1 for root), branch length (float array, NaN if
    absent) and names (list; '' for unnamed). Nodes are created in preorder, so parent index < child index."""
    text = re.sub(r"\[[^\]]*\]", "", text.strip())
    parent, blen, names = [], [], []
    stack, prev, last_closed = [], None, None

    def new(p):
        parent.append(p)
        blen.append(np.nan)
        names.append("")
        return len(parent) - 1

    def set_label(n, tok):
        name, _, length = tok.rpartition(":") if ":" in tok else (tok, "", "")
        names[n] = name.strip().strip("'\"")
        if length.strip():
            blen[n] = float(length)

    for m in re.finditer(r"[(),;]|[^(),;]+", text):
        tok = m.group(0)
        if tok == "(":
            stack.append(new(stack[-1] if stack else -1))
        elif tok == ")":
            last_closed = stack.pop()
        elif tok in (",", ";"):
            pass
        elif tok.strip():
            if prev in ("(", ","):
                set_label(new(stack[-1]), tok)
            elif prev == ")":
                set_label(last_closed, tok)
        prev = tok if tok.strip() else prev
    return np.array(parent, dtype=np.int64), np.array(blen, dtype=float), names


def structure(parent, blen, labelled):
    """Per-node counts by one reverse pass (children have larger indices than parents)."""
    n = len(parent)
    nchild = np.bincount(parent[parent >= 0], minlength=n)
    is_leaf = nchild == 0
    depth = np.zeros(n, dtype=np.int64)
    rdist = np.zeros(n)
    bl = np.nan_to_num(blen, nan=0.0)
    for i in range(1, n):
        p = parent[i]
        if p >= 0:
            depth[i] = depth[p] + 1
            rdist[i] = rdist[p] + bl[i]
    leaves = is_leaf.astype(np.int64)
    lab = (is_leaf & labelled).astype(np.int64)
    sdep = np.where(lab > 0, depth, 0).astype(float)
    sdist = np.where(lab > 0, rdist, 0.0)
    for i in range(n - 1, 0, -1):
        p = parent[i]
        if p >= 0:
            leaves[p] += leaves[i]
            lab[p] += lab[i]
            sdep[p] += sdep[i]
            sdist[p] += sdist[i]
    return {"nchild": nchild, "is_leaf": is_leaf, "depth": depth, "rdist": rdist, "leaves": leaves, "lab": lab,
            "sdep": sdep, "sdist": sdist}


def children_of(parent):
    order = np.argsort(parent, kind="stable")
    ps = parent[order]
    starts = np.searchsorted(ps, np.arange(len(parent)))
    ends = np.searchsorted(ps, np.arange(len(parent)), side="right")
    return order, starts, ends


def pick_pairs(parent, s, cap, strat=False):
    """C17-S1P/C17-S1T rule with counts: for each node with <= cap leaves and >= 2 children, its two largest children
    (by leaves) as arms, both with >= MIN_ARM labelled leaves; disjoint, smallest clades first.
    Returns rows (node, size, tau, k_edges, t_since, lab_a, lab_b)."""
    order, starts, ends = children_of(parent)
    cand = []
    for nd in np.nonzero((s["nchild"] >= 2) & (s["leaves"] <= cap))[0]:
        ch = order[starts[nd]:ends[nd]]
        ch = ch[np.argsort(-s["leaves"][ch], kind="stable")][:2]
        a, b = ch
        if s["lab"][a] >= MIN_ARM and s["lab"][b] >= MIN_ARM:
            nl = s["lab"][a] + s["lab"][b]
            k = (s["sdep"][a] + s["sdep"][b]) / nl - s["depth"][nd]
            t = (s["sdist"][a] + s["sdist"][b]) / nl - s["rdist"][nd]
            cand.append((int(s["leaves"][nd]), int(nd), int(s["depth"][nd]), float(k), float(t),
                         int(s["lab"][a]), int(s["lab"][b])))
    if strat:  # amendment 2: 5 equal-count bins by node date, hash order within bin, round-robin across bins
        dates = np.array([s["rdist"][r[1]] for r in cand])
        bins = np.minimum((np.argsort(np.argsort(dates, kind="stable"), kind="stable") * N_BINS) // len(cand), N_BINS - 1)
        hkey = [hashlib.sha256(f"{PICK_SALT}:{r[1]}".encode()).hexdigest() for r in cand]
        rank = np.zeros(len(cand), dtype=np.int64)
        for b in range(N_BINS):
            ix = np.nonzero(bins == b)[0]
            rank[ix[np.argsort([hkey[i] for i in ix], kind="stable")]] = np.arange(len(ix))
        cand = [cand[i] for i in np.lexsort((bins, rank))]
    else:
        cand.sort(key=lambda r: (r[0], r[1]))
    used = np.zeros(len(parent), bool)
    desc_used = np.zeros(len(parent), bool)
    out = []
    for size, nd, tau, k, t, la, lb in cand:
        if desc_used[nd] or used[nd]:
            continue
        p, hit = parent[nd], False
        while p >= 0:
            if used[p]:
                hit = True
                break
            p = parent[p]
        if hit:
            continue
        used[nd] = True
        p = parent[nd]
        while p >= 0 and not desc_used[p]:
            desc_used[p] = True
            p = parent[p]
        out.append((nd, size, tau, k, t, la, lb))
    return out


def zygote_split(parent, s):
    """First node from the root with >= 2 children; its two largest children are the first-cleavage halves."""
    order, starts, ends = children_of(parent)
    nd = int(np.nonzero(parent < 0)[0][0])
    while s["nchild"][nd] == 1:
        nd = int(order[starts[nd]])
    ch = order[starts[nd]:ends[nd]]
    return nd, [int(c) for c in ch[np.argsort(-s["leaves"][ch], kind="stable")][:2]]


def founders(parent, s):
    """Maximal clades with <= FOUNDER_MAX leaves: node with leaves <= max whose parent has more (or is the root)."""
    ok = s["leaves"] <= FOUNDER_MAX
    par_big = np.ones(len(parent), bool)
    has_p = parent >= 0
    par_big[has_p] = s["leaves"][parent[has_p]] > FOUNDER_MAX
    return np.nonzero(ok & par_big)[0]


def ancestor_in(parent, nd, target):
    while nd >= 0 and not target[nd]:
        nd = parent[nd]
    return int(nd)


# ---------------------------------------------------------------- stages
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    sums = []
    for name, want in FILES.items():
        p = RAW / name
        if not p.exists():
            req = urllib.request.Request(BASE + name, headers={"User-Agent": "cellware-fetch/1.0"})
            with urllib.request.urlopen(req, timeout=600) as r, open(p, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
        got = sha256(p)
        if want and got != want:
            raise SystemExit(f"sha256 mismatch for {name}: {got} != {want}")
        sums.append(f"{got}  {name}")
        print(name, p.stat().st_size, got, "verified" if want else "UNPINNED (commit this hash)")
    (RAW / "SHA256SUMS").write_text("\n".join(sums) + "\n")


def schema():
    """Column names and dtypes; example values only for ID-like columns that are not type-like."""
    import pandas as pd

    head = pd.read_csv(RAW / META, sep="\t", nrows=2000)
    with gzip.open(RAW / META, "rt") as f:
        n_rows = sum(1 for _ in f) - 1
    cols = []
    for c in head.columns:
        ex = None
        if ID_LIKE.search(str(c)) and not TYPE_LIKE.search(str(c)):
            ex = [str(v) for v in head[c].head(3)]
        cols.append({"column": str(c), "dtype": str(head[c].dtype), "examples": ex})
    tree_head = (RAW / TREE).read_text()[:300]
    leaf_examples = re.findall(r"[(,]([^(),:;]+):", tree_head)[:3]
    res = {"n_rows": n_rows, "columns": cols, "tree_leaf_name_examples": leaf_examples}
    print(json.dumps(res, indent=1))
    return res


def load():
    import pandas as pd

    if not ID_COL:
        raise SystemExit("ID_COL not set: write the mapping amendment after `schema` first")
    parent, blen, names = parse_newick((RAW / TREE).read_text())
    ids = set(pd.read_csv(RAW / META, sep="\t", usecols=[ID_COL])[ID_COL].astype(str))
    labelled = np.array([nm in ids for nm in names])
    return parent, blen, names, labelled


def gate():
    parent, blen, names, labelled = load()
    s = structure(parent, blen, labelled)
    res = {"nodes": int(len(parent)), "leaves": int(s["is_leaf"].sum()), "labelled_leaves": int(labelled.sum()),
           "branch_lengths_present": float(np.mean(~np.isnan(blen[1:]))), "gate_rules": GATE}
    zs, halves = zygote_split(parent, s)
    res["zygote_split"] = {"node_depth": int(s["depth"][zs]), "half_leaves": [int(s["leaves"][h]) for h in halves]}
    fnd = founders(parent, s)
    is_f = np.zeros(len(parent), bool)
    is_f[fnd] = True
    res["founder_clades"] = int(len(fnd))
    for cap, strat, key in ((CAP, True, "stratified"), (CAP, False, "smallest_first")):
        rows = pick_pairs(parent, s, cap, strat)
        if not rows:
            res[key] = {"eligible_pairs": 0}
            continue
        _, size, tau, k, t, la, lb = (np.array(x) for x in zip(*rows))
        f_of = np.array([ancestor_in(parent, r[0], is_f) for r in rows])
        res[key] = {
            "eligible_pairs": int(len(rows)),
            "k_pct_10_50_90": np.percentile(k, [10, 50, 90]).round(2).tolist(),
            "k_range_10_90": float(np.percentile(k, 90) - np.percentile(k, 10)),
            "tau_pct_10_50_90": np.percentile(tau, [10, 50, 90]).round(2).tolist(),
            "t_since_pct_10_50_90": np.percentile(t, [10, 50, 90]).round(3).tolist(),
            "corr_tau_k": float(np.corrcoef(tau, k)[0, 1]) if len(rows) > 2 else None,
            "clade_size_pct_10_50_90": np.percentile(size, [10, 50, 90]).tolist(),
            "labelled_per_arm_median": float(np.median(np.r_[la, lb])),
            "founders_with_pairs": int(len(set(f_of[f_of >= 0]))),
        }
    p = res["stratified"]
    checks = {
        "G1": p["eligible_pairs"] >= GATE["G1_min_pairs"],
        "G2": p.get("k_range_10_90", 0) >= GATE["G2_min_k_range"],
        "G3": p.get("founders_with_pairs", 0) >= GATE["G3_min_founders"],
        "G4": p.get("corr_tau_k") is not None and abs(p["corr_tau_k"]) <= GATE["G4_max_abs_corr"],
    }
    res["checks"] = checks
    res["gate"] = "PASS" if all(checks.values()) else "FAIL"
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def seal():
    from partitions import balanced_split

    if json.loads(OUT.read_text()).get("gate") != "PASS":
        raise SystemExit("gate has not passed: no sealing")
    parent, blen, names, labelled = load()
    s = structure(parent, blen, labelled)
    _, halves = zygote_split(parent, s)
    fnd = founders(parent, s)
    half_of = {}
    for h_i, h in enumerate(halves):
        for f in fnd:
            if ancestor_in(parent, int(f), np.isin(np.arange(len(parent)), [h])) == h:
                half_of[int(f)] = f"half{h_i + 1}"
    unit = {int(f): f"F{int(f)}" for f in fnd}
    sizes = {unit[f]: int(s["lab"][f]) for f in fnd}
    strata = {unit[f]: half_of.get(f, "other") for f in fnd}
    part = balanced_split(sizes, SALT, strata)
    MAP.parent.mkdir(exist_ok=True)
    lines = ["founder\troot_node\tpartition"] + [f"{unit[f]}\t{f}\t{part[unit[f]]}" for f in fnd]
    MAP.write_text("\n".join(lines) + "\n")
    print({p: sum(v == p for v in part.values()) for p in ("train", "A", "B", "C")})


# ---------------------------------------------------------------- selftest
def selftest():
    import c17s1t_stage as t

    rng = np.random.default_rng(20261120)
    t.MIN_ARM = MIN_ARM
    for rep in range(30):
        n_leaves = int(rng.integers(20, 400))
        # random binary tree by successive merges; names L<i>, internal I<j>; random branch lengths
        pool = [(f"L{i}", f"L{i}:{rng.uniform(0.1, 2):.3f}") for i in range(n_leaves)]
        edges, j = {}, 0
        while len(pool) > 1:
            a, b = (pool.pop(int(rng.integers(len(pool)))) for _ in range(2))
            nm = f"I{j}"
            j += 1
            edges[nm] = [a[0], b[0]]
            pool.append((nm, f"({a[1]},{b[1]}){nm}:{rng.uniform(0.1, 2):.3f}"))
        newick = pool[0][1].rsplit(":", 1)[0] + ";"
        lab = {f"L{i}": "x" for i in range(n_leaves) if rng.random() < 0.8}
        cap = int(rng.integers(20, 300))
        t.MAX_CLADE = cap
        ref = sorted((sz, tau, round(k, 6), len(la), len(lb)) for sz, tau, k, la, lb in t.pairs_of({"edges": edges}, lab))
        parent, blen, names = parse_newick(newick)
        assert sum(1 for nm in names if nm.startswith("L")) == n_leaves
        s = structure(parent, blen, np.array([nm in lab for nm in names]))
        got = sorted((sz, tau, round(k, 6), la, lb) for _, sz, tau, k, _, la, lb in pick_pairs(parent, s, cap))
        assert got == ref, (rep, got[:3], ref[:3])
    # time since split on a toy dated tree: ((A:1,B:1):2,(C:3,D:3):0);
    parent, blen, names = parse_newick("((A:1,B:1)X:2,(C:3,D:3)Y:0)R;")
    s = structure(parent, blen, np.array([nm in "ABCD" and nm != "" for nm in names]))
    assert names.index("A") > names.index("X") and abs(s["rdist"][names.index("C")] - 3) < 1e-9
    # amendment 2: stratified picker is disjoint, deterministic, and reaches older splits than smallest-first
    pool = [(f"L{i}", f"L{i}:{rng.uniform(0.1, 2):.3f}") for i in range(3000)]
    j = 0
    while len(pool) > 1:
        a, b = (pool.pop(int(rng.integers(len(pool)))) for _ in range(2))
        pool.append((f"I{j}", f"({a[1]},{b[1]})I{j}:{rng.uniform(0.1, 2):.3f}"))
        j += 1
    parent, blen, names = parse_newick(pool[0][1].rsplit(":", 1)[0] + ";")
    s = structure(parent, blen, np.array([nm.startswith("L") for nm in names]))
    r1, r2 = pick_pairs(parent, s, 400, True), pick_pairs(parent, s, 400, True)
    assert r1 == r2, "not deterministic"
    nodes = [r[0] for r in r1]
    for nd in nodes:  # no picked node is an ancestor of another
        p = parent[nd]
        while p >= 0:
            assert p not in set(nodes), "overlap"
            p = parent[p]
    old = pick_pairs(parent, s, 400, False)
    assert np.mean([r[3] for r in r1]) > np.mean([r[3] for r in old]), "stratified does not reach older splits"
    print("selftest ok: 30 random trees match c17s1t_stage.pairs_of; Newick parser and root distances ok; "
          f"stratified picker disjoint and deterministic, mean k {np.mean([r[3] for r in r1]):.2f} vs "
          f"smallest-first {np.mean([r[3] for r in old]):.2f}")


if __name__ == "__main__":
    {"selftest": selftest, "fetch": fetch, "schema": schema, "gate": gate, "seal": seal}[sys.argv[1]]()
