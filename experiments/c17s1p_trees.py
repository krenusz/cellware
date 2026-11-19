"""C17-S1P: sister-subclade heritable share in the embryo lineage trees, TRAIN embryos only.

Bence 2026-11-19: "1 go". Serves: C17-S1 (C17-S1R: grow the systems). Pre-registered in EXPERIMENT_LOG (C17-S1P).

The two children of a tree node are the two sister arms of the B38 heritable share (sis_data.heritable_share,
unchanged): H = (D_rand - D_sis) / (D_rand - D_noise) over the obs `lineage` categories, random partners within
the same embryo. Loading goes through c14s0_gate (train embryos filtered with partitions.train_only before any
other column; trees of other partitions skipped by key before parsing).

    python -I experiments/c17s1p_trees.py            # Actions, --petracer; writes results_c17/c17s1p_trees.json
    python -I experiments/c17s1p_trees.py selftest   # offline: pair picker on a toy tree, synthetic H checks
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sis_data as sd  # noqa: E402
from partitions import train_only  # noqa: E402,F401  (guard: partitioned reads go through train_only, in c14s0_gate)

OUT = HERE / "results_c17" / "c17s1p_trees.json"
MAX_CLADE, MIN_ARM, MIN_PAIRS = 300, 4, 30
SEED, B = 20261119, 300
MISSING = {"", "nan", "na", "none", "unknown"}


def pairs_of(tree, lab):
    """Disjoint sister-subclade pairs: (clade size, labels of arm 1, labels of arm 2), smallest clades first."""
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
    for n in reversed(order):  # children before parents
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
        a, b = sorted(ch, key=lambda c: -len(lists[c]))[:2]
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


def counts(pairs, cats):
    ix = {c: i for i, c in enumerate(cats)}
    k1 = np.zeros((len(pairs), len(cats)), int)
    k2 = np.zeros_like(k1)
    for r, (_, la, lb) in enumerate(pairs):
        for x in la:
            k1[r, ix[x]] += 1
        for x in lb:
            k2[r, ix[x]] += 1
    return k1, k2


def estimate(k1, k2, group, rng, n_perm=2000, n_noise=500):
    return {k: float(v) for k, v in sd.heritable_share(k1, k2, group, rng, n_perm, n_noise).items()}


def cluster_ci(k1, k2, group, rng, b=B):
    """Resample embryos with replacement (all their pairs), recompute H."""
    embs = np.unique(group)
    vals = []
    for _ in range(b):
        pick = rng.choice(embs, len(embs), replace=True)
        ix = np.concatenate([np.flatnonzero(group == e) for e in pick])
        g = np.concatenate([np.full((group == e).sum(), j) for j, e in enumerate(pick)])
        vals.append(sd.heritable_share(k1[ix], k2[ix], g, rng, 200, 100)["H"])
    return np.nanpercentile(vals, [2.5, 97.5]).tolist()


def smoke(k1, k2, group, rng, reps=20):
    """Mean H over `reps` synthetic draws on the real arm sizes (amendment 1: one draw has sd 0.15-0.26)."""
    out = {}
    for mode in ("syn0", "syn1"):
        vals = []
        for _ in range(reps):
            s1, s2 = sd.synthetic(k1, k2, group, mode, rng)
            vals.append(estimate(s1, s2, group, rng, 300, 100)["H"])
        out[mode] = float(np.mean(vals))
        out[f"{mode}_sd"] = float(np.std(vals))
    out["pass"] = abs(out["syn0"]) < 0.15 and out["syn1"] > 0.80
    return out


def selftest():
    leaves4 = lambda p: [f"{p}{i}" for i in range(4)]  # noqa: E731
    edges = {"r": ["n1", "n2"], "n1": ["x", "y"], "n2": ["u", "v"], "x": leaves4("x"), "y": leaves4("y"),
             "u": leaves4("u"), "v": leaves4("v")}
    lab = {c: ("A" if c[0] in "xu" else "B") for p in "xyuv" for c in leaves4(p)}
    pr = pairs_of({"edges": edges}, lab)
    assert len(pr) == 2 and all(s == 8 for s, _, _ in pr), pr  # n1 and n2 taken, r overlaps both
    rng = np.random.default_rng(1)
    n = 200
    q = rng.dirichlet(np.ones(4) * 0.5, n)
    k1 = np.array([rng.multinomial(6, p) for p in q])
    k2 = np.array([rng.multinomial(6, p) for p in q])
    grp = np.repeat(np.arange(4), n // 4)
    sm = smoke(k1, k2, grp, rng)
    assert sm["pass"], sm
    return {"ok": True, "smoke": sm}


def main():
    if sys.argv[1:2] == ["selftest"]:
        print(json.dumps(selftest()))
        return
    import c14s0_gate as g0

    rng = np.random.default_rng(SEED)
    obs, train_embryos, raw = g0.load_obs()
    lab = {c: str(v) for c, v in obs["lineage"].items() if str(v).strip().lower() not in MISSING}
    del obs
    rows = []
    for key, tree in g0.iter_train_trees(raw, train_embryos):
        emb = key.rsplit("-C", 1)[0]
        rows += [(emb, *p) for p in pairs_of(tree, lab)]
    del raw
    per_emb = dict(collections.Counter(r[0] for r in rows))
    res = {"train_embryos": train_embryos, "n_pairs": len(rows), "pairs_per_embryo": per_emb,
           "rule": f"clade <= {MAX_CLADE} leaves, >= {MIN_ARM} labelled cells per arm, disjoint, smallest first"}
    if len(rows) < MIN_PAIRS:
        res["verdict"] = f"STOP: {len(rows)} pairs < {MIN_PAIRS} (H21)"
    else:
        cats = sorted({x for r in rows for x in r[2] + r[3]})
        k1, k2 = counts([r[1:] for r in rows], cats)
        embs = sorted(per_emb)
        group = np.array([embs.index(r[0]) for r in rows])
        res["n_categories"] = len(cats)
        res["arm_cells_median"] = [float(np.median(k1.sum(1))), float(np.median(k2.sum(1)))]
        res["smoke"] = smoke(k1, k2, group, rng)
        if not res["smoke"]["pass"]:
            res["verdict"] = "INVALID: smoke failed"
        else:
            res["estimate"] = estimate(k1, k2, group, rng)
            res["H_ci_embryo_bootstrap"] = cluster_ci(k1, k2, group, rng)
            sizes = np.array([r[1] for r in rows])
            cuts = np.quantile(sizes, [1 / 3, 2 / 3])
            tert = np.digitize(sizes, cuts)
            res["exploratory_H_by_clade_size_tertile"] = {
                f"t{t}": {"n": int((tert == t).sum()), "median_size": float(np.median(sizes[tert == t])),
                          "H": estimate(k1[tert == t], k2[tert == t], group[tert == t], rng, 500, 200)["H"]}
                for t in range(3) if (tert == t).sum() >= 10}
            res["per_embryo_H"] = {e: estimate(k1[group == j], k2[group == j], group[group == j], rng, 500, 200)["H"]
                                   for j, e in enumerate(embs) if (group == j).sum() >= 10}
            res["verdict"] = "ESTIMATED"
    OUT.parent.mkdir(exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: v for k, v in res.items() if k != "pairs_per_embryo"}, indent=1,
                     default=lambda o: o.item() if hasattr(o, "item") else str(o)))


if __name__ == "__main__":
    main()
