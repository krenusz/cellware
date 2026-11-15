"""C14-S0 chapter gate, TRAIN embryos only (petracer_embryo, claim H-b).

Question: does a progenitor cell's state beyond its cell type (author subtype, UMAP position, cycle
phase) predict the lineage composition of its nearest tree relatives, in an embryo the model never saw?

Design (pre-registered in EXPERIMENT_LOG, C14-S0):
- Embryo filter first: partitions.train_only("petracer_embryo", ...) on the embryo column; every other
  column and every tree is read or kept only for train embryos (sealed trees are skipped by key).
- Relatives of cell c: leaves of the smallest ancestor whose other leaves include >= K_REL cells of a
  cell type different from c's; ancestors with > MAX_CLADE leaves are not used (c is skipped).
- Per cell type T: L1, L2 = the two commonest lineages among relatives (different-type cells only).
  T is eligible if L1 and L2 each hold >= 20% of those relatives and >= MIN_CELLS cells are evaluable.
  Outcome y = 1 if relatives in L1 outnumber those in L2 (>= 3 relatives in L1 or L2; ties dropped).
- Base model: own lineage, germ layer, log total counts, detection rate, log clade size, ancestor depth,
  stage. Full model: base + subtype + UMAP x, y + phase. Logistic regression per type.
- Statistic: leave-one-embryo-out, within-type AUC(full) - AUC(base), cell-weighted mean over types.
- Readout gate (H21): outcome counts per side, cells and clusters (ancestors); >= 30 clusters per side.
- H17 power: bootstrap clusters from the train OOF predictions to partition A's projected cluster
  counts (stage-matched per-embryo means x A's public stages); null sd from paired swaps of
  full/base predictions per cluster; power = power.sim_power. Also at half the effect.

Run: .venv/Scripts/python.exe -I experiments/c14s0_gate.py   (prints and writes results_c14/c14s0_gate.txt)
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from partitions import partition_of, train_only  # noqa: E402
from power import sim_power  # noqa: E402

DS = "petracer_embryo"
RAW = HERE.parent / "data" / "ext_petracer" / "raw"
OUT = HERE / "results_c14"
K_REL, MAX_CLADE, MIN_CELLS, MIN_SHARE = 5, 300, 200, 0.20
SEED, B = 20261115, 400
# partition A's embryos come from the public sample names (partitions.py), not from data
NAMES = [f"E{s}-R{r}" for s in ["7.5", "8.0", "8.5", "9.0", "9.5"] for r in (1, 2, 3)] + ["E10.0-R1"]
A_EMBRYOS = [e for e in NAMES if partition_of(DS, e) == "A"]
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def stage_of(emb):
    return float(emb[1:].split("-")[0])


def _s(x):
    return x.decode() if isinstance(x, bytes) else x


def load_obs():
    (h5,) = sorted(RAW.glob("*.h5td"))
    f = h5py.File(h5, "r")
    cats = [_s(c) for c in f["obs/embryo/categories"][()]]
    emb = pd.DataFrame({"embryo": pd.Categorical.from_codes(f["obs/embryo/codes"][()], cats).astype(str)})
    emb["row"] = np.arange(len(emb))
    tr = train_only(DS, emb, unit_col="embryo")  # embryo filter before any other column
    mask = np.zeros(len(emb), bool)
    mask[tr["row"].to_numpy()] = True

    def col(name):
        g = f[f"obs/{name}"]
        if isinstance(g, h5py.Group):
            cc = [_s(c) for c in g["categories"][()]]
            return pd.Categorical.from_codes(g["codes"][mask], cc).astype(str)
        return g[mask]

    obs = pd.DataFrame({"cell": [_s(i) for i in f["obs/_index"][mask]]})
    for c in ["embryo", "cell_type", "cell_subtype", "lineage", "germ_layer", "phase"]:
        obs[c] = col(c)
    for c in ["total_counts", "detection_rate"]:
        obs[c] = col(c)
    um = f["obsm/X_umap"][mask]
    obs["umap1"], obs["umap2"] = um[:, 0], um[:, 1]
    train_embryos = sorted(set(obs["embryo"]))
    say(f"train embryos ({len(train_embryos)}): {train_embryos}; cells {len(obs)}")
    raw = f["obst"][()]
    f.close()
    return obs.set_index("cell"), train_embryos, raw


def iter_train_trees(raw, train_embryos):
    """Stream the obst JSON object; yield (key, tree) for train embryos only."""
    s = raw.decode()
    dec = json.JSONDecoder()
    i = s.index("{") + 1
    keep = set(train_embryos)
    n_seen = n_kept = 0
    while True:
        while s[i] in " \n\r\t,":
            i += 1
        if s[i] == "}":
            break
        key, i = dec.raw_decode(s, i)
        while s[i] in " :":
            i += 1
        val, i = dec.raw_decode(s, i)
        n_seen += 1
        if key.rsplit("-C", 1)[0] in keep:
            n_kept += 1
            yield key, val
        del val
    say(f"trees seen {n_seen}, kept (train) {n_kept}")


def relatives(tree, ctype, lin):
    edges = tree["edges"]
    parent, children = {}, collections.defaultdict(list)
    for p, ch in edges.items():
        for c in ch:
            parent[c] = p
            children[p].append(c)
    nodes = set(parent) | set(edges)
    leaves = [n for n in nodes if n not in children]
    roots = [n for n in nodes if n not in parent]
    depth, order, stack = {}, [], [(r, 0) for r in roots]
    while stack:
        n, d = stack.pop()
        depth[n] = d
        order.append(n)
        stack.extend((c, d + 1) for c in children.get(n, []))
    size, lists = {}, {}
    for n in reversed(order):  # children before parents
        if n not in children:
            size[n], lists[n] = 1, [n]
        else:
            size[n] = sum(size[c] for c in children[n])
            if size[n] <= MAX_CLADE:
                lists[n] = [x for c in children[n] for x in lists[c]]
    out = []
    for c in leaves:
        if c not in ctype:
            continue
        a = parent.get(c)
        while a is not None and a in lists:
            rel = [x for x in lists[a] if x != c and x in ctype and ctype[x] != ctype[c]]
            if len(rel) >= K_REL:
                out.append((c, a, size[a], depth[a], collections.Counter(lin[x] for x in rel)))
                break
            a = parent.get(a)
    return out, len(leaves), sum(1 for c in leaves if c in ctype)


def fit_pred(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xtr), ytr)
    return m.predict_proba(sc.transform(Xte))[:, 1]


def design(df, full):
    num = pd.DataFrame({"ltc": np.log1p(df["total_counts"]), "det": df["detection_rate"],
                        "lsz": np.log(df["clade"]), "dep": df["depth"], "stage": df["stage"]}, index=df.index)
    cats = ["lineage", "germ_layer"] + (["cell_subtype", "phase"] if full else [])
    X = pd.concat([num, pd.get_dummies(df[cats].astype(str)).astype(float)], axis=1)
    if full:
        X["u1"], X["u2"] = df["umap1"], df["umap2"]
    return X


def auc_delta(s):
    if s["y"].nunique() < 2:
        return None
    return roc_auc_score(s["y"], s["p_full"]) - roc_auc_score(s["y"], s["p_base"])


def main():
    OUT.mkdir(exist_ok=True)
    obs, train_embryos, raw = load_obs()
    obs["stage"] = obs["embryo"].map(stage_of)
    ctype, lin = obs["cell_type"].to_dict(), obs["lineage"].to_dict()
    recs, n_leaves, n_match = [], 0, 0
    for key, tree in iter_train_trees(raw, train_embryos):
        r, nl, nm = relatives(tree, ctype, lin)
        n_leaves += nl
        n_match += nm
        recs += [(key,) + x for x in r]
    del raw
    say(f"train leaves {n_leaves}, matched to obs {n_match} ({n_match / max(n_leaves, 1):.3f}); "
        f"cells with relatives {len(recs)}")
    if n_match / max(n_leaves, 1) < 0.9:
        say("STOP: leaf names do not match the cell index")
        return
    R = pd.DataFrame(recs, columns=["tree", "cell", "anc", "clade", "depth", "relcnt"]).set_index("cell")
    R = R.join(obs)
    R["cluster"] = R["tree"] + ":" + R["anc"]

    rows, gate = [], []
    for T, g in R.groupby("cell_type"):
        tot = collections.Counter()
        for c in g["relcnt"]:
            tot.update(c)
        if len(tot) < 2:
            continue
        (L1, n1), (L2, n2) = tot.most_common(2)
        allr = sum(tot.values())
        if n1 / allr < MIN_SHARE or n2 / allr < MIN_SHARE:
            continue
        a = g["relcnt"].map(lambda c: c.get(L1, 0))
        b = g["relcnt"].map(lambda c: c.get(L2, 0))
        ok = (a + b >= 3) & (a != b)
        h = g[ok].copy()
        h["y"] = (a[ok] > b[ok]).astype(int)
        if len(h) < MIN_CELLS or h["y"].nunique() < 2:
            continue
        h["T"] = T
        rows.append(h)
        gate.append(dict(type=T, L1=L1, L2=L2, cells=len(h), y1=int(h.y.sum()), y0=int((1 - h.y).sum()),
                         clusters_y1=h.loc[h.y == 1, "cluster"].nunique(),
                         clusters_y0=h.loc[h.y == 0, "cluster"].nunique(),
                         embryos=h["embryo"].nunique(), subtypes=h["cell_subtype"].nunique()))
    if not rows:
        say("STOP: no eligible progenitor type")
        return
    D = pd.concat(rows)
    say("\nREADOUT GATE (H21), per eligible type:")
    for gr in gate:
        say(f"  {gr}")

    D["p_base"] = np.nan
    D["p_full"] = np.nan
    for T, g in D.groupby("T"):
        Xb, Xf = design(g, False), design(g, True)
        for e in train_embryos:
            te = (g["embryo"] == e).to_numpy()
            if te.sum() == 0 or g.loc[~te, "y"].nunique() < 2:
                continue
            D.loc[g.index[te], "p_base"] = fit_pred(Xb[~te], g.loc[~te, "y"], Xb[te])
            D.loc[g.index[te], "p_full"] = fit_pred(Xf[~te], g.loc[~te, "y"], Xf[te])
    D = D.dropna(subset=["p_base", "p_full"])
    per, w = {}, {}
    for T, g in D.groupby("T"):
        d = auc_delta(g)
        if d is not None:
            per[T], w[T] = d, len(g)
    delta = sum(per[t] * w[t] for t in per) / sum(w.values())
    say(f"\nLOEO delta AUC (full - base), cell-weighted over {len(per)} types: {delta:.4f}")
    for t in per:
        say(f"  {t}: {per[t]:+.4f} (n {w[t]})")
    for e in train_embryos:
        ds = [(auc_delta(g), len(g)) for _, g in D[D.embryo == e].groupby("T")]
        ds = [(d, n) for d, n in ds if d is not None]
        if ds:
            say(f"  embryo {e}: {sum(d * n for d, n in ds) / sum(n for _, n in ds):+.4f} "
                f"(cells {sum(n for _, n in ds)})")

    rng = np.random.default_rng(SEED)
    proj = {}
    for T, g in D.groupby("T"):
        per_emb = g.groupby("embryo")["cluster"].nunique()
        st_mean = per_emb.groupby(per_emb.index.map(stage_of)).mean()
        proj[T] = int(sum(st_mean.get(stage_of(e), 0) for e in A_EMBRYOS))
    say(f"\nA embryos (public names): {A_EMBRYOS}; projected clusters per type: {proj}")
    sub = {T: g.reset_index(drop=True) for T, g in D.groupby("T")}
    groups = {T: list(g.groupby("cluster").indices.values()) for T, g in sub.items()}

    def draw(swap):
        num = den = 0.0
        for T, s0 in sub.items():
            cl, n = groups[T], proj[T]
            if n < 2:
                continue
            pick = rng.integers(0, len(cl), n)
            idx = np.concatenate([cl[k] for k in pick])
            s = s0.iloc[idx].copy()
            if swap:
                flip = np.repeat(rng.random(n) < 0.5, [len(cl[k]) for k in pick])
                pb, pf = s["p_base"].to_numpy(), s["p_full"].to_numpy()
                s["p_base"], s["p_full"] = np.where(flip, pf, pb), np.where(flip, pb, pf)
            d = auc_delta(s)
            if d is not None:
                num += d * len(s)
                den += len(s)
        return num / den if den else float("nan")

    stats = np.array([draw(False) for _ in range(B)])
    null = np.array([draw(True) for _ in range(B)])
    null_sd = float(np.nanstd(null))
    ok_s = stats[~np.isnan(stats)]
    pw = sim_power(ok_s, null_sd)
    pw_half = sim_power(ok_s - ok_s.mean() / 2, null_sd)
    say(f"simulated delta AUC on A: mean {ok_s.mean():.4f}, sd {ok_s.std():.4f}; null sd {null_sd:.4f}")
    say(f"H17 power on A: {pw:.3f} at the train effect; {pw_half:.3f} at half the effect")
    res = dict(delta_loeo=delta, per_type=per, n_per_type=w, gate=gate, projected_clusters_A=proj,
               power_A=pw, power_A_half=pw_half, null_sd=null_sd, sim_mean=float(ok_s.mean()))
    (OUT / "c14s0_gate.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    (OUT / "c14s0_gate.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
