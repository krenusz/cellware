"""C15-S1b: train-only readout + leave-one-host-out fit + per-host noise + H17 power (pre-registered in EXPERIMENT_LOG).

Readout (C14 transferred): for a leaf cell of state cluster T, relatives = other-cluster leaves of the smallest
clade around it holding >= 5 of them (clade <= 300 leaves). Per T (train pooled): L1, L2 = two most frequent
relative clusters, each >= 20%. y = 1 if the cell's relatives lean L1 over L2 (>= 3 informative, ties dropped).
Base: log counts, detection, log clade size, depth, log unit leaves, arm. Full: + 30 PCs of 2,000 HVGs (train).
Per-cluster logistic, leave-one-host-out; per-host cell-weighted within-cluster delta AUC.

TRAIN hosts only (partitions growth12_host_v2): sealed units' trees are skipped by file name; the h5ad's obs
index and cluster codes are read whole and immediately cut to train leaves (only train rows used).
Outputs recoded (hosts h.., clusters s.., arms arm..).

Run (GitHub Actions): python -I experiments/c15s1b_fit.py
Smoke only: python -I experiments/c15s1b_fit.py --synthetic OUTDIR
"""
from __future__ import annotations

import collections
import json
import re
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c15g_hosts as g  # noqa: E402
from partitions import partition_of  # noqa: E402
from power import sim_power  # noqa: E402

DS = "growth12_host_v2"
OUT = HERE / "results_c15"
TREE_RE = re.compile(r"^(\d{3,5}_[A-Za-z0-9]+_T\d+)_tree\.nwk$")
MIN_OTHER, MAX_CLADE, MIN_INFO, MIN_SHARE = 5, 300, 3, 0.20
MIN_CELLS, MIN_CLADES = 200, 30
N_HVG, N_PC, MIN_LEAVES, N_SIM, SEED = 2000, 30, 200, 20000, 20261117
N_SHUFFLE = 5
SEALED_LEAVES = {}  # filled from results_c15/c15s0_rebal.json
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def train_unit(u):
    return partition_of(DS, u.split("_")[0]) == "train"


# ---------- trees ----------
def parse_newick(s):
    """Returns (parent list, leaf name per node or None). Node 0 = root."""
    parent, name = [-1], [None]
    stack, cur, i, n = [], 0, 0, len(s)
    while i < n:
        c = s[i]
        if c == "(":
            parent.append(cur)
            name.append(None)
            stack.append(cur)
            cur = len(parent) - 1
            i += 1
        elif c == ",":
            cur = stack[-1]
            parent.append(cur)
            name.append(None)
            cur = len(parent) - 1
            i += 1
        elif c == ")":
            cur = stack.pop()
            i += 1
        elif c == ";":
            break
        else:
            j = i
            while j < n and s[j] not in "(),;":
                j += 1
            tok = s[i:j].split(":")[0].strip().strip("'")
            if tok and name[cur] is None:
                name[cur] = tok
            i = j
    # the first "(" made node 1 child of root 0; collapse: root is node 0 with child 1
    children = collections.defaultdict(list)
    for v, p in enumerate(parent):
        if p >= 0:
            children[p].append(v)
    leaves = {v for v in range(len(parent)) if not children[v]}
    nm = [name[v] if v in leaves else None for v in range(len(parent))]
    return parent, nm, children


def readout_tree(parent, nm, children, cluster_of):
    """Per leaf with a cluster: (T, clade node, Counter of other-cluster relatives, clade size, depth)."""
    n = len(parent)
    order = []
    st = [0]
    while st:
        v = st.pop()
        order.append(v)
        st.extend(children[v])
    cnt = [None] * n
    size = [0] * n
    for v in reversed(order):
        if not children[v]:
            c = collections.Counter()
            if nm[v] in cluster_of:
                c[cluster_of[nm[v]]] += 1
            cnt[v], size[v] = c, 1
        else:
            c = collections.Counter()
            s = 0
            for w in children[v]:
                c.update(cnt[w])
                s += size[w]
            cnt[v], size[v] = c, s
    depth = [0] * n
    for v in order[1:]:
        depth[v] = depth[parent[v]] + 1
    out = {}
    for v in range(n):
        if children[v] or nm[v] not in cluster_of:
            continue
        T = cluster_of[nm[v]]
        a = parent[v]
        while a >= 0:
            other = sum(cnt[a].values()) - cnt[a][T]
            if other >= MIN_OTHER:
                break
            a = parent[a]
        if a < 0 or size[a] > MAX_CLADE:
            continue
        rel = collections.Counter({k: x for k, x in cnt[a].items() if k != T})
        out[nm[v]] = (T, a, rel, size[a], depth[v])
    return out


# ---------- data ----------
def h5_read(path, wanted):
    """Train leaves only: obs index -> rows; cluster codes; raw counts rows (layers/counts or X)."""
    h5py = g.h5py_mod()
    import scipy.sparse as sp
    with h5py.File(path, "r") as f:
        obs = f["obs"]
        ik = obs.attrs.get("_index", "_index")
        ik = ik.decode() if isinstance(ik, bytes) else ik
        index = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in obs[ik][:]])
        rows = np.where(np.isin(index, list(wanted)))[0]
        cg = obs.get("__categories")
        col = "Cluster-Name" if "Cluster-Name" in obs else next(k for k in obs if re.search(r"cluster|leiden", k, re.I))
        if isinstance(obs[col], h5py.Group):
            cats, codes = obs[col]["categories"][:], obs[col]["codes"][:]
        else:
            cats, codes = cg[col][:], obs[col][:]
        codes = codes[rows]  # only train rows kept
        cats = [c.decode() if isinstance(c, bytes) else str(c) for c in cats]
        M = f["layers"]["counts"] if "layers" in f and "counts" in f["layers"] else f["X"]
        indptr = M["indptr"][:]
        shape = M.attrs.get("shape")
        ng = int(shape[1]) if shape is not None else int(M["indices"][:].max()) + 1
        data, ind, ptr = [], [], [0]
        for r in rows:
            a, b = indptr[r], indptr[r + 1]
            data.append(M["data"][a:b])
            ind.append(M["indices"][a:b])
            ptr.append(ptr[-1] + (b - a))
        X = sp.csr_matrix((np.concatenate(data) if data else np.array([]), np.concatenate(ind) if ind else np.array([], int),
                           np.array(ptr)), shape=(len(rows), ng))
    return index[rows], codes, cats, X


def scan(tar_path, wd):
    trees, h5 = {}, None
    with tarfile.open(tar_path, "r|gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            base = Path(m.name).name
            mt = TREE_RE.match(base)
            if mt and train_unit(mt.group(1)):
                trees[mt.group(1)] = tf.extractfile(m).read().decode("utf-8", "replace")
            elif base.lower().endswith(".h5ad") and "combined" in base.lower():
                p = Path(wd) / "combined.h5ad"
                f = tf.extractfile(m)
                with open(p, "wb") as out:
                    while chunk := f.read(1 << 24):
                        out.write(chunk)
                h5 = p
    return trees, h5


def build(trees, h5):
    parsed = {u: parse_newick(s) for u, s in trees.items()}
    leaves = {nm for (_, names, _) in parsed.values() for nm in names if nm}
    index, codes, cats, X = h5_read(h5, leaves)
    cluster_of = {c: int(k) for c, k in zip(index, codes) if k >= 0}
    row_of = {c: i for i, c in enumerate(index)}
    recs = []
    for u, (par, names, ch) in parsed.items():
        nleaves = sum(1 for x in names if x)
        for cell, (T, clade, rel, csize, depth) in readout_tree(par, names, ch, cluster_of).items():
            recs.append(dict(cell=cell, unit=u, host=u.split("_")[0], arm=u.split("_")[1], T=T, clade=f"{u}:{clade}",
                             rel=rel, csize=csize, depth=depth, uleaves=nleaves, row=row_of[cell]))
    say(f"train units {len(parsed)}, leaves {len(leaves)}, leaves in h5ad {len(index)}, cells with a readout {len(recs)}")
    return recs, X, cats


def label(recs):
    by_T = collections.defaultdict(collections.Counter)
    for r in recs:
        by_T[r["T"]].update(r["rel"])
    L = {}
    for T, c in by_T.items():
        tot = sum(c.values())
        top = [k for k, x in c.most_common(2) if x >= MIN_SHARE * tot]
        if len(top) == 2:
            L[T] = top
    out = []
    for r in recs:
        if r["T"] not in L:
            continue
        a, b = (r["rel"][k] for k in L[r["T"]])
        if a + b < MIN_INFO or a == b:
            continue
        out.append({**r, "y": int(a > b)})
    return out, L


def gate(rows):
    keep = set()
    by_T = collections.defaultdict(list)
    for r in rows:
        by_T[r["T"]].append(r)
    for T, rs in by_T.items():
        c1 = {r["clade"] for r in rs if r["y"] == 1}
        c0 = {r["clade"] for r in rs if r["y"] == 0}
        if len(rs) >= MIN_CELLS and len(c1) >= MIN_CLADES and len(c0) >= MIN_CLADES:
            keep.add(T)
    return keep


def pcs(X):
    tot = np.asarray(X.sum(1)).ravel()
    det = np.asarray((X > 0).sum(1)).ravel() / X.shape[1]
    Xn = X.multiply(1e4 / np.maximum(tot, 1)[:, None]).tocsr()
    Xn.data = np.log1p(Xn.data)
    mu = np.asarray(Xn.mean(0)).ravel()
    var = np.asarray(Xn.multiply(Xn).mean(0)).ravel() - mu ** 2
    hvg = np.argsort(-var)[:min(N_HVG, X.shape[1])]
    D = Xn[:, hvg].toarray()
    D = np.clip((D - D.mean(0)) / np.where(D.std(0) > 0, D.std(0), 1), -10, 10)
    from sklearn.decomposition import PCA
    P = PCA(n_components=min(N_PC, D.shape[1] - 1, D.shape[0] - 1), random_state=0).fit_transform(D)
    return P, np.log1p(tot), det


def design(rows, P, ltot, det, arms):
    base = np.array([[ltot[r["row"]], det[r["row"]], np.log(r["csize"]), r["depth"], np.log(r["uleaves"])]
                     + [float(r["arm"] == a) for a in arms[1:]] for r in rows])
    full = np.hstack([base, P[[r["row"] for r in rows]]])
    return base, full


def loho(rows, base, full, ykey):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    y = np.array([r[ykey] for r in rows])
    host = np.array([r["host"] for r in rows])
    T = np.array([r["T"] for r in rows])
    num = collections.Counter()
    den = collections.Counter()
    for t in np.unique(T):
        it = T == t
        for h in np.unique(host[it]):
            te = it & (host == h)
            tr = it & (host != h)
            if len(set(y[tr])) < 2 or len(set(y[te])) < 2 or te.sum() < 10:
                continue
            pr = {}
            for nm, Z in (("b", base), ("f", full)):
                m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
                m.fit(Z[tr], y[tr])
                pr[nm] = m.predict_proba(Z[te])[:, 1]
            d = roc_auc_score(y[te], pr["f"]) - roc_auc_score(y[te], pr["b"])
            num[h] += d * te.sum()
            den[h] += te.sum()
    return {h: (num[h] / den[h], int(den[h])) for h in den}


def summary(per_host):
    d = np.array([v[0] for v in per_host.values()])
    n = np.array([v[1] for v in per_host.values()], float)
    if len(d) == 0:
        return dict(mean=float("nan"), wmean=float("nan"), sd=float("nan"), positive=0, hosts=0)
    return dict(mean=float(d.mean()), wmean=float((d * n).sum() / n.sum()), sd=float(d.std(ddof=1)) if len(d) > 1 else float("nan"),
                positive=int((d > 0).sum()), hosts=len(d))


def variance_model(per_host, wmean):
    d = np.array([v[0] for v in per_host.values()])
    n = np.array([v[1] for v in per_host.values()], float)
    e2 = (d - wmean) ** 2
    A = np.vstack([np.ones_like(n), 1 / n]).T
    from scipy.optimize import nnls
    (a, b), _ = nnls(A, e2)
    return float(a), float(b)


def power(effect, a, b, ratio, rng):
    out = {}
    for p, lv in SEALED_LEAVES.items():
        n = np.array([x * ratio for x in lv if x >= MIN_LEAVES])
        if len(n) < 2:
            out[p] = dict(power=0.0, hosts=len(n))
            continue
        sd = np.sqrt(a + b / np.maximum(n, 1))
        w = n / n.sum()
        se = np.sqrt((w ** 2 * sd ** 2).sum())
        draws = effect + rng.normal(0, 1, (N_SIM, len(n))) * sd
        z = (draws * w).sum(1) / se
        out[p] = dict(power=round(sim_power(z, 1.0), 3), hosts=int(len(n)), se=round(float(se), 4))
    return out


def analyse(recs, X, outdir):
    rng = np.random.default_rng(SEED)
    rows, L = label(recs)
    keep = gate(rows)
    rows = [r for r in rows if r["T"] in keep]
    say(f"labelled cells {len(rows)}; clusters with L1/L2 {len(L)}; passing H21 gate {len(keep)}")
    if not rows:
        res = dict(verdict="NO READOUT (no cluster passes the H21 gate)")
        Path(outdir).mkdir(parents=True, exist_ok=True)
        (Path(outdir) / "c15s1b_fit.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        return res
    used = sorted({r["row"] for r in rows})
    sub = {r: i for i, r in enumerate(used)}
    P, ltot, det = pcs(X[used])
    rows = [{**r, "row": sub[r["row"]]} for r in rows]
    arms = sorted({r["arm"] for r in rows})
    base, full = design(rows, P, ltot, det, arms)
    by = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by[(r["T"], r["host"])].append(i)
    ypl = np.zeros(len(rows), int)
    sig = P[[r["row"] for r in rows], 0] + rng.normal(0, 1, len(rows))
    for t in {r["T"] for r in rows}:
        it = np.array([r["T"] == t for r in rows])
        ypl[it] = (sig[it] > np.median(sig[it])).astype(int)
    sh_runs = []
    for k in range(N_SHUFFLE):  # amendment 1: average of 5 within-(cluster, host) shuffles
        ysh = np.array([r["y"] for r in rows])
        for idx in by.values():
            ysh[idx] = rng.permutation(ysh[idx])
        for i, r in enumerate(rows):
            r["y_sh"] = int(ysh[i])
        sh_runs.append(loho(rows, base, full, "y_sh"))
    hs = set.intersection(*[set(x) for x in sh_runs])
    ph_sh = {h: (float(np.mean([x[h][0] for x in sh_runs])), sh_runs[0][h][1]) for h in hs}
    for i, r in enumerate(rows):
        r["y_pl"] = int(ypl[i])
    s_sh, s_pl = summary(ph_sh), summary(loho(rows, base, full, "y_pl"))
    se_sh = s_sh["sd"] / np.sqrt(max(s_sh["hosts"], 1))
    ok1, ok2 = abs(s_sh["wmean"]) < max(0.01, 2 * se_sh), s_pl["wmean"] > 0.05
    say(f"SMOKE shuffled (mean of {N_SHUFFLE}): {s_sh['wmean']:+.4f} (limit {max(0.01, 2 * se_sh):.4f}) -> {'PASS' if ok1 else 'FAIL'}; "
        f"planted: {s_pl['wmean']:+.4f} -> {'PASS' if ok2 else 'FAIL'}")
    ph = loho(rows, base, full, "y")
    s = summary(ph)
    a, b = variance_model(ph, s["wmean"])
    host_leaves = collections.defaultdict(set)
    for r in recs:
        host_leaves[r["host"]].add(r["unit"])
    readout_cells = collections.Counter(r["host"] for r in rows)
    uleaves = {r["unit"]: r["uleaves"] for r in recs}
    tot_leaves = {h: sum(uleaves[u] for u in us) for h, us in host_leaves.items()}
    ratio = sum(readout_cells.values()) / max(sum(tot_leaves[h] for h in readout_cells), 1)
    pw = power(0.75 * s["wmean"], a, b, ratio, rng) if s["wmean"] > 0 else {p: dict(power=0.0) for p in SEALED_LEAVES}
    pw_floor = power(0.75 * s["wmean"], max(a, 0.0374 ** 2), b, ratio, rng) if s["wmean"] > 0 else {}  # sensitivity only
    ok = [p for p in ("A", "B") if pw.get(p, {}).get("power", 0) >= 0.8]
    verdict = "INVALID (smoke failed)" if not (ok1 and ok2) else (f"PASS ({', '.join(ok)} >= 0.8)" if ok else "FAIL (A and B < 0.8)")
    hosts = {h: f"h{i + 1:02d}" for i, h in enumerate(sorted(ph))}
    say(f"train per-host delta: mean {s['mean']:+.4f}, weighted {s['wmean']:+.4f}, sd {s['sd']:.4f}, "
        f"positive {s['positive']}/{s['hosts']}; variance model sd^2 = {a:.5f} + {b:.3f}/n; readout/leaf ratio {ratio:.2f}")
    say(f"H17 power (effect 0.75 x weighted mean): {pw} -> {verdict}")
    say(f"sensitivity, between-host sd floored at C14's 0.0374: {pw_floor}")
    res = dict(verdict=verdict, smoke_shuffled=s_sh, smoke_planted=s_pl, smoke_pass=bool(ok1 and ok2), train=s,
               per_host={hosts[h]: dict(delta=round(v[0], 4), cells=v[1], leaves=tot_leaves.get(h)) for h, v in ph.items()},
               variance_model=dict(a=a, b=b), readout_per_leaf=ratio, power=pw, power_with_C14_sd_floor=pw_floor, effect_used=0.75 * s["wmean"],
               clusters_passing=len(keep), clusters_with_pair=len(L), labelled_cells=len(rows),
               y1_share=float(np.mean([r["y"] for r in rows])))
    Path(outdir).mkdir(parents=True, exist_ok=True)
    (Path(outdir) / "c15s1b_fit.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    (Path(outdir) / "c15s1b_fit.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    return res


def load_sealed_leaves():
    r = json.loads((OUT / "c15s0_rebal.json").read_text(encoding="utf-8"))
    SEALED_LEAVES.update({p: r["partitions"][p]["leaves_per_host"] for p in "ABC"})


def synthetic(outdir):
    """Fake hosts: a binary tree per unit; a hidden program z drives both y-relevant relatives and PC0."""
    import scipy.sparse as sp
    global train_unit
    train_unit = lambda u: True  # noqa: E731
    rng = np.random.default_rng(4)
    SEALED_LEAVES.update({"A": [800, 1200, 2000, 3000, 5000], "B": [900, 1500, 2500, 4000, 6000], "C": [700, 1000]})
    recs = []
    rows_X = []
    k = 0
    for h in range(10):
        for t in range(2):
            u = f"{3000 + h}_{'AB'[h % 2]}A_T{t + 1}"
            n = 256
            z = np.repeat(rng.normal(size=n // 16), 16)  # clade-level program, inherited
            clusters = np.where(rng.random(n) < 0.5, 0, rng.choice([1, 2], n, p=[0.5, 0.5]))
            lean = (z + rng.normal(0, 0.5, n)) > 0
            clusters = np.where(clusters == 0, 0, np.where(lean, 1, 2))
            names = [f"{u}.c{j}" for j in range(n)]

            def build_tree(lo, hi):
                if hi - lo == 1:
                    return names[lo]
                mid = (lo + hi) // 2
                return f"({build_tree(lo, mid)},{build_tree(mid, hi)})"
            par, nm, ch = parse_newick(build_tree(0, n) + ";")
            cof = {names[j]: int(clusters[j]) for j in range(n)}
            for cell, (T, clade, rel, csize, depth) in readout_tree(par, nm, ch, cof).items():
                j = names.index(cell)
                recs.append(dict(cell=cell, unit=u, host=str(3000 + h), arm=u.split("_")[1], T=T, clade=f"{u}:{clade}",
                                 rel=rel, csize=csize, depth=depth, uleaves=n, row=k + j))
            expr = rng.poisson(5, (n, 60)).astype(float)
            expr[:, :10] += np.maximum(0, 4 * z[:, None] + 4)
            rows_X.append(expr)
            k += n
    X = sp.csr_matrix(np.vstack(rows_X))
    global MIN_CELLS, MIN_CLADES
    MIN_CELLS, MIN_CLADES = 50, 10
    res = analyse(recs, X, outdir)
    assert res["smoke_pass"], res.get("smoke_shuffled")
    assert res["train"]["wmean"] > 0.02, res["train"]
    print("SYNTHETIC OK", round(res["train"]["wmean"], 3), res["power"])


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--synthetic":
        return synthetic(sys.argv[2])
    load_sealed_leaves()
    dl = HERE.parent / "data" / "ext_c15" / "raw"
    dl.mkdir(parents=True, exist_ok=True)
    tar = dl / "record_5847461.tar.gz"
    g.download(tar)
    with tempfile.TemporaryDirectory() as wd:
        trees, h5 = scan(tar, wd)
        recs, X, _ = build(trees, h5)
    analyse(recs, X, OUT)


if __name__ == "__main__":
    main()
