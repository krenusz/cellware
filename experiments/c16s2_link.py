"""C16-S2: is the C16 hidden inherited state readable in a cell's own expression? (pre-registered in EXPERIMENT_LOG)
TRAIN embryos only.

1. Fit the C16-S0 hidden-state model (M1) on all train trees.
2. For each P/L leaf: z_rel = P(z = 1 | every other leaf's label), from the outside (down-pass) message, so the
   leaf's own label is excluded (the relatives' signal).
3. Within each observed label (P, L), leave-one-embryo-out ridge regression of z_rel on base (C14 base design +
   cell-type dummies) vs base + 30 expression PCs (C14-S3 readout). Per embryo: Spearman(prediction, z_rel);
   delta = rho(base+PCs) - rho(base), cell-weighted over the two labels.

Run (GitHub Actions, --petracer): python -I experiments/c16s2_link.py
Smoke only: python -I experiments/c16s2_link.py --synthetic OUTDIR
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c16s0_hidden as h  # noqa: E402

OUT = HERE / "results_c16"
PCS = [f"pc{i}" for i in range(30)]
MIN_PER_LABEL = 30
SEED = 20261119
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def induced_ids(edges, label_of):
    """As c16s0_hidden.induced, also returning the cell id of each leaf (None for internal nodes)."""
    parent, children = {}, collections.defaultdict(list)
    for p, ch in edges.items():
        for c in ch:
            parent[c] = p
            children[p].append(c)
    nodes = set(parent) | set(edges)
    roots = [n for n in nodes if n not in parent]
    order, st = [], list(roots)
    while st:
        v = st.pop()
        order.append(v)
        st.extend(children.get(v, []))
    keep = {}
    for v in reversed(order):
        keep[v] = (v in label_of) if v not in children else any(keep[c] for c in children[v])
    P, Ln, Ob, ids = [], [], [], []
    for r in roots:
        if not keep[r]:
            continue
        st = [(r, -1, 0)]
        while st:
            v, p, length = st.pop()
            kids = [c for c in children.get(v, []) if keep[c]]
            while len(kids) == 1 and v in children:
                v = kids[0]
                length += 1
                kids = [c for c in children.get(v, []) if keep[c]]
            me = len(P)
            P.append(p)
            Ln.append(length)
            leaf = v not in children
            Ob.append(label_of[v] if leaf else -1)
            ids.append(v if leaf else None)
            for c in kids:
                st.append((c, me, 1))
    return P, Ln, Ob, ids


def load_train_ids():
    import c14s0_gate as g0
    obs, embryos, raw = g0.load_obs()
    lab = obs["lineage"].to_dict()
    label_of = {c: (0 if v == h.P_LAB else 1) for c, v in lab.items() if v in (h.P_LAB, h.L_LAB)}
    del obs
    per = collections.defaultdict(list)
    for key, tree in g0.iter_train_trees(raw, embryos):
        per[key.rsplit("-C", 1)[0]].append(induced_ids(tree["edges"], label_of))
    del raw
    out, ids = {}, {}
    for e, fs in per.items():
        p, ln, ob, idl, off = [], [], [], [], 0
        for f in fs:
            p.extend([x + off if x >= 0 else -1 for x in f[0]])
            ln.extend(f[1])
            ob.extend(f[2])
            idl.extend(f[3])
            off += len(f[0])
        out[e] = h.Emb((p, ln, ob))
        ids[e] = idl
    return out, ids


def zrel(theta, F):
    """P(z = 1 | all other leaves) for every node (meaningful at leaves)."""
    Q, pi = h.unpack(theta, 1)
    S, n = 4, len(F.parent)
    L = np.ones((n, S))
    for c in (0, 1):
        m = F.obs == c
        emit = np.zeros(S)
        emit[[c, 2 + c]] = 1
        L[m] = emit
    logmsg = np.zeros((n, S))
    acc = np.zeros((n, S))
    cache = {}

    def P(t):
        if t not in cache:
            cache[t] = expm(Q * t)
        return cache[t]

    for ch in F.levels:
        par = F.parent[ch]
        msg = np.empty((len(ch), S))
        for t in np.unique(F.length[ch]):
            m = F.length[ch] == t
            msg[m] = L[ch[m]] @ P(t).T
        lm = np.log(np.maximum(msg, 1e-300))
        lm -= lm.max(1, keepdims=True)
        logmsg[ch] = lm
        np.add.at(acc, par, lm)
        up = np.unique(par)
        L[up] = np.exp(acc[up] - acc[up].max(1, keepdims=True))
    out = np.zeros((n, S))
    out[F.roots] = np.log(np.maximum(pi, 1e-300))
    for ch in reversed(F.levels):
        par = F.parent[ch]
        a = out[par] + acc[par] - logmsg[ch]
        w = np.exp(a - a.max(1, keepdims=True))
        nxt = np.empty((len(ch), S))
        for t in np.unique(F.length[ch]):
            m = F.length[ch] == t
            nxt[m] = w[m] @ P(t)
        out[ch] = np.log(np.maximum(nxt, 1e-300))
    pz = np.exp(out - out.max(1, keepdims=True))
    return (pz[:, 2] + pz[:, 3]) / pz.sum(1)


def ridge_pred(Xtr, ytr, Xte, alpha=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[~(sd > 0)] = 1
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    A1 = np.c_[np.ones(len(A)), A]
    reg = alpha * np.eye(A1.shape[1])
    reg[0, 0] = 0
    w = np.linalg.solve(A1.T @ A1 + reg, A1.T @ (ytr - 0))
    return np.c_[np.ones(len(B)), B] @ w


def deltas(T, base_cols, extra_cols):
    """T: rows = leaves with columns emb, lab, z, base_cols, extra_cols. Per embryo cell-weighted delta rho."""
    per = {}
    for e in sorted(T["emb"].unique()):
        num, wsum, n_lab = 0.0, 0, {}
        ok = True
        for lab in (0, 1):
            te = T[(T.emb == e) & (T.lab == lab)]
            tr = T[(T.emb != e) & (T.lab == lab)]
            n_lab[lab] = len(te)
            if len(te) < MIN_PER_LABEL or len(tr) < MIN_PER_LABEL or te["z"].std() == 0:
                ok = False
                continue
            rb = spearmanr(ridge_pred(tr[base_cols].to_numpy(float), tr["z"].to_numpy(), te[base_cols].to_numpy(float)), te["z"])[0]
            cols = base_cols + extra_cols
            rf = spearmanr(ridge_pred(tr[cols].to_numpy(float), tr["z"].to_numpy(), te[cols].to_numpy(float)), te["z"])[0]
            num += len(te) * (np.nan_to_num(rf) - np.nan_to_num(rb))
            wsum += len(te)
        per[e] = dict(delta=num / wsum if wsum else float("nan"), n_P=n_lab[0], n_L=n_lab[1], in_rule=ok)
    return per


def summary(per):
    v = [r["delta"] for r in per.values() if r["in_rule"]]
    return (float(np.mean(v)) if v else float("nan")), sum(x > 0 for x in v), len(v)


def table(forests, ids, theta, D, base):
    rows = []
    for e, F in forests.items():
        z = zrel(theta, F)
        for v, cid in enumerate(ids[e]):
            if cid is not None:
                rows.append((cid, e, int(F.obs[v]), z[v]))
    Z = pd.DataFrame(rows, columns=["cell", "emb", "lab", "z"]).set_index("cell")
    T = Z.join(base, how="inner").join(D[PCS], how="inner")
    return T


def analyse(forests, ids, D, base, outdir, rng):
    allF = h.merge([f.raw for f in forests.values()])
    th, ll = h.fit(allF, 1, 3, rng)
    say(f"M1 all train: {h.params(th)}")
    T = table(forests, ids, th, D, base)
    T = T.sort_values(["emb"]).copy()
    bcols = list(base.columns)
    say(f"P/L leaves with expression PCs: {len(T)} of {sum(f.n_obs for f in forests.values())}")
    for e, g in T.groupby("emb"):
        say(f"  {e}: P {int((g.lab == 0).sum())}, L {int((g.lab == 1).sum())}, z_rel mean {g.z.mean():.3f} sd {g.z.std():.3f}")
    # smokes
    pl = T.copy()
    pl["planted"] = pl["z"] + rng.normal(0, pl["z"].std(), len(pl))
    sm1 = deltas(pl, bcols + PCS, ["planted"])
    ok1 = all(r["delta"] > 0.05 for r in sm1.values() if r["in_rule"]) and any(r["in_rule"] for r in sm1.values())
    sh = T.copy()
    sh["z"] = sh.groupby(["emb", "lab"])["z"].transform(lambda s: rng.permutation(s.to_numpy()))
    sm2 = deltas(sh, bcols, PCS)
    v2 = [r["delta"] for r in sm2.values() if r["in_rule"]]
    m2 = float(np.mean(v2)) if v2 else 0.0
    lim2 = max(0.03, 2 * float(np.std(v2, ddof=1)) / np.sqrt(len(v2))) if len(v2) > 1 else 0.03
    ok2 = abs(m2) <= lim2  # amendment 1: signed mean within max(0.03, 2 SE)
    say(f"SMOKE (i) planted column: min delta {min((r['delta'] for r in sm1.values() if r['in_rule']), default=float('nan')):+.3f} "
        f"-> {'PASS' if ok1 else 'FAIL'}")
    say(f"SMOKE (ii) shuffled z_rel: mean delta {m2:+.3f} (limit {lim2:.3f}) -> {'PASS' if ok2 else 'FAIL'}")
    per = deltas(T, bcols, PCS)
    m, pos, n = summary(per)
    if not (ok1 and ok2):
        verdict = "INVALID (smoke failed)"
    elif n and m > 0.05 and pos >= min(4, n):
        verdict = "READABLE (exploratory)"
    elif not n or m <= 0:
        verdict = "NOT READABLE" if n else "DATA-LIMITED"
    else:
        verdict = "MIXED"
    for e, r in per.items():
        say(f"  {e}: delta rho {r['delta']:+.3f} (P {r['n_P']}, L {r['n_L']}){'' if r['in_rule'] else ' [not in rule]'}")
    say(f"mean delta {m:+.3f}, positive {pos}/{n} -> {verdict}")
    res = dict(verdict=verdict, per_embryo=per, mean=m, positive=pos, n=n, m1=h.params(th),
               smoke=dict(planted=sm1, shuffled=sm2, pass_=bool(ok1 and ok2)))
    Path(outdir).mkdir(parents=True, exist_ok=True)
    (Path(outdir) / "c16s2_link.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    (Path(outdir) / "c16s2_link.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    return res


def base_design(D, s1):
    B = s1.X(D, [], None)
    return pd.concat([B, pd.get_dummies(D["cell_type"].astype(str), prefix="ct").astype(float)], axis=1)


def synthetic(outdir):
    import c16s1_time as t

    rng = np.random.default_rng(3)
    raw = t.rand_forests(rng, n_emb=6, n_leaves=250)
    truth = np.r_[np.log([0.03, 0.3, 0.3, 0.03, 0.08]), 0.0, 0.0, 0.0]
    forests, ids, rows = {}, {}, []
    for e, f in raw.items():
        obs = h.simulate(f, truth, 1, rng)
        forests[e] = h.Emb((f.raw[0], f.raw[1], list(obs)))
        ids[e] = [f"{e}_c{v}" if obs[v] >= 0 else None for v in range(len(obs))]
        rows += [ids[e][v] for v in range(len(obs)) if obs[v] >= 0]
    D = pd.DataFrame(rng.normal(size=(len(rows), 30)), index=rows, columns=PCS)
    base = pd.DataFrame(dict(x1=rng.normal(size=len(rows)), x2=rng.normal(size=len(rows))), index=rows)
    res = analyse(forests, ids, D, base, outdir, rng)
    assert res["smoke"]["pass_"], res["smoke"]
    assert res["verdict"] in ("NOT READABLE", "MIXED"), res["verdict"]  # random PCs carry no z
    print("SYNTHETIC OK", res["verdict"])


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--synthetic":
        return synthetic(sys.argv[2])
    rng = np.random.default_rng(SEED)
    forests, ids = load_train_ids()
    import c14s3_region as r

    D, _, s1 = r.readout_with_pcs()
    D = D[~D.index.duplicated()]
    analyse(forests, ids, D, base_design(D, s1), OUT, rng)


if __name__ == "__main__":
    main()
