"""C16-S0: hidden inherited state in a tree fate model, paraxial (P) vs lateral plate (L) (pre-registered in
EXPERIMENT_LOG). TRAIN embryos only (c14s0_gate.load_obs filters by partition before reading labels).

M0: 2-state CTMC on P/L along the tree. M1: adds a hidden binary state z (switch rate mu) that sets the P/L
rates. Felsenstein pruning, vectorised by node height. Leave-one-embryo-out held-out log-likelihood per leaf.

Run (GitHub Actions, --petracer): python -I experiments/c16s0_hidden.py
Smoke only: python -I experiments/c16s0_hidden.py --synthetic OUTDIR
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

OUT = HERE / "results_c16"
P_LAB, L_LAB = "Paraxial mesoderm", "Lateral plate mesoderm"
MIN_LEAVES = 100
SEED = 20261117
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


# ---------- forests ----------
class Forest:
    """Concatenated induced trees: parent index (-1 root), edge length, leaf observation (0 P, 1 L, -1 none)."""

    def __init__(self, parent, length, obs):
        self.parent = np.asarray(parent)
        self.length = np.asarray(length, float)
        self.obs = np.asarray(obs)
        n = len(self.parent)
        ch = collections.defaultdict(list)
        for v, p in enumerate(self.parent):
            if p >= 0:
                ch[p].append(v)
        self.is_leaf = np.array([not ch[v] for v in range(n)])
        height = np.zeros(n, int)
        order = sorted(range(n), key=lambda v: 0)
        # heights by repeated relaxation in reverse topological order
        topo, st = [], [v for v in range(n) if self.parent[v] < 0]
        while st:
            v = st.pop()
            topo.append(v)
            st.extend(ch[v])
        for v in reversed(topo):
            if ch[v]:
                height[v] = 1 + max(height[c] for c in ch[v])
        del order
        self.height = height
        self.roots = np.where(self.parent < 0)[0]
        nonroot = np.where(self.parent >= 0)[0]
        self.levels = []
        for h in range(1, height.max() + 1 if n else 1):
            e = nonroot[height[self.parent[nonroot]] == h]
            if len(e):
                self.levels.append(e)
        self.n_obs = int((self.obs >= 0).sum())


def induced(edges, label_of):
    """edges: {parent: [children]} (cell-id leaves). Keep nodes above P/L leaves; collapse 1-child chains."""
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
        if v not in children:
            keep[v] = v in label_of
        else:
            keep[v] = any(keep[c] for c in children[v])
    out_parent, out_len, out_obs, idx = [], [], [], {}

    def add(v, p, length):
        idx[v] = len(out_parent)
        out_parent.append(p)
        out_len.append(length)
        out_obs.append(label_of[v] if v not in children else -1)
        return idx[v]

    for r in roots:
        if not keep[r]:
            continue
        st = [(r, -1, 0)]
        while st:
            v, p, length = st.pop()
            kids = [c for c in children.get(v, []) if keep[c]]
            while len(kids) == 1 and v in children:  # collapse chains
                v = kids[0]
                length += 1
                kids = [c for c in children.get(v, []) if keep[c]]
            me = add(v, p, length)
            for c in kids:
                st.append((c, me, 1))
    return out_parent, out_len, out_obs


def merge(forests):
    parent, length, obs, off = [], [], [], 0
    for f in forests:
        p = np.where(f[0] if isinstance(f[0], np.ndarray) else np.array(f[0]) >= 0, np.array(f[0]) + off, -1)
        parent.extend(p.tolist())
        length.extend(list(f[1]))
        obs.extend(list(f[2]))
        off += len(f[0])
    return Forest(parent, length, obs)


# ---------- models ----------
def unpack(theta, model):
    e = np.exp(theta)
    if model == 0:
        a, b = e[0], e[1]
        Q = np.array([[-a, a], [b, -b]])
        pi = np.array([1 / (1 + np.exp(-theta[2])), 0.0])
        pi[1] = 1 - pi[0]
        return Q, pi
    a0, b0, a1, b1, mu = e[:5]
    # state index = 2*z + c
    Q = np.zeros((4, 4))
    Q[0, 1], Q[1, 0], Q[2, 3], Q[3, 2] = a0, b0, a1, b1
    Q[0, 2] = Q[2, 0] = Q[1, 3] = Q[3, 1] = mu
    np.fill_diagonal(Q, -Q.sum(1))
    w = np.r_[theta[5:8], 0.0]
    pi = np.exp(w - w.max())
    return Q, pi / pi.sum()


def loglik(theta, F, model):
    Q, pi = unpack(theta, model)
    S = Q.shape[0]
    n = len(F.parent)
    L = np.ones((n, S))
    for c in (0, 1):
        m = F.obs == c
        if m.any():
            emit = np.zeros(S)
            emit[[c] if S == 2 else [c, 2 + c]] = 1
            L[m] = emit
    logscale = np.zeros(n)
    acc = np.zeros((n, S))  # log of product of child messages
    has = np.zeros(n, bool)
    cache = {}
    for e in F.levels:  # edges whose parent sits at this height; children are done
        ch = e
        par = F.parent[ch]
        msg = np.empty((len(ch), S))
        for t in np.unique(F.length[ch]):
            if t not in cache:
                cache[t] = expm(Q * t)
            m = F.length[ch] == t
            msg[m] = L[ch[m]] @ cache[t].T
        np.add.at(acc, par, np.log(np.maximum(msg, 1e-300)))
        np.add.at(logscale, par, logscale[ch])
        has[par] = True
        up = np.unique(par)
        mx = acc[up].max(1, keepdims=True)
        L[up] = np.exp(acc[up] - mx)
        logscale[up] += mx[:, 0]
        acc[up] = 0
    r = F.roots
    return float(np.sum(np.log(np.maximum(L[r] @ pi, 1e-300)) + logscale[r]))


def fit(F, model, starts, rng):
    best = None
    k = 3 if model == 0 else 8
    for s in range(starts):
        x0 = np.r_[np.log(rng.uniform(0.02, 0.3, 2 if model == 0 else 5)), rng.normal(0, 0.5, 1 if model == 0 else 3)]
        r = minimize(lambda th: -loglik(th, F, model), x0, method="L-BFGS-B",
                     bounds=[(-9, 3)] * (2 if model == 0 else 5) + [(-8, 8)] * (1 if model == 0 else 3))
        if best is None or r.fun < best.fun:
            best = r
    assert len(best.x) == k
    return best.x, -best.fun


def loeo(forests, rng, starts=3):
    gains = {}
    for e in forests:
        if forests[e].n_obs < MIN_LEAVES:
            continue
        tr = merge([forests[o].raw for o in forests if o != e])
        t0, _ = fit(tr, 0, 1, rng)
        t1, _ = fit(tr, 1, starts, rng)
        F = forests[e]
        gains[e] = (loglik(t1, F, 1) - loglik(t0, F, 0)) / F.n_obs
    return gains


def params(theta):
    e = np.exp(theta[:5])
    return dict(alpha0=e[0], beta0=e[1], alpha1=e[2], beta1=e[3], mu=e[4], half_life_edges=float(np.log(2) / e[4]))


# ---------- simulation (smokes) ----------
def simulate(F, theta, model, rng):
    Q, pi = unpack(theta, model)
    S = Q.shape[0]
    n = len(F.parent)
    state = np.full(n, -1)
    ch = collections.defaultdict(list)
    for v, p in enumerate(F.parent):
        if p >= 0:
            ch[p].append(v)
    cache = {}
    st = list(F.roots)
    for r in F.roots:
        state[r] = rng.choice(S, p=pi)
    while st:
        v = st.pop()
        for c in ch[v]:
            t = F.length[c]
            if t not in cache:
                cache[t] = expm(Q * t)
            p = np.maximum(cache[t][state[v]], 0)
            state[c] = rng.choice(S, p=p / p.sum())
            st.append(c)
    obs = np.where(F.obs >= 0, state % 2 if S == 4 else state, -1)
    return obs


class Emb:
    def __init__(self, raw):
        self.raw = raw
        F = merge([raw])
        self.__dict__.update(F.__dict__)


def with_obs(forests, obs_by):
    out = {}
    for e, f in forests.items():
        raw = (f.raw[0], f.raw[1], list(obs_by[e]))
        out[e] = Emb(raw)
    return out


def smokes(forests, t0_all, rng):
    strong = np.r_[np.log([0.02, 0.25, 0.25, 0.02, 0.05]), 0.0, 0.0, 0.0]
    sim1 = with_obs(forests, {e: simulate(f, strong, 1, rng) for e, f in forests.items()})
    g1 = loeo(sim1, rng, starts=2)
    tr = merge([f.raw for f in sim1.values()])
    th1, _ = fit(tr, 1, 3, rng)
    mu_hat = params(th1)["mu"]
    ok1 = np.mean(list(g1.values())) > 0 and sum(v > 0 for v in g1.values()) >= min(5, len(g1)) and 0.05 / 3 <= mu_hat <= 0.05 * 3
    sim0 = with_obs(forests, {e: simulate(f, t0_all, 0, rng) for e, f in forests.items()})
    g0 = loeo(sim0, rng, starts=2)
    ok2 = np.mean(list(g0.values())) <= 0.002
    say(f"SMOKE (i) strong memory: mean gain {np.mean(list(g1.values())):+.4f}, positive {sum(v > 0 for v in g1.values())}/{len(g1)}, "
        f"mu hat {mu_hat:.3f} (true 0.05) -> {'PASS' if ok1 else 'FAIL'}")
    say(f"SMOKE (ii) no memory: mean gain {np.mean(list(g0.values())):+.4f} -> {'PASS' if ok2 else 'FAIL'}")
    return bool(ok1 and ok2), dict(strong=g1, mu_hat=mu_hat, none=g0)


def analyse(forests, outdir):
    rng = np.random.default_rng(SEED)
    for e, f in forests.items():
        say(f"{e}: P/L leaves {f.n_obs} (P {(f.obs == 0).sum()}, L {(f.obs == 1).sum()}), nodes {len(f.parent)}")
    allF = merge([f.raw for f in forests.values()])
    t0_all, ll0 = fit(allF, 0, 1, rng)
    t1_all, ll1 = fit(allF, 1, 3, rng)
    pr = params(t1_all)
    say(f"all train: LL M0 {ll0:.1f}, M1 {ll1:.1f} (+{(ll1 - ll0) / allF.n_obs:.4f}/leaf); M1 params {pr}")
    smoke_ok, smoke = smokes(forests, t0_all, rng)
    g = loeo(forests, rng)
    m = float(np.mean(list(g.values()))) if g else float("nan")
    pos = sum(v > 0 for v in g.values())
    if not smoke_ok:
        verdict = "INVALID (smoke failed)"
    elif m > 0 and pos >= len(g) - 1:
        verdict = "MEMORY TERM SUPPORTED (exploratory)"
    elif m <= 0:
        verdict = "NOT SUPPORTED"
    else:
        verdict = "MIXED"
    say("held-out gain per leaf: " + ", ".join(f"{k} {v:+.4f}" for k, v in g.items()) + f"; mean {m:+.4f}, positive {pos}/{len(g)} -> {verdict}")
    res = dict(verdict=verdict, heldout_gain_per_leaf=g, mean_gain=m, positive=pos, n_embryos=len(g),
               m0_all=dict(alpha=float(np.exp(t0_all[0])), beta=float(np.exp(t0_all[1]))), m1_all=pr,
               ll_all=dict(m0=ll0, m1=ll1, leaves=allF.n_obs), smoke_pass=smoke_ok, smoke=smoke,
               leaves={e: dict(P=int((f.obs == 0).sum()), L=int((f.obs == 1).sum())) for e, f in forests.items()})
    Path(outdir).mkdir(parents=True, exist_ok=True)
    (Path(outdir) / "c16s0_hidden.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    (Path(outdir) / "c16s0_hidden.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    return res


def load_train():
    import c14s0_gate as g0
    obs, embryos, raw = g0.load_obs()
    lab = obs["lineage"].to_dict()
    label_of = {c: (0 if v == P_LAB else 1) for c, v in lab.items() if v in (P_LAB, L_LAB)}
    per = collections.defaultdict(list)
    for key, tree in g0.iter_train_trees(raw, embryos):
        per[key.rsplit("-C", 1)[0]].append(induced(tree["edges"], label_of))
    out = {}
    for e, fs in per.items():
        p, ln, ob, off = [], [], [], 0
        for f in fs:
            p.extend([x + off if x >= 0 else -1 for x in f[0]])
            ln.extend(f[1])
            ob.extend(f[2])
            off += len(f[0])
        out[e] = Emb((p, ln, ob))
    return out


def synthetic(outdir):
    rng = np.random.default_rng(1)

    def rand_tree(n):
        parent, k = [-1], 1
        frontier = [0]
        while k < 2 * n - 1:
            v = frontier.pop(rng.integers(len(frontier)))
            for _ in range(2):
                parent.append(v)
                frontier.append(k)
                k += 1
        return parent, [1] * len(parent), [-1] * len(parent)

    forests = {}
    for i in range(6):
        p, ln, ob = rand_tree(400)
        isleaf = [True] * len(p)
        for v in p:
            if v >= 0:
                isleaf[v] = False
        ob = [0 if isleaf[v] else -1 for v in range(len(p))]
        forests[f"E{i}"] = Emb((p, ln, ob))
    # real labels: from a memory model so the real-data path is exercised too
    truth = np.r_[np.log([0.03, 0.3, 0.3, 0.03, 0.08]), 0.0, 0.0, 0.0]
    forests = with_obs(forests, {e: simulate(f, truth, 1, rng) for e, f in forests.items()})
    res = analyse(forests, outdir)
    assert res["smoke_pass"], res["smoke"]
    print("SYNTHETIC OK", res["verdict"], round(res["m1_all"]["mu"], 3))


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--synthetic":
        return synthetic(sys.argv[2])
    analyse(load_train(), OUT)


if __name__ == "__main__":
    main()
