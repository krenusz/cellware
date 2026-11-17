"""C16-S1: time-varying-rate control for the C16-S0 hidden inherited state (pre-registered in EXPERIMENT_LOG).
TRAIN embryos only (trees and labels via c16s0_hidden.load_train).

M0t: 2-state CTMC on P/L whose rates (a_k, b_k) differ by tree-depth bin k (3 bins, tertiles of pooled edge depth).
M1t: the C16-S0 hidden-state model with one rate multiplier per depth bin (bin 0 = 1) and one switch rate mu.
Primary: leave-one-embryo-out held-out log-likelihood per leaf, M1t minus M0t.

Run (GitHub Actions, --petracer): python -I experiments/c16s1_time.py
Smoke only: python -I experiments/c16s1_time.py --synthetic OUTDIR
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c16s0_hidden as h  # noqa: E402

OUT = HERE / "results_c16"
NB = 3
SEED = 20261118
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def depth_of(F):
    d = np.zeros(len(F.parent))
    for v, p in enumerate(F.parent):  # parents precede children (induced() and rand_tree order)
        if p >= 0:
            assert p < v
            d[v] = d[p] + F.length[v]
    return d


def set_bins(forests, cuts):
    for f in forests.values():
        f.bin = np.searchsorted(cuts, depth_of(f), side="right")


def cuts_from(forests):
    d = np.concatenate([depth_of(f)[f.parent >= 0] for f in forests.values()])
    return np.quantile(d, [1 / 3, 2 / 3])


class Pooled:
    def __init__(self, embs):
        self.raw = None
        F = h.merge([e.raw for e in embs])
        self.__dict__.update(F.__dict__)
        self.bin = np.concatenate([e.bin for e in embs])


def unpack(theta, model):
    """Per-bin Q list and root pi. model 0 = M0t, 1 = M1t, 'm0'/'m1' = S0 models (no bins)."""
    if model in ("m0", "m1"):
        Q, pi = h.unpack(theta, 0 if model == "m0" else 1)
        return [Q] * NB, pi
    e = np.exp(theta)
    if model == 0:
        Qs = [np.array([[-e[k], e[k]], [e[NB + k], -e[NB + k]]]) for k in range(NB)]
        p0 = 1 / (1 + np.exp(-theta[2 * NB]))
        return Qs, np.array([p0, 1 - p0])
    a0, b0, a1, b1, mu = e[:5]
    mult = np.r_[1.0, e[5:5 + NB - 1]]
    Qs = []
    for s in mult:
        Q = np.zeros((4, 4))
        Q[0, 1], Q[1, 0], Q[2, 3], Q[3, 2] = s * a0, s * b0, s * a1, s * b1
        Q[0, 2] = Q[2, 0] = Q[1, 3] = Q[3, 1] = mu
        np.fill_diagonal(Q, -Q.sum(1))
        Qs.append(Q)
    w = np.r_[theta[5 + NB - 1:8 + NB - 1], 0.0]
    pi = np.exp(w - w.max())
    return Qs, pi / pi.sum()


NPAR = {0: 2 * NB + 1, 1: 8 + NB - 1}


def loglik(theta, F, model):
    Qs, pi = unpack(theta, model)
    S = Qs[0].shape[0]
    n = len(F.parent)
    L = np.ones((n, S))
    for c in (0, 1):
        m = F.obs == c
        if m.any():
            emit = np.zeros(S)
            emit[[c] if S == 2 else [c, 2 + c]] = 1
            L[m] = emit
    logscale = np.zeros(n)
    acc = np.zeros((n, S))
    cache = {}
    for ch in F.levels:
        par = F.parent[ch]
        msg = np.empty((len(ch), S))
        keys = F.length[ch] * 10 + F.bin[ch]
        for kk in np.unique(keys):
            m = keys == kk
            t, b = F.length[ch][m][0], F.bin[ch][m][0]
            if (t, b) not in cache:
                cache[(t, b)] = expm(Qs[b] * t)
            msg[m] = L[ch[m]] @ cache[(t, b)].T
        np.add.at(acc, par, np.log(np.maximum(msg, 1e-300)))
        np.add.at(logscale, par, logscale[ch])
        up = np.unique(par)
        mx = acc[up].max(1, keepdims=True)
        L[up] = np.exp(acc[up] - mx)
        logscale[up] += mx[:, 0]
        acc[up] = 0
    r = F.roots
    return float(np.sum(np.log(np.maximum(L[r] @ pi, 1e-300)) + logscale[r]))


def fit(F, model, starts, rng):
    best = None
    k = NPAR[model]
    nr = 2 * NB if model == 0 else 5 + NB - 1
    for _ in range(starts):
        x0 = np.r_[np.log(rng.uniform(0.02, 0.3, nr)), rng.normal(0, 0.5, k - nr)]
        if model == 1:
            x0[5:5 + NB - 1] = rng.normal(0, 0.3, NB - 1)
        r = minimize(lambda th: -loglik(th, F, model), x0, method="L-BFGS-B",
                     bounds=[(-9, 3)] * nr + [(-8, 8)] * (k - nr))
        if best is None or r.fun < best.fun:
            best = r
    return best.x, -best.fun


def loeo(forests, rng, starts=3):
    rows = {}
    for e, F in forests.items():
        if F.n_obs < h.MIN_LEAVES:
            continue
        tr = Pooled([forests[o] for o in forests if o != e])
        t0, _ = fit(tr, 0, 1, rng)
        t1, _ = fit(tr, 1, starts, rng)
        s0, _ = h.fit(tr, 0, 1, rng)
        s1, _ = h.fit(tr, 1, starts, rng)
        ll = dict(M0t=loglik(t0, F, 0), M1t=loglik(t1, F, 1), M0=loglik(s0, F, "m0"), M1=loglik(s1, F, "m1"))
        rows[e] = {k: v / F.n_obs for k, v in ll.items()}
    return rows


def gains(rows, a, b):
    return {e: r[a] - r[b] for e, r in rows.items()}


def simulate(F, theta, model, rng):
    Qs, pi = unpack(theta, model)
    S = Qs[0].shape[0]
    state = np.full(len(F.parent), -1)
    cache = {}
    for v, p in enumerate(F.parent):
        if p < 0:
            state[v] = rng.choice(S, p=pi)
            continue
        key = (F.length[v], F.bin[v])
        if key not in cache:
            cache[key] = expm(Qs[key[1]] * key[0])
        pr = np.maximum(cache[key][state[p]], 0)
        state[v] = rng.choice(S, p=pr / pr.sum())
    return np.where(F.obs >= 0, state % 2 if S == 4 else state, -1)


def relabel(forests, obs_by):
    out = {}
    for e, f in forests.items():
        n = h.Emb((f.raw[0], f.raw[1], list(obs_by[e])))
        n.bin = f.bin
        out[e] = n
    return out


def smokes(forests, t0t_all, rng):
    # (i) depth-varying rates, no memory: M1t must not beat M0t
    sim0 = relabel(forests, {e: simulate(f, t0t_all, 0, rng) for e, f in forests.items()})
    r0 = loeo(sim0, rng, starts=2)
    g0 = gains(r0, "M1t", "M0t")
    g0_s0 = gains(r0, "M1", "M0")
    ok1 = np.mean(list(g0.values())) <= 0.002
    # (ii) strong memory, no depth variation: M1t must beat M0t
    strong = np.r_[np.log([0.02, 0.25, 0.25, 0.02, 0.05]), np.zeros(NB - 1), 0.0, 0.0, 0.0]
    sim1 = relabel(forests, {e: simulate(f, strong, 1, rng) for e, f in forests.items()})
    g1 = gains(loeo(sim1, rng, starts=2), "M1t", "M0t")
    ok2 = np.mean(list(g1.values())) > 0 and sum(v > 0 for v in g1.values()) >= min(5, len(g1))
    say(f"SMOKE (i) depth-varying, no memory: M1t-M0t {np.mean(list(g0.values())):+.4f} -> {'PASS' if ok1 else 'FAIL'}; "
        f"(info) S0 contrast M1-M0 on the same data {np.mean(list(g0_s0.values())):+.4f}")
    say(f"SMOKE (ii) strong memory: M1t-M0t {np.mean(list(g1.values())):+.4f}, positive {sum(v > 0 for v in g1.values())}/{len(g1)} "
        f"-> {'PASS' if ok2 else 'FAIL'}")
    return bool(ok1 and ok2), dict(no_memory=g0, no_memory_s0_contrast=g0_s0, strong=g1)


def analyse(forests, outdir):
    rng = np.random.default_rng(SEED)
    cuts = cuts_from(forests)
    set_bins(forests, cuts)
    say(f"depth bin cuts (edges from root): {cuts.round(1).tolist()}")
    for e, f in forests.items():
        leaf = f.obs >= 0
        md = depth_of(f)[leaf].mean() if leaf.any() else float("nan")
        say(f"{e}: P/L leaves {f.n_obs}, mean leaf depth {md:.1f}, leaf bins {np.bincount(f.bin[leaf], minlength=NB).tolist()}")
    allF = Pooled(list(forests.values()))
    t0t, ll0t = fit(allF, 0, 1, rng)
    t1t, ll1t = fit(allF, 1, 3, rng)
    e0 = np.exp(t0t)
    say(f"all train M0t rates per bin: a {e0[:NB].round(4).tolist()}, b {e0[NB:2 * NB].round(4).tolist()}; "
        f"M1t mu {np.exp(t1t[4]):.4f} (half-life {np.log(2) / np.exp(t1t[4]):.1f} edges), "
        f"bin multipliers {np.r_[1, np.exp(t1t[5:5 + NB - 1])].round(3).tolist()}; LL M0t {ll0t:.1f} M1t {ll1t:.1f}")
    smoke_ok, smoke = smokes(forests, t0t, rng)
    rows = loeo(forests, rng)
    g = gains(rows, "M1t", "M0t")
    m = float(np.mean(list(g.values())))
    pos = sum(v > 0 for v in g.values())
    if not smoke_ok:
        verdict = "INVALID (smoke failed)"
    elif m > 0 and pos >= len(g) - 1:
        verdict = "HIDDEN STATE SURVIVES THE TIME CONTROL (exploratory)"
    elif m <= 0:
        verdict = "EXPLAINED BY DEPTH-VARYING RATES"
    else:
        verdict = "MIXED"
    for name, (a, b) in dict(primary_M1t_vs_M0t=("M1t", "M0t"), time_alone_M0t_vs_M0=("M0t", "M0"),
                             S0_M1_vs_M0=("M1", "M0"), M1_vs_M0t=("M1", "M0t")).items():
        gg = gains(rows, a, b)
        say(f"{name}: " + ", ".join(f"{k} {v:+.4f}" for k, v in gg.items()) + f"; mean {np.mean(list(gg.values())):+.4f}")
    say(f"PRIMARY mean {m:+.4f}, positive {pos}/{len(g)} -> {verdict}")
    res = dict(verdict=verdict, primary=g, mean=m, positive=pos, rows=rows, cuts=cuts.tolist(),
               m0t_all=dict(a=e0[:NB].tolist(), b=e0[NB:2 * NB].tolist()),
               m1t_all=dict(mu=float(np.exp(t1t[4])), mult=np.exp(t1t[5:5 + NB - 1]).tolist()),
               smoke_pass=smoke_ok, smoke=smoke)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    (Path(outdir) / "c16s1_time.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    (Path(outdir) / "c16s1_time.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    return res


def rand_forests(rng, n_emb=6, n_leaves=300):
    def rand_tree(n):
        parent, k, frontier = [-1], 1, [0]
        while k < 2 * n - 1:
            v = frontier.pop(rng.integers(len(frontier)))
            for _ in range(2):
                parent.append(v)
                frontier.append(k)
                k += 1
        return parent

    forests = {}
    for i in range(n_emb):
        p = rand_tree(n_leaves)
        isleaf = np.ones(len(p), bool)
        isleaf[[v for v in p if v >= 0]] = False
        forests[f"E{i}"] = h.Emb((p, [1] * len(p), [0 if isleaf[v] else -1 for v in range(len(p))]))
    return forests


def synthetic(outdir):
    rng = np.random.default_rng(2)
    forests = rand_forests(rng, n_leaves=800)
    set_bins(forests, cuts_from(forests))
    truth = np.r_[np.log([0.03, 0.3, 0.3, 0.03, 0.08]), np.zeros(NB - 1), 0.0, 0.0, 0.0]
    forests = relabel(forests, {e: simulate(f, truth, 1, rng) for e, f in forests.items()})
    res = analyse(forests, outdir)
    assert res["smoke_pass"], res["smoke"]
    print("SYNTHETIC OK", res["verdict"])


# ---------- parallel execution (technical amendment 2026-11-17): one Actions job per fold task, 4 cores per job ----------
PARTS = OUT / "c16s1_parts"
_G = {}


def _x0(model, rng, s0=False):
    if s0:
        n = 2 if model == 0 else 5
        k_ = 3 if model == 0 else 8
        return np.r_[np.log(rng.uniform(0.02, 0.3, n)), rng.normal(0, 0.5, k_ - n)]
    k_ = NPAR[model]
    nr = 2 * NB if model == 0 else 5 + NB - 1
    x0 = np.r_[np.log(rng.uniform(0.02, 0.3, nr)), rng.normal(0, 0.5, k_ - nr)]
    if model == 1:
        x0[5:5 + NB - 1] = rng.normal(0, 0.3, NB - 1)
    return x0


def _one(job):
    """job = (tag, model, x0); tag 't' = binned models (this file), 's' = C16-S0 models."""
    tag, model, x0 = job
    F = _G["F"]
    if tag == "t":
        nr = 2 * NB if model == 0 else 5 + NB - 1
        bounds = [(-9, 3)] * nr + [(-8, 8)] * (len(x0) - nr)
        r = minimize(lambda th: -loglik(th, F, model), x0, method="L-BFGS-B", bounds=bounds)
    else:
        n = 2 if model == 0 else 5
        r = minimize(lambda th: -h.loglik(th, F, model), x0, method="L-BFGS-B", bounds=[(-9, 3)] * n + [(-8, 8)] * (len(x0) - n))
    return tag, model, r.x, -r.fun


def pfit(F, jobs, procs=4):
    """Run independent fits in parallel (fork shares F); return best theta per (tag, model)."""
    import multiprocessing as mp

    _G["F"] = F
    if procs > 1 and sys.platform != "win32":
        with mp.get_context("fork").Pool(procs) as pool:
            out = pool.map(_one, jobs)
    else:
        out = [_one(j) for j in jobs]
    best = {}
    for tag, model, x, ll in out:
        if (tag, model) not in best or ll > best[(tag, model)][1]:
            best[(tag, model)] = (x, ll)
    return best


def fold_jobs(rng, starts):
    return ([("t", 0, _x0(0, rng))] + [("t", 1, _x0(1, rng)) for _ in range(starts)]
            + [("s", 0, _x0(0, rng, True))] + [("s", 1, _x0(1, rng, True)) for _ in range(starts)])


def fold_rows(forests, e, rng, starts):
    tr = Pooled([forests[o] for o in forests if o != e])
    b = pfit(tr, fold_jobs(rng, starts))
    F = forests[e]
    ll = dict(M0t=loglik(b[("t", 0)][0], F, 0), M1t=loglik(b[("t", 1)][0], F, 1),
              M0=h.loglik(b[("s", 0)][0], F, 0), M1=h.loglik(b[("s", 1)][0], F, 1))
    return {k_: v / F.n_obs for k_, v in ll.items()}


def prepare(forests):
    cuts = cuts_from(forests)
    set_bins(forests, cuts)
    allF = Pooled(list(forests.values()))
    rng = np.random.default_rng(SEED)
    b = pfit(allF, [("t", 0, _x0(0, rng))])  # deterministic M0t all-train fit (truth for smoke i)
    return cuts, allF, b[("t", 0)][0]


def sim_forests(forests, kind, t0t):
    rng = np.random.default_rng(SEED + (1 if kind == "null" else 2))
    if kind == "null":
        theta, model = t0t, 0
    else:
        theta, model = np.r_[np.log([0.02, 0.25, 0.25, 0.02, 0.05]), np.zeros(NB - 1), 0.0, 0.0, 0.0], 1
    return relabel(forests, {e: simulate(f, theta, model, rng) for e, f in forests.items()})


def task_names(forests):
    emb = [e for e, f in forests.items() if f.n_obs >= h.MIN_LEAVES]
    return ["all"] + [f"{kind}:{e}" for kind in ("real", "null", "strong") for e in emb]


def run_task(forests, name, outdir):
    cuts, allF, t0t = prepare(forests)
    rng = np.random.default_rng(SEED + sum(map(ord, name)))  # deterministic per task
    if name == "all":
        b = pfit(allF, [("t", 0, _x0(0, rng)), *[("t", 1, _x0(1, rng)) for _ in range(3)]])
        t1t, ll1t = b[("t", 1)]
        ll0t = loglik(t0t, allF, 0)
        res = dict(task=name, cuts=cuts.tolist(), m0t=np.exp(t0t[:2 * NB]).tolist(), ll0t=ll0t, ll1t=ll1t,
                   mu=float(np.exp(t1t[4])), mult=np.exp(t1t[5:5 + NB - 1]).tolist(),
                   leaves={e: dict(n=f.n_obs, depth=float(depth_of(f)[f.obs >= 0].mean()) if f.n_obs else None,
                                   bins=np.bincount(f.bin[f.obs >= 0], minlength=NB).tolist()) for e, f in forests.items()})
    else:
        kind, e = name.split(":", 1)
        fs = forests if kind == "real" else sim_forests(forests, kind, t0t)
        res = dict(task=name, rows=fold_rows(fs, e, rng, 3 if kind == "real" else 2))
    Path(outdir).mkdir(parents=True, exist_ok=True)
    (Path(outdir) / f"{name.replace(':', '__')}.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(name, res.get("rows", ""), flush=True)
    return res


def combine(partdir, outdir):
    LOG.clear()
    P = {json.loads(p.read_text())["task"]: json.loads(p.read_text()) for p in Path(partdir).glob("*.json")}
    a = P["all"]
    say(f"depth bin cuts (edges from root): {np.round(a['cuts'], 1).tolist()}")
    for e, v in a["leaves"].items():
        say(f"{e}: P/L leaves {v['n']}, mean leaf depth {v['depth']}, leaf bins {v['bins']}")
    say(f"all train M0t rates a {np.round(a['m0t'][:NB], 4).tolist()}, b {np.round(a['m0t'][NB:], 4).tolist()}; "
        f"M1t mu {a['mu']:.4f} (half-life {np.log(2) / a['mu']:.1f} edges), bin multipliers {np.round([1] + a['mult'], 3).tolist()}; "
        f"LL M0t {a['ll0t']:.1f} M1t {a['ll1t']:.1f}")
    rows = {kind: {t.split(":", 1)[1]: v["rows"] for t, v in P.items() if t.startswith(kind + ":")} for kind in ("real", "null", "strong")}
    g0, g0s, g1 = gains(rows["null"], "M1t", "M0t"), gains(rows["null"], "M1", "M0"), gains(rows["strong"], "M1t", "M0t")
    ok1 = np.mean(list(g0.values())) <= 0.002
    ok2 = np.mean(list(g1.values())) > 0 and sum(v > 0 for v in g1.values()) >= min(5, len(g1))
    say(f"SMOKE (i) depth-varying, no memory: M1t-M0t {np.mean(list(g0.values())):+.4f} -> {'PASS' if ok1 else 'FAIL'}; "
        f"(info) S0 contrast M1-M0 on the same data {np.mean(list(g0s.values())):+.4f}")
    say(f"SMOKE (ii) strong memory: M1t-M0t {np.mean(list(g1.values())):+.4f}, positive {sum(v > 0 for v in g1.values())}/{len(g1)} "
        f"-> {'PASS' if ok2 else 'FAIL'}")
    r = rows["real"]
    g = gains(r, "M1t", "M0t")
    m = float(np.mean(list(g.values())))
    pos = sum(v > 0 for v in g.values())
    if not (ok1 and ok2):
        verdict = "INVALID (smoke failed)"
    elif m > 0 and pos >= len(g) - 1:
        verdict = "HIDDEN STATE SURVIVES THE TIME CONTROL (exploratory)"
    elif m <= 0:
        verdict = "EXPLAINED BY DEPTH-VARYING RATES"
    else:
        verdict = "MIXED"
    for nm, (x, y) in dict(primary_M1t_vs_M0t=("M1t", "M0t"), time_alone_M0t_vs_M0=("M0t", "M0"),
                           S0_M1_vs_M0=("M1", "M0"), M1_vs_M0t=("M1", "M0t")).items():
        gg = gains(r, x, y)
        say(f"{nm}: " + ", ".join(f"{k_} {v:+.4f}" for k_, v in gg.items()) + f"; mean {np.mean(list(gg.values())):+.4f}")
    say(f"PRIMARY mean {m:+.4f}, positive {pos}/{len(g)} -> {verdict}")
    res = dict(verdict=verdict, primary=g, mean=m, positive=pos, rows=r, all=a, smoke_pass=bool(ok1 and ok2),
               smoke=dict(no_memory=g0, no_memory_s0_contrast=g0s, strong=g1))
    Path(outdir).mkdir(parents=True, exist_ok=True)
    (Path(outdir) / "c16s1_time.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    (Path(outdir) / "c16s1_time.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    return res


def synthetic_tasks(outdir):
    rng = np.random.default_rng(2)
    forests = rand_forests(rng, n_leaves=800)
    set_bins(forests, cuts_from(forests))
    truth = np.r_[np.log([0.03, 0.3, 0.3, 0.03, 0.08]), np.zeros(NB - 1), 0.0, 0.0, 0.0]
    forests = relabel(forests, {e: simulate(f, truth, 1, rng) for e, f in forests.items()})
    parts = Path(outdir) / "parts"
    for name in task_names(forests):
        run_task(forests, name, parts)
    res = combine(parts, outdir)
    assert res["smoke_pass"], res["smoke"]
    print("SYNTHETIC TASKS OK", res["verdict"])


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--synthetic":
        return synthetic(sys.argv[2])
    if len(sys.argv) > 2 and sys.argv[1] == "--synthetic-tasks":
        return synthetic_tasks(sys.argv[2])
    if len(sys.argv) > 1 and sys.argv[1] == "--combine":
        return combine(PARTS, OUT)
    if len(sys.argv) > 2 and sys.argv[1] == "task":
        return run_task(h.load_train(), sys.argv[2], PARTS)
    analyse(h.load_train(), OUT)


if __name__ == "__main__":
    main()
