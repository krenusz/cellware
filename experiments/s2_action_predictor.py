"""S2: learned action predictor (day-2 state + cytokine -> clone N-share) vs E38.
See the S2 pre-registration in EXPERIMENT_LOG.md. Needs the cytokine expression matrix
(downloaded on GitHub Actions; local downloads fail on SSL).

    python experiments/s2_action_predictor.py
"""

from __future__ import annotations

import gzip
import json
import os

import e26_selection as e26
import e33_measured as e33
import e37_pool as e37
import numpy as np
import pandas as pd
import torch
import track_b24 as b24
from clone_state_model import OUT, shard_of
from fit_cytokine import DATA, pooled
from track_b13 import fetch

PRIMARY = ("full", "G", "M")
REPORT = ("Epo", "SCF")
N_HVG, N_PC, FOLDS, MIN_CLONES = 2000, 50, 5, 50
torch.set_num_threads(4)


# ---------- data ----------
def day2_matrix():
    """Day-2 cells of fit-set clones: (cell clone ids, log1p matrix over the 2000 HVGs)."""
    with gzip.open(os.path.join(DATA, "metadata.txt.gz"), "rt") as f:
        rows = [line.rstrip("\n").split("\t") for line in f][1:]
    day = np.array([r[2] for r in rows])
    m = np.loadtxt(
        gzip.open(os.path.join(DATA, "clone_matrix.mtx.gz"), "rt"),
        comments="%",
        skiprows=3,
        dtype=int,
    )
    clone = np.full(len(rows) + 1, -1)
    clone[m[:, 1]] = m[:, 0] - 1  # cells 1-based
    keep = {
        c
        for c in range(1, len(rows) + 1)
        if day[c - 1] == "2" and clone[c] >= 0 and shard_of(int(clone[c]), 2) != 1
    }
    cells, genes, vals = [], [], []
    with gzip.open(fetch("normed_counts.mtx.gz"), "rt") as f:
        line = f.readline()
        while line.startswith("%"):
            line = f.readline()
        n_genes = int(line.split()[1])
        for ch in pd.read_csv(f, sep=" ", header=None, names=["c", "g", "v"], chunksize=5_000_000):
            ch = ch[ch.c.isin(keep)]
            cells.append(ch.c.to_numpy())
            genes.append(ch.g.to_numpy() - 1)
            vals.append(np.log1p(ch.v.to_numpy()))
    c, g, v = (np.concatenate(x) for x in (cells, genes, vals))
    order = np.array(sorted(keep))
    row = np.searchsorted(order, c)
    n = len(order)
    mean = np.bincount(g, v, n_genes) / n
    var = np.bincount(g, v**2, n_genes) / n - mean**2
    hvg = np.argsort(-var)[:N_HVG]
    col = np.full(n_genes, -1)
    col[hvg] = np.arange(N_HVG)
    sel = col[g] >= 0
    x = np.zeros((n, N_HVG), np.float32)
    x[row[sel], col[g[sel]]] = v[sel]
    return clone[order], x


def outcomes(cnt, conds):
    p = {c: pooled(cnt[c]) for c in conds}
    return {c: {k: (v[0], v[0] + v[1]) for k, v in p[c].items() if v[0] + v[1] >= 1} for c in conds}


def e38_latent(clones, prm, rng, meas):
    """Mirror of e37.simulate returning expected N and M counts per clone and condition (the
    latent rates before Poisson sampling), so the predictive is not double-sampled."""
    n = len(clones)
    m, fn, fm = meas[:, 0], meas[:, 1], meas[:, 2]
    w = prm["w"]
    z = w * m + np.sqrt(1 - w**2) * rng.standard_normal(n)
    a = prm["sigma_a"] * rng.standard_normal(n)
    ra = prm["rho_a"]
    a_full = ra * a + np.sqrt(1 - ra**2) * prm["sigma_a"] * rng.standard_normal(n)
    cn = np.clip(prm["kappa"] * fn, 0, 1)
    cm = np.clip(prm["kappa"] * fm, 0, 1 - cn)
    free = 1 - cn - cm
    u = rng.random(n) < prm["phi"]
    eps = prm["eps"]
    out = {}
    for c in e37.CONDS:
        full = c == "full"
        rate = prm["nu_full"] * np.where(u, 1.0, eps) if full else prm["nu"] * np.where(u, eps, 1.0)
        k = rng.poisson(rate)
        zb = np.sqrt(e37.RHO_F) * z + np.sqrt(1 - e37.RHO_F) * rng.standard_normal(n)
        p = 1 / (1 + np.exp(-(prm["beta"] * zb + prm["tau"][c])))
        lam = k * np.exp(np.clip(prm["mu"][c] + (a_full if full else a), -30, 8))
        ln = np.clip(lam * (free * p + cn) * prm["s"][c], 0, 1e4)
        lm = np.clip(lam * (free * (1 - p) + cm), 0, 1e4)
        out[c] = (ln, lm)
    return out


def e38_pred(cnt, conds, nsim=300):
    ref = b24.best_e38()
    clones = e26.universe(cnt)
    meas = e33.measured(clones, np.random.default_rng(77))
    sims = [
        e38_latent(clones, ref["params"], np.random.default_rng(5000 + s), meas)
        for s in range(nsim)
    ]
    idx = {int(k): i for i, k in enumerate(clones)}
    pred = {
        c: (np.stack([s[c][0] for s in sims]), np.stack([s[c][1] for s in sims])) for c in conds
    }
    m3 = {int(k): meas[i] for i, k in enumerate(clones)}
    return (pred, idx), m3, f"e38_s{ref['seed']}"


# ---------- encoders ----------
def pca_fit(x, k=N_PC):
    mu = x.mean(0)
    _, _, vt = np.linalg.svd(x - mu, full_matrices=False)
    return mu, vt[:k]


class Net(torch.nn.Module):
    def __init__(self, d, h=128, e=64, k=64):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(d, h), torch.nn.GELU(), torch.nn.Linear(h, e)
        )
        self.head = torch.nn.Sequential(torch.nn.GELU(), torch.nn.Linear(e, k))

    def forward(self, x):
        return self.head(self.backbone(x))


def dino_sister(z, cl, seed, epochs=30, batch=256):
    """DINO with two sister cells (same clone) as views; returns the teacher backbone."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    zt = torch.tensor(z, dtype=torch.float32)
    sd = zt.std(0, keepdim=True)
    by = {}
    for i, c in enumerate(cl):
        by.setdefault(int(c), []).append(i)
    partner = np.array([by[int(c)] for c in cl], dtype=object)
    rng = np.random.default_rng(seed)
    student, teacher = Net(z.shape[1]), Net(z.shape[1])
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(student.parameters(), 1e-3)
    centre = torch.zeros(64)
    n = len(z)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g).numpy()
        for b in range(0, n, batch):
            i = perm[b : b + batch]
            j = np.array([rng.choice([q for q in partner[a] if q != a] or [a]) for a in i])
            v1 = zt[i] + 0.1 * sd * torch.randn(len(i), z.shape[1], generator=g)
            v2 = zt[j] + 0.1 * sd * torch.randn(len(i), z.shape[1], generator=g)
            loss = 0.0
            for s_in, t_in in ((v1, v2), (v2, v1)):
                with torch.no_grad():
                    t = torch.softmax((teacher(t_in) - centre) / 0.04, -1)
                s = torch.log_softmax(student(s_in) / 0.1, -1)
                loss = loss - (t * s).sum(-1).mean() / 2
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                for ps, pt in zip(student.parameters(), teacher.parameters(), strict=True):
                    pt.mul_(0.996).add_(0.004 * ps)
                centre = 0.9 * centre + 0.1 * teacher(torch.cat([v1, v2])).mean(0)
    teacher.eval()
    return lambda a: teacher.backbone(torch.tensor(a, dtype=torch.float32)).detach().numpy()


# ---------- predictor ----------
def ridge_logit(x, k, n, lam=0.01):
    mu, sd = x.mean(0), x.std(0) + 1e-6
    xt = torch.tensor((x - mu) / sd, dtype=torch.float32)
    y = torch.tensor(k / n, dtype=torch.float32)
    w8 = torch.tensor(n / n.sum(), dtype=torch.float32)
    w = torch.zeros(x.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = xt @ w + b
        loss = (
            w8 * torch.nn.functional.binary_cross_entropy_with_logits(z, y, reduction="none")
        ).sum()
        loss = loss + lam * (w**2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    wn, bn = w.detach().numpy(), float(b.detach())
    return lambda a: 1 / (1 + np.exp(-(((a - mu) / sd) @ wn + bn)))


def dev(p, k, n):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return -(k * np.log(p) + (n - k) * np.log(1 - p))  # per clone, summed over its cells


def dev_mix(e38, c, q, k, n):
    """E38's native predictive (amendment 2026-11-04): over simulations, weight each by the
    Poisson probability of the clone's observed total n, then average Binom(k | n, q_s) with
    q_s = lamN / (lamN + lamM). Same constant convention as dev()."""
    (pred, idx) = e38
    if q not in idx:
        return dev(0.5, k, n)
    ln, lm = pred[c][0][:, idx[q]], pred[c][1][:, idx[q]]
    tot = ln + lm
    ok = tot > 0
    if not ok.any():
        return dev(0.5, k, n)
    ln, tot = ln[ok], tot[ok]
    lw = n * np.log(tot) - tot  # Poisson log-weight up to a constant
    qs = np.clip(ln / tot, 1e-3, 1 - 1e-3)
    ll = k * np.log(qs) + (n - k) * np.log(1 - qs)
    a = lw + ll
    return -(np.logaddexp.reduce(a) - np.logaddexp.reduce(lw))


# ---------- main ----------
def main():
    cnt = e33.load_counts(e33.CONDS5)
    conds = PRIMARY + REPORT
    y = outcomes(cnt, conds)
    e38p, m3, ref = e38_pred(cnt, conds)
    cell_clone, x = day2_matrix()
    clones = np.array(sorted(set(cell_clone.tolist())))
    rng = np.random.default_rng(0)
    fold = dict(zip(clones.tolist(), rng.permutation(np.arange(len(clones)) % FOLDS), strict=True))
    rows = []  # (cond, clone, k, n, {model: p})
    for f in range(FOLDS):
        tr_cells = np.array([fold[int(c)] != f for c in cell_clone])
        mu, vt = pca_fit(x[tr_cells])
        z = (x - mu) @ vt.T
        enc_dino = dino_sister(z[tr_cells], cell_clone[tr_cells], seed=f)
        e = enc_dino(z)
        feats = {}
        for name, mat in (("PCA", z), ("DINO-sister", e)):
            df = pd.DataFrame(mat).groupby(cell_clone).mean()
            feats[name] = {int(k): v for k, v in zip(df.index, df.to_numpy(), strict=True)}
        feats["m3"] = {int(k): m3[int(k)] for k in clones if int(k) in m3}
        for c in conds:
            ks = [k for k in clones.tolist() if k in y[c]]
            tr = [k for k in ks if fold[k] != f]
            te = [k for k in ks if fold[k] == f]
            kk = {k: np.array([y[c][q][0] for q in s], float) for k, s in (("tr", tr), ("te", te))}
            nn = {k: np.array([y[c][q][1] for q in s], float) for k, s in (("tr", tr), ("te", te))}
            null = kk["tr"].sum() / nn["tr"].sum()
            preds = {"null": np.full(len(te), null)}
            for name, ft in feats.items():
                if not tr or not te:
                    continue
                model = ridge_logit(np.array([ft[q] for q in tr]), kk["tr"], nn["tr"])
                preds[name] = model(np.array([ft[q] for q in te]))
            for i, q in enumerate(te):
                k1, n1 = kk["te"][i], nn["te"][i]
                d = {m: float(dev(v[i], k1, n1)) for m, v in preds.items()}
                d["E38"] = float(dev_mix(e38p, c, q, k1, n1))
                rows.append((c, q, k1, n1, d))
        print(f"fold {f} done", flush=True)
    models = ["null", "m3", "PCA", "DINO-sister", "E38"]
    rep = {
        "reference": ref,
        "n_day2_cells": len(cell_clone),
        "n_clones": len(clones),
        "per_cond": {},
    }
    use = []
    for c in conds:
        r = [t for t in rows if t[0] == c]
        nc = len({t[1] for t in r})
        rep["per_cond"][c] = {"clones": nc, "cells": float(sum(t[3] for t in r))}
        if nc >= MIN_CLONES:
            for mname in models:
                d = sum(t[4][mname] for t in r) / sum(t[3] for t in r)
                rep["per_cond"][c][mname] = float(d)
            if c in PRIMARY:
                use.append(c)
    prim = [t for t in rows if t[0] in use]
    cl = sorted({t[1] for t in prim})
    idx = {q: i for i, q in enumerate(cl)}
    D = np.zeros((len(cl), len(models)))
    N = np.zeros(len(cl))
    for t in prim:
        i = idx[t[1]]
        N[i] += t[3]
        for j, mname in enumerate(models):
            D[i, j] += t[4][mname]
    rep["primary_conds"] = use
    rep["primary_dev_per_cell"] = {m: float(D[:, j].sum() / N.sum()) for j, m in enumerate(models)}
    boot = np.random.default_rng(1)
    bi = [boot.integers(0, len(cl), len(cl)) for _ in range(2000)]

    def diff(a, b):
        ja, jb = models.index(a), models.index(b)
        v = np.array([(D[i, ja].sum() - D[i, jb].sum()) / N[i].sum() for i in bi])
        est = (D[:, ja].sum() - D[:, jb].sum()) / N.sum()
        return {
            "est": float(est),
            "ci": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
        }

    best = min(("PCA", "DINO-sister"), key=lambda m: rep["primary_dev_per_cell"][m])
    rep["R1_best_learned_minus_E38"] = {"best": best, **diff(best, "E38")}
    rep["R2_m3_minus_E38"] = diff("m3", "E38")
    rep["R3_DINO_minus_PCA"] = diff("DINO-sister", "PCA")
    rep["learned_minus_null"] = {m: diff(m, "null") for m in ("m3", "PCA", "DINO-sister", "E38")}
    json.dump(rep, open(os.path.join(OUT, "s2_action_predictor.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
