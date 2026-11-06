"""E54: does the measured RNA axis carry the clone state the sister fates imply? SIS-seq on the
core model (state + sampling steps). See EXPERIMENT_LOG E54.

    uv run python core_sisseq.py recover   # R4: estimator recovery on the core simulator
    uv run python core_sisseq.py fit       # fit-set estimate of a_f, sigma_f, s_eps
    uv run python core_sisseq.py predict   # held-out predictive ranges (before the look)

The held-out scores are built only inside `scores("heldout")`, called by test_oracle.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os

import numpy as np
import pandas as pd

import b41_rna_fate as b41
import core_model as cm
import sis_data as sd

OUT = os.path.join(os.path.dirname(__file__), "results_sis")
FATES = sd.FATES


def _cells_sorted():
    meta, t = {}, None
    for line in gzip.open(sd.SEQ_SOFT, "rt", encoding="utf-8", errors="replace"):
        if line.startswith("!Sample_title"):
            t = line.split("= ", 1)[1].strip()
        elif line.startswith("!Sample_characteristics_ch1 = cells_sorted"):
            meta[t] = float(line.split(": ", 1)[1])
    return meta


def _clones(part, g, lib, meta):
    d = sd.load_seq(part)
    ok = ((d["k1"].sum(1) > 3) & (d["k2"].sum(1) > 3)
          & np.array([t in g.columns and lib.get(t, 0) >= 5000 and meta.get(t, 0) >= 1 for t in d["title"]]))
    return {k: v[ok] for k, v in d.items()}


def logits(k, batch):
    n = k.sum(1, keepdims=True)
    return b41.centre(np.log((k + 0.5) / (n - k + 0.5)), batch)


def scores(part="fit"):
    """RNA scores r (clones x 3) defined on the fit set: fit set = leave-one-out predictions;
    held-out = projection with fit-set means, SDs, PC loadings and weights."""
    g = pd.read_csv(b41.GENES, index_col=0)
    lib, meta = g.sum(0), _cells_sorted()
    F = _clones("fit", g, lib, meta)
    C = g[F["title"]].to_numpy(float).T
    det = (C > 0).mean(0) >= 0.10
    E = np.log1p(C / C.sum(1, keepdims=True) * 1e4)[:, det]
    stats = {}
    Z = np.zeros_like(E)
    for b in np.unique(F["batch"]):
        m = F["batch"] == b
        mu, sdv = E[m].mean(0), E[m].std(0)
        sdv = np.where(sdv > 0, sdv, 1.0)
        stats[b] = (mu, sdv)
        Z[m] = (E[m] - mu) / sdv
    top = np.argsort(Z.var(0))[::-1][:2000]
    cmu = Z[:, top].mean(0)
    _, _, Vt = np.linalg.svd(Z[:, top] - cmu, full_matrices=False)
    V = Vt[:3].T
    pcs = (Z[:, top] - cmu) @ V
    L = logits(F["k1"] + F["k2"], F["batch"])
    X1 = np.column_stack([np.ones(len(L)), pcs])
    W = np.linalg.lstsq(X1, L, rcond=None)[0]
    if part == "fit":
        r = np.empty_like(L)
        for i in range(len(L)):
            m = np.arange(len(L)) != i
            r[i] = X1[i] @ np.linalg.lstsq(X1[m], L[m], rcond=None)[0]
        return F, r
    H = _clones("heldout", g, lib, meta)
    Ch = g[H["title"]].to_numpy(float).T
    Eh = np.log1p(Ch / Ch.sum(1, keepdims=True) * 1e4)[:, det]
    Zh = np.zeros_like(Eh)
    for i, b in enumerate(H["batch"]):
        mu, sdv = stats[b]
        Zh[i] = (Eh[i] - mu) / sdv
    r = np.column_stack([np.ones(len(Zh)), (Zh[:, top] - cmu) @ V]) @ W
    return H, r


def estimate(r, l1, l2):
    """sigma_f = sqrt(cov(l1, l2)); a_f = cov(r_f, mean l) / (sd(r_f) sigma_f)."""
    lp = (l1 + l2) / 2
    sig = np.sqrt(np.maximum([np.cov(l1[:, j], l2[:, j])[0, 1] for j in range(3)], 1e-9))
    a = np.array([np.cov(r[:, j], lp[:, j])[0, 1] / (r[:, j].std(ddof=1) * sig[j]) for j in range(3)])
    return a, sig


def simulate(a, sig, mu, s_eps, n1, n2, r, rng):
    """Core state + sampling: z_f = mu_f + sig_f (a_f r_f + sqrt(1 - a_f^2) xi_f); each well
    Poisson per fate at its observed size from softmax(z + s_eps noise)."""
    a = np.clip(a, -0.999, 0.999)
    z = mu + sig * (a * r + np.sqrt(1 - a**2) * rng.standard_normal(r.shape))
    out = []
    for n in (n1, n2):
        p = cm.softmax(z + s_eps * rng.standard_normal(z.shape))
        out.append(cm.emit([(n.astype(float), p, None)], rng))
    return out


def standardise(r):
    return (r - r.mean(0)) / r.std(0)


def calibrate_noise(F, sig, mu, rng, grid=np.linspace(0.0, 2.0, 21)):
    """s_eps matching the observed mean var(l1 - l2) across fates (sigma, mu held at estimates)."""
    l1, l2 = logits(F["k1"], F["batch"]), logits(F["k2"], F["batch"])
    target = np.var(l1 - l2, axis=0).mean()
    n1, n2 = F["k1"].sum(1), F["k2"].sum(1)
    best, bd = 0.0, np.inf
    for s in grid:
        v = []
        for _ in range(20):
            r = rng.standard_normal((len(n1), 3))
            k1, k2 = simulate(np.zeros(3), sig, mu, s, n1, n2, r, rng)
            b = np.zeros(len(n1), dtype=object)
            v.append(np.var(logits(k1, b) - logits(k2, b), axis=0).mean())
        if abs(np.mean(v) - target) < bd:
            best, bd = float(s), abs(np.mean(v) - target)
    return best


def base_params():
    F, r = scores("fit")
    l1, l2 = logits(F["k1"], F["batch"]), logits(F["k2"], F["batch"])
    frac = (F["k1"] + F["k2"]).sum(0) / (F["k1"] + F["k2"]).sum()
    mu = np.log(frac) - np.log(frac[0])
    return F, standardise(r), l1, l2, mu


def recover(n_test=300, seed=54, boot=300):
    rng = np.random.default_rng(seed)
    F, r, l1, l2, mu = base_params()
    _, sig = estimate(r, l1, l2)
    s_eps = calibrate_noise(F, sig, mu, rng)
    n1, n2, b = F["k1"].sum(1), F["k2"].sum(1), np.zeros(len(r), dtype=object)
    rows = []
    for _ in range(n_test):
        a = rng.uniform(-0.9, 0.9, 3)
        rr = rng.standard_normal(r.shape)
        k1, k2 = simulate(a, sig, mu, s_eps, n1, n2, rr, rng)
        s1, s2 = logits(k1, b), logits(k2, b)
        est = estimate(rr, s1, s2)[0]
        bs = np.array([estimate(rr[ix], s1[ix], s2[ix])[0] for ix in rng.integers(0, len(rr), (boot, len(rr)))])
        lo, hi = np.nanpercentile(bs, [2.5, 97.5], axis=0)
        rows.append((a, est, lo, hi))
    A = np.array([x[0] for x in rows])
    E = np.array([x[1] for x in rows])
    cov = np.mean([(x[2] <= x[0]) & (x[0] <= x[3]) for x in rows], axis=0)
    res = {"s_eps": s_eps, "sigma": sig.tolist(), "mu": mu.tolist()}
    for j, f in enumerate(FATES):
        rmse = float(np.sqrt(np.nanmean((E[:, j] - A[:, j]) ** 2)))
        res[f] = {"RMSE": rmse, "coverage95": float(cov[j]),
                  "reading": "recoverable" if rmse <= 0.2 and cov[j] >= 0.90 else "not recoverable"}
    json.dump(res, open(os.path.join(OUT, "r4_recovery.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


def fit(seed=55, boot=2000):
    rng = np.random.default_rng(seed)
    F, r, l1, l2, mu = base_params()
    a, sig = estimate(r, l1, l2)
    bs = [estimate(r[ix], l1[ix], l2[ix]) for ix in rng.integers(0, len(r), (boot, len(r)))]
    A = np.array([x[0] for x in bs])
    S = np.array([x[1] for x in bs])
    s_eps = json.load(open(os.path.join(OUT, "r4_recovery.json")))["s_eps"]
    res = {"n_clones": int(len(r)), "s_eps": s_eps, "mu": mu.tolist()}
    for j, f in enumerate(FATES):
        lo, hi = np.nanpercentile(A[:, j], [2.5, 97.5])
        q = np.nanpercentile(A[:, j] ** 2, [2.5, 97.5])
        res[f] = {"a": float(a[j]), "a_ci": [float(lo), float(hi)], "a2": float(a[j] ** 2),
                  "a2_ci": [float(q[0]), float(q[1])], "sigma": float(sig[j])}
    np.savez_compressed(os.path.join(OUT, "e54_boot.npz"), A=A, S=S)
    json.dump(res, open(os.path.join(OUT, "e54_fit.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


def predict(seed=56, n_draws=1000):
    rng = np.random.default_rng(seed)
    F, r, l1, l2, mu = base_params()
    fitres = json.load(open(os.path.join(OUT, "e54_fit.json")))
    z = np.load(os.path.join(OUT, "e54_boot.npz"))
    n_ho = int(round(len(r) / 2))  # expected held-out size (shard 0 = half the fit set)
    n1, n2 = F["k1"].sum(1), F["k2"].sum(1)
    cors = []
    for i in rng.integers(0, len(z["A"]), n_draws):
        ix = rng.integers(0, len(r), n_ho)
        rr = r[ix]
        k1, k2 = simulate(z["A"][i], z["S"][i], mu, fitres["s_eps"], n1[ix], n2[ix], rr, rng)
        b = np.zeros(n_ho, dtype=object)
        lp = (logits(k1, b) + logits(k2, b)) / 2
        cors.append([np.corrcoef(rr[:, j], lp[:, j])[0, 1] for j in range(3)])
    cors = np.array(cors)
    res = {"n_expected": n_ho,
           "corr_95": {f: np.nanpercentile(cors[:, j], [2.5, 97.5]).tolist() for j, f in enumerate(FATES)}}
    json.dump(res, open(os.path.join(OUT, "e54_heldout_prediction.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


def heldout_stat():
    """Called by test_oracle only: corr(r_f, pooled logit_f) on held-out clones."""
    H, r = scores("heldout")
    lp = (logits(H["k1"], H["batch"]) + logits(H["k2"], H["batch"])) / 2
    return {"n": int(len(r)), "corr": {f: float(np.corrcoef(r[:, j], lp[:, j])[0, 1]) for j, f in enumerate(FATES)}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["recover", "fit", "predict"])
    a = ap.parse_args()
    {"recover": recover, "fit": fit, "predict": predict}[a.cmd]()
