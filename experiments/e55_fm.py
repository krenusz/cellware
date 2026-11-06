"""E55: do pretrained single-cell RNA models (Geneformer) read more of the clone fate state than
PCA? LARRY in vitro, day-2 cells -> clone's later N / (N + M). See EXPERIMENT_LOG E55.

    uv run --with torch --with transformers --with safetensors python e55_fm.py embed V1-10M
    uv run --with torch --with transformers --with safetensors python e55_fm.py embed V2-316M
    uv run python e55_fm.py cv          # fit-shard nested CV, names the best foundation candidate
    uv run python e55_fm.py sis         # SIS-seq exploratory (needs `embed-sis` first)

Held-out clones are scored only inside `heldout_stat()`, called by test_oracle.
Downloaded weights and dictionaries are data: pickles are read with a whitelist unpickler.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np
import scipy.sparse

import test_oracle as to
import track_b_state as tb

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "data")
GF = os.path.join(ROOT, "fm_weights", "geneformer")
ORTHO = os.path.join(ROOT, "orthologs", "mouse_human_orthologs.tsv")
OUT = os.path.join(HERE, "results_e55")
MODELS = {"V1-10M": ("gc30M", 2048, False), "V2-316M": ("gc104M", 4096, True)}
CANDIDATES = ["gf_V1-10M", "gf_V2-316M"]
LAMBDAS = 10.0 ** np.arange(-2, 5)


# ------------------------------------------------------------------ safe loading

class _SafeUnpickler(pickle.Unpickler):
    OK = {("numpy", "dtype"), ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
          ("builtins", "dict"), ("builtins", "set"), ("collections", "OrderedDict")}

    def find_class(self, module, name):
        if (module, name) in self.OK:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


def load_pkl(name):
    with open(os.path.join(GF, "dict", name), "rb") as f:
        return _SafeUnpickler(f).load()


def mouse_to_human():
    """Mouse symbol -> human Ensembl ID, one-to-one orthologs only."""
    m = {}
    with open(ORTHO, encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) == 4 and p[3] == "ortholog_one2one" and p[1] and p[2]:
                m.setdefault(p[1], p[2])
    return m


# ------------------------------------------------------------------ LARRY data

def larry():
    """Day-2 normalised counts (cells x genes, CSR), gene symbols, and clone table."""
    import gzip
    day, ctype, _, cid = tb.load()
    t = np.loadtxt(os.path.join(tb.DATA, "day2_normed_triplets.txt.gz"))
    X = scipy.sparse.csr_matrix((t[:, 2], (t[:, 0].astype(int) - 1, t[:, 1].astype(int) - 1)),
                                shape=(len(day), int(t[:, 1].max())))
    with gzip.open(os.path.join(tb.DATA, "stateFate_inVitro_gene_names.txt.gz"), "rt") as f:
        genes = [g.strip() for g in f][: X.shape[1]]
    late = (day > 2) & (cid >= 0)
    nc = cid.max() + 1
    n = np.bincount(cid[late & (ctype == "Neutrophil")], minlength=nc)
    m = np.bincount(cid[late & (ctype == "Monocyte")], minlength=nc)
    d2 = (day == 2) & (cid >= 0)
    n2 = np.bincount(cid[d2], minlength=nc)
    clones = np.flatnonzero((n + m >= 3) & (n2 >= 1))
    y = n[clones] / (n[clones] + m[clones])
    tot = (n + m)[clones]
    cells = np.flatnonzero(d2 & np.isin(cid, clones))
    return X, genes, cid, clones, y, tot, cells


def split(clones):
    ho = np.array([to.shard_of(f"e55:{c}", 3) == 0 for c in clones])
    return ~ho, ho


# ------------------------------------------------------------------ Geneformer

def tokenize(x, genes, model):
    """Rank-value tokens for one cell: expression / gene median, descending, nonzero only."""
    tag, size, special = MODELS[model]
    tok, med = load_pkl(f"token_dictionary_{tag}.pkl"), load_pkl(f"gene_median_dictionary_{tag}.pkl")
    m2h = mouse_to_human()
    idx, tid, mv = [], [], []
    for j, g in enumerate(genes):
        h = m2h.get(g)
        if h in tok and h in med:
            idx.append(j), tid.append(tok[h]), mv.append(float(med[h]))
    return np.array(idx), np.array(tid), np.array(mv), size, special, tok


def embed(model, rows=None, X=None, genes=None, out=None, batch=4):
    import torch
    from transformers import BertConfig, BertModel
    from safetensors.torch import load_file
    torch.set_num_threads(max(1, os.cpu_count() - 1))
    if X is None:
        X, genes, cid, clones, y, tot, rows = larry()
    idx, tid, mv, size, special, tok = tokenize(None, genes, model)
    cfg = BertConfig.from_json_file(os.path.join(GF, model, "config.json"))
    net = BertModel(cfg, add_pooling_layer=False)
    sd = {k.removeprefix("bert."): v for k, v in load_file(os.path.join(GF, model, "model.safetensors")).items()
          if k.startswith("bert.")}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    assert not [k for k in missing if "position_ids" not in k], missing
    net.eval()
    Xs = X[rows][:, idx].toarray()
    embs = np.zeros((len(rows), cfg.hidden_size), dtype=np.float32)
    seqs = []
    for x in Xs:
        v = x / max(x.sum(), 1e-9) * 1e4 / mv
        nz = np.flatnonzero(v > 0)
        order = nz[np.argsort(-v[nz], kind="stable")]
        ids = tid[order].tolist()
        if special:
            ids = [tok["<cls>"]] + ids[: size - 2] + [tok["<eos>"]]
        else:
            ids = ids[:size]
        seqs.append(ids)
    lens = np.array([len(s) for s in seqs])
    print(f"{model}: {len(seqs)} cells, tokens per cell median {np.median(lens):.0f}", flush=True)
    with torch.no_grad():
        for b in range(0, len(seqs), batch):
            ch = seqs[b:b + batch]
            L = max(len(s) for s in ch)
            ids = torch.zeros((len(ch), L), dtype=torch.long)
            att = torch.zeros((len(ch), L), dtype=torch.long)
            for i, s in enumerate(ch):
                ids[i, :len(s)] = torch.tensor(s)
                att[i, :len(s)] = 1
            hs = net(input_ids=ids, attention_mask=att, output_hidden_states=True).hidden_states[-2]
            for i, s in enumerate(ch):
                a, e = (1, len(s) - 1) if special else (0, len(s))  # genes only: drop <cls>, <eos>
                embs[b + i] = hs[i, a:e].mean(0).numpy()
            if b % 100 == 0:
                print(f"  {b}/{len(seqs)}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(out or os.path.join(OUT, f"emb_{model}.npz"), rows=rows, emb=embs)


# ------------------------------------------------------------------ features and prediction

def clone_mean(F, rows, cid, clones):
    pos = {r: i for i, r in enumerate(rows)}
    return np.stack([F[[pos[r] for r in rows if cid[r] == c]].mean(0) for c in clones])


def pca_feats(X, cells, cid, tr_clones, all_clones, k=50):
    """PCA-50 fitted on day-2 cells of training clones only; all clones projected."""
    xs = X[cells].copy()
    xs.data = np.log1p(xs.data)
    trm = np.isin(cid[cells], tr_clones)
    a = xs[trm]
    mu = np.asarray(a.mean(0)).ravel()
    var = np.asarray(a.multiply(a).mean(0)).ravel() - mu**2
    top = np.argsort(var)[-2000:]
    A = a[:, top].toarray() - mu[top]
    _, _, vt = np.linalg.svd(A, full_matrices=False)
    P = (xs[:, top].toarray() - mu[top]) @ vt[:k].T
    return clone_mean(P, cells, cid, all_clones)


def marker_feats(X, genes, cells, cid, clones):
    g = {s: j for j, s in enumerate(genes)}
    v = np.log1p(X[cells][:, [g["Cebpa"], g["Irf8"]]].toarray())
    v = (v - v.mean(0)) / (v.std(0) + 1e-9)
    return clone_mean((v[:, 0] - v[:, 1])[:, None], cells, cid, clones)


def ridge_fit(A, y, lam):
    mu, sd = A.mean(0), A.std(0) + 1e-9
    Z = np.c_[(A - mu) / sd, np.ones(len(A))]
    pen = lam * np.eye(Z.shape[1])
    pen[-1, -1] = 0
    w = np.linalg.solve(Z.T @ Z + pen, Z.T @ y)
    return lambda B: np.c_[(B - mu) / sd, np.ones(len(B))] @ w


def nested_ridge(A, y, B, seed=0):
    """Choose lambda by inner 5-fold CV on (A, y), fit on all of A, predict B."""
    f = np.random.default_rng(seed).integers(0, 5, len(y))
    err = []
    for lam in LAMBDAS:
        p = np.zeros_like(y)
        for k in range(5):
            p[f == k] = ridge_fit(A[f != k], y[f != k], lam)(A[f == k])
        err.append(((p - y) ** 2).mean())
    return ridge_fit(A, y, LAMBDAS[int(np.argmin(err))])(B)


def features(name, X, genes, cells, cid, clones, tr_clones, embs):
    if name == "pca50":
        return pca_feats(X, cells, cid, tr_clones, clones)
    if name == "marker":
        return marker_feats(X, genes, cells, cid, clones)
    if name.startswith("gf_"):
        z = embs[name]
        return clone_mean(z["emb"], z["rows"], cid, clones)
    if name.startswith("pca50+"):
        return np.c_[features("pca50", X, genes, cells, cid, clones, tr_clones, embs),
                     features(name[6:], X, genes, cells, cid, clones, tr_clones, embs)]
    raise ValueError(name)


def load_embs():
    return {f"gf_{m}": np.load(os.path.join(OUT, f"emb_{m}.npz")) for m in MODELS
            if os.path.exists(os.path.join(OUT, f"emb_{m}.npz"))}


def ceiling(y, tot):
    """Share of target variance not due to binomial sampling of later cells."""
    return float(1 - np.mean(y * (1 - y) / np.maximum(tot - 1, 1)) / y.var())


def cv(seed=55):
    X, genes, cid, clones, y, tot, cells = larry()
    fit, ho = split(clones)
    embs = load_embs()
    cf, yf = clones[fit], y[fit]
    folds = np.random.default_rng(seed).integers(0, 5, len(cf))
    res = {"n_fit": int(fit.sum()), "n_heldout_expected": int(ho.sum()), "ceiling_R2": ceiling(yf, tot[fit])}
    for name in ["pca50", "marker"] + [c for c in CANDIDATES if c in embs]:
        p = np.zeros_like(yf)
        for k in range(5):
            tr, te = folds != k, folds == k
            F = features(name, X, genes, cells, cid, cf, cf[tr], embs)
            p[te] = nested_ridge(F[tr], yf[tr], F[te])
        res[name] = {"cv_r": float(np.corrcoef(p, yf)[0, 1]), "R2": float(1 - ((p - yf) ** 2).mean() / yf.var())}
    fm = [c for c in CANDIDATES if c in res]
    res["best_foundation"] = max(fm, key=lambda c: res[c]["cv_r"]) if fm else None
    json.dump(res, open(os.path.join(OUT, "e55_cv.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


def heldout_stat(n_boot=5000, seed=56):
    """Called by test_oracle only."""
    X, genes, cid, clones, y, tot, cells = larry()
    fit, ho = split(clones)
    embs = load_embs()
    best = json.load(open(os.path.join(OUT, "e55_cv.json")))["best_foundation"]
    names = ["pca50", "marker", best, f"pca50+{best}"]
    pred = {}
    for nm in names:
        F = features(nm, X, genes, cells, cid, clones, clones[fit], embs)
        pred[nm] = nested_ridge(F[fit], y[fit], F[ho])
    yh = y[ho]
    r = lambda p, ix=slice(None): float(np.corrcoef(p[ix], yh[ix])[0, 1])  # noqa: E731
    rng = np.random.default_rng(seed)
    B = rng.integers(0, len(yh), (n_boot, len(yh)))
    d1 = np.array([r(pred[best], b) - r(pred["pca50"], b) for b in B])
    d2 = np.array([r(pred[f"pca50+{best}"], b) - r(pred["pca50"], b) for b in B])
    return {"n": int(ho.sum()), "best": best, "r": {k: r(v) for k, v in pred.items()},
            "delta_best_minus_pca": [float(np.mean(d1)), *np.percentile(d1, [2.5, 97.5]).tolist()],
            "delta_concat_minus_pca": [float(np.mean(d2)), *np.percentile(d2, [2.5, 97.5]).tolist()],
            "ceiling_R2": ceiling(yh, tot[ho])}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["embed", "cv"])
    ap.add_argument("model", nargs="?")
    a = ap.parse_args()
    embed(a.model) if a.cmd == "embed" else cv()
