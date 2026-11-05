"""Chapter 2 data: SIS-seq (Tian et al. 2021, GEO GSE119097) and SIS-skew (Mendeley vj9tskkvks,
ST072). Loaders apply the H2 held-out split: shard 0 of 3 is held out, shards 1-2 are the fit set.
Also the shared sister statistics used by B38 and B39.
"""

from __future__ import annotations

import gzip
import os

import numpy as np
import pandas as pd

from test_oracle import shard_of

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "sis_seq")
SEQ_SOFT = os.path.join(ROOT, "geo", "family.soft.gz")
SKEW_DIR = os.path.join(ROOT, "fig5")
FATES = ("cDC1", "cDC2", "pDC")
SHARDS, HELD_OUT = 3, 0


def _keep(keys, part):
    sh = np.array([shard_of(k, SHARDS) for k in keys])
    if part == "fit":
        return sh != HELD_OUT
    if part == "heldout":
        return sh == HELD_OUT
    return np.ones(len(keys), bool)


def load_seq(part="fit"):
    """One row per SIS-seq clone: k1, k2 (n x 3 integer counts in the two sister wells), bg1, bg2
    (filler-cell fate fractions per well), batch, title."""
    recs, cur = [], None
    for line in gzip.open(SEQ_SOFT, "rt", encoding="utf-8", errors="replace"):
        line = line.rstrip("\n")
        if line.startswith("^SAMPLE"):
            cur = {}
            recs.append(cur)
        elif cur is not None and line.startswith("!Sample_title"):
            cur["title"] = line.split("= ", 1)[1]
        elif cur is not None and line.startswith("!Sample_characteristics_ch1"):
            k, v = line.split("= ", 1)[1].split(": ", 1)
            cur[k] = v
    df = pd.DataFrame(recs)
    df["batch"] = df.title.str.split("_").str[0]
    df = df[_keep(["seq:" + t for t in df.title], part)].reset_index(drop=True)
    num = lambda cols: df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()  # noqa: E731
    out = {"title": df.title.to_numpy(), "batch": df.batch.to_numpy()}
    for s in ("1", "2"):
        out["k" + s] = np.rint(num([f"{f.lower()}_sis{s}" for f in FATES])).astype(int)
        out["bg" + s] = num([f"{f.lower()}_fill_sis{s}" for f in FATES])
    return out


def load_skew(part="fit", wells=("W1", "W2", "C1", "C2", "C3")):
    """One row per SIS-skew clone (barcode within well), authors' pipeline: reads -> cells with
    sorted cell numbers and Cherry adjustment, both PCR replicates required, replicates summed,
    clone size > 1. Returns well, id, P (Cherry+ cells x 3), N (Cherry- cells x 3)."""
    a = pd.read_csv(os.path.join(SKEW_DIR, "ST072_Counts.txt"), sep="\t", index_col=0)
    cn = pd.read_csv(os.path.join(SKEW_DIR, "ST072_Sorted_cell_number.txt"), sep="\t",
                     index_col=0)["Cell_number"].astype(float)
    cn[cn.index.str.contains("pos")] /= 0.5
    cn[cn.index.str.contains("neg")] /= 1.5
    wc = a[[c for c in a.columns if c.startswith(("ST072_W", "ST072_C"))]]
    wc = wc / (2 * wc.sum() / cn[wc.columns])
    A = wc[[c for c in wc.columns if c.endswith("_a")]]
    B = wc[[c for c in wc.columns if c.endswith("_b")]]
    keep = (A.to_numpy() > 0) & (B.to_numpy() > 0)
    AB = pd.DataFrame((A.to_numpy() + B.to_numpy()) * keep, index=A.index,
                      columns=[c[:-2] for c in A.columns])
    rows = []
    for w in wells:
        P = AB[[f"ST072_{w}_Cherry_pos_{f}" for f in FATES]].to_numpy()
        N = AB[[f"ST072_{w}_Cherry_neg_{f}" for f in FATES]].to_numpy()
        m = (P.sum(1) + N.sum(1)) > 1
        for i in np.flatnonzero(m):
            rows.append((w, AB.index[i], P[i], N[i]))
    rows = [r for r, k in zip(rows, _keep([f"skew:{r[0]}:{r[1]}" for r in rows], part)) if k]
    return {"well": np.array([r[0] for r in rows]), "id": np.array([r[1] for r in rows]),
            "P": np.array([r[2] for r in rows]), "N": np.array([r[3] for r in rows])}


# ---------------------------------------------------------------- sister statistics

def _frac(k):
    return k / k.sum(1, keepdims=True)


def _tv(p, q):
    return 0.5 * np.abs(p - q).sum(1)


def heritable_share(k1, k2, group, rng, n_perm=2000, n_noise=500):
    """H = (D_rand - D_sis)/(D_rand - D_noise) on integer count arrays (clones x fates)."""
    p1, p2 = _frac(k1), _frac(k2)
    d_sis = _tv(p1, p2).mean()
    idx = [np.flatnonzero(group == g) for g in np.unique(group)]
    d_rand = 0.0
    for _ in range(n_perm):
        perm = np.arange(len(p2))
        for ix in idx:
            perm[ix] = rng.permutation(ix)
        d_rand += _tv(p1, p2[perm]).mean() / n_perm
    q = (k1 + k2) / (k1 + k2).sum(1, keepdims=True)
    n1, n2 = k1.sum(1), k2.sum(1)
    d_noise = np.mean([_tv(_frac(rng.multinomial(n1, q)), _frac(rng.multinomial(n2, q))).mean()
                       for _ in range(n_noise)])
    return {"D_sis": d_sis, "D_rand": d_rand, "D_noise": d_noise,
            "H": (d_rand - d_sis) / (d_rand - d_noise)}


def clone_logit_sd(k1, k2):
    """Per fate (one vs rest): sqrt(max(cov(l1, l2), 0)), l = logit((k + 0.5)/(n + 1))."""
    out = []
    for j in range(k1.shape[1]):
        l = [np.log((k[:, j] + 0.5) / (k.sum(1) - k[:, j] + 0.5)) for k in (k1, k2)]
        out.append(float(np.sqrt(max(np.cov(l[0], l[1])[0, 1], 0.0))))
    return out


def spearman(x, y):
    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))


def bootstrap(fn, n, rng, b=1000):
    """95% percentile CI of fn(index array) over clone resamples."""
    vals = np.array([fn(rng.integers(0, n, n)) for _ in range(b)], dtype=float)
    return np.nanpercentile(vals, [2.5, 97.5], axis=0).tolist()


def synthetic(k1, k2, group, mode, rng):
    """Replace sister counts (same totals): 'syn0' both from the group's population fractions (no
    clone memory), 'syn1' both from the clone's pooled fractions (full memory)."""
    n1, n2 = k1.sum(1), k2.sum(1)
    if mode == "syn1":
        q = (k1 + k2) / (k1 + k2).sum(1, keepdims=True)
    else:
        q = np.zeros(k1.shape)
        for g in np.unique(group):
            m = group == g
            q[m] = (k1[m] + k2[m]).sum(0) / (k1[m] + k2[m]).sum()
    return rng.multinomial(n1, q), rng.multinomial(n2, q)
