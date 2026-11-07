"""C5-S4: does a clone's chromatin distinctness fade over passages? (EXPERIMENT_LOG, pre-registered 2026-11-07)

    uv run python c5s4_atac_memory.py          # train analysis (passages 0-1) + selection/C5-S4.json

Data: the authors' MACS2 peaks after duplicate removal, one file per sample (data/gastruloid_raj/atac_peaks/),
sample sheet atac_meta/align.txt. Partitioned by passage (partitions.raj_atac): 0-1 train, 2 = sealed A,
3 = sealed B (last, H13). Sealed passages are read only in test(part) / counts(part), called by test_oracle.
"""

from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np

import partitions

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PEAKS = os.path.join(ROOT, "data", "gastruloid_raj", "atac_peaks")
SHEET = os.path.join(ROOT, "data", "gastruloid_raj", "atac_meta", "align.txt")
OUT = os.path.join(HERE, "results_gastruloid", "c5s4_atac_train.json")
SEL = os.path.join(HERE, "selection", "C5-S4.json")
DS = "raj_atac"
TOP = 20000   # peaks per sample (pre-registered)
BIN = 500     # bp
CLONES = ["clone 1", "clone 2", "clone 3", "clone 4", "clone 5", "clone 6", "clone 7"]
Z_C3S1 = {"clone 1": -4.03, "clone 3": 4.15, "clone 5": 5.08, "clone 6": -5.33, "clone 7": 3.38}
G3_R = 0.60   # propensity retention per passage (G3)
SEED = 54


def sheet():
    rows = []
    with open(SHEET, encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 5 and p[1].startswith("V"):
                rows.append({"v": p[1], "clone": p[2], "rep": int(p[3]), "passage": int(p[4])})
    return rows


def _path(v):
    return os.path.join(PEAKS, f"Sorted_{v}_rg_nodup_Small_Paired_peaks.narrowPeak_peaks.narrowPeak")


def _bins(v):
    """Top-TOP peaks by signalValue -> set of 500-bp summit bins; None if fewer than TOP peaks (QC fail)."""
    chrom, pos, sig = [], [], []
    with open(_path(v), encoding="utf-8") as f:
        for line in f:
            p = line.split("\t")
            chrom.append(p[0])
            pos.append(int(p[1]) + int(p[9]))
            sig.append(float(p[6]))
    if len(sig) < TOP:
        return None
    keep = np.argsort(-np.asarray(sig), kind="stable")[:TOP]
    return {(chrom[i], pos[i] // BIN) for i in keep}


def _n_peaks(v):
    with open(_path(v), "rb") as f:
        return sum(1 for _ in f)


def load(passages, allow_sealed=False):
    """{(clone, passage): [bin sets of QC-passing replicates]}. Train passages unless allow_sealed (oracle only)."""
    out = {}
    for r in sheet():
        if r["passage"] not in passages:
            continue
        part = partitions.partition_of(DS, r["passage"])
        if part != "train" and not allow_sealed:
            raise RuntimeError(f"passage {r['passage']} is sealed ({part}); read only via test_oracle")
        b = _bins(r["v"])
        if b is not None:
            out.setdefault((r["clone"], r["passage"]), []).append(b)
    return out


def jac(a, b):
    return len(a & b) / len(a | b)


def stats(data, p, clones):
    """J_rep, J_btw, S = J_rep - J_btw per clone at passage p, J_btw over the given clone set."""
    res = {}
    for c in clones:
        mine = data.get((c, p), [])
        if not mine:
            continue
        others = [s for o in clones if o != c for s in data.get((o, p), [])]
        jb = float(np.mean([jac(a, b) for a in mine for b in others])) if others else None
        jr = float(np.mean([jac(a, b) for a, b in itertools.combinations(mine, 2)])) if len(mine) >= 2 else None
        res[c] = {"n_rep": len(mine), "J_rep": jr, "J_btw": jb,
                  "S": (jr - jb) if (jr is not None and jb is not None) else None}
    return res


def train():
    data = load({0, 1})
    p0_clones = [c for c in CLONES if (c, 0) in data]
    st1 = stats(data, 1, CLONES)
    d0 = {c: 1 - v["J_btw"] for c, v in stats(data, 0, p0_clones).items()}
    d1 = {c: 1 - v["J_btw"] for c, v in stats(data, 1, p0_clones).items()}  # same clone set as passage 0
    common = sorted(set(d0) & set(d1))
    x, y = np.array([d0[c] for c in common]), np.array([d1[c] for c in common])
    # candidates: forecast D(p1) from D(p0); score = -SSE (M3 leave-one-clone-out, H12)
    loo = np.array([y[i] - np.mean(np.delete(y / x, i)) * x[i] for i in range(len(x))])
    cands = {"M1_fade_G3": -float(np.sum((y - G3_R * x) ** 2)),
             "M2_no_fade": -float(np.sum((y - x) ** 2)),
             "M3_train_ratio": -float(np.sum(loo ** 2))}
    logr = np.log(y / x)
    sd = float(np.std(logr, ddof=1))
    s1 = {c: v["S"] for c, v in st1.items() if v["S"] is not None}
    from scipy.stats import spearmanr

    zc = [c for c in Z_C3S1 if c in s1]
    rho = spearmanr([s1[c] for c in zc], [abs(Z_C3S1[c]) for c in zc]).statistic if len(zc) >= 3 else None
    res = {"qc": {f"{c}|p{p}": len(v) for (c, p), v in sorted(data.items())},
           "p1": st1, "D_btw_p0": d0, "D_btw_p1_same_set": d1, "p0_to_p1_log_ratio": dict(zip(common, logr.tolist())),
           "ratio_mean": float(np.mean(y / x)), "candidates": cands,
           "secondary_spearman_S1_vs_absz": {"clones": zc, "rho": rho, "note": "5 points, described only"}}
    chosen = max(cands, key=cands.get)
    sel = {"experiment": "C5-S4", "test": "gastruloid_c5s4_atac", "higher_is_better": True,
           "score": "-SSE of D_btw(p1) forecast from D_btw(p0), train clones; M3 leave-one-clone-out",
           "candidates": cands, "chosen": chosen,
           "forecast": {"M1_fade_G3": G3_R, "M2_no_fade": 1.0, "M3_train_ratio": res["ratio_mean"]}[chosen],
           "power": {"sd": round(sd, 4), "effect": round(abs(float(np.log(G3_R))), 4), "alpha": 0.05,
                     "one_sided": True, "min_power": 0.8, "weights": {},
                     "source": "sd of per-clone log D_btw ratio p0->p1 (train); effect |log 0.6|; n = clones "
                               "with >= 2 QC-passing replicates at the sealed passage and at passage 1"}}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    json.dump(sel, open(SEL, "w"), indent=1)
    print(json.dumps({"train": res, "selection": sel}, indent=1))


def counts(part):
    """Oracle power gate: clones with >= 2 replicates of >= TOP peaks at the sealed passage (line counts only)."""
    p = {v: k for k, v in partitions.PARTITIONED[DS]["fixed"].items()}[part]
    ok = {}
    for r in sheet():
        if r["passage"] == int(p) and r["clone"] in CLONES and _n_peaks(r["v"]) >= TOP:
            ok[r["clone"]] = ok.get(r["clone"], 0) + 1
    return {"clones": sum(1 for v in ok.values() if v >= 2)}


def test(part):
    """One look (via test_oracle): R = mean S(c,p) / mean S(c,1), one-sided 95% upper bound by clone bootstrap."""
    p = int({v: k for k, v in partitions.PARTITIONED[DS]["fixed"].items()}[part])
    data = load({1, p}, allow_sealed=True)
    s1, sp = stats(data, 1, CLONES), stats(data, p, CLONES)
    cl = [c for c in CLONES if s1.get(c, {}).get("S") is not None and sp.get(c, {}).get("S") is not None]
    a, b = np.array([sp[c]["S"] for c in cl]), np.array([s1[c]["S"] for c in cl])
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(10000):
        i = rng.integers(0, len(cl), len(cl))
        if b[i].mean() > 0:
            boot.append(a[i].mean() / b[i].mean())
    R = float(a.mean() / b.mean())
    ub = float(np.quantile(boot, 0.95))
    sel = json.load(open(SEL, encoding="utf-8"))
    return {"partition": part, "passage": p, "clones": cl, "S_p1": dict(zip(cl, b.tolist())),
            "S_p": dict(zip(cl, a.tolist())), "R": R, "R_upper95": ub, "fading": bool(ub < 1),
            "forecast": sel["forecast"], "chosen": sel["chosen"]}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    train()
