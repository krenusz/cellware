"""C5-S5: do subclones carry their parent clone's expression signature through a single-cell bottleneck?
(EXPERIMENT_LOG, pre-registered 2026-11-08)

    uv run python c5s5_subclone_expr.py        # train (VA passage-1 ESC RNA) + split-replicate power gate

Train: VA_1..18 kallisto TPM (open). Test: the locked subclone matrix (rna_subclone/), read ONLY in test(),
which the oracle calls after Bence's verbatim OK (H13; one look, last of its set).
"""

from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAJ = os.path.join(ROOT, "data", "gastruloid_raj")
VA = os.path.join(RAJ, "rna_counts")
TX2G = os.path.join(ROOT, "data", "ensembl_ref", "tx2gene.tsv")
OUT = os.path.join(HERE, "results_gastruloid", "c5s5_train.json")
FAILED = {"VA_1", "VA_11"}                # QC: 0.75 M reads; 23% aligned
BULK_REF = ["VA_2", "VA_5", "VA_9"]       # bulk + rainbow
SPLIT1_BULK, SPLIT2_BULK = ["VA_5"], ["VA_2", "VA_9"]
CLONES = [f"clone_{i}" for i in range(1, 8)]
PARENT = {"clone1s2": "clone_1", "clone5s2": "clone_5", "clone6a": "clone_6"}
GROUP = {"clone_1": "P", "clone_6": "P", "clone_3": "A", "clone_5": "A", "clone_7": "A"}
NGENES, MIN_TPM = 2000, 10.0
SEED = 56


def design():
    d = pd.read_csv(os.path.join(RAJ, "rna_meta", "studydesign.tsv"), sep="\t")
    d["rep"] = d["ID"].str.rsplit("_", n=1).str[-1].astype(int)
    return d[~d["Sample"].isin(FAILED)]


def va_tpm():
    t2g = pd.read_csv(TX2G, sep="\t", header=None, names=["tx", "gene", "sym"]).set_index("tx")["gene"]
    cols = {}
    for s in design()["Sample"]:
        a = pd.read_csv(os.path.join(VA, s, "abundance.tsv"), sep="\t", usecols=["target_id", "tpm"])
        a["gene"] = a["target_id"].str.split(".").str[0].map(t2g)
        cols[s] = a.dropna(subset=["gene"]).groupby("gene")["tpm"].sum()
    return pd.DataFrame(cols).fillna(0.0)


def lg(x):
    return np.log2(x + 1.0)


def train_parts():
    d, tpm = design(), va_tpm()
    L = lg(tpm)
    cm = {c: L[d.loc[d["clone"] == c, "Sample"]].mean(axis=1) for c in CLONES}
    cmt = pd.DataFrame({c: tpm[d.loc[d["clone"] == c, "Sample"]].mean(axis=1) for c in CLONES})
    expressed = cmt.mean(axis=1) >= MIN_TPM
    panel = pd.DataFrame(cm)[expressed].var(axis=1).sort_values(ascending=False).index[:NGENES]
    bulk = L[BULK_REF].mean(axis=1)
    parent_sig = pd.DataFrame({c: cm[c] - bulk for c in CLONES}).loc[panel]
    return d, L, panel, parent_sig


def train():
    d, L, panel, parent_sig = train_parts()
    b1, b2 = L[SPLIT1_BULK].mean(axis=1), L[SPLIT2_BULK].mean(axis=1)
    reps = {c: d[d["clone"] == c].sort_values("rep")["Sample"].tolist() for c in CLONES}
    two = [c for c in CLONES if len(reps[c]) >= 2]
    s1 = {c: (L[reps[c][0]] - b1).loc[panel] for c in two}
    s2 = {c: (L[reps[c][-1]] - b2).loc[panel] for c in CLONES}  # clone 4: its one good library
    corr, ranks, gaps = {}, {}, {}
    for c in two:
        r = {o: float(spearmanr(s1[c], s2[o]).statistic) for o in CLONES}
        corr[c] = r
        order = sorted(CLONES, key=lambda o: -r[o])
        ranks[c] = order.index(c) + 1
        gaps[c] = r[c] - np.mean([r[o] for o in CLONES if o != c])
    # exact null of T = sum of 3 iid uniform ranks 1..7
    tdist = np.array([sum(t) for t in itertools.product(range(1, 8), repeat=3)])
    pval = lambda t: float(np.mean(tdist <= t))
    crit = max(t for t in range(3, 22) if pval(t) < 0.05)
    rng = np.random.default_rng(SEED)
    power = {}
    for shrink in (1.0, 0.5):
        hit = 0
        for _ in range(20000):
            T = 0
            for c in rng.choice(two, 3):
                r = corr[c]
                others = [r[o] for o in CLONES if o != c]
                self_ = np.mean(others) + shrink * (r[c] - np.mean(others))
                T += 1 + sum(o > self_ for o in others)
            hit += T <= crit
        power[f"gap_x{shrink}"] = hit / 20000
    res = {"n_genes_panel": len(panel), "clones_split": two, "self_rank": ranks,
           "self_minus_other_corr": {c: round(float(g), 4) for c, g in gaps.items()},
           "split_corr": {c: {o: round(v, 4) for o, v in r.items()} for c, r in corr.items()},
           "T_critical": crit, "p_at_critical": pval(crit), "power": power,
           "gate": {"rule": "power >= 0.8 with the train gap halved", "pass": bool(power["gap_x0.5"] >= 0.8)},
           "parent_sig_corr_between_clones": {f"{a}|{b}": round(float(spearmanr(parent_sig[a], parent_sig[b]).statistic), 3)
                                              for a, b in itertools.combinations(CLONES, 2)}}
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res, indent=1))


def test(part="A"):
    """ONE look via test_oracle (after Bence's verbatim OK). Parent rank of each subclone among the 7 clones."""
    _, _, panel, parent_sig = train_parts()
    m = pd.read_csv(os.path.join(RAJ, "rna_subclone", "F8MWB5-expression-matrix.tsv"), sep="\t", index_col=0)
    cpm = lambda k: lg(m[f"F8MWB5_{k}_cpm"])
    bulk = (cpm(1) + cpm(2)) / 2
    subs = {"clone1s2": (3, 4), "clone5s2": (5, 6), "clone6a": (7, 8)}
    genes = [g for g in panel if g in m.index]
    out, T = {}, 0
    for s, (a, b) in subs.items():
        sig = ((cpm(a) + cpm(b)) / 2 - bulk).loc[genes]
        r = {c: float(spearmanr(sig, parent_sig.loc[genes, c]).statistic) for c in CLONES}
        order = sorted(CLONES, key=lambda c: -r[c])
        rank = order.index(PARENT[s]) + 1
        mates = [c for c in GROUP if GROUP[c] == GROUP[PARENT[s]] and c != PARENT[s]]
        out[s] = {"corr": r, "parent_rank": rank, "beats_group_mates": all(r[PARENT[s]] > r[c] for c in mates)}
        T += rank
    tdist = np.array([sum(t) for t in itertools.product(range(1, 8), repeat=3)])
    return {"genes_used": len(genes), "subclones": out, "T": T, "p": float(np.mean(tdist <= T)),
            "confirmed": bool(np.mean(tdist <= T) < 0.05)}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    train()
