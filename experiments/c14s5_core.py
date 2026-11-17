"""C14-S5 (b): train-only memory-vs-boundary check of the expression-state gain (pre-registered in EXPERIMENT_LOG).

Boundary distance = kNN type purity: share of a cell's 30 nearest readout neighbours (30 PCs, same embryo) with
its own annotated type. Core = purity >= 0.9, edge = purity < 0.6. LOEO base / base+PCs predictions exactly as
C14-S3 (B vs BP); per-embryo cell-weighted within-type delta AUC on all, core and edge cells.

Run (GitHub Actions, --petracer): python -I experiments/c14s5_core.py
Smoke only (synthetic, scratch): python -I experiments/c14s5_core.py --synthetic OUTDIR
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c14s3_region as s3  # noqa: E402

OUT = HERE / "results_c14"
PCS = s3.PCS
K, CORE, EDGE = 30, 0.9, 0.6
SEED = 20261117
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def purity(D):
    out = pd.Series(np.nan, index=D.index)
    for _, g in D.groupby("embryo"):
        X = g[PCS].to_numpy()
        k = min(K + 1, len(g))
        _, idx = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(X)
        T = g["T"].to_numpy()
        same = (T[idx[:, 1:]] == T[:, None]).mean(1)
        out.loc[g.index] = same
    return out


def deltas(s1, D, embryos, ycol, pur):
    pb = s3.loeo(s1, D, embryos, ycol, [])
    pf = s3.loeo(s1, D, embryos, ycol, ["pcs"])
    res = {}
    for name, mask in [("all", pur >= 0), ("core", pur >= CORE), ("edge", pur < EDGE)]:
        sub = D[mask.to_numpy()]
        per = s3.per_embryo(s1, sub, embryos, ycol, pb.loc[sub.index], pf.loc[sub.index])
        res[name] = dict(per_embryo=per, mean=float(np.mean(list(per.values()))) if per else float("nan"),
                         cells=int(mask.sum()))
    res["retained_core"] = res["core"]["mean"] / res["all"]["mean"] if res["all"]["mean"] else float("nan")
    return res


def decide(c):
    pos = sum(v > 0 for v in c["core"]["per_embryo"].values())
    n = len(c["core"]["per_embryo"])
    if c["retained_core"] >= 0.5 and pos >= min(5, n):
        return "MEMORY-CONSISTENT", pos, n
    if c["retained_core"] < 0.25:
        return "BOUNDARY-DRIVEN", pos, n
    return "MIXED", pos, n


def analyse(D, embryos, s1, outdir):
    rng = np.random.default_rng(SEED)
    pur = purity(D)
    say(f"purity: median {pur.median():.2f}; core (>= {CORE}) {float((pur >= CORE).mean()):.2f}; "
        f"edge (< {EDGE}) {float((pur < EDGE).mean()):.2f} of {len(D)} cells")
    sig = D["pc0"] + rng.normal(0, 1, len(D))
    edge = (pur < EDGE).to_numpy()
    D["y_edge"] = np.where(edge, (sig > 0).astype(int), rng.integers(0, 2, len(D)))
    D["y_all"] = (sig > 0).astype(int)
    s_edge = deltas(s1, D, embryos, "y_edge", pur)
    s_all = deltas(s1, D, embryos, "y_all", pur)
    ok1, ok2 = s_edge["retained_core"] < 0.25, s_all["retained_core"] >= 0.5
    say(f"SMOKE (i) edge-only signal: retained_core {s_edge['retained_core']:.2f} -> {'PASS' if ok1 else 'FAIL'}")
    say(f"SMOKE (ii) signal everywhere: retained_core {s_all['retained_core']:.2f} -> {'PASS' if ok2 else 'FAIL'}")
    c = deltas(s1, D, embryos, "y", pur)
    verdict, pos, n = decide(c)
    if not (ok1 and ok2):
        verdict = "INVALID (smoke test failed)"
    say(f"d all {c['all']['mean']:+.4f} ({c['all']['cells']} cells); core {c['core']['mean']:+.4f} "
        f"({c['core']['cells']}); edge {c['edge']['mean']:+.4f} ({c['edge']['cells']}); "
        f"retained_core {c['retained_core']:.2f}; core > 0 in {pos}/{n} embryos -> {verdict}")
    say("per embryo core: " + ", ".join(f"{k} {v:+.3f}" for k, v in c["core"]["per_embryo"].items()))
    by_type = {}
    for T, g in D.assign(p=pur).groupby("T"):
        by_type[T] = dict(cells=len(g), core_frac=round(float((g["p"] >= CORE).mean()), 3),
                          edge_frac=round(float((g["p"] < EDGE).mean()), 3))
    res = dict(verdict=verdict, core_positive=pos, n_embryos=n, k=K, core_cut=CORE, edge_cut=EDGE,
               smoke_edge=s_edge, smoke_all=s_all, smoke_pass=bool(ok1 and ok2), by_type=by_type, **c)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    (Path(outdir) / "c14s5_core.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    (Path(outdir) / "c14s5_core.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    return res


def synthetic(outdir):
    import c14s1_select as s1
    rng = np.random.default_rng(3)
    rows = []
    for emb in ["E8.0-R1", "E8.5-R2", "E9.0-R1", "E9.5-R1", "E8.0-R3", "E8.5-R3"]:
        for j, t in enumerate(["T1", "T2", "T3"]):
            n = 400
            P = rng.normal(size=(n, len(PCS)))
            P[:, 5] += 3 * j  # types separate on pc5; pc0 carries the planted signals
            P[: n // 4, 5] -= 1.5  # a quarter of each type sits toward the neighbouring type
            y = ((1.2 * P[:, 1] + rng.normal(size=n)) > 0).astype(int)
            df = pd.DataFrame({"embryo": emb, "T": t, "y": y, "cluster": [f"{emb}{t}{i // 4}" for i in range(n)],
                               "total_counts": rng.integers(1000, 5000, n), "detection_rate": rng.random(n),
                               "clade": rng.integers(6, 50, n), "depth": rng.integers(1, 9, n), "stage": 8.5,
                               "lineage": "L", "germ_layer": "G", "L1": "A", "L2": "B"})
            for i, p in enumerate(PCS):
                df[p] = P[:, i]
            rows.append(df)
    D = pd.concat(rows, ignore_index=True)
    D.index = [f"c{i}" for i in range(len(D))]
    res = analyse(D, sorted(D.embryo.unique()), s1, outdir)
    assert res["smoke_pass"], "smoke failed on synthetic"
    print("SYNTHETIC OK", res["verdict"], round(res["retained_core"], 2))


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--synthetic":
        return synthetic(sys.argv[2])
    D, embryos, s1 = s3.readout_with_pcs()
    say(f"readout cells {len(D)}; types {D['T'].nunique()}; embryos {embryos}")
    analyse(D, embryos, s1, OUT)


if __name__ == "__main__":
    main()
