"""C14-S1a train-only selection (H12): feature ablation, smoke tests, embryo-level power.

Pre-registered in EXPERIMENT_LOG (C14-S1a). Reuses the C14-S0 loaders, which filter to train embryos
with partitions.train_only before any other column and skip sealed trees by key.

Feature sets added to the base model: S = author subtype; U = UMAP x/y + phase; F = S + U.
Smoke tests: (i) outcomes shuffled across clusters within type x embryo -> |delta| < 0.01 for every set;
(ii) planted feature y + N(0, 1.5) added to U -> delta > 0.05.
Embryo-level power: each of A's 5 embryos gets a random stage-matched train donor; clusters are
resampled within the donor; pooled delta; null sd from paired swaps; power.sim_power.
Selection rule: U is carried to the A look if LOEO delta(U) >= 0.02, embryo-level power(U) >= 0.8 at
half the effect, and both smoke tests pass; otherwise no A look before S1b or Bence's decision.

Run: .venv/Scripts/python.exe -I experiments/c14s1_select.py
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c14s0_gate as g0  # noqa: E402  (train_only filtering lives in g0.load_obs)
from partitions import train_only  # noqa: E402,F401  (guard: partitioned reads go through train_only)
from power import sim_power  # noqa: E402

OUT = HERE / "results_c14"
SEL = HERE / "selection" / "C14-S1.json"
MIN_CL = 30
SEED, B = 20261116, 400
DONORS = {8.0: ["E8.0-R1", "E8.0-R3"], 8.5: ["E8.5-R2", "E8.5-R3"], 9.0: ["E9.0-R1"], 9.5: ["E9.5-R1"]}
SETS = {"S": ["sub"], "U": ["umap", "phase"], "F": ["sub", "umap", "phase"]}
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def build():
    obs, train_embryos, raw = g0.load_obs()
    obs["stage"] = obs["embryo"].map(g0.stage_of)
    ctype, lin = obs["cell_type"].to_dict(), obs["lineage"].to_dict()
    recs = []
    for key, tree in g0.iter_train_trees(raw, train_embryos):
        r, _, _ = g0.relatives(tree, ctype, lin)
        recs += [(key,) + x for x in r]
    del raw
    R = pd.DataFrame(recs, columns=["tree", "cell", "anc", "clade", "depth", "relcnt"]).set_index("cell").join(obs)
    R["cluster"] = R["tree"] + ":" + R["anc"]
    rows = []
    for T, g in R.groupby("cell_type"):
        tot = collections.Counter()
        for c in g["relcnt"]:
            tot.update(c)
        if len(tot) < 2:
            continue
        (L1, n1), (L2, n2) = tot.most_common(2)
        allr = sum(tot.values())
        if n1 / allr < g0.MIN_SHARE or n2 / allr < g0.MIN_SHARE:
            continue
        a = g["relcnt"].map(lambda c: c.get(L1, 0))
        b = g["relcnt"].map(lambda c: c.get(L2, 0))
        ok = (a + b >= 3) & (a != b)
        h = g[ok].copy()
        h["y"] = (a[ok] > b[ok]).astype(int)
        if len(h) < g0.MIN_CELLS or h["y"].nunique() < 2:
            continue
        if min(h.loc[h.y == 1, "cluster"].nunique(), h.loc[h.y == 0, "cluster"].nunique()) < MIN_CL:
            continue  # H21 rule enforced before modelling
        h["T"] = T
        h["L1"], h["L2"] = L1, L2  # names only (C14-S4); not model features
        rows.append(h.drop(columns=["relcnt"]))
    D = pd.concat(rows)
    say(f"types kept {D['T'].nunique()}, cells {len(D)}, clusters {D['cluster'].nunique()}")
    return D, train_embryos


def X(df, parts, planted=None):
    num = pd.DataFrame({"ltc": np.log1p(df["total_counts"]), "det": df["detection_rate"],
                        "lsz": np.log(df["clade"]), "dep": df["depth"], "stage": df["stage"]}, index=df.index)
    cats = ["lineage", "germ_layer"] + (["cell_subtype"] if "sub" in parts else []) + (["phase"] if "phase" in parts else [])
    out = pd.concat([num, pd.get_dummies(df[cats].astype(str)).astype(float)], axis=1)
    if "umap" in parts:
        out["u1"], out["u2"] = df["umap1"], df["umap2"]
    if planted is not None:
        out["planted"] = planted.loc[df.index]
    return out


def loeo(D, embryos, ycol, parts, planted=None):
    p = pd.Series(np.nan, index=D.index)
    for T, g in D.groupby("T"):
        Xg = X(g, parts, planted)
        for e in embryos:
            te = (g["embryo"] == e).to_numpy()
            if te.sum() == 0 or g.loc[~te, ycol].nunique() < 2:
                continue
            p.loc[g.index[te]] = g0.fit_pred(Xg[~te], g.loc[~te, ycol], Xg[te])
    return p


def pooled_delta(df, ycol="y"):
    num = den = 0.0
    for _, g in df.groupby("T"):
        s = pd.DataFrame({"y": g[ycol], "p_base": g["p_base"], "p_full": g["p_full"]})
        d = g0.auc_delta(s)
        if d is not None:
            num += d * len(g)
            den += len(g)
    return num / den if den else float("nan")


def run_sets(D, embryos, ycol, planted=None, sets=SETS):
    base = loeo(D, embryos, ycol, [])
    out = {}
    for name, parts in sets.items():
        full = loeo(D, embryos, ycol, parts, planted)
        E = D.assign(p_base=base, p_full=full).dropna(subset=["p_base", "p_full"])
        out[name] = (pooled_delta(E, ycol), E)
    return out


def embryo_power(E, rng):
    flat = E.reset_index(drop=True)
    # positions in flat (g.index after reset_index), not positions within the subgroup g
    by = {(e, T): [g.index.to_numpy()[ix] for ix in g.groupby("cluster").indices.values()]
          for (e, T), g in flat.groupby(["embryo", "T"])}
    a_stages = [g0.stage_of(e) for e in g0.A_EMBRYOS]

    def draw(swap):
        parts = []
        for st in a_stages:
            donor = rng.choice(DONORS[st])
            for (e, T), cl in by.items():
                if e != donor or len(cl) < 2:
                    continue
                pick = rng.integers(0, len(cl), len(cl))
                s = flat.iloc[np.concatenate([cl[k] for k in pick])].copy()
                if swap:
                    flip = np.repeat(rng.random(len(cl)) < 0.5, [len(cl[k]) for k in pick])
                    pb, pf = s["p_base"].to_numpy(), s["p_full"].to_numpy()
                    s["p_base"], s["p_full"] = np.where(flip, pf, pb), np.where(flip, pb, pf)
                parts.append(s)
        return pooled_delta(pd.concat(parts))

    stats = np.array([draw(False) for _ in range(B)])
    null_sd = float(np.std([draw(True) for _ in range(B)]))
    return sim_power(stats, null_sd), sim_power(stats - stats.mean() / 2, null_sd), float(stats.mean()), \
        float(stats.std()), null_sd


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    D, embryos = build()

    # smoke (i): shuffle outcomes across clusters within type x embryo
    D["y_shuf"] = D["y"]
    for _, g in D.groupby(["T", "embryo"]):
        cy = g.groupby("cluster")["y"].first()
        perm = pd.Series(rng.permutation(cy.to_numpy()), index=cy.index)
        D.loc[g.index, "y_shuf"] = g["cluster"].map(perm).to_numpy()
    shuf = {k: v[0] for k, v in run_sets(D, embryos, "y_shuf").items()}
    smoke1 = all(abs(v) < 0.01 for v in shuf.values())
    say(f"SMOKE (i) shuffled null delta: {shuf} -> {'PASS' if smoke1 else 'FAIL'}")

    # smoke (ii): planted feature on U
    planted = D["y"] + rng.normal(0, 1.5, len(D))
    pl = run_sets(D, embryos, "y", planted=planted, sets={"U+planted": SETS["U"]})["U+planted"][0]
    smoke2 = pl > 0.05
    say(f"SMOKE (ii) planted delta (U + planted): {pl:.4f} -> {'PASS' if smoke2 else 'FAIL'}")
    if not (smoke1 and smoke2):
        say("STOP: smoke test failed; ablation not read as evidence")

    real = run_sets(D, embryos, "y")
    res = {"smoke_shuffled": shuf, "smoke_planted": pl, "smoke_pass": bool(smoke1 and smoke2), "sets": {}}
    for name, (d, E) in real.items():
        per_emb = {e: pooled_delta(E[E.embryo == e]) for e in embryos if (E.embryo == e).any()}
        pw, pwh, m, sd, nsd = embryo_power(E, rng)
        res["sets"][name] = dict(loeo_delta=d, per_embryo=per_emb, power_A=pw, power_A_half=pwh,
                                 sim_mean=m, sim_sd=sd, null_sd=nsd)
        say(f"SET {name}: LOEO delta {d:+.4f}; per embryo "
            + ", ".join(f"{e} {v:+.3f}" for e, v in per_emb.items())
            + f"; embryo-level power on A {pw:.3f} (half effect {pwh:.3f}); sim mean {m:.4f} sd {sd:.4f}, null sd {nsd:.4f}")
    u = res["sets"]["U"]
    carry = bool(res["smoke_pass"] and u["loeo_delta"] >= 0.02 and u["power_A_half"] >= 0.8)
    res["selection"] = ("U carried to the A look" if carry else
                        "no A look: U fails the rule (S1b expression or Bence decides)")
    say(f"SELECTION: {res['selection']}")
    SEL.write_text(json.dumps(dict(experiment="C14-S1", rule="pre-registered C14-S1a", model="U" if carry else None,
                                   features_U="base + UMAP x/y + phase", **res), indent=1, default=str), encoding="utf-8")
    (OUT / "c14s1_select.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    (OUT / "c14s1_select.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
