"""G2b: robustness of "any mixing rescues" (#22). See EXPERIMENT_LOG G2b.

    uv run python g2b_robust.py
"""

from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd

from g2_elongation import ANT, DATA, OUT, POST

CLASSES = ("comp", "same", "all")


def load():
    d = pd.read_csv(DATA).dropna(subset=["Total", "Elongated"])
    d = d[d["Total"] > 0].copy()
    pure = d[d["Composition"].str.match(r"^100% clone \d$")].copy()
    pure["c"] = pure["Composition"].str[-1].astype(int)
    pr = d[d["Composition"].str.match(r"^50% clone \d \+ 50% clone \d$")].copy()
    ab = pr["Composition"].apply(lambda s: [int(x) for x in re.findall(r"clone (\d)", s)])
    pr["a"], pr["b"] = ab.str[0], ab.str[1]

    def cls(a, b):
        if a in (2, 4) or b in (2, 4):
            return "unl"
        return "comp" if (a in ANT) != (b in ANT) and {a, b} <= ANT | POST else "same"
    pr["cls"] = [cls(a, b) for a, b in zip(pr["a"], pr["b"])]
    return pure, pr


def rates(pure):
    g = pure.groupby("c")[["Elongated", "Total"]].sum()
    return ((g["Elongated"] + 0.5) / (g["Total"] + 1)).to_dict()


def logit(p):
    return np.log(p / (1 - p))


def nulls(pa, pb):
    n1 = 1 / (1 + np.exp(-(logit(pa) + logit(pb)) / 2))
    return {"N1": n1, "N2": (pa + pb) / 2, "N3": np.maximum(pa, pb)}


def excess(pr, p_of_row):
    """pooled observed - pooled null per class, for each null."""
    out = {}
    for c in CLASSES:
        m = pr if c == "all" else pr[pr["cls"] == c]
        if len(m) == 0:
            out[c] = None
            continue
        pa, pb = p_of_row(m)
        nn = nulls(pa, pb)
        w = m["Total"].to_numpy(float)
        obs = m["Elongated"].sum() / w.sum()
        out[c] = {k: float(obs - (v * w).sum() / w.sum()) for k, v in nn.items()}
        out[c]["obs"] = float(obs)
        out[c]["n_gastruloids"] = int(w.sum())
        out[c]["n_rows"] = int(len(m))
    return out


def pooled_fn(pure):
    r = rates(pure)
    return lambda m: (m["a"].map(r).to_numpy(float), m["b"].map(r).to_numpy(float))


def matched(pure, pr):
    rows = []
    for _, x in pr.iterrows():
        s = pure[pure["Date"] == x["Date"]]
        ga, gb = s[s["c"] == x["a"]], s[s["c"] == x["b"]]
        if len(ga) and len(gb):
            y = x.copy()
            y["pa"] = (ga["Elongated"].sum() + 0.5) / (ga["Total"].sum() + 1)
            y["pb"] = (gb["Elongated"].sum() + 0.5) / (gb["Total"].sum() + 1)
            rows.append(y)
    return pd.DataFrame(rows)


def boot(fn, frames, dates, B, rng):
    """Cluster bootstrap over dates: fn(*resampled frames) -> nested dict of floats."""
    vals = []
    for _ in range(B):
        pick = rng.choice(dates, len(dates))
        rs = [pd.concat([f[f["Date"] == u] for u in pick]) for f in frames]
        try:
            vals.append(fn(*rs))
        except Exception:
            continue
    return vals


def ci_of(vals, path):
    xs = []
    for v in vals:
        for p in path:
            v = None if v is None else v.get(p)
        if v is not None and np.isfinite(v):
            xs.append(v)
    return np.percentile(xs, [2.5, 97.5]).tolist() if xs else [np.nan, np.nan]


def main(B=2000, seed=65):
    rng = np.random.default_rng(seed)
    pure, pr = load()
    dates = np.union1d(pure["Date"].unique(), pr["Date"].unique())
    res = {}

    t1 = lambda pu, p: excess(p, pooled_fn(pu))
    est1 = t1(pure, pr)
    b1 = boot(t1, [pure, pr], dates, B, rng)
    res["test1_pure_only"] = {c: {k: [est1[c][k], *ci_of(b1, [c, k])] for k in ("N1", "N2", "N3")}
                              | {"obs": est1[c]["obs"], "n": est1[c]["n_gastruloids"]} for c in CLASSES}
    d1 = [v["comp"]["N2"] - v["same"]["N2"] for v in b1 if v["comp"] and v["same"]]
    res["test1_comp_minus_same_N2"] = [est1["comp"]["N2"] - est1["same"]["N2"], *np.nanpercentile(d1, [2.5, 97.5]).tolist()]

    mt = matched(pure, pr)
    t2 = lambda m: excess(m, lambda z: (z["pa"].to_numpy(float), z["pb"].to_numpy(float)))
    est2 = t2(mt)
    b2 = boot(t2, [mt], mt["Date"].unique(), B, rng)
    res["test2_date_matched"] = {c: (None if est2[c] is None else
                                     {k: [est2[c][k], *ci_of(b2, [c, k])] for k in ("N2", "N3")}
                                     | {"obs": est2[c]["obs"], "n": est2[c]["n_gastruloids"], "rows": est2[c]["n_rows"]})
                                 for c in CLASSES}

    loo = {}
    for k in range(1, 8):
        p2 = pr[(pr["a"] != k) & (pr["b"] != k)]
        e = t1(pure, p2)
        bk = boot(t1, [pure, p2], dates, 500, rng)
        loo[f"drop_clone{k}"] = {c: (None if e[c] is None else [e[c]["N2"], *ci_of(bk, [c, "N2"])]) for c in CLASSES}
    res["test3_leave_one_clone_out_N2"] = loo

    # per-pair table (pure-only nulls)
    r = rates(pure)
    tab = []
    for comp, g in pr.groupby("Composition"):
        a, b = g["a"].iloc[0], g["b"].iloc[0]
        obs = g["Elongated"].sum() / g["Total"].sum()
        tab.append({"pair": f"{a}+{b}", "cls": g["cls"].iloc[0], "n": int(g["Total"].sum()), "obs": round(obs, 3),
                    "pure_a": round(r[a], 3), "pure_b": round(r[b], 3), "N2": round((r[a] + r[b]) / 2, 3),
                    "beats_best": bool(obs > max(r[a], r[b]))})
    res["pairs"] = tab
    res["pure_rates"] = {int(k): round(v, 3) for k, v in r.items()}
    json.dump(res, open(os.path.join(OUT, "g2b_robust.json"), "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
