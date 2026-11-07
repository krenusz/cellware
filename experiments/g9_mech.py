"""G9: does signalling set the clones' division of labour? Train partitions only. See EXPERIMENT_LOG G9.

    uv run --with openpyxl python g9_mech.py
Reads only train drug series (Dec 2024, Feb/Mar 2025) and train Wnt gastruloids (train_only).
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from g7_model import pol, resample
from partitions import train_only

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "data", "gastruloid_raj", "raw_ls2")
OUT = os.path.join(HERE, "results_gastruloid")
SEL = os.path.join(HERE, "selection")

TRAIN_DRUG = {  # file -> (series, {tag: (pair, condition)})
    "clone1+3_clone1+6_drugtx_11282024.xlsx": ("dec24", {
        "1+3_ctrl": ("1+3", "ctrl"), "1+3_RA": ("1+3", "RA"), "1+3_SB": ("1+3", "SB"), "1+3_AGN": ("1+3", "AGN"),
        "1+6_ctrl": ("1+6", "ctrl"), "1+6_RA": ("1+6", "RA"), "1+6_SB": ("1+6", "SB")}),
    "clone1+3_drugtx_02112024.xlsx": ("feb25", {
        "control linescan": ("1+3", "ctrl"), "ra linescan": ("1+3", "RA"), "sb linescan": ("1+3", "SB"),
        "agn linescan": ("1+3", "AGN")}),
    "clone1+6_drugtx_02142024.xlsx": ("feb25", {
        "1+6_nodrug": ("1+6", "ctrl"), "1+6_ra": ("1+6", "RA"), "1+6_sb": ("1+6", "SB"), "1+6_agn": ("1+6", "AGN")}),
}
DRUGS = ("RA", "SB", "AGN")


def lines(d):
    """Per (Annotation ID, line index) dict of header -> values (an ID may hold several lines)."""
    out = []
    for (aid, tag), g in d.groupby(["Annotation ID", "Tags"]):
        by = {h: list(x["Values"]) for h, x in g.groupby("Header")}
        n = min(len(v) for v in by.values())
        for i in range(n):
            out.append((aid, tag, {h: v[i] for h, v in by.items()}))
    return out


def drug_D():
    rows = []
    for f, (series, tags) in TRAIN_DRUG.items():
        d = pd.read_excel(os.path.join(RAW, f))
        for aid, tag, h in lines(d[d["Tags"].isin(tags)]):
            if "Channel 2" in h and "Channel 0" in h:
                D = pol(resample(h["Channel 2"])) - pol(resample(h["Channel 0"]))
                if np.isfinite(D):
                    pair, cond = tags[tag]
                    rows.append({"series": series, "pair": pair, "cond": cond, "D": D, "id": aid})
    return pd.DataFrame(rows)


def cond_means(df):
    g = df.groupby(["series", "pair", "cond"])["D"]
    m = g.agg(["mean", "std", "count"]).reset_index()
    m["se"] = (m["std"] / np.sqrt(m["count"])).clip(lower=0.05)
    return m


# ---------- G9a drug models ----------

def predict(model, th, keys, train):
    """keys: list of (pair, cond). th: parameters fitted on train (dict)."""
    out = []
    for pair, cond in keys:
        c = th["c"][pair]
        if model == "D0":
            out.append(th["m"][pair])
        elif model == "D1":
            out.append(c * (1.0 if cond == "ctrl" else th["g"][cond]))
        elif model == "D2":
            out.append(c + (0.0 if cond == "ctrl" else th["s"][cond]))
        else:  # D3
            out.append(th["pc"].get((pair, cond), c))
    return np.array(out)


def fit_drug(model, m):
    pairs = sorted(m["pair"].unique())
    y, se = m["mean"].to_numpy(), m["se"].to_numpy()
    keys = list(zip(m["pair"], m["cond"]))
    w = 1 / se ** 2
    if model == "D0":
        mm = {p: float(np.average(y[m["pair"] == p], weights=w[m["pair"] == p])) for p in pairs}
        return {"m": mm, "c": mm}
    if model == "D3":
        pc = {k: float(np.average(y[[kk == k for kk in keys]], weights=w[[kk == k for kk in keys]])) for k in set(keys)}
        c = {p: pc.get((p, "ctrl"), float(np.mean(y[m["pair"] == p]))) for p in pairs}
        return {"pc": pc, "c": c}
    drugs = [d for d in DRUGS if d in set(m["cond"])]

    def unpack(x):
        th = {"c": dict(zip(pairs, x[:len(pairs)]))}
        th["g" if model == "D1" else "s"] = dict(zip(drugs, x[len(pairs):]))
        return th

    def loss(x):
        return float(np.sum(((y - predict(model, unpack(x), keys, None)) / se) ** 2))
    x0 = np.r_[[np.mean(y[(m["pair"] == p).to_numpy()]) for p in pairs], np.full(len(drugs), 1.0 if model == "D1" else 0.0)]
    res = minimize(loss, x0, method="Nelder-Mead", options={"maxiter": 40000, "xatol": 1e-6, "fatol": 1e-9})
    th = unpack(res.x)
    for d in DRUGS:  # drug absent from training: no effect
        th.setdefault("g" if model == "D1" else "s", {}).setdefault(d, 1.0 if model == "D1" else 0.0)
    return th


def gll(y, mu, se):
    return float(np.sum(-0.5 * ((y - mu) / se) ** 2 - np.log(se * np.sqrt(2 * np.pi))))


def g9a():
    df = drug_D()
    m = cond_means(df)
    res = {"n_by_condition": m.assign(mean=m["mean"].round(3), se=m["se"].round(3))[
        ["series", "pair", "cond", "count", "mean", "se"]].to_dict("records"), "models": {}}
    for model in ("D0", "D1", "D2", "D3"):
        cv = {}
        for s in ("dec24", "feb25"):
            tr, te = m[m["series"] != s], m[m["series"] == s]
            th = fit_drug(model, tr)
            cv[s] = gll(te["mean"].to_numpy(), predict(model, th, list(zip(te["pair"], te["cond"])), None), te["se"].to_numpy())
        th = fit_drug(model, m)
        pred = predict(model, th, list(zip(m["pair"], m["cond"])), None)
        z = (m["mean"].to_numpy() - pred) / m["se"].to_numpy()
        misses = [f"{r.series} {r.pair} {r.cond}: z {zz:.2f}" for r, zz in zip(m.itertuples(), z) if abs(zz) > 2.5]
        par = {k: ({kk if isinstance(kk, str) else "|".join(kk): round(float(vv), 3) for kk, vv in v.items()}) for k, v in th.items()}
        res["models"][model] = {"cv_loglik": round(sum(cv.values()), 2), "cv_by_series": {k: round(v, 2) for k, v in cv.items()},
                                "params": par, "consistency_misses": misses}
    return res


# ---------- G9b Wnt ----------

WNT = {"clone1_Wnt_04262025.csv": (1, "Channel 0", -1, {"bulk+1": "normal", "bulk+1_hi": "hi", "bulk+1_lo": "lo"}),
       "clone3_Wnt_04262025.csv": (3, "Channel 2", 1, {"bulk+3": "normal", "bulk+3_hi": "hi", "bulk+3_lo": "lo"})}


def wnt_P():
    rows = []
    for f, (clone, ch, sgn, tags) in WNT.items():
        d = pd.read_csv(os.path.join(RAW, f))
        d = train_only("raj_wnt", d, unit_col="Annotation ID")
        for aid, tag, h in lines(d[d["Tags"].isin(tags)]):
            if ch in h:
                p = sgn * pol(resample(h[ch]))
                if np.isfinite(p):
                    rows.append({"clone": clone, "dose": tags[tag], "P": p, "id": aid})
    return pd.DataFrame(rows)


def wnt_fit(model, d):
    out = {}
    for c in (1, 3):
        x = d[d["clone"] == c]
        base = x[x["dose"] == "normal"]["P"].mean() if (x["dose"] == "normal").any() else x["P"].mean()
        out[c] = {"normal": base}
        for dose in ("hi", "lo"):
            if model == "W0":
                out[c][dose] = x["P"].mean()
            elif model == "W2":
                xs = x[x["dose"] == dose]["P"]
                out[c][dose] = xs.mean() if len(xs) else base
        if model == "W0":
            out[c]["normal"] = x["P"].mean()
    if model == "W1":
        for dose in ("hi", "lo"):
            num = sum(d[(d["clone"] == c) & (d["dose"] == dose)]["P"].sum() * out[c]["normal"] for c in (1, 3))
            den = sum(((d["clone"] == c) & (d["dose"] == dose)).sum() * out[c]["normal"] ** 2 for c in (1, 3))
            f = num / den if den > 0 else 1.0
            for c in (1, 3):
                out[c][dose] = f * out[c]["normal"]
            out[f"factor_{dose}"] = f
    return out


def g9b():
    d = wnt_P().reset_index(drop=True)
    res = {"n": {f"clone{c}_{s}": int(k) for (c, s), k in d.groupby(["clone", "dose"]).size().items()}, "models": {}}
    for model in ("W0", "W1", "W2"):
        ll = 0.0
        for i in d.index:  # leave one gastruloid ID out (lines of one ID go together)
            tr = d[d["id"] != d.loc[i, "id"]]
            fit = wnt_fit(model, tr)
            mu = fit[d.loc[i, "clone"]][d.loc[i, "dose"]]
            resid = tr["P"].to_numpy() - np.array([fit[c][s] for c, s in zip(tr["clone"], tr["dose"])])
            sd = max(float(resid.std()), 0.05)
            ll += -0.5 * ((d.loc[i, "P"] - mu) / sd) ** 2 - np.log(sd * np.sqrt(2 * np.pi))
        fit = wnt_fit(model, d)
        cm = d.groupby(["clone", "dose"])["P"].agg(["mean", "std", "count"]).reset_index()
        misses = []
        for r in cm.itertuples():
            se = max((r.std if np.isfinite(r.std) else 0.5) / np.sqrt(r.count), 0.05)
            z = (r.mean - fit[r.clone][r.dose]) / se
            if abs(z) > 2.5:
                misses.append(f"clone{r.clone} {r.dose}: z {z:.2f}")
        res["models"][model] = {"cv_loglik": round(float(ll), 2),
                                "fit": {str(k): ({kk: round(float(vv), 3) for kk, vv in v.items()} if isinstance(v, dict) else round(float(v), 3))
                                        for k, v in fit.items()},
                                "condition_means": {f"clone{r.clone}_{r.dose}": [round(r.mean, 3), int(r.count)] for r in cm.itertuples()},
                                "consistency_misses": misses}
    return res


def select(res, test, exp):
    ranking = sorted(res["models"], key=lambda k: -res["models"][k]["cv_loglik"])
    qual = [k for k in ranking if not res["models"][k]["consistency_misses"]]
    res["ranking_by_cv"], res["best_qualifying"] = ranking, (qual[0] if qual else None)
    if qual and qual[0] == ranking[0]:
        os.makedirs(SEL, exist_ok=True)
        json.dump({"test": test, "candidates": {k: res["models"][k]["cv_loglik"] for k in ranking},
                   "higher_is_better": True, "chosen": ranking[0],
                   "note": f"{exp}: leave-one-out CV on train partition; consistency within 2.5 SE"},
                  open(os.path.join(SEL, f"{exp}.json"), "w"), indent=1)
    return res


def main():
    a = select(g9a(), "gastruloid_g9_drug", "G9a")
    b = select(g9b(), "gastruloid_g9_wnt", "G9b")
    json.dump({"G9a": a, "G9b": b}, open(os.path.join(OUT, "g9_mech.json"), "w"), indent=1, default=str)
    for name, r in (("G9a drugs", a), ("G9b Wnt", b)):
        print(f"== {name}: ranking {r['ranking_by_cv']}, best qualifying {r['best_qualifying']}")
        for k in r["ranking_by_cv"]:
            v = r["models"][k]
            print(f"  {k}: CV {v['cv_loglik']}  misses {v['consistency_misses']}")
            print(f"     {v.get('params', v.get('fit'))}")
    print(json.dumps(a["n_by_condition"], indent=0)[:2500])
    print(json.dumps(b["models"]["W2"]["condition_means"]))


if __name__ == "__main__":
    main()
