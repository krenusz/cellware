"""G4: mixing-rescue mechanism. M0, Mgen, H1 (latent average + window, d=1,2), H2 (OR/AND
complementation). Leave-one-composition-out CV. See EXPERIMENT_LOG G4.

    uv run python g4_mechanism.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, log_expit
from scipy.stats import binom, spearmanr

from g2_elongation import COMP, DATA, OUT, load, pair_class

RIDGE = 1e-3
T_DATA = os.path.join(os.path.dirname(DATA), "T_overlap - Sheet1.csv")
G1A_POLARITY = {1: -0.92, 3: 0.88, 5: 1.05, 6: -1.13, 7: 0.73}  # G1a fit-shard clone means


def ll_rows(eta, k, n):
    return k * log_expit(eta) + (n - k) * log_expit(-eta)


class Model:
    restarts = 1

    def __init__(self, F, cls):
        self.F, self.cls = F, cls

    def nll(self, th, ix, k, n):
        return -ll_rows(self.eta(th, ix), k[ix], n[ix]).sum() + RIDGE * np.sum(th ** 2)

    def fit(self, ix, k, n, rng):
        best = None
        for r in range(self.restarts):
            x0 = self.init(rng) if r else self.init0()
            o = minimize(self.nll, x0, args=(ix, k, n), method="L-BFGS-B")
            if best is None or o.fun < best.fun:
                best = o
        return best.x


class M0(Model):
    npar = len(COMP)

    def init0(self):
        return np.zeros(self.npar)

    def init(self, rng):
        return rng.normal(0, 1, self.npar)

    def eta(self, th, ix):
        return self.F[ix] @ th[:len(COMP)]


class Mgen(M0):
    npar = len(COMP) + 3

    def eta(self, th, ix):
        e = self.F[ix] @ th[:len(COMP)]
        for j, t in enumerate(("comp", "same", "unl")):
            e = e + th[len(COMP) + j] * (self.cls[ix] == t)
        return e


class H1(Model):
    restarts = 10

    def __init__(self, F, cls, d):
        super().__init__(F, cls)
        self.d = d
        self.npar = len(COMP) * d + 1

    def init0(self):
        return np.r_[np.zeros(len(COMP) * self.d) + 0.1, 1.0]

    def init(self, rng):
        return np.r_[rng.normal(0, 1, len(COMP) * self.d), rng.normal(1, 1)]

    def z(self, th):
        return th[:len(COMP) * self.d].reshape(len(COMP), self.d)

    def eta(self, th, ix):
        Z = self.F[ix] @ self.z(th)
        return th[-1] - (Z ** 2).sum(1)


class H2(Model):
    restarts = 10
    K = 2

    def __init__(self, F, cls):
        super().__init__(F, cls)
        self.npar = len(COMP) * self.K + 2

    def init0(self):
        return np.r_[np.ones(len(COMP) * self.K), 1.0, 1.0]

    def init(self, rng):
        return np.r_[rng.normal(1, 1.5, len(COMP) * self.K), rng.normal(1, 1), abs(rng.normal(1, 1))]

    def eta(self, th, ix):
        c = log_expit(th[:len(COMP) * self.K].reshape(len(COMP), self.K))  # log capacity
        pres = self.F[ix] >= 0.2 - 1e-9
        C = np.stack([np.where(pres, c[:, k][None, :], -np.inf).max(1) for k in range(self.K)], 1)
        return th[-2] + th[-1] * C.sum(1)


def cv(model, d, F, k, n, groups, rng):
    held = [g for g in np.unique(groups) if not (g.startswith("100%") or g == "bulk")]
    tot, per = 0.0, {}
    for g in held:
        te = groups == g
        tr = np.where(~te)[0]
        th = model.fit(tr, k, n, rng)
        v = float(ll_rows(model.eta(th, np.where(te)[0]), k[te], n[te]).sum())
        per[g] = v
        tot += v
    return tot, per


def main(seed=66):
    rng = np.random.default_rng(seed)
    d, F, cls = load()
    k, n = d["Elongated"].to_numpy(float), d["Total"].to_numpy(float)
    groups = d["Composition"].str.strip().to_numpy()
    models = {"M0": M0(F, cls), "Mgen": Mgen(F, cls), "H1_d1": H1(F, cls, 1), "H1_d2": H1(F, cls, 2),
              "H2_K2": H2(F, cls)}
    res = {"n_rows": int(len(d)), "cv_loglik": {}, "full_fit_loglik": {}, "npar": {}}
    per = {}
    allix = np.arange(len(d))
    fits = {}
    for name, m in models.items():
        tot, p = cv(m, d, F, k, n, groups, rng)
        th = m.fit(allix, k, n, rng)
        fits[name] = th
        res["cv_loglik"][name] = tot
        res["full_fit_loglik"][name] = float(ll_rows(m.eta(th, allix), k, n).sum())
        res["npar"][name] = m.npar
        per[name] = p
        print(name, round(tot, 2), flush=True)
    cvl = res["cv_loglik"]
    best = max(cvl, key=cvl.get)
    res["best"] = best
    res["within_2_of_best"] = [m for m in cvl if cvl[best] - cvl[m] < 2]
    res["reading"] = (f"{best} wins" if len(res["within_2_of_best"]) == 1
                      else "not distinguished: " + ", ".join(res["within_2_of_best"]))
    res["per_composition_cv"] = {g: {m: round(per[m][g], 2) for m in per} for g in per["M0"]}

    # H2 diagnostic
    pure = d[d["Composition"].str.match(r"^100% clone \d$")]
    pr = pure.assign(c=pure["Composition"].str[-1].astype(int)).groupby("c")[["Elongated", "Total"]].sum()
    rate = (pr["Elongated"] / pr["Total"]).to_dict()
    below = []
    for g in np.unique(groups):
        if "+" not in g or "33%" in g:
            continue
        m = groups == g
        cl = [int(c) for c in g.replace("50% clone ", "").split(" + ")]
        best_p = max(rate[c] for c in cl)
        kk, nn = int(k[m].sum()), int(n[m].sum())
        p = float(binom.cdf(kk, nn, best_p))
        below.append({"pair": g, "obs": kk / nn, "better_member": best_p, "p_below": p})
    res["H2_pairs_below_better_member"] = [b for b in below if b["p_below"] < 0.05]
    res["H2_contradicted"] = len(res["H2_pairs_below_better_member"]) >= 2

    # H1 emergent axis (d = 1)
    z1 = H1(F, cls, 1).z(fits["H1_d1"])[:, 0]
    zc = {i: float(z1[COMP.index(f"c{i}")]) for i in range(1, 8)}
    rho = spearmanr([zc[c] for c in G1A_POLARITY], list(G1A_POLARITY.values())).statistic
    res["H1_d1_positions"] = {"bulk": float(z1[0]), **{f"c{i}": zc[i] for i in zc}}
    res["H1_axis_vs_G1a_polarity_spearman"] = float(rho)

    # H3 descriptive
    t = pd.read_csv(T_DATA)
    ov = t[t["Clone"] == "overall"].groupby("Condition")["percent_overlap"].agg(["mean", "count"])
    res["H3_T_overall"] = {c: [round(float(r["mean"]), 3), int(r["count"])] for c, r in ov.iterrows()}
    pure_T = {int(c.split()[-1]): ov.loc[c, "mean"] for c in ov.index if c.startswith("pure clone")}
    res["H3_pure_T_vs_elongation"] = {c: [round(float(pure_T[c]), 3), round(rate[c], 3)] for c in pure_T}
    pairs = []
    for c in ov.index:
        if "+" in c:
            a, b = [int(x.split()[-1]) for x in c.split("+")]
            if a in pure_T and b in pure_T:
                pairs.append({"pair": c, "T_obs": round(float(ov.loc[c, "mean"]), 3),
                              "T_mean_pure": round(float((pure_T[a] + pure_T[b]) / 2), 3)})
    res["H3_pairs_T"] = pairs
    res["H3_bulk_T"] = round(float(ov.loc["bulk", "mean"]), 3)
    json.dump(res, open(os.path.join(OUT, "g4_mechanism.json"), "w"), indent=1, default=float)
    print(json.dumps({k_: v for k_, v in res.items() if k_ != "per_composition_cv"}, indent=1, default=float))


if __name__ == "__main__":
    main()
