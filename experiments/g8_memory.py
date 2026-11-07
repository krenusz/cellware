"""G8: memory model selection on open + spent data (no sealed file read). See EXPERIMENT_LOG G8.

    uv run --with openpyxl python g8_memory.py
"""

from __future__ import annotations

import csv
import json
import os

import numpy as np
from scipy.optimize import minimize

import g7_model

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_gastruloid")
P1 = {1: 0.922, 3: 0.881}  # G1a fit-shard passage-1 |means| (G7b reference)


def ledger(test):
    with open(os.path.join(HERE, "test_ledger.tsv"), encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["test"] == test:
                return json.loads(r["result"])
    raise KeyError(test)


def se(lo, hi):
    return max((hi - lo) / 3.92, 0.03)


def observations(boot=2000, seed=81):
    obs = []  # (series, clones tuple, passages elapsed k, ratio, se)
    # April 2024 1+3 pairs (G3, open): per-clone signed P ratio to passage 1
    df = g7_model.open_pairs()
    rng = np.random.default_rng(seed)
    for col, c, sg in (("P3", 3, 1), ("P1", 1, -1)):
        g = {p: sg * df[df["passage"] == p][col].to_numpy() for p in (1, 2, 3, 4)}
        for p in (2, 3, 4):
            est = g[p].mean() / g[1].mean()
            bs = [rng.choice(g[p], len(g[p])).mean() / rng.choice(g[1], len(g[1])).mean() for _ in range(boot)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            obs.append(("apr24_pair", (c,), p - 1, float(est), se(lo, hi)))
    # Sep 2024 subclones (G5, open): p2 / p1 subclone, one passage
    mem = json.load(open(os.path.join(OUT, "g5_memory.json")))["ratios"]
    for key, c in (("head", 3), ("tail", 1)):
        e, lo, hi = mem[key]["p2_over_p1"]
        obs.append(("sep24_subclone", (c,), 1, e, se(lo, hi)))
    # late 2024 / Jan 2025 clone + bulk passage 4 (G6a, spent, ledger)
    g6a = ledger("gastruloid_p4_memory")["per_clone_ratio"]
    for c in (1, 3, 5, 6):
        e, lo, hi = g6a[f"clone{c}"]
        obs.append(("late24_bulk", (c,), 3, e, se(lo, hi)))
    # Jan 2024 1+3 pairs (G6b, spent): pair S ratio passage 4 / 1
    e, lo, hi = ledger("gastruloid_jan24_pairs")["p4_over_p1"]
    obs.append(("jan24_pair", (1, 3), 3, e, se(lo, hi)))
    # July 2024 clone in bulk (G7b, spent): signed P / G1a passage-1 mean
    g7 = ledger("gastruloid_g7_memory")
    for key, c, k in (("clone1_p2", 1, 1), ("clone3_p2", 3, 1), ("clone3_p3", 3, 2), ("clone3_p4", 3, 3)):
        m, lo, hi = g7[key]["mean_signed_P"]
        obs.append(("jul24_bulk", (c,), k, m / P1[c], se(lo, hi) / P1[c]))
    return obs


CLONES = (1, 3, 5, 6)
SERIES = ("apr24_pair", "sep24_subclone", "late24_bulk", "jan24_pair", "jul24_bulk")


def clip(r):
    return np.clip(r, 0.0, 1.2)


class Model:
    """r(series, clone) from parameters; random-effect models have a series term."""
    name = ""
    random_series = False

    def n_par(self, series):
        raise NotImplementedError

    def r(self, th, s, c, series):
        raise NotImplementedError

    def pred(self, th, o, series):
        s, cl, k, _, _ = o
        return float(np.mean([clip(self.r(th, s, c, series)) ** k for c in cl]))

    def nll(self, th, obs, series):
        return 0.5 * sum(((o[3] - self.pred(th, o, series)) / o[4]) ** 2 for o in obs)

    def fit(self, obs):
        self.seen = sorted({c for o in obs for c in o[1]})
        series = sorted({o[0] for o in obs})
        best = None
        for x0 in (0.5, 0.8, 0.95):
            res = minimize(lambda th: self.nll(th, obs, series), np.full(self.n_par(series), x0), method="Nelder-Mead",
                           options={"maxiter": 20000, "xatol": 1e-5, "fatol": 1e-7})
            if best is None or res.fun < best.fun:
                best = res
        return best.x, series

    def heldout_ll(self, th, series, test_obs, rng, draws=4000):
        """Log-likelihood of a new series' observations."""
        if not self.random_series:
            return float(sum(-0.5 * ((o[3] - self.pred(th, o, series)) / o[4]) ** 2 - np.log(o[4] * np.sqrt(2 * np.pi))
                             for o in test_obs))
        # new series: draw its series effect from the fitted series effects (empirical + jitter)
        eff = self.series_effects(th, series)
        sd = np.std(eff) if len(eff) > 1 else 0.1
        ll = 0.0
        new = rng.choice(eff, draws) + rng.normal(0, sd * 0.5, draws)
        for o in test_obs:
            p = np.array([self.pred_new(th, o, e, series) for e in new])
            v = o[4] ** 2 + p.var()
            ll += -0.5 * (o[3] - p.mean()) ** 2 / v - 0.5 * np.log(2 * np.pi * v)
        return float(ll)


class M0(Model):
    name = "M0 no fade"
    def n_par(self, series): return 1
    def r(self, th, s, c, series): return 1.0
    def fit(self, obs): return np.zeros(1), sorted({o[0] for o in obs})


class M1(Model):
    name = "M1 one rate"
    def n_par(self, series): return 1
    def r(self, th, s, c, series): return th[0]


class M2(Model):
    name = "M2 rate per clone"
    def n_par(self, series): return len(CLONES)
    def r(self, th, s, c, series):
        if c not in self.seen:  # clone absent from training: mean of the fitted clones
            return float(np.mean([th[CLONES.index(x)] for x in self.seen]))
        return th[CLONES.index(c)]


class M3(Model):
    name = "M3 rate per series"
    random_series = True
    def n_par(self, series): return len(series)
    def r(self, th, s, c, series): return th[series.index(s)]
    def series_effects(self, th, series): return np.asarray(th)
    def pred_new(self, th, o, e, series):
        return float(np.mean([clip(e) ** o[2] for _ in o[1]]))


class M4(Model):
    name = "M4 clone rate x series multiplier"
    random_series = True
    def n_par(self, series): return len(CLONES) + len(series) - 1  # first series multiplier = 1
    def mult(self, th, s, series):
        i = series.index(s)
        return 1.0 if i == 0 else th[len(CLONES) + i - 1]
    def base(self, th, c):
        if c not in self.seen:
            return float(np.mean([th[CLONES.index(x)] for x in self.seen]))
        return th[CLONES.index(c)]
    def r(self, th, s, c, series): return self.base(th, c) * self.mult(th, s, series)
    def series_effects(self, th, series): return np.array([self.mult(th, s, series) for s in series])
    def pred_new(self, th, o, e, series):
        return float(np.mean([clip(self.base(th, c) * e) ** o[2] for c in o[1]]))


def main():
    obs = observations()
    models = [M0(), M1(), M2(), M3(), M4()]
    rng = np.random.default_rng(82)
    out = {"observations": [dict(series=o[0], clones=list(o[1]), k=o[2], ratio=round(o[3], 3), se=round(o[4], 3))
                            for o in obs], "models": {}}
    for m in models:
        cv = {}
        for s in SERIES:
            train = [o for o in obs if o[0] != s]
            test = [o for o in obs if o[0] == s]
            th, ser = m.fit(train)
            if not m.random_series:
                # clones absent from training (e.g. clone 5/6 only in late24) are predicted at the mean rate
                pass
            cv[s] = m.heldout_ll(th, ser, test, rng)
        th, ser = m.fit(obs)
        resid = [(o[0], o[1], o[2], round(o[3], 3), round(m.pred(th, o, ser), 3), round((o[3] - m.pred(th, o, ser)) / o[4], 2))
                 for o in obs]
        misses = [r for r in resid if abs(r[5]) > 2.5]
        out["models"][m.name] = {"cv_loglik_total": round(sum(cv.values()), 2),
                                 "cv_by_heldout_series": {k: round(v, 2) for k, v in cv.items()},
                                 "params": [round(float(x), 3) for x in th], "series_order": ser,
                                 "consistency_misses": [list(map(str, r)) for r in misses]}
    ranking = sorted(out["models"], key=lambda k: -out["models"][k]["cv_loglik_total"])
    out["ranking_by_cv"] = ranking
    qual = [k for k in ranking if not out["models"][k]["consistency_misses"]]
    out["best_qualifying"] = qual[0] if qual else None
    json.dump(out, open(os.path.join(OUT, "g8_memory.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("ranking_by_cv", "best_qualifying")}, indent=1))
    for k in ranking:
        v = out["models"][k]
        print(f"{k:38s} CV {v['cv_loglik_total']:9.2f}  misses {len(v['consistency_misses'])}  params {v['params']}")
        print("     by series:", v["cv_by_heldout_series"])
        for r in v["consistency_misses"]:
            print("     miss:", r)


if __name__ == "__main__":
    main()
