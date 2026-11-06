"""SIS-skew configuration of the core model (C1) and the editing-aware fit (R3 / E52, B42).

Mapping (core step -> SIS-skew):
  state      z ~ N(mu_group, diag(SIGMA_Z^2)) over (cDC1, cDC2, pDC); each arm adds logit noise s_eps
  split      partition: c = 1 + Poisson(c0) cells, Cherry+ with probability t_W / t_C
  growth     lines exp(m_g + s_g noise + s_line noise), extinction q; + arm adds lam_L; in C wells
             a fraction e of + lines is edited and adds lam_K + gamma . zs (selection on the clone's
             standardised cDC2 / pDC logits relative to cDC1)
  output     fate logits + label delta (+ arm) + tau (edited lines); sfate on edited lines
  sampling   Poisson per fate

    uv run python core_sis.py equiv      # C1: SIS config (e = 1, sfate = 0) vs r2_skew_sbi.simulate
    uv run python core_sis.py bank       # R3: 20,000 prior simulations
    uv run python core_sis.py recover    # R3: 300 recovery tests
    uv run python core_sis.py fit        # E52: fit set
    uv run python core_sis.py b42        # B42: pDC gain test
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

import core_model as cm
import r2_skew_sbi as r2
from r1_skew_sim import N_KO, N_NULL, SIGMA_Z

OUT = r2.OUT
BANK = os.path.join(OUT, "r3_bank.npz")
PRIOR = {**r2.PRIOR, "e": (0.15, 0.55), "sfate_cDC2": (-1.5, 1.5), "sfate_pDC": (-1.5, 1.5)}
NAMES = list(PRIOR)
MAIN = ["gamma_cDC2", "gamma_pDC", "tau_cDC2", "tau_pDC", "sfate_cDC2", "sfate_pDC", "e"]


def simulate(p, n, ko, rng):
    w = "C" if ko else "W"
    m = int(n * 1.8) + 60
    mu = np.array([0.0, p[f"mu_cDC2_{w}"], p[f"mu_pDC_{w}"]])
    z = mu + rng.standard_normal((m, 3)) * SIGMA_Z                              # 1. state
    rel = z[:, 1:] - z[:, :1]
    zs = (rel - rel.mean(0)) / rel.std(0)
    b, nb = cm.split_partition(m, p["c0"], p[f"t_{w}"], rng)                   # 2. split
    g = p["m_g"] + p["s_g"] * rng.standard_normal(m)                           # 3. growth
    e = p.get("e", 1.0) if ko else 1.0
    edit_add = (p["lam_K"] + p["gamma_cDC2"] * zs[:, 0] + p["gamma_pDC"] * zs[:, 1]) if ko else 0.0
    gp_u, gp_e = cm.grow(b, g + p["lam_L"], rng, p["s_line"], p["q"], e, edit_add)
    gn_u, gn_e = cm.grow(nb, g, rng, p["s_line"], p["q"], 1.0, 0.0)
    gn = gn_u + gn_e
    delta = np.array([0.0, p["delta_cDC2"], p["delta_pDC"]])
    tau = np.array([0.0, p["tau_cDC2"], p["tau_pDC"]]) if ko else np.zeros(3)
    if ko and p.get("dlabel_C", 0.0):  # S1: extra label shift of the + arm in C wells only
        delta = delta + np.array([0.0, p["dlabel_C"], p["dlabel_C"]])
    sf = np.array([0.0, p.get("sfate_cDC2", 0.0), p.get("sfate_pDC", 0.0)]) if ko else None
    lp = z + delta + p["s_eps"] * rng.standard_normal((m, 3))                  # 4. output
    ln = z + p["s_eps"] * rng.standard_normal((m, 3))
    if ko:
        P = cm.emit([(gp_u, cm.softmax(lp), None), (gp_e, cm.softmax(lp + tau), sf)], rng)
    else:
        P = cm.emit([(gp_u + gp_e, cm.softmax(lp), None)], rng)
    N = cm.emit([(gn, cm.softmax(ln), None)], rng)                             # 5. sampling
    P, N = P.astype(float), N.astype(float)
    keep = np.flatnonzero((P.sum(1) + N.sum(1)) > 1)[:n]
    return P[keep], N[keep]


def sim_stats(p, rng, n_null=N_NULL, n_ko=N_KO):
    return np.concatenate([r2.svec(*simulate(p, n_null, False, rng)), r2.svec(*simulate(p, n_ko, True, rng))])


def draw(rng):
    return {k: rng.uniform(*r) for k, r in PRIOR.items()}


# ------------------------------------------------------------------ C1 equivalence (SIS)

def equiv(nsim=300, seed=31):
    post = json.load(open(os.path.join(OUT, "e51_fit.json")))["posterior"]
    p = {k: v["median"] for k, v in post.items()}
    p1 = {**p, "e": 1.0, "sfate_cDC2": 0.0, "sfate_pDC": 0.0}
    rng = np.random.default_rng(seed)
    a = np.array([r2.sim_stats(p, rng) for _ in range(nsim)])
    b = np.array([sim_stats(p1, rng) for _ in range(nsim)])
    d = np.abs(a.mean(0) - b.mean(0))
    se = np.sqrt(a.var(0) / nsim + b.var(0) / nsim)
    ok = (d <= 3 * se) | (d <= 0.01)
    res = {"n_stats": int(len(d)), "share_pass": float(ok.mean()), "fails": [int(i) for i in np.flatnonzero(~ok)],
           "equivalent": bool(ok.mean() >= 0.95)}
    json.dump(res, open(os.path.join(OUT, "c1_equiv_sis.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


# ------------------------------------------------------------------ R3 / E52

def tag(d):
    return "" if d is None else f"_d{'+' if d > 0 else '-'}{abs(d):.1f}"


def bank(n=20000, seed=32, d=None):
    rng = np.random.default_rng(seed)
    th, S = np.zeros((n, len(NAMES))), []
    for i in range(n):
        p = draw(rng)
        th[i] = [p[k] for k in NAMES]
        if d is not None:
            p["dlabel_C"] = d
        S.append(sim_stats(p, rng))
    np.savez_compressed(BANK.replace(".npz", tag(d) + ".npz"), theta=th, S=np.array(S))
    print("bank", th.shape)


class ABC3(r2.ABC):
    def posterior(self, s):
        t = self.project(s)
        dist = np.sqrt((((self.proj - t) / self.ps) ** 2).sum(1))
        k = int(len(dist) * self.frac)
        ix = np.argsort(dist)[:k]
        h = dist[ix].max() + 1e-12
        wts = 1 - (dist[ix] / h) ** 2
        X = np.column_stack([np.ones(k), self.proj[ix] - t])
        sw = np.sqrt(wts)[:, None]
        B = np.linalg.lstsq(X * sw, self.th[ix] * sw, rcond=None)[0]
        adj = self.th[ix] - (self.proj[ix] - t) @ B[1:]
        lo = np.array([PRIOR[q][0] for q in NAMES])
        hi = np.array([PRIOR[q][1] for q in NAMES])
        return np.clip(adj, lo, hi), wts, ix


def summarise(adj, wts):
    out = {}
    for j, nm in enumerate(NAMES):
        o = np.argsort(adj[:, j])
        cw = np.cumsum(wts[o]) / wts.sum()
        q = lambda a: float(adj[o, j][min(np.searchsorted(cw, a), len(o) - 1)])  # noqa: E731
        out[nm] = {"median": q(0.5), "lo": q(0.025), "hi": q(0.975)}
    return out


def load_abc(d=None):
    z = np.load(BANK.replace(".npz", tag(d) + ".npz"))
    return ABC3(z["theta"], z["S"])


def recover(n_test=300, seed=33):
    rng = np.random.default_rng(seed)
    abc = load_abc()
    rows = []
    for _ in range(n_test):
        p = draw(rng)
        rows.append((p, summarise(*abc.posterior(sim_stats(p, rng))[:2])))
    res = {"n_test": n_test}
    for nm in NAMES:
        tru = np.array([p[nm] for p, _ in rows])
        med = np.array([q[nm]["median"] for _, q in rows])
        res[nm] = {"RMSE": float(np.sqrt(np.mean((med - tru) ** 2))),
                   "coverage95": float(np.mean([q[nm]["lo"] <= p[nm] <= q[nm]["hi"] for p, q in rows])),
                   "mean_width": float(np.mean([q[nm]["hi"] - q[nm]["lo"] for _, q in rows])),
                   "R2": float(1 - np.mean((med - tru) ** 2) / np.var(tru))}
    for nm in MAIN:
        r = res[nm]
        r["reading"] = ("recoverable" if r["RMSE"] <= 0.3 and 0.90 <= r["coverage95"] <= 0.99
                        else "miscalibrated" if r["coverage95"] < 0.90 else "not recoverable")
    json.dump(res, open(os.path.join(OUT, "r3_recovery.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in MAIN}, indent=1))


def predictive(adj, wts, rng, n_draws, n_null=N_NULL, n_ko=N_KO, override=None, fn=sim_stats):
    ix = rng.choice(len(adj), n_draws, p=wts / wts.sum())
    out = []
    for i in ix:
        p = dict(zip(NAMES, adj[i]))
        if override:
            p.update(override)
        out.append(fn(p, rng, n_null, n_ko))
    return np.array(out)


def fit(seed=34, d=None):
    rng = np.random.default_rng(seed)
    abc = load_abc(d)
    s = r2.observed_fit()
    adj, wts, _ = abc.posterior(s)
    post = summarise(adj, wts)
    sims = predictive(adj, wts, rng, 200, override=None if d is None else {"dlabel_C": d})
    lo, hi = np.percentile(sims, 2.5, axis=0), np.percentile(sims, 97.5, axis=0)
    inside = (s >= lo) & (s <= hi)
    g = {nm: post[nm] for nm in ("gamma_cDC2", "gamma_pDC")}
    res = {"posterior": post, "fit_adequacy_share_inside": float(inside.mean()),
           "misses": [int(i) for i in np.flatnonzero(~inside)],
           "reading_selection": ("selection" if any(v["lo"] > 0 or v["hi"] < 0 for v in g.values())
                                 else "none (negligible)" if all(-0.25 <= v["lo"] and v["hi"] <= 0.25 for v in g.values())
                                 else "undetermined")}
    name = "e52" if d is None else "s1" + tag(d)
    np.savez_compressed(os.path.join(OUT, f"{name}_posterior.npz"), adj=adj, wts=wts)
    json.dump(res, open(os.path.join(OUT, f"{name}_fit.json"), "w"), indent=1)
    print(json.dumps({"posterior": post, "adequacy": res["fit_adequacy_share_inside"],
                      "misses": res["misses"], "reading_selection": res["reading_selection"]}, indent=1))


# ------------------------------------------------------------------ B42

def pdc_gain(P, N):
    both = (P.sum(1) > 1) & (N.sum(1) > 1)
    pp, pn = P[both, 2] > 1, N[both, 2] > 1
    G, L = float(np.mean(pp & ~pn)), float(np.mean(~pp & pn))
    return np.array([G, L, G - L])


def gain_stats(p, rng, n_null=N_NULL, n_ko=N_KO):
    return np.concatenate([pdc_gain(*simulate(p, n_null, False, rng)), pdc_gain(*simulate(p, n_ko, True, rng))])


def b42(seed=42, n_draws=400, d=None):
    import sis_data as sd
    from b39_sisskew import GROUPS
    rng = np.random.default_rng(seed)
    d = sd.load_skew("fit")
    obs = np.concatenate([pdc_gain(d["P"][np.isin(d["well"], GROUPS[g])], d["N"][np.isin(d["well"], GROUPS[g])])
                          for g in ("null", "KO")])
    name = "e52" if d is None else "s1" + tag(d)
    extra = {} if d is None else {"dlabel_C": d}
    z = np.load(os.path.join(OUT, f"{name}_posterior.npz"))
    full = predictive(z["adj"], z["wts"], rng, n_draws, fn=gain_stats, override=extra or None)
    nofate = predictive(z["adj"], z["wts"], rng, n_draws, fn=gain_stats,
                        override={"tau_cDC2": 0.0, "tau_pDC": 0.0, "sfate_cDC2": 0.0, "sfate_pDC": 0.0, **extra})
    lab = ["null_G", "null_L", "null_Net", "KO_G", "KO_L", "KO_Net"]
    res = {}
    for j, nm in enumerate(lab):
        res[nm] = {"obs": float(obs[j]),
                   "full_95": np.percentile(full[:, j], [2.5, 97.5]).tolist(),
                   "no_fate_effect_95": np.percentile(nofate[:, j], [2.5, 97.5]).tolist()}
    inn = lambda nm, key: res[nm][key][0] <= res[nm]["obs"] <= res[nm][key][1]  # noqa: E731
    in_nf = inn("KO_G", "no_fate_effect_95") and inn("KO_Net", "no_fate_effect_95")
    in_f = inn("KO_G", "full_95") and inn("KO_Net", "full_95")
    res["reading"] = ("expansion plus sampling explains it" if in_nf
                      else "needs a fate effect" if in_f else "unexplained")
    json.dump(res, open(os.path.join(OUT, "b42.json" if d is None else f"b42_{name}.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["equiv", "bank", "recover", "fit", "b42"])
    ap.add_argument("--d", type=float, default=None, help="S1: fixed C-well label shift")
    a = ap.parse_args()
    if a.cmd in ("bank", "fit", "b42"):
        {"bank": bank, "fit": fit, "b42": b42}[a.cmd](d=a.d)
    else:
        {"equiv": equiv, "recover": recover}[a.cmd]()
