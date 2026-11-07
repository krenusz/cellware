"""C3-S2: elongation rescue from the core configuration's sorting index. See EXPERIMENT_LOG C3-S2.

    uv run python c3s2_rescue_sorting.py

Model S: logit p = sum_i f_i e_i + kappa * E[H], H = B / (1 + B), B = sum_i f_i (zb_i - zbar)^2,
zb_i = per-gastruloid copy of clone state z_i (core_model.state_copy, rho_g); bulk score mean 0, no
copy. z and rho_g fixed from C3-S1 (polarity data only). Primary: G4's leave-one-composition-out CV
against M0 and Mgen. Data: G2 morphology table (open, used).
"""

from __future__ import annotations

import json
import os

import numpy as np

import core_model as cm
import g2b_robust as g2b
from g2_elongation import COMP, OUT, load
from g4_mechanism import M0, Mgen, cv, ll_rows

C3S1 = json.load(open(os.path.join(OUT, "c3s1_core_gastruloid.json")))
Z = {int(c): v for c, v in C3S1["states_z"].items()}  # clones 1, 3, 5, 6, 7
RHO_G = C3S1["rho_g"]
MC = 4000


def expected_H(F, zmap, rho_g, seed=2):
    """E[H] per row over per-gastruloid copies of each clone's state."""
    rng = np.random.default_rng(seed)
    z = np.array([0.0] + [zmap.get(i, 0.0) for i in range(1, 8)])
    is_clone = np.array([0.0] + [1.0] * 7)
    zb = cm.state_copy(np.broadcast_to(z, (MC, len(z))), rho_g, rng)
    zb = np.where(is_clone[None, :] > 0, zb, 0.0)  # bulk: score mean 0, no copy
    out = np.empty(len(F))
    for r, f in enumerate(F):
        zbar = zb @ f
        B = ((zb - zbar[:, None]) ** 2) @ f
        out[r] = float(np.mean(B / (1 + B)))
    return out


class S(M0):
    npar = len(COMP) + 1

    def __init__(self, F, cls, H):
        super().__init__(F, cls)
        self.H = H

    def eta(self, th, ix):
        return self.F[ix] @ th[:len(COMP)] + th[-1] * self.H[ix]


def class_excess(d, cls, p_row, rates):
    """Pooled (observed or predicted) rate minus pooled N2 (mean pure probability), 50:50 pairs."""
    out = {}
    comp = d["Composition"].str.strip()
    is_pair = comp.str.match(r"^50% clone \d \+ 50% clone \d$").to_numpy()
    ab = comp.str.findall(r"clone (\d)")
    for c in ("comp", "same", "unl"):
        m = is_pair & (cls == c)
        w = d["Total"].to_numpy(float)[m]
        n2 = np.array([(rates[int(a)] + rates[int(b)]) / 2 for a, b in ab[m]])
        out[c] = float((p_row[m] * w).sum() / w.sum() - (n2 * w).sum() / w.sum())
    out["comp_minus_same"] = out["comp"] - out["same"]
    return out


def main(seed=67, B=1000):
    rng = np.random.default_rng(seed)
    d, F, cls = load()
    k, n = d["Elongated"].to_numpy(float), d["Total"].to_numpy(float)
    groups = d["Composition"].str.strip().to_numpy()
    dates = d["Date"].to_numpy()
    H = expected_H(F, Z, RHO_G)
    allix = np.arange(len(d))
    res = {"z": Z, "rho_g": RHO_G, "E_H_by_composition": {}}
    for g in np.unique(groups):
        res["E_H_by_composition"][g] = round(float(H[groups == g][0]), 3)

    models = {"M0": M0(F, cls), "Mgen": Mgen(F, cls), "S": S(F, cls, H)}
    res["cv_loglik"], res["full_fit_loglik"], fits = {}, {}, {}
    for name, m in models.items():
        tot, _ = cv(m, d, F, k, n, groups, rng)
        th = m.fit(allix, k, n, rng)
        fits[name] = th
        res["cv_loglik"][name] = round(tot, 2)
        res["full_fit_loglik"][name] = round(float(ll_rows(m.eta(th, allix), k, n).sum()), 2)
        print(name, round(tot, 2), flush=True)
    kappa = float(fits["S"][-1])

    # kappa: cluster bootstrap over dates
    ud = np.unique(dates)
    ks = []
    for _ in range(B):
        pick = rng.choice(ud, len(ud))
        ix = np.concatenate([np.where(dates == u)[0] for u in pick])
        ks.append(models["S"].fit(ix, k, n, rng)[-1])
    res["kappa"] = [round(kappa, 3)] + [round(float(v), 3) for v in np.percentile(ks, [2.5, 97.5])]

    cvl = res["cv_loglik"]
    if kappa <= 0 or cvl["S"] < cvl["M0"] + 2:
        reading = "fails"
    elif cvl["S"] >= cvl["Mgen"] - 2:
        reading = "rescue from local rules"
    else:
        reading = "partial"
    res["reading"] = reading

    # secondary 1: complementary - same excess over N2 (G2b test 1)
    pure, _ = g2b.load()
    rates = g2b.rates(pure)
    obs_p = k / n
    pS = 1 / (1 + np.exp(-models["S"].eta(fits["S"], allix)))
    res["excess_over_N2"] = {"observed": {a: round(b, 3) for a, b in class_excess(d, cls, obs_p, rates).items()},
                             "S_predicted": {a: round(b, 3) for a, b in class_excess(d, cls, pS, rates).items()}}

    # secondary 2: clone + bulk chimeras, pooled per composition
    p0 = 1 / (1 + np.exp(-models["M0"].eta(fits["M0"], allix)))
    cb = {}
    for g in np.unique(groups):
        if "clone" in g and "+" not in g and not g.startswith("100%"):
            m = groups == g
            w = n[m]
            cb[g] = {"obs": round(float(k[m].sum() / w.sum()), 3), "S": round(float((pS[m] * w).sum() / w.sum()), 3),
                     "M0": round(float((p0[m] * w).sum() / w.sum()), 3), "n": int(w.sum()), "E_H": round(float(H[m][0]), 3)}
    res["clone_plus_bulk"] = cb
    res["clone_plus_bulk_abs_err"] = {mn: round(float(np.mean([abs(v[mn] - v["obs"]) for v in cb.values()])), 3)
                                      for mn in ("S", "M0")}

    # sensitivity: drop rows with clones 2 or 4
    keep = (F[:, COMP.index("c2")] == 0) & (F[:, COMP.index("c4")] == 0)
    dk = d[keep].reset_index(drop=True)
    sens = {}
    for name, m in (("M0", M0(F[keep], cls[keep])), ("Mgen", Mgen(F[keep], cls[keep])), ("S", S(F[keep], cls[keep], H[keep]))):
        sens[name] = round(cv(m, dk, F[keep], k[keep], n[keep], groups[keep], rng)[0], 2)
    res["sensitivity_without_2_4_cv"] = sens

    json.dump(res, open(os.path.join(OUT, "c3s2_rescue_sorting.json"), "w"), indent=1)
    print(json.dumps({a: b for a, b in res.items() if a != "E_H_by_composition"}, indent=1))


if __name__ == "__main__":
    main()
