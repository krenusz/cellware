"""B39: what does Bcor loss do to sister clones? SIS-skew, KO wells (C1-C3) vs null wells (W1, W2),
fit set only. Arm + = Cherry+ (KO in C wells, control vector in W wells), arm - = Cherry- (WT).

    uv run python b39_sisskew.py [--synthetic syn0|syn1] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

import sis_data as sd

GROUPS = {"KO": ("C1", "C2", "C3"), "null": ("W1", "W2")}
SIZE_BINS = [(2, 5), (6, 20), (21, 100), (101, np.inf)]


def decompose(P, N):
    """Population shift f+ - f- per fate = within + reweighting + interaction (both-arm clones)."""
    wp, wn = P.sum(1) / P.sum(), N.sum(1) / N.sum()
    pp, pn = sd._frac(P), sd._frac(N)
    within = (wn[:, None] * (pp - pn)).sum(0)
    rew = ((wp - wn)[:, None] * pn).sum(0)
    inter = ((wp - wn)[:, None] * (pp - pn)).sum(0)
    return np.concatenate([within, rew, inter, within + rew + inter])


def size_ratio_rho(P, N):
    m = N.sum(1) > 10
    lr = np.log(P[m].sum(1) / N[m].sum(1))
    pn = sd._frac(N[m])
    return [sd.spearman(lr, pn[:, j]) for j in range(3)]


def one_arm(P, N):
    tot = P.sum(1) + N.sum(1)
    pres_p, pres_n = P.sum(1) > 1, N.sum(1) > 1
    out = {}
    for lo, hi in SIZE_BINS:
        m = (tot >= lo) & (tot <= hi)
        key = f"{lo}-{hi if np.isfinite(hi) else 'inf'}"
        out[key] = {"n": int(m.sum()),
                    "one_arm": float(np.mean(pres_p[m] ^ pres_n[m])) if m.any() else None,
                    "plus_only": float(np.mean(pres_p[m] & ~pres_n[m])) if m.any() else None,
                    "minus_only": float(np.mean(~pres_p[m] & pres_n[m])) if m.any() else None}
    return out


def analyse(P, N, well, rng, boot):
    both = (P.sum(1) > 3) & (N.sum(1) > 3)
    Pb, Nb, wb = P[both], N[both], well[both]
    k1, k2 = np.rint(Pb).astype(int), np.rint(Nb).astype(int)
    ok = (k1.sum(1) > 0) & (k2.sum(1) > 0)
    k1, k2, wk = k1[ok], k2[ok], wb[ok]
    n = len(Pb)
    h = sd.heritable_share(k1, k2, wk, rng)
    h_ci = sd.bootstrap(lambda ix: sd.heritable_share(k1[ix], k2[ix], wk[ix], rng, 100, 50)["H"],
                        len(k1), rng, boot)
    dec = decompose(Pb, Nb)
    dec_ci = sd.bootstrap(lambda ix: decompose(Pb[ix], Nb[ix]), n, rng, boot)
    names = ["within", "reweighting", "interaction", "total"]
    dec_out = {nm: {f: {"est": float(dec[i * 3 + j]), "ci": [dec_ci[0][i * 3 + j], dec_ci[1][i * 3 + j]]}
                    for j, f in enumerate(sd.FATES)} for i, nm in enumerate(names)}
    rho = size_ratio_rho(Pb, Nb)
    rho_ci = sd.bootstrap(lambda ix: size_ratio_rho(Pb[ix], Nb[ix]), n, rng, boot)
    return {"n_clones": int(len(P)), "n_both_arm": int(n), **h, "H_ci": h_ci,
            "decomposition": dec_out,
            "size_ratio_rho": {f: {"est": rho[j], "ci": [rho_ci[0][j], rho_ci[1][j]]}
                               for j, f in enumerate(sd.FATES)},
            "n_wt_gt10": int((Nb.sum(1) > 10).sum()),
            "one_arm": one_arm(P, N)}


def main(synthetic=None, out=None, boot=1000):
    rng = np.random.default_rng(39)
    d = sd.load_skew("fit")
    res = {"synthetic": synthetic}
    for g, wells in GROUPS.items():
        m = np.isin(d["well"], wells)
        P, N, well = d["P"][m], d["N"][m], d["well"][m]
        if synthetic:
            both = (P.sum(1) > 3) & (N.sum(1) > 3)
            k1, k2 = sd.synthetic(np.rint(P[both]).astype(int), np.rint(N[both]).astype(int),
                                  well[both], synthetic, rng)
            P, N = P.copy(), N.copy()
            P[both], N[both] = k1.astype(float), k2.astype(float)
        res[g] = analyse(P, N, well, rng, boot)
    res["reading"] = ("underpowered" if res["KO"]["n_both_arm"] < 150 or res["null"]["n_both_arm"] < 100
                      else "powered")
    out = out or os.path.join(os.path.dirname(__file__), "results_sis")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"b39{'_' + synthetic if synthetic else ''}.json")
    json.dump(res, open(path, "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=lambda x: round(float(x), 3)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", choices=["syn0", "syn1"])
    ap.add_argument("--out")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    main(a.synthetic, a.out, a.boot)
