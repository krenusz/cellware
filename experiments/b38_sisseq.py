"""B38: is DC fate fixed in the clone before the split? SIS-seq sister wells, fit set only.

    uv run python b38_sisseq.py [--synthetic syn0|syn1] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

import sis_data as sd

LARGE = ("ILSGR0260r", "JS96")


def analyse(k1, k2, bg1, bg2, batch, rng, boot=1000):
    n = len(k1)
    h = sd.heritable_share(k1, k2, batch, rng)
    h_ci = sd.bootstrap(lambda ix: sd.heritable_share(k1[ix], k2[ix], batch[ix], rng, 100, 50)["H"],
                        n, rng, boot)
    sig = sd.clone_logit_sd(k1, k2)
    sig_ci = sd.bootstrap(lambda ix: sd.clone_logit_sd(k1[ix], k2[ix]), n, rng, boot)
    dp = sd._frac(k1) - sd._frac(k2)
    dbg = bg1 - bg2
    env = [sd.spearman(dp[:, j], dbg[:, j]) for j in range(3)]
    env_ci = sd.bootstrap(lambda ix: [sd.spearman(dp[ix, j], dbg[ix, j]) for j in range(3)], n, rng,
                          boot)
    return {"n_clones": n, **h, "H_ci": h_ci,
            "sigma_logit": dict(zip(sd.FATES, sig)),
            "sigma_ci": {f: [sig_ci[0][j], sig_ci[1][j]] for j, f in enumerate(sd.FATES)},
            "env_rho": dict(zip(sd.FATES, env)),
            "env_ci": {f: [env_ci[0][j], env_ci[1][j]] for j, f in enumerate(sd.FATES)}}


def main(synthetic=None, out=None, boot=1000):
    rng = np.random.default_rng(38)
    d = sd.load_seq("fit")
    m = (d["k1"].sum(1) > 3) & (d["k2"].sum(1) > 3)
    k1, k2, bg1, bg2, batch = d["k1"][m], d["k2"][m], d["bg1"][m], d["bg2"][m], d["batch"][m]
    if synthetic:
        k1, k2 = sd.synthetic(k1, k2, batch, synthetic, rng)
    res = {"synthetic": synthetic, "fit_clones_total": int(len(d["title"])),
           "batches": {b: int((batch == b).sum()) for b in np.unique(batch)}}
    if len(k1) < 40:
        res["reading"] = "underpowered"
    res["all"] = analyse(k1, k2, bg1, bg2, batch, rng, boot)
    lg = np.isin(batch, LARGE)
    res["large_batches"] = analyse(k1[lg], k2[lg], bg1[lg], bg2[lg], batch[lg], rng, boot)
    out = out or os.path.join(os.path.dirname(__file__), "results_sis")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"b38{'_' + synthetic if synthetic else ''}.json")
    json.dump(res, open(path, "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=lambda x: round(float(x), 3)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", choices=["syn0", "syn1"])
    ap.add_argument("--out")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    main(a.synthetic, a.out, a.boot)
