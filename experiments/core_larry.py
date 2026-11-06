"""LARRY cytokine configuration of the core model (C1): reproduces e46/e48 `simulate`.

Mapping (E46/E48 parameter -> core step):
  w, measured m           -> state z = w m + sqrt(1 - w^2) noise (1-d)
  RHO_F                   -> per-well state copy zb (fate memory over the split)
  beta, tau_c             -> fate logits beta zb + tau_c for neutrophil vs monocyte (K = 2)
  nu, nu_full, phi, eps   -> split: independent Poisson founders, two clone classes
  mu_c, sigma_a, rho_a    -> growth (clone growth a; the full arm sees a correlated copy)
  gamma, SIGN             -> selection: log growth + gamma SIGN_c zb (G +, M -)
  kappa, measured frac_N/M -> committed fractions
  s_c                     -> sfate: fate-specific expansion of neutrophils, log s_c
"""

from __future__ import annotations

import numpy as np

import core_model as cm
import e33_measured as e33
from e20_gap import RHO_F

CONDS = e33.CONDS
SIGN = {"G": 1.0, "M": -1.0}


def simulate(clones, prm, rng, meas):
    n = len(clones)
    m, fn, fm = meas[:, 0], meas[:, 1], meas[:, 2]
    w = prm["w"]
    z = w * m + np.sqrt(1 - w**2) * rng.standard_normal(n)                      # 1. state
    a = prm["sigma_a"] * rng.standard_normal(n)                                  # 3. clone growth
    ra = prm["rho_a"]
    a_full = ra * a + np.sqrt(1 - ra**2) * prm["sigma_a"] * rng.standard_normal(n)
    cn = np.clip(prm["kappa"] * fn, 0, 1)                                        # 4. committed
    cmt = np.clip(prm["kappa"] * fm, 0, 1 - cn)
    committed = np.column_stack([cn, cmt])
    u = rng.random(n) < prm["phi"]                                               # clone class
    eps = prm["eps"]
    gamma = prm.get("gamma", 0.0)
    out, cl = {}, np.asarray(clones)
    for c in CONDS:
        full = c == "full"
        rate = prm["nu_full"] * np.where(u, 1.0, eps) if full else prm["nu"] * np.where(u, eps, 1.0)
        k = cm.split_poisson(rate, rng)                                          # 2. split
        zb = np.sqrt(RHO_F) * z + np.sqrt(1 - RHO_F) * rng.standard_normal(n)   # 1. per-well copy
        logit = prm["beta"] * zb + prm["tau"][c]
        p_n = 1 / (1 + np.exp(-logit))
        p = np.column_stack([p_n, 1 - p_n])
        log_g = prm["mu"][c] + (a_full if full else a) + gamma * SIGN.get(c, 0.0) * zb
        _, g = cm.grow(k, log_g, rng)                                            # 3. growth
        sf = np.array([np.log(prm["s"][c]), 0.0])                                # 4. sfate
        x = cm.emit([(g, p, sf)], rng, committed=committed)                      # 5. sampling
        keep = np.flatnonzero(x.sum(1) > 0)
        out[c] = {4: {kk: [a1, a2, 0] for kk, a1, a2 in zip(cl[keep].tolist(), x[keep, 0].tolist(),
                                                           x[keep, 1].tolist())}}
    return out
