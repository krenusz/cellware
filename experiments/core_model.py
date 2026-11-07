"""One generative core for clone-level fate data (C1, EXPERIMENT_LOG 2026-11-06).

Every data set is a configuration of the same five steps, per clone:

1. **State.** A clone fate state z; each condition (well or sister arm) sees a noisy copy of it
   (fate memory). Fate logits = f(state) + instruction shift tau_c (+ label shift).
2. **Split.** Founder cells per condition: independent Poisson draws (cells seeded into wells,
   LARRY) or a partition of the clone's c cells across arms (transduction, SIS-skew).
3. **Growth.** Each founder line grows to exp(log_growth) (clone growth + condition growth +
   selection gamma x selection state), with optional per-line noise s_line and extinction q.
   In a perturbed condition a fraction e of lines carries the perturbation (editing efficiency).
4. **Output.** Mean cells of fate f = growth x (free x p_f + committed_f) x exp(sfate_f): p from
   the fate logits, committed_f = measured committed fraction (LARRY day 2), sfate = fate-specific
   cell-level expansion (LARRY's s_c; SIS's sfate).
5. **Sampling.** Poisson counts per fate (equivalent in distribution to a Poisson total split
   multinomially when sfate = 0).

Configurations: `core_larry.py` (reproduces E46/E48) and `core_sis.py` (R3/E52, reproduces R2
when e = 1 and sfate = 0).
"""

from __future__ import annotations

import numpy as np


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def state_copy(z, rho, rng):
    """Step 1 fate memory: a condition's copy of the clone state, corr(copy, z) = sqrt(rho) per unit
    variance (core_larry's RHO_F per well; gastruloid: per gastruloid and per passage event)."""
    z = np.asarray(z, float)
    return np.sqrt(rho) * z + np.sqrt(1 - rho) * rng.standard_normal(z.shape)


def split_poisson(rate, rng):
    """Independent founders per condition: rate is an array (clones,)."""
    return rng.poisson(rate)


def split_partition(n, c0, t, rng):
    """c = 1 + Poisson(c0) cells per clone, each joins the perturbed arm with probability t."""
    c = 1 + rng.poisson(c0, n)
    b = rng.binomial(c, t)
    return b, c - b


def grow(k, log_g, rng, s_line=0.0, q=0.0, e=1.0, log_g_edit=0.0, clip=(-30.0, 8.0)):
    """Total growth of k founder lines per clone, split into (unedited, edited) parts.

    log_g: (clones,) log growth shared by all lines of a clone in this condition; edited lines add
    log_g_edit. With s_line = q = 0 and e = 1 this is k * exp(log_g + log_g_edit) (LARRY)."""
    if s_line == 0.0 and q == 0.0 and e >= 1.0:
        return np.zeros(len(k)), k * np.exp(np.clip(log_g + log_g_edit, *clip))
    kmax = int(k.max()) if len(k) else 0
    if kmax == 0:
        z = np.zeros(len(k))
        return z, z.copy()
    idx = np.arange(kmax)[None, :]
    alive = (idx < k[:, None]) & (rng.random((len(k), kmax)) >= q)
    noise = s_line * rng.standard_normal((len(k), kmax))
    edited = rng.random((len(k), kmax)) < e
    base = np.exp(np.clip(log_g[:, None] + noise, *clip))
    g_u = (base * (alive & ~edited)).sum(1)
    g_e = (base * np.exp(np.clip(np.broadcast_to(np.asarray(log_g_edit, float), (len(k),))[:, None], *clip))
           * (alive & edited)).sum(1)
    return g_u, g_e


def emit(groups, rng, committed=None, sfate=None, cap=1e4):
    """Poisson counts per fate. groups: list of (growth (clones,), p (clones, K), sfate_group (K,)
    or None). committed: (clones, K) committed fractions (rows sum <= 1) or None."""
    n, K = groups[0][1].shape
    mean = np.zeros((n, K))
    free = 1.0 if committed is None else (1 - committed.sum(1, keepdims=True))
    com = 0.0 if committed is None else committed
    for g, p, sf in groups:
        m = g[:, None] * (free * p + com)
        if sf is not None:
            m = m * np.exp(np.asarray(sf))[None, :]
        if sfate is not None:
            m = m * np.exp(np.asarray(sfate))[None, :]
        mean += m
    return rng.poisson(np.clip(mean, 0, cap))
