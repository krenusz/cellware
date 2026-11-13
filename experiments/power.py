"""One power rule for the three gates (program.md: scan H19.2, selection H17, oracle H14) and the checks
on a selection file's power and readout blocks (H21). Stdlib only, so the guard tests need no scipy.

Every new script takes its power from here. The closed-chapter scripts listed in
tests/test_guards.py (LEGACY_POWER) keep their own copies, because their looks map to past commits.

    from power import mean_power, rho_power, sim_power
"""

from __future__ import annotations

import math
from statistics import NormalDist

MIN_POWER = 0.8  # floor: a selection file may ask for more, never for less
MAX_ALPHA = 0.05
MIN_SIDE = 30  # H21: units on each side of a binary split
STRICT_FROM = "2026-11-13"  # pre-registrations from this date also need sources and a readout block
_N = NormalDist()


def z_crit(alpha, one_sided):
    if not 0 < alpha <= MAX_ALPHA:
        raise ValueError(f"alpha {alpha} outside (0, {MAX_ALPHA}]")
    return _N.inv_cdf(1 - alpha) if one_sided else _N.inv_cdf(1 - alpha / 2)


def z_power(effect, se, alpha=0.025, one_sided=True):
    """Power of a z test for a true effect in the pre-registered direction (effect > 0 = predicted sign).
    The far tail of a two-sided test is ignored, so the value is slightly conservative."""
    if not se > 0 or math.isnan(effect):
        raise ValueError(f"se {se} must be > 0 and effect {effect} a number")
    return 1 - _N.cdf(z_crit(alpha, one_sided) - effect / se)


def contrast_se(sd, counts, weights=None):
    """SE of a weighted contrast of group means: sd * sqrt(sum_g w_g^2 / n_g). counts = {group: n units}."""
    weights = weights or {}
    if not counts or min(counts.values()) < 1:
        raise ValueError(f"counts {counts}: a group is empty")
    return sd * math.sqrt(sum(weights.get(g, 1.0) ** 2 / n for g, n in counts.items()))


def mean_power(effect, sd, counts, weights=None, alpha=0.025, one_sided=True):
    """Power for a mean or a contrast of means. sd is the per-unit sd at the test's own unit."""
    return z_power(effect, contrast_se(sd, counts, weights), alpha, one_sided)


def rho_power(rho, n, alpha=0.025, one_sided=True, fisher_sd=1.0):
    """Power for a correlation by the Fisher z transform: se = fisher_sd / sqrt(n - 3).
    fisher_sd is 1 for Pearson; about 1.03 for Spearman."""
    if n <= 3:
        return 0.0
    return z_power(math.atanh(rho), fisher_sd / math.sqrt(n - 3), alpha, one_sided)


def sim_power(stats, null_sd, alpha=0.025, one_sided=True):
    """Power from simulated test statistics under the model's own prediction (H17): the share of
    simulated statistics beyond the critical value z * null_sd."""
    stats = list(stats)
    if not stats:
        raise ValueError("no simulated statistics")
    crit = z_crit(alpha, one_sided) * null_sd
    return sum(s > crit for s in stats) / len(stats)


def block_errors(sel, strict):
    """Problems in a selection file's power block (and, when strict, its sources and readout block)."""
    p = sel.get("power")
    if not isinstance(p, dict):
        return ["no power block (H14)"]
    errs = []
    for k in ("sd", "effect", "alpha"):
        if not isinstance(p.get(k), (int, float)) or isinstance(p.get(k), bool):
            errs.append(f"power.{k} missing or not a number (H14)")
    if errs:
        return errs
    if not p["sd"] > 0:
        errs.append("power.sd must be > 0 (H14)")
    if not 0 < p["alpha"] <= MAX_ALPHA:
        errs.append(f"power.alpha {p['alpha']} outside (0, {MAX_ALPHA}] (H14)")
    if not strict:
        return errs
    src = p.get("sources")
    for k in ("effect", "sd"):
        if not (isinstance(src, dict) and str(src.get(k, "")).strip()):
            errs.append(
                f"power.sources.{k} missing: name the used result it comes from (power rule)"
            )
    r = sel.get("readout_gate")
    if not isinstance(r, dict):
        return errs + ["no readout_gate block (H21)"]
    if r.get("pass") is not True:
        errs.append("readout_gate.pass is not true (H21)")
    sides = r.get("sides")
    if r.get("binary_split") is False:
        if not str(r.get("why_not_binary", "")).strip():
            errs.append("readout_gate.binary_split is false without why_not_binary (H21)")
    elif not (isinstance(sides, dict) and len(sides) >= 2):
        errs.append(
            "readout_gate.sides needs the unit count on each side, or binary_split: false (H21)"
        )
    else:
        low = {k: v for k, v in sides.items() if not isinstance(v, (int, float)) or v < MIN_SIDE}
        if low:
            errs.append(f"readout_gate.sides below {MIN_SIDE} units: {low} (H21)")
    return errs


def gate(sel, counts, strict=False):
    """The oracle's power gate (H14). Returns (power or None, refusal message or None).
    counts come from the sealed partition's structure only (H9)."""
    errs = block_errors(sel, strict)
    if errs:
        return None, "; ".join(errs)
    if not counts or min(counts.values()) < 2:
        return None, f"counts {counts}: a group is empty (H14)"
    p = sel["power"]
    need = max(float(p.get("min_power", MIN_POWER)), MIN_POWER)
    power = mean_power(
        p["effect"], p["sd"], counts, p.get("weights"), p["alpha"], bool(p.get("one_sided"))
    )
    if power < need:
        return power, f"power {power:.2f} < {need} (counts {counts}); partition NOT spent (H14)"
    return power, None
