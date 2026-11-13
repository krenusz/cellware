"""C10-S0 amendment 2: SYNTHETIC smoke test of the candidate instruction statistic (no data read).

S = log F(drug library) - log F(control library), F = between-family residual MS of post family means
(after regressing them on the pre-sister state) / (within-family post-cell MS / mean n).
Checks: (1) invariant to a constant / scale on one library; (2) ~0 in a selection-only world with range
restriction and one noisy pre cell per family; (3) > 0 when family-specific instruction is added.
"""
import numpy as np

D = 5            # state dimensions (stand-in for top PCs)
SIG_CELL = 1.0   # cell-level noise (pre and post)


def library(rng, n_fam, keep_frac, tau, memory=0.7, drift=0.5, shift=0.0, scale=1.0, n_post=(2, 5)):
    """One drug/control pair arm. Lineage states x ~ N(0, I); survivors = top keep_frac on x[0] (selection).
    Post family mean = memory*x + drift + instruction u_i (sd tau). One pre cell per family (noisy)."""
    n_all = int(n_fam / keep_frac)
    x = rng.normal(size=(n_all, D))
    keep = np.argsort(-x[:, 0])[:n_fam]
    x = x[keep]
    pre = x + rng.normal(scale=SIG_CELL, size=x.shape)             # one pre sister cell
    mu = memory * x + drift + rng.normal(scale=tau, size=x.shape) if tau else memory * x + drift
    fams = []
    for i in range(n_fam):
        k = rng.integers(n_post[0], n_post[1] + 1)
        cells = mu[i] + rng.normal(scale=SIG_CELL, size=(k, D))
        fams.append(scale * cells + shift)
    return scale * pre + shift, fams


def logF(pre, fams):
    means = np.array([f.mean(0) for f in fams])
    X = np.column_stack([np.ones(len(pre)), pre])
    beta, *_ = np.linalg.lstsq(X, means, rcond=None)
    resid = means - X @ beta
    dof_b = len(means) - X.shape[1]
    ms_between = (resid ** 2).sum() / (dof_b * D)
    within = sum(((f - f.mean(0)) ** 2).sum() for f in fams)
    dof_w = sum(len(f) - 1 for f in fams) * D
    nbar = np.mean([len(f) for f in fams])
    # F compares between-family residual variance to what cell noise alone would give a family mean
    return np.log(ms_between / (within / dof_w / nbar))


def S(rng, n_fam, drug_keep, ctrl_keep, tau, **kw):
    pd_, fd = library(rng, n_fam, drug_keep, tau, **kw)
    pc, fc = library(rng, n_fam, ctrl_keep, 0.0)
    return logF(pd_, fd) - logF(pc, fc)


def main():
    out = {}
    # (1) invariance: identical seeds, transformed vs untransformed drug library
    r1, r2 = np.random.default_rng(1), np.random.default_rng(1)
    a = S(r1, 40, 0.08, 0.15, 0.0)
    b = S(r2, 40, 0.08, 0.15, 0.0, shift=3.0, scale=2.5)
    out["1_invariance_diff"] = round(abs(a - b), 10)
    for n_fam in (40, 80):
        rng = np.random.default_rng(7)
        null = np.array([S(rng, n_fam, 0.08, 0.15, 0.0) for _ in range(1000)])
        crit = np.quantile(null, 0.95)
        out[f"2_selection_only_n{n_fam}"] = {"mean": round(null.mean(), 3), "sd": round(null.std(), 3),
                                              "crit95": round(crit, 3)}
        for tau in (0.25, 0.5, 0.75):
            alt = np.array([S(rng, n_fam, 0.08, 0.15, tau) for _ in range(500)])
            out[f"3_instruction_tau{tau}_n{n_fam}"] = {"mean": round(alt.mean(), 3),
                                                      "power_vs_null95": round((alt > crit).mean(), 3)}
        # selection-only but much stronger restriction in drug (1%) vs control (50%)
        sel = np.array([S(rng, n_fam, 0.01, 0.5, 0.0) for _ in range(500)])
        out[f"2b_strong_restriction_n{n_fam}"] = {"mean": round(sel.mean(), 3),
                                                  "false_pos_vs_crit": round((sel > crit).mean(), 3)}
        # selection-only, drug lowers memory (state partly reset) - is that 'instruction'? report it
        mem = np.array([S(rng, n_fam, 0.08, 0.15, 0.0, memory=0.3) for _ in range(500)])
        out[f"2c_lower_memory_n{n_fam}"] = {"mean": round(mem.mean(), 3), "pos_rate": round((mem > crit).mean(), 3)}
    for k, v in out.items():
        print(k, v)


if __name__ == "__main__":
    main()
