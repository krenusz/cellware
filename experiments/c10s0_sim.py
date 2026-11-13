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


# ---------- amendment 3: real structure ----------
PROBS = {1: 0.40, 2: 0.35, 3: 0.12, 4: 0.07, 5: 0.06}


def lib_real(rng, n_fam, keep_frac, tau, memory=0.7, drift_sd=0.3, sub_frac_A=None, sub_drift=0.0, n_fixed=None):
    n_all = int(n_fam / keep_frac)
    x = rng.normal(size=(n_all, D))
    sub = None
    if sub_frac_A is not None:                      # genetic subclones: offset in pre AND subclone drift in post
        sub_all = (rng.random(n_all) < 0.5).astype(int)
        off = np.zeros(D); off[1] = 1.0
        x = x + np.outer(sub_all, off)
        # selection mixes survival on x[0] and subclone: keep top keep_frac with subclone A share = sub_frac_A
        order = np.argsort(-x[:, 0])
        A = [i for i in order if sub_all[i] == 1][: int(round(n_fam * sub_frac_A))]
        B = [i for i in order if sub_all[i] == 0][: n_fam - len(A)]
        keep = np.array(A + B); sub = sub_all[keep]
    else:
        keep = np.argsort(-x[:, 0])[:n_fam]
    x = x[keep]
    pre = x + rng.normal(scale=SIG_CELL, size=x.shape)
    mu = memory * x + rng.normal(scale=drift_sd, size=x.shape)
    if tau:
        mu = mu + rng.normal(scale=tau, size=x.shape)
    if sub is not None:
        d = np.zeros(D); d[2] = sub_drift
        mu = mu + np.outer(sub, d)
    ks = list(PROBS); ps = np.array(list(PROBS.values())); ps = ps / ps.sum()
    fams = []
    for i in range(n_fam):
        k = n_fixed or rng.choice(ks, p=ps)
        fams.append(mu[i] + rng.normal(scale=SIG_CELL, size=(k, D)))
    return pre, fams


def logF_all(pre, fams):
    means = np.array([f.mean(0) for f in fams]); n = np.array([len(f) for f in fams])
    X = np.column_stack([np.ones(len(pre)), pre])
    beta, *_ = np.linalg.lstsq(X, means, rcond=None)
    resid = means - X @ beta
    ms_b = (resid ** 2).sum() / ((len(means) - X.shape[1]) * D)
    multi = [f for f in fams if len(f) >= 2]
    sw2 = sum(((f - f.mean(0)) ** 2).sum() for f in multi) / (sum(len(f) - 1 for f in multi) * D)
    return np.log(ms_b / (sw2 * np.mean(1.0 / n)))


def S_real(rng, nd, nc, est, tau=0.0, drug_kw=None, ctrl_kw=None):
    drug_kw = drug_kw or {}; ctrl_kw = ctrl_kw or {}
    if est == "F2":
        # n = families with >= 2 post cells; draw sizes from PROBS conditional on >= 2
        pdd, fd = lib_real(rng, nd, 0.08, tau, **drug_kw); pc, fc = lib_real(rng, nc, 0.15, 0.0, **ctrl_kw)
        keep_d = [i for i, f in enumerate(fd) if len(f) >= 2]; keep_c = [i for i, f in enumerate(fc) if len(f) >= 2]
        return logF(pdd[keep_d], [fd[i] for i in keep_d]) - logF(pc[keep_c], [fc[i] for i in keep_c])
    pdd, fd = lib_real(rng, nd, 0.08, tau, **drug_kw); pc, fc = lib_real(rng, nc, 0.15, 0.0, **ctrl_kw)
    return logF_all(pdd, fd) - logF_all(pc, fc)


def real():
    # sealed-third family counts: F2 uses >=2-post families (drawn as total linked = n2 / 0.6)
    pairs = {"Olaparib_1 vs Control_1": (58, 37), "NK_2 vs Control_1": (53, 37), "NK_1 vs Control_1": (41, 37),
             "Olaparib_1 vs Control_2": (58, 18)}
    out = {}
    R = 400
    for name, (n2d, n2c) in pairs.items():
        nd, nc = int(n2d / 0.6), int(n2c / 0.6)
        for est in ("F2", "Fall"):
            rng = np.random.default_rng(11)
            null = np.array([S_real(rng, nd, nc, est) for _ in range(R)])
            crit = np.quantile(null, 0.95)
            row = {"null_mean": round(null.mean(), 3), "crit95": round(crit, 3)}
            for tau in (0.5, 0.75, 1.0):
                alt = np.array([S_real(rng, nd, nc, est, tau=tau) for _ in range(R // 2)])
                row[f"power_tau{tau}"] = round((alt > crit).mean(), 3)
            # null worlds
            subA = np.array([S_real(rng, nd, nc, est, drug_kw={"sub_frac_A": 0.9, "sub_drift": 1.0},
                                    ctrl_kw={"sub_frac_A": 0.5, "sub_drift": 1.0}) for _ in range(R // 2)])
            row["nullA_subclone_fp"] = round((subA > crit).mean(), 3)
            longer = np.array([S_real(rng, nd, nc, est, drug_kw={"drift_sd": 0.3 * np.sqrt(13 / 8)})
                               for _ in range(R // 2)])
            row["nullB_longer_drug_drift_fp"] = round((longer > crit).mean(), 3)
            out[f"{name} [{est}]"] = row
    for k, v in out.items():
        print(k, v)


if __name__ == "__main__":
    import sys
    real() if sys.argv[1:] == ["real"] else main()
