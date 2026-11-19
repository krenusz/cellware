"""C17-S1: joint clone model, train only. Bence 2026-11-19: "go 1 2" (smoke tests, then train checks).

Serves: C17-S1 (EXPERIMENT_LOG C17-S1/S2/S3 + amendments 1-3; lit/novelty/C17-S1.md).

--stage smoke   synthetic data only, no real data read:
    T-split': size-adjusted, size-binned clone-swap ICC on scenarios clone-driven / size-only / host-only,
              plus the planted expansion-agreement control. Reports rejection rates over replicates.
    T-cortical: instruction (clone-count fate fraction, guide - NT) and selection (log mean size by fate)
              on pure-selection, pure-instruction and null simulations, with and without a detection floor;
              the size-matched variant from the pre-registration is reported alongside.
--stage checks  train rows only (partitions.train_only), counts only (H9), no outcome:
    rf2020_clone: library compartments per sample set; train clones shared across two recipients with
              >= 3 Kit+ cells in each (gate >= 30).
    cortical_clone: train clones and donor batches per guide target (gate >= 3 batches per arm).

    python -I experiments/c17s1_joint.py --stage smoke|checks
"""

from __future__ import annotations

import argparse
import collections
import gzip
import io
import json
import os
import sys
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "results_c17")
ALPHA = 0.025


# ---------------------------------------------------------------- statistics (shared with S1/S3)
def icc_pairs(x1, x2):
    """One-way random-effects ICC(1) for two measurements per unit."""
    x = np.stack([x1, x2], 1)
    n = len(x)
    gm = x.mean()
    msb = 2 * ((x.mean(1) - gm) ** 2).sum() / (n - 1)
    msw = ((x - x.mean(1, keepdims=True)) ** 2).sum() / n
    return (msb - msw) / (msb + msw)


def size_adjust(b, logsize):
    """Residual of bias on log size (per recipient), so size cannot carry agreement."""
    X = np.c_[np.ones_like(logsize), logsize]
    beta, *_ = np.linalg.lstsq(X, b, rcond=None)
    return b - X @ beta


def tsplit_stat(b1, b2, s1, s2, group, rng, n_null=1000, n_bins=5):
    """Size-adjusted ICC of lineage bias; null swaps recipient-2 values within group x size quintile."""
    r1, r2 = size_adjust(b1, s1), size_adjust(b2, s2)
    obs = icc_pairs(r1, r2)
    tot = s1 + s2
    cells = collections.defaultdict(list)
    for g in np.unique(group):
        idx = np.where(group == g)[0]
        q = np.quantile(tot[idx], np.linspace(0, 1, n_bins + 1)[1:-1]) if len(idx) >= n_bins else []
        bins = np.searchsorted(q, tot[idx])
        for i, bn in zip(idx, bins, strict=True):
            cells[(g, bn)].append(i)
    null = np.empty(n_null)
    for k in range(n_null):
        perm = np.arange(len(b1))
        for members in cells.values():
            m = np.array(members)
            perm[m] = rng.permutation(m)
        null[k] = icc_pairs(r1, r2[perm])
    p = (1 + (null >= obs).sum()) / (1 + n_null)
    return obs, p


def cortical_stats(fate, size, guide):
    """fate in {0,1} per clone, size = cells, guide = bool (True = targeting). Returns
    instruction (clone-count fate fraction diff), selection (diff in log-mean-size gap between fates),
    size-matched instruction (pre-registered variant: mean within-size-bin fate diff)."""
    instr = fate[guide].mean() - fate[~guide].mean()

    def gap(m):
        return np.log(size[m & (fate == 1)].mean()) - np.log(size[m & (fate == 0)].mean())

    sel = gap(guide) - gap(~guide)
    q = np.quantile(size, [0.2, 0.4, 0.6, 0.8])
    b = np.searchsorted(q, size)
    diffs = [fate[(b == k) & guide].mean() - fate[(b == k) & ~guide].mean()
             for k in range(5) if ((b == k) & guide).sum() >= 5 and ((b == k) & ~guide).sum() >= 5]
    return instr, sel, float(np.mean(diffs)) if diffs else float("nan")


def truncnb_fit(x, floor):
    """MLE of a negative binomial (mean mu, size k) truncated below at floor; returns mu, k, P(detect)."""
    from scipy import optimize, stats  # noqa: PLC0415

    x = np.asarray(x)

    def nll(t):
        mu, k = np.exp(t)
        p = k / (k + mu)
        return -(stats.nbinom.logpmf(x, k, p).sum() - len(x) * stats.nbinom.logsf(floor - 1, k, p))

    t = optimize.minimize(nll, [np.log(max(x.mean(), 1.0)), 0.0], method="Nelder-Mead").x
    mu, k = np.exp(t)
    return mu, k, float(stats.nbinom.sf(floor - 1, k, k / (k + mu)))


def cortical_corrected(fate, size, guide, floor):
    """Amendment 4: detection-aware decomposition. Per arm and fate class, fit a truncated NB to clone
    sizes; undetected clones are added back as n / P(detect). Instruction = corrected fate fraction
    (guide - NT); selection = difference in log fitted mean size between fates (guide - NT)."""
    frac, lmu = {}, {}
    for arm in (False, True):
        m = guide == arm
        nh, mus = {}, {}
        for f in np.unique(fate):
            xs = size[m & (fate == f)]
            mu, _k, d = truncnb_fit(xs, floor)
            nh[f], mus[f] = len(xs) / max(d, 1e-6), mu
        tot = sum(nh.values())
        frac[arm] = {f: v / tot for f, v in nh.items()}
        lmu[arm] = mus
    instr = frac[True][1] - frac[False][1]
    sel = (np.log(lmu[True][1]) - np.log(lmu[True][0])) - (np.log(lmu[False][1]) - np.log(lmu[False][0]))
    return instr, sel


# ---------------------------------------------------------------- smoke
def sim_split(rng, scenario, n=60, groups=4):
    g = rng.integers(0, groups, n)
    latent = rng.normal(2.0, 1.0, n)  # clone potency (log size)
    host = rng.normal(0, 0.3, (n, 2))
    s = latent[:, None] + rng.normal(0, 0.5, (n, 2))  # expansion agrees (published control)
    cells = np.maximum(3, rng.poisson(np.exp(s)))
    intrinsic = rng.normal(0, 1.0, n)
    hshift = np.array([0.0, 0.8])  # recipient-level host effect
    if scenario == "clone":
        eta = 0.8 * intrinsic[:, None] + hshift + host
    elif scenario == "size":
        eta = -0.9 * (np.log(cells) - 2.0) + hshift + host
    else:  # host
        eta = hshift + host + rng.normal(0, 0.8, (n, 2))
    p = 1 / (1 + np.exp(-eta))
    k = rng.binomial(cells, p)
    bias = np.log((k + 0.5) / (cells - k + 0.5))  # CLR-type log ratio with 0.5 pseudocount
    return bias[:, 0], bias[:, 1], np.log(cells[:, 0]), np.log(cells[:, 1]), g


def sim_cortical(rng, scenario, floor, n_nt=560, n_g=200):
    def arm(nc, p_fate, growth_fate1):
        fate = rng.binomial(1, p_fate, nc)
        mu = np.where(fate == 1, growth_fate1, 1.0) * 5.0
        size = rng.negative_binomial(2, 2 / (2 + mu))
        keep = size >= floor
        return fate[keep], size[keep]

    fn, sn = arm(n_nt, 0.4, 1.0)
    pg = 0.55 if scenario == "instruction" else 0.4
    gr = 2.0 if scenario == "selection" else 1.0
    fg, sg = arm(n_g, pg, gr)
    fate, size = np.r_[fn, fg], np.r_[sn, sg]
    guide = np.r_[np.zeros(len(fn), bool), np.ones(len(fg), bool)]
    return (*cortical_stats(fate, size, guide), *cortical_corrected(fate, size, guide, floor))


def smoke(reps=200, cortical_only=False):
    rng = np.random.default_rng(17)
    res = {"tsplit": {}, "expansion_control": {}, "cortical": {}}
    for sc in ("clone", "size", "host"):
        rej, iccs = 0, []
        for _ in range(reps):
            b1, b2, s1, s2, g = sim_split(rng, sc)
            icc, p = tsplit_stat(b1, b2, s1, s2, g, rng, n_null=300)
            rej += p < ALPHA
            iccs.append(icc)
        res["tsplit"][sc] = {"reject_rate": rej / reps, "icc_mean": float(np.mean(iccs))}
    ctrl = [icc_pairs(*sim_split(rng, "host")[2:4]) for _ in range(reps)]
    res["expansion_control"] = {"icc_mean": float(np.mean(ctrl)), "min": float(np.min(ctrl))}
    for floor in (1, 3, 5):
        for sc in ("null", "selection", "instruction"):
            st = np.array([sim_cortical(rng, sc, floor) for _ in range(reps)])
            res["cortical"][f"{sc}_floor{floor}"] = {
                "instruction_mean": float(st[:, 0].mean()), "instruction_sd": float(st[:, 0].std()),
                "selection_mean": float(st[:, 1].mean()),
                "size_matched_instruction_mean": float(np.nanmean(st[:, 2])),
                "corrected_instruction_mean": float(st[:, 3].mean()),
                "corrected_instruction_sd": float(st[:, 3].std()),
                "corrected_selection_mean": float(st[:, 4].mean())}
    t, c = res["tsplit"], res["cortical"]
    res["pass"] = {
        "tsplit_clone_power": t["clone"]["reject_rate"] >= 0.8,
        "tsplit_size_fpr": t["size"]["reject_rate"] <= 0.06,
        "tsplit_host_fpr": t["host"]["reject_rate"] <= 0.06,
        "expansion_control_recovered": res["expansion_control"]["icc_mean"] > 0.3,
        "cortical_corrected_selection_not_instruction": all(
            abs(c[f"selection_floor{f}"]["corrected_instruction_mean"])
            < 0.5 * c[f"selection_floor{f}"]["corrected_instruction_sd"] for f in (1, 3, 5)),
        "cortical_corrected_instruction_detected": all(
            c[f"instruction_floor{f}"]["corrected_instruction_mean"] > 0.1 for f in (1, 3, 5)),
        "cortical_corrected_selection_detected": all(
            c[f"selection_floor{f}"]["corrected_selection_mean"] > 0.4 for f in (1, 3, 5)),
    }
    return res


# ---------------------------------------------------------------- train checks (counts only)
GEO = "https://ftp.ncbi.nlm.nih.gov/geo/series"
RF = f"{GEO}/GSE134nnn/GSE134242/suppl/GSE134242_ARF2019_"
CORT = f"{GEO}/GSE284nnn/GSE284197/suppl/GSE284197_clones.h5ad"


def fetch(url):
    for i in range(6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "cellware-c17/1.0"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            print(f"retry {i + 1}: {e}", flush=True)
            time.sleep(5 * (i + 1))
    raise RuntimeError(url)


def lines(name):
    return gzip.decompress(fetch(RF + name + ".txt.gz")).decode().splitlines()


def recipient(sid):
    t = sid.split("_")
    return t[1] if t[0] in ("HSC", "Kit", "KIT") and len(t) > 1 else t[0]


def compartment(sid):
    s = sid.lower()
    return "kit" if "kit" in s else ("hsc" if "hsc" in s else "other")


def checks():
    import pandas as pd  # noqa: PLC0415
    from partitions import train_only  # noqa: PLC0415

    rows = []
    sets = {"Sample1": "samp_id", "Sample2": "samp_id_flat", "Sample3": "samp_id_flat",
            "Sample4": "samp_id_flat", "Sample5": "samp_id_flat"}
    libs = {}
    for s, id_f in sets.items():
        bcs, ids = lines(f"{s}_barcode_list"), lines(f"{s}_{id_f}")
        if len(ids) == len(bcs) + 1:
            ids = ids[1:]
        libs[s] = dict(collections.Counter(compartment(i) for i in ids))
        for b, i in zip(bcs, ids, strict=True):
            b = b.strip().strip("_")
            if b:
                rows.append((f"{s}:{b}", s, recipient(i), compartment(i)))
    df = train_only("rf2020_clone", pd.DataFrame(rows, columns=["unit", "set", "rec", "comp"]), "unit")
    kit = df[df.comp == "kit"].groupby(["unit", "rec"]).size().unstack(fill_value=0)
    shared = kit[(kit >= 3).sum(1) >= 2]
    per_set = collections.Counter(u.split(":")[0] for u in shared.index)
    rf = {"cells_by_compartment_all_partitions": libs, "train_clones": int(df.unit.nunique()),
          "train_shared_kit3_by_set": dict(per_set), "train_shared_kit3_total": int(len(shared)),
          "gate_ge_30": bool(len(shared) >= 30)}

    import h5py  # noqa: PLC0415

    with h5py.File(io.BytesIO(fetch(CORT)), "r") as h:
        obs = h["obs"]
        idx = obs.attrs.get("_index", "_index")
        index = [x.decode() if isinstance(x, bytes) else str(x) for x in obs[idx][:]]

        def cat(col):
            g = obs[col]
            cats = [c.decode() if isinstance(c, bytes) else str(c) for c in g["categories"][:]]
            return [cats[k] if k >= 0 else "NA" for k in g["codes"][:]]

        h5_ncells = obs["n_cells"][:]
        cd = pd.DataFrame({"unit": index, "target": cat("Gene_target"), "batch": cat("batch")})
    cd = train_only("cortical_clone", cd, "unit")
    nc = np.asarray(h5_ncells)[np.isin(index, cd.unit.values)]
    floor_counts = {"train_min_n_cells": int(nc.min()), "train_clones_by_n_cells_1_to_5":
                    {int(v): int((nc == v).sum()) for v in range(1, 6)}}
    arms = {t: {"train_clones": int(len(g)), "donor_batches": int(g.batch.nunique()),
                "gate_ge_3_batches": bool(g.batch.nunique() >= 3)} for t, g in cd.groupby("target")}
    return {"rf2020_clone": rf, "cortical_clone": arms, "cortical_detection_floor": floor_counts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["smoke", "checks"], required=True)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    res = smoke() if a.stage == "smoke" else checks()
    path = os.path.join(OUT, f"c17s1_{a.stage}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(res.get("pass", res), indent=1)[:3000])


if __name__ == "__main__":
    main()
