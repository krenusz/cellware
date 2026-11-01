"""Clone-level statistics from the Weinreb 2020 LARRY in vitro data (E6).

Writes the fit-set statistics (well 1) to larry_fit.json. The held-out cross-well statistics are
computed only with --test, which counts as one look at the test set (see EXPERIMENT_LOG.md).
"""

import argparse
import gzip
import json
import os

import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "larry")
OUT = os.path.join(os.path.dirname(__file__), "results_larry")


def load():
    with gzip.open(os.path.join(DATA, "metadata.txt.gz"), "rt") as f:
        rows = [line.rstrip("\n").split("\t") for line in f][1:]
    with gzip.open(os.path.join(DATA, "clone_matrix.mtx.gz"), "rt") as f:
        m = np.loadtxt(f, comments="%", skiprows=3, dtype=int)
    clone = np.full(len(rows), -1)
    clone[m[:, 0] - 1] = m[:, 1] - 1
    day = np.array([float(r[2]) for r in rows])
    well = np.array([int(r[5]) for r in rows])
    kind = np.array([r[4] for r in rows])
    fate = np.full(len(rows), "U")
    fate[kind == "Neutrophil"] = "N"
    fate[kind == "Monocyte"] = "M"
    fate[(kind != "Undifferentiated") & (kind != "Neutrophil") & (kind != "Monocyte")] = "O"
    return clone, day, well, fate


def counts(clone, day, well, fate, w):
    """clone id -> (N, M, O) mature-cell counts in well w, days 4 and 6."""
    keep = (clone >= 0) & (day > 2) & (well == w) & (fate != "U")
    out = {}
    for c, f in zip(clone[keep], fate[keep], strict=True):
        out.setdefault(c, [0, 0, 0])["NMO".index(f)] += 1
    return out


def fit_stats(cnt):
    """Composition statistics over clones with >= 2 mature cells (the fit target)."""
    arr = np.array([v for v in cnt.values() if sum(v) >= 2], dtype=float)
    n, m, o = arr.T
    nm = n + m
    frac = n[nm > 0] / nm[nm > 0]
    return {
        "clones": int(len(arr)),
        "n_only": float(np.mean((n > 0) & (m == 0) & (o == 0))),
        "m_only": float(np.mean((m > 0) & (n == 0) & (o == 0))),
        "mixed_nm": float(np.mean((n > 0) & (m > 0))),
        "has_o": float(np.mean(o > 0)),
        "o_share": float(o.sum() / arr.sum()),
        "nfrac_hist": np.histogram(frac, bins=5, range=(0, 1))[0].tolist(),
        "mean_size": float(arr.sum(1).mean()),
    }


def test_stats(c1, c2, min_nm=2, boot=2000, seed=0):
    pairs = []
    for c in set(c1) & set(c2):
        a, b = c1[c], c2[c]
        if a[0] + a[1] >= min_nm and b[0] + b[1] >= min_nm:
            pairs.append((a[0] / (a[0] + a[1]), b[0] / (b[0] + b[1])))
    p = np.array(pairs)
    rng = np.random.default_rng(seed)

    def stat(q):
        return np.corrcoef(q[:, 0], q[:, 1])[0, 1], np.mean((q[:, 0] > 0.5) == (q[:, 1] > 0.5))

    r, agree = stat(p)
    bs = np.array([stat(p[rng.integers(0, len(p), len(p))]) for _ in range(boot)])
    return {
        "clones": int(len(p)), "r": float(r), "r_ci": np.percentile(bs[:, 0], [2.5, 97.5]).tolist(),
        "agree": float(agree), "agree_ci": np.percentile(bs[:, 1], [2.5, 97.5]).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="compute held-out stats (one look)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    data = load()
    c1 = counts(*data, w=1)
    fit = fit_stats(c1)
    json.dump(fit, open(os.path.join(OUT, "larry_fit.json"), "w"), indent=1)
    print("fit set:", fit)
    if args.test:
        test = test_stats(c1, counts(*data, w=2))
        json.dump(test, open(os.path.join(OUT, "larry_test.json"), "w"), indent=1)
        print("TEST:", test)


if __name__ == "__main__":
    main()
