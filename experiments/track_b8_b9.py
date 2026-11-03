"""B8 (same-cocktail negative control for B7) and B9 (day-2 state of the selected clones).
See the 2026-11-03 (4) batch pre-registration in EXPERIMENT_LOG.md.

    uv run python experiments/track_b8_b9.py
"""

from __future__ import annotations

import gzip
import json
import os

import numpy as np
from e25_clone_types import groups
from fit_cytokine import CONDS5, load_counts
from fit_cytokine import DATA as CYTO
from kin_corr_e19 import load as load_invitro
from scipy.stats import kruskal, mannwhitneyu

HERE = os.path.dirname(__file__)


def b8(rng):
    g = load_invitro()  # {clone: {(day, well): [N, M]}} for clones in >= 2 wells
    # include single-well clones too: reload without the >= 2 wells filter
    import kin_corr_e19 as k

    rows = [
        line.rstrip("\n").split("\t")
        for line in gzip.open(os.path.join(k.DATA, "metadata.txt.gz"), "rt")
    ][1:]
    m = np.loadtxt(
        gzip.open(os.path.join(k.DATA, "clone_matrix.mtx.gz"), "rt"),
        comments="%",
        skiprows=3,
        dtype=int,
    )
    per = {}
    for cell, clone in zip(m[:, 0] - 1, m[:, 1] - 1, strict=True):
        day, kind, well = rows[cell][2], rows[cell][4], rows[cell][5]
        if day == "2.0" or kind not in ("Neutrophil", "Monocyte"):
            continue
        per.setdefault(int(clone), {}).setdefault(well, [0, 0])[
            0 if kind == "Neutrophil" else 1
        ] += 1
    out = {"clones_in_2plus_wells_E19": len(g)}
    for w, o in (("1", "2"), ("2", "1")):
        ok = {c: v for c, v in per.items() if sum(v.get(w, [0, 0])) >= 2}
        both = [c for c, v in ok.items() if sum(v.get(o, [0, 0])) >= 2]
        only = [c for c, v in ok.items() if sum(v.get(o, [0, 0])) < 2]
        a = np.array([per[c][w][0] / sum(per[c][w]) for c in both])
        b = np.array([per[c][w][0] / sum(per[c][w]) for c in only])
        bs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(2000)]
        out[f"well{w}"] = {
            "n_both": len(a),
            "n_only": len(b),
            "mean_both": float(a.mean()),
            "mean_only": float(b.mean()),
            "diff": float(a.mean() - b.mean()),
            "ci": np.percentile(bs, [2.5, 97.5]).tolist(),
        }
    return out


def b9():
    rows = [
        line.rstrip("\n").split("\t")
        for line in gzip.open(os.path.join(CYTO, "metadata.txt.gz"), "rt")
    ][1:]
    m = np.loadtxt(
        gzip.open(os.path.join(CYTO, "clone_matrix.mtx.gz"), "rt"),
        comments="%",
        skiprows=3,
        dtype=int,
    )
    d2 = [i for i, r in enumerate(rows) if r[2] == "2"]
    xy = np.array([[float(rows[i][5]), float(rows[i][6])] for i in d2])
    ct = np.array([rows[i][3] for i in d2])
    cn, cm = xy[ct == "Neu"].mean(0), xy[ct == "Mo"].mean(0)
    pos = {i: j for j, i in enumerate(d2)}
    cells = {}
    for clone, cell in zip(m[:, 0] - 1, m[:, 1] - 1, strict=True):
        if cell in pos:
            cells.setdefault(clone, []).append(pos[cell])
    _, _, both, g_only, m_only = groups(load_counts(CONDS5))
    feats = {}
    for name, grp in (("both", both), ("G_only", g_only), ("M_only", m_only)):
        f = []
        for c in grp:
            if c in cells:
                j = np.array(cells[c])
                score = np.mean(
                    np.linalg.norm(xy[j] - cm, axis=1) - np.linalg.norm(xy[j] - cn, axis=1)
                )
                f.append(
                    (
                        np.mean(ct[j] == "Undifferentiated"),
                        np.mean(ct[j] == "Neu"),
                        np.mean(ct[j] == "Mo"),
                        score,
                    )
                )
        feats[name] = np.array(f)
    out = {k: {"clones_with_day2": len(v)} for k, v in feats.items()}
    for i, name in enumerate(("undiff_share", "neu_share", "mo_share", "neu_priming_score")):
        for k, v in feats.items():
            out[k][name] = (
                float(np.median(v[:, i])) if name.endswith("score") else float(v[:, i].mean())
            )
        out[f"kruskal_{name}"] = float(kruskal(*[v[:, i] for v in feats.values()]).pvalue)
        out[f"G_only_vs_both_{name}_p"] = float(
            mannwhitneyu(feats["G_only"][:, i], feats["both"][:, i]).pvalue
        )
        out[f"M_only_vs_both_{name}_p"] = float(
            mannwhitneyu(feats["M_only"][:, i], feats["both"][:, i]).pvalue
        )
    return out


def main():
    out = {"b8": b8(np.random.default_rng(0)), "b9": b9()}
    json.dump(out, open(os.path.join(HERE, "results_e18", "b8_b9.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
