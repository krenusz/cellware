"""QS2-N2-A: division-axis memory from mother to daughter division, train scan.

Pre-registered 2026-11-27 (EXPERIMENT_LOG.md, "## QS2-N2-A"). Serves: C12 CLOSED. Three stages, each gated on the one
before:

    python experiments/n2_axis.py selftest   # synthetic structure and a toy XML only, no real file
    python experiments/n2_axis.py all        # stage0 -> smoke -> train; writes experiments/results_qs2/n2_axis.json

Stage 0 (structure only): R, its ratio to the tissue extent, the time window (smallest of W_GRID with enough usable
trees), usable pairs and trees.
Smoke (H21 item 3): simulated axes on the real train structure (positions, times, pairs, control sets); no real axis.
Train (H21 item 1): real train axes; the d distribution, T and its two-sided control-swap p.

Train units only (partitions.train_only). Spot positions are read for train-unit spots only and are never saved or
printed; outputs are counts, scalars (R, extent, lag), simulated rejection rates and train d summaries.
"""

import io
import json
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ds_emb_seal  # noqa: E402
import partitions  # noqa: E402
import qs_e5  # noqa: E402

OUT = os.path.join(HERE, "results_qs2", "n2_axis.json")
SEED = 20261127
W_GRID = (10.0, 20.0, 30.0, 60.0)  # control time window (min) around the mother division; stage 0 takes the
#                                    smallest with >= TREES_MIN usable trees (structure only)
BAND = 0.25  # control: | distance to the daughter division - the true mother's distance | <= BAND x R
K_MIN = 2  # a pair needs >= 2 controls (swap pool >= 3)
R_EXTENT_MAX = 0.25  # stage 0: R <= 0.25 x tissue extent, else the control is not local
TREES_MIN = 40  # stage 0: usable train trees
TAU = 0.5  # smoke: field noise (strong field)
L_FACTORS = (0.5, 1.0, 2.0, 4.0, 8.0)  # smoke: field length scale in units of R
DRIFT_L, DRIFT_LAGS = 2.0, 4.0  # smoke: extra null with the field also rotating in time (period 4 x median lag)
Q_PASS = 0.5
Q_GRID = (0.2, 0.3, 0.5, 0.7)
NULL_MAX, ALT_MIN = 0.08, 0.80
N_NULL, N_ALT, N_PERM, N_PERM_TRAIN = 400, 200, 500, 5000
ALPHA = 0.05


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def p2(a, b):
    """Second Legendre polynomial of the angle between axes: 1 aligned, -0.5 orthogonal, about 0 if random (3D)."""
    c = np.sum(a * b, axis=-1)
    return 1.5 * c * c - 0.5


# ---------- structure ----------

def load_positions(src, keep_spots):
    pos = {}
    for _, el in ET.iterparse(src, events=("end",)):
        if el.tag == "Spot":
            i = el.get("ID")
            if i in keep_spots:
                pos[i] = (float(el.get("POSITION_X")), float(el.get("POSITION_Y")), float(el.get("POSITION_Z")))
            el.clear()
    return pos


def div_table(frame, edges, unit_of, keep, pos):
    """Divisions (a node with exactly 2 kids at one frame) of the units in `keep`: tree, t (kids' frame), midpoint,
    axis (unit vector between the kids). Pairs: one per mother division, (mother, next division along the chain of
    its kid with the smaller spot ID that reaches a division); axis-blind, and no two pairs share a mother."""
    kids = defaultdict(list)
    for a, b in edges:
        if a not in frame or b not in frame or unit_of.get(a) not in keep:
            continue
        s, t = (a, b) if frame[a] < frame[b] else (b, a)
        kids[s].append(t)
    skipped, rec = Counter(), {}
    for v, ks in list(kids.items()):
        if len(ks) != 2:
            if len(ks) > 2:
                skipped["more_than_2_kids"] += 1
            continue
        k1, k2 = sorted(ks, key=int)
        if frame[k1] != frame[k2]:
            skipped["kids_at_different_frames"] += 1
            continue
        if k1 not in pos or k2 not in pos:
            skipped["no_position"] += 1
            continue
        a1, a2 = np.array(pos[k1]), np.array(pos[k2])
        n = np.linalg.norm(a2 - a1)
        if n == 0:
            skipped["zero_axis"] += 1
            continue
        rec[v] = (unit_of[v], frame[k1], (a1 + a2) / 2, (a2 - a1) / n)

    def next_div(k):
        x = k
        while True:
            ks = kids.get(x, [])
            if len(ks) == 2:
                return x
            if len(ks) != 1:
                return None
            x = ks[0]

    ids = sorted(rec, key=lambda v: (rec[v][1], int(v)))
    ix = {v: i for i, v in enumerate(ids)}
    pairs = []
    for v in ids:
        nd = [w for k in sorted(kids[v], key=int) if (w := next_div(k)) is not None and w in ix]
        if nd:
            pairs.append((ix[v], ix[nd[0]]))
    return {"tree": np.array([rec[v][0] for v in ids], dtype=object),
            "t": np.array([rec[v][1] for v in ids], dtype=float),
            "pos": np.array([rec[v][2] for v in ids], dtype=float).reshape(-1, 3),
            "axis": np.array([rec[v][3] for v in ids], dtype=float).reshape(-1, 3),
            "pairs": np.array(pairs, dtype=int).reshape(-1, 2)}, skipped


def pool_movies(tables):
    """tables: [(movie, frame interval in min, table)] -> one structure dict over movies."""
    off, parts = 0, {"movie": [], "dt": [], "tree": [], "t": [], "pos": [], "axis": [], "pairs": []}
    for m, dt, tb in tables:
        n = len(tb["t"])
        parts["movie"].append(np.full(n, m))
        parts["dt"].append(np.full(n, dt, dtype=float))
        for k in ("tree", "t", "pos", "axis"):
            parts[k].append(tb[k])
        parts["pairs"].append(tb["pairs"] + off)
        off += n
    return {k: np.concatenate(v) if k != "pairs" else np.concatenate(v).reshape(-1, 2) for k, v in parts.items()}


def add_centres(S):
    """Per pair: centre = daughter position minus the movie's mean mother-to-daughter displacement (structure only),
    and the residual distance of the true mother from that centre. Returns {movie: |mean displacement|}."""
    pr = S["pairs"]
    disp = S["pos"][pr[:, 1]] - S["pos"][pr[:, 0]]
    pm = S["movie"][pr[:, 0]]
    vbar = {m: disp[pm == m].mean(0) for m in np.unique(pm)}
    S["centre"] = S["pos"][pr[:, 1]] - np.array([vbar[m] for m in pm]).reshape(-1, 3)
    S["resid"] = np.linalg.norm(S["pos"][pr[:, 0]] - S["centre"], axis=1)
    return {str(m): float(np.linalg.norm(v)) for m, v in vbar.items()}


def controls(S, w_min, band):
    """Per pair (M, D): divisions C of other trees in the same movie with |t_C - t_M| <= w_min minutes and
    | |C - centre| - |M - centre| | <= band, centre = D minus the movie's mean displacement (add_centres). Controls
    sit on the true mother's shell around where a mother is expected, so a shared drift does not favour M."""
    out = []
    for p, (i, _) in enumerate(S["pairs"].tolist()):
        dist = np.linalg.norm(S["pos"] - S["centre"][p], axis=1)
        ok = ((S["movie"] == S["movie"][i]) & (S["tree"] != S["tree"][i])
              & (np.abs(S["t"] - S["t"][i]) * S["dt"][i] <= w_min + 1e-9) & (np.abs(dist - S["resid"][p]) <= band))
        out.append(np.flatnonzero(ok))
    return out


def extent(S):
    """Median over movies of the 5-95 percentile range of division midpoints along their first principal axis."""
    ext = []
    for m in np.unique(S["movie"]):
        P = S["pos"][S["movie"] == m]
        P = P - P.mean(0)
        v = np.linalg.svd(P, full_matrices=False)[2][0]
        x = P @ v
        ext.append(float(np.percentile(x, 95) - np.percentile(x, 5)))
    return float(np.median(ext))


# ---------- statistic and control-swap null ----------

class Design:
    """Usable pairs (>= K_MIN controls), optionally restricted to a set of trees. One pair per mother, so each pair
    has its own pool {true mother + controls} and its own swap draw."""

    def __init__(self, S, ctrl, trees=None):
        keep = [p for p, (i, _) in enumerate(S["pairs"].tolist())
                if len(ctrl[p]) >= K_MIN and (trees is None or S["tree"][i] in trees)]
        self.n_pairs = len(keep)
        self.mothers, self.ntree = [], 0
        if not keep:
            return
        self.mothers = [int(S["pairs"][p, 0]) for p in keep]
        maxm = 1 + max(len(ctrl[p]) for p in keep)
        self.pool = np.full((len(keep), maxm), -1)
        for k, p in enumerate(keep):
            self.pool[k, :1 + len(ctrl[p])] = [S["pairs"][p, 0], *ctrl[p].tolist()]
        self.msize = (self.pool >= 0).sum(1)
        self.pd = np.array([S["pairs"][p, 1] for p in keep])
        tl = sorted({S["tree"][i] for i in self.mothers})
        ti = {t: k for k, t in enumerate(tl)}
        self.ntree = len(tl)
        W = np.zeros((len(keep), self.ntree))
        W[np.arange(len(keep)), [ti[S["tree"][i]] for i in self.mothers]] = 1.0
        self.W = W / W.sum(0)

    def s(self, A):
        """P2 of each pair's daughter axis with every member of its pool (column 0 = true mother)."""
        v = p2(A[self.pd][:, None, :], A[np.maximum(self.pool, 0)])
        return np.where(self.pool >= 0, v, np.nan)

    def d_of(self, s, jj):
        """Pair d when pool member jj is the 'mother': P2(daughter, jj) - mean P2(daughter, rest of the pool).
        jj: (n_pairs,) or (B, n_pairs)."""
        tot = np.nansum(s, 1)
        sj = s[np.arange(s.shape[0]), jj]
        return sj - (tot - sj) / (self.msize - 1)

    def test(self, A, rng, B):
        """T = mean over trees of the tree-mean d; two-sided p from B swaps, one draw per pair, uniform over its pool
        {true mother + controls}."""
        s = self.s(A)
        d = self.d_of(s, np.zeros(self.n_pairs, int))
        tm = d @ self.W
        T = float(tm.mean())
        J = (rng.random((B, self.n_pairs)) * self.msize).astype(int)
        Ts = (self.d_of(s, J) @ self.W).mean(1)
        p = (1 + int(np.sum(np.abs(Ts) >= abs(T) - 1e-12))) / (B + 1)
        return T, p, d, tm, s


# ---------- smoke: simulated axes on the real structure ----------

def sim_axes(S, rng, L, model, q, period_min=None):
    """Field: per movie, axis = norm(u + TAU*g), u rotating in a random plane with period L along a random direction
    (and, if period_min is set, also in time with that period). Then, in daughter-time order, with probability q a
    daughter division's axis is replaced: memory = norm(a_M + 0.3 g); orthogonal = random, perpendicular to a_M."""
    A = np.empty_like(S["pos"])
    for m in np.unique(S["movie"]):
        ix = S["movie"] == m
        e = unit(rng.normal(size=3))
        e1 = unit(rng.normal(size=3))
        e2 = unit(np.cross(e1, unit(rng.normal(size=3))))
        ph = 2 * np.pi * (S["pos"][ix] @ e) / L + rng.uniform(0, 2 * np.pi)
        if period_min:
            ph = ph + 2 * np.pi * S["t"][ix] * S["dt"][ix] / period_min
        u = np.cos(ph)[:, None] * e1 + np.sin(ph)[:, None] * e2
        A[ix] = unit(u + TAU * rng.normal(size=(int(ix.sum()), 3)))
    if model != "null" and q > 0:
        order = np.argsort(S["t"][S["pairs"][:, 1]], kind="stable")
        for i, j in S["pairs"][order].tolist():
            if rng.random() < q:
                g = rng.normal(size=3)
                if model == "memory":
                    A[j] = unit(A[i] + 0.3 * g)
                else:
                    A[j] = unit(g - (g @ A[i]) * A[i])
    return A


def reject_rate(S, ctrl, trees, n_trees, model, q, L, reps, rng, perms=N_PERM, period_min=None):
    hits, Tm = 0, []
    trees = sorted(trees)
    for _ in range(reps):
        sub = set(rng.choice(trees, size=n_trees, replace=False).tolist()) if n_trees < len(trees) else set(trees)
        D = Design(S, ctrl, sub)
        T, p, *_ = D.test(sim_axes(S, rng, L, model, q, period_min), rng, perms)
        hits += p < ALPHA
        Tm.append(T)
    return {"reject": hits / reps, "mean_T": float(np.mean(Tm)), "reps": reps}


# ---------- stages ----------

def load_train():
    import pandas as pd  # noqa: PLC0415

    dts = qs_e5.frame_intervals()
    tables, skipped, units = [], Counter(), {}
    for m in qs_e5.MOVIES:
        path = qs_e5.fetch_movie(m)
        frame, edges = ds_emb_seal.load(path)
        unit_of = qs_e5.components(frame, edges, m)
        df = partitions.train_only("mouse_gast_tree", pd.DataFrame({"unit": sorted(set(unit_of.values()))}), "unit")
        keep = set(df["unit"])
        pos = load_positions(path, {s for s, u in unit_of.items() if u in keep})
        tb, sk = div_table(frame, edges, unit_of, keep, pos)
        del pos
        tables.append((m, dts[m], tb))
        skipped.update(sk)
        units[f"M{m}"] = {"train_units": len(keep), "frame_interval_min": dts[m],
                          "divisions": int(len(tb["t"])), "pairs": int(len(tb["pairs"]))}
    return pool_movies(tables), units, dict(skipped)


def stage0(S):
    pr = S["pairs"]
    drift = add_centres(S)
    R = float(np.median(S["resid"]))
    lag = float(np.median((S["t"][pr[:, 1]] - S["t"][pr[:, 0]]) * S["dt"][pr[:, 0]]))
    ext = extent(S)
    by_w, chosen = {}, None
    for w in W_GRID:
        ctrl = controls(S, w, BAND * R)
        D = Design(S, ctrl)
        by_w[str(w)] = {"usable_pairs": D.n_pairs, "usable_trees": D.ntree}
        if D.ntree >= TREES_MIN:
            chosen = w
            break
    n_ctrl = np.array([len(c) for c in ctrl if len(c) >= K_MIN] or [0])
    res = {"R": R, "mean_displacement_over_R": {m: v / R for m, v in drift.items()},
           "median_raw_displacement_over_R": float(np.median(np.linalg.norm(S["pos"][pr[:, 1]] - S["pos"][pr[:, 0]],
                                                                            axis=1)) / R),
           "median_lag_min": lag, "extent": ext, "R_over_extent": R / ext, "pairs_all": int(len(pr)),
           "trees_with_pairs": int(len({S["tree"][i] for i in pr[:, 0]})), "by_window_min": by_w,
           "window_min": chosen, "usable_pairs": D.n_pairs, "usable_trees": D.ntree,
           "controls_per_usable_pair": {"median": float(np.median(n_ctrl)), "p10": float(np.percentile(n_ctrl, 10)),
                                        "p90": float(np.percentile(n_ctrl, 90))}}
    fails = []
    if R / ext > R_EXTENT_MAX:
        fails.append(f"R/extent {R / ext:.3f} > {R_EXTENT_MAX}")
    if chosen is None:
        fails.append(f"usable trees {D.ntree} < {TREES_MIN} at every window up to {W_GRID[-1]} min")
    res["pass"], res["fails"] = not fails, fails
    return res, ctrl, D


def smoke(S, ctrl, D0, R, lag, rng):
    trees = set(S["tree"][D0.mothers].tolist())
    n_look = int(round(0.5 * D0.ntree))
    res = {"n_look_trees": n_look, "n_train_trees": D0.ntree, "null": {}, "alt": {}, "grid": {}}
    for f in L_FACTORS:
        res["null"][f"L={f}R"] = reject_rate(S, ctrl, trees, n_look, "null", 0, f * R, N_NULL, rng)
    res["null"][f"L={DRIFT_L}R+drift"] = reject_rate(S, ctrl, trees, n_look, "null", 0, DRIFT_L * R, N_NULL, rng,
                                                    period_min=DRIFT_LAGS * lag)
    for model in ("memory", "orthogonal"):
        res["alt"][model] = reject_rate(S, ctrl, trees, n_look, model, Q_PASS, R, N_ALT, rng)
    fails = [f"null {k} rejects {v['reject']:.3f} > {NULL_MAX}" for k, v in res["null"].items()
             if v["reject"] > NULL_MAX]
    fails += [f"{k} at q={Q_PASS} rejects {v['reject']:.3f} < {ALT_MIN}" for k, v in res["alt"].items()
              if v["reject"] < ALT_MIN]
    res["pass"], res["fails"] = not fails, fails
    if res["pass"]:
        for model in ("memory", "orthogonal"):
            for n in (n_look, D0.ntree):
                res["grid"][f"{model}|n={n}"] = {str(q): reject_rate(S, ctrl, trees, n, model, q, R, N_ALT, rng)
                                                for q in Q_GRID}
        res["mde_q"] = {k: next((float(q) for q, v in g.items() if v["reject"] >= ALT_MIN), None)
                        for k, g in res["grid"].items()}
    return res


def q5(x):
    return {str(p): float(np.percentile(x, p)) for p in (5, 25, 50, 75, 95)}


def train(S, ctrl, D0, R, w, rng):
    T, p, d, tm, s = D0.test(S["axis"], rng, N_PERM_TRAIN)
    sd = float(np.std(tm, ddof=1))
    res = {"T": T, "p_two_sided": p, "n_trees": D0.ntree, "n_pairs": D0.n_pairs,
           "pair_d": q5(d), "tree_mean_d": q5(tm), "tree_mean_d_sd": sd,
           "mean_P2_true_mother": float(np.mean(s[:, 0])), "mean_P2_controls": float(np.nanmean(s[:, 1:])),
           "mde_sd_units_label_only": {f"n={n}": 2.8 * sd / np.sqrt(n)
                                       for n in (int(round(0.5 * D0.ntree)), D0.ntree)},
           "exploratory": {}}
    for f in (0.5, 2.0):
        D = Design(S, controls(S, w, f * BAND * R))
        if D.n_pairs:
            Tf, pf, *_ = D.test(S["axis"], rng, N_PERM_TRAIN)
            res["exploratory"][f"band x{f}"] = {"T": Tf, "p_two_sided": pf, "n_trees": D.ntree,
                                                "n_pairs": D.n_pairs}
    return res


def run_all():
    rng = np.random.default_rng(SEED)
    S, units, skipped = load_train()
    out = {"note": "train trees only; positions never saved; stages gated in order", "movies": units,
           "skipped_divisions": skipped}
    out["stage0"], ctrl, D0 = stage0(S)
    s0 = out["stage0"]
    print("stage0:", json.dumps(s0))
    if s0["pass"]:
        out["smoke"] = smoke(S, ctrl, D0, s0["R"], s0["median_lag_min"], rng)
        print("smoke pass:", out["smoke"]["pass"], out["smoke"]["fails"])
        if out["smoke"]["pass"]:
            out["train"] = train(S, ctrl, D0, s0["R"], s0["window_min"], rng)
            print("train:", json.dumps({k: out["train"][k] for k in ("T", "p_two_sided", "n_trees", "n_pairs")}))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return {"written": "experiments/results_qs2/n2_axis.json"}


# ---------- selftest ----------

TOY = b"""<root><AllSpots>
<Spot ID="1" FRAME="0" POSITION_X="0" POSITION_Y="0" POSITION_Z="0"/>
<Spot ID="2" FRAME="1" POSITION_X="-1" POSITION_Y="0" POSITION_Z="0"/>
<Spot ID="3" FRAME="1" POSITION_X="1" POSITION_Y="0" POSITION_Z="0"/>
<Spot ID="4" FRAME="2" POSITION_X="-1" POSITION_Y="0" POSITION_Z="0"/>
<Spot ID="5" FRAME="3" POSITION_X="-1" POSITION_Y="-1" POSITION_Z="0"/>
<Spot ID="6" FRAME="3" POSITION_X="-1" POSITION_Y="1" POSITION_Z="0"/>
</AllSpots><AllTracks><Track>
<Edge SPOT_SOURCE_ID="1" SPOT_TARGET_ID="2"/><Edge SPOT_SOURCE_ID="1" SPOT_TARGET_ID="3"/>
<Edge SPOT_SOURCE_ID="2" SPOT_TARGET_ID="4"/><Edge SPOT_SOURCE_ID="4" SPOT_TARGET_ID="5"/>
<Edge SPOT_SOURCE_ID="4" SPOT_TARGET_ID="6"/>
</Track></AllTracks></root>"""


def synth(rng, n_trees=80, movies=2, box=30.0, step=8.0, depth=3, cyc=10, dt=2.0, drift=0.0):
    """Synthetic division structure: binary trees of `depth` division generations, random-walk positions plus a
    shared per-movie drift of `drift` x step per generation."""
    tables = []
    for m in range(1, movies + 1):
        dv = unit(rng.normal(size=3)) * step * drift
        tree, t, pos, pairs = [], [], [], []
        for k in range(n_trees // movies):
            frontier = [(None, rng.uniform(0, box, 3), int(rng.integers(0, 4)))]
            for _ in range(depth):
                nxt = []
                for par, p, tt in frontier:
                    i = len(t)
                    tree.append(f"M{m}:{k}")
                    t.append(tt)
                    pos.append(p)
                    if par is not None and (not pairs or pairs[-1][0] != par):
                        pairs.append((par, i))
                    for _ in range(2):
                        kid = p + dv + rng.normal(0, step / np.sqrt(3), 3)
                        nxt.append((i, kid, tt + cyc + int(rng.integers(-1, 2))))
                frontier = nxt
        n = len(t)
        tables.append((m, dt, {"tree": np.array(tree, dtype=object), "t": np.array(t, float),
                               "pos": np.array(pos), "axis": unit(rng.normal(size=(n, 3))),
                               "pairs": np.array(pairs, int)}))
    return pool_movies(tables)


def selftest():
    rng = np.random.default_rng(1)
    r = {}
    a = np.array([[1.0, 0, 0]])
    r["p2_aligned"] = bool(np.isclose(p2(a, a)[0], 1.0))
    r["p2_orthogonal"] = bool(np.isclose(p2(a, np.array([[0, 1.0, 0]]))[0], -0.5))
    frame, edges = ds_emb_seal.load(io.BytesIO(TOY))
    unit_of = qs_e5.components(frame, edges, 9)
    pos = load_positions(io.BytesIO(TOY), set(frame))
    tb, _ = div_table(frame, edges, unit_of, set(unit_of.values()), pos)
    r["toy_divisions_2_pairs_1"] = len(tb["t"]) == 2 and tb["pairs"].tolist() == [[0, 1]]
    r["toy_axes"] = bool(np.allclose(tb["axis"][0], [1, 0, 0]) and np.allclose(tb["axis"][1], [0, 1, 0]))
    nulls = []
    for drift in (0.0, 2.0):
        S = synth(rng, drift=drift)
        r["one_pair_per_mother"] = len(set(S["pairs"][:, 0].tolist())) == len(S["pairs"])
        add_centres(S)
        R = float(np.median(S["resid"]))
        ctrl = controls(S, 6.0, BAND * R)
        D = Design(S, ctrl)
        r[f"synth_usable_trees_drift{drift}"] = D.ntree
        trees = set(S["tree"][D.mothers].tolist())
        for f in (1.0, 4.0, 8.0):
            k = f"null_reject_drift{drift}_L{f}R"
            r[k] = reject_rate(S, ctrl, trees, len(trees), "null", 0, f * R, 150, rng, perms=300)["reject"]
            nulls.append(r[k])
    mem = reject_rate(S, ctrl, trees, len(trees), "memory", 0.8, R, 40, rng, perms=300)
    ort = reject_rate(S, ctrl, trees, len(trees), "orthogonal", 0.8, R, 40, rng, perms=300)
    r["memory_reject"], r["orthogonal_reject"] = mem["reject"], ort["reject"]
    r["memory_T_positive"], r["orthogonal_T_negative"] = mem["mean_T"] > 0, ort["mean_T"] < 0
    nulls_ok = all(x <= 0.12 for x in nulls)
    r["pass"] = bool(r["p2_aligned"] and r["p2_orthogonal"] and r["toy_divisions_2_pairs_1"] and r["toy_axes"]
                     and r["one_pair_per_mother"] and nulls_ok and mem["reject"] >= 0.8 and ort["reject"] >= 0.8
                     and r["memory_T_positive"] and r["orthogonal_T_negative"])
    return r


if __name__ == "__main__":
    arg = sys.argv[1:] or ["selftest"]
    if arg[0] == "selftest":
        print(json.dumps(selftest(), default=float))
    elif arg[0] == "all":
        print(json.dumps(run_all()))
    else:
        raise SystemExit(f"unknown stage {arg[0]}")
