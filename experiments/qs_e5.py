"""QS-E5 stage 0b: readout inventory of the E5 track trees (mouse gastrulation manual tracks, S-BIAD1682).

Pre-registered 2026-11-25 (EXPERIMENT_LOG.md, "## QS-E5", Stage 0b). Counts and structure only.

    python experiments/qs_e5.py selftest     # toy XML strings, no real file
    python experiments/qs_e5.py inventory    # downloads Movie1-4 MaMuT XML at run time (sha256 pinned), writes
                                             # experiments/results_c17/qs_e5_inventory.json

Pass 1 reads spot ID, FRAME and edges only, builds units ('M<movie>:<earliest spot ID>', ds_emb_seal naming) and keeps
train units via partitions.train_only("mouse_gast_tree", ...). Pass 2 reads attributes of train-unit spots, edges and
tracks only, and records presence shares and numbers of distinct values (counts only). No attribute value, distribution,
intensity or position is saved or printed. The file list is read for the time-resolution column and the file path only.
"""

import csv
import hashlib
import io
import json
import os
import sys
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ds_emb_seal  # noqa: E402
import partitions  # noqa: E402

DL = os.path.join(HERE, "..", "data", "ext_mouse_gast_tracks", "inv_raw")  # fresh directory, not raw/
OUT = os.path.join(HERE, "results_c17", "qs_e5_inventory.json")
BASE = "https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/682/S-BIAD1682/Files/"
LIST_URL = BASE + "File_list_no_zarr.tsv"
MOVIES = [1, 2, 3, 4]
SHA256 = {  # data/ext_mouse_gast_tracks/raw/SHA256SUMS
    1: "906844b721243f3a27df6c86ec2ac66ed6a57e494159fb3d4753300530c54871",
    2: "2c4f937c83638495119c1cdb280c1ac744a2c20acd535ec246fa6d7e5e3fca7a",
    3: "58b930f30ccb79edb52511d8465155c8e66f2ca9f5fa8ece06bbc2b539069525",
    4: "360952ac155657c0af99bae1b2c7790cf47988e4cabfb570f2b912ce14a9fe38",
}
# attributes whose number of distinct values is counted (values never kept in the output)
NAMED_DISTINCT = {"MANUAL_SPOT_COLOR", "MANUAL_EDGE_COLOR"}
LABEL_HINTS = ("NAME", "LABEL", "COLOR")


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "cellware-inventory/1.0"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read()


def fetch_movie(m):
    os.makedirs(DL, exist_ok=True)
    path = os.path.join(DL, f"Movie{m}-mamut.xml")
    if not os.path.exists(path):
        raw = _get(f"{BASE}Movie{m}/Movie{m}-mamut.xml")
        with open(path, "wb") as fh:
            fh.write(raw)
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != SHA256[m]:
        raise SystemExit(f"Movie{m}: sha256 mismatch")
    return path


# ---------- structure ----------

def components(frame, edges, movie):
    """{spot: unit}, unit named exactly as ds_emb_seal.trees: 'M<movie>:<earliest spot ID>'."""
    adj = defaultdict(set)
    for a, b in edges:
        if a in frame and b in frame:
            adj[a].add(b)
            adj[b].add(a)
    seen, out = set(), {}
    for s in frame:
        if s in seen:
            continue
        comp, stack = [], [s]
        seen.add(s)
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        root = min(comp, key=lambda x: (frame[x], int(x)))
        for x in comp:
            out[x] = f"M{movie}:{root}"
    return out


def cycles(frame, edges, unit_of, keep, last_frame):
    """Cell-cycle segments of the units in `keep`, per generation (= divisions above the segment start).

    A segment is a maximal non-branching chain. Complete = born at a division and ends at a division. Censored-start =
    root segment (no observed birth) that ends at a division. Censored-end = born at a division and ends at the last
    frame or at a leaf before it. Both = root segment with a censored end.
    Returns {generation: Counter(complete, censored_start, censored_end, both_censored)}."""
    kids, parents = defaultdict(list), defaultdict(list)
    for a, b in edges:
        if a not in frame or b not in frame or unit_of.get(a) not in keep:
            continue
        s, t = (a, b) if frame[a] < frame[b] else (b, a)
        kids[s].append(t)
        parents[t].append(s)
    nodes = [n for n in frame if unit_of.get(n) in keep]
    gen = {}
    for n in sorted(nodes, key=lambda x: frame[x]):  # parents come earlier in time
        ps = parents[n]
        gen[n] = min((gen[p] + (len(kids[p]) == 2) for p in ps), default=0)
    res = defaultdict(Counter)
    for n in nodes:
        ps = parents[n]
        born_at_div = len(ps) == 1 and len(kids[ps[0]]) == 2
        if len(ps) == 1 and not born_at_div:
            continue  # inside a chain, not a segment start
        if len(ps) > 1:
            born_at_div = False  # merge: treated as not observed birth
        end = n
        while len(kids[end]) == 1 and len(parents[kids[end][0]]) == 1:
            end = kids[end][0]
        ends_at_div = len(kids[end]) == 2
        if born_at_div and ends_at_div:
            k = "complete"
        elif not born_at_div and ends_at_div:
            k = "censored_start"
        elif born_at_div:
            k = "censored_end"
        else:
            k = "both_censored"
        res[gen[n]][k] += 1
        res[gen[n]]["end_at_last_frame"] += (not ends_at_div) and frame[end] >= last_frame
    return res


# ---------- pass 2 ----------

class Attr:
    """Presence count and distinct-value count per attribute (values held in memory only, never output)."""

    def __init__(self):
        self.n = 0
        self.present = Counter()
        self.vals = defaultdict(set)

    def add(self, attrib):
        self.n += 1
        for k, v in attrib.items():
            self.present[k] += 1
            if k in NAMED_DISTINCT or any(h in k.upper() for h in LABEL_HINTS):
                self.vals[k].add(v)

    def summary(self):
        return {"n": self.n,
                "present_share": {k: round(c / self.n, 4) for k, c in sorted(self.present.items())} if self.n else {},
                "distinct_values": {k: len(v) for k, v in sorted(self.vals.items())}}


def pass2(path, unit_of, keep):
    spots, edges, tracks = Attr(), Attr(), Attr()
    pending, no_unit_tracks = [], 0
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag == "Spot":
            if unit_of.get(el.get("ID")) in keep:
                spots.add(el.attrib)
            el.clear()
        elif el.tag == "Edge":
            u = unit_of.get(el.get("SPOT_SOURCE_ID"))
            if u in keep:
                edges.add(el.attrib)
            pending.append(u)
            el.clear()
        elif el.tag == "Track":
            us = {u for u in pending if u is not None}
            pending = []
            if not us:
                no_unit_tracks += 1
            elif us & keep:
                tracks.add(el.attrib)
            el.clear()
    return {"spots": spots.summary(), "edges": edges.summary(), "tracks": tracks.summary(),
            "tracks_without_edges": no_unit_tracks}


# ---------- file list (time resolution only) ----------

def time_resolution(raw):
    """{movie: sorted set of the time-resolution strings of that movie's rows}. Reads two columns only."""
    rows = csv.reader(io.StringIO(raw.decode("utf-8", "ignore")), delimiter="\t")
    head = next(rows)
    low = [h.strip().lower() for h in head]
    ci_path = next((i for i, h in enumerate(low) if h in ("path", "file", "files", "name")
                    or "path" in h), None)
    ci_t = next((i for i, h in enumerate(low) if "time" in h and ("resol" in h or "interval" in h)), None)
    if ci_path is None or ci_t is None:
        raise SystemExit(f"file list columns not recognised (path={ci_path}, time={ci_t}); headers: {head}")
    out = defaultdict(set)
    for row in rows:
        if len(row) <= max(ci_path, ci_t):
            continue
        for m in MOVIES:
            if f"Movie{m}/" in row[ci_path] or f"Movie{m}-" in row[ci_path]:
                if row[ci_t].strip():
                    out[m].add(row[ci_t].strip())
    return {m: sorted(v) for m, v in out.items()}


# ---------- stages ----------

def selftest():
    # --- unit naming equals ds_emb_seal.trees naming
    frame = {"5": 0, "6": 1, "7": 2, "8": 2, "9": 0, "10": 3}
    edges = [("5", "6"), ("6", "7"), ("6", "8"), ("9", "10")]  # tree rooted at 5; tree rooted at 9
    u = components(frame, edges, 2)
    assert set(u.values()) == {"M2:5", "M2:9"} == set(ds_emb_seal.trees(frame, edges, 2)), u
    assert u["7"] == "M2:5" and u["10"] == "M2:9"
    # merge of two roots -> one unit named by the earliest spot
    um = components({"1": 0, "2": 0, "3": 1}, [("1", "3"), ("2", "3")], 1)
    assert set(um.values()) == {"M1:1"}

    # --- train_only filtering via a monkeypatched map
    import pandas as pd
    real = partitions.partition_of
    partitions.partition_of = lambda ds, unit: "train" if unit == "M2:5" else "A"
    try:
        df = partitions.train_only("mouse_gast_tree", pd.DataFrame({"unit": sorted(set(u.values()))}), "unit")
    finally:
        partitions.partition_of = real
    keep = set(df["unit"])
    assert keep == {"M2:5"}, keep

    # --- cycle classification on a toy tree (frames in 0..9, last frame 9)
    # root segment 0-1-2 divides at 2 -> A: 3-4 divides at 4 -> A1: 5 (leaf at frame 5, censored end),
    #                                                         A2: 6-7-8-9 (ends at last frame)
    #                                    -> B: 10 (frame 3, leaf) ; second root C 20-21 (leaf, both censored)
    fr = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 5, "7": 6, "8": 7, "9": 9, "10": 3, "20": 0, "21": 1}
    ed = [("0", "1"), ("1", "2"), ("2", "3"), ("3", "4"), ("4", "5"), ("4", "6"), ("6", "7"), ("7", "8"), ("8", "9"),
          ("2", "10"), ("20", "21")]
    uo = components(fr, ed, 1)
    assert set(uo.values()) == {"M1:0", "M1:20"}
    cy = cycles(fr, ed, uo, {"M1:0", "M1:20"}, 9)
    assert dict(cy[0]) == {"censored_start": 1, "both_censored": 1, "end_at_last_frame": 0}, dict(cy[0])
    # generation 1: segments born at the first division (3-4 complete; 10 leaf censored end)
    assert cy[1]["complete"] == 1 and cy[1]["censored_end"] == 1, dict(cy[1])
    # generation 2: segment 5 leaf censored end; segment 6-7-8-9 ends at last frame
    assert cy[2]["censored_end"] == 2 and cy[2]["end_at_last_frame"] == 1 and cy[2]["complete"] == 0, dict(cy[2])
    only_c = cycles(fr, ed, uo, {"M1:20"}, 9)  # unit filtering
    assert set(only_c) == {0} and only_c[0]["both_censored"] == 1

    # --- presence and distinct-value counting on toy MaMuT-like XML (pass 2)
    xml = ("<TrackMate><Model><AllSpots><SpotsInFrame frame='0'>"
           "<Spot ID='5' FRAME='0' MANUAL_SPOT_COLOR='1' X='1'/><Spot ID='6' FRAME='1' MANUAL_SPOT_COLOR='2'/>"
           "<Spot ID='9' FRAME='0' MANUAL_SPOT_COLOR='7'/></SpotsInFrame></AllSpots><AllTracks>"
           "<Track name='a' TRACK_ID='0'><Edge SPOT_SOURCE_ID='5' SPOT_TARGET_ID='6' MANUAL_EDGE_COLOR='3'/>"
           "<Edge SPOT_SOURCE_ID='6' SPOT_TARGET_ID='7'/></Track>"
           "<Track name='b' TRACK_ID='1'><Edge SPOT_SOURCE_ID='9' SPOT_TARGET_ID='10' MANUAL_EDGE_COLOR='4'/></Track>"
           "</AllTracks></Model></TrackMate>")
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.xml")
        with open(p, "w") as fh:
            fh.write(xml)
        uu = {"5": "M2:5", "6": "M2:5", "7": "M2:5", "8": "M2:5", "9": "M2:9", "10": "M2:9"}
        r = pass2(p, uu, {"M2:5"})
    assert r["spots"]["n"] == 2 and r["spots"]["present_share"]["MANUAL_SPOT_COLOR"] == 1.0, r
    assert r["spots"]["present_share"]["X"] == 0.5 and r["spots"]["distinct_values"] == {"MANUAL_SPOT_COLOR": 2}, r
    assert r["edges"]["n"] == 2 and r["edges"]["present_share"]["MANUAL_EDGE_COLOR"] == 0.5, r
    assert r["edges"]["distinct_values"] == {"MANUAL_EDGE_COLOR": 1}, r
    assert r["tracks"]["n"] == 1 and r["tracks"]["distinct_values"] == {"name": 1}, r

    # --- file list: two columns read, genotype column ignored
    tsv = b"Files\tTime resolution (min)\tGenotype\nMovie1/Movie1-mamut.xml\t10\tSECRET\nMovie2/x.tif\t5\tSECRET\n"
    assert time_resolution(tsv) == {1: ["10"], 2: ["5"]}
    return {"ok": True}


def inventory():
    raw_list = _get(LIST_URL)
    tres = time_resolution(raw_list)
    res = {"file_list_sha256": hashlib.sha256(raw_list).hexdigest(), "movies": {}}
    for m in MOVIES:
        path = fetch_movie(m)
        # pass 1: spot ID, FRAME, edges only
        frame, edges = ds_emb_seal.load(path)
        unit_of = components(frame, edges, m)
        import pandas as pd  # noqa: PLC0415
        df = partitions.train_only("mouse_gast_tree", pd.DataFrame({"unit": sorted(set(unit_of.values()))}), "unit")
        keep = set(df["unit"])
        last = max(frame.values())
        cyc = cycles(frame, edges, unit_of, keep, last)
        # pass 2: attributes of train-unit spots, edges and tracks
        attrs = pass2(path, unit_of, keep)
        res["movies"][f"M{m}"] = {
            "frame_interval": tres.get(m, []),
            "frames": last + 1,
            "train_units": len(keep),
            "train_spots": sum(unit_of[s] in keep for s in frame),
            "cycles_per_generation": {str(g): dict(c) for g, c in sorted(cyc.items())},
            **attrs,
        }
        print(f"M{m}: train units {len(keep)}, spots {attrs['spots']['n']}, edges {attrs['edges']['n']}, "
              f"tracks {attrs['tracks']['n']}, frame interval {tres.get(m, [])}, frames {last + 1}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    tot = Counter()
    for v in res["movies"].values():
        for c in v["cycles_per_generation"].values():
            tot.update({k: n for k, n in c.items() if k != "end_at_last_frame"})
    print("cycles (all generations, train):", dict(tot))
    return {"written": os.path.relpath(OUT, os.path.join(HERE, ".."))}


# ---------- QE1 gate 2: pair structure and power (structure only; observed cycle lengths feed one threshold) ----------

PCT = 95  # censoring window: birth frame + this percentile of the movie's complete cycle length <= last frame
KIN_MIN, WIN_MIN = 10.0, 20.0  # non-kin birth tolerance (+-10 min) and permutation window (20 min)
RHO_SIS, RHO_MD = 0.5, 0.3
SEED = 20261125


def build_cells(frame, edges, unit_of, keep, last_frame):
    """Cell-cycle segments of the units in `keep` (same definition as cycles()). Returns numpy arrays per segment:
    tree (unit name), gen, birth (frame), parent (segment index of the mother, -1 = none), eligible (bool).
    Eligible = complete (born at a division, ends at a division) and birth + PCT-th percentile of the complete cycle
    length (frames) of this movie <= last_frame. The percentile is a threshold used only to decide eligibility; no
    cycle length is returned or saved."""
    import numpy as np  # noqa: PLC0415

    kids, parents = defaultdict(list), defaultdict(list)
    for a, b in edges:
        if a not in frame or b not in frame or unit_of.get(a) not in keep:
            continue
        s, t = (a, b) if frame[a] < frame[b] else (b, a)
        kids[s].append(t)
        parents[t].append(s)
    nodes = [n for n in frame if unit_of.get(n) in keep]
    gen = {}
    for n in sorted(nodes, key=lambda x: frame[x]):
        gen[n] = min((gen[p] + (len(kids[p]) == 2) for p in parents[n]), default=0)
    starts = [n for n in nodes if len(parents[n]) != 1 or len(kids[parents[n][0]]) >= 2]
    seg_of_end, seg = {}, []  # seg: [start, end, born_at_div]
    for n in sorted(starts, key=lambda x: (frame[x], int(x))):
        end = n
        while len(kids[end]) == 1 and len(parents[kids[end][0]]) == 1:
            end = kids[end][0]
        born = len(parents[n]) == 1 and len(kids[parents[n][0]]) == 2
        seg_of_end[end] = len(seg)
        seg.append([n, end, born])
    par = np.full(len(seg), -1)
    for i, (n, end, born) in enumerate(seg):
        if born:
            par[i] = seg_of_end.get(parents[n][0], -1)
    complete = np.array([b and len(kids[e]) == 2 for _, e, b in seg], dtype=bool)
    length = np.array([frame[e] - frame[n] + 1 for n, e, _ in seg])
    birth = np.array([frame[n] for n, _, _ in seg])
    thr = np.percentile(length[complete], PCT) if complete.any() else np.inf
    elig = complete & (birth + thr <= last_frame)
    return {"tree": np.array([unit_of[n] for n, _, _ in seg], dtype=object),
            "gen": np.array([gen[n] for n, _, _ in seg]), "birth": birth, "parent": par, "eligible": elig}


def pair_index(cells, dt):
    """Index arrays of sister pairs, mother-daughter pairs and time-matched non-kin pairs among eligible cells.
    Sisters: both daughters of one division, both eligible. Mother-daughter: mother and a daughter both eligible.
    Non-kin: eligible cells of the same generation with births within KIN_MIN minutes and different mothers."""
    import numpy as np  # noqa: PLC0415

    par, el, gen, birth = cells["parent"], cells["eligible"], cells["gen"], cells["birth"]
    idx = np.flatnonzero(el & (par >= 0))
    byp = defaultdict(list)
    for i in idx:
        byp[par[i]].append(i)
    sis = np.array([v for v in byp.values() if len(v) == 2], dtype=int).reshape(-1, 2)
    md = np.array([(par[i], i) for i in idx if el[par[i]]], dtype=int).reshape(-1, 2)
    tol = int(KIN_MIN / dt + 1e-9)
    nk = []
    for g in np.unique(gen[idx]):
        ii = idx[gen[idx] == g]
        for k, i in enumerate(ii):
            for j in ii[k + 1:]:
                if abs(birth[i] - birth[j]) <= tol and par[i] != par[j]:
                    nk.append((i, j))
    return sis, md, np.array(nk, dtype=int).reshape(-1, 2)


def pairs_counts(cells, dt):
    sis, md, nk = pair_index(cells, dt)
    tree, gen = cells["tree"], cells["gen"]
    per_tree = defaultdict(Counter)
    for t in set(tree):
        per_tree[t]["eligible_cells"] = int(((tree == t) & cells["eligible"]).sum())
    for a, b in sis:
        per_tree[tree[a]]["sister_pairs"] += 1
    for a, b in md:
        per_tree[tree[a]]["mother_daughter_pairs"] += 1
    for a, b in nk:
        for t in {tree[a], tree[b]}:
            per_tree[t]["nonkin_pairs"] += 1
    by_gen = defaultdict(Counter)
    for a, b in sis:
        by_gen[int(gen[a])]["sister_pairs"] += 1
    for a, b in md:
        by_gen[int(gen[b])]["mother_daughter_pairs"] += 1
    for a, b in nk:
        by_gen[int(gen[a])]["nonkin_pairs"] += 1
    return {"segments": int(len(tree)), "eligible_cells": int(cells["eligible"].sum()),
            "sister_pairs": int(len(sis)), "mother_daughter_pairs": int(len(md)), "nonkin_pairs": int(len(nk)),
            "by_generation": {str(g): dict(c) for g, c in sorted(by_gen.items())},
            "per_tree": {t: dict(c) for t, c in sorted(per_tree.items())}}


def load_train_cells(m):
    """Download (sha256-checked), pass 1 only (spot ID, FRAME, edges), train units only."""
    import pandas as pd  # noqa: PLC0415

    frame, edges = ds_emb_seal.load(fetch_movie(m))
    unit_of = components(frame, edges, m)
    df = partitions.train_only("mouse_gast_tree", pd.DataFrame({"unit": sorted(set(unit_of.values()))}), "unit")
    keep = set(df["unit"])
    return build_cells(frame, edges, unit_of, keep, max(frame.values())), keep


def frame_intervals():
    tres = time_resolution(_get(LIST_URL))
    return {m: float(tres[m][0]) for m in MOVIES}


def pairs_stage():
    dts = frame_intervals()
    res = {"note": "train trees only; counts of eligible cycles and pairs; nonkin_pairs per tree counts each pair once "
                   "for each distinct tree of its two members", "movies": {}}
    for m in MOVIES:
        cells, keep = load_train_cells(m)
        v = res["movies"][f"M{m}"] = {"frame_interval_min": dts[m], "train_units": len(keep),
                                      **pairs_counts(cells, dts[m])}
        print(f"M{m}: trees {len(keep)}, eligible cells {v['eligible_cells']}, sister pairs {v['sister_pairs']}, "
              f"mother-daughter pairs {v['mother_daughter_pairs']}, non-kin pairs {v['nonkin_pairs']}")
    out = os.path.join(HERE, "results_c17", "qs_e5_pairs.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    return {"written": "experiments/results_c17/qs_e5_pairs.json"}


# ---------- power ----------

class Struct:
    """Pair structure pooled over movies (arrays only; no observed cycle lengths)."""

    def __init__(self, cell_sets, dts):
        import numpy as np  # noqa: PLC0415

        parts, off = [], 0
        for m, cells in cell_sets:
            sis, md, nk = pair_index(cells, dts[m])
            parts.append((m, cells, sis + off, md + off, nk + off, off))
            off += len(cells["tree"])
        self.movie = np.concatenate([np.full(len(c["tree"]), m) for m, c, *_ in parts])
        names = [f"M{m}|{t}" for m, c, *_ in parts for t in c["tree"]]
        self.trees = sorted(set(names))
        tid = {t: i for i, t in enumerate(self.trees)}
        self.tree = np.array([tid[x] for x in names])
        self.tree_movie = np.array([int(t.split("|")[0][1:]) for t in self.trees])
        self.gen = np.concatenate([c["gen"] for _, c, *_ in parts])
        self.birth = np.concatenate([c["birth"] for _, c, *_ in parts])
        self.parent = np.concatenate([np.where(c["parent"] >= 0, c["parent"] + p[5], -1)
                                      for (_, c, *_), p in zip(parts, parts)])
        self.elig = np.concatenate([c["eligible"] for _, c, *_ in parts])
        self.sis = np.concatenate([p[2] for p in parts])
        self.md = np.concatenate([p[3] for p in parts])
        self.nk = np.concatenate([p[4] for p in parts])
        win = np.array([int(round(WIN_MIN / dts[m])) for m in self.movie])
        keys = list(zip(self.movie.tolist(), self.gen.tolist(), (self.birth // win).tolist()))
        u = {k: i for i, k in enumerate(sorted(set(keys)))}
        self.stratum = np.array([u[k] for k in keys])


def simulate(S, rng, rho_sis, rho_md):
    """Gaussian log cycle lengths over every segment: x = r*mother + s*(shared by the two daughters) + e*eps,
    unit variance, sister covariance r^2+s^2 = rho_sis, mother-daughter covariance r = rho_md, roots N(0,1)."""
    import numpy as np  # noqa: PLC0415

    r = rho_md
    s = np.sqrt(max(rho_sis - r * r, 0.0))
    e = np.sqrt(1.0 - r * r - s * s)
    n = len(S.gen)
    x = rng.standard_normal(n)
    u = rng.standard_normal(n)
    for g in range(1, int(S.gen.max()) + 1):
        i = np.flatnonzero((S.gen == g) & (S.parent >= 0))
        p = S.parent[i]
        x[i] = r * x[p] + s * u[p] + e * x[i]
    return x


def _sums(z, a, b, tr, T):
    import numpy as np  # noqa: PLC0415

    return np.bincount(tr, weights=z[a] * z[b], minlength=T), np.bincount(tr, minlength=T).astype(float)


def _ratio(S, N, w=None):
    import numpy as np  # noqa: PLC0415

    if w is None:
        return S.sum() / max(N.sum(), 1.0)
    return (w @ S) / np.maximum(w @ N, 1.0)


def one_test(S, x, sel, rng, perms, boots):
    """Returns (reject_sister, reject_mother_daughter). D = mean standardised product over kin pairs minus the same over
    time-matched non-kin pairs. Null centre = mean D over censoring-aware permutations (cells of the same movie,
    generation and 20-min birth window); SE = tree-cluster bootstrap; reject if |D - centre| > 1.96 SE.
    Non-kin pairs are clustered by the tree of their first member."""
    import numpy as np  # noqa: PLC0415

    T = len(S.trees)
    ci = np.flatnonzero(S.elig & sel[S.tree])
    if len(ci) < 4:
        return False, False
    sets = {}
    for name, P in (("sis", S.sis), ("md", S.md), ("nk", S.nk)):
        keep = sel[S.tree[P[:, 0]]] & sel[S.tree[P[:, 1]]] if len(P) else np.zeros(0, dtype=bool)
        sets[name] = (P[keep, 0], P[keep, 1], S.tree[P[keep, 0]])
    if any(len(v[0]) == 0 for v in sets.values()):
        return False, False
    chosen = np.flatnonzero(sel)

    def stats(vals):
        z = np.zeros(len(x))
        v = vals[ci]
        z[ci] = (v - v.mean()) / (v.std() or 1.0)
        return {k: _sums(z, a, b, tr, T) for k, (a, b, tr) in sets.items()}

    def D(st, w=None):
        nk = _ratio(*st["nk"], w)
        return _ratio(*st["sis"], w) - nk, _ratio(*st["md"], w) - nk

    obs = stats(x)
    d_obs = D(obs)
    s_ci = S.stratum[ci]
    base = np.argsort(s_ci, kind="stable")
    dp = []
    for _ in range(perms):
        rnd = np.lexsort((rng.random(len(ci)), s_ci))
        v = np.zeros(len(x))
        v[ci[base]] = x[ci[rnd]]
        dp.append(D(stats(v)))
    centre = np.mean(dp, axis=0)
    pick = chosen[rng.integers(0, len(chosen), size=(boots, len(chosen)))]
    w = np.stack([np.bincount(p, minlength=T) for p in pick]).astype(float)
    nkb = _ratio(*obs["nk"], w)
    se = [np.std(_ratio(*obs[k], w) - nkb) for k in ("sis", "md")]
    return tuple(bool(abs(d_obs[i] - centre[i]) > 1.96 * se[i]) for i in (0, 1))


def stratified_subsample(S, n_total, rng):
    """Select n_total of the trees, stratified by movie (largest remainder), as the seal."""
    import numpy as np  # noqa: PLC0415

    sel = np.zeros(len(S.trees), dtype=bool)
    movies = sorted(set(S.tree_movie.tolist()))
    share = {m: n_total * (S.tree_movie == m).sum() / len(S.trees) for m in movies}
    k = {m: int(share[m]) for m in movies}
    for m in sorted(movies, key=lambda m: -(share[m] - k[m]))[: n_total - sum(k.values())]:
        k[m] += 1
    for m in movies:
        sel[rng.choice(np.flatnonzero(S.tree_movie == m), size=k[m], replace=False)] = True
    return sel


def power_run(S, reps, n_trees, rho_sis, rho_md, seed, perms=100, boots=200):
    import numpy as np  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    hit = np.zeros(2)
    for _ in range(reps):
        sel = np.ones(len(S.trees), dtype=bool) if n_trees >= len(S.trees) else stratified_subsample(S, n_trees, rng)
        hit += one_test(S, simulate(S, rng, rho_sis, rho_md), sel, rng, perms, boots)
    return {"reps": reps, "trees": int(min(n_trees, len(S.trees))), "rho_sister": rho_sis,
            "rho_mother_daughter": rho_md, "power_sister": float(hit[0] / reps),
            "power_mother_daughter": float(hit[1] / reps)}


def power_stage(reps=1000):
    dts = frame_intervals()
    S = Struct([(m, load_train_cells(m)[0]) for m in MOVIES], dts)
    counts = {"trees": len(S.trees), "eligible_cells": int(S.elig.sum()), "sister_pairs": int(len(S.sis)),
              "mother_daughter_pairs": int(len(S.md)), "nonkin_pairs": int(len(S.nk))}
    print("train structure:", counts)
    res = {"counts": counts,
           "test": "D = kin product mean - non-kin product mean; permutation centre, cluster-bootstrap SE, alpha 0.05 "
                   "two-sided; 100 permutations, 200 bootstrap resamples per replicate",
           "partition_A_64_trees_planted": power_run(S, reps, 64, RHO_SIS, RHO_MD, SEED),
           "partition_A_64_trees_null": power_run(S, reps, 64, 0.0, 0.0, SEED + 1),
           "all_127_trees_planted": power_run(S, reps, len(S.trees), RHO_SIS, RHO_MD, SEED + 2)}
    for k in ("partition_A_64_trees_planted", "partition_A_64_trees_null", "all_127_trees_planted"):
        print(k, res[k])
    out = os.path.join(HERE, "results_c17", "qs_e5_power.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    return {"written": "experiments/results_c17/qs_e5_power.json"}


def selftest_qe1():
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    frame, edges, nid = {}, [], [0]

    def chain(f0, f1, parent=None):  # spots at frames f0..f1 in a chain, linked from `parent`
        ids = []
        for f in range(f0, f1 + 1):
            nid[0] += 1
            frame[str(nid[0])] = f
            ids.append(str(nid[0]))
        edges.extend(zip(ids[:-1], ids[1:]))
        if parent:
            edges.append((parent, ids[0]))
        return ids[0], ids[-1]

    # tree 1: root 0-4 divides; sisters a (5-9, len 5) and b (5-11, len 7); a divides into a1 (10-12, len 3,
    # divides again) and a leaf; b divides into leaves
    _, r1 = chain(0, 4)
    _, a_e = chain(5, 9, r1)
    _, b_e = chain(5, 11, r1)
    _, a1_e = chain(10, 12, a_e)
    chain(10, 16, a_e)
    chain(13, 14, a1_e)
    chain(13, 14, a1_e)
    chain(12, 16, b_e)
    chain(12, 16, b_e)
    # tree 2: kids c (born 6, len 6, divides) and a leaf d
    _, r2 = chain(0, 5)
    _, c_e = chain(6, 11, r2)
    chain(6, 8, r2)
    chain(12, 13, c_e)
    chain(12, 13, c_e)
    # tree 3: kid e born 28, len 5, divides: beyond the censoring window (last frame 33)
    _, r3 = chain(0, 27)
    _, e_e = chain(28, 32, r3)
    chain(28, 29, r3)
    chain(33, 33, e_e)
    chain(33, 33, e_e)
    uo = components(frame, edges, 1)
    assert len(set(uo.values())) == 3
    real = partitions.partition_of
    partitions.partition_of = lambda ds, unit: "train"
    try:
        df = partitions.train_only("mouse_gast_tree", pd.DataFrame({"unit": sorted(set(uo.values()))}), "unit")
    finally:
        partitions.partition_of = real
    keep = set(df["unit"])
    cells = build_cells(frame, edges, uo, keep, 33)
    # complete cells a(5) b(7) a1(3) c(6) e(5); 95th percentile 6.8 -> e (28 + 6.8 > 33) is outside the window
    assert int(cells["eligible"].sum()) == 4, int(cells["eligible"].sum())
    pc = pairs_counts(cells, 2.0)
    assert pc["sister_pairs"] == 1 and pc["mother_daughter_pairs"] == 1, pc  # (a,b); (a,a1)
    assert pc["nonkin_pairs"] == 2, pc  # gen-1 cells a, b, c: a-c and b-c (different mothers, births within 5 frames)
    assert sum(v.get("sister_pairs", 0) for v in pc["per_tree"].values()) == 1
    # train filtering: only tree-1 train -> tree 2 and 3 cells absent
    first = min(uo.values())
    partitions.partition_of = lambda ds, unit: "train" if unit == first else "A"
    try:
        k1 = set(partitions.train_only("mouse_gast_tree", pd.DataFrame({"unit": sorted(set(uo.values()))}), "unit")["unit"])
    finally:
        partitions.partition_of = real
    c1 = build_cells(frame, edges, uo, k1, 33)
    assert set(c1["tree"]) == k1 == {first}

    # power stage on a synthetic structure: 2 movies x 40 trees of perfect binary lineages (gen 0-3)
    sets = []
    rng = np.random.default_rng(1)
    for m in (1, 2):
        T, gen, birth, parent = [], [], [], []
        for t in range(40):
            base, start = len(T), 0
            for g, cnt in enumerate((1, 2, 4, 8)):
                for k in range(cnt):
                    T.append(f"M{m}:{t}")
                    gen.append(g)
                    birth.append(g * 40 + int(rng.integers(0, 4)))
                    parent.append(-1 if g == 0 else base + (2 ** (g - 1) - 1) + k // 2)
        sets.append((m, {"tree": np.array(T, dtype=object), "gen": np.array(gen), "birth": np.array(birth),
                         "parent": np.array(parent), "eligible": np.array(gen) >= 1}))
    S = Struct(sets, {1: 2.0, 2: 2.0})
    assert len(S.sis) > 0 and len(S.md) > 0 and len(S.nk) > 0
    # simulated moments: sister covariance 0.5, mother-daughter 0.3 (large draw)
    xs = np.array([simulate(S, np.random.default_rng(i), RHO_SIS, RHO_MD) for i in range(200)])
    assert abs(np.mean(xs[:, S.sis[:, 0]] * xs[:, S.sis[:, 1]]) - 0.5) < 0.05
    assert abs(np.mean(xs[:, S.md[:, 0]] * xs[:, S.md[:, 1]]) - 0.3) < 0.05
    pw = power_run(S, 60, 64, RHO_SIS, RHO_MD, 5, perms=40, boots=100)
    nl = power_run(S, 60, 64, 0.0, 0.0, 6, perms=40, boots=100)
    assert pw["power_sister"] > 0.7 and nl["power_sister"] < 0.2 and nl["power_mother_daughter"] < 0.2, (pw, nl)
    sel = stratified_subsample(S, 64, np.random.default_rng(0))
    assert sel.sum() == 64 and sel[S.tree_movie == 1].sum() == 32
    return {"pairs_toy": pc["sister_pairs"], "power_toy": pw["power_sister"], "null_toy": nl["power_sister"]}


if __name__ == "__main__":
    arg = sys.argv[1:] or ["selftest"]
    if arg[0] == "selftest":
        print(json.dumps(selftest()))
        print(json.dumps(selftest_qe1()))
    elif arg[0] == "pairs":
        print(json.dumps(pairs_stage()))
    elif arg[0] == "power":
        print(json.dumps(power_stage(int(arg[1]) if len(arg) > 1 else 1000)))
    elif arg[0] == "inventory":
        print(json.dumps(inventory()))
    else:
        raise SystemExit("usage: qs_e5.py selftest|inventory|pairs|power")
