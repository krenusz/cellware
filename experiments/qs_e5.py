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


if __name__ == "__main__":
    arg = sys.argv[1:] or ["selftest"]
    if arg[0] == "selftest":
        print(json.dumps(selftest()))
    elif arg[0] == "inventory":
        print(json.dumps(inventory()))
    else:
        raise SystemExit("usage: qs_e5.py selftest|inventory")
