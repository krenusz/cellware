"""DS-WIDE-G: S0 structure gate for the track-type leads W2 (six regeneration-recording projects) and W3 (one embryo).

Pre-registered in EXPERIMENT_LOG (DS-WIDE-G, amendments 1 and 2). Structure counts only: spots (frame) and links.
No tag, feature, position, name or id is read, printed or saved. Specimens carry neutral labels only.

    python -I experiments/ds_wide_gate.py --out experiments/results_c17/ds_wide_gate_counts.json   # fetch + count
    python -I experiments/ds_wide_gate.py --selftest                                              # toy forests + toy zip

Fetch:
  W2: the six .mastodon members of two zips by HTTP Range (tail, central directory, local header, member bytes;
      one request per range, Retry-After honoured once on 429/503; a non-206 answer aborts, so no zip body is pulled).
  W3: the one .mastodon file from its record (full GET, size and md5 checked against the record).
  File names are resolved at run time from the record listing, and are never printed or saved.
  Downloads go to data/ext_wide_g_tracks/raw under neutral names, with sha256 (and SHA256SUMS).
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import re
import struct
import sys
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mastodon_read  # noqa: E402

DATA = HERE.parent / "data" / "ext_wide_g_tracks" / "raw"
MIN_ARM = 5
GATE_ELIGIBLE = 75  # 0.4 x eligible >= 30
W2_REC, W3_REC = 15181497, 7296222
W2_LIMBS = ("li13", "li15", "li48", "li51", "li52")
UA = {"User-Agent": "research-structure-check/1.0"}


# ---------------------------------------------------------------- range access
class HttpRange:
    """Range reader that accepts only 206 answers with a matching Content-Range."""

    def __init__(self, url):
        self.url, self.requests, self.bytes = url, 0, 0

    def _req(self, rng):
        for attempt in (0, 1):
            req = urllib.request.Request(self.url, headers={**UA, "Range": rng})
            try:
                self.requests += 1
                with urllib.request.urlopen(req, timeout=120) as r:
                    if r.status != 206:
                        raise RuntimeError(f"range request answered {r.status}, expected 206; aborting")
                    cr = r.headers.get("Content-Range", "")
                    body = r.read()
                    self.bytes += len(body)
                    return body, cr
            except urllib.error.HTTPError as e:
                if e.code in (429, 503) and attempt == 0:
                    time.sleep(min(float(e.headers.get("Retry-After", "30") or 30), 120))
                    continue
                raise
        raise RuntimeError("unreachable")

    def tail(self, n):
        body, cr = self._req(f"bytes=-{n}")
        m = re.match(r"bytes (\d+)-(\d+)/(\d+)$", cr)
        if not m or int(m.group(2)) - int(m.group(1)) + 1 != len(body):
            raise RuntimeError("Content-Range mismatch on tail")
        return body, int(m.group(3))

    def get(self, start, n):
        body, cr = self._req(f"bytes={start}-{start + n - 1}")
        m = re.match(r"bytes (\d+)-(\d+)/(\d+)$", cr)
        if not m or int(m.group(1)) != start or len(body) != n:
            raise RuntimeError("Content-Range mismatch")
        return body


class BytesRange:
    """Local stand-in for the selftest: same interface, serves ranges from a buffer, counts requests."""

    def __init__(self, buf):
        self.buf, self.requests, self.bytes = buf, 0, 0

    def tail(self, n):
        self.requests += 1
        body = self.buf[-n:]
        self.bytes += len(body)
        return body, len(self.buf)

    def get(self, start, n):
        self.requests += 1
        body = self.buf[start:start + n]
        assert len(body) == n, "range beyond buffer"
        self.bytes += len(body)
        return body


# ---------------------------------------------------------------- zip members by range
def _zip64_extra(extra, need):
    """Return the values of the zip64 extra field for the requested 'need' flags in order (usize, csize, loff)."""
    e = 0
    while e + 4 <= len(extra):
        hid, dsz = struct.unpack("<HH", extra[e:e + 4])
        if hid == 1:
            d, q, vals = extra[e + 4:e + 4 + dsz], 0, []
            for _ in need:
                vals.append(struct.unpack("<Q", d[q:q + 8])[0])
                q += 8
            return vals
        e += 4 + dsz
    raise RuntimeError("zip64 field missing")


def central_directory(src):
    tail, total = src.tail(65536)
    i = tail.rfind(b"PK\x05\x06")
    if i < 0:
        raise RuntimeError("end of central directory not found")
    _, _, _, n_ent, _, cd_size, cd_off, _ = struct.unpack("<IHHHHIIH", tail[i:i + 22])
    if cd_off == 0xFFFFFFFF or n_ent == 0xFFFF or cd_size == 0xFFFFFFFF:
        j = tail.rfind(b"PK\x06\x07", 0, i)
        if j < 0:
            raise RuntimeError("zip64 locator not found")
        _, _, eocd64_off, _ = struct.unpack("<IIQI", tail[j:j + 20])
        base = total - len(tail)
        if eocd64_off >= base:
            rec = tail[eocd64_off - base:eocd64_off - base + 56]
        else:
            rec = src.get(eocd64_off, 56)
        _, _, _, _, _, _, n_ent, _, cd_size, cd_off = struct.unpack("<IQHHIIQQQQ", rec)
    cd = src.get(cd_off, cd_size)
    members, pos = [], 0
    while pos + 46 <= len(cd) and cd[pos:pos + 4] == b"PK\x01\x02":
        (_, _, _, _, meth, _, _, crc, csz, usz, nlen, elen, clen, _, _, _, loff) = struct.unpack(
            "<IHHHHHHIIIHHHHHII", cd[pos:pos + 46])
        name = cd[pos + 46:pos + 46 + nlen].decode("utf-8", "replace")
        extra = cd[pos + 46 + nlen:pos + 46 + nlen + elen]
        need = [k for k, v in (("u", usz), ("c", csz), ("o", loff)) if v == 0xFFFFFFFF]
        if need:
            vals = dict(zip(need, _zip64_extra(extra, need)))
            usz, csz, loff = vals.get("u", usz), vals.get("c", csz), vals.get("o", loff)
        members.append({"name": name, "method": meth, "crc": crc, "csize": csz, "usize": usz, "loff": loff})
        pos += 46 + nlen + elen + clen
    if len(members) != n_ent:
        raise RuntimeError(f"central directory entries {len(members)} != {n_ent}")
    return members, total


def extract_member(src, m):
    """Local header, then the compressed bytes (sizes from the central directory), inflate, verify CRC32 and size."""
    lh = src.get(m["loff"], 30)
    if lh[:4] != b"PK\x03\x04":
        raise RuntimeError("local header signature not found")
    nlen, elen = struct.unpack("<HH", lh[26:30])
    comp = src.get(m["loff"] + 30 + nlen + elen, m["csize"])
    if m["method"] == 8:
        data = zlib.decompress(comp, -15)
    elif m["method"] == 0:
        data = comp
    else:
        raise RuntimeError(f"unsupported compression method {m['method']}")
    if len(data) != m["usize"] or (zlib.crc32(data) & 0xFFFFFFFF) != m["crc"]:
        raise RuntimeError("member size or CRC32 mismatch")
    return data


def tracking_members(members, labels_expected):
    """Pick .mastodon members (skipping archive-metadata folders). Returns {label: member}; names are used for the
    limb label only and never returned or saved."""
    cand = [m for m in members if m["name"].lower().endswith(".mastodon") and not m["name"].startswith("__MACOSX")]
    if labels_expected is None:
        if len(cand) != 1:
            raise RuntimeError(f"expected 1 tracking member, found {len(cand)}")
        return {None: cand[0]}
    out = {}
    for m in cand:
        hit = [lab for lab in labels_expected if re.search(rf"(?<![0-9A-Za-z]){lab}(?![0-9])", m["name"].rsplit("/", 1)[-1])]
        if len(hit) != 1 or hit[0] in out:
            raise RuntimeError("limb label not unique in member name")
        out[hit[0]] = m
    if set(out) != set(labels_expected):
        raise RuntimeError(f"limb labels found {sorted(out)}")
    return out


# ---------------------------------------------------------------- counts (mirrors DS-EMB-G E5 spot_gate)
def gate_counts(frames, links, min_arm=MIN_ARM):
    """frames: int array (index = spot id); links: Nx2 array. Links are oriented forward in frame; same-frame links are
    counted and excluded. Division = spot with exactly 2 forward links. Eligible: an earlier division on the path
    back to the root (depth >= 1) and each arm (daughter plus all descendants, as E5: summed subtree sizes) has >=
    min_arm spots. Also reported: arm size by distinct spots (differs from the sum only under merges)."""
    n = len(frames)
    fs, ft = frames[links[:, 0]], frames[links[:, 1]]
    same = int((fs == ft).sum())
    keep = fs != ft
    a = np.where(fs <= ft, links[:, 0], links[:, 1])[keep].astype(np.int64)
    b = np.where(fs <= ft, links[:, 1], links[:, 0])[keep].astype(np.int64)
    outdeg, indeg = np.bincount(a, minlength=n), np.bincount(b, minlength=n)
    order_l = np.argsort(a, kind="stable")
    kids = b[order_l]
    start = np.concatenate([[0], np.cumsum(outdeg)])
    nodes = np.argsort(frames, kind="stable")
    isdiv = outdeg == 2
    anc = np.zeros(n, bool)
    for v in nodes:
        if outdeg[v] and (anc[v] or isdiv[v]):
            anc[kids[start[v]:start[v + 1]]] = True
    sub = np.ones(n, np.int64)
    for v in nodes[::-1]:
        if outdeg[v]:
            sub[v] = 1 + sub[kids[start[v]:start[v + 1]]].sum()

    def distinct_ge(root):  # >= min_arm distinct spots reachable (capped search)
        seen, stack = {int(root)}, [int(root)]
        while stack and len(seen) < min_arm:
            v = stack.pop()
            for c in kids[start[v]:start[v + 1]]:
                if int(c) not in seen:
                    seen.add(int(c))
                    stack.append(int(c))
        return len(seen) >= min_arm

    div_ids = np.flatnonzero(isdiv)
    elig = np.zeros(n, bool)
    elig_distinct = 0
    for v in div_ids:
        d1, d2 = kids[start[v]:start[v + 1]]
        if anc[v] and sub[d1] >= min_arm and sub[d2] >= min_arm:
            elig[v] = True
        if anc[v] and distinct_ge(d1) and distinct_ge(d2):
            elig_distinct += 1
    anc_e = np.zeros(n, bool)
    for v in nodes:
        if outdeg[v] and (anc_e[v] or elig[v]):
            anc_e[kids[start[v]:start[v + 1]]] = True
    # connected components (union-find over retained links)
    par = list(range(n))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for x, y in zip(a.tolist(), b.tolist()):
        rx, ry = find(x), find(y)
        if rx != ry:
            par[rx] = ry
    has_link = (outdeg + indeg) > 0
    roots = {find(int(v)) for v in np.flatnonzero(has_link)}
    elig_roots = {find(int(v)) for v in np.flatnonzero(elig)}
    hist = np.bincount(frames[frames >= 0]) if n else np.array([], int)
    hist_txt = ";".join(f"{i}:{c}" for i, c in enumerate(hist.tolist()) if c)
    return {
        "spots": n, "links": int(len(links)), "same_frame_links_excluded": same,
        "divisions": int(isdiv.sum()), "eligible_pairs": int(elig.sum()),
        "track_trees": len(roots), "isolated_spots": int((~has_link).sum()),
        "trees_with_eligible_pair": len(elig_roots), "non_nested_eligible_pairs": int((elig & ~anc_e).sum()),
        "spots_with_gt2_forward_links": int((outdeg > 2).sum()), "spots_with_gt1_incoming_link": int((indeg > 1).sum()),
        "eligible_pairs_arm_by_distinct_spots": elig_distinct,
        "frame_range": [int(frames.min()), int(frames.max())] if n else None,
        "frame_hist_sha256": hashlib.sha256(hist_txt.encode()).hexdigest(),
    }


# ---------------------------------------------------------------- fetch
def _json(url):
    req = urllib.request.Request(url, headers=UA)
    for attempt in (0, 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt == 0:
                time.sleep(min(float(e.headers.get("Retry-After", "30") or 30), 120))
                continue
            raise


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch_all(data_dir=DATA):
    """Returns {label: path} and a fetch log (counts, hashes; no names)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    paths, log = {}, {}
    rec = _json(f"https://zenodo.org/api/records/{W2_REC}")
    keys = {f["key"]: f for f in rec["files"]}
    for prefix, labels in (("Suppl_data_9_", W2_LIMBS), ("Suppl_data_5_", None)):
        hit = [k for k in keys if k.startswith(prefix) and k.endswith(".zip")]
        if len(hit) != 1:
            raise RuntimeError(f"zip with prefix {prefix}: {len(hit)} candidates")
        f = keys[hit[0]]
        src = HttpRange(f["links"]["self"])
        members, total = central_directory(src)
        if total != f["size"]:
            raise RuntimeError("archive size differs from the record")
        sel = tracking_members(members, labels)
        for lab, m in sel.items():
            label = f"W2-{lab}" if lab else "W2-F5"
            data = extract_member(src, m)
            p = data_dir / f"{label}.mastodon"
            p.write_bytes(data)
            paths[label] = p
            log[label] = {"bytes": len(data), "crc32_matches_directory": True, "sha256": _sha(p)}
        for lab in sel:
            log[f"W2-{lab}" if lab else "W2-F5"]["range_requests_for_archive"] = src.requests
    rec3 = _json(f"https://zenodo.org/api/records/{W3_REC}")
    hit = [f for f in rec3["files"] if f["key"].endswith(".mastodon")]
    if len(hit) != 1:
        raise RuntimeError("expected one tracking file in the second record")
    f = hit[0]
    req = urllib.request.Request(f["links"]["self"], headers=UA)
    h, body = hashlib.md5(), io.BytesIO()
    with urllib.request.urlopen(req, timeout=300) as r:
        while chunk := r.read(1 << 20):
            h.update(chunk)
            body.write(chunk)
    p = data_dir / "W3-e1.mastodon"
    p.write_bytes(body.getvalue())
    md5_ok = f"md5:{h.hexdigest()}" == f["checksum"] and len(body.getvalue()) == f["size"]
    if not md5_ok:
        raise RuntimeError("md5 or size mismatch against the record")
    paths["W3-e1"] = p
    log["W3-e1"] = {"bytes": len(body.getvalue()), "md5_matches_record": True, "sha256": _sha(p)}
    (data_dir / "SHA256SUMS").write_text("".join(f"{v['sha256']}  {k}.mastodon\n" for k, v in sorted(log.items())))
    for k in sorted(log):
        print(f"sha256 {k} {log[k]['sha256']} {log[k]['bytes']} bytes")
    return paths, log


# ---------------------------------------------------------------- gate
def specimen(path):
    try:
        rec = mastodon_read.read_mastodon(str(path))
        val = mastodon_read.validate(rec)
    except Exception as e:  # structural detail only
        return {"read_error": f"{type(e).__name__}: {str(e)[:300]}", "valid": False}
    out = {"validation": val}
    if val["valid"]:
        out["counts"] = gate_counts(rec["frames"], rec["links"])
    return out


def duplicate_check(per):
    keys = {k: (v["counts"]["spots"], v["counts"]["links"], v["counts"]["frame_hist_sha256"])
            for k, v in per.items() if "counts" in v}
    groups = collections.defaultdict(list)
    for k, v in keys.items():
        groups[v].append(k)
    ident = [sorted(g) for g in groups.values() if len(g) > 1]
    sc = collections.defaultdict(list)
    for k, v in keys.items():
        sc[v[:2]].append(k)
    same_counts = [sorted(g) for g in sc.values() if len(g) > 1 and len({keys[x] for x in g}) > 1]
    return {"identical_tracking_groups": ident, "same_spot_and_link_counts_different_histogram": same_counts}


def lead_verdicts(per, dup):
    dropped = {x for g in dup["identical_tracking_groups"] for x in g[1:]}
    out = {}
    w2 = [k for k in per if k.startswith("W2-") and k != "W2-F5"]
    ok = all("counts" in per[k] for k in w2)
    w2u = [k for k in w2 if k not in dropped and "counts" in per[k]]
    e2 = sum(per[k]["counts"]["eligible_pairs"] for k in w2u)
    out["W2"] = {"limbs_in_gate": sorted(w2u), "limbs_counted": len(w2u), "animals": "unknown",
                 "replication_line": f"{len(w2u)} limbs, animals unknown", "eligible_pairs": e2,
                 "expected_train_eligible": 0.4 * e2, "reader_validation_all_pass": ok,
                 "verdict": ("fail at L (reader validation)" if not ok else
                             "pass (limbs of unknown animals)" if 0.4 * e2 >= 30 else "fail (eligible pairs < 75)"),
                 "F5_reported_separately": per.get("W2-F5", {}).get("counts", {}).get("eligible_pairs")}
    e3 = per["W3-e1"].get("counts", {}).get("eligible_pairs", 0) if "counts" in per.get("W3-e1", {}) else None
    out["W3"] = {"embryos": 1, "eligible_pairs": e3, "expected_train_eligible": None if e3 is None else 0.4 * e3,
                 "verdict": ("fail at L (reader validation)" if e3 is None else
                             "pass, low replication (1 embryo)" if 0.4 * e3 >= 30 else "fail (eligible pairs < 75)")}
    return out


def run(out_path):
    paths, flog = fetch_all()
    per = {}
    for lab in ("W2-li13", "W2-li15", "W2-li48", "W2-li51", "W2-li52", "W2-F5", "W3-e1"):
        per[lab] = specimen(paths[lab])
        per[lab]["file"] = flog[lab]
    dup = duplicate_check(per)
    res = {"experiment": "DS-WIDE-G structure gate (W2, W3)", "min_arm_spots": MIN_ARM, "gate_eligible_pairs": GATE_ELIGIBLE,
           "reader_pinned_commit": mastodon_read.PINNED_COMMIT, "per_specimen": per, "duplicates": dup,
           "leads": lead_verdicts(per, dup),
           "note": "counts only; no ids, positions, tags, features or file names; same-frame links are counted and excluded"}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res["leads"], indent=1))
    for k, v in per.items():
        print(k, json.dumps(v.get("counts", v.get("read_error"))))
    return res


# ---------------------------------------------------------------- selftest
def _toy(chains):
    """chains: list of (name, start_frame, length, parent_name or None). Builds spots/links for a toy forest."""
    frames, links, last, first = [], [], {}, {}
    for name, st, ln, par in chains:
        ids = []
        for i in range(ln):
            frames.append(st + i)
            ids.append(len(frames) - 1)
            if i:
                links.append((ids[-2], ids[-1]))
        first[name], last[name] = ids[0], ids[-1]
        if par:
            links.append((last[par], ids[0]))
    return np.array(frames, np.int32), np.array(links, np.int32).reshape(-1, 2), first, last


def selftest():
    # R (6) divides: A (6) and B (6); A divides: C (6) and D (3, short arm); B divides: E (6), F (6); E divides: G (6), H (6).
    # Divisions R, A, B, E. R has depth 0 (not eligible). A: short arm D -> not eligible. B: eligible. E: eligible, nested in B.
    ch = [("R", 0, 6, None), ("A", 6, 6, "R"), ("B", 6, 6, "R"), ("C", 12, 6, "A"), ("D", 12, 3, "A"),
          ("E", 12, 6, "B"), ("F", 12, 6, "B"), ("G", 18, 6, "E"), ("H", 18, 6, "E")]
    fr, lk, _, _ = _toy(ch)
    c = gate_counts(fr, lk)
    assert (c["divisions"], c["eligible_pairs"], c["non_nested_eligible_pairs"], c["track_trees"],
            c["trees_with_eligible_pair"], c["isolated_spots"]) == (4, 2, 1, 1, 1, 0), c
    # second tree: a division at the root only (no depth) plus a lone spot and a same-frame link
    ch2 = ch + [("S", 0, 7, None), ("T", 7, 7, "S"), ("U", 7, 7, "S")]
    fr2, lk2, first, last = _toy(ch2)
    fr2 = np.concatenate([fr2, [3, 3]]).astype(np.int32)  # lone spot, then a spot that shares a frame with the last of 'R'
    n0 = len(fr2) - 2
    lk2 = np.vstack([lk2, [[last["R"] - 2, n0 + 1]]]).astype(np.int32)  # frame 3 -> frame 3: same-frame link to a new spot
    c2 = gate_counts(fr2, lk2)
    assert (c2["divisions"], c2["eligible_pairs"], c2["track_trees"], c2["trees_with_eligible_pair"],
            c2["isolated_spots"], c2["same_frame_links_excluded"]) == (5, 2, 2, 1, 2, 1), c2
    # link stored backwards (target earlier) is re-oriented, same result
    c2b = gate_counts(fr2, lk2[:, ::-1].copy())
    assert {k: v for k, v in c2b.items()} == {k: v for k, v in c2.items()}, (c2b, c2)
    # a merge: two short chains join into one longer chain; a division inside the merged branch
    ch3 = [("R", 0, 6, None), ("A", 6, 6, "R"), ("B", 6, 6, "R"), ("C", 12, 6, "A"), ("D", 12, 6, "A")]
    fr3, lk3, first3, last3 = _toy(ch3)
    lk3 = np.vstack([lk3, [[last3["B"], first3["C"]]]]).astype(np.int32)  # B also feeds C: merge at C's first spot
    c3 = gate_counts(fr3, lk3)
    assert c3["spots_with_gt1_incoming_link"] == 1 and c3["divisions"] == 2, c3
    # more than two forward links is not a division
    fr4 = np.array([0, 1, 1, 1], np.int32)
    c4 = gate_counts(fr4, np.array([[0, 1], [0, 2], [0, 3]], np.int32))
    assert c4["spots_with_gt2_forward_links"] == 1 and c4["divisions"] == 0, c4
    # zip member extraction from a local buffer: plain zip and a zip64-marked central directory
    payload = {"d1/PH-li13-T4_all-division.mastodon": mastodon_read.toy_project([0, 1, 2], [(0, 1), (1, 2)]),
               "d1/Suppl_li48_x_all-division.mastodon": mastodon_read.toy_project([0, 1], [(0, 1)]),
               "__MACOSX/d1/._PH-li13-T4_all-division.mastodon": b"x" * 176, "d1/other.xml": b"<a/>" * 500}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for k, v in payload.items():
            z.writestr(k, v)
    zbuf = buf.getvalue()
    for variant in ("plain", "zip64"):
        data = zbuf if variant == "plain" else _to_zip64(zbuf)
        src = BytesRange(data)
        members, total = central_directory(src)
        assert total == len(data) and len(members) == 4, (variant, len(members))
        sel = tracking_members(members, ("li13", "li48"))
        for lab, name in (("li13", "d1/PH-li13-T4_all-division.mastodon"), ("li48", "d1/Suppl_li48_x_all-division.mastodon")):
            got = extract_member(src, sel[lab])
            assert got == payload[name], (variant, lab)
        assert src.bytes < len(data) + 70000
        try:
            tracking_members(members, ("li13", "li99"))
            raise AssertionError("missing limb must fail")
        except RuntimeError:
            pass
        r = mastodon_read.read_mastodon(extract_member(src, sel["li13"]))
        assert gate_counts(r["frames"], r["links"])["links"] == 2
    print("ds_wide_gate selftest: ok")


def _to_zip64(zbuf):
    """Rewrite the end record as zip64 (locator + 64-bit end record, 0xFFFFFFFF markers); offsets stay valid."""
    i = zbuf.rfind(b"PK\x05\x06")
    _, _, _, n_ent, n_tot, cd_size, cd_off, clen = struct.unpack("<IHHHHIIH", zbuf[i:i + 22])
    e64_off = i
    e64 = struct.pack("<IQHHIIQQQQ", 0x06064B50, 44, 45, 45, 0, 0, n_ent, n_tot, cd_size, cd_off)
    loc = struct.pack("<IIQI", 0x07064B50, 0, e64_off, 1)
    eocd = struct.pack("<IHHHHIIH", 0x06054B50, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0)
    return zbuf[:i] + e64 + loc + eocd


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=str(HERE / "results_c17" / "ds_wide_gate_counts.json"))
    args = ap.parse_args()
    if args.selftest:
        mastodon_read.selftest()
        selftest()
    else:
        run(args.out)
