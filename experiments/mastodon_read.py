"""Minimal reader for Mastodon tracking projects (.mastodon = zip of project.xml and model.raw).

Returns only spots (row id, frame) and links (source row, target row). Tag sets and feature columns are
never opened or decoded; spot positions and covariances are skipped by offset, not stored.

Binary layout learned by READING (not installing, importing or running) the format developers' importer,
github.com/mastodon-sc/python-mastodon-importer, file mastodon_reader/mastodon_reader.py, at the pinned commit
87eb001fb36e1850203c30fc155928b4e21acd6a (class JavaRawReader and MastodonReader.read_tables):
  model.raw = Java object stream, big-endian header: short -21267 (magic), short 5 (version), then data blocks.
  Block = key byte 119 + 1-byte length, or key byte 122 + 4-byte big-endian length; the payload of all blocks is
  one continuous byte stream (reads may span blocks).
  Stream: int32 BE n_spots; n_spots records of 84 bytes: 3 doubles (x y z) LE, int32 LE frame, 7 doubles LE
  (covariance terms, bsrs); int32 BE n_links; n_links records of 4 x int32 BE: source row, target row,
  and two further ints (edge slots, ignored here).
  A spot's id is its row index in the spot table; link endpoints are row indices.
Departures from the importer (recorded): the 1-byte block length is read UNSIGNED (the importer reads it signed,
which fails for lengths 128-255); data after the edge table is not decoded, only its byte count is reported.

    python -I experiments/mastodon_read.py --selftest
"""

from __future__ import annotations

import io
import struct
import sys
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

PINNED_COMMIT = "87eb001fb36e1850203c30fc155928b4e21acd6a"
MAGIC, VERSION = -21267, 5
SPOT_DTYPE = np.dtype([("xyz", "<f8", (3,)), ("t", "<i4"), ("cov", "<f8", (7,))])
assert SPOT_DTYPE.itemsize == 84


class FormatError(Exception):
    """Carries structural detail only (sizes and counts), never content."""


def _blocks(raw: bytes):
    if len(raw) < 4:
        raise FormatError(f"model.raw too short: {len(raw)} bytes")
    magic, ver = struct.unpack(">hh", raw[:4])
    if magic != MAGIC or ver != VERSION:
        raise FormatError(f"header magic/version {magic}/{ver}, expected {MAGIC}/{VERSION}")
    pos, parts, nblocks, end_key = 4, [], 0, None
    while pos < len(raw):
        key = raw[pos]
        if key == 119:
            n, hdr = raw[pos + 1], 2
        elif key == 122:
            n, hdr = struct.unpack(">i", raw[pos + 1:pos + 5])[0], 5
        else:
            end_key = key
            break
        if n < 0 or pos + hdr + n > len(raw):
            raise FormatError(f"block {nblocks} overruns file: length {n}, available {len(raw) - pos - hdr}")
        parts.append(raw[pos + hdr:pos + hdr + n])
        pos += hdr + n
        nblocks += 1
    return b"".join(parts), {"blocks": nblocks, "bytes_after_blocks": len(raw) - pos, "end_key": end_key}


def read_mastodon(src):
    """src: path, bytes or file object. Returns dict with frames (int32, index = spot id), links (int32 Nx2) and diag."""
    fh = io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src
    with zipfile.ZipFile(fh) as z:
        names = set(z.namelist())
        if "model.raw" not in names or "project.xml" not in names:
            raise FormatError(f"members model.raw/project.xml present: {'model.raw' in names}/{'project.xml' in names}")
        version = ET.fromstring(z.read("project.xml")).attrib.get("version")
        raw = z.read("model.raw")
    data, diag = _blocks(raw)
    diag["project_version"] = version
    diag["stream_bytes"] = len(data)
    if len(data) < 4:
        raise FormatError(f"stream too short: {len(data)}; {diag}")
    n_spots = struct.unpack(">i", data[:4])[0]
    need = 4 + 84 * n_spots + 4
    if n_spots < 0 or need > len(data):
        raise FormatError(f"n_spots {n_spots}: need {need} bytes, have {len(data)}; {diag}")
    spots = np.frombuffer(data, dtype=SPOT_DTYPE, count=n_spots, offset=4)
    n_links = struct.unpack(">i", data[need - 4:need])[0]
    need_end = need + 16 * n_links
    if n_links < 0 or need_end > len(data):
        raise FormatError(f"n_links {n_links}: need {need_end} bytes, have {len(data)}; n_spots {n_spots}; {diag}")
    links = np.frombuffer(data, dtype=">i4", count=4 * n_links, offset=need).reshape(n_links, 4)[:, :2]
    diag.update(n_spots=n_spots, n_links=n_links, leftover_stream_bytes=len(data) - need_end)
    return {"frames": spots["t"].astype(np.int32), "links": links.astype(np.int32), "diag": diag}


def validate(rec):
    """Structural validation. Cycles are tested on the links as stored (before any re-orientation)."""
    fr, lk, d = rec["frames"], rec["links"], rec["diag"]
    n, m = len(fr), len(lk)
    out = {"n_spots": n, "n_links": m, "project_version": d["project_version"],
           "stored_counts_fit_stream": True,  # the two stored counts are read from the stream and consumed in full
           "leftover_stream_bytes": d["leftover_stream_bytes"], "bytes_after_blocks": d["bytes_after_blocks"]}
    out["frames_min"] = int(fr.min()) if n else None
    out["frames_max"] = int(fr.max()) if n else None
    out["frames_nonnegative"] = bool(n == 0 or fr.min() >= 0)
    inr = bool(m == 0 or (lk.min() >= 0 and lk.max() < n))
    out["endpoints_in_range"] = inr
    if not inr:
        out.update(same_frame_links=None, reversed_links=None, cycle_free_as_stored=None, links_join_different_frames=None,
                   links_forward_as_stored=None, valid=False)
        return out
    fs, ft = fr[lk[:, 0]], fr[lk[:, 1]]
    same, rev = int((fs == ft).sum()), int((fs > ft).sum())
    out["same_frame_links"], out["reversed_links"] = same, rev
    out["self_links"] = int((lk[:, 0] == lk[:, 1]).sum())
    out["links_join_different_frames"] = same == 0  # task wording / amendment 1 (strict flag)
    out["links_forward_as_stored"] = same == 0 and rev == 0  # amendment 1 wording "to a later frame" (strict flag)
    out["cycle_free_as_stored"] = _acyclic(n, lk)
    # core validity: layout parsed, indices and frames sane, no cycle as stored. Same-frame and reversed links are
    # reported, not decided here (amendment 2 counts and excludes them).
    # DS-WIDE-G amendment 3: same-frame links above 1% of links suggest a misread layout -> invalid.
    out["same_frame_within_1pct"] = bool(same <= 0.01 * m)
    out["valid"] = bool(
        out["frames_nonnegative"] and out["cycle_free_as_stored"] and out["self_links"] == 0 and out["same_frame_within_1pct"]
    )
    return out


def _acyclic(n, lk):
    """Kahn's algorithm on the stored directed graph."""
    if len(lk) == 0:
        return True
    src, dst = lk[:, 0], lk[:, 1]
    indeg = np.bincount(dst, minlength=n).astype(np.int64)
    order = np.argsort(src, kind="stable")
    start = np.concatenate([[0], np.cumsum(np.bincount(src, minlength=n))])
    dsts = dst[order]
    stack = list(np.flatnonzero(indeg == 0))
    seen = 0
    while stack:
        v = stack.pop()
        seen += 1
        for w in dsts[start[v]:start[v + 1]]:
            indeg[w] -= 1
            if indeg[w] == 0:
                stack.append(int(w))
    return seen == n


# ---- toy writer for the selftest (mirrors the layout above) ----
def toy_project(frames, links, version="0.3", block=100):
    body = struct.pack(">i", len(frames))
    for t in frames:
        body += struct.pack("<ddd", 1.0, 2.0, 3.0) + struct.pack("<i", t) + struct.pack("<7d", *range(7))
    body += struct.pack(">i", len(links))
    for a, b in links:
        body += struct.pack(">iiii", a, b, 0, 0)
    raw = struct.pack(">hh", MAGIC, VERSION)
    for i in range(0, len(body), block):
        chunk = body[i:i + block]
        raw += (bytes([119, len(chunk)]) if len(chunk) < 256 else b"z" + struct.pack(">i", len(chunk))) + chunk
    raw += bytes([120])  # end-of-block-data marker
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("project.xml", f'<?xml version="1.0"?><MamutProject version="{version}"/>')
        z.writestr("model.raw", raw)
        z.writestr("tags.raw", b"not decoded")
    return buf.getvalue()


def selftest():
    frames = [0, 1, 2, 2, 3, 1]
    links = [(0, 1), (1, 2), (1, 3), (2, 4)]
    for block in (7, 200, 254, 255, 100000):  # 128-255 exercises the unsigned length; 100000 a single 122 block
        r = read_mastodon(toy_project(frames, links, block=block))
        assert r["frames"].tolist() == frames and r["links"].tolist() == [list(x) for x in links], block
        v = validate(r)
        assert v["valid"] and v["same_frame_links"] == 0 and v["reversed_links"] == 0 and v["cycle_free_as_stored"], v
        assert v["leftover_stream_bytes"] == 0 and v["bytes_after_blocks"] == 1 and v["project_version"] == "0.3", v
    # same-frame link, reversed link, cycle as stored
    v = validate(read_mastodon(toy_project([0, 1, 1], [(0, 1), (1, 2), (2, 0)])))
    assert v["same_frame_links"] == 1 and v["reversed_links"] == 1 and not v["cycle_free_as_stored"] and not v["valid"], v
    v = validate(read_mastodon(toy_project([0, 1, 2], [(1, 0)])))
    assert v["reversed_links"] == 1 and v["cycle_free_as_stored"] and not v["links_forward_as_stored"], v
    # endpoint out of range and truncated stream
    v = validate(read_mastodon(toy_project([0, 1], [(0, 5)])))
    assert not v["endpoints_in_range"] and not v["valid"], v
    good = toy_project([0, 1, 2], [(0, 1)])
    with zipfile.ZipFile(io.BytesIO(good)) as z:
        raw = z.read("model.raw")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("project.xml", "<a version='0.3'/>")
        z.writestr("model.raw", raw[:30])
    try:
        read_mastodon(buf.getvalue())
        raise AssertionError("truncated stream must fail")
    except FormatError:
        pass
    print("mastodon_read selftest: ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        print(__doc__)
