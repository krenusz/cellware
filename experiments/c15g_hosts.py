"""C14-S5 (c): structure-only read of C15 candidate #12's processed data (pre-registered in EXPERIMENT_LOG).

Downloads Zenodo record 5847461 (one tarball) on the runner, checks size and md5 against the API record, streams
the tar. Reads only member names/sizes and, from cell-metadata tables (csv/tsv/txt) and h5ad obs groups, the one
column holding unit IDs. Never reads expression, state labels, tree shapes or result tables; never loads pickles.
Output is recoded (arm1.., h01..): no raw sample names.

Run (GitHub Actions): python -I experiments/c15g_hosts.py
Smoke only: python -I experiments/c15g_hosts.py --synthetic OUTDIR
"""
from __future__ import annotations

import collections
import csv
import hashlib
import io
import json
import re
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "results_c15"
RECORD = "https://zenodo.org/api/records/5847461"
UNIT_RE = re.compile(r"(?<![0-9])(\d{3,5})_([A-Za-z0-9]+?)_(T\d+)")
META_EXT = (".csv", ".tsv", ".txt", ".csv.gz", ".tsv.gz", ".txt.gz")
META_MAX = 200e6
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def ext_of(name):
    n = name.lower()
    for e in (".csv.gz", ".tsv.gz", ".txt.gz", ".tar.gz"):
        if n.endswith(e):
            return e
    return Path(n).suffix or "(none)"


def units_in(values):
    c = collections.Counter()
    for v in values:
        m = UNIT_RE.search(str(v))
        if m:
            c["_".join(m.groups())] += 1
    return c


def h5py_mod():
    try:
        import h5py
    except ImportError:
        subprocess.run(["uv", "pip", "install", "h5py"], check=True)
        import h5py
    return h5py


def obs_unit_counts(path):
    """Cells per unit from an h5ad obs group: only the column whose values match the unit pattern."""
    h5py = h5py_mod()
    best = collections.Counter()
    with h5py.File(path, "r") as f:
        if "obs" not in f:
            return best
        obs = f["obs"]
        cols = []
        if isinstance(obs, h5py.Dataset):  # old compound layout
            for name in obs.dtype.names or []:
                cols.append([x.decode() if isinstance(x, bytes) else x for x in obs[name][:]])
        else:
            for name, item in obs.items():
                if isinstance(item, h5py.Group) and "categories" in item and "codes" in item:
                    cats = [x.decode() if isinstance(x, bytes) else str(x) for x in item["categories"][:]]
                    if not any(UNIT_RE.search(c) for c in cats[:50]):
                        continue
                    codes = item["codes"][:]
                    cols.append([cats[i] for i in codes if i >= 0])
                elif isinstance(item, h5py.Dataset) and item.dtype.kind in "OSU" and item.ndim == 1:
                    cats_grp = obs.get("__categories")
                    if cats_grp is not None and name in cats_grp:
                        cats = [x.decode() if isinstance(x, bytes) else str(x) for x in cats_grp[name][:]]
                        cols.append([cats[i] for i in item[:]])
                    else:
                        head = [x.decode() if isinstance(x, bytes) else str(x) for x in item[:50]]
                        if any(UNIT_RE.search(h) for h in head):
                            cols.append([x.decode() if isinstance(x, bytes) else str(x) for x in item[:]])
                elif isinstance(item, h5py.Dataset) and item.ndim == 1 and name != "_index":
                    cats_grp = obs.get("__categories")
                    if cats_grp is not None and name in cats_grp:
                        cats = [x.decode() if isinstance(x, bytes) else str(x) for x in cats_grp[name][:]]
                        if any(UNIT_RE.search(c) for c in cats[:50]):
                            cols.append([cats[i] for i in item[:] if i >= 0])
        for col in cols:
            c = units_in(col)
            if sum(c.values()) > sum(best.values()):
                best = c
    return best


def table_unit_counts(fobj, name):
    raw = fobj.read()
    if name.lower().endswith(".gz"):
        import gzip
        raw = gzip.decompress(raw)
    text = raw.decode("utf-8", "replace")
    delim = "\t" if "\t" in text[:2000] else ","
    rows = list(csv.reader(io.StringIO(text), delimiter=delim))
    if len(rows) < 2:
        return collections.Counter()
    best = collections.Counter()
    for j in range(len(rows[0])):
        c = units_in(r[j] for r in rows[1:] if len(r) > j)
        if sum(c.values()) > sum(best.values()):
            best = c
    return best


def scan(tar_path, workdir):
    members, exts, sizes = 0, collections.Counter(), collections.Counter()
    name_units = collections.defaultdict(set)  # unit -> member kinds naming it
    tables = []
    with tarfile.open(tar_path, "r|gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            members += 1
            e = ext_of(m.name)
            exts[e] += 1
            sizes[e] += m.size
            for u in units_in([m.name]):
                name_units[u].add(e)
            low = m.name.lower()
            if e == ".h5ad" or (low.endswith(META_EXT) and m.size < META_MAX):
                f = tf.extractfile(m)
                if e == ".h5ad":
                    p = Path(workdir) / f"obj{members}.h5ad"
                    with open(p, "wb") as out:
                        while chunk := f.read(1 << 24):
                            out.write(chunk)
                    c = obs_unit_counts(p)
                    p.unlink()
                else:
                    c = table_unit_counts(f, m.name)
                if c:
                    tables.append((e, c))
    return members, exts, sizes, name_units, tables


def summarise(members, exts, sizes, name_units, tables):
    cell_counts = max((c for _, c in tables), key=lambda c: sum(c.values()), default=collections.Counter())
    units = set(name_units) | set(cell_counts)
    arms = sorted({u.split("_")[1] for u in units})
    hosts = sorted({u.split("_")[0] for u in units})
    arm_code = {a: f"arm{i + 1}" for i, a in enumerate(arms)}
    host_code = {h: f"h{i + 1:02d}" for i, h in enumerate(hosts)}
    per_host = collections.defaultdict(lambda: collections.Counter())
    for u in units:
        h, a, _ = u.split("_")
        per_host[host_code[h]][arm_code[a]] += 1
    per_arm = collections.defaultdict(lambda: {"hosts": set(), "units": 0})
    for h, c in per_host.items():
        for a, n in c.items():
            per_arm[a]["hosts"].add(h)
            per_arm[a]["units"] += n
    cells = sorted(cell_counts.values())
    q = lambda p: cells[int(p * (len(cells) - 1))] if cells else None  # noqa: E731
    return dict(
        members=members, by_extension={e: dict(files=exts[e], mb=round(sizes[e] / 1e6, 1)) for e in exts},
        tables_with_unit_ids=len(tables),
        n_units=len(units), n_units_named_in_files=len(name_units), n_units_with_cells=len(cell_counts),
        n_hosts=len(hosts), n_arms=len(arms),
        units_per_host=dict(sorted(collections.Counter(sum(c.values()) for c in per_host.values()).items())),
        hosts_with_more_than_one_arm=sum(len(c) > 1 for c in per_host.values()),
        per_arm={a: dict(hosts=len(v["hosts"]), units=v["units"]) for a, v in sorted(per_arm.items())},
        cells_per_unit=dict(min=q(0), q25=q(0.25), median=q(0.5), q75=q(0.75), max=q(1),
                            units_ge_100=sum(x >= 100 for x in cells), total=sum(cells)),
        per_host=[dict(host=h, **dict(c)) for h, c in sorted(per_host.items())],
    )


def download(dest):
    rec = json.loads(urllib.request.urlopen(RECORD, timeout=60).read())
    f = rec["files"][0] if isinstance(rec["files"], list) else rec["files"]["entries"][0]
    url = f["links"]["self"] if "links" in f else f"{RECORD}/files/{f['key']}/content"
    size, md5 = f["size"], f["checksum"].split(":")[-1]
    say(f"record file: {size / 1e6:.1f} MB")
    h = hashlib.md5()
    with urllib.request.urlopen(url, timeout=600) as r, open(dest, "wb") as out:
        while chunk := r.read(1 << 24):
            h.update(chunk)
            out.write(chunk)
    assert Path(dest).stat().st_size == size, "size mismatch"
    assert h.hexdigest() == md5, "md5 mismatch"
    say("size and md5 match the Zenodo record")


def run(tar_path, outdir):
    with tempfile.TemporaryDirectory() as wd:
        res = summarise(*scan(tar_path, wd))
    say(f"members {res['members']}; units {res['n_units']} (named in files {res['n_units_named_in_files']}, "
        f"with cell counts {res['n_units_with_cells']}); hosts {res['n_hosts']}; arms {res['n_arms']}")
    say(f"units per host {res['units_per_host']}; hosts with >1 arm {res['hosts_with_more_than_one_arm']}")
    say(f"per arm {res['per_arm']}; cells per unit {res['cells_per_unit']}")
    Path(outdir).mkdir(parents=True, exist_ok=True)
    (Path(outdir) / "c15g_hosts.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (Path(outdir) / "c15g_hosts.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    return res


def synthetic(outdir):
    with tempfile.TemporaryDirectory() as wd:
        wd = Path(wd)
        src = wd / "src"
        src.mkdir()
        units = ["1001_AA_T1", "1001_AA_T2", "1002_AA_T1", "1003_BB_T1", "1003_BB_T4", "1004_CC_T2"]
        for u in units:
            (src / f"{u}_tree.nwk").write_text("(a,b);")
        with open(src / "cell_meta.tsv", "w") as f:
            f.write("cell\tsample\tother\n")
            for i, u in enumerate(units):
                for j in range(100 + 50 * i):
                    f.write(f"c{i}_{j}\t{u}\tx\n")
        (src / "model.pkl").write_bytes(b"not loaded")
        tar = wd / "t.tar.gz"
        with tarfile.open(tar, "w:gz") as tf:
            tf.add(src, arcname="data")
        res = run(tar, outdir)
    assert res["n_units"] == 6 and res["n_hosts"] == 4 and res["n_arms"] == 3, res
    assert res["cells_per_unit"]["min"] == 100 and res["hosts_with_more_than_one_arm"] == 0
    assert "AA" not in json.dumps(res) and "1001" not in json.dumps(res), "raw names leaked"
    print("SYNTHETIC OK")


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--synthetic":
        return synthetic(sys.argv[2])
    dl = HERE.parent / "data" / "ext_c15" / "raw"
    dl.mkdir(parents=True, exist_ok=True)
    tar = dl / "record_5847461.tar.gz"
    download(tar)
    run(tar, OUT)


if __name__ == "__main__":
    main()
