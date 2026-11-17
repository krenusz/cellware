"""C15-S1a: H9 format check of candidate #12 on TRAIN hosts only (pre-registered in EXPERIMENT_LOG).

Reports structure: h5ad layout (obs columns, dtypes, category counts), X shape/dtype/integer-ness, var columns,
fraction of train tree leaves found in each h5ad's obs index (under a few name transforms), .nwk vs .tree, and
file-kind counts. Names are sanitised (sensitive words masked) before printing or saving; values never saved.
Sealed hosts: their trees are skipped by unit ID before reading; obs rows are not counted.

Run (GitHub Actions): python -I experiments/c15s1a_format.py
Smoke only: python -I experiments/c15s1a_format.py --synthetic OUTDIR
"""
from __future__ import annotations

import collections
import json
import re
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c15g_hosts as g  # noqa: E402
from partitions import partition_of  # noqa: E402

DS = "growth12_host_v2"
OUT = HERE / "results_c15"
MASK = re.compile(r"tumou?r|cancer|metast|kras|lkb1|stk11|\bapc\b|p53|trp53|lung|adeno|onc|malig|carcin", re.I)
LEAF_RE = re.compile(r"[(,]\s*([^(),:;\s]+)")
LOG = []


def clean(s):
    return MASK.sub("X", str(s))


def say(*a):
    s = clean(" ".join(str(x) for x in a))
    print(s, flush=True)
    LOG.append(s)


def is_train_unit(u):
    return partition_of(DS, u.split("_")[0]) == "train"


def dec(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def h5_layout(path, train_leaves):
    h5py = g.h5py_mod()
    import numpy as np
    info = {}
    with h5py.File(path, "r") as f:
        info["top"] = sorted(clean(k) for k in f.keys())
        obs = f["obs"]
        idx_key = obs.attrs.get("_index", "_index")
        idx_key = dec(idx_key)
        index = [dec(x) for x in obs[idx_key][:]] if idx_key in obs else []
        info["n_obs"] = len(index)
        cols = []
        cg = obs.get("__categories")
        for name, item in obs.items():
            if name in (idx_key, "__categories"):
                continue
            if isinstance(item, h5py.Group) and "categories" in item:
                cols.append((clean(name), "categorical", len(item["categories"])))
            elif cg is not None and name in cg:
                cols.append((clean(name), "categorical(old)", len(cg[name])))
            else:
                cols.append((clean(name), str(item.dtype), None))
        info["obs_columns"] = cols
        X = f["X"]
        if isinstance(X, h5py.Group):
            info["X"] = dict(encoding=dec(X.attrs.get("encoding-type", "sparse")), shape=[int(v) for v in X.attrs.get("shape", [])],
                             dtype=str(X["data"].dtype), nnz=int(X["data"].shape[0]))
            sample = X["data"][: min(20000, X["data"].shape[0])]
        else:
            info["X"] = dict(encoding="dense", shape=list(X.shape), dtype=str(X.dtype))
            sample = X[: min(200, X.shape[0])].ravel()
        info["X"]["integer_valued_sample"] = bool(np.all(np.mod(sample, 1) == 0)) if len(sample) else None
        var = f["var"]
        info["var_columns"] = sorted(clean(k) for k in var.keys())
        info["layers"] = sorted(clean(k) for k in f["layers"].keys()) if "layers" in f else []
        info["obsm"] = sorted(clean(k) for k in f["obsm"].keys()) if "obsm" in f else []
        idx = set(index)
        tf = {"exact": lambda s: s, "strip -1": lambda s: s.rsplit("-", 1)[0], "add -1": lambda s: s + "-1"}
        idx_t = {k: {fn(x) for x in idx} for k, fn in tf.items()}
        leaves = [lf for lv in train_leaves.values() for lf in lv]
        info["leaf_match"] = {k: round(sum(fn(lf) in idx_t["exact"] or lf in idx_t[k] for lf in leaves) / max(len(leaves), 1), 3)
                              for k, fn in tf.items()}
        info["example_index"] = clean(index[0]) if index else None
        info["example_leaf"] = clean(leaves[0]) if leaves else None
    return info


def scan(tar_path, wd):
    kinds = collections.Counter()
    train_leaves, nwk_vs_tree, h5 = {}, {}, []
    raw_tree = {}
    with tarfile.open(tar_path, "r|gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            name = m.name
            us = list(g.units_in([name]))
            pattern = g.UNIT_RE.sub("<unit>", Path(name).name)
            kinds[clean(re.sub(r"\d+", "#", pattern))] += 1
            low = name.lower()
            if low.endswith(".h5ad"):
                p = Path(wd) / f"obj{len(h5) + 1}.h5ad"
                f = tf.extractfile(m)
                with open(p, "wb") as out:
                    while chunk := f.read(1 << 24):
                        out.write(chunk)
                h5.append(p)
            elif us and len(us) == 1 and is_train_unit(us[0]) and low.endswith((".nwk", ".tree")):
                txt = tf.extractfile(m).read().decode("utf-8", "replace")
                if low.endswith(".nwk"):
                    train_leaves[us[0]] = LEAF_RE.findall(txt)
                raw_tree.setdefault(us[0], {})[Path(low).suffix] = txt
    for u, d in raw_tree.items():
        if ".nwk" in d and ".tree" in d:
            nwk_vs_tree = dict(identical=d[".nwk"].strip() == d[".tree"].strip(), nwk_len=len(d[".nwk"]),
                               tree_len=len(d[".tree"]), tree_head=clean(d[".tree"][:60]))
            break
    layouts = [h5_layout(p, train_leaves) for p in h5]
    return dict(file_kinds=dict(kinds.most_common(40)), train_units_with_nwk=len(train_leaves),
                train_leaves=sum(len(v) for v in train_leaves.values()), nwk_vs_tree=nwk_vs_tree, h5ad=layouts)


def run(tar, outdir):
    with tempfile.TemporaryDirectory() as wd:
        res = scan(tar, wd)
    say(f"train units with nwk {res['train_units_with_nwk']}, train leaves {res['train_leaves']}")
    say("nwk vs tree:", res["nwk_vs_tree"])
    for i, h in enumerate(res["h5ad"]):
        say(f"h5ad{i + 1}: n_obs {h['n_obs']}, X {h['X']}, leaf match {h['leaf_match']}")
        say(f"  top {h['top']}; layers {h['layers']}; obsm {h['obsm']}; var {h['var_columns']}")
        say(f"  obs columns {h['obs_columns']}")
        say(f"  example index {h['example_index']} / leaf {h['example_leaf']}")
    say("file kinds:", res["file_kinds"])
    Path(outdir).mkdir(parents=True, exist_ok=True)
    (Path(outdir) / "c15s1a_format.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (Path(outdir) / "c15s1a_format.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    return res


def synthetic(outdir):
    import numpy as np
    h5py = g.h5py_mod()
    global is_train_unit
    is_train_unit = lambda u: True  # noqa: E731
    with tempfile.TemporaryDirectory() as wd:
        wd = Path(wd)
        src = wd / "src"
        src.mkdir()
        cells = []
        for u in ["2001_AA_T1", "2002_BB_T1"]:
            names = [f"{u}_c{j}" for j in range(20)]
            cells += names
            nwk = "((" + ",".join(names[:10]) + "),(" + ",".join(names[10:]) + "));"
            (src / f"{u}_tree.nwk").write_text(nwk)
            (src / f"{u}_tree.tree").write_text(nwk)
        with h5py.File(src / "x.h5ad", "w") as f:
            o = f.create_group("obs")
            o.attrs["_index"] = "_index"
            o["_index"] = np.array([c.encode() for c in cells])
            c = o.create_group("leiden_tumor_state")
            c["categories"] = np.array([b"0", b"1"])
            c["codes"] = np.zeros(len(cells), "int8")
            X = f.create_group("X")
            X.attrs["encoding-type"] = "csr_matrix"
            X.attrs["shape"] = [len(cells), 5]
            X["data"] = np.ones(10, "float32")
            X["indices"] = np.arange(10) % 5
            X["indptr"] = np.r_[np.arange(0, 10), np.full(len(cells) - 9, 10)]
            f.create_group("var")["gene_ids"] = np.array([b"g"] * 5)
        tar = wd / "t.tar.gz"
        with tarfile.open(tar, "w:gz") as tf:
            tf.add(src, arcname="data")
        res = run(tar, outdir)
    h = res["h5ad"][0]
    assert h["leaf_match"]["exact"] == 1.0 and h["X"]["integer_valued_sample"] and res["nwk_vs_tree"]["identical"]
    assert "tumor" not in json.dumps(res).lower(), "mask failed"
    print("SYNTHETIC OK")


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--synthetic":
        return synthetic(sys.argv[2])
    dl = HERE.parent / "data" / "ext_c15" / "raw"
    dl.mkdir(parents=True, exist_ok=True)
    tar = dl / "record_5847461.tar.gz"
    g.download(tar)
    run(tar, OUT)


if __name__ == "__main__":
    main()
