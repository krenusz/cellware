"""C15-S0: structural counts of candidate #12 per host partition (pre-registered in EXPERIMENT_LOG).

Partition growth12_host (partitions.py) is applied to the host number of every unit ID before anything is
counted. Reads only: unit IDs in h5ad obs (cells per unit, per h5ad), comma counts of .nwk files (leaves per
unit), and the NUMBER of categories of an obs column named like leiden/cluster/state. Never expression, state
composition, tree topology or results. Output recoded; per-unit sizes as sorted lists without IDs.

Run (GitHub Actions): python -I experiments/c15s0_struct.py
Smoke only: python -I experiments/c15s0_struct.py --synthetic OUTDIR
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

DS = "growth12_host"
OUT = HERE / "results_c15"
STATE_RE = re.compile(r"leiden|cluster|state", re.I)
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def n_state_categories(path):
    """Largest category count among obs columns named like leiden/cluster/state (count only)."""
    h5py = g.h5py_mod()
    best = 0
    with h5py.File(path, "r") as f:
        obs = f.get("obs")
        if obs is None or isinstance(obs, h5py.Dataset):
            return best
        cg = obs.get("__categories")
        for name, item in obs.items():
            if not STATE_RE.search(name):
                continue
            if isinstance(item, h5py.Group) and "categories" in item:
                best = max(best, len(item["categories"]))
            elif cg is not None and name in cg:
                best = max(best, len(cg[name]))
    return best


def scan(tar_path, wd):
    cells = {}  # h5ad member index -> Counter(unit -> cells)
    leaves = collections.Counter()
    n_states = 0
    k = 0
    with tarfile.open(tar_path, "r|gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            low = m.name.lower()
            if low.endswith(".h5ad"):
                k += 1
                p = Path(wd) / f"obj{k}.h5ad"
                f = tf.extractfile(m)
                with open(p, "wb") as out:
                    while chunk := f.read(1 << 24):
                        out.write(chunk)
                cells[f"h5ad{k}"] = g.obs_unit_counts(p)
                n_states = max(n_states, n_state_categories(p))
                p.unlink()
            elif low.endswith(".nwk"):
                us = list(g.units_in([m.name]))
                if len(us) == 1:
                    txt = tf.extractfile(m).read().decode("utf-8", "replace")
                    leaves[us[0]] = max(leaves[us[0]], txt.count(",") + 1)
    return cells, leaves, n_states


def summarise(cells, leaves, n_states):
    units = set(leaves).union(*[set(c) for c in cells.values()]) if cells else set(leaves)
    arms = sorted({u.split("_")[1] for u in units})
    arm_code = {a: f"arm{i + 1}" for i, a in enumerate(arms)}
    main = max(cells, key=lambda k: len(cells[k]), default=None)
    parts = {}
    for p in ["train", "A", "B", "C"]:
        us = sorted(u for u in units if partition_of(DS, u.split("_")[0]) == p)
        hosts = collections.defaultdict(list)
        for u in us:
            hosts[u.split("_")[0]].append(u)
        host_leaves = sorted(sum(leaves.get(u, 0) for u in hu) for hu in hosts.values())
        host_cells = sorted(sum(cells[main].get(u, 0) for u in hu) for hu in hosts.values()) if main else []
        parts[p] = dict(
            hosts=len(hosts), units=len(us),
            units_per_arm=dict(sorted(collections.Counter(arm_code[u.split("_")[1]] for u in us).items())),
            leaves_per_unit=sorted(leaves.get(u, 0) for u in us),
            cells_per_unit=sorted(cells[main].get(u, 0) for u in us) if main else [],
            leaves_per_host=host_leaves, cells_per_host=host_cells,
            hosts_with_unit_ge_200_leaves=sum(any(leaves.get(u, 0) >= 200 for u in hu) for hu in hosts.values()),
        )
    return dict(dataset=DS, n_units=len(units), n_arms=len(arms), n_state_clusters=n_states,
                cells_source=f"{main} obs unit-ID column (largest of {len(cells)} h5ad)",
                units_per_h5ad={k: len(v) for k, v in cells.items()},
                cells_per_h5ad={k: sum(v.values()) for k, v in cells.items()},
                leaves_source=".nwk comma count", units_with_tree=len(leaves), partitions=parts)


def run(tar, outdir):
    with tempfile.TemporaryDirectory() as wd:
        res = summarise(*scan(tar, wd))
    say(f"units {res['n_units']}, arms {res['n_arms']}, state clusters {res['n_state_clusters']}; "
        f"cells per h5ad {res['cells_per_h5ad']}; units with tree {res['units_with_tree']}")
    for p, v in res["partitions"].items():
        lv = v["leaves_per_unit"]
        say(f"{p}: hosts {v['hosts']}, units {v['units']} {v['units_per_arm']}, hosts with a unit >= 200 leaves "
            f"{v['hosts_with_unit_ge_200_leaves']}, leaves per unit median {lv[len(lv) // 2] if lv else 0}, "
            f"leaves per host {v['leaves_per_host']}")
    Path(outdir).mkdir(parents=True, exist_ok=True)
    (Path(outdir) / "c15s0_struct.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (Path(outdir) / "c15s0_struct.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    return res


def synthetic(outdir):
    import numpy as np
    h5py = g.h5py_mod()
    with tempfile.TemporaryDirectory() as wd:
        wd = Path(wd)
        src = wd / "src"
        src.mkdir()
        units = [f"{h}_{a}_T{t}" for h, a in [(1001, "AA"), (1002, "AA"), (1003, "BB"), (1004, "BB"), (1005, "CC"),
                                             (1006, "CC"), (1007, "AA"), (1008, "BB")] for t in (1, 2)]
        for i, u in enumerate(units):
            (src / f"{u}_tree.nwk").write_text("(" + ",".join(f"c{j}" for j in range(150 + 10 * i)) + ");")
        with h5py.File(src / "x.h5ad", "w") as f:
            o = f.create_group("obs")
            s = o.create_group("sample")
            s["categories"] = np.array([u.encode() for u in units])
            s["codes"] = np.repeat(np.arange(len(units)), 120).astype("int16")
            c = o.create_group("leiden_x")
            c["categories"] = np.array([b"0", b"1", b"2", b"3"])
            c["codes"] = np.zeros(120 * len(units), "int8")
        tar = wd / "t.tar.gz"
        with tarfile.open(tar, "w:gz") as tf:
            tf.add(src, arcname="data")
        res = run(tar, outdir)
    assert res["n_units"] == 16 and res["n_state_clusters"] == 4, res
    assert sum(v["hosts"] for v in res["partitions"].values()) == 8
    assert sum(v["units"] for v in res["partitions"].values()) == 16
    s = json.dumps(res)
    assert "AA" not in s and "1001" not in s, "raw names leaked"
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
