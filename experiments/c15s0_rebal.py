"""C15-S0 amendment 1: rebalance the sealed hosts of growth12_host to 5/5/5 (pre-registered in EXPERIMENT_LOG).

Train hosts (v1) unchanged. v1-sealed hosts sorted by total tree leaves (desc), ties by sha256(salt v2 + host),
dealt A B C C B A ... Output: host number -> partition map (no arm names) and per-partition structure in the
c15s0_struct layout, so c15s0_power can read it. Structure only.

Run (GitHub Actions): python -I experiments/c15s0_rebal.py
Smoke only: python -I experiments/c15s0_rebal.py --synthetic
"""
from __future__ import annotations

import collections
import hashlib
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c15g_hosts as g  # noqa: E402
import c15s0_struct as st  # noqa: E402
from partitions import partition_of  # noqa: E402

V1 = "growth12_host"
SALT2 = "cellware-growth12-host-v2"
SNAKE = "ABCCBA"
OUT = HERE / "results_c15"


def assign(host_leaves, v1_of):
    """host_leaves: {host: total leaves}; v1_of: host -> v1 partition. Returns {host: v2 partition}."""
    out = {h: "train" for h in host_leaves if v1_of(h) == "train"}
    sealed = [h for h in host_leaves if v1_of(h) != "train"]
    key = lambda h: (-host_leaves[h], hashlib.sha256(f"{SALT2}:{h}".encode()).hexdigest())  # noqa: E731
    for i, h in enumerate(sorted(sealed, key=key)):
        out[h] = SNAKE[i % len(SNAKE)]
    return out


def structure(units, leaves, cells_main, mapping):
    arms = sorted({u.split("_")[1] for u in units})
    arm_code = {a: f"arm{i + 1}" for i, a in enumerate(arms)}
    parts = {}
    for p in ["train", "A", "B", "C"]:
        us = sorted(u for u in units if mapping[u.split("_")[0]] == p)
        hosts = collections.defaultdict(list)
        for u in us:
            hosts[u.split("_")[0]].append(u)
        parts[p] = dict(
            hosts=len(hosts), units=len(us),
            units_per_arm=dict(sorted(collections.Counter(arm_code[u.split("_")[1]] for u in us).items())),
            leaves_per_unit=sorted(leaves.get(u, 0) for u in us),
            cells_per_unit=sorted(cells_main.get(u, 0) for u in us),
            leaves_per_host=sorted(sum(leaves.get(u, 0) for u in hu) for hu in hosts.values()),
            hosts_with_unit_ge_200_leaves=sum(any(leaves.get(u, 0) >= 200 for u in hu) for hu in hosts.values()),
        )
    return parts


def run(cells, leaves, outdir, v1_of=lambda h: partition_of(V1, h)):
    main = max(cells, key=lambda k: len(cells[k]), default=None)
    cells_main = cells[main] if main else {}
    units = set(leaves) | set(cells_main)
    host_leaves = collections.Counter()
    for u in units:
        host_leaves[u.split("_")[0]] += leaves.get(u, 0)
    mapping = assign(dict(host_leaves), v1_of)
    parts = structure(units, leaves, cells_main, mapping)
    for p, v in parts.items():
        print(f"{p}: hosts {v['hosts']}, units {v['units']} {v['units_per_arm']}, leaves per host {v['leaves_per_host']}")
    res = dict(dataset="growth12_host_v2", rule="train = v1 train; v1-sealed hosts by leaves desc, snake ABCCBA",
               mapping=dict(sorted(mapping.items())), partitions=parts)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    (Path(outdir) / "c15s0_rebal.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def synthetic():
    leaves = {f"{1000 + i}_AA_T1": 100 * (i + 1) for i in range(20)}
    v1 = lambda h: "train" if int(h) < 1005 else "A"  # noqa: E731
    with tempfile.TemporaryDirectory() as wd:
        r1 = run({}, leaves, wd, v1)
        r2 = run({}, leaves, wd, v1)
    assert r1["mapping"] == r2["mapping"]
    assert [r1["partitions"][p]["hosts"] for p in "ABC"] == [5, 5, 5]
    assert all(r1["mapping"][str(1000 + i)] == "train" for i in range(5))
    assert r1["mapping"]["1019"] == "A" and r1["mapping"]["1018"] == "B" and r1["mapping"]["1016"] == "C" \
        and r1["mapping"]["1015"] == "B" and r1["mapping"]["1013"] == "A", r1["mapping"]
    print("SYNTHETIC OK")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--synthetic":
        return synthetic()
    dl = HERE.parent / "data" / "ext_c15" / "raw"
    dl.mkdir(parents=True, exist_ok=True)
    tar = dl / "record_5847461.tar.gz"
    g.download(tar)
    with tempfile.TemporaryDirectory() as wd:
        cells, leaves, _ = st.scan(tar, wd)
    run(cells, leaves, OUT)


if __name__ == "__main__":
    main()
