"""The only door to held-out test statistics. Enforces the autoresearch test budget.

Each test may be split by clone into sealed shards (clone-level hash, so a clone is never in two
shards); every (test, shard) answers exactly once. A look requires a pre-registration heading for
the experiment in EXPERIMENT_LOG.md and is appended to test_ledger.tsv with the commit it ran at.
Parallel research tracks get different shards of the same test, or share one joint look.

    uv run python experiments/test_oracle.py status
    uv run python experiments/test_oracle.py look epo_scf_pooled --exp E12
    uv run python experiments/test_oracle.py look epo_scf_pooled --exp E13 --shard 0 --shards 2
"""

from __future__ import annotations

import argparse
import datetime
import functools
import hashlib
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
from fit_cytokine import CONDS5, load_counts, pooled_pairs_r  # noqa: E402

HERE = os.path.dirname(__file__)
LEDGER = os.path.join(HERE, "test_ledger.tsv")
LOG = os.path.join(HERE, "EXPERIMENT_LOG.md")
COLUMNS = ["date", "experiment", "test", "shard", "commit", "result"]
SALT = "cellware-test-v1"  # fixed forever: changing it would reshuffle sealed shards


@functools.lru_cache(maxsize=None)  # pure; the hash dominated fit time before caching
def shard_of(clone, shards):
    return int(hashlib.sha256(f"{SALT}:{clone}".encode()).hexdigest(), 16) % shards


def keep_shard(cnt, shard, shards):
    """Restrict {cond: {day: {clone: counts}}} to one clone shard."""
    if shards == 1:
        return cnt
    return {c: {d: {k: v for k, v in by.items() if shard_of(k, shards) == shard}
                for d, by in days.items()} for c, days in cnt.items()}


def epo_scf_pooled(shard, shards):
    return pooled_pairs_r(keep_shard(load_counts(CONDS5), shard, shards))


def presence_heldout(shard, shards):
    """B37: B31 presence statistics on Epo/SCF clone shard 1 of 2 (whole test; shard args unused)."""
    import b31_copresence as b31
    from clone_state_model import split

    s = b31.stats(split(load_counts(CONDS5), 1), {})
    keys = ("a_lift_ge2", "a_den_ge2", "a_lift_ge3", "a_den_ge3", "b_PG_given_M6+", "b_den_M6+")
    return {k: float(s[k]) for k in keys}


def skew_heldout(shard, shards):
    """E51: SIS-skew statistics on the H2 held-out shard (shard 0 of 3; shard args unused)."""
    import numpy as np

    import sis_data as sd
    from b39_sisskew import GROUPS
    from r2_skew_sbi import heldout_index, svec

    d = sd.load_skew("heldout")
    blocks, n = [], {}
    for g in ("null", "KO"):
        m = np.isin(d["well"], GROUPS[g])
        n[g] = int(m.sum())
        blocks.append(svec(d["P"][m], d["N"][m]))
    s = np.concatenate(blocks)
    return {"n": n, "index": heldout_index(), "values": [float(s[i]) for i in heldout_index()]}


# name -> (function, description). Tests already spent elsewhere are recorded in the ledger.
TESTS = {
    "epo_scf_pooled": (epo_scf_pooled, "cytokine expt: N-fraction concordance pooled over every "
                                       "condition pair with Epo or SCF"),
    "presence_heldout": (presence_heldout, "cytokine expt: G-M lift (>= 2, >= 3 cells) and "
                                           "P(G | M size 6+) on clone shard 1 of 2 (B37)"),
    "skew_heldout": (skew_heldout, "SIS-skew: 29 B39/B40 statistics on H2 held-out shard 0 of 3 (E51)"),
}


def read_ledger():
    if not os.path.exists(LEDGER):
        return []
    with open(LEDGER, encoding="utf-8") as f:
        rows = [line.rstrip("\n").split("\t") for line in f][1:]
    return [dict(zip(COLUMNS, r, strict=True)) for r in rows if r and r[0]]


def append_ledger(row):
    new = not os.path.exists(LEDGER)
    with open(LEDGER, "a", encoding="utf-8") as f:
        if new:
            f.write("\t".join(COLUMNS) + "\n")
        f.write("\t".join(str(row[c]) for c in COLUMNS) + "\n")


def preregistered(exp):
    with open(LOG, encoding="utf-8") as f:
        return any(line.startswith(f"## {exp}:") and "pre-registered" in line for line in f)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    lk = sub.add_parser("look")
    lk.add_argument("test", choices=sorted(TESTS))
    lk.add_argument("--exp", required=True, help="experiment id with a pre-registration, e.g. E12")
    lk.add_argument("--shard", type=int, default=0)
    lk.add_argument("--shards", type=int, default=1)
    args = ap.parse_args()
    ledger = read_ledger()

    if args.cmd == "status":
        for name, (_, desc) in TESTS.items():
            print(f"{name}: {desc}")
        print("\nledger:")
        for r in ledger:
            print(f"  {r['date']}  {r['experiment']:5s} {r['test']:28s} shard {r['shard']}")
        return

    shard_id = f"{args.shard}/{args.shards}"
    taken = [r for r in ledger if r["test"] == args.test]
    if any(r["shard"] in (shard_id, "0/1") for r in taken) or (
            args.shards == 1 and taken):
        sys.exit(f"refused: {args.test} shard {shard_id} (or the whole test) was already seen")
    if any(r["shard"].split("/")[1] != str(args.shards) for r in taken):
        sys.exit(f"refused: {args.test} is already split into a different number of shards")
    if not preregistered(args.exp):
        sys.exit(f"refused: no '## {args.exp}: ... (pre-registered ...)' heading in the log")
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "experiments/*.py"],
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        sys.exit("refused: commit experiment code first, so the look maps to a commit")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    result = TESTS[args.test][0](args.shard, args.shards)
    append_ledger({"date": datetime.date.today().isoformat(), "experiment": args.exp,
                   "test": args.test, "shard": shard_id, "commit": commit,
                   "result": json.dumps(result)})
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
