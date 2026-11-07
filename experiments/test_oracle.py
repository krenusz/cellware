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

import numpy as np

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


def sisseq_rna_heldout(shard, shards):
    """E54: corr(RNA score, pooled sister logit) per fate on SIS-seq H2 held-out clones."""
    import core_sisseq

    return core_sisseq.heldout_stat()


def larry_fm_heldout(shard, shards):
    """E55: Geneformer vs PCA-50 clone-fate prediction on LARRY in vitro e55 shard 0 of 3."""
    import e55_fm

    return e55_fm.heldout_stat()


def gastruloid_polarity_heldout(shard, shards):
    """G1a: per clone x replicate held-out polarity means vs fit-shard intervals (g1 shard 0 of 3)."""
    import g1_gastruloid

    return g1_gastruloid.heldout_stat()


def gastruloid_p4_memory(shard, shards):
    """G6a: sealed passage-4 clone + bulk polarity / G1a passage-1 means vs r^3 (whole files)."""
    import g6_confirm

    return g6_confirm.p4_memory()


def gastruloid_jan24_pairs(shard, shards):
    """G6b: sealed Jan 2024 clone 1+3 pairs, passage 1 propensity and passage 4 / 1 fade."""
    import g6_confirm

    return g6_confirm.jan24_pairs()


def gastruloid_g7_pairs(shard, shards):
    """G7a: sealed 3+7 / 6+7 pair ordering vs simulator v2."""
    import g7_confirm

    return g7_confirm.g7a_pairs()


def gastruloid_g7_memory(shard, shards):
    """G7b: sealed July 2024 clone-in-bulk passages 2-4 vs simulator v2 / v1 memory."""
    import g7_confirm

    return g7_confirm.g7b_memory()


# name -> (function, description). Tests already spent elsewhere are recorded in the ledger.
TESTS = {
    "epo_scf_pooled": (epo_scf_pooled, "cytokine expt: N-fraction concordance pooled over every "
                                       "condition pair with Epo or SCF"),
    "gastruloid_g7_memory": (gastruloid_g7_memory, "Raj gastruloids: sealed July 2024 clone 1/3 in bulk, "
                                                   "passages 2-4 vs simulator v2 (G7b)"),
    "gastruloid_g7_pairs": (gastruloid_g7_pairs, "Raj gastruloids: sealed 3+7 / 6+7 pairs vs simulator v2 (G7a)"),
    "gastruloid_jan24_pairs": (gastruloid_jan24_pairs, "Raj gastruloids: sealed Jan 2024 clone 1+3 pairs, "
                                                       "passage 1 and 4 (G6b)"),
    "gastruloid_p4_memory": (gastruloid_p4_memory, "Raj gastruloids: sealed passage-4 clone + bulk files (G6a)"),
    "gastruloid_polarity_heldout": (gastruloid_polarity_heldout, "Raj gastruloids: clone A-P polarity, "
                                                                 "g1 shard 0 of 3 (G1a)"),
    "larry_fm_heldout": (larry_fm_heldout, "LARRY in vitro: Geneformer vs PCA-50 day-2 state -> clone N "
                                           "fraction, e55 shard 0 of 3 (E55)"),
    "presence_heldout": (presence_heldout, "cytokine expt: G-M lift (>= 2, >= 3 cells) and "
                                           "P(G | M size 6+) on clone shard 1 of 2 (B37)"),
    "sisseq_rna_heldout": (sisseq_rna_heldout, "SIS-seq: RNA score vs sister fate logit, H2 held-out shard 0 of 3 (E54)"),
    "skew_heldout": (skew_heldout, "SIS-skew: 29 B39/B40 statistics on H2 held-out shard 0 of 3 (E51)"),
}


# Partitioned tests (H11-H13): name -> (function(partition), dataset, description). One look
# spends one sealed partition of the dataset for every test; the last one needs Bence's OK.
def _g11_drug(part):
    import g9_confirm

    return g9_confirm.g11_drug(part)


def _g12_drug(part):
    import g9_confirm

    return g9_confirm.g12_drug(part)


def _g12_counts(part):
    import g9_confirm

    return g9_confirm.g12_counts(part)


PARTITION_TESTS = {
    "gastruloid_g12_drug": (_g12_drug, "raj_drug", "G12: pooled drugs vs control, one-sided, drug partition B"),
    "gastruloid_g11_drug": (_g11_drug, "raj_drug", "G11: drug compression ratio RA/SB vs control on sealed drug "
                                                   "series (A = May 3 2025), one partition per look"),
}
SELECTION_DIR = os.path.join(HERE, "selection")


def selection_gate(exp, test):
    """H12: a look needs experiments/selection/<exp>.json, committed, choosing the test's model as
    the best train-fit (cross-validated on the train partition) among the candidates."""
    path = os.path.join(SELECTION_DIR, f"{exp}.json")
    if not os.path.exists(path):
        return f"no train-fit selection file selection/{exp}.json (H12)"
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", path], capture_output=True).returncode == 0
    dirty = subprocess.run(["git", "status", "--porcelain", "--", path], capture_output=True, text=True).stdout.strip()
    if not tracked or dirty:
        return f"selection/{exp}.json must be committed before the look (H12)"
    s = json.load(open(path, encoding="utf-8"))
    cands = s.get("candidates", {})
    if s.get("test") != test or len(cands) < 2 or s.get("chosen") not in cands:
        return f"selection/{exp}.json must name test={test}, >= 2 candidates with train scores, and chosen (H12)"
    best = (max if s.get("higher_is_better", True) else min)(cands, key=cands.get)
    if s["chosen"] != best:
        return f"chosen {s['chosen']} is not the best train fit ({best}) (H12)"
    return None


PARTITION_COUNTS = {"gastruloid_g12_drug": _g12_counts}  # test -> function(partition) -> {group: n units}; counts only (H9)


def power_gate(exp, test, part):
    """H14: power of the pre-registered contrast from counts only; refuse below the minimum."""
    from scipy.stats import norm

    s = json.load(open(os.path.join(SELECTION_DIR, f"{exp}.json"), encoding="utf-8"))
    p = s.get("power")
    if not p or test not in PARTITION_COUNTS:
        return f"no power block in selection/{exp}.json or no count function for {test} (H14)"
    n = PARTITION_COUNTS[test](part)
    if min(n.values()) < 2:
        return f"partition {part} counts {n}: a group is empty (H14)"
    se = p["sd"] * np.sqrt(sum(1 / v for v in n.values()))
    z = norm.ppf(1 - p["alpha"]) if p.get("one_sided") else norm.ppf(1 - p["alpha"] / 2)
    power = float(1 - norm.cdf(z - p["effect"] / se))
    print(f"H14 power: counts {n}, sd {p['sd']}, effect {p['effect']}, power {power:.2f} (min {p['min_power']})")
    if power < p["min_power"]:
        return f"power {power:.2f} < {p['min_power']} for partition {part} (counts {n}); partition NOT spent (H14)"
    return None


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


PREFLIGHT_FROM = "2026-11-08"  # H15 applies to pre-registrations dated from this day on


def preregistered(exp):
    """Heading '## <exp>: ... (pre-registered <date>)'; from PREFLIGHT_FROM on, the section must
    also hold a 'Pre-flight:' block (H15)."""
    import re

    text = open(LOG, encoding="utf-8").read()
    m = re.search(rf"^## {re.escape(exp)}:.*pre-registered (\d{{4}}-\d{{2}}-\d{{2}}).*$", text, re.M)
    if not m:
        return False
    if m.group(1) < PREFLIGHT_FROM:
        return True
    nxt = re.search(r"^## ", text[m.end():], re.M)
    section = text[m.end(): m.end() + nxt.start()] if nxt else text[m.end():]
    if "Pre-flight:" not in section:
        sys.exit(f"refused: {exp} pre-registration has no 'Pre-flight:' block (H15)")
    if "Serves:" not in section:
        sys.exit(f"refused: {exp} pre-registration has no 'Serves:' line (H16 scope guard)")
    return True


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    lk = sub.add_parser("look")
    lk.add_argument("test", choices=sorted(TESTS) + sorted(PARTITION_TESTS))
    lk.add_argument("--exp", required=True, help="experiment id with a pre-registration, e.g. E12")
    lk.add_argument("--shard", type=int, default=0)
    lk.add_argument("--shards", type=int, default=1)
    lk.add_argument("--partition", choices=["A", "B", "C"], help="sealed partition (partitioned tests)")
    lk.add_argument("--last-ok", default="", help="Bence's verbatim OK to spend the LAST partition (H13)")
    ct = sub.add_parser("count", help="structure only: units per tag in a sealed drug partition (H9/H14)")
    ct.add_argument("--partition", choices=["A", "B"], required=True)
    args = ap.parse_args()
    if args.cmd == "count":
        import g9_confirm

        print(json.dumps(g9_confirm.tag_counts(args.partition), indent=1))
        return
    ledger = read_ledger()

    if args.cmd == "look" and args.test in PARTITION_TESTS:
        import partitions

        func, dataset, _ = PARTITION_TESTS[args.test]
        if not args.partition:
            sys.exit("refused: partitioned test needs --partition A|B|C (H11)")
        used = {r["shard"] for r in ledger if r["shard"].startswith(f"{dataset}:")}
        tag = f"{dataset}:{args.partition}"
        if tag in used:
            sys.exit(f"refused: partition {tag} already spent (H11)")
        left = [p for p in partitions.SEALED if f"{dataset}:{p}" not in used]
        if len(left) == 1 and len(args.last_ok.strip()) < 10:
            sys.exit(f"refused: {tag} is the LAST sealed partition; needs --last-ok with Bence's verbatim OK (H13)")
        if not preregistered(args.exp):
            sys.exit(f"refused: no '## {args.exp}: ... (pre-registered ...)' heading in the log")
        gate = selection_gate(args.exp, args.test)
        if gate:
            sys.exit("refused: " + gate)
        pw = power_gate(args.exp, args.test, args.partition)
        if pw:
            sys.exit("refused: " + pw)
        import check_sealed
        problems = check_sealed.check()
        if problems:
            sys.exit("refused: hold-out guard failed:\n  " + "\n  ".join(problems))
        if subprocess.run(["git", "status", "--porcelain", "--", "experiments/*.py"],
                          capture_output=True, text=True).stdout.strip():
            sys.exit("refused: commit experiment code first, so the look maps to a commit")
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True).stdout.strip()
        result = func(args.partition)
        if args.last_ok:
            result = {"last_partition_ok": args.last_ok, **result}
        append_ledger({"date": datetime.date.today().isoformat(), "experiment": args.exp,
                       "test": args.test, "shard": tag, "commit": commit, "result": json.dumps(result)})
        print(json.dumps(result, indent=1))
        return

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
    gate = selection_gate(args.exp, args.test)  # H12 applies to every look from 2026-11-07
    if gate:
        sys.exit("refused: " + gate)
    import check_sealed  # hold-out guard (rules H1-H13, program.md)
    problems = check_sealed.check()
    if problems:
        sys.exit("refused: hold-out guard failed:\n  " + "\n  ".join(problems))
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
