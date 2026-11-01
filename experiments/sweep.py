"""Run a list of experiment commands with N parallel workers, skipping finished ones.

One command per line in the sweep file; blank lines and `#` comments are ignored. A line may
start with `[path] ` to name the output file it writes: if that file exists, the line is skipped,
so a sweep can be resumed after an interruption. Logs go to `<logdir>/<line number>.log`.

    # this PC: one worker only (it hard-resets under parallel load)
    uv run python experiments/sweep.py sweeps/e7_cue.txt --workers 1
    # Colab / cloud: one worker per core
    python experiments/sweep.py sweeps/e7_cue.txt --workers 8
    # GitHub Actions: one job per line (.github/workflows/sweep.yml)
    gh workflow run sweep.yml --ref modernize -f sweep=sweeps/e7_cue.txt
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def parse(path: Path):
    jobs = []
    for i, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = re.match(r"\[(.+?)\]\s+(.*)", line)
        out, cmd = (m.group(1), m.group(2)) if m else (None, line)
        jobs.append((i, out, cmd))
    return jobs


def run(job, logdir: Path, threads: int):
    i, out, cmd = job
    if out and Path(out).exists():
        return i, "skip", 0.0
    env = {**os.environ, "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)}
    start = time.time()
    with open(logdir / f"{i}.log", "w") as log:
        code = subprocess.call(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT, env=env)
    return i, "ok" if code == 0 else f"fail({code})", time.time() - start


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sweep", type=Path)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--threads", type=int, default=2, help="BLAS/torch threads per job")
    p.add_argument("--logdir", type=Path, default=None)
    p.add_argument("--lines", default=None, help="only these line numbers, e.g. 3,5,10-20")
    p.add_argument("--list", action="store_true", help="print pending line numbers as JSON")
    args = p.parse_args()
    jobs = parse(args.sweep)
    if args.lines:
        keep = set()
        for part in args.lines.split(","):
            a, _, b = part.partition("-")
            keep.update(range(int(a), int(b or a) + 1))
        jobs = [j for j in jobs if j[0] in keep]
    if args.list:
        print(json.dumps([i for i, out, _ in jobs if not (out and Path(out).exists())]))
        return
    logdir = args.logdir or Path("experiments/logs_sweep") / args.sweep.stem
    logdir.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} jobs, {args.workers} workers", flush=True)
    failed = 0
    with ThreadPoolExecutor(args.workers) as pool:
        for i, status, secs in pool.map(lambda j: run(j, logdir, args.threads), jobs):
            failed += status.startswith("fail")
            print(f"line {i:3d}: {status} ({secs:.0f}s)", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
