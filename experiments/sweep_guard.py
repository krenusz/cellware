"""Cancel a running sweep as soon as its pre-registered stability rule can no longer be met.

Standing rule (2026-11-04): if a sweep needs at least `need` of its seeds to pass the gate, cancel
it once more than (seeds - need) have failed. Finished seeds are read from their per-line
artifacts as each job completes. The gate is E36's: E33's gate plus full-G / full-M co-presence
within ±30% of the fit set's counts.

    uv run python experiments/sweep_guard.py RUN_ID SWEEP_FILE NEED
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import e32_multinomial as e32
import e33_measured as e33
import numpy as np
from sweep import parse

CO = ("full-G", "full-M")


def gh(*args):
    return subprocess.run(["gh", *args], capture_output=True, text=True, check=False)


def gated(rep, co):
    if not rep["adequate"]:
        return False
    z = rep["z"]
    for k in CO:
        if f"co_{k}" not in z:
            return True  # no co-presence target: E33's gate alone
        sim = co[k] + z[f"co_{k}"] * np.sqrt(max(co[k], 4))
        if abs(sim / co[k] - 1) > 0.3:
            return False
    return True


def main():
    run, need = sys.argv[1], int(sys.argv[3])
    outs = {str(i): out for i, out, _ in parse(Path(sys.argv[2]))}
    co = e32.copresence(e33.split(e33.load_counts(e33.CONDS5), None))
    seen, fails, tries = set(), 0, 0
    tmp = tempfile.mkdtemp(prefix="guard_")
    while True:
        r = gh("run", "view", run, "--json", "status,jobs")
        if r.returncode:
            time.sleep(60)  # network blips: keep waiting
            continue
        info = json.loads(r.stdout)
        final = info["status"] == "completed"  # snapshot before judging, so no job is skipped
        jobs = [j for j in info["jobs"] if j["name"].startswith("run (")]
        for j in jobs:
            line = j["name"][5:-1]
            if j["status"] != "completed" or line in seen:
                continue
            d = os.path.join(tmp, line)
            if gh("run", "download", run, "-n", f"line-{line}", "-D", d).returncode:
                continue  # artifact not ready yet; retry next poll
            seen.add(line)
            f = os.path.join(d, outs[line])
            ok = os.path.exists(f) and gated(json.load(open(f)), co)
            fails += not ok
            tag = "gated" if ok else "FAIL"
            print(f"line {line}: {tag} ({fails} fails, {len(seen)}/{len(jobs)})")
            if fails > len(jobs) - need:
                print(f"stability rule ({need}/{len(jobs)}) can no longer be met: cancelling {run}")
                gh("run", "cancel", run)
                return
        if final:
            done = [j for j in jobs if j["conclusion"] == "success"]
            if len(seen) < len(done) and tries < 10:
                tries += 1  # artifacts of the last jobs can lag the run's completion
                time.sleep(30)
                continue
            print(f"run finished: {len(seen) - fails}/{len(jobs)} gated ({len(seen)} judged)")
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
