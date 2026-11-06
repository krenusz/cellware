"""Hold-out guard. Fails (exit 1) if any rule in sealed.py / program.md (H1-H8) is broken.

    uv run python experiments/check_sealed.py

Checks:
  1. every data/ subdirectory has a DATASETS entry (seal decided before analysis);
  2. sealed file names appear only in their allowed reader modules (and the registry);
  3. sealed files with a recorded sha256 are unchanged;
  4. held-out access patterns ("heldout" loaders, include_heldout=True, heldout_stat calls)
     appear only in HELDOUT_READERS modules;
  5. include_heldout=True is used only inside functions named fit / heldout_stat.
Run by test_oracle before every look.
"""

from __future__ import annotations

import glob
import hashlib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import sealed  # noqa: E402

PATTERN = re.compile(r"""["']heldout["']|include_heldout\s*=\s*True|heldout_stat\(|shard_of\(""")
SELF = {"check_sealed.py", "sealed.py"}


def check(py_dir=HERE, data_dir=None):
    errors = []
    data_dir = data_dir or os.path.join(ROOT, "data")
    for d in sorted(os.listdir(data_dir)):
        if os.path.isdir(os.path.join(data_dir, d)) and d not in sealed.DATASETS:
            errors.append(f"data/{d}: no DATASETS entry in sealed.py (seal before any statistic, H1)")
    py = {os.path.basename(p): open(p, encoding="utf-8", errors="replace").read()
          for p in glob.glob(os.path.join(py_dir, "*.py"))}
    for path, (sha, readers, _status) in sealed.SEALED_FILES.items():
        base = os.path.basename(path)
        for name, src in py.items():
            if name in SELF or name in readers:
                continue
            if base in src:
                errors.append(f"{name} names sealed file {base} (allowed: {readers}, H3)")
        full = os.path.join(ROOT, path)
        if sha and os.path.exists(full):
            if hashlib.sha256(open(full, "rb").read()).hexdigest() != sha:
                errors.append(f"{path} changed since sealing (H4)")
    for name, src in py.items():
        if name in SELF:
            continue
        if PATTERN.search(src) and name not in sealed.HELDOUT_READERS:
            for i, line in enumerate(src.splitlines(), 1):
                if PATTERN.search(line):
                    errors.append(f"{name}:{i} held-out access outside approved readers (H2): {line.strip()[:90]}")
        if "include_heldout=True" in src:
            func = None
            for i, line in enumerate(src.splitlines(), 1):
                m = re.match(r"def (\w+)\(", line)
                if m:
                    func = m.group(1)
                if "include_heldout=True" in line and not line.lstrip().startswith(("def ", "#", '"""')) \
                        and "include_heldout=True, which" not in line and func not in ("fit", "heldout_stat"):
                    errors.append(f"{name}:{i} include_heldout=True outside fit/heldout_stat (H2)")
    return errors


if __name__ == "__main__":
    errs = check()
    for e in errs:
        print("FAIL", e)
    print("sealed check:", "OK" if not errs else f"{len(errs)} problem(s)")
    sys.exit(1 if errs else 0)
