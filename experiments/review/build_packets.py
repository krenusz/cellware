"""Build replay packets: the pre-registration as it stood at a past commit, the scripts it names, its
selection file, and program.md and DISCOVERIES.md at that commit. Text from git only; no data file is read."""
import json, os, re, subprocess, sys
REPO, OUT = sys.argv[1], sys.argv[2]
def git(*a):
    r = subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.stdout if r.returncode == 0 else None
def first_commit(exp):
    out = git("log", "--reverse", "--format=%h", f"-S## {exp}:", "--", "experiments/EXPERIMENT_LOG.md")
    return out.split()[0]
def section(log, exp):
    m = re.search(rf"^## {re.escape(exp)}:.*$", log, re.M)
    if not m:
        return None
    n = re.search(r"^## ", log[m.end():], re.M)
    return log[m.start(): m.end() + n.start()] if n else log[m.start():]
CASES = json.load(open(sys.argv[3], encoding="utf-8"))
rows = []
for c in CASES:
    commit = c.get("commit") or first_commit(c["at"])
    commit = git("rev-parse", "--short", commit).strip()
    log = git("show", f"{commit}:experiments/EXPERIMENT_LOG.md")
    secs = [section(log, e) for e in c["sections"]]
    assert all(secs), (c["id"], [e for e, s in zip(c["sections"], secs) if not s])
    d = os.path.join(OUT, c["id"]); os.makedirs(os.path.join(d, "scripts"), exist_ok=True)
    text = "\n".join(secs)
    open(os.path.join(d, "PREREG.md"), "w", encoding="utf-8", newline="\n").write(text)
    names = sorted(set(re.findall(r"\b([a-z][a-z0-9_]*\.py)\b", text)) | set(c.get("scripts", [])))
    got = []
    for n in names:
        if n in c.get("skip", []):
            continue
        src = git("show", f"{commit}:experiments/{n}")
        if src and len(src.splitlines()) <= 450:
            open(os.path.join(d, "scripts", n), "w", encoding="utf-8", newline="\n").write(src); got.append(n)
    sels = []
    for e in c.get("selection", c["sections"]):
        s = git("show", f"{commit}:experiments/selection/{e}.json")
        if s:
            os.makedirs(os.path.join(d, "selection"), exist_ok=True)
            open(os.path.join(d, "selection", e + ".json"), "w", encoding="utf-8", newline="\n").write(s); sels.append(e)
    prog = git("show", f"{commit}:experiments/program.md") or ""
    open(os.path.join(d, "program.md"), "w", encoding="utf-8", newline="\n").write(prog)
    disc = git("show", f"{commit}:experiments/DISCOVERIES.md") or ""
    open(os.path.join(d, "DISCOVERIES.md"), "w", encoding="utf-8", newline="\n").write(disc)
    outcome = bool(re.search(r"(?im)^\*\*(outcome|result)", text))
    rows.append((c["id"], commit, len(text.splitlines()), got, sels, len(disc.splitlines()), "OUTCOME-IN-TEXT" if outcome else ""))
for r in rows:
    print(*r)
