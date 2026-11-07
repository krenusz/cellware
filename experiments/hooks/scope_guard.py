"""Scope guard against research drift (program.md H15/H16; Bence 2026-11-07: "create hooks to defense
us from drifting like this again!").

Modes
  claude-pre     Claude Code PreToolUse hook (Bash / PowerShell / Write). Blocks
                 - running an experiment script (name like g13_diff, c3s1_core_sim) unless it is in
                   FOCUS.json allowed_scripts or named in an EXPERIMENT_LOG section whose `Serves:` line
                   names the current FOCUS step;
                 - writing a NEW experiment script before such a pre-registration exists.
  claude-prompt  Claude Code UserPromptSubmit hook: in Cellware sessions, injects the current focus.
  commit-msg F   git commit-msg hook: FOCUS.json changes need 'Bence:' in the message; new
                 pre-registrations (dated >= 2026-11-08) need `Pre-flight:` and a `Serves:` line naming
                 the current step (or quoting 'Bence OK').

Stdlib only; run with -I.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOCUS = os.path.join(EXP, "FOCUS.json")
LOG = os.path.join(EXP, "EXPERIMENT_LOG.md")
SCRIPT_RE = re.compile(r"\b([a-z]+\d+[a-z0-9]*_[a-z0-9_]+)(?:\.py)?\b")
HEAD_RE = re.compile(r"^## (\S+?):.*\(pre-registered (\d{4}-\d{2}-\d{2})", re.M)
FROM = "2026-11-08"


def focus():
    return json.load(open(FOCUS, encoding="utf-8"))


def sections(text):
    heads = list(HEAD_RE.finditer(text))
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        nxt = re.search(r"^## ", text[h.end():end], re.M)
        body = text[h.start(): h.end() + nxt.start()] if nxt else text[h.start():end]
        yield h.group(1), h.group(2), body


def serves_step(body, step):
    return any(step in line for line in body.splitlines() if "Serves:" in line)


def allowed_script(name, f):
    if name in f.get("allowed_scripts", []):
        return True
    text = open(LOG, encoding="utf-8").read()
    return any(serves_step(b, f["step"]) and name in b for _, _, b in sections(text))


def deny(reason):
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                             "permissionDecisionReason": reason}}))
    sys.exit(0)


def claude_pre():
    d = json.load(sys.stdin)
    tool, ti = d.get("tool_name", ""), d.get("tool_input", {}) or {}
    f = focus()
    msg = (f"Cellware scope guard (H16). Current focus {f['step']}: {f['question']} "
           f"Pre-register with 'Pre-flight:' and 'Serves: {f['step']} ...' naming the script, or get Bence's OK "
           f"in chat and record it in FOCUS.json (commit message quoting 'Bence:').")
    if tool in ("Bash", "PowerShell"):
        cmd = ti.get("command", "")
        if "cellware" not in cmd.lower() and "cellware" not in str(d.get("cwd", "")).lower():
            return
        if not re.search(r"\bpython\b", cmd):
            return
        names = {m.group(1) for m in SCRIPT_RE.finditer(cmd)
                 if os.path.exists(os.path.join(EXP, m.group(1) + ".py"))}  # real experiment scripts only
        bad = sorted(n for n in names if not allowed_script(n, f))
        if bad:
            deny(f"Blocked: {', '.join(bad)} is not part of the current focus. {msg}")
    elif tool == "Write":
        p = str(ti.get("file_path", "")).replace("\\", "/")
        if "cellware/experiments/" not in p.lower() or not p.endswith(".py") or os.path.exists(ti.get("file_path", "")):
            return
        name = os.path.basename(p)[:-3]
        if SCRIPT_RE.fullmatch(name) and not allowed_script(name, f):
            deny(f"Blocked: new experiment script {name}.py has no pre-registration serving {f['step']}. {msg}")


def claude_prompt():
    d = json.load(sys.stdin)
    tp = d.get("transcript_path", "")
    hit = "cellware" in str(d.get("prompt", "")).lower()
    if not hit and tp and os.path.exists(tp):
        with open(tp, "rb") as fh:
            fh.seek(max(0, os.path.getsize(tp) - 400_000))
            hit = b"cellware" in fh.read().lower()
    if not hit:
        return
    f = focus()
    ctx = (f"[Cellware focus, FOCUS.json] Step {f['step']}: {f['question']} Next: {'; '.join(f['next_steps'])}. "
           f"Parked (no work without Bence's OK): {', '.join(f['parked'])}. Every proposed step must serve this "
           f"focus (H16) and pass pre-flight (H15); propose nothing else.")
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": ctx}}))


def commit_msg(path):
    msg = open(path, encoding="utf-8").read()
    staged = subprocess.run(["git", "diff", "--cached", "--name-only"], capture_output=True, text=True).stdout.split()
    errs = []
    if "experiments/FOCUS.json" in staged and "Bence:" not in msg:
        errs.append("FOCUS.json changed: the commit message must quote Bence's OK ('Bence: \"...\"').")
    if "experiments/EXPERIMENT_LOG.md" in staged:
        f = json.loads(subprocess.run(["git", "show", ":experiments/FOCUS.json"], capture_output=True,
                                      text=True).stdout or open(FOCUS, encoding="utf-8").read())
        text = subprocess.run(["git", "show", ":experiments/EXPERIMENT_LOG.md"], capture_output=True,
                              text=True, encoding="utf-8").stdout
        for exp, date, body in sections(text):
            if date < FROM:
                continue
            if "Pre-flight:" not in body:
                errs.append(f"{exp}: pre-registration lacks 'Pre-flight:' (H15).")
            if not (serves_step(body, f["step"]) or "Bence OK" in body):
                errs.append(f"{exp}: 'Serves:' must name current step {f['step']} or quote 'Bence OK' (H16).")
    if errs:
        sys.stderr.write("commit refused by scope guard:\n  " + "\n  ".join(errs) + "\n")
        sys.exit(1)


if __name__ == "__main__":
    mode = sys.argv[1]
    try:
        if mode == "claude-pre":
            claude_pre()
        elif mode == "claude-prompt":
            claude_prompt()
        elif mode == "commit-msg":
            commit_msg(sys.argv[2])
    except SystemExit:
        raise
    except Exception as e:  # never break the session on a guard bug; say so
        if mode == "commit-msg":
            sys.stderr.write(f"scope guard error: {e}\n")
            sys.exit(1)
        print(json.dumps({"systemMessage": f"Cellware scope guard error: {e}"}))
