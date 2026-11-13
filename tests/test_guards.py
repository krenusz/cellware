"""End-to-end tests of the research guards (experiments/program.md H1-H21).

Each test builds a throwaway git repo that holds the REAL oracle, sealed-file check, power rule and
scope-guard hook, plus synthetic stubs for the data side. Nothing here reads real data: the stub
test function only writes a marker file, which proves whether the oracle let a look through.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments"
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP / "hooks"))

import power  # noqa: E402
import scope_guard  # noqa: E402

TEST = "celltag_c9s2_atac"
MARKER = "<!-- scope-guard: sections below need Pre-flight and Serves -->"
SEALED_NAME = "toy_sealed_k7.csv"
# Closed-chapter scripts that keep their own power arithmetic (their looks map to past commits).
LEGACY_POWER = {
    "c4s2a_morph_confirm.py",
    "c6s4_fit.py",
    "c6s7_checks.py",
    "c6s7_sort.py",
    "c6s7a_sort.py",
    "c6s7k_ppost.py",
    "c6s7l_axis.py",
    "c6s8a_power.py",
    "c6s8d_size.py",
    "c7s2a_diag.py",
    "c7s5_reprog.py",
    "c7s7_confirm.py",
    "c8s0_scan.py",
    "c8s1_confirm.py",
    "c9s0_scan.py",
    "c9s1_confirm.py",
    "kin_corr_e19.py",
}


def prereg(exp="T1", date="2026-11-14", serves="C9-S2", preflight=True, practice=True, extra=""):
    lines = [f"## {exp}: toy test (pre-registered {date})", f"Serves: {serves} toy question."]
    if preflight:
        lines.append(
            "Pre-flight: (1) unit ok (2) readout ok (3) used data (4) power (5) none (6) ok (7) ok"
        )
    if practice:
        lines.append("Practice: statistic list ticked.")
    lines.append(f"Script: c9s2_confirm.py {extra}")
    return "\n".join(lines) + "\n\n"


def selection(**over):
    sel = {
        "test": TEST,
        "higher_is_better": True,
        "candidates": {"a": 0.2, "b": 0.1},
        "chosen": "a",
        "power": {
            "sd": 1.0,
            "effect": 0.5,
            "alpha": 0.025,
            "one_sided": True,
            "min_power": 0.8,
            "sources": {"effect": "train CV", "sd": "train CV"},
        },
        "readout_gate": {"pass": True, "sides": {"neu": 60, "mono": 181}},
    }
    sel.update(over)
    return sel


def git(root, *args, check=True):
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=check)


class Box:
    """A throwaway repo with the real guards and synthetic data stubs."""

    def __init__(self, root):
        self.root = root
        self.exp = root / "experiments"
        (self.exp / "hooks").mkdir(parents=True)
        (self.exp / "selection").mkdir()
        for name in ("test_oracle.py", "check_sealed.py", "power.py", "RULES_DIGEST.md"):
            shutil.copy(EXP / name, self.exp / name)
        shutil.copy(EXP / "hooks" / "scope_guard.py", self.exp / "hooks" / "scope_guard.py")
        (root / ".githooks").mkdir()
        hook = root / ".githooks" / "commit-msg"
        shutil.copy(REPO / ".githooks" / "commit-msg", hook)
        hook.chmod(hook.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        data = root / "data" / "toy"
        data.mkdir(parents=True)
        (data / SEALED_NAME).write_text("unit,outcome\nu1,1\n", encoding="utf-8")
        sha = hashlib.sha256((data / SEALED_NAME).read_bytes()).hexdigest()
        self.write(
            "fit_cytokine.py",
            "CONDS5 = []\n\n\ndef load_counts(*a, **k):\n    return {}\n\n\n"
            "def pooled_pairs_r(*a, **k):\n    return 0.0\n",
        )
        self.write(
            "sealed.py",
            f'DATASETS = {{"toy": "sealed"}}\n'
            f'SEALED_FILES = {{"data/toy/{SEALED_NAME}": '
            f'("{sha}", ["c9s2_confirm.py"], "sealed")}}\n'
            'HELDOUT_READERS = {"test_oracle.py", "c9s2_confirm.py"}\n',
        )
        self.write("partitions.py", 'PARTITIONED = {"celltag_lsk": {}}\nSEALED = ("A", "B", "C")\n')
        self.write(
            "c9s2_confirm.py",
            "import json\nimport os\n\nHERE = os.path.dirname(os.path.abspath(__file__))\n"
            f'FILE = "{SEALED_NAME}"\n\n\n'
            "def counts_lsk(part):\n"
            '    return json.load(open(os.path.join(HERE, "toy_counts.json")))\n\n\n'
            "def test_lsk(part):\n"
            '    open(os.path.join(HERE, "LOOKED_" + part), "w").write("x")\n'
            '    return {"rho": 0.3}\n',
        )
        self.write("toy_counts.json", json.dumps({"clones": 100}))
        self.write(
            "EXPERIMENT_LOG.md",
            "# log\n\n## E1: old (pre-registered 2026-11-01)\n\n" + MARKER + "\n\n",
        )
        self.append_log(prereg())
        self.focus(step="C9-S2")
        self.write("selection/T1.json", json.dumps(selection()))
        git(root, "init", "-q")
        git(root, "config", "user.email", "guard@example.invalid")
        git(root, "config", "user.name", "guard test")
        git(root, "config", "core.autocrlf", "false")
        git(root, "add", "-A")
        git(root, "commit", "-q", "-m", "sandbox")
        git(root, "config", "core.hooksPath", ".githooks")  # the real commit-msg hook, from here on

    def write(self, rel, text):
        (self.exp / rel).write_text(text, encoding="utf-8", newline="\n")

    def append_log(self, text):
        with open(self.exp / "EXPERIMENT_LOG.md", "a", encoding="utf-8", newline="\n") as f:
            f.write(text)

    def focus(self, **over):
        f = {"step": "C9-S2", "question": "toy question.", "next_steps": ["n"], "parked": ["p"]}
        f.update(over)
        self.write("FOCUS.json", json.dumps(f))

    def commit(self, msg="change"):
        git(self.root, "add", "-A")
        return git(self.root, "commit", "-q", "-m", msg, check=False)

    def look(self, part="A", exp="T1", last_ok=None):
        cmd = [
            sys.executable,
            "experiments/test_oracle.py",
            "look",
            TEST,
            "--exp",
            exp,
            "--partition",
            part,
        ]
        if last_ok:
            cmd += ["--last-ok", last_ok]
        return subprocess.run(cmd, cwd=self.root, capture_output=True, text=True)

    def looked(self):
        return sorted(p.name for p in self.exp.glob("LOOKED_*"))

    def ledger_rows(self):
        p = self.exp / "test_ledger.tsv"
        return len(p.read_text(encoding="utf-8").splitlines()) - 1 if p.exists() else 0

    def hook(self, mode, payload):
        return subprocess.run(
            [sys.executable, "-I", "experiments/hooks/scope_guard.py", mode],
            cwd=self.root,
            input=json.dumps(payload),
            capture_output=True,
            text=True,
        )


@pytest.fixture
def box(tmp_path):
    return Box(tmp_path / "cellware")  # the hook only acts on paths that contain "cellware"


def refused(box, res, needle):
    assert res.returncode != 0, res.stdout
    assert needle in res.stderr, res.stderr
    assert box.looked() == [], "the sealed test function ran"
    assert box.ledger_rows() == 0, "a refused look was written to the ledger"


# --- oracle: a look goes through only when every gate passes -------------------------------------


def test_oracle_allows_a_complete_look(box):
    res = box.look("A")
    assert res.returncode == 0, res.stderr
    assert box.looked() == ["LOOKED_A"]
    assert box.ledger_rows() == 1


def test_oracle_refuses_without_preregistration(box):
    refused(box, box.look("A", exp="T9"), "no '## T9:")


@pytest.mark.parametrize(
    ("kwargs", "needle"),
    [
        ({"preflight": False}, "Pre-flight"),
        ({"practice": False}, "Practice"),
    ],
)
def test_oracle_refuses_incomplete_preregistration(box, kwargs, needle):
    box.append_log(prereg(exp="T2", **kwargs))
    box.write("selection/T2.json", json.dumps(selection()))
    assert box.commit("add T2").returncode != 0  # the commit hook refuses it first
    git(
        box.root, "commit", "-q", "--no-verify", "-m", "force T2"
    )  # sandbox only: bypass to reach the oracle
    refused(box, box.look("A", exp="T2"), needle)


def test_oracle_practice_line_not_needed_before_cutoff(box):
    box.append_log(prereg(exp="T0", date="2026-11-11", practice=False))
    sel = selection()
    del sel["readout_gate"], sel["power"]["sources"]  # not required before the cutoff either
    box.write("selection/T0.json", json.dumps(sel))
    assert box.commit("add T0").returncode == 0
    assert box.look("A", exp="T0").returncode == 0


def test_oracle_refuses_without_selection_file(box):
    (box.exp / "selection" / "T1.json").unlink()
    box.commit("drop selection")
    refused(box, box.look("A"), "H12")


def test_oracle_refuses_uncommitted_selection(box):
    box.write("selection/T1.json", json.dumps(selection(chosen="b")))
    refused(box, box.look("A"), "must be committed")


def test_oracle_refuses_when_chosen_is_not_best_train_fit(box):
    box.write("selection/T1.json", json.dumps(selection(chosen="b")))
    box.commit("pick b")
    refused(box, box.look("A"), "not the best train fit")


def test_oracle_refuses_low_power_and_keeps_partition(box):
    box.write("toy_counts.json", json.dumps({"clones": 4}))
    box.commit("small partition")
    refused(box, box.look("A"), "partition NOT spent")
    box.write("toy_counts.json", json.dumps({"clones": 100}))
    box.commit("full partition")
    assert box.look("A").returncode == 0  # the refused look did not spend A


def test_oracle_floors_min_power_at_08(box):
    sel = selection()
    sel["power"]["min_power"] = 0.1
    box.write("selection/T1.json", json.dumps(sel))
    box.write("toy_counts.json", json.dumps({"clones": 4}))
    box.commit("min_power 0.1")
    refused(box, box.look("A"), "partition NOT spent")


@pytest.mark.parametrize(
    ("change", "needle"),
    [
        (lambda s: s["power"].pop("sources"), "power.sources"),
        (lambda s: s.pop("readout_gate"), "readout_gate"),
        (lambda s: s["readout_gate"].update(sides={"neu": 12, "mono": 181}), "below 30"),
        (lambda s: s["readout_gate"].update({"pass": False}), "pass is not true"),
        (lambda s: s["power"].update(alpha=0.2), "alpha"),
    ],
)
def test_oracle_refuses_weak_power_or_readout_block(box, change, needle):
    sel = selection()
    change(sel)
    box.write("selection/T1.json", json.dumps(sel))
    box.commit("weak block")
    refused(box, box.look("A"), needle)


def test_oracle_refuses_uncommitted_experiment_code(box):
    with open(box.exp / "c9s2_confirm.py", "a", encoding="utf-8") as f:
        f.write("# edited after pre-registration\n")
    refused(box, box.look("A"), "commit experiment code first")


def test_oracle_refuses_changed_sealed_file(box):
    (box.root / "data" / "toy" / SEALED_NAME).write_text("unit,outcome\nu1,0\n", encoding="utf-8")
    refused(box, box.look("A"), "changed since sealing")


def test_oracle_refuses_when_another_script_names_a_sealed_file(box):
    box.write("z9_peek.py", f'PATH = "data/toy/{SEALED_NAME}"\n')
    box.commit("peek")
    refused(box, box.look("A"), "names sealed file")


def test_oracle_refuses_heldout_access_outside_readers(box):
    box.write("z9_rogue.py", "x = load(include_heldout=True)\n")
    box.commit("rogue")
    refused(box, box.look("A"), "held-out access outside approved readers")


def test_oracle_refuses_unregistered_data_directory(box):
    (box.root / "data" / "new_set").mkdir()
    refused(box, box.look("A"), "no DATASETS entry")


def test_partition_is_spent_once_and_last_needs_ok(box):
    assert box.look("A").returncode == 0
    again = box.look("A")
    assert again.returncode != 0 and "already spent" in again.stderr
    assert box.look("B").returncode == 0
    last = box.look("C")
    assert last.returncode != 0 and "LAST sealed partition" in last.stderr
    assert box.looked() == ["LOOKED_A", "LOOKED_B"]
    assert box.ledger_rows() == 2
    assert box.look("C", last_ok="ok spend the last one").returncode == 0
    assert box.ledger_rows() == 3


def test_union_look_refused_when_part_of_it_is_spent(box):
    assert box.look("A").returncode == 0
    res = box.look("AB")  # a union that contains a spent partition
    assert res.returncode != 0 and "already spent" in res.stderr
    assert box.ledger_rows() == 1


# --- commit-msg hook, through a real `git commit` ------------------------------------------------


def test_commit_refuses_focus_change_without_bence_quote(box):
    box.focus(next_steps=["other"])
    res = box.commit("update focus")
    assert res.returncode != 0 and "Bence" in res.stderr
    assert box.commit('update focus\n\nBence: "ok"').returncode == 0


def test_commit_refuses_new_chapter_without_value_and_literature(box):
    box.focus(step="C10-S1")
    res = box.commit('open chapter 10\n\nBence: "go"')
    assert res.returncode != 0 and "H20" in res.stderr
    box.focus(step="C10-S1", value="adds x", literature="top 5 read")
    assert box.commit('open chapter 10\n\nBence: "go"').returncode == 0


def test_commit_allows_pause_and_same_chapter_step(box):
    box.focus(step="C9-S3")
    assert box.commit('next step\n\nBence: "go"').returncode == 0
    box.focus(step="PAUSED")
    assert box.commit('pause\n\nBence: "pause"').returncode == 0


@pytest.mark.parametrize(
    ("kwargs", "needle"),
    [
        ({"preflight": False}, "Pre-flight"),
        ({"practice": False}, "Practice"),
        ({"serves": "C3-S1"}, "Serves"),
    ],
)
def test_commit_refuses_incomplete_preregistration(box, kwargs, needle):
    box.append_log(prereg(exp="T2", **kwargs))
    res = box.commit("add T2")
    assert res.returncode != 0 and needle in res.stderr


def test_commit_allows_complete_and_bence_ok_preregistrations(box):
    box.append_log(prereg(exp="T2"))
    assert box.commit("add T2").returncode == 0
    box.append_log(prereg(exp="T3", serves="C3-S1", extra="Bence OK: 'do it'"))
    assert box.commit("add T3").returncode == 0


def test_commit_refuses_log_without_marker(box):
    text = (box.exp / "EXPERIMENT_LOG.md").read_text(encoding="utf-8").replace(MARKER, "")
    box.write("EXPERIMENT_LOG.md", text)
    res = box.commit("drop marker")
    assert res.returncode != 0 and "marker" in res.stderr


# --- Claude Code hooks ---------------------------------------------------------------------------


def test_pretool_blocks_script_outside_focus(box):
    box.write("z9_rogue.py", "print(1)\n")
    cmd = "cd cellware && python experiments/z9_rogue.py"
    out = box.hook("claude-pre", {"tool_name": "Bash", "tool_input": {"command": cmd}}).stdout
    assert '"deny"' in out and "z9_rogue" in out


def test_pretool_allows_script_named_in_current_preregistration(box):
    cmd = "cd cellware && python experiments/c9s2_confirm.py"
    out = box.hook("claude-pre", {"tool_name": "Bash", "tool_input": {"command": cmd}}).stdout
    assert out.strip() == ""


def test_pretool_blocks_new_experiment_script_without_preregistration(box):
    path = str(box.exp / "z9_new.py")
    out = box.hook("claude-pre", {"tool_name": "Write", "tool_input": {"file_path": path}}).stdout
    assert '"deny"' in out


def test_prompt_hook_injects_focus_and_rule_digest(box):
    out = box.hook("claude-prompt", {"prompt": "cellware status?"}).stdout
    ctx = json.loads(out)["hookSpecificOutput"]["additionalContext"]
    assert "C9-S2" in ctx and "H21" in ctx and "Self-challenge" in ctx
    assert box.hook("claude-prompt", {"prompt": "what is for dinner"}).stdout.strip() == ""


# --- the power rule ------------------------------------------------------------------------------


def test_power_known_values():
    z = power.z_crit(0.025, True)
    assert abs(z - 1.959964) < 1e-5
    assert abs(power.z_power(z, 1.0) - 0.5) < 1e-9
    assert abs(power.z_power(z + 0.841621, 1.0) - 0.8) < 1e-5
    assert abs(power.z_crit(0.05, False) - z) < 1e-9
    assert abs(power.contrast_se(2.0, {"a": 16, "b": 16}) - 2.0 * (2 / 16) ** 0.5) < 1e-12
    assert abs(power.contrast_se(1.0, {"a": 4}, {"a": 2.0}) - 1.0) < 1e-12
    assert power.rho_power(0.3, 3) == 0.0
    assert power.rho_power(0.3, 200) > power.rho_power(0.3, 50) > power.rho_power(0.1, 50)
    assert power.sim_power([3.0, 3.0, 0.0, 0.0], 1.0) == 0.5


@pytest.mark.parametrize("alpha", [0.0, 0.051, 0.5, -0.1])
def test_power_rejects_alpha_outside_range(alpha):
    with pytest.raises(ValueError):
        power.z_crit(alpha, True)


def test_power_gate_refuses_empty_group_and_bad_blocks():
    assert power.gate(selection(), {"a": 100, "b": 1})[1]
    assert power.gate({}, {"a": 100})[1] == "no power block (H14)"
    pw, msg = power.gate(selection(), {"a": 100}, strict=True)
    assert msg is None and pw > 0.99
    not_binary = selection(readout_gate={"pass": True, "binary_split": False})
    assert "why_not_binary" in power.gate(not_binary, {"a": 100}, strict=True)[1]


def test_existing_selection_files_still_pass_the_power_block_check():
    files = sorted((EXP / "selection").glob("*.json"))
    assert files
    for f in files:
        sel = json.loads(f.read_text(encoding="utf-8"))
        if "power" in sel:
            assert power.block_errors(sel, strict=False) == [], f.name


def test_new_scripts_take_power_from_power_py():
    own = re.compile(r"norm\.(ppf|cdf|sf)\(|NormalDist")
    offenders = sorted(
        p.name
        for p in EXP.glob("*.py")
        if p.name not in LEGACY_POWER | {"power.py"}
        and own.search(p.read_text(encoding="utf-8", errors="replace"))
    )
    assert offenders == [], (
        f"use experiments/power.py instead of a private power formula: {offenders}"
    )


# --- the rule text itself ------------------------------------------------------------------------


def test_every_rule_is_in_program_and_digest():
    program = (EXP / "program.md").read_text(encoding="utf-8")
    digest = (EXP / "RULES_DIGEST.md").read_text(encoding="utf-8")
    for n in range(1, 22):
        assert re.search(rf"\*\*H{n} ", program), f"H{n} missing from program.md"
        assert re.search(rf"\bH{n}\b", digest), f"H{n} missing from RULES_DIGEST.md"
    assert len(digest) < 4000, "the digest is injected every turn; keep it short"
    for heading in ("## Practice checklist", "## Result reports: self-challenge block"):
        assert heading in program


def test_repo_wiring():
    hook = (REPO / ".githooks" / "commit-msg").read_text(encoding="utf-8")
    assert "scope_guard.py commit-msg" in hook
    assert scope_guard.MARKER in (EXP / "EXPERIMENT_LOG.md").read_text(encoding="utf-8")
    assert scope_guard.PRACTICE_FROM == power.STRICT_FROM
    assert os.path.exists(scope_guard.DIGEST)


# --- the real experiment code, checked statically (no data directory needed) ---------------------


def test_real_experiment_code_passes_the_static_sealed_check(tmp_path):
    """No script outside the approved readers may hold a sealed-reader call (H2, H3, H11).
    A withdrawn test loses its reader functions with its HELDOUT_READERS entry (C9-S1)."""
    import check_sealed

    problems = check_sealed.check(py_dir=str(EXP), data_dir=str(tmp_path))
    assert problems == [], "\n".join(problems)
