# Retro sweep: all 16 retros read end to end (2026-11-13)

Bence: "do a sweep ... on all of the retros". Scope: RETRO_2026-11-03 to RETRO_2026-11-11c_EXTENDED (16 files,
1,219 lines), checked against program.md (H1-H19) and the DISCOVERIES M-lessons. No data was opened. Cellware
stays PAUSED; nothing here changes a rule. Section 5 lists proposals for Bence.

## 1. The arc in four phases

| phase | retros | what kept going wrong | what fixed it |
|---|---|---|---|
| Model churn | 11-03 to 11-05b | new simulators instead of diagnostics; scoring rules that rewarded noise | stopping rule; fit-set scoring; data diagnostics first |
| Integrity | 11-06 to 11-07b | hold-out read outside the oracle; partition spent without power; wrong unit | H1-H15, enforced in code |
| Feasibility | 11-07c to 11-11 | chapters 4-7 built first, found unpowerable later | H17, H18, H19 |
| Value | 11-11b, 11-11c | confirmed results that the literature already had | **not fixed** (H20 proposed only) |

## 2. Patterns that recur across retros

1. **Bence caught it, not me.** In 9 of 16 retros the key failure was found by Bence's challenge:
   - "this is suspicious" (B7 audit);
   - the two scoring rules (B29, E47);
   - the goal question that produced the stopping rule;
   - "check the full 120GB";
   - "are you sure there is nothing else?";
   - "feels weak";
   - used data first (DiffAssay);
   - the low-power overstatement (C4-S2b);
   - the kill-rule rationale.

   The rules that followed are all reactive. None of them makes me ask those questions unprompted.
2. **Data diagnostics beat model iteration, every time it was compared** (11-03, 11-04, 11-04b, 11-05, 11-05b,
   11-10). Yet chapter 6 ran 13 model versions in one step. The lesson was written five times before H18 made
   it binding.
3. **Literature read after building.**
   - 11-05b: the paper's design contradicted a structure we had carried for a day.
   - 11-06: the 40% editing efficiency was in the paper.
   - 11-07b: an open-access paper was called unread.
   - 11-09: the closest published model, with code, was in our own citation map.
   - 11-11c: novelty was never checked at chapter choice.

   Five retros, one cause: the reading step is not gated.
4. **Power computed from the wrong thing**, in six variants:
   - no power at all (G11);
   - assumed effect sizes instead of the model's own (ch6);
   - a plug-in sd (C5-S4);
   - the dataset-wide ratio for a subset (T2);
   - one memory rate where a spent result gave two (C6-S8a);
   - a degenerate outcome (C9).

   Each got its own patch. The extended retro already says to merge them.
5. **Small-n statistics treated as evidence.** 8 informative clones (11-04), one clone (B26), 2-4 gastruloids
   (C4-S2b), 5 clones reused four times (ch6), 7 GM clones (C9).
6. **Tooling lessons repeat because they live only in retros or memory:** smoke tests touching real output
   (11-04b), shell quoting (11-05), heredoc loss (ch7), `\r` in plans (11-10b), watcher never fired (11-10).

## 3. Lessons stated in a retro but in no rule (program.md or M-list)

Checked by search of program.md. "Standing practice" in a retro is not enforced anywhere.

| lesson | source retro | where it lives now |
|---|---|---|
| Adequacy gate before any cross reading | 11-03 | retro only |
| Check how many units drive a statistic before using it as a gate | 11-04 | partly in H14; readout gate proposed |
| Leave uninformative statistics out of any summed score | 11-05 | retro only |
| Cold starts after a structural model change | 11-04 | retro only |
| Smoke tests use synthetic outcomes and write to the scratchpad | 11-04b | retro only |
| Probe whether a fit can stay at the favourable point before registering | 11-05b | retro only |
| Rank candidates against stated criteria before calling one best | 11-08 | retro only |
| Simulated readouts copy the measurement physics | 11-09 | retro only |
| Check a known model's analytic regime before choosing parameters | 11-09 | retro only |
| Extend seeds and read the top 5 papers before code, at chapter opening | 11-09 | retro only |
| Step 0 hygiene: >= 10 reps, settling check, float tolerance | 11-10 | retro only |
| List which spent results constrain each power input | 11-10b | retro only |
| Read the methods definition of every readout before proposing it | 11-10b | retro only |
| Fate-free validity gate before any fate statistic on a new featurisation | 11-11 | retro only |
| Pre-registration tick-list against the script | 11-11 | retro only |
| Re-check the private accessions at every retro | 11-10b, 11-11 | FOCUS.json watch list |
| Readout gate (>= 30 per side; statistic varies with the model) | 11-11c | proposed, not adopted |
| H20 value gate | 11-11c | proposed, not adopted |
| Merge H14 / H17 / H19.2 into one power rule | 11-11c | proposed, not adopted |

19 lessons, 0 enforced. This is the main finding of the sweep.

## 4. Decisions made in retros that were never carried out

| decision | retro | status |
|---|---|---|
| Shared core model: LARRY and SIS as two configurations of one code base | 11-06 | `core_model.py` was built (11-06) and used in chapters 3 and 5. The log says the LARRY refit through it was "still to do"; no later entry found. **Open** |
| Reconnect the 2023-24 tissue simulator as the spatial layer | 11-06 | not done; chapter 6 built new simulators |
| One parametrised analysis script; fix the launcher exit code | 11-05b | not done: `analyze_e43.py` to `analyze_e50.py` are still separate copies. **Open** (moot unless LARRY modelling resumes) |
| Read the Marangoni preprint before any further model design | 11-10 | not read: it appears in the gap map and the log only as a to-do. **Open** (moot while chapter 6 is closed) |
| Read the Cell Rep 2022 chromatin-priming paper before C7-S4 | 11-11 | no reading note found in the log or `lit/`. **Open** |
| Epo presence test with all cell types | 11-05b | became discovery #12 (done) |

## 5. Proposals (Bence decides; nothing adopted)

1. **One "practice" block in program.md** that holds the 16 retro-only lessons as a checklist, in three groups:
   before a model run, before a statistic, before a chapter. That is cheaper than 16 new H-rules.
2. **Adopt the three open proposals from 11-11c:** readout gate, H20 value gate, merged power rule.
3. **A literature gate at chapter opening**, enforced like pre-flight:
   - the source paper's methods read;
   - the top 5 map papers read;
   - the minimal published mechanism stated;
   - novelty stated.

   This targets pattern 3, the most repeated failure without a rule.
4. **A self-challenge block in every result report**, to target pattern 1. It uses the questions Bence keeps
   asking:
   - is this suspicious?
   - what else could explain it?
   - how many units drive it?
   - can used data answer it?
5. **Close or drop the open decisions in section 4** at resume, so they stop being carried.

## 6. What the retros got right

- Every retro after 11-06 states what was sealed and untouched. The count of sealed looks matches the
  ledger (20, none re-run).
- Withdrawals are recorded as withdrawals: E26-E29, #16, C4-S2b, C9-S1, and the "confirmed" overclaim in the
  novelty map.
- The retro cadence itself worked: every hard rule (H11-H19) came out of a retro within a day of the failure.
