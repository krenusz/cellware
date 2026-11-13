# Reviewer replay benchmark: results

## Run 1 (brief v1, 2026-11-13): FAIL, stopped early

Isolation canary (K13, live case): the reviewer saw no project context beyond the working-directory path
(`...\cellware\experiments\outreach`), read every packet file in full, and produced a substantive review (verdict
block on power and spent results). `files_read` is self-reported.

| case | kind | verdict | blocking checks | scored | note |
|---|---|---|---|---|---|
| K01 G11 | fault | pass | - | **miss** | flagged "no power calculation" as a note only |
| K03 C4-S2b | fault | pass | - | **miss** | flagged "2-6 units, no inference" as notes; packet had no script |
| K04 C5-S4 | fault | block | power, unit, spent-results, script | **catch** | named the known defect: sd not from the exact S-ratio statistic |
| K06 C6-S7g | fault | pass | - | **miss** | flagged "7th sub-version" and "6 reps" as notes |
| K07 C6-S8a | fault | pass | - | **miss** | flagged the exact defect (one rate where sealed G6 gave clone-specific rates) as a note |
| K08 C8-S1b T2 | fault | block | power, spent-results | **catch** | power fragile, 0.46 if shrunk; did not name the subset-ratio defect |
| K02 G7 | fault | pass | - | **miss** | flagged "spends all remaining hold-out" and "single rate rejected by spent G6" as notes |
| K09, K16 | fault | not run | | | stopped: rule unreachable |
| K11, K12, K17, K15, K19 | sound | not run | | | stopped |

**Score:** 2 of 7 scored fault cases caught; at most 4 of 9 reachable, so the 7 of 9 rule cannot be met. Stopped
under the early-stop rule; no sound case was run.

**Diagnosis:** detection worked: in all four misses the known defect appears in the findings, with evidence. The
verdict threshold failed: brief v1 blocks only for a defect that would waste a sealed partition or break a rule in
*the packet's* program.md. Steps with no sealed spend (K06, K07) and steps from before a rule existed (K01, pre-H14)
were therefore passed with the defect as a note.

## Revision (brief v2, the one revision BENCHMARK.md allows)
See REVIEWER_BRIEF.md "Verdict" (v2). Run 2 reruns all 14 scored cases on the same packets and the same prompt
template; both runs are reported here.
