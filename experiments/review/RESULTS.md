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

**Diagnosis:** detection worked: in all five misses the known defect appears in the findings, with evidence. The
verdict threshold failed: brief v1 blocks only for a defect that would waste a sealed partition or break a rule in
*the packet's* program.md. Steps with no sealed spend (K06, K07) and steps from before a rule existed (K01, pre-H14)
were therefore passed with the defect as a note.

## Revision (brief v2, the one revision BENCHMARK.md allows)
See REVIEWER_BRIEF.md "Verdict" (v2). Run 2 reruns all 14 scored cases on the same packets and the same prompt
template; both runs are reported here.

## Run 2 (brief v2, 2026-11-13): FAIL (sensitivity fixed, specificity broken)

All 14 scored cases ran. Every reviewer reported seeing only the outreach working-directory path (no focus line,
digest or memory) and reading every packet file in full; no void cases. Condensed verdicts:
`results/run2/verdicts.json`.

| case | kind | verdict | blocking checks | scored | note |
|---|---|---|---|---|---|
| K01 G11 | fault | block | power, readout | **catch** | "No power is computed for the primary test" |
| K02 G7 | fault | block | power, readout | **miss** | blocked, but not under reserve / spent-results; the spent-G6 clone 5 contradiction was a note |
| K03 C4-S2b | fault | block | readout, power | **catch** | 2-6 gastruloids per condition |
| K04 C5-S4 | fault | block | power, readout, spent-results | **catch** | fade input contradicted by spent #23 |
| K06 C6-S7g | fault | block | amendments, power, spent-results | **catch** | "7th lettered version (a-g)", no retro |
| K07 C6-S8a | fault | block | spent-results, power | **catch** | single retention 0.46 vs #23; C4-S4 overprediction |
| K08 C8-S1b T2 | fault | block | power | **catch** | packet's own 0.46-if-shrunk figure |
| K09 C9-S1 | fault | block | readout, power | **catch** | MegE share 0.97, GM units not gated |
| K16 C7-S3 | fault | block | power, script, readout | **catch** | new ATAC readout, no validity check |
| K11 C7-S7 | sound | block | power | **false block** | ~0.68 under a 30% shrink that spent #17 implies; joint power of 4 tests |
| K12 C8-S1c | sound | pass | - | ok | power holds to rho 0.27 |
| K17 C8-S1a T1 | sound | pass | - | ok | power holds to 46% of the train effect |
| K15 C7-S6 | sound | block | power | **false block** | train-only descriptive step, no look |
| K19 C7-S5 | sound | block | power, readout | **false block** | train-only step, "No look in this step" |

**Score:** 8 of 9 faults caught (rule: >= 7), 3 of 5 sound cases blocked (rule: <= 1). **FAIL.** The one revision
is used, so the reviewer stays advisory (`STATUS.json` `"enforced": false`) and nothing is wired into the oracle or
the commit hook.

**Diagnosis:** v2 swung from too lenient to too strict. Two of the three false blocks (K15, K19) come from block
rule 1 ("no power computed") firing on train-only steps that spend nothing and claim nothing; the brief does not
scope the power rule to looks. K11 is a closer call: the shrink argument is the same one that correctly caught K08,
and C7-S7 happened to confirm. A reviewer that blocks 9 of 14 steps would stop most work, so as a gate it fails.

**What it is still good for (advisory):** on fault cases it named the retro defect in 8 of 9, usually with the
packet's own line as evidence. Used as a pre-look checklist, its block findings are worth reading; it is not a
pass/fail gate.

**Not adopted now:** a v3 that scopes the power rule to steps that spend sealed data or make a confirmatory claim.
Testing it on these same 14 cases would be tuning on the answer key; it needs fresh cases (new steps after Cellware
resumes) before it could gate anything.
