# Reviewer replay benchmark (fixed 2026-11-13, before any reviewer run)

Bence: "go with your recommended actions and wire it fully". Purpose: measure whether an independent
reviewer subagent catches the failures our retros found, using only what was on disk at the time.

## Design
- **Packets:** `build_packets.py` rebuilds each case from git: the pre-registration section(s) as they stood at
  the listed commit, the scripts it names (<= 450 lines), its selection file, and DISCOVERIES.md and program.md at
  that commit. Text from git only; no data file. Cases are in `cases.json`.
- **Reviewer:** a fresh read-only subagent (Explore type, model sonnet: a different model from the author, for
  independence and cost) with `REVIEWER_BRIEF.md` and one packet directory. It must read nothing outside the packet
  and list every file it read; a case where it read outside the packet is void.
- **Hindsight limit, disclosed:** the brief's checks were written from these same failures, so the benchmark
  measures whether the reviewer *applies* the checks, not whether it finds new failure classes.

## Answer key (written before any run)
A **catch** = verdict `block` **and** at least one blocking finding in the listed class.

| case | step | known defect (from the retros) | class |
|---|---|---|---|
| K01 | G11 | no power computed before spending drug partition A; minimum 4 per condition (~15% power) | power |
| K02 | G7 | spends all remaining same-design hold-out in one round; memory rule already undermined by spent G6 | reserve / spent-results |
| K03 | C4-S2b | ordering reading from 2-6 gastruloids per condition; no chance of the pattern computed; set spent | small n / power |
| K04 | C5-S4 | power sd from D_btw log ratios, not the S-ratio test statistic; 7 clones | power from wrong quantity |
| K06 | C6-S7g | 7th model version within one step (forking path); 6 reps per step-0 point | amendments / step-0 hygiene |
| K07 | C6-S8a | one retention rate (0.46) where spent G6a gave clone-specific rates | spent-results constraint |
| K08 | C8-S1b | subset (exp 7) power from a dataset-wide ratio (struct_ratio 1.0); plan power 0.86 marginal, 0.46 if shrunk | subset power |
| K09 | C9-S1 | degenerate outcome (mean MegE share 0.97; few units carry the statistic) | readout gate |
| K16 | C7-S3 | new ATAC featurisation with no fate-free validity check; chromatin-vs-RNA delta unpowerable from paired counts | validity / power |

Sound cases (no retro-identified defect; looks ran and confirmed, or train-only steps that held):
K11 C7-S7, K12 C8-S1c, K17 C8-S1a, K18 G6, K15 C7-S6. A **false block** = verdict `block` on a sound case.

Live case (unscored): K13 C9-S2, the ready LSK chromatin test. Its review is reported to Bence as information.

## Acceptance rule (fixed now)
- **Pass:** at least **7 of 9** fault cases caught, and at most **1 of 5** sound cases blocked.
- Pass -> `STATUS.json` set to `"enforced": true`; the oracle and commit hook then require a passing review.
- Fail -> the reviewer stays advisory (`"enforced": false`); the brief may be revised once and the benchmark
  re-run on the same cases with the result of both runs reported.
- Scoring of each case is written into `RESULTS.md` with the reviewer's findings quoted.
