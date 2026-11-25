# Full retro 2026-11-18: chapters 10-16, novelty audit, lesson landing (since the 2026-11-13 sweep)

Bence: "do a full retro!! lot to injest". Scope: the 8 retros since RETRO_SWEEP_2026-11-13 (RETRO_2026-11-13 to RETRO_2026-11-17c, 470 lines), chapters 10-16, the new rules H22-H25, a novelty audit of every confirmed or exploratory result under the H25 lens, the lesson-landing audit, the model ledger, the sealed and exposure inventory, the watch list and the gap map. No data was opened. Cellware stays PAUSED. All work by Opus 5.5.

## 1. Scorecard, chapters 10-16

| ch | question | ended | sealed result | already published? (H25 lens) |
|---|---|---|---|---|
| 10 | ReSisTrace: instruction vs selection under treatment | primary stopped at S0 (library-level treatment, 2 per arm; H22 born) | rescue CONFIRMED on resistrace:A (state beyond subclone + QC predicts survival, delta AUC 0.062) | **partly:** the source paper (Dai 2024) reports pre-resistant signatures in the same data; ours adds the increment beyond a subclone proxy, tested held-out |
| 11 | SG_kit1 gate | failed on published design counts in one step (H22; H20 already published) | none | **published** (gate said so) |
| 12 | expansion-culture lineage set: state beyond size predicts in vivo presence | paused after S1 (train dAUC 0.17); single-partition power too low; pooled power provisional (z_power, needs sim_power) | none; A, B, C sealed | **partly:** overlaps the source paper's own association (noted 11-14b, never resolved) |
| 13 | spatial morphogenesis on embryo trees | closed on a data fact (spatial scores are inferred, not measured) | none | n/a |
| 14 | state beyond cell type predicts clade fate in held-out embryos | CONFIRMED on petracer_embryo:A (z 3.0, mean delta AUC 0.084, 4/5); survives region scores on train (0.82); origin program (S4); memory-consistent core vs edge (S5) | A spent; B (one usable embryo) and C sealed | **partly:** the source paper (Colgan 2026, claims read 2026-11-18) reports gene regulatory programs underlying fate specification and cell types that keep "traces of origin" when converging. Its held-out predictive test is not in the claims |
| 15 | carry C14 to an in vivo growth model (#12, 32 hosts) | closed underpowered; S1b INVALID by smoke design | none; growth12_host_v2 sealed | not checked (no claim) |
| 16 | hidden inherited state for P vs L on embryo trees | closed exploratory (B one usable embryo) | none | **largely published** (Colgan 2026, Liu 2026); new only as method |

**Confirmed since the sweep:** 2 looks (C10-S2, C14-S2), both by Opus, both one look, no re-run. Total looks to date: 22.

## 2. Patterns

### 2.1 Source-paper pre-emption (new, the main finding)
In **four of the five data chapters** (10, 12, 14, 16) the data set's own source paper already reported our headline in kind. Cause:
- We pick data sets because a paper analysed them, so that paper is the most likely to have asked our question.
- Our sealing rule kept its results unread, so the most likely pre-emption was the one we never checked.
- H20 required the "top 5 papers" but the source paper was exempt from reading in practice.

Landed as **H25** (2026-11-18, two commits): a novelty file per new step, and the source paper read at claims level.

**Consequence for choosing work:** what is ours is what the source paper structurally cannot do:
- held-out prediction across independent units (embryos, hosts, libraries);
- a fitted mechanistic model compared against alternatives;
- transfer of one result across data sets.

Description of the same data set (timing, programs, biases) is the authors' ground.

### 2.2 Data-limited by independent units (continues)
- C10 (2 libraries per arm), C12 (single partitions underpowered), C15 (5 hosts per partition), C16 and C14-B (one usable embryo).
- Powered results all used many lineages within a unit, with no between-condition contrast (C7, C8, C10-S2, C14-S2).
- Whole-embryo sets give about 3 embryos per partition, and stage heterogeneity cuts that to 1 usable. H24 now balances by structure; it cannot create units.

### 2.3 Bence still catches the key failure
Since the sweep, at least 5 key catches came from Bence, not from my analysis:
- "this is why we have the rule not to test thing we end up data limitation" (C10);
- "did we do retro after c14??" (H23);
- "but i meant to use the 4 core of the action!" (C16-S1);
- "why dont you want to test claims on holdout?" (C16-S4);
- the C16 novelty outburst (H25).

The self-challenge block asks "already published?" at the result, which is too late. H25 moves it to step choice.

### 2.4 Rules work when they are code
- H22, H23 and H25 are enforced by the commit hook; H14 by the oracle.
- Lessons left as text (practice lines, retro items) are the ones that slip.
- Pending items pile up (section 4).

## 3. What worked (keep)
1. **Gates fire before spending:** SG_kit1 in one step, C9-S3, C12 (power), C15 (power), C16-B (oracle n >= 2). Nothing sealed was spent on an unpowered look.
2. **Hold-out integrity:** no sealed file read outside the oracle; every look committed first; nothing re-run.
3. **Actions as default compute** after two misses; the 4-process pool per job cut C16-S1 from about 3 h to 25 min.
4. **Synthetic path tests** caught design errors before real data in C14, C15 and four times in C16.
5. **Same-session retros** (H23) at every close since C14.

## 4. Lesson-landing audit: what is still open
| item | source | status | recommendation |
|---|---|---|---|
| k cap <= n_units / 5 for regressions on learned features | RETRO_2026-11-13 | pending Bence since 11-14 | adopt as practice |
| Library placebos matched for unit count | RETRO_2026-11-13 | pending Bence | adopt as practice |
| Power for a sealed look from the out-of-sample effect on spent partitions when any exist | RETRO_2026-11-14 | pending Bence | adopt as practice |
| Data search scored by the shape of past successes (lineage unit, >= ~100 units per partition, state before fate, public) | RETRO_2026-11-14 | pending Bence | adopt as practice; add "source paper has not answered it" (H25) |
| Guard hook for the `Compute:` line | RETRO_2026-11-15 | pending Bence | adopt (cheap, code) |
| scope_guard `allowed_scripts` compares bare names to `.py` entries, so the allow-list never matches | RETRO_2026-11-17 | pending Bence | fix (one line, plus a test) |
| `drug_resistance` topic in lit_gap.py | RETRO_2026-11-13 | owed before any compound chapter | keep owed |
| C12 power redo by `sim_power` | RETRO_2026-11-14b | owed before any C12 look | keep owed |
| DISCOVERIES.md has no entries for C7, C8, C12 or C14 | found today | the register lags 4 chapters | C14 entry added today (#31); C7, C8, C12 next |
| Five open decisions from the 11-13 sweep (core-model LARRY refit, tissue simulator, one analysis script, Marangoni preprint, Cell Rep 2022 paper) | RETRO_SWEEP_2026-11-13 s.4 | never closed | drop all five: their chapters are closed |
| Dataset-specific and sensitive wording in files written before 2026-11-14 (e.g. the DISCOVERIES chapter 10 heading) | RETRO_2026-11-14b | Bence's call | rewrite neutrally (text only; touches no seal) |
| Public Actions log of run 44073504820 (two gene names and a tool name) | RETRO_2026-11-17b | Bence's call | delete the log |

**Decided 2026-11-18 (Bence: "Adopt all seven"):**
1. k cap and 2. count-matched placebos: practice lines, "Before a statistic".
3. Out-of-sample power from spent partitions: practice line, "Before a power claim".
4. Data search by shape of past successes plus H25: practice line, "When searching for data".
5. `Compute:` line: enforced by the commit hook for pre-registrations dated 2026-11-18 or later (scope_guard.compute_error; test added).
6. Allow-list bug: fixed (scope_guard.allowed_script strips `.py`; test added).
7. Housekeeping:
   - the five open decisions from RETRO_SWEEP_2026-11-13 s.4 are **dropped** (their chapters are closed);
   - prose files rewritten in neutral wording (15 files: line, condition and agent names replaced; treatment arms now T1, T2, NK). Code keeps its data keys and the mask list; run logs and generated gap maps (public paper titles) are unchanged; git history still holds the old text;
   - the public log of run 44073504820 is deleted (API 404 since). Its two artifacts (`results`, `line-1`) were deleted on Bence's OK the same day ("Yes, delete them"); the run now has 0 artifacts.

## 5. Model ledger (since 2026-11-13)
- 762 Opus, 35 Sonnet, 45 Fable rows. 0 RISK rows.
- 15 CHECK rows:
  - **Fable, 11-13 (5 rows):** the C9-S1 withdrawal, i.e. removing its sealed-reader functions and the program split. Reviewed today for the first time: no look, no sealed read; the oracle ledger has no 11-13 look except C10-S2 (Opus).
  - **Sonnet, 11-14 (7 rows):** C12-S1 selection and tests; reviewed in RETRO_2026-11-14b.
  - **Sonnet, 11-15 (3 rows):** local C14-S0/S1 runs; reviewed in RETRO_2026-11-16.
- Since 11-16, all rows are Opus: the Sonnet data lane is not offered in these sessions.

## 6. Sealed and exposure inventory
**Sealed, unspent:**
- celltag_lsk:C and celltag_reprog:C (last partitions);
- chspctrack:B, C;
- resistrace:B, C;
- lsc_larry:A, B, C;
- petracer_embryo:B (one usable embryo), C;
- growth12_host_v2:A, B, C;
- plus the older reserves listed in RETRO_2026-11-14.

None has a powered pre-registered test today.

**Accepted exposure:**
- C10: per-sample pre-resistant counts (2026-11-13).
- C12: the exposure statement in every report.
- petracer_embryo:
  - snippet-level aggregate results of Colgan 2026 (2026-11-17);
  - its claims level, read for this retro under H25: abstract, Results headings, figure titles and Discussion conclusions; no numbers.
  - Any later look on petracer_embryo B or C states both.

## 7. Watch list (re-checked today, due about 21 Nov)
- GSE324482: not public (0 GEO hits).
- E-MTAB-17351: still 403.
- hESC endoderm clonal memory (Linneberg-Agerholm 2026, bioRxiv 10.64898/2026.09.02.748863): no data statement found in snippets.
- "DeepTrack 2026": no match under that name. The watch entry needs its real title or DOI, otherwise drop it.
- PEtracer GSE342607: public, in use.

## 8. Literature
**Gap map today:** Actions run 44291080990, `results_c16/gap_2026-11-18_full.json`.
- 66 seeds, 5,747 works relevant to at least two open questions; no new works since yesterday's C16 map.
- Top 40 unread: 36 open access, 4 paywalled (The Company of Biologists 3, 1 other). No subscription is justified.
- **Only 26 of the 5,747 relevant works are marked read.** The map keeps ranking; the reading has not kept up. H25 should change that, because each new step now forces a per-finding search.
- To read before any next embryo or mesoderm chapter (all open access, titles only so far):
  - #10, whole-embryo barcoding, convergent mesoderm origins (bioRxiv 2026.11.11);
  - #7, human neuromesoderm organoid lineage tracing (bioRxiv 2026.07.29);
  - #1, size-dependent decoupling of morphogenesis and transcription in pseudoembryos (Sci Adv 2025): the morphogenesis goal.
- #3 (Development 2026, a preprint highlight on mammalian embryo lineage maps) is paywalled and probably discusses the source paper; read at claims level only if needed.

**Source paper claims (H25):** Colgan 2026, read at claims level today (`lit/source_claims_petracer_2026-11-18.md`).

## 9. How the project served the long-term goal
- **Long term:** model stem-cell fate behaviour, and eventually morphogenesis.
- **Fate behaviour:** done in culture (C1-C3: instruction vs selection, fate memory) and in several systems as "state beyond type carries lineage fate" (C7, C8, C10, C14 confirmed). The C14 and C16 embryo work is partly pre-empted by the source paper. What remains ours is held-out prediction and the fitted model.
- **Morphogenesis:** still blocked by data (no public whole-embryo lineage with measured position). C13 closed on that fact.
- **The honest arc:** the method (sealed partitions, oracle, power gates, now novelty gate) is the most durable product. The biology replicates and quantifies more often than it discovers.

## 10. Lessons from this retro, each landed
| # | Lesson | Landing |
|---|---|---|
| 1 | The source paper is the most likely pre-emption; check it at claims level before any step | -> H25 (done 2026-11-18, two commits) |
| 2 | Choose questions the source paper cannot answer by design: held-out prediction across units, model comparison, cross-data-set transfer | -> H25 template ("What is left that is ours", done); practice line added today |
| 3 | The discoveries register is updated at every chapter close, with its novelty verdict | -> practice line added today; C14 entry added (#31) |
| 4 | Pending lessons older than 3 days go to Bence as one decision list at the next retro, not one per message | -> practice line added today; section 4 is the first such list |
| 5 | Non-Opus ledger rows are reviewed in the next retro even when another retro intervenes | -> not adopted as new: already a retro step; the 11-13 Fable rows slipped and are reviewed here |
| 6 | Whole-embryo sets: count usable units per partition (units that carry the statistic) at sealing, not at the look | -> not adopted as new: practice line from RETRO_2026-11-17c covers it |

## 11. Status
- Cellware PAUSED.
- Decisions for Bence: section 4 (items pending him), and the direction of the next chapter under H25.
