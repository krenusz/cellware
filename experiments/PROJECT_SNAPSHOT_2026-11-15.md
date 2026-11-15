# Cellware: project snapshot, 2026-11-15

Asked by Bence (2026-11-15): "make a full project document where you explain our ongoing project as current snapshots claims already collected data already tested and showing also the negatives, like the failed exps".

Written from the repository files (FOCUS.json, test_ledger.tsv, EXPERIMENT_LOG.md, the RETRO_* files, WRITEUP_* files, DISCOVERIES.md, selection/*.json, program.md, partitions.py). Numbers are copied from those files, not recalled. Dataset wording is neutral by Bence's rule of 2026-11-14: two datasets are called "treatment-response lineage set (C10)" and "expansion-culture lineage set (C12)", and older files that use other wording were not copied. Status: PAUSED. Nothing in this file is a new result.

---

## 1. What Cellware is

**Question.** How do stem cells decide their fate, and eventually how do cells that share local rules build a patterned tissue (embryo morphogenesis)? The long-term goal was set by Bence on 2026-11-05. It is tested on public lineage-tracing data, where cells carry heritable barcodes so that relatives (clones, sister cells, lineages) can be compared.

**Method: a self-correcting autoresearch loop.**
- An AI agent (Claude Code) runs the analysis. Bence sets the questions, picks chapters and approves anything that spends held-out data. A cheaper model (Sonnet 5.5) handles steps that touch a chapter's biology data, and the larger model keeps planning, rules, review and commits (program.md, "Model routing").
- Every hypothesis and prediction is written in EXPERIMENT_LOG.md before the run (pre-registration), every run is logged including failures, and every claim traces to a commit.
- **Hold-out discipline.** Each dataset is split before any statistic by a salted hash of its biological unit: train 40%, and sealed partitions A, B and C at 20% each. Sealed data is read only through a one-look oracle (`test_oracle.py`) that refuses a look without a pre-registration, a train-chosen model, a power check of at least 0.8, a `Pre-flight:` block, a `Serves:` line, and a `Practice:` line. A spent partition is never reused for confirmation.
- **Retros turn failures into rules.** Every retro must land each lesson as an H-rule, a practice-checklist line, a memory item, or an explicit "not adopted".

**Counts (verified 2026-11-15).**

| item | count | source |
|---|---|---|
| commits on the repo | 662 (first 2023-09-07; latest 24daa14, 2026-11-15) | `git log --oneline`; the 2023-24 commits are the earlier tissue-simulator setup, the autoresearch program starts 2026-10-31 (first ledger entry) |
| hold-out and process rules | 22 (H1-H22) | program.md |
| practice-checklist lines | 16 | program.md |
| methodological lessons in the discoveries register | 13 (M1-M13) | DISCOVERIES.md |
| retros | 19 dated retros plus 1 all-retro sweep (20 `RETRO_*.md` files) | experiments/ |
| guard tests | 45 tests collected by pytest (some parametrised), run before every push | tests/test_guards.py |
| sealed looks (oracle ledger rows) | 21 rows in test_ledger.tsv (the first row records four looks) | test_ledger.tsv |
| discoveries register entries | numbered 1-29 | DISCOVERIES.md |
| chapters | 12 opened (C1-C10, C12 worked; C11 stopped at a one-step gate) | FOCUS.json, log |

Of the 21 ledger rows, 6 are the recent powered looks (C7-S7a/b/c, C8-S1a/c, C10-S2) with 8 confirmatory statistics, all confirmed. The rest are earlier looks from 2026-10-31 to 2026-11-07, with mixed outcomes listed in section 3.2 and section 5.

---

## 2. Current snapshot (2026-11-15)

**State: PAUSED since 2026-11-14.** Bence: "keep sealed, pause cellware" (FOCUS step `PAUSED-2026-11-14b`). An earlier pause was set on 2026-11-11 ("leave all sealed, and pause cellware"). Nothing sealed was spent after chapter 8 except C10-S2 on 2026-11-13.

**Why paused.** Cellware's tests work when the unit is a clone or lineage and there are about 100 or more units per sealed partition. They fail when the question needs replicated conditions (libraries, wells, plates, gastruloids). The last chapter opened (C12) found a large effect on train, but no single sealed partition has the power to test it (section 3.3). Spending sealed data on an under-powered look would waste the partition, so the work stopped until Bence decides.

**Sealed and unspent right now** (none has a powered pre-registered test today; they are reserves for the reopen conditions in section 7):
- celltag_lsk: C (the last partition).
- celltag_reprog: C (last).
- chspctrack: B and C.
- C10 set: B and C.
- C12 set: A, B and C.
- Gastruloid reserves: signalling-perturbation partition C, mixing-rescue partition C (C4), ATAC partitions A and B (C5), the C5-S3 reserve, the C6 sets S1 and S2:C, DiffAssay replicate 2.
- Locked files: F8MWB5, revision-era chimeras, the Triandafillou atlas.

**Decisions pending with Bence.**
1. **C12 pooled look.** A single look on pooled partitions A+B for the D20 in vivo readout would leave C as the last reserve (H13 needs his verbatim OK). The power quoted for it (0.85) is provisional and probably optimistic (it must be redone with `sim_power`, see 3.3). Alternatively stay paused.
2. **Retro lines still pending his OK** (RETRO_2026-11-14.md): out-of-sample effect from spent partitions as the power input; a data-search scoring rule by the shape of past successes; a weekly cap on re-checking the unchanged watch list. From RETRO_2026-11-13.md: cap k at no more than n_units/5, and unit-count-matched library placebos. The C12 retro lines were adopted on 2026-11-14 ("1").
3. **Exposure statements.** C10 exposure was accepted by Bence on 2026-11-13 ("exposure is acceptable if we keep in mind always we use this"); every C10 report states it. Every C12 report must state its own exposure (3.3).
4. **Existing files with older dataset wording.** Files written before the neutral-wording rule still contain dataset-specific terms; nothing was rewritten and that is Bence's call (RETRO_2026-11-14b).
5. **Watch list re-check about 2026-11-21:** a human embryonic-cell endoderm sister-well multiome release, GSE324482, E-MTAB-17351, DeepTrack 2026, and the PEtracer embryo set GSE342607. On 2026-11-14 nothing was public; GSE324482 was private and E-MTAB-17351 returned 403.

---

## 3. Claims

### 3.1 Rules for reading this section

H8: "moderate" or "strong" is used only for oracle-confirmed results. Open-data and train-only results are "unconfirmed" or "exploratory". Spent partitions never count as confirmation again (H7).

### 3.2 Oracle-confirmed claims (sealed, one look each)

| # | claim | statistic | n | partition | date | commit |
|---|---|---|---|---|---|---|
| C7-a | A clone's fate state is heritable in blood-progenitor culture (sister-well agreement) | H 0.75, 95% CI 0.65-0.82, one-sided lower bound 0.66 | 158 clones with 2 or more cells | celltag_lsk:A | 2026-11-11 | 5e69507 |
| C7-b | Early RNA carries clone fate, blood-progenitor system | G_rna 0.50, lower bound 0.33 | 196 | celltag_lsk:A | 2026-11-11 | 5e69507 |
| C7-c | Early chromatin carries clone fate in the same system, but less | G_atac 0.24, lower bound 0.097 | 137 | celltag_lsk:A | 2026-11-11 | 5e69507 |
| C7-d | Early RNA carries clone fate, reprogramming system | G_rna 0.35, lower bound 0.21 | 113 | celltag_reprog:A | 2026-11-11 | 5e69507 |
| C7-e | Early RNA carries clone fate, human progenitor clones | G_rna 0.40, lower bound 0.13 | 122 | chspctrack:A | 2026-11-11 | 5e69507 |
| C8-T1 | A clone's day-2.5 RNA predicts its specific neutrophil-vs-monocyte split on unseen clones | Spearman rho 0.45, lower bound 0.32 (train 0.46) | 153 | celltag_lsk:B | 2026-11-11 | 5ca00c4 |
| C8-T3 | A clone's day-3 RNA predicts the share of its descendants that reprogram, on unseen clones | rho 0.59, lower bound 0.46 (train 0.66) | 105 | celltag_reprog:B | 2026-11-11 | 31345fc |
| C10-S2 | In the treatment-response lineage set (C10), a lineage's pre-treatment expression state predicts which lineages persist after treatment, beyond library, a coarse genetic-subclone proxy and capture quality | AUC 0.660 to 0.722, delta 0.062, lower 97.5% bound 0.046 (train CV 0.055) | 5,194 lineages, 420 persisted | C10 partition A | 2026-11-13 | 545ae77 |

Notes on what these mean and do not mean:
- **C7 and C8 are "many clones, no between-condition contrast"** (pattern named in RETRO_2026-11-14). They are narrow: the questions are already largely answered by the source papers, so they are disciplined replications (RETRO_2026-11-11c_EXTENDED).
- **C7-c shrank** from 0.28 on train to 0.24 on A. Chromatin is not privileged over RNA: on train, RNA 0.54 vs chromatin 0.31 bits per cell in blood, and 0.27 vs 0.29 in reprogramming (C7-S3 and C7-S5, exploratory, no look).
- **C10-S2 size:** it adds about 40% to what library + subclone proxy + QC give above chance (AUC above 0.5 goes from 0.160 to 0.222). Real but modest.
- **C10-S2 cannot say:** that the state is non-genetic (the genetic proxy has 3 subclones and is inferred from expression), that survival is fully separated from capture, anything per treatment type (arms are pooled with library as a fixed effect), or anything about instruction versus selection. Dai 2024 already reports pre-survivor signatures in this data, so the direction is published.
- **C10 exposure (accepted):** pooled per-library counts of pre-survivor families and matched post cells were seen before sealing; no lineage-level state link.

Earlier-era confirmed or replicated claims, with the discoveries register's own strength label (these used earlier shard hold-outs and sealed partitions, before the H11 scheme of 2026-11-07):
- **#9** Small clones of one condition are private to that well (2-cell clones seen in another well 21% vs model 51%); replicated on held-out clones (B37: lift at 2 or more cells 0.29 vs fit set 0.31). Strong (replicated).
- **#20** A gastruloid clone's anterior-posterior propensity is a stable clone state: the clone explains 0.56 (fit) and 0.69 (held-out) of per-gastruloid polarity (G1a, G6b). Strong.
- **#26** Clone states plus rank sorting predicted the order of two unseen clones: 12/12 for one pair, about 0.50 for a pair predicted to barely sort. The magnitude was off (separation -1.60 vs predicted -2.67). Moderate.
- **#8, #13 (H 0.71), #1 and others** (chapters 1 and 2) are labelled strong or moderate in DISCOVERIES.md; #1 and #13 rest on several pre-registered tests and open-data analyses rather than one oracle look, so read their evidence column there. I did not re-verify each of those entries for this document.

### 3.3 Train-only or provisional results (not confirmed)

**C12-S1, expansion-culture lineage set (train partition only, 712 clones).** Counts are from `selection/C12-S1.json` and the log.
- On train cross-validation, a clone's day-20 expression state beyond its log day-20 size, QC, library, barcode level, multi-clone fraction and hashtag adds delta AUC **0.166** for presence in vivo at day 20 (101 positives of 712), and **0.060** for presence ex vivo at day 40 (158 positives of 712).
- Controls: state placebo about -0.01, label shuffle -0.010 (in vivo) and +0.016 (day 40, so day 40 is nearer +0.04 net, RETRO_2026-11-14b point 6). Nuisance variables alone give AUC 0.50 and 0.52.
- **This is train CV, not a held-out result (H8).** Not excluded: well effects beyond the hashtags, the composition of the inoculum, and overlap with the source paper's own association (H20).
- **Power is PROVISIONAL and must be redone with `sim_power`.** The figures 0.55-0.57 per single partition and 0.85 pooled A+B for in vivo came from `power.z_power` on a fixed out-of-fold standard error. That breaks H17, which requires simulating the refit pipeline at the sealed n. Treat them as probably optimistic. For day 40, power per partition was 0.25-0.27 and never reaches 0.8 even with all three partitions (0.63).
- **Exposure to state in every C12 report:** C12-S0 per-partition D20 unit counts (train 712, A 304, B 316, C 298 under the highest-UMI rule); train-only presence fractions and size quantiles; pooled assigned and ambiguous cell counts per sample; train library counts. Nothing from A, B or C outcomes.

**C9-S2 (chromatin early state to neutrophil-vs-monocyte split, celltag_lsk:C), never run.** Train CV rho 0.30 (n 195, chosen C 0.01), plan-stage power 0.84. Out-of-sample on the spent partitions A and B (209 clones), the model gives rho 0.26 (A 0.22, B 0.38), which puts power on C at 0.71 (C9-S3). Exploratory: 0.26 is about 3.7 standard errors above 0 on spent data, so a chromatin signal is likely, but it is not confirmation.

**Model-family results from chapters 1-3** (open data, not oracle-confirmed unless marked in 3.2): selection on a fate state present at day 2 explains most of the cytokine effect, with instruction modest (about 0.5 logits each way); discoveries #1-#6 and #10-#12 in DISCOVERIES.md. #7 (separate seeding of the full-cocktail culture) is "in tension" with the source paper's design.

---

## 4. Data inventory

Partitions follow H11: train 40% (buckets 0-1), then A, B, C. "Spent" means an oracle look was used.

| dataset (neutral name) | source | unit | train / A / B / C | state |
|---|---|---|---|---|
| Cytokine-perturbation lineage set (chapter 1) | Weinreb 2020 | clone | hash shards | looks logged (E6, E6b, E9+E10, E14, B37); current state of the Epo/SCF shard was not re-verified here |
| SIS-seq / SIS-skew (chapter 2) | Tian 2021 | clone | shards | held-out looks spent: E51, E54 |
| Gastruloid clones, Raj lab (chapter 3-6) | Ayyappan 2025 plus Aug-Sep 2026 revision files; DATA_MAP_RAJ.md (12,049 files, 226 GB) | gastruloid, passage or replicate | per-file locks | G1, G6, G7 spent; signalling-perturbation partition A spent (G11), B refused by the power gate (G12, 0.54), C sealed; mixing-rescue A+B spent as one look (C4-S4), C sealed; ATAC A and B sealed; DiffAssay replicate 2 sealed; locked: F8MWB5, revision chimeras, atlas |
| Gastruloid founder clusters, GSE291244 (chapter 4-5) | Regalado/Qiu 2026 | founder cluster, 28 | partitioned by founder cluster | **first version VOID** (partitioned by RT well, which is not a gastruloid; never read). Replacement fails power (G14). No look |
| celltag_lsk | Jindal 2024, 6,822 clones | clone | 40/20/20/20 | A spent (C7), B spent (C8 T1), **C sealed (last)** |
| celltag_reprog | Jindal 2024, 8,050 clones | clone | 40/20/20/20 | A spent (C7), B spent (C8 T3), **C sealed (last)** |
| chspctrack | human CD34+ clones, 13,700 clones | clone | 40/20/20/20 | A spent (C7); **B, C sealed** |
| Treatment-response lineage set (C10) | Dai 2024, GEO GSE223003 processed tables | pre-treatment lineage | train 10,132 lineages in the treated libraries; A 5,194 | A spent (C10-S2); **B, C sealed** |
| Expansion-culture lineage set (C12) | public Zenodo deposit, 2026 | clone (connected component of co-called barcodes at D20) | train 712 / A 304 / B 316 / C 298 | **A, B, C all sealed, unspent** |
| SG_kit1 gastruloid seqFISH (C11) | Ayyappan 2025 folders | gastruloid | not partitioned | **not downloaded** (failed its gate, 5.1) |
| PEtracer embryo lineage trees | Cell 2026, GSE342607 | embryo | not partitioned | **not downloaded** (H20 risk high; on the watch list) |

Exposure notes kept in `partitions.py` (`prior_exposure`) apply to every set. Where the registry key of the C12 set abbreviates the original dataset name, it was left as is because renaming it would change the salt and the seal (RETRO_2026-11-14b point 8).

---

## 5. Negatives

These are as important as the confirmed claims. Each line states what failed and why.

### 5.1 Chapters that ended without a confirmed primary result

| chapter | what was asked | outcome | one-line reason |
|---|---|---|---|
| 4 mixing rescue | Does mixing clones rescue elongation, and why? | closed, data-limited (2026-11-07) | The one powered sealed test (C4-S4, partitions A+B, power 0.80) did **not replicate**: T2 +0.25 SD vs predicted +0.81, lower bound below 0. C partition kept sealed |
| 4 sorting-index test (C4-S2b) | Does the data support the sorting idea? | **withdrawn as unsupported** | Power of the ordering check 0.20 (2-4 gastruloids per condition); the earlier "does not support" wording was an overstatement |
| 4 seqFISH chimeras (C4-S3) | Test the chimera design | failed pre-flight | 5 control gastruloids; 1+3 (n 2) vs 1+6 (n 3): minimum p 0.10; no download |
| 4 pre-registered test C4-S2a | Mixed-plate rescue | **NOT run on that design** | No single partition had power; Bence kept all sealed. The later union look (C4-S4) failed (above) |
| 5 clone memory | Does clone state fade with passages, and what carries it? | parked, data-limited (2026-11-08) | Passage-1 A-P chromatin axis is real (gap 0.38, about 5.6 SE out of fold) but fade power 0.14 / 0.53; S-ratio power 0.77; expression memory split-replicate power 0.41; five clones, two replicates. No single memory law (blood loses about 4% per division; ES clones lose about half per passage) |
| 5 DiffAssay (C5-S7) | Growth bias vs head-to-tail preference | failed pre-flight | Growth only, n 5, replicate 1 null; replicate 2 sealed |
| 6 cells into space | Can clone state plus local rules build A-P domains and elongation? | closed as **exploratory** (2026-11-10), credit about 3.25 in the ledger | No robustly powered unseen test existed; S1 power at most 0.20 under the core model; clone-coupling fits failed leave-one-clone-out (signs 3/5 and 4/5); 13 model versions; the one mechanism that passed step-0 checks has Q2 and Q3 exposed and arm-dependent; the atlas alignment is circular |
| 8 T2 | Does early human-progenitor state beat cytokine condition? | **REFUSED by the power gate** | 0.78 below 0.8 at the partition's own structural count (32 effective units); the planning figure 0.86 had used a dataset-wide ratio. chspctrack:B not spent |
| 9 (HSPC erythroid vs myeloid, C9-S1) | Early state vs a split | **WITHDRAWN at train selection** | Only 7 of 179 train clones carry any neutrophil or monocyte cell, experiment 1 none; the pooled rho 0.57 was a tie artefact. No oracle entry, sealed B and C unread |
| 9 (chromatin, C9-S2) | Early chromatin vs neutrophil/monocyte split | **never run** | Power on the last partition 0.71 (out-of-sample rho 0.26) after three rescue designs (C9-S3: 0.71, 0.67, 0.09); none reaches 0.8; celltag_lsk:C sealed |
| 10 instruction vs selection | Does treatment instruct survival or only select lineages? | **stopped at S0** after 4 amendments and placebos | Treatment is assigned per library, 2 libraries per arm (1 pair once time-matched). Untreated libraries already differ by 0.84 (count-matched retry 0.72; bar 0.5). Became H22 |
| 11 SG_kit1 seqFISH | Chimera segregation graded by clone propensity | **failed the chapter gate in one step** | 1+3 (2) vs 1+6 (3) per arm under H22; pure-vs-chimera already published (H20); 1-2 chimeras per sealed partition. Not downloaded |
| 12 expansion-culture lineage set | Does D20 state predict later presence beyond size? | large train effect, **no powered single sealed test** | One partition: power 0.55-0.57 (provisional) in vivo and 0.25-0.27 for day 40. Pooled A+B 0.85 is provisional and needs `sim_power`; day 40 never reaches 0.8 |

### 5.2 Individual failed, unconfirmed or voided experiments

- **E26-E29 verdicts withdrawn (RETRO_2026-11-03).** The fits never matched the basic levels (loss badly scaled), so four "ruled out" verdicts were retracted.
- **Instruction vs selection ratio** was never resolved as a ratio. Instruction stays modest and stable; selection varies with the model (RETRO_2026-11-05b).
- **E55 (pretrained single-cell foundation model).** On 197 held-out clones the model's embeddings predicted clone fate at r 0.58 versus 0.64 for plain PCA (delta -0.06, interval -0.13 to -0.0001); concatenating added nothing (#19). Negative.
- **E54 (SIS-seq RNA trace).** Held-out correlations 0.25-0.31, about half the fit-set strength and not individually significant (n 34). #17 stays weak on held-out clones.
- **E51 and #16.** The fit-set "expand-and-shift" pattern did not replicate (+0.09 held-out vs -0.17 fit). #16 withdrawn; the Bcor fate-gain claim #18 also withdrawn as a fate claim (disappears under a +/-0.5 label shift).
- **G7b (memory over passages).** Both the single-rate and the clone-specific-rate predictions failed on sealed series. #23: fading happens in some series, but no model predicts a new series.
- **G11 (signalling perturbations, partition A).** On the prediction but only 3 control gastruloids (about 15% power): underpowered, not a confirmation. **G12:** partition B refused by the power gate (0.54), B and C stay sealed; testing needs about 24 controls plus 48 treated in one series.
- **Mixing rescue #22.** The +0.30 pair effect holds for the 2024-25 table only; it did not replicate out of sample (C4-S4).
- **Mechanism of rescue #24.** Latent-average window, OR/AND complementation and T+ fraction all fail as formulated; no model is supported.
- **Gastruloid founder clones (G14), GSE291244.** 28 founder clusters; the authors' own correlation is 0.052; fails power.
- **Chapter 6 model versions.** C6-S4 coupling fit failed leave-one-out (signs 3/5, rho 0.6); C6-S5 right direction (5/5 in sample) but about 10 times too weak, CV 4/5; Potts versions failed or fell apart (C6-S7b to f).
- **#28 (clonal takeover).** An observation only, n = 1 winner; not testable with public data.
- **C4-S1 and others:** earlier rescue mechanism claims "not supported" were withdrawn when their power was found to be 0.20.
- **#27 (signalling perturbations compress clone separation 11/11)** stays unconfirmed: no model passed H12 and the sealed look was underpowered.
- **Void partition:** the first GSE291244 partition (by RT well) was void because the paper pools and dissociates gastruloids before RT, so every gastruloid fell into every partition. No statistic was computed on it.

### 5.3 Power refusals, voids and a near miss (what the gates saved)

| event | date | what the gate did |
|---|---|---|
| Signalling-perturbation partition B | 2026-11-07 | refused at power 0.54 |
| C8 T2 | 2026-11-11 | refused at 0.78; B not spent |
| C9-S1 | 2026-11-11 | withdrawn at train selection |
| C9-S2 / S3 | 2026-11-11 / 14 | never run; best power 0.71 |
| C10 instruction | 2026-11-13 | stopped at S0 |
| C11 | 2026-11-14 | failed at the proposal gate |
| C12 single partition | 2026-11-14 | not looked; power 0.55-0.57 (provisional) |

---

## 6. Process failures and what changed

| failure | when | consequence | what changed |
|---|---|---|---|
| Held-out data read outside the oracle (G5 read G1a's sealed rows); five unsealed chapter-3 tests. Bence "furious" | 2026-11-06/07 | #21-#25 had been over-rated; the sealed re-check (G6) kept the direction of the fading claim and rejected its rate | H1-H10, loaders exclude sealed data by default (85faf94); M13 |
| G7 spent all remaining same-design gastruloid hold-out in one round, on a memory rule G6 had already undermined | 2026-11-07 | one failed look where open data would have shown the problem | H11 (several sealed partitions), H12 (train-chosen model), H13 (keep a reserve) |
| Partition A of the signalling set spent with 3 controls (about 15% power) | 2026-11-07 | one of three partitions wasted | H14 (power gate in the oracle) |
| GSE291244 partitioned by RT well without reading how the experiment ran; DiffAssay mislabelled twice; proposed spending DiffAssay's only sealed replicate when used data answered it; a count that was an outcome printed during a "structure" check | 2026-11-07 | a whole partition scheme void; wrong questions proposed twice | H15 (pre-flight block, enforced by the oracle) |
| Data-driven drift ("now you are drifting!") | 2026-11-07 | work outside the chapter | H16 (Serves line, scope-guard hooks) |
| Model churn (nine simulators in two days) | 2026-11-03/04 | long searches for no new information | model-iteration stopping rule; no new simulator without a diagnostic; early-stop guard on sweeps |
| Two flawed scoring rules (B29 and E47) found by Bence, not by the analysis | 2026-11-05 | noise picked the reference model | M1, M2; check each statistic's width before summing |
| Missed half the data store; low-power result worded as evidence against a hypothesis (C4-S2b) | 2026-11-07 | revision-era data found only when Bence asked | list the whole tree before saying data is absent; never word a low-power look as evidence against |
| C6-S7 reached 13 model versions; PC used for jobs the Actions runner could do | 2026-11-10 | a forking path toward a pass; hours of wall time | H18 (retro before a 5th amendment), step-0 hygiene (at least 10 repeats, settling check, float tolerance), standing use of GitHub Actions |
| ATAC pipeline went to the fate statistic without a validity check; C7-S3 delta unpowerable by design | 2026-11-11 | two amendments, near-miss wrong conclusion | fate-free validity gate; pre-registration tick-list; plan-stage power |
| Plan-stage power for a subset used the dataset-wide ratio (C8 T2: 0.86 vs real 0.78) | 2026-11-11 | caught by the oracle before a statistic | H19 amendment: a subset uses its own structural count |
| Chapters chosen for interest ended data-limited three times (4, 5, 7) | 2026-11-11 | Bence: "3rd fcking chapter we went with your suggestion and ending up as data limitation" | H19 (achievable chapter goals only) |
| Confirmed chapters 7-9 repeated what source papers already had | 2026-11-11 | low novelty | H20 (value and literature gate before code) |
| Degenerate outcome passed the scan (C9-S1) | 2026-11-11 | withdrawn | H21 (readout gate); the withdrawn test's reader functions stayed in place and the oracle refused every look for two days (found 2026-11-13), so removal rules were added to the practice checklist |
| Treatment assigned per library, 2 per arm, read as lineage-level power (C10); the step was framed as a "scan" so the gate was never applied; 4 amendments narrowed it before a placebo stopped it | 2026-11-13 | the third chapter lost to a design flaw visible at S0; Bence's anger | H22 (name the assignment unit at S0; fewer than 3 per arm for a between-condition contrast means stop), adopted 2026-11-14 |
| C9-S2 declared "ready" on an optimistic power (0.84 from train CV with no shrink) | 2026-11-11 | spent partitions later said 0.71; the look was never run, so nothing was lost | pending: use the out-of-sample effect on spent partitions when they exist |
| First oracle call on the C10 set crashed on repeated barcodes across libraries | 2026-11-13 | ledger not written, A not spent | dry-run the sealed test function on train rows through the same code path before a look (memory item) |
| First C12 clone-linking rule gave one giant component | 2026-11-14 | map deleted unread before any count; rule amended | H1 held |
| Pushed with a failing guard test (a23997e, fixed in c352b14) | 2026-11-14 | a broken tip on origin for one commit | run tests/test_guards.py before every push (memory item) |
| FOCUS.json edited by a script, two `next_steps` lines dropped (caught in the diff) | 2026-11-14 | none lost | edit FOCUS.json only with the Edit tool and read the diff |
| C12 selection power computed with `z_power` on a fixed out-of-fold SE, not `sim_power`; label-shuffle bias not subtracted; retro skipped its literature gap map | 2026-11-14 | the 0.55-0.57 and 0.85 figures are provisional | practice lines adopted 2026-11-14: positives-based counts for presence outcomes, train confound battery, shuffle-net deltas, pre-registration template, `sim_power`-only selection power |
| Data search reactive: fixed watch list re-read each time, while the PEtracer embryo set and the C12 set were found only by a broader scan | 2026-11-14 | a step per unchanged re-check | pending: score searches by the shape of past successes |
| 9 of 16 retros: the key failure was found by Bence's question, not by the analysis | whole project | all rules were reactive | self-challenge block (five questions in every result report, from 2026-11-13) |
| 19 lessons lived only in retros (RETRO_SWEEP_2026-11-13) | 2026-11-13 | repeated | the rule that every lesson lands as H-rule, practice line, memory item or "not adopted"; practice checklist created |
| Platform safety classifier repeatedly stopped the larger models on the C12 dataset; about six turns lost | 2026-11-14 | tooling, not science | model routing rule: blocked steps go to the data-lane agent on Sonnet 5.5 |

Four retro decisions were never carried out (RETRO_SWEEP_2026-11-13 section 4): refactor the chapter 1 and 2 models into one code base (partly done as `core_model.py`), reconnect the 2023-24 tissue simulator as the spatial layer, one parametrised analysis script, and reading the Marangoni preprint before further chapter 6 design. They are moot while the chapters are closed.

---

## 7. Open questions and reopen conditions

| parked or closed chapter | open question | reopen when |
|---|---|---|
| 4 mixing rescue (closed, partition C sealed) | Is rescue a per-clone competence supplied by a diverse partner? | pure and mixed gastruloids on the same plate with per-clone composition become public |
| 5 clone memory (parked) | Does the passage-1 A-P chromatin axis fade with passages? | the human embryonic-cell endoderm sister-well multiome is released, or any set with 10 or more clones x 3 or more passages x 2 or more replicates. C5-S6 tests are ready, ATAC A and B sealed |
| 6 cells into space (closed, exploratory) | Which crawl arm fits, and what is clone-specific memory loss? | clone-specific memory data over passages 1-4, or crawl-orientation imaging. S1, S2:C, revision chimeras and the atlas stay sealed or locked |
| 7 fate carrier (closed, confirmed) | RNA vs chromatin in the same cell; coverage | same-cell clone data (the endoderm multiome, or a similar release); B/C of celltag_lsk, celltag_reprog and chspctrack sealed |
| 8 fate readout (closed, T1 and T3 confirmed) | T2: human progenitor state vs condition | a partition with at least 40 effective units in the subset; chspctrack:B/C sealed |
| 9 chromatin to neutrophil-vs-monocyte split (paused) | Is the chromatin signal real on a partition? | an out-of-sample rho of about 0.29 or more at 97 units, or a new data release; celltag_lsk:C is the last partition and needs Bence's verbatim OK (H13) |
| 10 treatment-response lineage set (closed, confirmed narrow) | Is the state non-genetic? Instruction vs selection? | per-barcode variant calls from the reads (about 564 GB, not downloaded) separate genetic from non-genetic state; partition B is the natural test with the same frozen pipeline plus a genetic term. Instruction vs selection needs at least 3 libraries per arm |
| 12 expansion-culture lineage set (paused) | Does day-20 state predict later presence beyond size? | Bence decides: a pooled A+B look only after `sim_power` confirms 0.8; the day-40 readout never reaches 0.8 |
| Parked ideas | signalling-perturbation compression (#27), G13 DiffAssay, dominance (#28), GSE291244 founder state | new data meeting H19 and H22 |

---

## 8. Reading guide

- Rules in full: program.md (H1-H22, practice checklist, self-challenge block, guard tests, model routing); short digest: RULES_DIGEST.md.
- Claims with evidence and literature checks: DISCOVERIES.md. Sealed looks: test_ledger.tsv; train selections: `selection/*.json`.
- Chapter close-outs: the WRITEUP_* files, the C10 write-up (WRITEUP file for C10), the RETRO_* files; full-project recap of 2026-11-11: PROJECT_RECAP_2026-11-11.md; extended retro RETRO_2026-11-11c_EXTENDED.md.
- Open data searches: DATA_SEARCH_2026-11-14.md, NEXT_DATASETS_2026-11.md.

**Ranked next steps** (my pick first):
1. Stay paused until after the 2026-11-21 watch-list check; nothing sealed is spent by waiting.
2. If Bence wants C12 moved forward, redo the power with `sim_power` on the refit pipeline at the sealed n (train-only, spends nothing) before any look is considered.
3. Decide whether to adopt the pending retro practice lines (section 2, decision 2) so the next chapter opens under them.
