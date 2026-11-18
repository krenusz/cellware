# Novelty audit, all chapters (2026-11-18)

Bence: "so based on what we have now check all the chapters and discoveries whether we have anything novel". Builds on `novelty_map_2026-11-12.md` (chapters 1-8, abstract search plus a full-text read of 283 open citers of the LARRY source paper and all 12 citers of the SIS-skew source paper) and the H25 lens adopted today: each data set's source paper checked at claims level.

**Searched today:** 6 targeted web searches (the LARRY reanalysis question, the SIS-skew/Bcor reanalysis, gastruloid clone memory and chromatin, pre-resistant state beyond subclone, held-out-embryo prediction, the C12 source); claims-level reads of the embryo source paper (`source_claims_petracer_2026-11-18.md`) and the C12 source (title and abstract only, via its data record). Snippet and abstract level; the 2026-11-12 full-text citer check is not redone.

## Verdicts

| ch | result | sealed / held-out? | verdict | closest prior work |
|---|---|---|---|---|
| 1 | #2-3 pure instruction cannot fit; instruction about 0.5 logit under each cytokine | within-model, sealed cross-well looks | **NOVEL** (no reanalysis found, again today) | source paper left it open; live imaging shows instruction exists |
| 1 | #9 small clones private to one condition well | held-out replicated | **NOVEL** | restricted progenitors make small clones (mechanism only) |
| 1 | #11 day-2 abundance predicts wells reached | fit set | novel but follows from the design | none |
| 1 | #12 one well's lineage comes from clones absent elsewhere | fit set | novel, unconfirmed; in tension with published instruction | |
| 1 | #4-5 survival advantage size and selection bound | fit set | known phenomenon, new numbers | |
| 1 | #1, #6, #10 | | known (replications) | source paper; heritable imprinting work |
| 2 | #14/18 the KO fate effect is not identifiable once editing efficiency and sampling are modelled | design-level recovery study | **NOVEL** (no follow-up of the source found today) | source paper claims a fate effect |
| 2 | #15 the control wells are not null | fit set, strong CIs | **NOVEL** (finer than the source's test) | source paper |
| 2 | #13, #17 | | known | |
| 3 | #21 domain size set by the aggregate, not clone composition (quantitative) | open data | novel quantification, unconfirmed | source paper (qualitative) |
| 3 | #25 memory lost at passaging events, not divisions | fit set | new test of a stated idea | source paper (qualitative) |
| 3 | #26 clone states plus sorting predict unseen pairs (12/12) | sealed | known idea, **sealed prediction is new** | source paper |
| 3 | #20, #23 | | known | |
| 4 | #22 rescue (withdrawn); #24 none of three mechanisms fits | | negative, unconfirmed | |
| 5 | clones chromatin-distinct at passage 1 along an A-P axis | exploratory, 5 clones | **partly published, other system:** hESC clonal memory preprint (2026) reports fate-biased clones differ in chromatin at lineage elements | Linneberg-Agerholm 2026 |
| 6 | axis needs short-range adhesion plus long-range pull | exploratory | **published** | Guan/Hormoz 2025 |
| 7 | fate state heritable and readable early, one metric, 3 systems | sealed, 5/5 on A | known phenomenon; **one sealed metric across systems is new** (method) | source papers; FateLimit, LineageFlow 2026 |
| 8 | Neu/Mono and reprogramming success predictable on unseen clones | sealed | known; sealed confirmation | source papers; LineageFlow 2026 (clone-disjoint held-out) |
| 10 | #29 state beyond a subclone proxy and QC predicts survival | sealed on A | **partly published:** the source paper reports pre-resistant states in the same data; the increment beyond a subclone proxy, held out, is ours (modest) | source paper |
| 11 | gate only | | published (gate said so) | |
| 12 | day-20 state beyond clone size predicts later in vivo presence | train only | **published in kind:** the source preprint's state-fate analysis links clonal programs to self-renewal, in vivo initiation and treatment response (claims level) | source preprint |
| 14 | #31 state beyond type predicts relatives' fate in unseen embryos | sealed on A | **partly published:** source paper reports fate programs and origin traces; lineage inpainting (2026) reports fates separate late; the **held-out-embryo predictive test is ours** | source paper; arXiv 2610.40447 |
| 16 | #30 hidden inherited P/L state | train only | **largely published**; method new | source paper; Liu 2026 |

## What is novel, ranked
1. **Corrections of two published data sets, held on sealed or recovery checks (chapters 1-2):** instruction size and the exclusion of pure instruction (#2-3); private small clones (#9); the KO fate effect is not identifiable (#14/18); the control wells are not null (#15). Still unchallenged today. This is the strongest unit: a reanalysis note per data set.
2. **Held-out predictive tests of known phenomena:** C14 on unseen embryos (#31), C10's increment beyond a subclone proxy (#29), C3 sealed pair-order prediction (#26), C7/C8 one metric across systems. The biology is known; the held-out test is not common. Modest novelty, method-level.
3. **The method itself:** sealed partitions, oracle looks, power gates, the novelty gate. Rare in this field. Not a biological finding.

## What is not novel
- Chapters 6, 11, 12 and 16: published (C12 and C16 by their own source).
- Chapters 5, 10 and 14: partly published.
- Most of the replicated biology in chapters 1-3, 7 and 8.

## Limits
- Snippet and abstract level today; no new full-text citer check.
- "No reference found" is not proof of novelty.
- The four chapter 1-2 items are corrections of 2020-2021 papers. Their value depends on the source authors' own raw-data checks (editing efficiency per clone, vector effects).
