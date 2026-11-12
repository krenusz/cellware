# Novelty map: what we found vs what was already known (2026-11-12)

**Method:** each discovery (DISCOVERIES.md #1-#27, plus chapter 5-8 results) was checked against:
- the cross-check already recorded in DISCOVERIES.md;
- targeted Europe PMC searches (abstracts, 2019-2026; script in the session scratchpad);
- the 2026-11-11 literature re-review.

**Limits:**
- Abstract level only. "No reference found" means no abstract match, not proven novelty.
- Supplements and full texts were not searched, except Weinreb 2020 SI and Tian 2021, which we read earlier.
- The CellTag-multi Results stay unread (sealed sets remain).

## Verdict at a glance

| class | count | items |
|---|---|---|
| **A. Genuinely novel and confirmed on held-out data** | 4 | #2-#3 instruction size, #9 private small M-CSF clones, #14/#18 Bcor reinterpretation, #15 null wells not null |
| **B. Novel but unconfirmed / exploratory** | 6 | #11, #12, #21 (partly), #24, #27, chapter 5 chromatin axis |
| **C. Known phenomenon, new quantification or a stricter test** | 7 | #4-#5, #19, #25, #26, ch7 cross-system, ch8 |
| **D. Already known (we replicated it)** | 8 | #1, #6, #10, #13, #17, #20, #23, chapter 6 rule |
| **E. Withdrawn / artefact / housekeeping** | 4 | #7, #8, #16, #22 |

## A. Genuinely novel and confirmed

| # | finding | why novel | closest prior work | evidence |
|---|---|---|---|---|
| 2-3 | In the LARRY cytokine experiment, pure instruction cannot fit the data. Instruction is modest and stable, about 0.5 logit each way under G-CSF and M-CSF, whatever selection is added | Weinreb 2020 showed fate bias at day 2 but left instruction vs selection open ("cannot be ruled out"). We found no reanalysis that **quantifies** both in one model (0 hits for LARRY + instruction/selection) | Weinreb 2020; Rieger 2009 (instruction exists, in sorted GMPs) | strong within the model family; sealed cross-well looks |
| 9 | Small M-CSF clones are private to the M-CSF well (21% seen elsewhere vs 51% by model) | not reported in Weinreb 2020. Lin/Naik 2024 gives a plausible mechanism (restricted progenitors make small clones) but not this observation | Lin/Naik 2024 | strong, replicated on held-out clones (B37) |
| 14/18 | Bcor loss acts on persistence and expansion. **A fate-level effect is not identifiable** in SIS-skew once 40% editing efficiency and sampling are modelled | **reinterprets the source paper's claim** (Tian 2021: Bcor restricts pDC potential). No later paper re-examines it (Bcor search: only Tian 2021 and its preview) | Tian 2021 Immunity | moderate; recovery checks + held-out E51 |
| 15 | The SIS-skew control ("null") wells are not null: control-vector sisters shift away from cDC1 within clones | **contradicts the source paper's** report of no change in control wells | Tian 2021 | strong (CIs exclude 0) |

**Caveat for A:** all four come from reanalysing published datasets. Their value is as corrections or
quantifications of those papers. A reviewer would want the authors' raw-data checks (for #15: vector or
population effects; for #14: editing efficiency per clone).

## B. Novel but not confirmed

| # | finding | status | prior work |
|---|---|---|---|
| 11 | A clone's day-2 abundance predicts how many wells it reaches, beyond independent seeding | moderate, fit set; no reference found | follows from the Weinreb design, never stated |
| 12 | Epo-well erythroid cells come from clones absent from the G/M wells (not diversion of shared clones) | moderate, fit set | Grover 2014 (Epo instructs MPPs): **in tension** |
| 21 | Gastruloid domain size is set by the aggregate, not by clone composition (slope 0.17 vs 1) | unconfirmed (open data) | Ayyappan 2025 notes the crossing is "not sensitive to the ratio" (qualitative). Our quantitative rejection of sorting-by-state is new |
| 24 | Mixing rescue is not buffering, strict complementation or T+ fraction as formulated | unconfirmed negative | Regalado 2025, Braccioli 2025, Oriola 2024 proposed these |
| 27 | RA, SB and AGN erode the clones' division of labour (11/11 comparisons) | unconfirmed (train only; sealed drug partitions unspent) | Ayyappan 2025 tested drugs on chimeras: **partly known** |
| ch5 | Gastruloid clones are chromatin-distinct at passage 1, along an A-P axis (gap 0.38, about 5.6 SE) | exploratory, 5 clones | no reference found (0 hits) |

## C. Known phenomenon, new numbers or a stricter test

| # | finding | what is new | prior work |
|---|---|---|---|
| 4-5 | G-CSF gives neutrophil-leaning clones a 2.5-6.5x survival/expansion advantage; plausible lineage-matched selection gamma 0.3-0.6 | the size and bound | Kimura 2009, Weinreb 2020 Supp Fig 7g (qualitative) |
| 19 | Geneformer reads the clone fate state no better than PCA (r 0.58 vs 0.64, held out) | first test on **clone fate**, sealed | Kedzierska 2025 (zero-shot models no better than PCA, in general) |
| 25 | Gastruloid memory is lost at passaging events, not with divisions (subclones 1.21x parent; a per-division loss would leave <= 0.34) | the division-vs-event test | Ayyappan 2025: "robust to large-scale expansion" but "not preserved through serial passaging" (same idea, stated qualitatively) |
| 26 | Clone states + rank sorting predict the order of two unseen clones (12/12) | a sealed prediction | Ayyappan 2025 (pairs sort by propensity) |
| ch7 | Fate state heritable (H 0.75) and readable early from RNA in 3 systems (0.35-0.50 bits/cell) and from chromatin in LSK (0.24) | **one metric, sealed, across 3 systems** | Weinreb 2020, Jindal 2024, the 2025 human clonal-fate predictors paper, FateLimit 2026, LineageFlow 2026 |
| ch8 | Neu vs Mono (rho 0.45) and reprogramming success (rho 0.59) predictable on unseen clones | sealed confirmation | Weinreb 2020, Biddy 2018, CoSpar 2022, Deep Lineage 2026 |

## D. Already known (replications)

| # | finding | known from |
|---|---|---|
| 1 | Cytokines act mainly by selection on a fate state present at day 2 | Weinreb 2020; Lin 2021 |
| 6 | Fate memory near-full over 2 days | Naik 2013; Nat Commun 2026 (human HSPC clonal memory) |
| 10 | G-CSF-private clones are neutrophil-committed | Weinreb 2020 Supp Fig 7g |
| 13 | DC fate fixed in the clone before the split (H 0.71) | Naik 2013; Lin 2018; Tian 2021 |
| 17 | The clone state leaves only a weak trace in day-2.5 RNA | Weinreb 2020 (fate bias partly invisible); Tian 2021 |
| 20 | Gastruloid clone A-P propensity is a stable clone state | Ayyappan 2025 (our source) |
| 23 | Propensity fades over passages | Ayyappan 2025 |
| ch6 | Axis needs short-range adhesion **plus** a long-range posterior pull | **"Robust axis formation requires both short-range adhesion and long-range attraction"** (bioRxiv 2025, doi 10.1101/2025.09.27.678924). Our C6-S7l model converged on the same mechanism independently |

## E. Withdrawn, artefacts, housekeeping

- #7 full-cocktail seeding: a fitted structure; Weinreb's design contradicts it.
- #8 detection does not differ by cell type: a control result.
- #16 did not replicate.
- #22 mixing rescue did not replicate out of sample.

## Bottom line

- **What is genuinely ours:** corrections and quantifications of two published datasets.
  - LARRY: instruction is about 0.5 logit; pure instruction is ruled out; small M-CSF clones are private.
  - SIS-skew: the Bcor fate claim is not identifiable; the control wells are not null.
  - All four held on sealed data. This is the strongest publishable unit: a reanalysis / "matters arising"
    style note on each dataset.
- **Methodological contribution:** sealed, oracle-gated confirmation of clone-fate readout across three systems
  with one metric. The phenomenon is known; the rigour is not common in this field.
- **Most of the biology replicates known findings.** That is useful as validation, not as discovery.
- **Watch:** the 2025 preprint on adhesion plus long-range attraction pre-empts our chapter-6 mechanism. Cite
  it before saying anything about axis formation.
