# Discoveries register

Every finding about the data or the biology, with its evidence (experiment ids in
EXPERIMENT_LOG.md), its strength, and a literature cross-check. Status of the check:
**consistent**, **in tension**, or **not yet checked**. Methodological lessons are at the end.
Started 2026-11-05; entries before that date are collected from the log and FINDINGS_2026-11.md.

Data set for entries 1-12: Weinreb et al. 2020, LARRY cytokine-perturbation experiment
(day-2 sisters split into full cocktail, G-CSF, M-CSF, Epo, SCF; read out days 4 and 6).

## Fate decisions

| # | discovery | evidence | strength | literature | check |
|---|---|---|---|---|---|
| 1 | Cytokines act mainly by **selection on a fate state that already exists at day 2**; G-M co-present clones are monocyte-leaning under G-CSF. | B7, B9, B11, B13-B16, B21 | strong | Weinreb 2020 (fate bias readable at day 2; Supp Fig 7g, lineage-matched expansion, survival "cannot be ruled out"); Lin 2021 Nat Cell Biol (Flt3L acts by selective expansion of primed clones, "clonal tuning"); Calabria 2024 Nature (long-term clonal commitment in patients). Counterpoint: Tsapogas 2014 (Flt3L instructive in vivo, population level) | consistent (literature split on Flt3L) |
| 2 | **Pure instruction cannot produce the data** under G-CSF (no fit with selection switched off). | E40 | strong within the model family | Rieger 2009: instruction in >= 34% (G) / 65% (M) of sorted GMPs, so instruction exists, but GMPs are a later, sorted stage | consistent (both act) |
| 3 | **Instruction is modest and stable**: about 0.5 logits each way under G-CSF and M-CSF, almost unchanged across E38, E47, E48, E49 whatever selection is added. | E38-E49 | moderate | Rieger 2009, Mossadegh-Keller 2013 (M-CSF instructs HSCs), Endele 2017 (CSF-1/Src instructs) | consistent that both act; differing emphasis (Rieger: instruction in most sorted GMPs; ours: selection dominates in a mixed day-2 population) |
| 4 | G-CSF gives neutrophil-leaning clones a **2.5-6.5x survival/expansion advantage**, trading off against instruction along a ridge. | E40 | moderate | Kimura 2009 (GM-CSF/STAT5 controls GMP proliferation and survival) | consistent (qualitative) |
| 5 | **Lineage-matched selection at a plausible strength** (gamma ~0.3-0.6: at most about one extra doubling per SD of bias over 48 h) fits B7 G; the fit pushes gamma to whatever cap is set. E46's unbounded 1.3 is about twice the plausible bound. | E46, E47, E48 | moderate | Rieger 2009 supplement: ~19% early death in both M-CSF and G-CSF, ~2 doublings on day 1, slower on day 2 | consistent (bound derived from it) |
| 6 | **Fate memory is near-full over 2 days**; the apparent fading (B21) is sampling noise. | E41b, B21 | moderate | Naik 2013 (heritable lineage imprinting in sister progenitors); Nat Commun 2026 (clonal memory of division in human HSPCs, >= 2 divisions) | consistent |

## Clonal structure of the experiment

| # | discovery | evidence | strength | literature | check |
|---|---|---|---|---|---|
| 7 | The **full-cocktail culture was seeded from a largely separate set of clones** (about 22% of clones mainly there). | B12, E37, E38 | moderate (fitted structure) | Weinreb 2020 SI 8.1 and Supp Fig 7: day-2 cells split evenly into four wells, Mix included, so not a separate seeding; alternative: LSK-like cells expand mainly in the full cocktail (SCF, TPO) | **in tension** |
| 8 | **Detection does not differ by cell type** (neutrophil/monocyte barcode capture ratio 0.91 [0.72, 1.20]); sister competition at the split cannot explain co-presence. | B28 | moderate | Weinreb 2020 SI (design, capture) | consistent |
| 9 | **Small M-CSF clones are private to the M-CSF well**: 2-cell clones seen in any other well 21% (model 51%); real cells spread over libraries, not an artefact; half as often seen at day 2. They explain the M-presence and low G-M co-presence misfits. | B31, B32, B34; replicated on held-out clones (B37: lift at >= 2 cells 0.29 vs fit set 0.31) | strong (replicated) | Lin/Naik 2024: restricted downstream progenitors (MDPs, CMPs) make small clones | consistent (candidate explanation, untested) |
| 10 | **G-CSF-private clones are neutrophil-committed at every size** (70% neutrophils vs 41% in shared clones; privacy grows with size). | B34 (E50 tried a G-only neutrophil class: 0/10 gated, it strips mixed G clones) | moderate | Weinreb 2020 Supp Fig 7g (neutrophil-only clones expand more in G-CSF) | consistent |
| 11 | **A clone's day-2 abundance predicts how many wells it reaches**, beyond independent seeding (gradient 0.13 vs model [-0.09, 0.09]). Clones differ in how many cells they had at the split. | B35 | moderate | no direct reference found; follows from splitting each clone's day-2 cells into wells (Weinreb SI: day-2 sample measures clone size at the split) | expected from design |
| 12 | Every model **over-shares clones with the Epo and SCF wells** (about 2x). For SCF it is a counting artefact: SCF wells are 71% undifferentiated, and counting all cell types fixes it. For Epo it is real: the erythroid cells come from clones absent from the G/M wells; clones shared with G/M make myeloid/basophil cells in Epo. | B32, B36 | moderate | Grover 2014 (Epo pushes multipotent progenitors to erythroid fate): diversion of shared clones tested and not supported; an erythroid-lineage clone subset fits the data | SCF explained; Epo in tension with diversion |

## Chapter 2: SIS-seq / SIS-skew (Tian et al. 2021; Flt3L DC cultures)

| # | discovery | evidence | strength | literature | check |
|---|---|---|---|---|---|
| 13 | **DC fate is largely fixed in the clone before the split.** Sisters in separate wells agree on cDC1/cDC2/pDC split far beyond chance: heritable share H 0.71 [0.59, 0.78] (large batches 0.81); clone-level logit SD 2.0 (cDC2) and 2.7 (pDC). No effect of the well's background on sister differences. A second system showing LARRY's strong clone fate state (#1). | B38 | strong (69 clones) | Naik 2013 Nature (sibling LMPPs share fate, DC-biased imprinting); Lin 2018 Cell Rep (DC trajectories intrinsically programmed, cDC vs pDC split at HSPC stage); Tian 2021 (control clones concordant) | consistent (abstracts + Tian full text read 2026-11-06) |
| 14 | **Bcor loss acts on whether sister clones persist and on how much the KO arm grows, not detectably on their fate split.** One-arm clones are 2-4x commoner in KO wells at every size; no KO within-clone or clone-reweighting fate term beyond the null. E51: uniform KO-arm expansion about 10x (lam_K 2.3 [1.5, 3.2]); fate-linked selection gamma not detected (intervals about +-0.4), model adequate on held-out clones (24/29). The fit-set "expand-and-shift" interaction did **not replicate** on held-out clones. | B39, E51 | moderate (399 both-arm KO clones; baseline problem, #15) | Tian 2021 full text: mean 5x expansion (up to 1,000x), **Bcor editing efficiency only 40%** (our model assumed 100%); claims "gain of pDC fate potential" in 78% of the 20% SkewID-skewed clones (binary presence). Cao 2016, Kelly 2019: Bcor loss raises myeloid proliferation | expansion consistent; **pDC fate gain in tension** (may be expansion + sampling; B42 to test; editing efficiency to add) |
| 15 | **The SIS-skew null wells are not null:** control-vector sisters shift away from cDC1 toward cDC2/pDC within clones (-0.12 / +0.08 / +0.04), and their size ratio tracks fate. Vector/transduction/sorting effect, or W (WT HSPC) vs C (Cas9 LSK) population difference. | B39 | strong (CIs exclude 0) | Tian 2021 full text: W = total cKit+ WT progenitors; they report no change in barcode number, no major expansion, concordant binary fate, 1% skewed; no directional fraction test | not contradicted, not reported (our test is finer) |
| 16 | ~~**Bcor loss lets non-cDC1 clones persist.**~~ **Not replicated** on held-out clones (E51: +0.09 vs fit -0.17); keep as a fit-set pattern only. Clones present only in the KO arm are cDC1-poor (rho -0.17 [-0.25, -0.08]; null +0.05); KO-arm persistence does not depend on the WT sister's fate. One-arm odds 6.2x the null's; large one-arm KO clones are KO-only. | B40, E51 | withdrawn (did not replicate) | Tian 2021 (Bcor restricts cDC2 expansion and pDC potential) | n/a |
| 17 | **The clone fate state is visible in day-2.5 RNA as a lymphoid-vs-myeloid priming axis, not as early lineage factors.** PC1-3 predict pDC (LOO R^2 0.38) and cDC2 (0.27) fate, about 45% / 32% of the heritable part; pre-chosen marker programmes (Tcf4, Spib, Siglech, Id2, Klf4...) do not. The axis: Satb1, Tox, Lrmp toward pDC; Csf1r, Grn, H2-Aa toward cDC2. | B41 | moderate (65 clones; gene axis descriptive) | pDC from lymphoid-primed progenitors (Rodrigues 2018 Nat Immunol; Dress 2019 Nat Immunol); Tox and Satb1 in pDC development; Tian 2021 found non-classic fate-correlated genes | consistent (Rodrigues/Dress abstracts read 2026-11-06; Satb1/Tox in pDC not yet checked) |

## Methodological lessons (how not to fool ourselves)

- **M8 (2026-11-05).** Descriptive fit-set patterns of about one CI width (B39 interaction, B40 persistence-by-fate) both failed to replicate on held-out clones. A registered reading met "weakly" is a hypothesis, not a discovery, until held-out clones agree.

- **M7 (2026-11-05).** Run a recovery check before fitting: in SIS-skew, clone-level selection is identifiable (R^2 0.96) but within-clone instruction is not quite (RMSE 0.36) and absorbs any baseline-label mismatch (bias 0.48). Know which parameter the data can carry before reading it (R1).

- **M6 (2026-11-05).** A sister-agreement ratio (H) is unstable when there is no memory (its denominator goes to 0); read it only through its CI. Check a "null" arm before using it as baseline: in SIS-skew it was not null (#15).

| # | lesson | evidence |
|---|---|---|
| M1 | Pass/fail at p >= 0.05 rewards wide predictions; compare models with a proper score (CRPS) and only over informative statistics. | B29, E47 amendment |
| M2 | Statistics computed on all clones leak the held-out shard; compute on the fit set. | B7 audit, B29 |
| M3 | A good fit to an aggregate statistic can come from compensating errors; decompose it (e.g. by clone size) before crediting it. | B31 (E46 co-presence) |
| M4 | A held-out test must be powered for the question: the Epo/SCF concordance cannot see selection. | B30, B33 |
| M5 | Check every model change against every condition before registering it; two designs were rejected this way before any fit. | E49 restricted class, E43 |

## Cross-check log

- 2026-11-06 (RETRO_2026-11-06): goal and literature audit. Literature columns of #1, #3, #13, #14, #15, #17 updated from abstracts (Lin 2018, Lin 2021, Naik 2013, Rodrigues 2018, Dress 2019, Cao 2016, Kelly 2019, Tsapogas 2014, Calabria 2024) and the Tian 2021 full text. #14 has a modelling gap (40% editing efficiency) and a direct tension (pDC fate gain).

- 2026-11-05 (RETRO_2026-11-05b): entries 6, 7, 11, 12 cross-checked; 7 moved to "in tension". B36 tested 12's candidate: SCF explained, Epo not.
- 2026-11-05 (B37): held-out look replicated discovery 9's small-clone co-presence deficit; strong vs plausible selection not decided.
