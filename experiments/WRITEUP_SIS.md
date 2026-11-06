# Does the LARRY model generalise? Chapter 2: dendritic-cell sister splits (SIS-seq, SIS-skew)

Working write-up, 2026-11-06 (final for this data set). Not a paper. Every test was pre-registered
in EXPERIMENT_LOG.md (B38-B42, R1-R4, E51-E54, C1, S1); discoveries #13-#18 and their literature
checks are in DISCOVERIES.md. Chapter 1 is WRITEUP_LARRY_CYTOKINE.md.

## Where this sits in the project

Cellware aims to model stem-cell fate behaviour in general and, in the long run, embryo
morphogenesis. The current step is to show that one model fits *conditional* data: the same
clones split across conditions. Chapter 1 (LARRY, G-CSF vs M-CSF) answered "mostly selection on a
strong clone fate state, modest instruction". Chapter 2 asks whether that model family carries to
a second system, and whether its hidden clone state can be tied to measured biology.

## The data

Tian et al. 2021 (Immunity, Naik lab), all public files (GEO GSE119097, GSE153676; supplementary
tables).

- **SIS-seq.** Barcoded mouse progenitors divide in culture; at day 2.5 each clone's cells are
  split: a few sisters are sequenced (RNA), the rest go into two separate wells and are read out
  for dendritic-cell fate (cDC1, cDC2, pDC). 158 clones, 105 with more than 3 cells in both
  sister wells.
- **SIS-skew.** The same split, but the two arms differ. In KO wells (Cas9 LSK cells), one arm
  carries a Bcor guide (Cherry+) and the other is wild-type sisters. In null wells (wild-type
  cKit+ progenitors) both arms are wild type, one with a control vector. About 830 KO and 450
  null clones in the fit set.
- A third of clones in each data set were sealed as held-out before any fate statistic (H2) and
  read only through `test_oracle.py`, once per test.

## The questions and the answers

| question | answer | strength |
|---|---|---|
| Is DC fate fixed in the clone before the split? (LARRY's main finding, second system) | **Yes.** Heritable share 0.71 [0.59, 0.78]; clone logit SD 2.0-2.7 | strong (#13) |
| Does Bcor loss instruct fate, or select clones by their state? | **Neither is detectable.** Bcor expands the KO arm about 10x and lets more one-arm clones persist; a fate-linked term is not identifiable, and the one marginal fate signal disappears under a plausible label shift | moderate, a null (#14, #18) |
| Can the hidden clone state be read from the sisters' RNA? | **Only weakly.** Held-out correlations 0.25-0.31, half the fit-set estimate | weak (#17) |
| Does one model cover both systems? | **Yes.** One core simulator reproduces LARRY's answer and the SIS-skew fits | strong for the machinery (C1, E53) |

## The evidence, in brief

1. **The clone fate state replicates (B38).** Sisters in separate wells agree on their
   cDC1/cDC2/pDC split far beyond chance, with no effect of the well's background. This is the
   same strong clone state that drives LARRY (beta 7-8 logits per SD), now in a second lineage
   and lab. Literature: consistent with Naik 2013 (sibling progenitors share DC fate) and Lin 2018
   (DC trajectories intrinsically programmed).
2. **The null wells are not null (B39, #15).** Control-vector sisters shift within clones away
   from cDC1. The W (wild-type cKit+) and C (Cas9 LSK) wells differ in population as well as in
   vector, so every Bcor effect is measured against an imperfect baseline. Tian 2021 used a binary
   test and did not report this.
3. **What Bcor does is growth and persistence (B39, B40, E51).** One-arm clones are 2-4x commoner
   in KO wells at every size; the KO arm grows about 10x (lam_K 2.3 [1.5, 3.2], consistent with
   Tian's mean 5x). The fit model passed its held-out look (24/29 statistics inside). A fit-set
   "KO-only clones are cDC1-poor" pattern did not replicate (#16, withdrawn).
4. **The fate term is not identifiable (R2/R3, E52).** With Bcor editing efficiency (40% in Tian
   2021) put into the model, a synthetic recovery check showed no fate parameter can be recovered
   at this sample size; the fit leaves selection on clone state (gamma) undetermined.
5. **Tian's "gain of pDC potential" is expansion plus sampling (B42, S1).** The KO net pDC gain
   (0.156) sat just above what the model gives with no fate effect (0.149). Shifting the W-vs-C
   label difference by +-0.5 logits moves that limit to 0.165-0.182, so the excess is within the
   model's uncertainty. Gamma stays undetermined under every labelling. Tian's reading came from
   a binary per-clone test with no model of sampling small wells; ours does not contradict their
   data, only the inference that fate potential changed.
6. **The RNA trace of the state is weak (B41, E54).** On the fit set a lymphoid-vs-myeloid
   priming axis (Satb1/Tox/Lrmp toward pDC, Csf1r/Grn toward cDC2) appeared to carry most of the
   state (loading 0.75-0.80; LARRY's day-2 score: 0.95). On 34 held-out clones the correlations
   were 0.25-0.31: positive for all three fates, but below the predicted range for pDC and cDC2.
   Pre-chosen marker programmes (Tcf4, Spib, Id2, Klf4) did not predict fate at all.
7. **One core model (C1, E53).** `core_model.py` (state, split, growth, sampling) reproduces the
   chapter-1 LARRY fit (instruction 0.52 / 0.70, gamma 0.56, beta 7.4, all inside E48's ranges)
   and the SIS-skew fits. This is the clone-level prior planned for the tissue simulator.

## What the chapter changes in the project's picture

- **Kept and strengthened:** a strong heritable clone fate state is the central quantity. It now
  holds in two systems, two labs and two lineage choices.
- **Not generalised:** the instruction-vs-selection split. LARRY could separate them because
  cytokines change growth by fate a lot and the data were large; SIS-skew's knockout mainly
  changes growth regardless of fate, the baseline wells differ, and the sample is too small to
  identify a fate term. A "selection" answer is not a property of the model family; it has to be
  earned per data set, and a recovery check must come first (lesson from R3).
- **Measured state, partly:** the hidden clone state is real (sisters agree) but only partly
  visible in a few sisters' RNA at day 2.5. For the spatial chapter, clone state should be treated
  as a latent variable estimated from outcomes, with RNA as a weak prior, not as a readout.

## Limits

- Small samples: SIS-seq 65 fit and 34 held-out clones; two of four batches have under 12 cells
  per well.
- The W-vs-C confound cannot be removed with these data; S1 bounds its effect only for +-0.5
  logits.
- E54's fit-set estimate was inflated because the gene space (PCA) was chosen on the same clones
  that were scored (lesson M10). The held-out figure is the honest one.
- One model family. Cell-level selection within a clone (fate-specific proliferation of
  precursors) and instruction are not separable from endpoint counts; every reading here is
  clone-level.

## Methodological lessons from this chapter

- **M6-M9 (DISCOVERIES.md):** read sister-agreement ratios only through their CI and check a
  "null" arm before using it as baseline (M6); run a recovery check before fitting (M7); fit-set
  patterns about one CI wide did not replicate (M8); put known experimental imperfections, such as
  editing efficiency, in the model before reading a parameter (M9).
- **M10:** leave-one-out does not protect the choice of feature space; only sealed clones give the
  honest effect size.

## What would settle the open parts

Matched-population null wells (Cas9 LSK with a non-targeting guide), more SIS-seq clones with
deeper sister RNA, or barcoded time-lapse of KO vs WT sisters. Recorded only: no outreach.

## Next

Chapter 3: Raj-lab monoclonal gastruloids (NEXT_DATASETS_2026-11.md, spatial rank 1), data
downloaded step by step. The core model supplies the cell-level clone state; the first test is
whether clone states plus local rules reproduce the clones' anterior/posterior propensities and
the pure-vs-mixed elongation difference.
