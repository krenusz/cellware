# Do G-CSF and M-CSF instruct or select myeloid fate? An answer from LARRY clonal data

Working write-up, 2026-11-05 (final for this data set: modelling stopped under the stopping rule after E49/E50). Not a paper. Every test was pre-registered in EXPERIMENT_LOG.md;
discoveries and their literature checks are in DISCOVERIES.md; earlier work is in
WRITEUP_E5_E25.md and FINDINGS_2026-11.md.

## Where this sits in the project

Cellware aims to model stem-cell fate behaviour in general and, in the long run, embryo
morphogenesis. The current step is to show that a model can fit *conditional* data: experiments
where the same clones are split across conditions. This is the first such data set.

## The data

Weinreb et al. 2020 (Science), LARRY cytokine experiment. About 20,000 mouse LSK/LK progenitors,
roughly 30% lentivirally barcoded, grown 2 days in a broad cocktail. At day 2, 25% of cells were
profiled by single-cell RNA-seq; the rest were split evenly into wells with the full cocktail,
G-CSF, M-CSF, Epo (and, in the data, SCF), and sampled at days 4 and 6. Sister cells of one clone
therefore experienced different cytokines, and their day-2 relatives' transcriptomes are known.

## The question

When G-CSF yields neutrophils and M-CSF yields monocytes, do the cytokines **instruct**
undecided cells (shift each cell's choice) or **select** (favour survival and growth of cells
whose fate is already set)? The model separates the two: instruction is a per-condition shift of
the neutrophil/monocyte logit (tau); selection is a condition-specific advantage for neutrophil
output (s) and, in later models, a lineage-matched growth advantage for clones by fate bias
(gamma).

## The answer

**Both act, and selection on a fate state already present at day 2 carries most of the effect.**

| model (gated seeds) | instruction, G shift (logits) | instruction, M shift | G-CSF neutrophil advantage s_G | M-CSF s_M | lineage-matched selection gamma | clone fate-state effect beta (logits per SD) |
|---|---|---|---|---|---|---|
| E38, no lineage-matched selection (9) | 0.49 [0.38, 0.68] | 0.47 [0.40, 1.70] | 4.7 [4.1, 5.3] | 0.11 | 0 | 6.6 |
| E47, gamma <= 0.35 (9) | 0.47 [0.39, 1.40] | 0.62 [0.46, 1.15] | 3.0 [2.6, 3.8] | 0.19 | 0.31 | 7.7 |
| **E48, gamma <= 0.6, working model (9)** | **0.50 [0.38, 0.70]** | **0.56 [0.40, 0.92]** | **2.1 [1.6, 2.9]** | **0.23** | **0.56** | **7.8** |
| E46, unbounded gamma (16) | 0.86 [0.41, 7.5] | 4.8 [0.6, 7.5] (ridge) | 0.6 | 0.69 | 1.29 | 8.9 |

Medians over gated seeds, [min, max].

- **Instruction is small and stable:** about 0.5 logits each way in every biologically plausible
  model (E38, E47, E48). At an even baseline, 0.5 logits moves a cell's neutrophil probability by
  about 12 percentage points.
- **The clone's own state dominates:** a one-SD difference in a clone's day-2 fate state is worth
  7-8 logits, about 15 times the cytokine's instructive shift.
- **Selection is strong:** under G-CSF, neutrophil output is favoured about 2-5x (s_G; the uniform
  part shrinks as the lineage-matched part grows). Under M-CSF it is disfavoured about 5-9x.
  Lineage-matched selection sits at the biological limit (gamma ~0.56; Rieger 2009 bounds it at
  about 0.3-0.6).
- **Only implausible selection changes the instruction estimate.** E46's unbounded gamma (1.3,
  about twice the plausible bound) moves the M-CSF instruction onto a ridge (0.6-7.5). Within the
  plausible range, the answer does not depend on how selection is modelled.

## The evidence, in brief

1. **Fate-selected presence (B7).** Clones present under both G-CSF and M-CSF are
   monocyte-leaning under G-CSF; clones seen only under G-CSF are neutrophil-pure. Models that fix
   which clones are present cannot produce this.
2. **The fate state exists at day 2 (B9-B16).** Day-2 Cebpa/Gfi1 vs Irf8/PU.1 expression predicts
   later fate and presence (AUC 0.81), and replicates on the main LARRY data set.
3. **Pure instruction fails (E40).** With selection switched off, no fit passes, even with very
   strong instruction.
4. **Plausible selection fits the key pattern (E48).** Lineage-matched selection capped at the
   Rieger bound reproduces B7 G, and the fit pushes gamma to the cap.

## Literature

- **Consistent:** Rieger 2009 (time-lapse: M-CSF and G-CSF instruct a majority of sorted GMPs,
  and early death is low and equal in both, so selection must act through division rates);
  Mossadegh-Keller 2013 and Endele 2017 (M-CSF/CSF-1 instructs); Kimura 2009 (GM-CSF/STAT5
  controls proliferation and survival); Weinreb 2020 Supp Fig 7g (lineage-matched expansion,
  survival "cannot be ruled out").
- **Reconciliation:** the single-cell imaging studies use sorted GMPs, a later and more uniform
  stage; LARRY starts from LSK/LK cells, many of which already carry a fate bias at day 2. Both
  pictures hold: cytokines can instruct, but in a mixed progenitor population most of the
  difference between conditions comes from which clones grow.

## Limits

- **One model family, one data set.** The ratio is an estimate within this family.
- **Presence features the model does not reproduce** (each recorded in DISCOVERIES.md, none
  moves the instruction estimate under the stopping rule). Two targeted models failed: a limited
  sister split (E49, 6/18 gated) and a neutrophil-committed G-only class (E50, 0/10 gated; it
  removes too many mixed G clones):
  - small M-CSF clones private to the M-CSF well (B32, B34);
  - neutrophil-committed G-CSF clones private to the G-CSF well (B34);
  - clones that were rare at day 2 reach fewer wells (B35);
  - Epo wells hold a largely separate, erythroid-lineage clone set (B36).
- **The full-cocktail arm** is fitted as drawn from a separate clone subset (about 22%). The paper's
  design says the day-2 cells were split evenly into four wells, full cocktail included, so this
  is a fitted property, not the design. An alternative is that stem-like cells grow mainly in the
  full cocktail. Like Epo, the arm may look separate because it supports lineages the G/M wells
  do not.
- **Held-out test not yet spent.** The original held-out statistic (Epo/SCF concordance) cannot
  separate the models (B30). A presence-based shard-1 look can separate strong selection (E46)
  from plausible or none (B33). It cannot separate plausible selection from none.
- **No absolute rates.** Division and death rates per cytokine are bounded from the literature,
  not measured here.

## What would settle it

Per-clone survival and division under each cytokine (time-lapse with barcodes), deeper sampling
of M-CSF wells, or sister splits read out at matched times. Recorded only: no outreach.

## Next

1. The held-out look (strong vs plausible/no selection), pre-registered with E46 and E48
   predictions.
2. Choose the next conditional data set (other progenitor systems or perturbations), reusing the
   protocol: fit-set scoring, every-condition checks, the stopping rule and the discoveries
   register.
