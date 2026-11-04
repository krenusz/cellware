# Cellware findings, November 2026: how G-CSF and M-CSF shape neutrophil vs monocyte fate

Working notes, not a paper. One data set (Weinreb et al. 2020, LARRY, cytokine perturbation
experiment), one family of models. Every test below was pre-registered in EXPERIMENT_LOG.md
before it ran. The earlier programme (E5-E25, methylation, in vivo) is in WRITEUP_E5_E25.md.

## The question

Day-2 sister cells of one barcoded progenitor clone were split into wells with the full cocktail,
G-CSF, M-CSF, Epo or SCF, and read out at days 4 and 6. Sisters under G-CSF and M-CSF still agree
on their neutrophil:monocyte balance (r 0.51). Do the cytokines **instruct** fate (tell
undecided cells what to become) or **select** it (let cells of an already-set fate survive and
expand)?

## Findings, by strength of evidence

### 1. Cytokines act mainly by selection on a fate state that already exists (strong)

- **The G-M clones are fate-selected (B7).** Clones present under both G-CSF and M-CSF are
  monocyte-leaning under G-CSF (mean N-share 0.35) while G-only clones are neutrophil-pure (0.72).
  Every simulator that fixed which clones are present predicted no difference.
- **The state is already there at day 2 (B9, B11, B13-B15).** Clones that keep making neutrophils
  under M-CSF were neutrophil-primed at day 2 (partial rho 0.24, p 1e-5). The Cebpa/Gfi1 vs
  Irf8/PU.1 programme carries it (Cebpa 0.28, Gfi1 strongest); receptor balance Csf3r - Csf1r
  passes all three primary tests (rho 0.14-0.19). Presence in a condition is readable from
  day-2 expression at AUC 0.81.
- **It replicates (B16).** On the main in vitro LARRY data (571 clones), Cebpa - Irf8 predicts
  fate at rho 0.30 and Gfi1 at 0.34.

### 2. A model with measured day-2 state, modest instruction and strong selection reproduces the G-vs-M structure (moderate)

- Inputs are measured, not free: each clone's day-2 Cebpa - Irf8 score and committed fractions
  (E33). Instruction signs are fixed by the literature (E35).
- **Estimates are stable across seeds** (E38, 18 seeds): neutrophil survival is about 4-5x under
  G-CSF and about 0.1x under M-CSF, with instruction modest (tau shifted about 0.6-1.3 logits).
  This is consistent with the published single-cell evidence for instruction (Rieger 2009,
  Mossadegh-Keller 2013): both happen, but selection carries most of the effect here.
- G-M agreement, B7 M, and which M-CSF clones carry the rare neutrophils (B19, 0.51 vs model
  0.53) all match.
- **Not a stable pass.** Under the pre-registered stability rule, no model passes every primary
  component in enough seeds. G-M splits (5/9 gated seeds in E38), and B7 G undershoots (-0.27
  vs -0.37, 7/9 fail). The rare full passes (E35 s0, E38 s8, E39 s11) sit in different corners
  of parameter space.

### 3. The full-cocktail culture was seeded from a largely separate set of clones (moderate)

- Full-arm clones are co-present with G and M at 0.3-0.4x independence (B12). Every model drawing
  all arms from one pool overshoots that co-presence (E32, E34, E36).
- A clone-level pool split (E37) fixes it. In E38, 15/18 seeds find the same basin: about 22% of
  clones seed mainly the full culture, with about 10% leak, and the full arm's clone count is
  right in every gated seed.
- **This is an inference from structure.** The paper's design figure (Science Supp Fig 7) was
  not read.

### 4. What does not explain the remaining misfits (pre-registered negatives)

- **Sampling day (B18).** Day-matched full6-M6 agreement is still 0.04 (76 clones). Same-day
  G-M is higher than different-day (0.50 vs 0.23), but not significantly (CI of the difference
  -0.12 to 0.74). Day-matched full6-G6 is 0.92 (12 clones): the full cocktail behaves like G-CSF.
- **Clone-level M-CSF expansion (E39).** Making a whole neutrophil-leaning clone fail to expand
  under M-CSF fixes the composition of the G-M clones (B20's C1) but the fit uses it in place of
  per-cell selection, so G-M overshoots (0.62-0.75). 1/18 gated.

### 5. The limits of this data set

- **Full-M (0.24) is below resolution.** It rests on about 8 informative clones, the M-CSF
  clones in the full arm that carry any neutrophil (B19). M-CSF day-6 wells hold 114 neutrophils
  against 4,526 monocytes.
- **B7 G's shortfall is localised (B20)**: too many G-M co-present clones in the model (114 vs
  78), and they are too neutrophilic (0.45 vs 0.35). G-only clones fit.
- **The held-out test (Epo/SCF shard 1) is still unseen.** No model's prediction for it is
  stable enough to make the look informative (E38: gated shard-1 IQR 0.65).

## How sure are we

| claim | evidence | status |
|---|---|---|
| Selection on a pre-existing state dominates | B7, B9, B11, B13-B16, stable s estimates | strong |
| Day-2 Cebpa/Gfi1 vs Irf8/PU.1 state predicts fate and presence | B11, B13-B15, replicated B16 | strong |
| Some instruction in the literature's direction | E35 improves fit with signs fixed | moderate |
| Full culture seeded from a separate clone subset | B12, E32-E38 | moderate, design unverified |
| A single model fits everything | E35-E39 | **not shown** |

## What a new experiment would need

The data cannot settle full-M or the instruction:selection ratio. The decisive measurements are
per-clone survival or death under each cytokine (time-lapse or apoptosis reporters), deeper
sampling of M-CSF wells, or a design that splits sisters into G-CSF and M-CSF at matched times
with both read out. Per the project rule, no outreach: this is recorded, not pursued.

## Next analytical steps (no new data)

1. **Map the ridge instead of point-fitting.** Profile the loss along the G-CSF selection vs
   instruction axis, or use simulation-based inference, to state the range of compatible
   instruction:selection ratios (LITERATURE_SWEEP_2026-11d, Xu et al.).
2. **B21:** a fluctuation-test analysis (Luria-Delbrück style) of N-share variance across sister
   wells and days, a simulator-free estimate of fate memory.
3. **B22:** do M-only clones carry a monocyte-DC progenitor (MDP) day-2 signature (Yáñez 2017)?
   That would explain low full-M without any cytokine effect.
