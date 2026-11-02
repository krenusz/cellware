# Cellware E5-E17 and wider programme: what a learned cell rule can and cannot explain about blood-cell fate

Write-up of experiments E5-E17 (November 2026). Details, pre-registrations and every run are in
[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md); held-out test looks are recorded in
[test_ledger.tsv](test_ledger.tsv).

## Question

Can a single decision rule shared by all cells, either written as equations or learned by a
neural policy, reproduce how sister cells from one blood stem/progenitor clone choose their
fates? We test against LARRY lineage-tracing data (Weinreb et al. 2020, Science), where clones
are barcoded, split at day 2 into different wells, mice or cytokines, and read out at days 4-6.

## Method

- **Simulators.** E5-E8: a spatial tissue simulator trained with PPO or evolution strategies (ES).
  E6 onward: a non-spatial clone simulator matching LARRY: cells divide daily, an undecided cell's
  fate state follows a pitchfork drift and commits to neutrophil (N), monocyte (M) or other (O).
- **Discipline.** Every experiment is pre-registered (predictions written before running). Each
  held-out statistic is looked at once, only after the fit passes pre-set adequacy gates, and
  then becomes training data. Since E11, test statistics are released only by
  `test_oracle.py`, which refuses repeat looks and unregistered experiments.
- **Compute.** Sweeps run as one GitHub Actions job per run (`run_sweep.sh`), free for a
  public repository, about 2.5x faster than the local PC.

## Results

### 1. The simulator and learners work (E5, E7, E8)

- **E5:** a totipotency window and an annealed entropy bonus removed training collapse (0/25).
  In the 20-seed rerun, knocking out the morphogen (score -0.20) and the gap junctions (-0.11)
  hurt, both CIs excluding 0.
- **E7:** cells can learn to read inherited memory. A cue given only to the founder is followed by
  86% of tissues four divisions later, against 50% without memory. The 20-seed rerun confirms a
  memory effect (-0.10). Sister concordance is identical with and without memory, so it is not a
  read-out of memory.
- **E8:** ES matches stabilised PPO (difference +0.03, CI -0.04 to +0.10) and never collapses.
  It is the trainer used from E10 on.

### 2. Faithful memory is ruled out; the environment matters (E6, E6b)

- Copying fate memory intact at each division overpredicts sister concordance across wells
  (0.86-0.89 vs observed 0.74). Fate bias carried by the founder's state and the timing of
  commitment fits within the held-out CI.
- Across mice, sisters agree far less (r = 0.46 [0.22, 0.68], 66 clones) than across wells. Every
  culture-calibrated model predicts 0.70-0.76. The missing factor is the environment.

### 3. A learned rule beats fitted equations (E9, E10)

Cytokine experiment: sisters split at day 2 into the full cocktail, G-CSF or M-CSF. Single joint
held-out look at cross-condition concordance:

| Pair | Observed r [95% CI] | E9 equations (resample / inherit) | E10 learned (resample / inherit) |
|---|---|---|---|
| full-G | 0.69 [0.40, 0.92] | -0.08 / 0.07 | 0.46 / 0.56 |
| full-M | 0.24 [-0.11, 0.55] | 0.15 / 0.17 | 0.20 / 0.07 |
| G-M | 0.51 [0.33, 0.65] | 0.19 / 0.14 | 0.11 / 0.03 |

The learned rule hits 4 of 6 predictions inside the CI, the equations 2 of 6. The reason is
diagnosable. The equations commit 35% of cells before day 2. The learned rule keeps 75%
undecided, close to the 71% in the data; this is post hoc evidence, not a test. Distilled, the
policy's rule is a restoring drift that delays commitment, plus a cytokine-specific push; it
ignores memory.

### 4. Every model underpredicts how strongly clones keep their bias (E11-E17)

G-CSF and M-CSF push in opposite directions, yet sisters in them agree (G-M r = 0.51).

| Exp. | Added mechanism | G-M in band (0.33-0.65) | Fit acceptable | Verdict |
|---|---|---|---|---|
| E11 | learned selection (divide / arrest), 5 conditions | 0/10 | yes | refuted |
| E12 | heritable clone trait tilting N vs M expansion | 0/10 | yes | refuted |
| E13 | learned founder fate bias in the drift | 1/10 | 9/10 | refuted; ES switches the bias off |
| E13 scan | fixed bias x cross weight | trade-off | - | concordance costs fit |
| E14 | fixed founder bias 0.35, cross weight 5 | 9/10 | 3/10 | refuted; M-CSF fit broken (M-only ~0.7 vs 0.88) |
| E15 | clone rank in the N/M choice, cytokine sets level | 1/10 | 10/10 | refuted; levels stuck near start (optimisation-limited) |
| E16 | E15 with levels started from the data (fair test) | 1/10 | 10/10 | refuted; best fits, clone rank shrunk again |
| E17 | conflict test: down-weight within-condition fit | max 0.46 | breaks | conflict plus a hard limit |

- **Ceiling.** With 4-5 sampled cells per clone, even a fully clone-set propensity gives G-M r of
  only about 0.36-0.47. The observed 0.51 is at the ceiling: in real cells the N/M propensity is
  essentially fixed per clone before day 2, broad across clones, and survives opposing cytokines.
- **Trade-off.** An additive clone bias competes with the cytokine. It buys concordance but drags
  cells out of M under M-CSF. Moving the bias into the N/M choice (E15-E16) avoids that
  competition but the optimiser still prefers a weak clone bias.
- **Conflict test (E17).** Trained on the cross statistic alone, the model reaches G-M of only
  0.46, with a within-condition fit 4-9x worse. So there is both a real conflict and a hard limit
  at about the sampling ceiling. The observed 0.51 rests on 55 clones; it is likely partly a high
  draw, with the true value near the ceiling (clone fate essentially fixed before day 2). This is
  an interpretation, not a test.

### 5. Where the clone bias lives (wider programme, tracks A-D and B3)

- **The "time" statistic was noise.** Every model overshot M-CSF's day-4 vs day-6 clone
  correlation (0.72-0.83 vs 0.32), and track A's late-commitment models did not fix it. A
  stable-clone simulation at the real sampling depth gives 0.81 [0.03, 0.99]; the observed CI
  is [-0.02, 0.88]. Under M-CSF almost every clone is pure M, so the statistic hangs on a handful
  of clones. Perfectly stable clones are compatible with the data.
- **Robustness (track C, 80 runs).** The ~0.3 limit on G-CSF vs M-CSF agreement holds across
  seeds, weightings and simulation size; weighting agreement more always costs fit. A stricter
  convergence check later showed most 150-generation runs were still improving, so key models
  are being rerun at 250 generations / 4000 clones.
- **Day-2 expression carries part of the bias (track B).** In LARRY in vitro, the expression of
  a clone's day-2 cells predicts its later N/M fate with CV r 0.66 (43% of variance; noise allows
  ~96%). More day-2 cells per clone do not help (r 0.67 / 0.62 / 0.68 for 1 / 2 / >= 3), so
  the unexplained ~55% is not a sampling artefact. Feeding the measured propensity into the
  simulator (B2) did not raise agreement (0.31).
- **Methylation carries more (track B3, EPI-Clone data).** In the same LARRY transplant
  experiments, with the same design, the methylation of a clone's multipotent cells predicts its
  myeloid / erythroid bias better than their expression: CV r 0.41 [0.30, 0.52] vs 0.25-0.28
  [0.13, 0.36], difference CI [0.01, 0.32], at similar noise ceilings, and unlike expression not
  through subtype composition. Different cells and clones, so suggestive, not a test.

## Conclusions

1. Learning the cell rule from data beats hand-written equations on held-out lineage statistics,
   because it recovers when cells commit.
2. Faithfully inherited fate memory is ruled out for this system; the environment matters across
   animals.
3. The best models fit within-condition data well and predict G-CSF vs M-CSF sister agreement
   of 0.23-0.34, at or just below the observed CI (0.33-0.65). No model in this family can fit
   both within-condition data and the full observed agreement; the statistic is at its sampling
   ceiling. The cytokine modelling line is closed for this dataset; deeper per-clone sampling
   would be needed to settle it.
4. The clone-level fate bias is partly visible in day-2 expression (~40%) and, in a second
   dataset, more visible in DNA methylation than in expression. The working hypothesis is an
   epigenetically held clone state set before day 2. The decisive test needs methylation,
   expression and lineage in the same cells (DARLIN Camellia-seq, GSE222246 / GSE222328).

## Held-out tests

| Test | Status |
|---|---|
| In vitro cross-well (E6) | spent |
| In vivo cross-mouse (E6b) | spent |
| Cytokine full/G/M cross-condition (E9+E10) | spent |
| Cytokine Epo/SCF pooled (reserved for a model passing its fit gates) | **unseen** |

## Limits

- One dataset and one statistic family; the cytokine cross-condition data are thin (wide CIs).
- 5 seeds per variant; ES optima are seed-dependent (E13 scan, E14 show two optima).
- Two of six cross-only E17 runs degenerated (no valid clones).
- E12 ceiling and E10 timing evidence are post hoc.
