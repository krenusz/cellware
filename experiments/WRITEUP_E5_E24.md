# Cellware E5-E24 and wider programme: what a learned cell rule can and cannot explain about blood-cell fate

Write-up of experiments E5-E24 and tracks A-B6 (November 2026). Literature context: [LITERATURE_SWEEP_2026-11.md](LITERATURE_SWEEP_2026-11.md). Working notes, not a paper; the line is open. Details, pre-registrations and every run are in
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
  ~96%). It survives holding out whole libraries (0.67) and removing starting population,
  library and clone sizes (0.60; B1c). More day-2 cells per clone do not help (r 0.67 / 0.62 / 0.68 for 1 / 2 / >= 3), so
  the unexplained ~55% is not a sampling artefact. Feeding the measured propensity into the
  simulator (B2) did not raise agreement (0.31).
- **Methylation carries more (track B3, EPI-Clone data).** In the same LARRY transplant
  experiments, with the same design, the methylation of a clone's multipotent cells predicts its
  myeloid / erythroid bias better than their expression: CV r 0.41 [0.30, 0.52] vs 0.25-0.28
  [0.13, 0.36], difference CI [0.01, 0.32], at similar noise ceilings, and unlike expression not
  through subtype composition. Different cells and clones, so suggestive, not a test.
- **Same cells, same clones (track B5, DARLIN Camellia-seq).** In 182 HSCs from 82 clones
  (mouse x clone label; labels repeat across mice) with methylation and RNA in the same cell,
  sister cells are recognisable from methylation (AUC 0.63, null 0.55) but not from expression
  (0.51, at its null). Pre-registered. Controlling for global methylation, coverage and
  proliferation leaves it unchanged (0.62). Identical sisters would score 0.95 at this depth, so
  the shared part is ~28% of the maximum.
- **What kind of signal it is (B6, B6d, revised reading).** The clone signal sits only in
  genome-wide 100-kb bins (0.63). Windows around 40 lineage genes (0.50), promoters (0.52), gene
  bodies (0.53) and random gene sets (~0.52) carry none. It holds when each mouse is left out
  (0.59-0.63) and across plates as well as within them (0.63 vs 0.62), so it is not a batch
  effect. On 11 labelled clones, methylation does not predict Meg bias better than expression
  (0.60 vs 0.70). **Reading:** the Camellia signal looks like a heritable, large-scale,
  largely neutral record of ancestry, not a fate-instructive state at lineage genes. This
  replaces the earlier "drifting clone state" reading.
- **Where EPI-Clone's fate prediction comes from (B3e-B3g).** Using the authors' own site
  labels, cell-state ("Dynamic") CpGs predicted myeloid bias better than clone-record ("Static")
  CpGs (0.39 vs 0.24), but at matched panel size the gap disappears (Static at the 22nd
  percentile of size-matched Dynamic sets). Sisters share Static sites very strongly (AUC 0.93)
  and Dynamic sites clearly (0.66 size-matched), yet are no closer in pseudotime than random
  same-type cells (0.49). Against a record-only benchmark built from the data (one half of the
  record sites conditioned on the other), cell-state-site sharing net of the record is far below
  what the record alone produces (0.59 vs 0.84), so record leakage explains it (B3h). Fate
  information beyond the record at cell-state sites is suggestive but does not clear its
  benchmark. **Then the confound check (B3i):** with cross-validation leaving one mouse out, or
  with mouse and clone size removed, the record sites' fate prediction disappears (0.10 and 0.01,
  CIs include 0). Cell-state sites keep it (0.24-0.25). Mouse and clone size alone predict as
  well as methylation did (0.42). **Reading:** the heritable record is neutral in both datasets.
  The fate information sits in cell-state methylation of the multipotent cells, beyond mouse and
  clone size. Earlier B3-family numbers are inflated by the confound. Per site (B3j), three
  enhancer CpGs pass FDR 10% and 24 reach p < 0.01 (4.5 expected); descriptively, sites the panel
  picked as MPP-high lean myeloid and HSC-high sites lean erythroid, i.e. clones whose
  multipotent cells look further along are more myeloid. EPI-Clone's own test calls 93% of the
  "cell-state" sites clone-associated, confirming the leakage reading. On a pre-defined
  whole-class index (B3k), centred within cell type and controlling mouse, size and multipotent
  composition, more MPP-like methylation goes with more myeloid bias: 0.19 [0.04, 0.33], positive
  in 5 of 6 mice (two near zero). A narrow confirmation on the discovery data, not a replication.
- **Models built on this (E18, E18b, E18c).** E18 tied LARRY sister fate sharing to the Camellia
  signal through a drift rate. One grid cell reproduced all three cross-condition statistics
  (G-M 0.42, the best in the programme), but only with fast drift and near-deterministic fate.
  E18b found fetal-liver and bone-marrow sisters share about the same record (rho 0.18 vs
  0.21) despite very different division counts, so constant drift was rejected: the record looks
  stable. E18c drops the link: a neutral
  record (Camellia) plus a separate, unmeasured fate state.
  Profiling the fate-state sharing directly, settings from 0.6 to 0.95 put all three cross
  statistics in or at the edge of their CIs (G-M 0.33-0.39). That range includes E18's value
  (0.86), so E18 never needed the Camellia link. LARRY alone does not pin the fate-state sharing,
  and G-M still sits near the bottom of its CI.
- **Kin correlations pin the missing parameter (E19).** Using two kinship distances in LARRY
  in vitro (same well vs different wells, kin-correlation idea from Hormoz et al. 2015/2016), the
  latent fate correlation is 0.94 within a well and 0.83 across wells, so the fate-state sharing
  between sister branches is 0.88 [0.82, 0.92]. That narrows E18c's range and matches its
  near-deterministic fits. With it, the model still predicts G-M ~0.33-0.39.
- **The gap is real (E20, E21).** With fate-state sharing fixed at 0.88, 200 simulations at the
  observed sampling depth give G-M 0.25 +/- 0.06; none reaches the observed 0.51, while full-G
  and full-M are reproduced. Letting each condition have its own gain raises G-M to 0.33 but
  breaks full-M and still never reaches 0.51. The cross-condition limit is not a sampling ceiling.
  With a warm-started optimiser (E21b) the gain model fits the within-condition data far better,
  and G-M drops to 0.21. A second clone trait acting only under single cytokines (E22) is fitted
  to ~0 by the within-condition data and leaves G-M at 0.21. Across E15-E22, the better the
  within-condition fit, the lower the predicted G-M: the two parts of the data pull against each
  other in every model of this family. Clone size coupling (E23; the original paper notes
  sisters shift clone size and fate together) and carry-over of cells already committed before
  the split (E24; 16% of day-2 cells are Neu/Mo) both fail too: G-M stays at 0.20-0.21.
  The G-M statistic rests on 78 clones. A second literature sweep
  ([LITERATURE_SWEEP_2026-11b.md](LITERATURE_SWEEP_2026-11b.md)) points to clone *types*
  (intrinsically fated vs cytokine-plastic clones; two monocyte routes) as the best-supported
  untested explanation.
- **Clone control is much weaker in vivo (E19b).** In transplanted mice, cells of one clone agree
  on N vs M only weakly even within the same mouse (latent r 0.22, vs 0.94 in culture), and about
  as much across mice. The E6b cross-mouse drop is a general loosening in vivo, not a
  recipient effect.

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
4. The clone-level fate bias is partly visible in day-2 expression (~40%) and in cell-state DNA
   methylation of multipotent cells (r ~0.25 beyond mouse and clone size, B3i). Clones also
   carry a heritable methylation record, but it is neutral in both datasets: large-scale,
   heterochromatic, not at lineage genes, and not fate-predictive once mouse and clone size are
   controlled. The record part replicates published work (Camellia-seq, EPI-Clone).
5. Sister branches in culture share ~88% of their latent fate state (E19), and fate is
   near-deterministic given it. A simulator with these values fits within-condition data and
   two of three cross-condition statistics, but G-CSF vs M-CSF agreement is out of its reach
   (p < 0.005, E20), also with per-condition gains (E21, E21b) or a second clone trait (E22).
   Better within-condition fits push it further away. That gap is a real, unexplained feature.
6. In vivo, the clone sets the N/M choice far less than in culture (E19b).

## Open (deliberately not closed)

- What is the fate-relevant cell-state methylation (B3i survivors), and when is it set?
- What links a clone's G-CSF and M-CSF outcomes more than either to the full cocktail? Not
  sampling (E20), per-condition gain (E21, E21b), a fitted second trait (E22), size coupling
  (E23) or pre-split carry-over (E24). Next: a mixture of clone types (fixed vs plastic), and
  whether the 78 G-M clones are a selected subset.
- Does the MPP-like methylation axis (B3k) replicate in independent data?
- What sets the fate state, and when?
- Division counts: Camellia barcodes were induced at E10.0 and sampled ~9 weeks later (Li et al.
  preprint). Not yet used.
- The Epo/SCF shard-1 held-out test is still unseen. It is not powered to separate the
  current models (~24 clones).

## Held-out tests

| Test | Status |
|---|---|
| In vitro cross-well (E6) | spent |
| In vivo cross-mouse (E6b) | spent |
| Cytokine full/G/M cross-condition (E9+E10) | spent |
| Cytokine Epo/SCF shard 0 (E14) | spent: 0.44 [0.03, 0.78], non-discriminating |
| Cytokine Epo/SCF shard 1 | **unseen** (power rule: skipped in E18) |

## Limits

- One dataset and one statistic family; the cytokine cross-condition data are thin (wide CIs).
- 5 seeds per variant; ES optima are seed-dependent (E13 scan, E14 show two optima).
- Two of six cross-only E17 runs degenerated (no valid clones).
- E12 ceiling and E10 timing evidence are post hoc.
