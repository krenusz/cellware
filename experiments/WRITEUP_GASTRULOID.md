# Clone states in a self-organising tissue. Chapter 3: gastruloid clones (Raj lab)

Working write-up, 2026-11-07 (chapter closed on this data set). Not a paper. Every test was
pre-registered in EXPERIMENT_LOG.md (G1-G12); discoveries #20-#27 and their literature checks are in
DISCOVERIES.md; process failures and the literature map are in RETRO_2026-11-07.md. Chapters 1 and 2
are WRITEUP_LARRY_CYTOKINE.md and WRITEUP_SIS.md.

## Where this sits in the project

Cellware aims to model stem-cell fate and, in the long run, embryo morphogenesis. Chapters 1-2
showed that a heritable clone state dominates fate in two blood systems. Chapter 3 asks the first
spatial question: in a self-organising mouse gastruloid, does a clone's state decide *where* its
cells go along the head-tail (anterior-posterior, A-P) axis, can we simulate that, and how stable is
the state?

## The data

Ayyappan et al. 2025 (Raj lab), public line scans and tables. Mouse ES clones carry fluorescent
labels; gastruloids are grown from one clone, a clone in unlabelled bulk, or two clones mixed. Per
gastruloid, a line scan along the axis gives each clone's signal profile.

- **Polarity P:** profile resampled to 30 points, z-normalised, smoothed, first-half minus
  second-half mean (positive = anterior). Pairs: D = P(red clone) - P(green clone).
- Experiments span series from Jan 2024 to Jul 2025: clone in bulk over passages, subclones, 1+3,
  1+6, 3+7, 6+7 pairs, drug treatments (RA, SB431542, AGN193109), Wnt "hi/lo", elongation scoring
  (3,597 gastruloids).
- **Hold-out.** First a sealed file set (G1-G7, rules H1-H10). After two failures (below), every
  remaining source was split into train 40% + sealed partitions A/B/C, with a train-fit selection
  file (H11-H13) and a power gate (H14) before any look.

## Results

| # | claim | evidence | status |
|---|---|---|---|
| 20 | A clone's A-P preference is a stable clone state (clone explains 0.69 of held-out polarity; batch 0.04) | G1a sealed, G6b second sealed series | **strong** |
| 26 | Clone states + rank sorting predict how two unseen clones order (6+7: 12/12; 3+7 barely sort, as predicted) | G7a sealed | **moderate** (magnitude off for 6+7) |
| 23 | Preference fades over passages, but the rate depends on series x clone; no model predicts a new series | G3, G6, G7b sealed, G8 | moderate (direction); **rate unpredictable** |
| 27 | Perturbing signalling erodes the division of labour: conditional on a series' own control, drugs multiply clone separation by ~0.35 (one shared factor); high Wnt erases propensity | G9, G11 (train + sealed A), G12 (gate) | **unconfirmed: consistent, not testable with these data** |
| 21 | Domain size is set by the gastruloid, not the clone mix | G1b open | unconfirmed |
| 22 | Any mixing rescues elongation (+0.2-0.3), small complementarity bonus | G2, G2b open | unconfirmed |
| 24 | Rescue is not buffering, strict complementation or T+ area | G4 open | unconfirmed (negative) |
| 25 | Memory is lost at passaging, not per division | G5 open | unconfirmed |

### The simulator

A 1D A-P model (simulator v2, `g7_model.py`): each clone has a state; cells sort by rank of state;
proportion feedback sets domain size; gastruloid-level noise (tau 0.26). The **sorting part passed a
sealed test** (G7a). The **memory part failed** (G7b): neither one fade rate nor a clone-specific rate
predicts a new series, and no replacement qualified on train data (G8). Elongation rescue is a
descriptive rule; its mechanism is open.

### Drug compression: why it stays unconfirmed (G9, G11, G12)

- Train (Dec 2024, Feb/Mar 2025): every drug reduces 1+3 and 1+6 clone separation in 11/11
  comparisons. Predicting each drug from its own series' control, one shared factor 0.35 wins
  leave-one-series-out CV; additive shifts and "no effect" lose by 17-32 log-likelihood units. So the
  series' strength and the drug effect multiply.
- Sealed partition A (May 2025): RA 0.24, SB 0.33 of control, on the prediction (|z| <= 0.15), but
  only 3 controls; the pre-registered CI test failed. Power was ~15%, which nobody checked before
  the look.
- Partition B (Jul 2025): 22 controls, 14 drug-treated. The new power gate (H14) gave 0.54 for the
  pooled one-sided test and refused the look; **B and C stay sealed**. Per-gastruloid SD (~1.6) is
  large relative to the effect (~0.9): reaching 80% power needs about 24 controls and 48 treated
  gastruloids in one series, more than any series here has.

## What this chapter established

1. **A heritable clone state generalises to a third system** and now to space: it decides where a
   clone's cells go in a self-organising tissue (strong).
2. **Sorting by state is predictive** for unseen clone pairs (moderate): the first validated spatial
   rule in the project.
3. **The state's strength is not fixed.** It varies by experiment series in fade (G8), in pair
   separation (G10: 1+3 separation 3.0 / 1.3 / 1.8 across series) and under signalling perturbation
   (G9/G11). A "series gain" that scales every clone difference fits all three on train data; it is
   not confirmed on sealed data. The Cell 2026 monoclonal-gastruloid paper (heritable biases at
   imprinted, methylation-sensitive loci) suggests a culture-history mechanism.

## What is open

- **Memory:** what sets the fade rate. Needs culture-history metadata or epigenetic data.
- **Elongation rescue (division of labour):** why mixing rescues; no elongation readout in the
  untouched files. Candidate mechanism: Wnt-responsive cells (Suppinger 2023), untested.
- **Domain size (#21):** DiffAssay data locked; partition before any statistic.

## Process: what went wrong, what is fixed

| failure | fix |
|---|---|
| G5 read sealed rows outside the oracle | loaders exclude sealed data; guard H1-H10 |
| G7 spent all remaining same-design hold-out in one round | partitions + train-fit selection + reserve (H11-H13) |
| G11 spent partition A at ~15% power | power gate from counts only (H14); G12 shows it refusing B |

## Data still sealed

Raj reserve (clone3% passage 3, Jul 2024; last Raj sealed file, needs Bence's OK); drug partitions
B (Jul 2025) and C (May 2025 overlap metrics); Wnt partitions A-C; GSE291244 partitions A-C (chapter
4 candidate); locked: DiffAssay, AP_CloneFraction, GFP-spike, sgbc2.
