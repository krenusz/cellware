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
- **Simulator-free check (B21, fluctuation-test logic).** Cells of a clone are strongly alike in
  N vs M in every condition and day (beta-binomial intra-clone correlation 0.62-0.86), so cells do
  not decide independently with a condition-wide probability. Replicate libraries agree at
  0.84-0.91 under G-CSF and the full cocktail. Caveat: induction in the first divisions after
  plating, followed by expansion, would look the same.
- **Pure instruction under G-CSF is excluded by the model profile (E40).** With G-CSF selection
  switched off (s_G = 1 or 1.5), no fit passes the gate even with very strong instruction; G-M
  agreement falls to 0.21-0.33 and the fate-selected presence pattern disappears.

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
- **The corners are a bounded ridge (E40).** Fixing G-CSF neutrophil survival s_G and refitting
  everything else, gated fits exist at s_G 2.5, 4 and 6.5, with instruction compensating exactly
  (3.7 logits at 2.5, about 0.5 at 4-6.5; Spearman -1.0). Fits fail at s_G <= 1.5 (too little
  selection) and at 10 (G-CSF wells too neutrophilic). The best fits are at 4-6.5x. **Headline
  range: a 2.5-6.5x neutrophil survival/expansion advantage under G-CSF, with 0.5-3.7 logits of
  instruction, the small-instruction end fitting best.** B7 G fails at every grid point, so its
  misfit is not a ridge effect.
- **Fate memory fades slowly (B21).** Under G-CSF, day-4 vs day-6 agreement is 0.68 of same-day
  agreement (CI 0.40-0.92), a half-life of about 3.6 days. No simulator includes this yet.

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
- **A separate monocyte-DC progenitor lineage (B22).** M-CSF-only clones carry no stronger day-2
  MDP signature (Flt3, Cd74, MHC-II) than M-CSF clones seen elsewhere, once the Cebpa - Irf8 axis
  and size are controlled (partial rho -0.01, p 0.90, 246 clones).

### 5. The limits of this data set

- **Full-M (0.24) is below resolution.** It rests on about 8 informative clones, the M-CSF
  clones in the full arm that carry any neutrophil (B19). M-CSF day-6 wells hold 114 neutrophils
  against 4,526 monocytes. Even replicate M-CSF libraries agree at only 0.28 (B21), so agreement
  statistics built on M-CSF wells mostly measure noise.
- **Agreement has a ceiling.** Replicate libraries of one condition agree at 0.85-0.91 (B21), so
  G-M agreement (0.51) is about 60% of the attainable maximum.
- **B7 G's shortfall is localised (B20)**: too many G-M co-present clones in the model (114 vs
  78), and they are too neutrophilic (0.45 vs 0.35). G-only clones fit.
- **The held-out test (Epo/SCF shard 1) is still unseen.** No model's prediction for it is
  stable enough to make the look informative (E38: gated shard-1 IQR 0.65).

## How sure are we

| claim | evidence | status |
|---|---|---|
| Selection on a pre-existing state dominates | B7, B9, B11, B13-B16, B21, stable s estimates | strong |
| Pure instruction under G-CSF cannot produce the data | E40 (s_G <= 1.5 fails) | strong (within this model family) |
| G-CSF neutrophil advantage 2.5-6.5x, instruction 0.5-3.7 logits, trading off | E40 ridge | moderate |
| Day-2 Cebpa/Gfi1 vs Irf8/PU.1 state predicts fate and presence | B11, B13-B15, replicated B16 | strong |
| Fate memory fades slowly (half-life about 3.6 days under G-CSF) | B21 | moderate |
| Full culture seeded from a separate clone subset | B12, E32-E38 | moderate, design unverified |
| M-CSF-only clones are a separate MDP lineage | B22 | not supported |
| A single model fits everything | E35-E40 | **not shown** |

## What a new experiment would need

The data cannot settle full-M or the instruction:selection ratio. The decisive measurements are
per-clone survival or death under each cytokine (time-lapse or apoptosis reporters), deeper
sampling of M-CSF wells, or a design that splits sisters into G-CSF and M-CSF at matched times
with both read out. Per the project rule, no outreach: this is recorded, not pursued.

## Next analytical steps (no new data)

Done on 2026-11-04: the ridge profile (E40), the fluctuation analysis (B21) and the MDP check
(B22). Remaining:

1. **E41b (done, negative):** fading fate memory is not needed (median r46 0.915) and does not
   fix B7 G (0/9 gated seeds). Near-full memory already reproduces B21's day-4 vs day-6 ratio once
   17-clone sampling noise is simulated, so B21's 0.68 is not evidence of real fading. s_G stays
   on the ridge (median 5.1). G-M agreement now fails in all 9. **Next: B23**, testing whether
   day-2 Gfi1 predicts the G-CSF benefit beyond Cebpa-Irf8 (Sierra 2010: Gfi1 gates G-CSF
   responsiveness), as a data diagnostic before any model.
2. **B23-B25 (done).** B23: day-2 Gfi1 adds to G-CSF N-share beyond Cebpa-Irf8 (partial
   rho 0.17, p 0.01) but does not explain B7 G (2% shrink). B24: B7 G sits entirely in 2-cell
   G-CSF samples (-0.47 vs sim -0.22) and in a small set of clones that are large under M-CSF
   with only 2-3 monocytes under G-CSF (15% vs 2%). Not same-day contamination (the other-day gap
   is as large), and the model makes too many shared clones (57 vs 39). Best candidate:
   M-biased clones are dropped (die) under G-CSF more often than the model allows. B25: the low
   M-CSF replicate agreement is sampling (null 0.37 [0.11, 0.64], observed 0.28), so the M-CSF
   statistics need no down-weighting.
3. **Held-out test:** still unspent; spend it only on a model whose shard-1 prediction is stable.

## Correction (2026-11-04, fit-set audit of B7)

B7's data value and simulated interval had both been computed on all clones, held-out shard
included. Re-scored on the fit set only, **B7 G is a marginal misfit**: E38 passes it in 5/9
gated seeds (was 2/9) and the full primary in 3/9 (was 1/9); E40 passes it at s_G 4. Earlier
statements that "B7 G fails at every grid point / in every model" are withdrawn. E38 is still not
stable, and E41b's discard stands. B24's fit-set patterns (M-large / G-tiny clones 0.15 vs 0.02;
57 vs 39 shared clones) stand. E42 probes: neither clone dropout nor shrinkage of M-biased clones
under G-CSF can produce them (both move B7 G the wrong way).


## Update (2026-11-05): E43-E45 negative; a presence vs B7 G trade-off

E43 and E44 (an M-committed clone class) and E45 (limited sister cells split between wells) all
fail the protocol; E38 stays the reference. E45 fixes most of B27's presence misfit (G-M lift
0.47 -> 0.31-0.39, M-present clones right) but gates only 3/13 seeds and loses B7 G in all of
them. Shared G/M presence and the B7 G difference trade against each other in every model tried.

## Update (2026-11-05, later): E46, selective expansion fits much better

E46 adds one parameter: each cytokine expands the clones whose fate bias matches its lineage
(G-CSF neutrophil-leaning, M-CSF monocyte-leaning). It gates 16/18 seeds (E38 9/18), passes B7 G
in all gated seeds, fits G/M co-presence, and gamma is consistent (1.2-1.35) in every seed. It
misses the stability rule only on the full-M component (7/16), and still has too few M-present
clones and too few large M-pure clones under G-CSF (D3). E38 stays the formal reference. Weinreb's
Supp Fig 7g shows the same pattern descriptively (read after E46 was registered). The Weinreb
supplement also fixes the design: B28 shows sister competition at the split cannot explain
co-presence, and that detection does not differ materially by cell type.

## Correction (2026-11-05, B29): the primary concordances had held-out clones and little power

Prompted by Bence's question, an audit found that the primary concordances (G-M, full-G, full-M)
used all clones, held-out shard included, and that pass/fail at p >= 0.05 rewards wider
predictions. Re-scored on the fit set: the concordances pass for every seed of both models (they
carry little information), E38 is still unstable (B7 G 5/9), and **E46 is stable and passing (13/16
primary)**. By the registered rule E46 is the new reference. A proper score (CRPS) shows its only
clear gain is B7 G; on the concordances E38 scores better. Held-out test still unspent; the power
check comes first. From now on the primary concordances use the fit set and comparisons report
CRPS.

## Update 2026-11-05: E47, selection capped at a plausible strength

Rieger 2009 (GMPs almost death-free in single M-CSF or G-CSF, ~12 h cycle) implies E46's
selection strength is about four times too large. E47 caps it at about one doubling per SD of
bias. Result: stable and passing on the fit set (9/9 gated seeds), summed CRPS 0.340 (E46 0.376,
E38 0.320), so by the registered rule **E47 is the reference**. But the B7 G centre moves only
from -0.27 (E38) to -0.29 (data -0.38; E46 -0.41): E46's one real gain does not survive a
plausible cap. Instruction estimates are back to E38's (G and M shifts ~0.5 each). M presence,
D3 and B26 remain unexplained by all three models.
