# Research program (autoresearch rules for this repo)

How experiments are run and logged here. An agent or a person follows these rules for every run.

## Goal (set by Bence, 2026-11-05)
- **Long term:** model stem-cell fate behaviour in general, and eventually answer embryo
  morphogenesis questions (how cells sharing local rules build a patterned tissue).
- **Current step:** show that the model can fit conditional data, i.e. perturbation experiments
  where the same clones are split across conditions. First data set: Weinreb 2020 LARRY cytokine
  experiment (closed 2026-11-05, WRITEUP_LARRY_CYTOKINE.md). Second: SIS-seq / SIS-skew
  (chapter 2 below; closed 2026-11-06, WRITEUP_SIS.md).
- **Short-term question (chapter 1, answered):** do G-CSF and M-CSF instruct or select neutrophil
  vs monocyte fate, and in what proportion (instruction shifts tau, selection s and gamma)?
  Answer: mostly clone-level selection on a strong fate state, modest instruction.
- **Along the way:** every discovery goes into `DISCOVERIES.md` with its evidence and a
  literature cross-check (consistent / in tension / not yet checked).

### Stopping rule for model iterations (from 2026-11-05)
A misfit gets a new model only if a quick check shows that fixing it could move the short-term
answer (instruction shifts or selection strength) by more than their current spread across
standing models. Otherwise it is recorded in DISCOVERIES.md as a property of the data or a known
limitation, and the project moves on (write-up, held-out check, next data set).

## Chapter 2 plan: SIS-seq and SIS-skew (agreed 2026-11-05; closed 2026-11-06)
**Outcome (WRITEUP_SIS.md):** clone fate state replicates strongly; Bcor acts on growth and
persistence, no identifiable fate term; RNA carries only a weak trace of the state (held-out).

Data: Tian et al. 2021 Immunity (Naik lab), all public files local in `data/sis_seq/`
(see NEXT_DATASETS_2026-11.md, checks 1-2c). Downloads go stepwise, as each step needs them.

**Why this data set.** The LARRY answer rests on a latent clone fate state (beta 7-8 logits per
SD) that drives selection. SIS tests it three ways LARRY could not: the clone state is measured
(RNA of sisters before the split, plus two independent sister fate wells), the condition splits
sisters of one clone (Bcor KO vs WT, with null wells where both arms are WT), and the lineage is
new (cDC1 / cDC2 / pDC).

**Chapter question.** Does the same model family generalise to a second system and give a
trustworthy instruction-vs-selection split? Success means fitting the null wells, the KO shift
and the presence statistics on held-out clones.

**Levels must be kept apart.** In SIS-skew a within-clone shift in fate fractions can be
instruction or fate-specific proliferation of precursors inside the clone (cell-level
selection); endpoint counts cannot tell these apart. What the data can separate: clone-level
selection (does KO expansion depend on the clone's existing bias?) vs within-clone change (does
KO change the fate split relative to the sisters?). Every reading and the write-up name the level.
LARRY's answer is also clone-level.

**Steps (each gated by the previous one).**
1. Seal held-out clones in both data sets before any fate statistic: SIS-skew by clone within
   well (barcodes collide across wells, so a clone is defined only within a well); SIS-seq by
   clone, stratified by batch.
2. **B38, SIS-seq:** is fate fixed before the split? Sister-sister agreement vs random pairs and
   vs the per-well background, weighted by cell count. Gives a heritable-fate estimate to compare
   with LARRY's beta: the first cross-data-set test of the main LARRY discovery. Minimum 105
   clones with both sister wells.
3. **B39, SIS-skew:** what Bcor does. Null wells: sister disagreement when nothing differs.
   KO wells: fate shift beyond the null; KO/WT size ratio against the clone's WT fate; one-arm
   clones by size (LARRY's private-clone question). Minimum 200 both-arm clones per arm type.
4. **Port the model** to 3 fates and two sister arms per clone: instruction as a KO-arm logit
   shift, selection as KO-arm growth tied to latent state. **First, a synthetic recovery check:**
   can known instruction and selection be recovered at this sample size? (Not done for LARRY;
   part of why E46 vs E48 stayed open.)
5. Fit, then one held-out look. The stopping rule applies.
6. **Link latent state to biology:** does the inferred clone state match SIS-seq RNA (a
   pre-chosen DC-priming gene score, or 2-3 principal components)? This ties the model to a
   measured fate code, the result that matters most for the long-term goal.
7. Literature cross-check (Naik 2013 Nature, DC fate imprinted in single progenitors; Lin/Naik
   2018; Tian 2021); discoveries into DISCOVERIES.md.
8. Then chapter 3: monoclonal gastruloids (GSE291244), on a model tested in two systems and tied
   to measured state.

**Known risks.** SIS-seq is small (2 of 4 batches under 12 cells per well); barcode collisions
across SIS-skew wells; the 25% transduction adjustment in the authors' cell conversion; batch
effects. Each pre-registration states its minimum sample.

## Plan toward space (agreed direction 2026-11-06; RETRO_2026-11-06)
The clone-level core model (`core_model.py`: clone state, instruction shift, selection on state,
split, growth, sampling) becomes the **cell-level prior of the tissue simulator**:
- each cell agent carries its clone state z (inherited with the fate memory rho the core fits);
- the cell's fate logits are the core's (B z + tau), with tau now a function of the local signal
  the agent sees (instruction becomes local);
- growth and survival follow the core's selection term, also allowed to depend on position;
- the 2023-24 learned-policy simulator supplies the spatial layer (neighbour signals, movement).
First spatial test (chapter 3): Raj-lab gastruloids (NEXT_DATASETS, spatial rank 1). Pure clones
carry anterior or posterior biases; mixtures restore the axis. The model must reproduce clone
spatial propensities and the pure-vs-mixed elongation difference from clone states plus local
rules. Data download stepwise, when the chapter starts.
**Chapter 3 status (2026-11-06):** G1 (#20 clone A-P propensity is a stable state; #21 the
gastruloid sets domain size), G2 (#22 mixing any two clones rescues elongation), G3 (#23
propensity fades over passages). **Priority question (Bence): division of labour.** Sweep in
LITERATURE_SWEEP_2026-11e.md: rescue is generic in our data; three explanations to test before
the model (H1 buffering of clone extremes, H2 complementation of missing capacities, H3
proportion feedback on the starting T+ fraction). Model architecture from the literature:
sorting on clone state (positions) + a communication feedback (proportions).

**Chapter 3 model (G5, 2026-11-06):** `g5_gastruloid.py`, cell-based 1D simulator: clone
state + rank sorting (positions) + proportion feedback g 0.83, a* 0.38 (domain sizes) +
memory lost per passage event, retention 0.60 (#25) + descriptive elongation rescue rule.
Reproduces #20-#23 (calibrated); misses 3+7 and 4+6 elongation. Next for chapter 3: write-up.

**Candidate chapter 4 (Bence, 2026-11-06): the mechanism of mixing rescue / division of labour
(#22, #24).** Mixing any two gastruloid clones rescues elongation, with a modest bonus for
opposite A-P propensities; buffering, strict complementation and a T+ window do not explain it
from elongation counts (G4). Search for data that can: per-clone cell-type composition of pure
and mixed gastruloids (Shendure monoclonal gastruloids GSE291244 / Cell 2026 "A fluctuation test
for gastruloid heterogeneity"; needs the cell-to-gastruloid assignment from tape files),
time-resolved T+ and pluripotency fractions (Oriola 2024 style), per-clone signalling
(Wnt/Nodal/RA) and adhesion. Literature start: LITERATURE_SWEEP_2026-11e.md. Not started; chapter
3 first finishes with the gastruloid model (G5) using a descriptive rescue rule.

## Original question (2023-24 programme)
Can cells that share one learned policy, and see only local signals, grow a tissue with target
size, fate proportions and spatial pattern, and which signalling channels does that rely on?
Later: does the model reproduce measured clonal statistics (Weinreb 2020 LARRY)?

## Fixed setup
- Primary metric, tissue studies: `score` (0 = perfect; size error + proportion error + pattern
  error when `--spatial`). Dish studies: `dish_total_return`. Reported as the mean of the last
  30 iterations per run, then mean and bootstrap 95% CI across seeds.
- Fixed budget per run: 300-400 PPO iterations, 16 tissues/dishes in parallel, CPU, 2 threads.
- Hardware rule: one run at a time on this PC (it hard-resets under parallel load).
- 5 seeds (0-4) per condition. A difference counts only if its 95% CI excludes 0.

## Loop
1. Write the hypothesis and the prediction in EXPERIMENT_LOG.md **before** running.
2. Change one thing (a flag or one mechanism). Commit the code first, so every run maps to a commit.
3. Run all conditions x seeds sequentially; write per-run JSON to a new `results_<study>/`.
4. Register the study in `build_log.py` and run it: every run goes into `results.tsv`
   (status `ok` / `collapsed`). No run is deleted, failed ones included.
5. Analyse (`analyze_tissue.py` / `analyze.py`), then fill in outcome and decision
   (keep / discard / fix) in EXPERIMENT_LOG.md. Report failures as plainly as successes.
6. A health check (no collapse: min size over the last 100 iterations > 1.5) must pass
   before knockouts are interpreted.

## Retro (standing steps, from 2026-11-04)
1. Write `RETRO_<date>.md`: what went well, what went badly, decisions.
2. **Literature gap map:** run `uv run python experiments/lit_gap.py`. It maps the citation
   neighbourhood of the core papers (SEEDS in the script) through the free OpenAlex API, ranks
   unread papers by how many open questions they touch, and marks each open access or paywalled
   (by publisher). Read the top open-access ones, add DOIs read to `experiments/lit/read.txt`,
   and record in the retro which paywalled papers matter and behind which publisher. Add new
   core papers to SEEDS when the project adopts them.
3. Sweeps run with `--guard N`; status updates show `./sweep_status.sh`.

## Out of bounds
- Changing the metric or the target after seeing results of the same study.
- Interpreting a knockout comparison where conditions collapse at different rates.
- Copying copyleft code (VertAX, foambryo, deltaMic): reimplement from papers.

## Test oracle (from 2026-11-01)

Held-out statistics are computed only by `experiments/test_oracle.py look <test> --exp <id>`. It refuses without a pre-registration heading, with uncommitted experiment code, or for a (test, shard) already seen, and logs every look in `experiments/test_ledger.tsv`. Parallel tracks use disjoint clone shards (`--shard i --shards K`) or one joint look. Agents working a track get the oracle, never the raw test statistic code paths.
