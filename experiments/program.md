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
(Applies to the 2023-24 PPO tissue runs only; not to chapters 1-3, which use likelihood fits.)
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
4. **Every lesson lands somewhere** (RETRO_SWEEP_2026-11-13: 19 lessons lived only in retros). Each lesson in
   a retro ends with one of: `-> H<n>` (rule added or amended), `-> practice` (line added to the practice
   checklist below), `-> memory` (tooling pitfall, saved to the assistant's memory), or `-> not adopted`
   with the reason. "Standing practice" without one of these is not allowed.
5. Re-check the private accessions on the FOCUS.json watch list.

## Out of bounds
- Changing the metric or the target after seeing results of the same study.
- Interpreting a knockout comparison where conditions collapse at different rates.
- Copying copyleft code (VertAX, foambryo, deltaMic): reimplement from papers.

## Test oracle (from 2026-11-01)

Held-out statistics are computed only by `experiments/test_oracle.py look <test> --exp <id>`. It refuses without a pre-registration heading, with uncommitted experiment code, or for a (test, shard) already seen, and logs every look in `experiments/test_ledger.tsv`. Parallel tracks use disjoint clone shards (`--shard i --shards K`) or one joint look. Agents working a track get the oracle, never the raw test statistic code paths.


## Hold-out rules (Bence, 2026-11-06: "be really careful with the holdout data")

Enforced by `experiments/check_sealed.py` (registry `experiments/sealed.py`); the oracle runs it
before every look and refuses on any failure.
- **H1 Seal first.** Before the first statistic on any data set, register it in `sealed.py` with
  its hold-out (hash shard or untouched files). A data set without a sealed part supports only
  "exploratory / unconfirmed" results.
- **H2 Oracle only.** Held-out data is read only inside test_oracle test functions and the
  approved loaders in `HELDOUT_READERS`; every loader excludes sealed data by default.
- **H3 Named readers.** A sealed file's name may appear only in its listed reader module.
- **H4 Integrity.** Record the sha256 when sealing a file; a changed file cannot be looked at.
- **H5 Provenance.** Before calling a file untouched, check whether any processed table already
  used (the authors' or ours) was built from it.
- **H6 One look.** Predictions, readings and code are committed before the look; one look per
  test; the result is reported as it comes, with no re-analysis of the sealed data afterwards.
- **H7 Spent is spent.** A looked-at set never serves as confirmation again; reusing it for fitting
  is disclosed in the log.
- **H8 Labels.** "moderate" or "strong" in DISCOVERIES only for oracle-confirmed results; open-data
  results are "unconfirmed".
- **H9 Blind format checks.** Inspecting a new file shows columns, tags and counts only, never
  rates, means or other outcome summaries (G2 lesson).
- **H10 When unsure, ask.** If it is not clear whether data is held out, stop and ask Bence.
- **H11 Several sealed partitions, never one hold-out.** Bence (2026-11-07): "you cannot use up
  the holdout set ... we need several sealed partition". Every new data set is split before any
  statistic by a salted hash of its biological unit (`partitions.py`): train 40% (the only data
  analysis code reads, via `train_only`) + sealed partitions A, B, C (20% each). One look spends
  one partition; it is never redrawn or reused.
- **H12 The train fit decides what earns a look.** Candidate models are compared by cross-validation
  on the train partition only; the winner is recorded in `selection/<exp>.json` (test, >= 2
  candidates with scores, chosen = best), committed before the look. The oracle refuses otherwise.
  A model must also be consistent with every open and spent result first (G7 lesson: a memory
  rule already undermined by G6 was spent on the last Raj files).
- **H13 Keep a reserve.** At most one partition per look. The last unspent partition of a source
  (or the last sealed file of an unpartitioned source, e.g. the Raj reserve) needs Bence's verbatim
  OK (`--last-ok`), logged in the ledger.
- **H14 Power before a look.** (G11 lesson: drug partition A was spent with 3 controls, about 15%
  power.) A sealed look needs a power block in `selection/<exp>.json` (effect size and SD from
  train / open / spent data, one- or two-sided alpha, minimum power 0.8). The oracle first reads
  **counts only** from the partition (units per group, H9), computes power, and refuses the look
  below 0.8 without computing any statistic or spending the partition. `test_oracle.py count`
  shows a partition's tag counts (structure only, never outcomes).
- **H15 Pre-flight before pre-registering.** (Retro 2026-11-07b: wrong unit, mislabelled file,
  unpowered look, sealed data proposed where used data sufficed.) Every pre-registration has a
  `Pre-flight:` block covering (1) unit verified from the source, quoted; (2) readout verified by a
  structure check; (3) used data first: open / train / spent data tried before any sealed look;
  (4) power (H14); (5) exposure: any count or summary already seen that is an outcome for this
  question; (6) literature access checked; (7) readout gate (H21, from 2026-11-13). The oracle refuses a
  look whose section lacks the block.
- **H16 Scope guard.** (Bence 2026-11-07: "now you are drifting! ... biology is really wide topic".)
  Every pre-registration has a `Serves:` line: the goal element and the current chapter question it
  serves, and how its result could change that chapter's answer (stopping rule). Work outside the
  current chapter question needs Bence's OK in chat, quoted in the entry. The oracle refuses a look
  without `Serves:`. Current chapter and order of work: GOAL_CHECK_2026-11-07.md section 5.
  **Enforced by hooks** (Bence 2026-11-07: "create hooks to defense us from drifting like this again!"):
  `experiments/FOCUS.json` holds the current step; `experiments/hooks/scope_guard.py` runs as a Claude
  Code PreToolUse hook (blocks running or creating an experiment script not named in a pre-registration
  whose `Serves:` names the current step), as a UserPromptSubmit hook (injects the focus into every
  Cellware turn), and as the git commit-msg hook (`.githooks/`, `core.hooksPath`): FOCUS.json changes
  need a quoted `Bence:` OK; new pre-registrations need `Pre-flight:` and `Serves: <step>`.
- **H17 Power from the model's own prediction, at chapter opening.** (Retro 2026-11-09; Bence: "1 ok!".
  C6-S1 tabulated S1 power at assumed shares 0.66-0.76; the core model itself predicts 0.51-0.59, power <= 0.20,
  so chapter 6's clone-placement claim had no possible test from day 1.) At a chapter's opening pre-flight, every
  candidate sealed test gets its power from the **current model's predicted effect** (simulate the exact test
  statistic at the sealed n), not from assumed or hoped-for effect sizes. If the chapter's central claim has no
  test reaching 0.8 under its own model, the chapter is reframed, or its work is labelled exploratory in every
  pre-registration and summary until a powered test exists (new data or a sharper prediction). Re-run the power
  block whenever the model changes.
- **H18 Mandatory retro after 4 amendments within one step.** (Retro 2026-11-10; Bence: "ok add h18".
  C6-S7 reached 12 model versions, S7 to S7l, each pre-registered, but together a forking path toward a pass.)
  - Before a 5th amendment (model version) within one step is pre-registered, write a retro
    (`RETRO_<date>.md`, with the literature gap map).
  - The retro answers in writing:
    1. What is a pass worth after N versions? (Record it in the step's credit ledger.)
    2. Should the step close instead of being amended again?
  - Every later amendment of the same step cites that retro. Another retro is due after 4 more.
  - C6-S7 is covered by RETRO_2026-11-10.
- **H19 Chapter selection gate: achievable goals only.** (2026-11-11; Bence: "how about selecting a chapter goal
  with achievable goals???? 3rd fcking chapter we went with your suggestion and ending up as data limitation",
  then "3" = adopt this gate, close chapter 7 with powered looks, choose chapter 8 under it.)
  - Chapters 4, 5 and 7 were chosen for interest, and power was checked after building. All three ended
    data-limited. Chapter 6 had no possible unseen test.
  - **A chapter goal may be proposed only if, before any work starts, all three hold:**
    1. **Data in hand:** the data is on disk or public, and it has passed the structural pre-flight (unit,
       readout, counts).
    2. **Powered primary test:** the primary sealed test has power >= 0.8, from structural counts of a sealed
       partition. The effect size comes from data already used (a spent result or train), never from hope.
       **Amendment 2026-11-11b (RETRO_2026-11-11b):** when the test uses a subset (an experiment, condition or
       stratum), the count must be that subset's own structural count in the sealed partition. That count
       reads no fate labels and uses the oracle count-function code. Dataset-wide clone ratios are not
       allowed for subsets (C8 T2: estimate 0.86, real 0.78).
    3. **Short path:** that test can be run within about 3 steps of the chapter start.
  - The proposal shows the three checks with numbers. If no candidate passes, say so; don't pitch an
    ungated idea.
  - **Exploratory side questions** may follow inside a chapter only after its primary test has been run.
- **H20 Value and literature gate, at chapter opening.** (RETRO_2026-11-11c, RETRO_SWEEP_2026-11-13; Bence
  2026-11-13: "alright go 1 2 3". Five retros found the literature read after building. Chapters 7-9 confirmed
  results the source papers already had: H19 checks feasibility, not value.)
  - A chapter proposal shows, next to the three H19 checks and **before any code**:
    1. **Source methods read:** the methods of the source paper, with the definition of every readout the
       chapter would use, quoted.
    2. **Top 5 read:** the five highest-ranked unread papers of the gap map for the chapter question, DOIs
       added to `lit/read.txt`.
    3. **Minimal published mechanism:** the simplest published model that already explains the data, and
       whether its code is public.
    4. **Value:** whether the source paper or a 2024+ paper already reports the answer, and what our result
       would add. If it adds only a disciplined replication, say so up front; Bence decides.
  - **Hold-out limit:** reading stops at any section, figure or table that reports the outcome of a sealed
    test (H9). The proposal lists what was skipped for that reason.
  - **Enforced by the commit-msg hook:** a FOCUS.json change that opens a new chapter is refused unless
    FOCUS.json holds non-empty `value` and `literature` fields.
- **H21 Readout gate.** (RETRO_2026-11-11b/c; C9-S1 passed the scan with a degenerate outcome; 11-04, B26,
  C4-S2b and chapter 6 leaned on a handful of units.) Pre-flight item 7. Before a statistic is pre-registered
  as a test or used as a gate:
  1. the outcome distribution is shown at the scan, from train / open / spent data only;
  2. a binary split has >= 30 units on each side, by structural count of the partition to be looked at;
  3. the statistic varies with the model: simulate it under at least two candidate models and show that it
     differs. A statistic that does not is left out of any summed score;
  4. the number of units that carry the statistic is stated. A statistic carried by a handful of units is
     reported as exploratory and cannot be a gate.
- **Power: one rule at three gates** (H14, H17 and H19.2 read together; numbers kept because code and log
  cite them).

  | gate | when | power comes from | rule |
  |---|---|---|---|
  | scan | chapter proposal | structural counts of a sealed partition; effect from used data | H19.2 |
  | selection | chapter opening, and after every model change | the current model's simulated test statistic at the sealed n | H17 |
  | oracle | before the look | counts only, read by the oracle; refuses below 0.8 | H14 |

  At every gate:
  - effect size and sd come from named used results (train, open or spent), never from assumed values;
  - the power block lists which spent results constrain each input, and uses all of them (C6-S8a: one
    memory rate was used where a spent result gave two);
  - the sd is estimated at the test's own unit (C5-S4);
  - a subset uses its own structural count (H19.2 amendment);
  - the outcome passes the readout gate (H21) first.

## Practice checklist (from 2026-11-13; RETRO_SWEEP_2026-11-13)

Lessons that retros called standing practice. They are checks, not H-rules: each is one line in a
pre-registration. Every pre-registration from 2026-11-13 has a `Practice:` line that names the list it
ticked and any item that does not apply, with the reason. The oracle and the commit-msg hook refuse a
pre-registration dated 2026-11-13 or later without it.

**Before a model run**
- Smoke tests use synthetic outcomes and write to the scratchpad, never to real results (11-04b).
- Cold start after any structural model change (11-04).
- Check a known model's analytic regime before choosing parameters (11-09).
- Simulated readouts copy the measurement physics (11-09).
- Probe whether the fit can stay at the favourable point before registering (11-05b).
- Step 0 hygiene: at least 10 repeats, a settling check, a float tolerance (11-10).

**Before a statistic**
- Adequacy gate before any cross reading (11-03).
- On a new featurisation, pass a fate-free validity check before any fate statistic (11-11).
- Tick the pre-registration against the script, line by line, before the run (11-11).

**Before calling a candidate best**
- Rank the candidates against stated criteria first (11-08).

## Result reports: self-challenge block (from 2026-11-13)

In 9 of 16 retros the key failure was found by Bence's question, not by the analysis. Every result report,
in chat and in the outcome entry of EXPERIMENT_LOG.md (`Challenge:` line), answers these before the
conclusion, one line each:
1. Is anything suspicious (too clean, too large, too convenient)?
2. What else could explain it?
3. How many units carry it?
4. Could data already used have answered it?
5. Is it already in the literature?

"Nothing" is an allowed answer only with the check that was done.
