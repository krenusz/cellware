# Research program (autoresearch rules for this repo)

How experiments are run and logged here. An agent or a person follows these rules for every run.

## Goal (set by Bence, 2026-11-05)
- **Long term:** model stem-cell fate behaviour in general, and eventually answer embryo
  morphogenesis questions (how cells sharing local rules build a patterned tissue).
- **Current step:** show that the model can fit conditional data, i.e. perturbation experiments
  where the same clones are split across conditions. First data set: Weinreb 2020 LARRY cytokine
  experiment.
- **Short-term question:** do G-CSF and M-CSF instruct or select neutrophil vs monocyte fate, and
  in what proportion (instruction shifts tau, selection s and gamma)?
- **Along the way:** every discovery goes into `DISCOVERIES.md` with its evidence and a
  literature cross-check (consistent / in tension / not yet checked).

### Stopping rule for model iterations (from 2026-11-05)
A misfit gets a new model only if a quick check shows that fixing it could move the short-term
answer (instruction shifts or selection strength) by more than their current spread across
standing models. Otherwise it is recorded in DISCOVERIES.md as a property of the data or a known
limitation, and the project moves on (write-up, held-out check, next data set).

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
