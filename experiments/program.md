# Research program (autoresearch rules for this repo)

How experiments are run and logged here. An agent or a person follows these rules for every run.

## Question
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

## Out of bounds
- Changing the metric or the target after seeing results of the same study.
- Interpreting a knockout comparison where conditions collapse at different rates.
- Copying copyleft code (VertAX, foambryo, deltaMic): reimplement from papers.
