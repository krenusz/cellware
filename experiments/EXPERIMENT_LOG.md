# Experiment log

One entry per study, in order. Each entry: hypothesis and prediction, change, setup, outcome,
decision. Per-run numbers: [results.tsv](results.tsv) (95 runs, rebuilt by `build_log.py`).
Rules: [program.md](program.md). All studies E1-E5 ran on 2026-10-31.

Note on pre-registration: E1-E5 were run before this log existed. Their hypotheses below are
reconstructed from the plan stated before each run, not written in advance. From E6 on, entries
are written before running.

## Summary

| ID | Commit | Question | Runs | Collapsed | Outcome | Decision |
|---|---|---|---|---|---|---|
| E1 | e85d3b6 | Collective vs individual memory, depleting dish | 10 | - | Ceiling: both reach 99% of the map | Make task harder |
| E2 | 8380ed5 | Same, with regrowth + cooperative cells | 10 | - | Equal harvest; collective crowds and shares, individual splits territory | Keep; move to tissue |
| E3 | 4fd1901 | Tissue knockouts, cells see global size | 25 | 0 | Target learned; no knockout matters | Discard: confound |
| E4 | 7263c29 | Local signals only + spatial target | 25 | 9 | PPO collapses to one founder cell | Discard: unstable |
| E5 | c1acd59 | Same + totipotency window + entropy anneal | 25 | 0 | Stable; morphogen > gap junctions > contact, memory | Keep as baseline |

---

## E1: collective vs individual memory (dish)

- **Hypothesis.** Agents that learn from pooled trajectories (collective memory) harvest more
  than agents that learn only from their own, as the 2023 Tufts interim result suggested.
- **Prediction.** Higher `dish_total_return` for collective.
- **Change.** First run of the PyTorch port (`cellware.envs.PetriDish`, exact port of the 2023 env).
- **Setup.** 4 agents, 16 dishes, horizon 128, 300 iterations, PPO (lr 3e-4, clip 0.2,
  entropy 0.01), 5 seeds x 2 conditions.
- **Outcome.** Both reach about 780 of 790 points within 50 iterations: the task saturates.
  Individual is 4.7 points higher (significant, but 0.6%). Collective reaches the ceiling faster
  (~30 vs ~45 iterations) and its agents converge on one policy (JS divergence 0.0009 vs 0.18;
  partly built in, as they train on identical data). Reward split unequal in both (CV ~1.05).
- **Decision.** Prediction not confirmed; uninformative because of the ceiling. Harder env (E2).

## E2: harder dish

- **Hypothesis.** With scarce, regrowing resources and rewards that need two agents, collective
  memory improves coordination and so total harvest.
- **Prediction.** Higher harvest and more cooperative payouts for collective.
- **Change.** `--regrow_rate 0.02 --coop_threshold 50 --see_others`.
- **Setup.** As E1, 5 seeds x 2 conditions.
- **Outcome.** No ceiling, but equal harvest (~1,820 per dish; CI includes 0). Two social
  organisations: individual agents split the dish (overlap 0.46, unequal reward, CV 0.54);
  collective agents crowd together (overlap 0.73, fairer split, CV 0.40). Both learn cooperation
  (~6 joint payouts per dish per episode), collective earlier (~70 vs ~150 iterations).
- **Decision.** Harvest prediction not confirmed; the organisation difference is the finding.
  Keep. Move to the stem-cell tissue model.

## E3: tissue knockouts, global readout

- **Hypothesis.** Growing a tissue to target size and fate proportions requires cell-cell
  signalling and lineage memory.
- **Prediction.** Each knockout (contact, morphogen, gap junctions, memory) lowers `score`.
- **Change.** New `cellware.tissue` model (pitchfork fate landscape, irreversible commitment,
  heritable memory, three signalling channels) and shared-policy PPO trainer.
- **Setup.** 12x12 lattice, target 60 cells at 10/60/30% stem/A/B, horizon 40, 300 iterations,
  5 conditions x 5 seeds.
- **Outcome.** Target learned (65 cells, 6/62/33%, score -0.32 from -1.40). No knockout changes
  the score; sister concordance 0.66 with and without memory.
- **Diagnosis.** Cells observed global tissue size and episode time. That broadcast makes local
  signalling redundant; real cells have no such readout.
- **Decision.** Discard as a test of the hypothesis. Remove global inputs (`--global_readout`,
  default off) and add a spatial target that needs signalling.

## E4: local signals only, spatial target

- **Hypothesis.** As E3, now testable: without global inputs, size, proportions and an A-core /
  B-rim pattern must be coordinated locally.
- **Prediction.** Knockouts lower `score`; the morphogen knockout most (only long-range channel).
- **Change.** `global_readout=False`; `--spatial` adds pattern error (A within radius 3.4 of the
  founder site, B outside).
- **Setup.** As E3, 400 iterations.
- **Outcome.** Training collapses: tissues grow to 20-55 cells, then the policy stops dividing
  and the founder commits to A (score about -1.78). By the health check (min size over the last
  100 iterations <= 1.5), 9/25 runs collapsed: full 4/5, nomorphogen 5/5, others 0/5.
  (The README's "2 of 5 full, 4 of 5 nomorphogen" counted only the final size; the stricter
  check finds more.) Conditions that collapsed less scored better, which says nothing about
  the channels.
- **Decision.** Discard. Fix training stability before any knockout claim.

## E5: stabilised training

- **Hypothesis.** The collapse comes from the founder committing early (a local optimum
  reachable in the first steps) and the policy losing entropy before growth pays off.
- **Prediction.** A totipotency window plus an annealed entropy bonus removes collapse in all
  conditions; then the morphogen knockout lowers `score` most.
- **Change.** `--min_commit_size 16` (no commitment until 16 cells; x held just below the
  threshold), entropy coefficient annealed linearly 0.05 -> 0.005. Test
  `test_totipotency_window_blocks_early_commitment` added (30 tests pass).
- **Setup.** As E4. Health check on the full model (5 seeds) first, then 20 knockout runs,
  one at a time.
- **Outcome.** 0/25 collapsed (min size over the last 100 iterations 28-41).
  Full model: 53.2 cells, stem 0.19 / A 0.66 / B 0.15, score -1.05 (sd 0.02), pattern 0.63,
  sister concordance 0.88. Differences from full (* = CI excludes 0):

  | Knockout | Score | Stem share | B share | Pattern | Sister conc. |
  |---|---|---|---|---|---|
  | no morphogen | -0.17* | +0.23* | -0.12* | +0.11* | +0.06* |
  | no gap junctions | -0.11* | +0.10* | -0.06* | +0.02 | +0.02 |
  | no contact | -0.02 | 0.00 | 0.00 | -0.04* | -0.02 |
  | no memory | -0.01 | -0.03* | +0.04 | -0.03* | -0.02 |

  The morphogen (secreted by A cells) acts as negative feedback on stem renewal; without it the
  stem share doubles. Its higher pattern accuracy is an artefact: fewer cells commit, so fewer
  are scored. Memory inheritance still does not drive sister concordance.
- **Decision.** Prediction confirmed. Keep E5 as the baseline config.
- **Open issues.** Stem share twice the target, B share half; pattern only partly formed (0.63).
  Report pattern accuracy alongside the committed fraction. 5 seeds: effect sizes preliminary.

---

## E6: fit to LARRY clonal data (pre-registered 2026-11-01, before any statistic was computed)

- **Data.** Weinreb et al. 2020 (Science 367:eaaw3381), in vitro haematopoiesis: 130,887 cells,
  5,864 clones, days 2/4/6, LSK and LK starting populations. After day 2, each clone's cells were
  split into two sets of wells (`Well` 1 and 2). Only the clone matrix and metadata are used
  (`data/larry/`, not committed). So far only the column structure and category counts were read.
- **Reduction.** Mature cells (days 4 and 6) are grouped as N (neutrophil), M (monocyte) and O
  (all other mature types); undifferentiated cells are ignored. Unit of analysis is a clone.
- **Fit set (well 1 only).** For clones with >= 2 mature cells in well 1: fraction of clones that
  are N-only, M-only, mixed N+M, and containing O; distribution of the N fraction N/(N+M).
- **Held-out test (never used in fitting).** Clones with >= 2 N/M cells in each well. Primary
  test statistic: Pearson r of the N fraction between well 1 and well 2 across clones (fate bias
  shared by cells that were separated at day 2). Secondary: fraction of clones whose majority
  fate (N or M) agrees across wells. 95% CIs by bootstrap over clones.
- **Model.** Non-spatial version of `cellware.tissue`: founder cells divide (cycle ~ 0.5 day) for
  6 days; each cell carries a fate state x with pitchfork drift `dx = (x - x^3 + w*m)dt + noise`
  and commits at |x| > 0.7 (x > 0 -> N, x < 0 -> M; a fixed fraction goes to O); a memory value m
  is drawn per founder and copied to daughters with noise. At day 2 descendants are split at random
  into two wells; a fraction of cells is sampled at days 4 and 6. The RL policy is replaced by
  these fitted parameters: this tests the landscape + memory mechanism, not the learned policy.
- **Fitting.** Simulation-based: minimise the distance between simulated and observed fit-set
  statistics (grid search, then Nelder-Mead), 5 simulation seeds per point.
- **Hypothesis.** Inherited memory (founder bias copied with small noise) explains why sister
  cells in separate wells keep the same fate bias.
- **Predictions.** (1) The full model, fitted on well 1 alone, predicts the cross-well r within
  the observed 95% CI. (2) Null A (no memory, w = 0) predicts r near 0. (3) Null B (memory but
  resampled at every division) also predicts r near 0. Weinreb reported that fate bias is
  partly but not fully heritable, so the observed r is expected to be clearly above 0 and below 1.
- **Budget on the test set.** At most 4 looks: one per model version, each recorded below with its
  result, failed ones included. Model changes between looks may use only the fit set.

### E6 runs

**Fit set (well 1), from `larry_stats.py`:** 1,398 clones with >= 2 mature cells; N-only 0.28,
M-only 0.26, mixed N+M 0.12, containing O 0.35, O share of cells 0.34, mean 7.2 sampled mature
cells per clone. The N fraction is strongly bimodal (423 clones near 0, 453 near 1, 119 between).

**v1 fits** (`fit_larry.py --tag v1`, 150 random + 150 local evaluations, ~3 min each):

| Model | Fit loss | Mixed N+M (obs 0.12) | Predicted cross-well r | Predicted agreement |
|---|---|---|---|---|
| inherit | 0.012 | 0.07 | 0.89 (0.87-0.91) | 0.93 |
| resample | 0.006 | 0.11 | 0.79 (0.79-0.81) | 0.86 |
| none | 0.031 | 0.00 | 1.00 | 1.00 |

Before looking at the test set: predictions 2 and 3 are already wrong *in the model*. Neither
null predicts r near 0. Clonal fate concordance does not need inherited memory: once a cell
commits, its daughters keep the fate, so a clone that commits before the day-2 split agrees across
wells anyway. "No memory" fits by committing every founder almost immediately (k = 3.9), which
gives no mixed clones at all (0.00 vs 0.12 observed). The test can still tell the models apart,
because they predict different r (0.79 / 0.89 / 1.00).

**Look 1 at the test set (1 of 4):** 291 clones with >= 2 N/M cells in both wells.
Observed **r = 0.74 (95% CI 0.66-0.80)**, majority-fate agreement **0.82 (0.78-0.86)**.
- `resample` (0.79, agreement 0.86): inside the r CI, at the upper edge for agreement.
- `inherit` (0.89, 0.93): outside, overpredicts heritability.
- `none` (1.00): outside.
- Prediction 1 (inherited memory predicts r within CI) **not confirmed** for v1. Sisters separated
  at day 2 share fate bias less than a faithfully copied memory implies, which matches Weinreb's
  "missing heritability": part of the fate choice happens after the split.
- Caveat: v1 searches were short, and the inherit fit is worse (0.012 vs 0.006), so part of the
  gap could be the optimiser.

**v2 (fit set only):** add `--memory decay`, where each division keeps a share a of the parent
memory and refills the rest (a = 1 is inherit, a = 0 is resample), and refit all four models with
a longer search (250 random + 300 local). If a lands between 0 and 1 and the fit improves, the
data favour partly heritable memory.

**v2 outcome (no test look spent):** the search failed for three of four models. `decay`,
`inherit` and `resample` landed in a worse optimum (loss ~0.03 vs 0.006-0.012 in v1): fast
commitment, no mixed clones, r ~0.99. Only `none` found a good fit, and a better one than v1
(loss 0.0085, mixed 0.11, predicted r 0.81, agreement 0.87). The random + local search is
unreliable, so model comparisons at this stage reflect the optimiser, not the models.

**v3 (fit set only):** Nelder-Mead (scipy) started from every earlier best (v1 and v2, all models)
plus 2 random starts, 200 evaluations each. The loss is deterministic (fixed simulation seeds), so
Nelder-Mead is appropriate. Only after this will look 2 be spent.

**v3 outcome:**

| Model | Fit loss | Mixed N+M (obs 0.12) | Predicted r (range over 5 sims) | Agreement |
|---|---|---|---|---|
| resample | 0.0042 | 0.11 | 0.80 (0.78-0.82) | 0.87 |
| decay (a = 0.50) | 0.0043 | 0.08 | 0.81 (0.80-0.82) | 0.87 |
| inherit | 0.0068 | 0.08 | 0.87 (0.85-0.90) | 0.91 |
| none | 0.0083 | 0.11 | 0.81 (0.79-0.83) | 0.87 |

**Look 2 (2 of 4), against observed r 0.74 (0.66-0.80), agreement 0.82 (0.78-0.86):** all four
models overpredict. `resample` is at the edge of the r interval and outside for agreement;
`inherit` is clearly outside. Differences between `resample`, `decay` and `none` are small (fit
loss 0.004-0.008, predicted r 0.80-0.81): with this data, inherited memory adds little over
commitment timing plus noise. The faithful-copy model (`inherit`) is the worst on both fit and
prediction. Note: Nelder-Mead left `a` at its start value (0.50) in `decay`, so its memory
share is not well identified.

**v4 (fit set only):** a systematic overprediction suggests something lowers observed
cross-well agreement that the model lacks. Candidate chosen from the fit set: N/M annotation error
(each sampled N or M cell is given the other label with probability eps), which also shows up
within well 1 as stray minority cells in otherwise single-fate clones. Refit all four models with
`--label_noise`, Nelder-Mead from the v3 bests + 2 random starts. Look 3 follows.

---

## Planned (write hypothesis before running)

- **E7, memory test.** A task where lineage memory should matter (fate set by an early cue that
  is then removed), to see whether memory inheritance can be used at all.
- **E8, ES outer loop.** Evolution strategies instead of PPO, same budget; compare stability.
