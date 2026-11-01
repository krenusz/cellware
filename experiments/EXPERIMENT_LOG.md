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
(filled in as they happen)

---

## Planned (write hypothesis before running)

- **E7, memory test.** A task where lineage memory should matter (fate set by an early cue that
  is then removed), to see whether memory inheritance can be used at all.
- **E8, ES outer loop.** Evolution strategies instead of PPO, same budget; compare stability.
