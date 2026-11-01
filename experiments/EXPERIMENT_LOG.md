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

**v4 outcome and look 3 (3 of 4):** label noise improves the fit (loss 0.0022 for `resample`,
eps ~0.02) but moves the prediction the wrong way: r 0.86 (`resample`), 0.87 (`decay`), 0.88
(`inherit`), 0.98 (`none`), all outside the observed 0.66-0.80. A better within-well fit gave a
worse held-out prediction. Lesson: the v1-v4 fit statistics (clone composition) cannot separate
fate bias that is inherited across the split from bias set later within each sub-lineage, so
fitting them harder does not constrain heritability.

**v5 (fit set only):** add a statistic that carries heritability information without touching
well 2: the correlation of the N fraction between day-4 and day-6 cells of the same clone in well 1
(`r_d4_d6`). Observed 0.91; the v3 models give 0.97-0.99, so the fit set alone already says the
models are too heritable. Refit all four models with `--use_time` (no label noise), Nelder-Mead
from the v3 bests. Look 4, the last, follows.

**v5 outcome:**

| Model | Fit loss | Mixed N+M (obs 0.12) | r_d4_d6 (obs 0.91) | Predicted r | Agreement |
|---|---|---|---|---|---|
| resample | 0.0075 | 0.13 | 0.98 | 0.76 (0.75-0.77) | 0.84 |
| decay | 0.0084 | 0.08 | 0.98 | 0.81 (0.79-0.82) | 0.87 |
| none | 0.0111 | 0.10 | 0.97 | 0.81 (0.80-0.83) | 0.87 |
| inherit | 0.0106 | 0.08 | 0.99 | 0.86 (0.85-0.89) | 0.91 |

**Look 4 (4 of 4, budget used up):** against observed r 0.74 (0.66-0.80) and agreement 0.82
(0.78-0.86), `resample` is inside both intervals; `decay` and `none` are just outside on r;
`inherit` is clearly outside, as in every version.

### E6 conclusion

- **Confirmed across all five versions:** faithfully inherited memory overpredicts how strongly
  sister cells separated at day 2 share a fate (predicted 0.86-0.89 vs observed 0.74). Lineage
  memory that is copied intact at every division is ruled out for this system.
- **Best model:** `resample` (fate bias carried only by the founder's starting state and the
  timing of commitment; memory redrawn at each division) predicts the held-out cross-well
  statistics within their 95% CIs (r 0.76, agreement 0.84). No-memory and half-memory models are
  close behind, so the data do not demand a memory channel beyond commitment timing.
- **Not cracked:** no model reproduces the within-well day-4 vs day-6 correlation (0.97-0.99 vs
  0.91). In the model a committed cell's fate passes unchanged to all its descendants, so a clone
  cannot change composition over time; in the data it does, probably because undifferentiated
  cells keep committing between day 4 and day 6 under changing conditions. Since v5 also did not
  match this statistic, its hit on the test set is partly a different optimum (slower commitment,
  more mixed clones), not a mechanism the fit set confirmed. Treat it as supportive, not decisive.
- **Original predictions:** (1) inherited memory predicts r within the CI: **not confirmed**;
  (2) and (3) nulls predict r near 0: **wrong**, because commitment timing alone gives clonal
  concordance.
- **Next test (new pre-registration needed, fresh data):** add time-varying commitment (cells
  can stay undifferentiated and commit late; division slows after commitment), fit on the in vitro
  data, and predict the Weinreb in vivo or cytokine-perturbation experiments, which were never
  looked at.

## E6b: late commitment, predicted on in vivo data (pre-registered 2026-11-01)

- **Data.** Weinreb 2020 in vivo experiment (`data/larry_invivo/`, not committed): barcoded
  progenitors profiled at day 2, transplanted into mice, profiled at days 9 and 16. 23,241 clones;
  852 clones have cells in more than one recipient mouse. Only structure (clone x mouse x day
  counts) has been read; no fate statistic has been computed.
- **Model change (v6, motivated by the fit set only).** In E6, committed cells kept dividing, so
  early-committed cells dominated later samples and no model could lower the day-4 vs day-6
  correlation (0.97-0.99 vs 0.91). v6 adds `q`, the probability that a committed cell still
  divides; with q < 1 later samples are dominated by cells that committed later.
- **Fit.** In vitro only: the well-1 statistics plus `r_d4_d6` (as v5). The in vitro well 2 is
  no longer a valid test (its budget is used up) and is reported for information only.
- **Test (in vivo, never used for fitting).** Clones with >= 2 N+M cells (neutrophil, monocyte)
  in each of two different mice (days 9 and 16 pooled; for clones in 3+ mice, the two with the most
  N+M cells). Primary statistic: Pearson r of N/(N+M) across clones between mice; secondary:
  majority-fate agreement. Bootstrap 95% CI.
- **Predictions.** (1) v6 matches the in vitro `r_d4_d6` better than v5 (fit set). (2) The best
  v6 model's cross-split r, simulated with the in vitro parameters (split at day 2), lies inside the
  in vivo 95% CI. (3) Faithful inheritance (`inherit`) again predicts higher r than observed.
  Main caveat, stated in advance: the in vivo timescale (9-16 days, in mice) differs from the model's
  6-day culture, so a miss on (2) may reflect the environment rather than the mechanism.
- **Budget.** One look at the in vivo test per model, all four memory variants in one look.

### E6b result

v6 fits (first attempt crashed when a sparse simulation left < 2 comparable clones; fixed in
`larry_stats.test_stats` and rerun, same settings):

| Model | Fit loss | q | Mixed (obs 0.12) | r_d4_d6 (obs 0.91) | In vitro cross-well r (obs 0.74, info only) | Predicted cross-split r |
|---|---|---|---|---|---|---|
| resample | 0.0088 | 0.67 | 0.13 | 0.97 | 0.70 | 0.70 (0.68-0.73) |
| decay | 0.0084 | 0.79 | 0.13 | 0.97 | 0.75 | 0.75 |
| none | 0.0106 | 0.78 | 0.11 | 0.96 | 0.75 | 0.75 |
| inherit | 0.0074 | 0.73 | 0.12 | 0.96 | 0.76 | 0.76 |

**The single in vivo look:** only 66 clones meet the criterion (>= 2 N+M cells in each of two
mice). Observed cross-mouse **r = 0.46 (95% CI 0.22-0.68)**, majority agreement **0.61
(0.48-0.71)**.

- Prediction 1 (v6 matches r_d4_d6 better than v5): **marginally**: 0.96-0.97 vs 0.97-0.99,
  still far from 0.91. Committed cells fitted to divide less (q ~0.7-0.8), which helps a little.
- Prediction 2 (best v6 model within the in vivo CI): **not confirmed.** All models predict
  0.70-0.76, above the upper bound of 0.68 (resample is closest).
- Prediction 3 (`inherit` higher than observed): true, but now every model is, so it says nothing.
- Side effect: with q fitted, all four models reproduce the in vitro cross-well r (0.70-0.76 vs
  0.74). Allowing committed cells to stop dividing made memory variants indistinguishable on that
  statistic, so the E6 ranking between them was fragile.
- **Interpretation.** Sister cells transplanted into different mice share fate much less (0.46)
  than sisters in different culture wells (0.74). A model calibrated on culture cannot know that;
  the stated caveat (different environment and timescale) is the most likely reason. The data are
  thin (66 clones, wide CI).

### E6 overall status

The clearest results are negative and robust: faithfully copied memory never fits better than
alternatives, and fate concordance between sisters falls from culture (0.74) to separate animals
(0.46), pointing to the environment rather than inheritance as the missing factor. What the model
still lacks: within-clone change over time (0.91 not reached) and any environment term. Next model
step, if pursued: a per-split environment bias (well or mouse) fitted on in vitro wells and the
cytokine-perturbation experiment, tested on in vivo.

## E9: environment vs inheritance, cytokine perturbation (pre-registered 2026-11-01)

- **Motivation.** E6b: sister cells agree on fate much less in separate mice (0.46) than in
  separate culture wells (0.74). The model has no environment term. Weinreb's third experiment
  splits each clone's cells after day 2 into different cytokine conditions, which tests directly
  how far the environment overrides inherited fate bias.
- **Data.** `data/larry_cytokine/` (not committed): 65,075 cells, 5,257 clones, days 2/4/6;
  conditions: full cocktail, M-CSF, G-CSF, Epo, SCF. Only column structure and category counts
  read so far; no fate statistic computed.
- **Model (v7).** E6 lineage model (memory variant `resample`, the E6 best, plus `inherit` for
  comparison; q fitted as in v6) with one extra drift term per condition after day 2:
  `dx = k (x - x^3 + m + b_c) dt + noise`, `b_full = 0` fixed, `b_G` and `b_M` fitted. Cells
  present at day 2 are split at random across conditions.
- **Fit set.** Within-condition statistics only, for full cocktail, G-CSF and M-CSF separately:
  clone composition (N-only, M-only, mixed, has O, O share, N-fraction histogram, size) and the
  within-condition day-4 vs day-6 correlation.
- **Held-out test (never fitted).** Cross-condition sister concordance: Pearson r of the N
  fraction between the same clone's cells in two conditions, for (full, G-CSF), (full, M-CSF) and
  (G-CSF, M-CSF); clones with >= 2 N+M cells in both; bootstrap 95% CIs. Epo and SCF are not used.
- **Predictions.** (1) Fitted biases have the expected signs: b_G > 0 (towards N), b_M < 0
  (towards M). (2) Observed cross-condition r is lower than the in vitro cross-well r (0.74): a
  changed environment overrides part of the inherited bias. (3) The `resample` v7 model predicts
  each of the three cross-condition r values within its 95% CI. (4) `inherit` overpredicts them.
- **Budget.** One look at the test, both models together. Any later model change needs a new
  held-out set.
- **Fit-set note (before fitting, test unseen).** Within-condition clones with >= 2 mature cells:
  full 1,015, G-CSF 666, M-CSF 1,043. The cytokines shift composition as expected (N-only 0.13 /
  0.45 / 0.03; M-only 0.39 / 0.20 / 0.88). Day 4 was barely sampled in full cocktail (128 clones),
  so only 4 clones have cells on both days there (G-CSF 39, M-CSF 131); the day-4 vs day-6
  correlation is therefore fitted for M-CSF only. The model samples days 4 and 6 equally, unlike
  the data; this affects composition statistics, not the test.

- **v7 fits (2026-11-01, test unseen).** Loss: resample 0.74, inherit 0.90. Both are inadequate:
  M-CSF M-only 0.46 simulated vs 0.88 observed; full-cocktail has-O 0.13 vs 0.36; q collapsed to
  0.02-0.07; inherit's b_M sits at the -1.5 bound. Prediction 1 (signs) holds for both. The test
  look is **not** spent on fits that miss the fit set.
- **v8 (pre-registered 2026-11-01, before fitting, test unseen).** Changes, all motivated only by
  the fit-set misfit above: (a) b_G, b_M bounds widened to +-3; (b) per-condition shift of the O
  threshold after the split (t_full, t_G, t_M), since the cocktail makes far more O than M-CSF;
  (c) 4 Nelder-Mead starts (the three v7 starts plus the v7 optimum), maxfev 600 each.
  Predictions 1-4 above carry over unchanged. Adequacy gate before the look: M-CSF M-only and
  full-cocktail has-O each within 0.10 of observed for the `resample` model; if it fails, report
  the misfit and still make no look. One look, both models, `fit_cytokine.py --version v8 --test`.
- **v8 run 1 discarded (implementation bug).** The O shifts started at exactly 0, where
  Nelder-Mead's initial simplex steps only 0.00025, so they never moved (all fits had t = 0).
  Restarted with t_full = -0.5, t_G = 0.3, t_M = 0.5 as starting values; nothing else changed.
  Run 1 best: resample 0.61 (M-CSF M-only 0.74), inherit 0.72. Raw files kept outside the repo.
- **v8 result (run 2, GitHub Actions run 41334724959): adequacy gate FAILED, no look.** Best loss:
  resample 0.65 (start 3), inherit 0.64 (start 2). Full-cocktail has-O is now fitted (resample
  0.34 vs 0.36) but M-CSF M-only is not (resample 0.53, best over starts 0.61, vs 0.88; gate needs
  >= 0.78). Even b_M near -1.0 to -1.3 leaves many N-containing clones under M-CSF. Reading:
  cells already committed before the day-2 split are untouched by an *instructive* drift term,
  so the model cannot empty N out of M-CSF clones. That points to a *selective* effect (M-CSF
  fails to support neutrophil survival or expansion), which the model lacks. Nelder-Mead also
  moves little in 12 dimensions on a noisy loss; a better optimiser is a secondary fix.
  Test set still unseen; the E9 budget of one look is intact.

## E10: learn the decision rule from data (pre-registered 2026-11-01)

- **Motivation.** So far the learned (PPO/ES) cells were trained on invented targets, and every
  real-data result came from a few fitted equations. E10 trains the cells' decision rule on the
  LARRY data itself, so the policy becomes a flexible, data-constrained hypothesis that can be
  compared with the equation model and then inspected.
- **Environment (`train_clone_es.py`).** Non-spatial clones as in E9: daily division (committed
  cells with probability q), pitchfork fate drift with k, sigma, sampling f and q fixed from the
  E6 v6 `resample` fit (not refitted). Each quarter-day an uncommitted cell sees [x, age, memory
  (2), condition one-hot] and picks REST / PUSH_N / PUSH_M / COMMIT_O. The cytokine condition is
  only an input; the response to it is learned. No environment bias is built in.
- **Training.** ES (8 antithetic pairs, sigma 0.05, Adam 0.03, 150 generations, common random
  numbers per generation), fitness = minus the E9 fit-set distance (within-condition statistics
  for full, G-CSF, M-CSF). Memory `inherit` (copy + noise 0.1) vs `resample`. 5 seeds each.
- **Test.** The same cross-condition sister concordance as E9 (full-G, full-M, G-M), one look,
  taken together with E9's look so both models face the same unseen data.
- **Predictions.** (1) The learned policy fits the fit set at least as well as the E9 equation
  model (loss within 25% of E9's). (2) Its held-out predictions are no worse than E9's (absolute
  error summed over the three pairs). (3) Inspection: the policy pushes toward N under G-CSF and
  toward M under M-CSF (it rediscovers the known cytokine roles from clone statistics alone).
  (4) `inherit` vs `resample`: no difference in held-out error (consistent with E6).
- **If (2) fails,** the flexible policy overfits the fit set and the equation model is the better
  scientific description; that will be reported as such.

## E7: can the cells use lineage memory at all? (pre-registered 2026-11-01)

- **Motivation.** In E3-E5, removing memory inheritance never changed the outcome, and E6 found no
  need for faithful memory in real data. Before concluding that the mechanism is useless, check
  that the learned policy *can* use it when the task demands it.
- **Task (`--cue`, `TissueConfig.cue_target`).** Each tissue's founder gets +1 or -1 in memory
  slot 0, visible through no other channel. For -1 tissues the A and B target shares swap
  (10/30/60 instead of 10/60/30). No spatial target. Totipotency window 16 cells, so the cue must
  survive at least four rounds of division before anyone can commit. Entropy 0.05 -> 0.005.
- **Conditions.** `full` (memory inherited with noise 0.1) vs `nomemory` (resampled at division,
  so the cue is lost after the first division). 5 seeds x 400 iterations each.
- **Metrics.** Primary: `score`. Secondary: `cue_accuracy` (share of tissues whose A-vs-B majority
  matches the cue; 0.5 = chance).
- **Predictions.** (1) `full` reaches cue accuracy clearly above chance (CI excluding 0.5).
  (2) `nomemory` stays near 0.5. (3) `full` scores better than `nomemory`, CI excluding 0.
- **If (1) fails,** the shared PPO policy cannot learn to read and propagate memory in this
  budget, and the earlier knockout null results for memory say nothing about the mechanism.

### E7 result (10 runs, 0 collapsed; min size over the last 100 iterations 38-52)

| Condition | Score | Cue accuracy | Size | Stem / A / B | Sister conc. |
|---|---|---|---|---|---|
| full | -0.640 (sd 0.03) | **0.86** (95% CI 0.82-0.90) | 61 | 0.21 / 0.45 / 0.35 | 0.84 |
| nomemory | -0.747 (sd 0.02) | 0.50 (0.47-0.53) | 65 | 0.17 / 0.48 / 0.35 | 0.84 |
| difference | **+0.107 (0.07-0.14)** | **+0.36 (0.31-0.41)** | | | |

(A and B shares are averaged over both cue types, so they sit near 0.45 / 0.45 by design.)

- All three predictions **confirmed.** With memory, 86% of tissues follow a cue that was given
  only to the founder and passed down at least four divisions; without memory, chance (0.50).
- So the shared policy *can* learn to read and propagate lineage memory. The null results for
  memory in E3-E5 mean those tasks did not need it, not that the mechanism or the learner fails.
  This matches E6: in the LARRY data too, nothing required faithful memory.
- Sister concordance is the same in both (0.84): it is driven by local signals and the timing of
  commitment, not by memory, in every study so far. It is therefore not a valid read-out of memory.

## E8: evolution strategies instead of PPO (pre-registered 2026-11-01)

- **Motivation.** PPO needed two fixes (E5) to avoid collapse. ES optimises the final tissue score
  directly, with no credit assignment across cells or steps, and fits the Watson/Levin framing of
  development as an evolved policy.
- **Setup (`train_tissue_es.py`).** Same environment as E5 (spatial target, totipotency window
  16), same network. OpenAI-ES: 8 antithetic pairs per generation, 8 tissues per evaluation,
  sigma 0.05, Adam lr 0.03, rank-normalised fitness, 50 generations = 6,400 training tissue
  episodes, the same number PPO used in E5 (400 iterations x 16 tissues). Evaluation each
  generation on 16 separate tissues (not counted in the budget). 5 seeds. Hyperparameters fixed in
  advance, no tuning.
- **Comparison.** ES vs E5 PPO `full`, same seeds. PPO metrics: mean of the last 30 iterations
  (as before). ES metrics: mean of the last 4 generations (the same share of the budget, ~7.5%).
- **Predictions.** (1) ES never collapses (min size > 1.5 in all seeds). (2) At matched episodes,
  ES scores *worse* than stabilised PPO (ES is less sample-efficient for a 64-unit network);
  stated in advance so that a win is not post-hoc. (3) ES `nomemory` vs ES `full`: no difference,
  as with PPO.

### E8 result (10 ES runs, ~2.2 min each)

| Method | Score | Size | Stem / A / B | Pattern | Min size (late) |
|---|---|---|---|---|---|
| PPO (E5 full, seeds 0-4) | -1.049 (sd 0.02) | 53 | 0.19 / 0.66 / 0.15 | 0.63 | 33 |
| ES full | -1.020 (sd 0.09) | 51 | 0.12 / 0.67 / 0.21 | 0.63 | 44 |
| ES nomemory | -0.943 (sd 0.04) | 51 | 0.14 / 0.61 / 0.25 | 0.58 | 44 |

- (1) **Confirmed:** no ES run collapsed, with neither of the PPO fixes' entropy schedule needed
  (the totipotency window is part of the environment and was kept).
- (2) **Not confirmed:** ES is not worse. ES - PPO = +0.03 (95% CI -0.04 to +0.10): equal at
  matched episodes, with a stem share closer to the target (0.12 vs 0.19) and more B cells.
  ES reaches about -0.9 within ~7 generations (900 episodes), then plateaus with noisy evaluation.
- (3) **Roughly confirmed, with a twist:** ES nomemory scores slightly *better* than ES full
  (-0.08, CI -0.15 to +0.00, just touching 0). Plausible reason: resampled memory gives cells
  random but useful individual variation (symmetry breaking), while inherited memory makes
  neighbours alike. Not established at 5 seeds.
- **Decision.** ES is a valid alternative trainer: as good as stabilised PPO, more robust, much
  simpler, and naturally parallel (ideal for Colab). Keep both; the 20-seed reruns use PPO. ES
  seed-to-seed spread is larger (sd 0.09 vs 0.02), so ES comparisons need more seeds.

---

## Planned (write hypothesis before running)

- **E7, memory test.** A task where lineage memory should matter (fate set by an early cue that
  is then removed), to see whether memory inheritance can be used at all.
- **E8, ES outer loop.** Evolution strategies instead of PPO, same budget; compare stability.

### E10 result, fit set only (2026-11-01, test unseen)

- 10/10 runs finished (5 seeds x {inherit, resample}, 150 generations, ~345 s each on the PC).
  Final loss: resample 0.49 (sd 0.10), inherit 0.51 (sd 0.18); best seeds 0.26-0.28. The fitted
  equations (E9 v8) reach 0.65 at best. The policy has ~1,400 weights against 9-12 equation
  parameters, so a better fit set score is expected and is not evidence by itself.
- **Every run passes the E9 adequacy gate:** M-CSF M-only 0.87-0.94 (observed 0.88); full-cocktail
  has-O 0.24-0.51 (0.36); G-CSF N-only 0.39-0.54 (0.45).
- **What the policies learned (undecided cells near x = 0):** all 10 respond to the condition
  input the same way: under G-CSF mostly PUSH_N (0.44-0.78), under M-CSF mostly PUSH_M
  (0.76-0.97), in full cocktail a mix. A consistent instructive response, discovered with no
  sign constraint. COMMIT_O stays at 0.00-0.02 per step.
- **Open question:** why the policy fits M-CSF and the equations do not. Candidates: E10 fixes
  q from v6 (~0.75, committed cells keep dividing) where the E9 fits drove q to 0.01-0.2; and
  the policy's push can depend on x and memory. To check before claiming a mechanism.
- Memory variant makes no clear difference on the fit set (0.49 vs 0.51, overlapping).

### E10 vs E9 diagnosis (2026-11-01, held-out cross-condition test still unseen)

Question: why do the learned policies fit M-CSF (M-only ~0.9) when the E9 equations cannot (0.53)?

1. **Not q.** E10 policies evaluated with q forced to 0.67 / 0.20 / 0.07: M-CSF M-only 0.91 /
   0.93 / 0.94. Conversely, E9 v8 refit with q fixed at the v6 value (`--fix_q`, sweep
   e9_v8_fixq, Actions run 41337607689): best resample loss 0.63 (v8: 0.65); one start reaches
   M-only 0.88 but then over-predicts G-CSF N-only (0.64 vs 0.45) and full-cocktail has-O
   (0.52 vs 0.36). No start fits all three. q is at most a minor factor.
2. **Timing of commitment at the day-2 split.** Share of cells undecided / N / M / O at the end
   of day 2, when the conditions start:

   | | undecided | N | M | O |
   |---|---|---|---|---|
   | E9 v8 equations (best resample) | 0.27 | 0.35 | 0.32 | 0.05 |
   | E10 policies (mean of 10) | 0.75 | 0.02 | 0.17 | 0.05 |
   | **Data, day-2 cells (cytokine experiment)** | **0.71** | **0.07** | **0.11** | **0.12** |

   The equations commit most cells before the cytokines act, so N-committed cells carry over into
   M-CSF and the drift term cannot remove them. The policies keep cells undecided until the
   condition is known (and lean towards M beforehand), which is what lets M-CSF empty the N fate.
3. **Out-of-sample check (post hoc, not pre-registered, so only supporting evidence).** Day-2
   cell states were never in any fit set. The data show 71% of day-2 cells still
   undifferentiated; E10 gives 75%, the equations 27%. The learned rule recovered the real
   commitment timing from day-4/6 clone composition alone.

**Conclusion.** The E9 failure is structural: in the equation model the fate landscape commits cells
too early, whatever q or bias. The data-trained rule found "stay undecided until the
environment is known", which matches day-2 cell states. Next: the held-out cross-condition
test, E9 (reported as failing its adequacy gate) and E10 together.

### E9 + E10 held-out test: THE single look (2026-11-01)

`fit_cytokine.py --version v8 --test` (E9: best v8 start per memory variant, copied to
fit_v8_{memory}.json; E10: mean of the 5 seeds' predictions). Output: cytokine_test_v8.json.
Test budget for the cytokine cross-condition statistics is now spent.

| Pair | Observed r [95% CI] (clones) | E9 resample | E9 inherit | E10 resample (seed range) | E10 inherit (seed range) |
|---|---|---|---|---|---|
| full-G | 0.69 [0.40, 0.92] (29) | -0.08 | 0.07 | **0.46** (0.37-0.51) | **0.56** (0.47-0.69) |
| full-M | 0.24 [-0.11, 0.55] (84) | **0.15** | **0.17** | **0.20** (0.00-0.36) | **0.07** (0.00-0.18) |
| G-M | 0.51 [0.33, 0.65] (78) | 0.19 | 0.14 | 0.11 (0.00-0.18) | 0.03 (0.01-0.06) |

Bold = inside the observed 95% CI.

- **P1 (bias signs b_G > 0, b_M < 0): confirmed** for both E9 variants; E10 learned the same
  directions unconstrained.
- **P2 (cross-condition r below the in vitro cross-well 0.74): mostly confirmed.** full-M and G-M
  are clearly lower (CIs end at 0.55 and 0.65); full-G (0.69) is not distinguishable from 0.74.
- **P3 (resample predicts all three within CI): not confirmed.** E9 resample 1/3; E10 resample
  2/3 (misses G-M).
- **P4 (inherit overpredicts): refuted.** Every model *under*predicts, inherit as much as resample.
- **Main finding.** Sisters sent into opposite cytokines (G-CSF vs M-CSF) still agree on their
  N/M balance (r = 0.51). Every model predicts near 0-0.2 there: in all of them the opposing
  cytokines largely erase the clone's inherited bias. In the real cells they do not. The
  environment shifts the *level* (M-CSF clones are mostly M) but the clone's *rank* among clones
  survives. Both model families treat cytokine response and inherited bias as competing for the
  same drift; the data say they act more independently (e.g. bias sets the clone's relative
  output, cytokine sets the absolute mix).
- **E10 vs E9.** The data-trained rule is better on the held-out test (4/6 cells inside the CI vs
  2/6), as well as on the fit set and on day-2 states. Neither family passes.
- **Next (new held-out data needed for any refit):** a model where clone bias persists under
  condition-specific selection, tested on the Epo and SCF conditions (never used) or the in vivo
  set.

