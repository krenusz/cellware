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

### E10 distilled (2026-11-01)

Linear fit of each policy's expected push (P(PUSH_N) - P(PUSH_M)) over 20,000 random
observations (results_clone_es/distil/linear_drive.json). R^2 0.76-0.92 per policy. Mean over 10:

    drive = -0.26 - 0.55 x + {full +0.27, G +0.66, M -0.53} (+ small memory terms)

- **Restoring term -0.55 x (negative in all 10 runs, sd 0.24):** cells push against the fate they
  are drifting towards, which flattens the pitchfork and delays commitment; this is the
  mechanism behind the 75% undecided at day 2.
- **Memory terms are small and sign-inconsistent** (|coef| <= 0.34, signs vary across seeds): the
  learned rule essentially ignores clone memory. That is why sisters split into G-CSF and M-CSF
  decorrelate in E10 (predicted r ~ 0.1) while the data keep r = 0.51.

## E11: learned rule with selection, five conditions (pre-registered 2026-11-01)

- **Motivation.** E9/E10 test: every model erases clone bias across opposing cytokines; E10
  ignores memory because nothing in its fitness rewarded it.
- **Model** (`train_clone_e11.py`). E10 plus: (1) selection: committed cells choose divide or
  arrest at each daily division via the policy (replaces fixed q), so a cytokine can favour a
  lineage without redirecting undecided cells; (2) observation adds the committed fate and a
  five-condition one-hot (full, G-CSF, M-CSF, Epo, SCF); (3) clones 1,500.
- **Fit set.** Within-condition statistics for all five conditions (Epo 323, SCF 137 clones), M-CSF
  day-4 vs day-6 r, and the now-spent cross-condition concordances full-G 0.69, full-M 0.24,
  G-M 0.51 (weight 2).
- **Held-out test (never fitted, not yet computed).** N-fraction concordance pooled over every
  condition pair that includes Epo or SCF (55 clone pairs by count; statistic not computed).
  Bootstrap 95% CI.
- **Runs.** 5 seeds x {inherit, resample}, 150 generations, 8 antithetic pairs; GitHub Actions.
- **Predictions.** (1) `inherit` fits G-M concordance (simulated r within 0.33-0.65); `resample`
  cannot (r < 0.33), i.e. clone memory is needed to keep bias across opposing cytokines.
  (2) The distilled `inherit` rule has consistent, non-negligible memory terms (|coef| > 0.2,
  same sign in >= 4 of 5 seeds). (3) `inherit` predicts the pooled Epo/SCF concordance within
  its 95% CI; (4) `resample` underpredicts it (below the CI).
- **Caveat recorded before the test.** An untrained smoke-test policy gave pooled r ~0.99: with
  few N/M cells in Epo/SCF, many pairs are all-N or all-M clones, so this statistic may be high
  for any model that commits cells early. Prediction 4 is therefore the informative one.
- **Budget.** One look, all 10 runs together: `train_clone_e11.py --test`.

### E11 result, fit set only (2026-11-01, held-out Epo/SCF test NOT yet looked at)

10/10 runs (GitHub Actions). Loss levelled off at ~0.8-0.9 by generation ~120 (all ten).

| | loss | G-M r (obs 0.51) | full-G (0.69) | full-M (0.24) | M-CSF M-only (0.88) |
|---|---|---|---|---|---|
| inherit (5) | 0.78-1.01 | 0.17-0.29 | 0.61-0.68 | 0.26-0.37 | 0.75-0.85 |
| resample (5) | 0.65-0.95 | 0.21-0.29 | 0.44-0.69 | 0.26-0.42 | 0.70-0.81 |

- **P1 refuted.** Neither variant reaches the G-M band (0.33-0.65), even with G-M in the loss;
  resample does as well as inherit. Selection lifts G-M from ~0.1 (E10) to ~0.25, not to 0.51.
- **P2 refuted.** Distilled inherit rules have tiny memory terms (|coef| <= 0.15). The policy does
  not use clone memory even when the fitness rewards cross-condition agreement.
- **Selection is not identifiable:** P(divide) by fate and condition varies wildly across seeds
  (e.g. M-CSF keeps M cells dividing in some seeds and arrests them in others). Many selection
  schedules give similar losses.
- Restoring term weaker than E10 (x coef -0.03 to -0.38), and the M-CSF fit got slightly worse
  (0.70-0.85 vs 0.87-0.94): adding cross-condition terms traded off against within-condition fit.
- **Reading.** Either (a) the memory channel as built (2 numbers, noise 0.1 per division, only
  read by the policy) is too weak a carrier for ES to find, or (b) clone bias in real cells
  sits in something the model lacks, e.g. heritable differences in division/survival rate
  rather than in fate choice. These are not distinguishable from the fit set.
- **Test look withheld (2026-11-01).** P1 and P2 failed on the fit set, and the pooled Epo/SCF
  statistic is weak (55 pairs; an untrained policy already scores ~0.99), so the look would say
  little about E11. The Epo/SCF test stays unseen and moves to `test_oracle.py` (one look per
  pre-registered experiment, logged in test_ledger.tsv); `train_clone_e11.py --test` is disabled.

## E12: heritable clone expansion trait (pre-registered 2026-11-01)

- **Motivation.** E11 reading (b): clone bias may live in heritable differences in how a clone's
  N vs M cells expand, not in fate choice. If so, the clone's rank survives opposing cytokines.
- **Model.** E11 (`train_clone_e11.py --trait`) plus one clone-level value u ~ N(0,1), fixed for
  the clone and shared by all its cells. At each division, a committed cell's divide logit gets
  `+ s u` if N and `- s u` if M; s is learned with the policy (init 0.5). Nothing else changes.
- **Fit set.** Same as E11 (five conditions within-condition, M-CSF day-4 vs day-6, full-G /
  full-M / G-M concordance).
- **Runs.** 5 seeds x {inherit, resample}, 150 generations, GitHub Actions. Control: E11 runs.
- **Predictions (fit set).** (1) G-M r reaches the 0.33-0.65 band in >= 3 of 5 runs per memory
  variant (E11: 0/10). (2) Learned s > 0.5 (the trait is used, not switched off) in >= 4 of 5.
  (3) Within-condition fit no worse than E11 (loss within E11's range).
- **Held-out test.** Pooled Epo/SCF concordance, through test_oracle.py, only if (1) passes.

### E12 result, fit set only (2026-11-01; Epo/SCF test still unseen)

| | loss | learned s | G-M r (obs 0.51, band 0.33-0.65) | M-CSF M-only (0.88) |
|---|---|---|---|---|
| inherit (5) | 0.76-1.17 | 0.38-0.98 | 0.08-0.29 | 0.78-0.90 |
| resample (5) | 0.62-0.95 | 0.43-1.15 | 0.15-0.31 | 0.74-0.93 |

- **P1 refuted:** 0/10 in the G-M band (E11 0/10); best 0.31.
- **P2 partly:** s > 0.5 in 7/10 (inherit 3/5, resample 4/5). The trait is used but does not raise G-M.
- **P3 holds:** loss range comparable to E11 (0.62-1.17 vs 0.65-1.01); M-CSF fit slightly better.
- **No test look** (pre-registered condition: P1 must pass).
- **Ceiling check (toy, not a fit).** With clone N-propensity fully shared across conditions
  (Beta(1,1) across clones, cytokines shifting only the level) and the observed sampling depth
  (~4-5 mature cells per clone), the G-M r this statistic can reach is only ~0.36 (3 cells) to
  0.47 (6 cells). The observed 0.51 [0.33, 0.65] is at that ceiling: in real cells the N/M
  propensity is essentially fully clone-determined before the split and survives opposing
  cytokines intact, with a broad spread across clones. Our models reach about half of that.
- **Reading.** What is missing is not a weak extra term but a strong, broad, clone-level fate
  propensity fixed by day 2 that the cytokines rescale without reshuffling. E9 inherit had memory
  in the drift but mixed it with fresh noise each division (s_mem) and fixed early commitment;
  E10-E12 policies ignore memory or use the trait weakly. Next model should make the founder
  propensity the dominant, exactly inherited variable and let the data set its spread.


## E13: exactly inherited founder fate propensity (pre-registered 2026-11-02)

- **Motivation.** E12 ceiling check: the observed G-M concordance (0.51) is at the ceiling for
  ~4-5 cells per clone, so the clone's N/M propensity must be broad, fixed before day 2 and
  rescaled (not reshuffled) by cytokines. E10-E12 lack a strong clone-level fate variable.
- **Model.** E11 (`train_clone_e11.py --founder`) plus one clone value u ~ N(0,1), shared exactly
  by every cell of the clone for all six days, added to the undecided cells' fate drift as
  `b u` (positive pushes N). b is learned with the policy (init 1.0) and sets the spread across
  clones; the policy still sees the condition and can rescale everything else. No E12 trait.
- **Fit set, runs, control.** Same as E11/E12: 5 seeds x {inherit, resample}, 150 generations,
  GitHub Actions; controls E11 and E12.
- **Predictions (fit set).** (1) G-M r in the 0.33-0.65 band in >= 3 of 5 runs per memory
  variant (E11 0/10, E12 0/10). (2) Learned b > 1.0 in >= 4 of 5 (the propensity is used and
  broad). (3) Loss within E11/E12 range (<= 1.17) in >= 4 of 5.
- **Held-out test.** Pooled Epo/SCF concordance through test_oracle.py, whole test (1 shard),
  only if (1) passes for both memory variants; else only for a passing variant's runs, one look.

### E13 result, fit set only (2026-11-02; Epo/SCF test still unseen)

| run | loss | learned b | G-M r (obs 0.51) | full-G (0.69) | full-M (0.24) |
|---|---|---|---|---|---|
| inherit s0-s4 | 0.60, 0.94, 1.86, 0.77, 1.01 | -0.06, -0.13, 0.43, 0.04, 0.01 | 0.11, 0.28, **0.58**, 0.15, 0.11 | 0.33-0.74 | 0.18-0.60 |
| resample s0-s4 | 0.98, 0.55, 0.95, 1.07, 0.91 | 0.20, -0.04, -0.06, -0.02, 0.09 | 0.22-0.28 | 0.48-0.64 | 0.28-0.31 |

- **P1 refuted:** 1/5 inherit, 0/5 resample in the G-M band.
- **P2 refuted:** b > 1.0 in 0/10; ES shrinks the founder term towards zero in 9/10 (|b| <= 0.2).
- **P3 holds in 9/10** (loss <= 1.17); the exception is the one run that kept b (inherit s2).
- **No test look** (P1 failed).
- **Reading (post hoc, one run).** The only run that kept a founder propensity (b = 0.43) is also
  the only one with G-M in band (0.58), but its within-condition loss is the worst of all (1.86).
  In this architecture a shared clone fate bias that raises cross-condition concordance costs
  within-condition fit, and the optimiser prefers the within-condition terms (5 conditions vs
  3 cross pairs at weight 2). Either the drift-bias form is wrong (it distorts clone size / fate
  mix distributions), or the weighting decides. Next diagnostic, no test needed: scan fixed b
  (0, 0.25, 0.5, 1) and W_CROSS, map the loss trade-off frontier.

## E13 scan: founder strength vs cross weight (diagnostic, fit set only, 2026-11-02)

- Fixed b in {0, 0.25, 0.5, 1} x W_CROSS in {2, 5}, inherit, seeds 0-1, 150 generations.
- Question: the loss cost of raising G-M towards 0.51. Expectation (from E13 s2): G-M rises with
  b, within-condition loss rises too. No test look in this diagnostic.

### E13 scan result (2026-11-02; fit set only, no test look)

| b | W=2: loss / G-M (s0, s1) | W=5: loss / G-M (s0, s1) |
|---|---|---|
| 0 | 0.80 / 0.25, 0.77 / 0.18 | 1.03 / 0.36, 1.06 / 0.26 |
| 0.25 | 0.94 / 0.26, 0.93 / 0.31 | 0.89 / 0.34, 1.12 / 0.39 |
| 0.5 | 0.95 / 0.35, 1.79 / 0.54 | 1.19 / 0.45, 2.32 / 0.70 |
| 1 | 1.33 / 0.36, 1.82 / 0.73 | 1.44 / 0.53, 2.36 / 0.45 |

(Loss includes the weighted cross terms, so W=2 vs W=5 losses are not strictly comparable.)
- **Frontier.** G-M rises with b and with W. In-band G-M at E11-like loss (<= ~1.2): b 0.25-0.5
  with W=5 (0.34, 0.39, 0.45). Reaching the observed 0.51 or above costs loss >= 1.4 in every run.
- **Seed instability.** At b >= 0.5 the seed-1 runs jump to high G-M with high loss (1.8-2.4):
  a second optimum that buys concordance with a poor within-condition fit.
- **Reading.** A moderate founder propensity (b ~ 0.25-0.5) plus a higher cross weight gets
  into the G-M band without wrecking the fit; the full 0.51 is not reachable at acceptable fit
  in this drift-bias form. The data's clone bias is stronger than this model's form can carry.
- **Candidate for a pre-registered test model:** b = 0.25-0.5, W = 5 (choose before any look).

## E14: moderate fixed founder propensity (pre-registered 2026-11-02)

- **Model.** E13 drift-bias founder term with b fixed at 0.35 and W_CROSS = 5, chosen from the
  E13 scan before any test look. Outputs e13scan_b0.35_w5_{memory}_seed{s}.json.
- **Runs.** 5 seeds x {inherit, resample}, 150 generations, GitHub Actions.
- **Predictions (fit set).** (1) G-M r in 0.33-0.65 in >= 3 of 5 per memory variant.
  (2) Loss <= 1.3 in >= 4 of 5 per variant. (3) M-CSF M-only within 0.10 of 0.88 in >= 4 of 5.
- **Held-out test.** If (1) and (2) pass for a variant, one whole-test look (1 shard) at pooled
  Epo/SCF via test_oracle.py --exp E14, comparing the mean prediction of that variant's passing
  runs with the observed r. Success: observed r inside the runs' prediction range +/- 0.10.
  If both variants pass, both predictions are recorded before the single look.

### E14 result, fit set only (2026-11-02; Epo/SCF test still unseen)

| run | loss | G-M r (band 0.33-0.65) | M-CSF M-only (0.88) |
|---|---|---|---|
| inherit s0-s4 | 1.41, 2.26, 1.18, 1.03, 2.05 | 0.41, 0.63, 0.47, 0.44, 0.58 | 0.72, 0.41, 0.73, 0.73, 0.37 |
| resample s0-s4 | 1.12, 2.46, 1.36, 2.48, 1.44 | 0.45, 0.57, 0.43, 0.69, 0.40 | 0.68, 0.41, 0.71, 0.38, 0.69 |

- **P1 holds:** inherit 5/5, resample 4/5 in the G-M band (E11-E13: 1/30).
- **P2 refuted:** loss <= 1.3 in inherit 2/5, resample 1/5.
- **P3 refuted:** M-CSF M-only within 0.10 of 0.88 in 0/10 (best 0.73; bimodal 0.37-0.41 vs ~0.70).
- **No test look** (P2 failed). Predicted pooled Epo/SCF r recorded in the result files (0.42-0.70);
  not compared with data.
- **Reading.** A fixed founder bias reliably buys sister concordance but breaks M-CSF: the clone
  bias in the drift pulls some cells out of M even in M-CSF, so pure-M clones fall from 0.88 to
  ~0.7 (or collapse to ~0.4 in a second optimum). In the data, M-CSF overrides the clone bias
  within condition while the clone's rank survives across conditions. That points to the bias
  acting on *which* lineage a clone expands relative to its sisters (threshold / rank), with
  the cytokine setting the absolute level, rather than an additive drift that competes with it.

## E15: clone rank in the N/M assignment, cytokine sets the level (pre-registered 2026-11-02)

- **Motivation.** E14: an additive founder bias in the drift buys G-M concordance but competes
  with the cytokine push (capped at PUSH = 0.6), breaking M-CSF (M-only ~0.7 vs 0.88).
- **Model.** E11 (`train_clone_e11.py --rank`). An undecided cell commits when |x| crosses the
  threshold (either side); N vs M is then drawn with p_N = sigmoid(2 sign(x) + b u - tau_c), with
  u ~ N(0,1) fixed and shared exactly by the clone, b (init 1) and a per-condition level tau_c
  (6 values incl. pre-split, init 0, unbounded) learned with the policy. The cytokine can set the
  level as far as the data need; the clone sets the rank. No additive drift bias.
- **Runs.** 5 seeds x {inherit, resample}, 150 generations, W_CROSS = 2, GitHub Actions.
- **Predictions (fit set).** (1) G-M r in 0.33-0.65 in >= 3 of 5 per variant. (2) Loss <= 1.3 in
  >= 4 of 5. (3) M-CSF M-only within 0.10 of 0.88 in >= 4 of 5. (4) Learned b > 1.0 in >= 4 of 5.
- **Held-out test.** If (1)-(3) pass for a variant: one whole-test look at pooled Epo/SCF via
  test_oracle.py --exp E15. Prediction = range of that variant's passing runs; success if the
  observed r lies within it +/- 0.10. If both pass, both ranges are recorded before the look.

### E15 result, fit set only (2026-11-02; Epo/SCF test still unseen)

| run | loss | b | G-M r (band 0.33-0.65) | M-CSF M-only (0.88) | tau_M |
|---|---|---|---|---|---|
| inherit s0-s4 | 0.66-0.93 | 0.67-1.73 | 0.20-0.31 | 0.65-0.77 | 0.4-1.0 |
| resample s0-s4 | 0.73-1.24 | 0.38-1.86 | 0.20-0.33 | 0.59-0.74 | 0.4-0.8 |

- **P1 refuted:** inherit 0/5, resample 1/5 (0.33, at the edge).
- **P2 holds:** loss <= 1.3 in 10/10 (best E11-E15 fits: 0.66).
- **P3 refuted:** M-CSF M-only 0.59-0.77, 0/10 within 0.10 of 0.88 (worse than E11, 0.78-0.90).
- **P4 holds:** b > 1.0 in 7/10; the clone rank is used and broad, yet G-M stays low.
- **No test look.**
- **Reading.** The rank term is kept (unlike E13) but G-M does not rise, and the M-CSF level tau_M
  stays small (0.4-1.0, logit units): with ES lr 0.03, the per-condition levels barely move from 0
  in 150 generations, so M-CSF cannot override the clone rank and pure-M clones fall. E15 is
  therefore not a clean test of the rank idea: the levels are optimisation-limited. A fair rerun
  needs tau initialised from the observed per-condition N fractions (or a larger step for tau).

## E16: E15 with cytokine levels started from the data (pre-registered 2026-11-02)

- **Motivation.** E15 was optimisation-limited: tau_c barely moved from 0, so M-CSF could not
  override the clone rank. E16 is the fair test of the rank idea.
- **Model.** E15 (`--rank --tau_obs`) with tau_c initialised to -logit(observed N share among
  N+M cells in condition c); pre-split uses the full cocktail. Start values: full 0.98,
  G -0.77, M 3.25, Epo -0.20, SCF -0.08 (fit-set statistics; Epo/SCF within-condition shares are
  fit data, not the held-out cross statistic). Everything else as E15.
- **Runs, predictions, test.** Identical to E15: (1) G-M in 0.33-0.65 in >= 3 of 5 per variant;
  (2) loss <= 1.3 in >= 4 of 5; (3) M-CSF M-only within 0.10 of 0.88 in >= 4 of 5; (4) b > 1.0
  in >= 4 of 5. One whole-test Epo/SCF look via test_oracle.py --exp E16 if (1)-(3) pass for a
  variant; success if the observed r lies within the passing runs' range +/- 0.10.

### E16 result, fit set only (2026-11-02; Epo/SCF test still unseen)

| run | loss | b | G-M r (band 0.33-0.65) | M-CSF M-only (0.88) | tau_M (start 3.25) |
|---|---|---|---|---|---|
| inherit s0-s4 | 0.49-0.71 | 0.09-0.88 | 0.28-0.31 | 0.70-0.77 | 3.2-3.8 |
| resample s0-s4 | 0.59-0.74 | 0.27-1.54 | 0.23-0.34 | 0.70-0.79 | 3.5-4.2 |

- **P1 refuted:** inherit 0/5, resample 1/5 (0.34).
- **P2 holds:** 10/10; best fits of the series (0.49-0.74).
- **P3 refuted:** 0/10 (0.70-0.79), even with M-CSF's level near 96% M.
- **P4 refuted:** b > 1.0 in 2/10; with a fair start ES shrinks the clone rank again.
- **No test look.**
- **Reading.** With the levels no longer optimisation-limited, the rank form behaves like E13:
  the fitness prefers a weak clone bias, and G-M stays ~0.3. Across E11-E16 (60 runs, five
  mechanisms) G-M reaches the band only when the within-condition fit is sacrificed (E13 s2,
  E14). Either the within-condition statistics the fit uses conflict with the cross statistic
  under any shared-rule model of this family, or the cross statistic (55 clones, CI 0.33-0.65)
  is too thin to pull against five within-condition terms. Next: test that conflict directly
  (fit only the cross terms and read the within-condition loss), not another mechanism.

## E17: conflict test, cross terms vs within-condition terms (diagnostic, 2026-11-02)

- **Question.** Is G-M ~0.3 in E11-E16 forced by a conflict between the within-condition
  statistics and the cross statistic, or is the cross term simply outweighed?
- **Setup.** E16 model (`--rank --tau_obs`), within-condition weight 0 (cross terms only) and
  0.25, 3 seeds x {inherit, resample}. Every run also reports the unweighted within-condition
  loss (`within_loss`, comparable to E16's 0.49-0.74, which includes no cross terms... E16 loss
  included cross terms at W=2, so compare within_loss to E16 runs re-scored the same way).
- **Expectation, written before running.** If G-M reaches ~0.5 with weight 0 and within_loss
  stays near E16, the earlier failure is weighting/optimisation. If G-M reaches ~0.5 only with a
  large within_loss rise, there is a real conflict for this model family. If G-M stays ~0.3 even
  with weight 0, the model family cannot produce the concordance at all. No test look.

### E17 result (2026-11-02; diagnostic, no test look)

Unweighted within-condition loss (E16 re-scored the same way: 0.34-0.61):

| within weight | within loss | G-M r | M-CSF M-only |
|---|---|---|---|
| 1 (E16, 10 runs) | 0.34-0.61 | 0.23-0.34 | 0.70-0.79 |
| 0.25 (6 runs) | 0.66-0.94 | 0.27-0.38 | 0.61-0.71 |
| 0, cross only (4 of 6; 2 degenerated, no valid clones) | 2.38-2.99 | 0.33-0.46 | 0.28-0.64 |

- **Even optimising only the cross terms, G-M tops out at 0.46** (mean of 5 simulations per run),
  with the within-condition fit 4-9x worse than E16 and M-CSF broken.
- **Reading.** Both: (a) a real conflict, since every step towards higher G-M costs
  within-condition fit; and (b) a hard limit, since this model family barely exceeds ~0.45 even
  unconstrained, which matches the E12 sampling ceiling (0.36-0.47 at 4-5 cells per clone). The
  observed 0.51 [0.33, 0.65] sits at or above that ceiling with a wide CI. The likeliest account:
  the observed value is partly a high draw of a thin statistic (55 clones), and the true
  concordance is near the ceiling, i.e. clone fate is essentially fixed before day 2. Additional
  mechanisms in this family will not resolve it; better data (deeper sampling per clone) would.
- **Status of the modelling line.** E11-E17 close the cytokine question for this dataset: best
  models fit within-condition data well (loss 0.34-0.61) and predict G-M 0.23-0.34, at or just
  below the lower CI bound. The Epo/SCF test remains unseen.

---

# Wider programme (2026-11-02): four parallel tracks

Rules for every track: the whole grid is pre-registered with its decision rule; 10 seeds per
setting; report median and spread, not the best run; `health_check.py` marks runs where a learned
scalar did not move (< 0.05), a condition kept < 50 clones, or the loss was still falling; failed
runs are training failures, excluded and counted. Held-out tests: Epo/SCF is split into 2 clone
shards (test_oracle `--shards 2`), one per track that passes its gates.

## Track A: late commitment / reversion (pre-registered 2026-11-02)

- **Target.** M-CSF day-4 vs day-6 clone N-fraction correlation: observed 0.32, E16 models
  0.59-0.87 (every model since E6 overshoots: clone composition is too stable over time).
- **Grid.** E16 model (`--rank --tau_obs`) x {day_obs: policy sees the day; revert: committed
  N/M cells revert to undecided at learned rate sigmoid(r0 + r1 (day - 4)); both} x {inherit,
  resample} x 10 seeds = 60 runs (`sweeps/track_a.txt`, outputs a_*).
- **Predictions.** (1) At least one form brings median M r_d4_d6 within 0.15 of 0.32 (E16
  median 0.83). (2) That form keeps median within-condition loss <= 0.8 (E16 re-scored 0.34-0.61).
  (3) revert forms learn a non-trivial rate (r0 > -2.5 in >= 6 of 10 healthy runs).
- **Gate for a test look.** (1) and (2) pass and median G-M in 0.33-0.65: one look at Epo/SCF
  shard 0 of 2 (`--exp A`), prediction = that form's healthy-run range, success if the observed
  shard value lies within it +/- 0.10.

## Track C: robustness map (pre-registered 2026-11-02)

- **Question.** Are the E11/E16 conclusions properties of the model family or of one setting?
- **Grid.** {E11 base, E16 rank} x W_CROSS {2, 5} x clones {1500, 4000} x 10 seeds, inherit
  (`sweeps/track_c.txt`, outputs c_*), 80 runs. No test look.
- **Predictions.** (1) Median G-M stays < 0.40 in every cell (the E17 limit is not an artefact
  of 5 seeds or 1500 clones). (2) More clones lowers seed spread of G-M (IQR at 4000 < at 1500)
  in >= 3 of 4 pairs. (3) W_CROSS 5 raises median G-M and raises median within loss in both
  models (the trade-off is systematic).

## Track B: day-2 expression as the clone state (step 1 pre-registered 2026-11-02)

- **Step 1 (local, descriptive, in vitro fit data only).** For clones with day-2 cells and >= 3
  day-4/6 N+M cells, predict the clone's later N fraction from its day-2 cells' state, 5-fold
  cross-validated by clone. State: (a) SPRING coordinates (metadata), (b) top-50 PCs of the
  normalised expression. Predictors: kNN regression and ridge. Report CV r and the share of
  across-clone variance explained.
- **Prediction.** Day-2 state explains a substantial part (CV r >= 0.5 with PCs), and less with
  SPRING-2D. Weinreb 2020 reported that day-2 state predicts fate only partially; the residual
  is the "hidden" clone component our models need.

## Track D: dataset search

Model-knowledge shortlist (unverified, no web search yet): Kucinski 2024 LARRY in vivo,
SPLINTR 2022 (AML), DARLIN 2023, Rodriguez-Fraticelli 2020, ReDeeM 2024. Verify accessions,
clone counts and per-clone depth before any download.

### Track B step 1 result (2026-11-02; in vitro fit data, descriptive)

571 clones with day-2 cells (mean 1.5 per clone) and >= 3 later N+M cells (mean 11). 5-fold CV
by clone:

| state features | kNN CV r (R2) | ridge CV r (R2) |
|---|---|---|
| SPRING 2D | 0.61 (0.37) | 0.48 (0.23) |
| top-50 PCs of expression (2000 most variable genes, log1p) | 0.51 (0.21) | **0.66 (0.43)** |

- **Prediction holds:** PCs reach CV r 0.66 (>= 0.5). SPRING is lower with ridge, close with kNN.
- **Ceiling:** sampling noise of the target allows R2 up to ~0.96, so ~55% of the across-clone
  variance in later fate is *not* explained by the day-2 state of 1-2 sampled cells. Part of it
  is that 1-2 cells are a noisy sample of the clone's state; part may be truly hidden
  (epigenetic, as DARLIN reports methylation tracks clonal memory better than expression).
- **Implication for the models.** The "clone bias" our simulators had to invent is largely
  measurable: a day-2 expression readout carries ~40% of it. Step 2: give the learned rule a
  measured state (e.g. the ridge-predicted N propensity of the founder's day-2 state, sampled
  from the empirical distribution) instead of a latent u, and test whether that closes G-M
  without breaking M-CSF.

### Track B depth check (2026-11-02)

CV r of the PC-ridge prediction by day-2 cells per clone: 1 cell 0.67 (382 clones), 2 cells 0.62
(118), >= 3 cells 0.68 (71). No rise with depth: the unexplained ~55% is not mainly from sampling
too few day-2 cells. It is either absent from expression (e.g. epigenetic) or set after day 2.

## Track B2: measured founder propensity (pre-registered 2026-11-02)

- **Model.** E16 (`--rank --tau_obs`) but the clone value is not latent N(0,1): each simulated
  founder draws a real clone's CV-predicted N propensity from day-2 expression
  (`results_track_b/propensity.npy`, 571 in vitro clones), used as z = logit(p); N/M assignment
  logit = 2 sign(x) + b z - tau_c, b learned (init 1). Assumes the in vitro clone-state
  distribution transfers to the cytokine experiment (same lab, same progenitor population).
- **Runs.** {inherit, resample} x 10 seeds (`sweeps/track_b2.txt`, outputs b2_*).
- **Predictions vs E16.** (1) Median G-M higher than E16 (0.30) and in 0.33-0.65 for >= 5 of 10
  healthy runs per variant. (2) Median within-condition loss <= 0.8. (3) M-CSF M-only median
  >= 0.80 (E16 0.70-0.79).
- **Gate.** (1)-(3) pass for a variant: Epo/SCF shard 1 of 2 (`--exp B2`), prediction = healthy-run
  range, success if the observed shard value is within it +/- 0.10.

### Track D result: dataset search (2026-11-02, web-verified where stated)

| Dataset | Design relevant to us | Access | Verdict |
|---|---|---|---|
| Rodriguez-Fraticelli 2020 Nature (TCF15), LARRY in transplanted HSCs | sisters across HSC and mature compartments, 16-24 wk | GEO GSE134242 (via OmicsDI; check on GEO) | best next test: clone state (HSC transcriptome) vs fate, in vivo |
| Scherer 2025 Nature (EPI-Clone), LARRY + single-cell methylation | clone fate vs methylation state: tests the "hidden = epigenetic" reading of track B | Nature / PMC12240852; accession not yet found | high value for B's open question |
| Singh 2025 Cell Stem Cell, LARRYv2 HSCs +/- leukemic drivers | clonal responses split by mutation | accession not checked | condition split, but leukemic |
| DARLIN (Li 2023 Cell) | single time point, no sister split | CoSpar link 404; Zenodo 10.8 GB | not a test for our statistics |
| Xue 2026 Adv Sci, human hematopoietic organoids, LARRY + MAESTER | day 4 vs day 8, human; reports LARRY barcode homoplasy | PMC13042525 | human in vitro time series; check |
| PolyloxExpress (GSE152555) | in vivo HSC clones | GEO | secondary |

No public dataset found that repeats LARRY's split-across-cytokines design with deeper clones.
Caveat for all LARRY-based analyses (Xue 2026, Scherer 2025): barcode homoplasy / multiple
integrations can merge distinct clones, which inflates "multi-lineage" clones.

### Track B2 result (2026-11-02; fit set only, Epo/SCF shards unseen)

Health: 11/20 healthy (5 runs: b did not move; 6: loss still falling, i.e. 150 generations
not converged). Healthy runs:

| variant (healthy) | within loss | b | G-M r (band 0.33-0.65) | M-CSF M-only (0.88) |
|---|---|---|---|---|
| inherit (4) | 0.49-0.68 | 0.51-0.90 | 0.29-0.33 (median 0.31) | 0.71-0.78 |
| resample (7) | 0.31-0.73 | 0.79-1.41 | 0.24-0.33 (median 0.31) | 0.72-0.80 |

- **P1 refuted:** median G-M 0.31 (E16 0.30); <= 2 runs per variant reach 0.33.
- **P2 holds:** within loss <= 0.8 in all healthy runs.
- **P3 refuted:** median M-only ~0.76 (< 0.80).
- **No look.**
- **Reading.** Replacing the latent clone value with the measured day-2 propensity changes
  nothing: G-M stays ~0.3. Together with E16, the shape of the clone-value distribution is not
  the limit. 9/20 runs failed health, mostly by not converging: future sweeps should use >= 250
  generations or the health rule needs a noise-tolerant convergence test.

### Track A result (2026-11-02; fit set only, no look)

Healthy runs, median [range] (`summarise_grid.py`); target M-CSF day-4 vs day-6 r = 0.32:

| form | healthy | within loss | M r_d4_d6 | G-M | M-only | revert r0 (start -3) |
|---|---|---|---|---|---|---|
| day, inherit / resample | 5, 6 | 0.64, 0.54 | 0.79, 0.77 | 0.31, 0.28 | 0.71, 0.73 | - |
| revert, inherit / resample | 6, 6 | 0.58, 0.57 | 0.79, 0.83 | 0.31, 0.31 | 0.75, 0.75 | -1.9 to -3.7 |
| day+revert, inherit / resample | 7, 9 | 0.59, 0.59 | 0.72, 0.79 | 0.29, 0.31 | 0.73, 0.73 | -2.2 to -3.2 |

- **P1 refuted:** best median 0.72, far from 0.32 +/- 0.15 (one run reached 0.27).
- **P2 holds:** within loss <= 0.8.
- **P3 refuted:** r0 > -2.5 in 3/28 healthy revert runs; reversion is not used.
- **Reading.** Neither time-aware commitment nor reversion lowers the day-4/6 correlation:
  the fitness does not reward it enough, or the observed 0.32 is dominated by sampling (few
  cells per clone per day). Next: compute the sampling ceiling of r_d4_d6 as for G-M (a model
  with perfectly stable clones, observed depth) before any further mechanism.

### Track C result (2026-11-02; robustness map, no look)

| model, W, clones | healthy | within loss | G-M | M r_d4_d6 | M-only |
|---|---|---|---|---|---|
| E11 W2 1500 / 4000 | 9, 10 | 0.71, 0.65 | 0.25, 0.27 | 1.00, 0.99 | 0.78, 0.82 |
| E11 W5 1500 / 4000 | 7, 9 | 0.88, 0.78 | 0.35, 0.33 | 0.98, 0.99 | 0.72, 0.79 |
| E16 W2 1500 / 4000 | 6, 8 | 0.55, 0.42 | 0.29, 0.29 | 0.79, 0.75 | 0.73, 0.76 |
| E16 W5 1500 / 4000 | 7, 8 | 0.60, 0.55 | 0.31, 0.32 | 0.73, 0.77 | 0.72, 0.75 |

- **P1 holds:** median G-M < 0.40 in all 8 cells (max 0.35); the limit is robust to seeds and
  simulation size.
- **P2 inconclusive:** ranges at 4000 are not consistently narrower.
- **P3 holds:** W 5 raises G-M and within loss in both models.
- **Extra.** 4000 clones improves within loss in all four pairs (simulation noise was part of the
  loss); E16's rank model fits best (0.42) and is the only family that lowers M r_d4_d6 (0.75 vs
  0.99), so it is the right base.
- Infra: two concurrent `run_sweep.sh` calls picked up the same run id (track A results fetched
  manually); run-id lookup is now locked.

### Day-4/6 sampling ceiling (2026-11-02; resolves track A's target)

`ceiling_d4d6.py`: perfectly stable clones (one N share per clone for both days), share drawn
from a Beta fitted to the observed clones after removing binomial noise, each clone sampled at
its observed day-4 / day-6 depth (131 clones, median 3 and 5 N+M cells).

- Observed M-CSF r_d4_d6 = 0.32, 95% bootstrap CI **[-0.02, 0.88]**.
- Stable-clone ceiling: median **0.81**, 95% range [0.03, 0.99].
- **Reading.** Under M-CSF almost every clone is pure M (Beta(0.02, 0.92)); r is driven by a
  handful of clones with any N, so the statistic is almost uninformative. The models' 0.72-0.83
  match perfectly stable clones; the observed 0.32 is compatible with them. Track A chased noise.
  This term should be dropped or replaced by a pooled, better-powered time statistic in any
  future fit (decide in a pre-registration, not post hoc for a test).

### Health re-check with a noise-tolerant convergence rule (2026-11-02)

New rule: loss trend over the last 60 generations must not fall by > 2x its residual noise and
> 5%. Applied to tracks A, B2, C (all 150 generations): **74/160 healthy**; 80 runs still
improving (75 only that, 5 with b frozen), 6 with b frozen only. Most 150-generation runs had not
converged, so A/B2/C medians are pre-convergence and may understate what each model can do.
Defaults raised: `--generations 250`, `--clones 4000` (C showed lower loss at 4000).

### Track D follow-up: data access (2026-11-02, verified)

- **GSE134242** (Rodriguez-Fraticelli 2020): confirmed on GEO. 90 samples, inDrop v3,
  barcoded HSC and KIT+ progenitor libraries from 5 primary and secondary recipients, plus
  CROP-seq and TetO-Tcf15 experiments. Supplementary per sample: counts_norm.csv.gz, genes,
  barcode lists, cell-barcode maps. GEO did not return file sizes to this machine.
- **EPI-Clone** (Scherer 2025): data on Figshare as Seurat .rds (needs R or an RDS reader):
  M.1-M.3 HSC methylation 173 MB (10.6084/m9.figshare.24204750); M.4 LARRY mature myeloid 10 MB
  (25472467); scRNA-seq of barcoded progenitors 866 MB (24260743); count matrices GSE282971;
  code github.com/veltenlab/EPI-clone.

## Converged reruns R: E14, E16, B2 at 250 generations / 4000 clones (pre-registered 2026-11-02)

- **Why these.** Failures that could be training artefacts: E14 (fit broke; ES may not have
  finished), E16 and B2 (most runs still improving at 150). Not rerun: E11-E13, E15 (superseded
  by E14/E16), track A (its target is sampling noise), E17 (diagnostic), C (E16 cells are covered).
- **Grid.** {E14 fixed b 0.35 W 5, E16, B2} x {inherit, resample} x 10 seeds, fit set unchanged
  (incl. the day-4/6 term, so only training length and simulation size change).
  `sweeps/reruns_250.txt`, outputs r_e14_*, r_e16_*, r_b2_*.
- **Predictions.** (1) >= 70% of runs healthy under the noise-tolerant rule. (2) Within loss
  falls vs the 150-generation runs (median, per model). (3) The G-M limit holds: median G-M of
  healthy E16 and B2 runs < 0.40. (4) E14's M-CSF M-only median rises to >= 0.80 if its failure
  was undertraining.
- **Gates.** Unchanged from the original pre-registrations (E14, E16, B2). A model passing its
  own gates gets its Epo/SCF shard (E16/B2: shard 1 of 2; E14: shard 0 of 2), one look.
- **Also from now on:** `--no_d4d6` available; the next new fit set drops the M-CSF day-4/6 term
  (sampling-dominated). Not used in these reruns.

### Track B3: EPI-Clone methylation vs clone fate (2026-11-02; exploratory, new dataset)

Data: Scherer 2025 Figshare M.4 (LARRY_mature.RDS, 10 MB) and M.1-M.3 (173 MB), read with the
Python `rdata` package (installed into the venv, not in pyproject). 5-fold CV by clone, ridge.

- **M.4 (mature tissues; lymphoid share of mature cells from progenitor state):** inconclusive.
  101 clones, but the target barely varies (mean 0.08, var 0.012): methylation CV r 0.08,
  protein 0.14, both R2 < 0.
- **M.1-M.3 (LARRY mice 1-4, progenitor compartment; myeloid share GMP / (GMP + MEP + EryP +
  MkP) from the methylation of the clone's HSC/MPP1-4 cells):** 181 clones, 14 multipotent and
  57 committed cells per clone.

  | state | CV r |
  |---|---|
  | methylation, top-20 PCs | **0.41** (R2 0.10; noise ceiling R2 0.44) |
  | permuted null | -0.04 |
  | multipotent cell-type composition only | 0.02 |
  | composition + methylation | 0.39 |

- **Reading.** The methylation state of a clone's multipotent cells predicts its later
  myeloid vs erythroid bias, and not through which multipotent subtypes it contains. It
  explains ~22% of the explainable variance (0.10 / 0.44). Different lineage axis, tissue and
  readout from LARRY in vitro, so not directly comparable with track B's expression result
  (43% of N/M variance). It supports the epigenetic-memory reading but does not show that
  methylation carries what expression missed: that needs both readouts in the same cells
  (EPI-Clone's scRNA dataset, 866 MB, is a separate experiment).

### Track B3c-d: methylation vs expression in the same EPI-Clone experiments (2026-11-02)

- **Same-clone comparison not possible:** the methylation (M.1-M.3) and scRNA objects share only
  ~55 LARRY barcodes (527 vs 606 clones); 12 pass the filters. `track_b_epiclone_joint.py`
  result (n = 12) is uninterpretable and not used. The scRNA object needs `rds_raw.py` (a
  raw-tree reader; rdata's converter fails on it).
- **Parallel comparison, same design, same experiments, separate cells and clones:**

  | readout of multipotent cells | clones | CV r (null) | R2 / noise ceiling R2 | composition only |
  |---|---|---|---|---|
  | DNA methylation (m13) | 181 | **0.41** (-0.04) | 0.10 / 0.44 | 0.02 |
  | expression (scRNA) | 218 | **0.28** (-0.12) | 0.02 / 0.43 | 0.16 |

- **Reading (suggestive, not a test).** With matched design and similar noise ceilings,
  methylation predicts a clone's myeloid / erythroid bias better than expression (0.41 vs
  0.28), and unlike expression its signal is not carried by multipotent subtype composition.
  This fits track B's finding that ~55% of the clone bias is invisible in expression, and
  DARLIN's report that clonal memory tracks methylation. Caveats: different cells and clone
  sets, no CI on the difference, one lineage axis. A proper test needs both readouts per clone.

### Methylation vs expression: bootstrap (2026-11-02)

`bootstrap_meth_vs_rna.py`, 5000 clone resamples per analysis, independent clone sets:
methylation CV r 0.41 [0.30, 0.52]; expression 0.25 [0.13, 0.36] (this run's folds; 0.28 in the
earlier run: the RNA script iterates a Python set, so fold assignment varies between runs);
difference [0.01, 0.32], P(diff <= 0) = 0.02. The gap is unlikely to be clone-sampling noise,
but the CI excludes 0 only narrowly and ignores fold-assignment variance. Still suggestive.

### Same-cell multi-omic data located (2026-11-02)

- **DARLIN Camellia-seq** (Li 2023, GEO SuperSeries GSE222486): lineage barcode + DNA methylation
  + chromatin accessibility + RNA in the same single HSCs across AGM, fetal liver and adult bone
  marrow. Processed RNA GSE222246; methylation / accessibility GSE222328. The right data for a
  same-clone test of "the hidden clone bias is epigenetic". Sizes not yet checked.
- **EPI-Clone X.1** (Figshare 27991574, 10 MB): methylation + 120 targeted genes in the same human
  CD34+ cells, but clones are defined from methylation itself, so circular for this question.

## Track B4: same-cell methylation vs expression vs clone fate, DARLIN Camellia-seq (pre-registered 2026-11-02)

- **Data.** DARLIN (Li 2023) Zenodo archive (processed outputs, Figs 4-5) and, if needed, GEO
  GSE222246 (RNA) / GSE222328 (CpG methylation, GpC accessibility) / GSE222479 (lineage).
  Design details (which cells, which fate readout) are fixed after inspecting the files and
  before computing any clone-fate statistic; any change to this plan is logged first.
- **Question.** For HSCs with lineage barcode + methylation + expression in the same cell, does
  methylation predict the clone's fate bias better than expression, and does it add to it?
- **Analysis.** Clone fate bias from the clone's non-HSC / mature cells (as in the paper);
  state = per-cell methylation (CpG, binned) and expression (log-normalised) PCs, averaged over
  the clone's HSCs. 5-fold CV by clone, ridge; bootstrap CIs over clones.
- **Predictions.** (1) Methylation CV r > expression CV r, difference CI excluding 0.
  (2) Methylation + expression > expression alone (CV r gain >= 0.05).
  (3) Both above a permuted null.
- If clones with both readouts number < 50, the test is reported as underpowered and not read.

### Converged reruns R: result (2026-11-02)

| model, memory | healthy | within loss | G-M | M r_d4_d6 | M-only |
|---|---|---|---|---|---|
| E14 inherit / resample | 4, 8 | 0.97, 0.86 | 0.41, 0.41 | 0.99, 0.96 | 0.75, 0.75 |
| E16 inherit / resample | 5, 4 | 0.36, 0.49 | 0.27, 0.29 | 0.73, 0.70 | 0.79, 0.78 |
| B2 inherit / resample | 6, 7 | 0.46, 0.53 | 0.25, 0.28 | 0.83, 0.81 | 0.81, 0.79 |

- **P1 refuted:** 34/60 healthy (57% < 70%); still 26 not converged or b frozen at 250.
- **P2 holds:** within loss falls (B2 0.60 -> 0.46-0.53; E16 re-scored 0.34-0.61 -> 0.36-0.49).
- **P3 holds:** E16 / B2 median G-M < 0.40 (0.25-0.29).
- **P4 refuted:** E14 M-only 0.75 (< 0.80); longer training does not repair M-CSF.
- **E14's own gate (E14 pre-registration: G-M in band in >= 3/5 and loss <= 1.3 in >= 4/5 per
  variant) passes in both variants:** all 20 runs have G-M 0.33-0.59; loss <= 1.3 in 8/10
  (inherit) and 9/10 (resample); on healthy runs only, 4/4 and 7/8. Per the R pre-registration
  E14 gets Epo/SCF shard 0 of 2, one look.

#### E14 test predictions, recorded before the look (healthy runs, pooled Epo/SCF r)

| variant | healthy runs | prediction range | median | success band (range +/- 0.10) |
|---|---|---|---|---|
| inherit | 4 | 0.538-0.662 | 0.559 | 0.438-0.762 |
| resample | 8 | 0.441-0.608 | 0.541 | 0.341-0.708 |

Caveats stated before the look: the predictions are for the whole test, the look is shard 0
(half the clones), so the observed value is noisier; E14 fails the M-CSF prediction, so a hit
would support concordance-by-founder-bias, not the full model.

#### E14 shard-0 look: THE single look (2026-11-02, test_oracle, ledger row appended)

Observed pooled Epo/SCF r on shard 0 of 2: **0.44 [0.03, 0.78]**, 24 clones (agreement 0.71).

- **Pre-registered criterion met for both variants**: inside inherit's band (0.438-0.762, at its
  lower edge) and resample's band (0.341-0.708).
- **How much it means: little.** 24 clones give a CI spanning 0.03-0.78; E16 and B2 runs predict
  0.4-0.5 and would also have passed, so this test does not discriminate between our models.
  The prediction was not refuted; it was not strongly confirmed either.
- Shard 1 of 2 remains unseen (reserved; E16 / B2 did not pass their gates).
- Lesson: the Epo/SCF test is too thin (~48 clones in total) to separate models. Future tests
  need a power check (the CI width a model difference would have to beat) before reserving them.

### DARLIN Zenodo archive contents (2026-11-02)

10.8 GB, 6968 files: Fig. 4 (skull 10x scRNA + DARLIN lineage, CA/TA/RA loci, one time point)
and Fig. 5 (tissue-overlap tables) plus snakemake intermediates. **No Camellia-seq data**: the
same-cell methylation + RNA + lineage test (B4) needs GEO GSE222246 / GSE222328 / GSE222479,
fetched on GitHub Actions (NCBI blocks this machine). `geo_probe.py` learns the formats first.

### Camellia-seq formats and B4 feasibility (2026-11-02, geo_probe.py on Actions)

- GSE222246: 12 plate-level RNA UMI count tables; all cells annotated "cell type: HSC" (AGM,
  fetal liver, bone marrow). GSE222328: 1078 cells, per-cell CpG and GpC tables
  (chr, pos, met_reads, nonmet_reads, rate; GpC ~26 MB per cell). GSE222479: 244 samples, all
  bulk / single-cell CARLIN *raw reads only* (no processed files), mostly the migration study.
- **B4 is not feasible as pre-registered.** (1) Every Camellia cell is an HSC: there is no fate
  readout per clone, so "predict clone fate" cannot be computed; the paper's question was clonal
  memory (do sister HSCs share methylation / expression), not fate. (2) No processed lineage
  calls for Camellia cells are deposited: clone identity would have to be rebuilt from raw
  CARLIN reads (SRA) with the DARLIN snakemake pipeline. B4 is withdrawn before any statistic
  was computed. A feasible re-scoped question is "which readout carries clone identity", i.e.
  sister-HSC similarity in methylation vs expression, which needs (2).

## Track B5: which readout carries clone identity? Camellia-seq sisters (pre-registered 2026-11-02)

- **Data.** Clone IDs from DARLIN_tutorial `DARLIN_HSC_sample_sheet.tsv.gz` (`large_clone_id`):
  27 clones, 272 HSCs (BM 162, FL 96, AGM 14; 6 mice), all with CpG methylation (232 pass
  metQC) and RNA (244 pass rnaQC). Per-cell CpG (GSE222328) and plate RNA tables (GSE222246)
  fetched on GitHub Actions. Fate labels cover too few clones (Meg bias 10 vs 40 cells), so the
  withdrawn B4 fate question is not attempted.
- **Features.** Methylation: CpG rate in 100-kb bins, bins covered in >= 50% of cells, cell-mean
  imputed, top-20 PCs. Expression: log-normalised UMI, 2000 most variable genes, top-20 PCs.
  Cells passing both QCs only.
- **Statistic.** For every pair of cells from the same mouse and tissue, similarity = Pearson r of
  their PC vectors; AUC for separating sister pairs (same clone) from non-sister pairs. Bootstrap
  over clones (2000) for CIs, and a clone-label permutation null (within mouse and tissue).
- **Predictions.** (1) Methylation AUC > 0.5 beyond its permutation null. (2) Methylation AUC >
  expression AUC, difference CI excluding 0 (DARLIN reported clonal memory tracks methylation).
  (3) Expression AUC above its null (weaker signal still present).

### Track B5 result (2026-11-02; GitHub Actions, run after the gene-alignment fix)

182 HSCs (both QCs, methylation fetched), 27 clones, 215 sister pairs within mouse and tissue;
25,740 100-kb methylation bins, 9,103 genes common to all 10 RNA plates. (First run stopped by
its own guard at 24 matched cells, no statistic computed; fixed gene alignment across plates.)

| readout | sister vs non-sister AUC [95% CI, clone bootstrap] | permutation null mean / 95th pct |
|---|---|---|
| DNA methylation | **0.63** [0.57, 0.69] | 0.51 / 0.55 |
| expression | 0.51 [0.48, 0.58] | 0.48 / 0.51 |
| difference | **[0.01, 0.18]** | |

- **P1 holds:** methylation separates sisters from non-sisters beyond its null.
- **P2 holds:** methylation > expression, difference CI excludes 0.
- **P3 not established:** expression sits at its null's 95th percentile (0.51 vs 0.51).
- **Reading.** In the same single HSCs, clone identity is readable from DNA methylation and
  essentially not from expression. Together with track B (expression explains ~40% of LARRY
  clone fate, the rest invisible), B3 (EPI-Clone: methylation predicts clone bias better than
  expression, 0.41 vs 0.25) and DARLIN's own report, this is now three independent lines
  pointing to a heritable, epigenetically held clone state. Caveats: 27 clones; the
  within-clone signal is modest (AUC 0.63); a methylation difference between clones could also
  reflect shared clone age / proliferation history rather than a fate-instructive state.

## Track B5b: confound control and measurement ceiling (pre-registered 2026-11-02)

- **Control.** Before the methylation PCs, regress out of every bin (across cells) the cell's
  global CpG methylation rate, log CpG coverage and an expression proliferation score (mean
  log-expression of Mki67, Top2a, Ccnb1, Cdk1, Mcm2, Pcna, present genes only). Recompute AUC,
  CI and null as in B5.
- **Ceiling.** Keep each cell's coverage (reads per bin) and replace its methylation with a
  binomial draw from its clone's pooled bin rate (sisters identical up to sampling). Same
  pipeline; median AUC over 20 draws = what perfectly identical sisters would score at this
  depth. Report observed AUC as a fraction of the way from 0.5 to the ceiling.
- **Predictions.** (1) Controlled methylation AUC stays above its null 95th percentile.
  (2) Controlled AUC >= 0.58 (keeps >= 60% of the observed excess over 0.5). (3) Ceiling AUC
  well above observed (>= 0.75): sisters' true methylation is less than identical but the
  observed 0.63 understates the shared state because of sparse coverage.
- **Decision rule for the simulator.** Build it only if (1) holds; calibrate its clone-state
  drift to the ceiling-adjusted fraction.

### Track B5b result (2026-11-02)

| | AUC [95% CI] | null 95th pct |
|---|---|---|
| methylation, as in B5 | 0.63 | 0.55 |
| methylation, controlled (global met rate, CpG coverage, proliferation score from Mki67, Top2a, Cdk1, Mcm2, Pcna) | **0.62** [0.56, 0.71] | 0.55 |
| identical-sister ceiling at observed depth (20 draws) | **0.95** [0.94, 0.96] | |

- **P1 and P2 hold:** controlling for global methylation, coverage and proliferation leaves the
  sister signal unchanged (0.626 -> 0.623). It is not explained by shared age / division state
  as far as these covariates capture it.
- **P3 holds:** the ceiling is 0.95, so the observed signal is (0.63 - 0.5) / (0.95 - 0.5) =
  **~28% of the maximum**. Sparse coverage is *not* why the signal is modest: sisters' methylation
  genuinely diverges; only part of the profile is clone-shared. (The ceiling is an upper bound:
  pooled clone rates include the cell itself and carry sampling noise, so 28% is a lower bound.)
- **Decision (pre-registered rule): build the simulator.** Calibration target: a heritable clone
  state of which ~30% (lower bound) of the cell-to-cell methylation variance at the measured
  scale is clone-shared after several divisions, i.e. substantial drift per division.

## E18: drifting heritable clone-state model (pre-registered 2026-11-02)

- **Model.** Each clone has a founder state z ~ N(0,1). The state is inherited with drift: after
  g divisions two lineages' states correlate rho = (1 - d)^g. In LARRY, each day-2 sister branch
  b has z_b = sqrt(rho_L) z + sqrt(1 - rho_L) e_b; each mature cell is N with probability
  sigmoid(beta z_b + tau_c + sigma_w w), w ~ N(0,1) per cell; tau_c per condition. Simulated on
  the observed clones with their observed N+M counts per condition and day, so sampling depth
  is exact, and scored with the same statistics as the data.
- **Calibration of rho_L (not fitted to LARRY).** (a) Camellia: find rho_cam such that a toy with
  the B5 design (27 clones, observed clone sizes and mouse|tissue groups, 20-dim state, clone-
  shared variance fraction rho, measurement noise set so rho = 1 gives the 0.95 ceiling) gives
  AUC 0.63. (b) d from rho_cam = (1 - d)^g_cam with g_cam in {10, 20, 40} divisions separating
  Camellia sisters (E-stage barcoding to adult HSC). (c) rho_L = (1 - d)^g_L with g_L in {1, 3}
  (LARRY sisters' common ancestor around day 1-2, fate state set by ~day 2-3). Predictions are
  reported over this 3 x 2 grid.
- **Fit (fit set only).** beta, sigma_w and tau_c (5 conditions) by least squares to the
  within-condition statistics (clone N-share mean and across-clone variance, mixed-clone
  fraction, per condition), cytokine data, fit-set clones (all clones outside Epo/SCF shard 1).
- **Predictions.** (P1) Consistency (spent data, not a test): full-G, full-M, G-M cross r inside
  the observed CIs for at least one grid cell. (P2) The held-out test: pooled Epo/SCF r on shard 1
  of 2 inside the grid's predicted range +/- 0.10. Power check before the look: if the shard's
  expected CI half-width exceeds the spread between the grid's extreme predictions, the look is
  skipped as uninformative (lesson from E14).

### E18 result (2026-11-02)

Calibration: rho_cam = 0.21 (toy noise 1.0 reproduces the 0.95 ceiling; 209 cells passing both
QCs, vs 182 analysed in B5 - small design mismatch, noted). Fit by Nelder-Mead on the fit set
(Epo/SCF shard-1 clones excluded); held-out clones' N+M depth (not outcomes) used to simulate.

| g_cam | g_L | rho_L | fit loss | beta | sigma_w | full-G (obs 0.69 [0.40, 0.92]) | full-M (0.24 [-0.11, 0.55]) | G-M (0.51 [0.33, 0.65]) | pred. Epo/SCF shard 1 (mean, 95%) |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 1 | 0.86 | 0.11 | 6.9 | 0.13 | 0.70 | 0.47 | **0.42** | 0.52 [0.31, 0.69] |
| 10 | 3 | 0.63 | 0.14 | 4.7 | 0.37 | 0.48 | 0.24 | 0.27 | 0.32 [0.05, 0.53] |
| 20 | 1 | 0.93 | 0.67* | 1.7 | 0.95 | 0.47 | 0.43 | 0.40 | 0.33 [0.06, 0.60] |
| 20 | 3 | 0.79 | 0.08 | 7.1 | 0.51 | 0.61 | 0.30 | 0.24 | 0.45 [0.19, 0.67] |
| 40 | 1 | 0.96 | 0.71* | 1.6 | 1.04 | 0.46 | 0.40 | 0.38 | 0.29 [-0.02, 0.59] |
| 40 | 3 | 0.89 | 0.47* | 1.9 | 0.12 | 0.51 | 0.42 | 0.34 | 0.33 [0.08, 0.56] |

\* poorly converged (noisy objective; Nelder-Mead stuck) - not interpreted.

- **P1 (consistency, spent data) holds:** cell (10, 1) puts all three cross statistics inside the
  observed CIs, including G-M 0.42, the highest G-M of any well-fitting model in this programme
  (E11-E17 well-fitting runs: 0.23-0.34). It needs strong, nearly deterministic clone bias
  (beta 6.9, tiny per-cell noise) and highly shared sister state (rho_L 0.86).
- **P2: look skipped (pre-registered power rule).** Shard 1 has ~24 clones; shard 0's CI half-width
  was ~0.37, larger than the spread of the grid's predicted means (0.29-0.52). The look could
  not discriminate; shard 1 stays unseen.
- **Reading.** A Camellia-calibrated, drifting heritable state reproduces the LARRY cross-
  condition concordance when drift is fast in HSC time but sisters in culture share most of their
  state, with fate set almost deterministically by that state. The division-count assumptions
  dominate the prediction; constraining them (division counts / clone ages) is now the limiting
  input, not the model.

### Division counts: literature and data check (2026-11-02)

- Fetal liver HSCs: essentially all cycle, transit ~10-14 h (Nygren et al., JCI; JCB review).
  Adult mouse HSCs divide rarely (H2B-GFP label retention: Bernitz 2016 Cell; revised upward
  by Morcos 2020 JEM). LARRY culture timing not stated in what I could access; murine progenitors
  in cytokines take ~36-40 h to the first division and ~12 h per later division (1997 Blood).
- DARLIN Camellia induction / sampling stages are not in GEO or accessible full text. Constraint
  from the design: AGM HSCs carry barcodes, so induction was <= ~E10.5.
- Note: `large_clone_id` labels are reused across mice (clone_0 in several mice). B5 compared
  pairs only within mouse|tissue, so sister calls were correct, but "27 clones" was 27 labels:
  there are 91 (mouse, clone) clones (BM 53, FL 34, AGM 4). B5's clone bootstrap pooled
  same-label clones across mice (CI slightly conservative-to-misstated; rerun below fixes it).
- Note: GEO text calls CpG "accessibility" and GpC "methylation"; the sample sheet (met_rate ~0.78
  on ~1.4 M CpG sites, acc_rate ~0.26 on GpC) confirms CpG = endogenous methylation as used.

## E18b: per-tissue drift test (pre-registered 2026-11-02)

- **Idea.** If the clone state drifts at a constant rate per division, sisters separated by
  fewer divisions share more state. Camellia FL sisters (induction <= E10.5 to FL sampling
  ~E14.5) are separated by fewer divisions than BM sisters (adult).
- **Measure.** Per-tissue sister AUC (BM; FL) with (mouse, clone) clone keys, clone bootstrap,
  within-mouse permutation null; per-tissue toy calibration (same design, same 0.95-ceiling noise)
  -> rho_FL, rho_BM.
- **Division assumptions (from the literature above).** g_FL in [8, 11] (~4 days at 10-14 h);
  g_BM in [20, 40] (FL expansion + perinatal cycling + few adult divisions).
- **Predictions.** (1) rho_FL > rho_BM. (2) Constant drift: d from FL (rho_FL^(1/g_FL)) predicts
  rho_BM = (1 - d)^g_BM; the measured rho_BM lies inside the predicted range (over the g ranges)
  or its CI overlaps it. (3) E18 refit with d from FL (better constrained than the 3 x 2 grid),
  g_L in {1, 3}, Nelder-Mead with 3 restarts and 5 common-random-number simulations per loss.
  The Epo/SCF shard-1 look stays gated by the power rule.

### E18b result: per-tissue drift test (2026-11-02)

B5 rerun with (mouse, clone) keys: global methylation AUC 0.63 [0.57, 0.68], expression 0.51,
difference [0.01, 0.20], controlled 0.62, ceiling 0.96 - B5/B5b conclusions unchanged.

| tissue | cells | clones | sister AUC [95% CI] | null 95th | rho (toy calibration) [CI] |
|---|---|---|---|---|---|
| bone marrow | 115 | 44 | 0.62 [0.54, 0.72] | 0.55 | 0.21 [0.08, 0.39] |
| fetal liver | 58 | 34 | 0.61 [0.54, 0.69] | 0.58 | 0.18 [0.06, 0.33] |

- **P1 refuted:** rho_FL is not larger than rho_BM (0.18 vs 0.21; CIs overlap fully).
- **P2 refuted:** constant drift calibrated on FL (d = 0.14-0.19 per division over g_FL 8-11)
  predicts rho_BM = 0.0002-0.014 after 20-40 divisions; measured 0.21 [0.08, 0.39]. Only the
  extreme corner (FL CI upper end, g_FL 11, g_BM 20 -> 0.13) touches the BM CI.
- **P3 not run (logged deviation):** the refit was to use d from FL under the constant-drift
  model that P2 just rejected; running it would calibrate a falsified mechanism.
- **Reading.** The clone-shared methylation component does not decay with divisions between
  mid-gestation and adulthood: it is about the same (~20% of the measured variation) in FL and
  BM. That fits an early-established, then stable clone component (a plateau), not a constant
  per-division drift. E18's grid (which assumed constant drift) is therefore the wrong form;
  its one well-fitting cell needed LARRY sisters to share ~86% of the fate state, which a stable
  ~20% methylation component does not deliver unless the fate-relevant part of the state is
  much more clone-shared than the genome-wide methylation proxy. That tension is now the open
  question.

## Track B6: where the clone signal lives, Meg fate, per mouse (pre-registered 2026-11-02)

Same 182 Camellia HSCs and (mouse, clone) keys as B5c; GitHub Actions; Ensembl GRCm38.102 genes.

- **B6a, genome location.** Feature sets (per-cell CpG rate per region; regions covered in >= 50%
  of cells; PCs = min(20, regions - 1); sister AUC as in B5):
  (i) lineage windows: TSS +/- 50 kb of 40 haematopoietic lineage / HSC genes, in 10-kb sub-bins;
  (ii) all promoters, TSS +/- 2 kb; (iii) gene bodies; (iv) intergenic 100-kb bins.
  Null for (i): 50 random gene sets of the same size, same windows -> AUC distribution.
  **Prediction:** (1) lineage-window AUC above the random-set 95th percentile and above the
  genome-wide 0.63. If it holds, the fate-relevant part of the clone state is more clone-shared
  than the genome-wide average (resolving the E18b tension); if not, the clone signal is diffuse.
- **B6b, Meg fate.** Clones with Meg-bias labels (cells labelled Meg / nonMeg; clone label =
  majority). Clone-mean methylation PCs vs expression PCs, leave-one-clone-out ridge, AUC for
  Meg vs nonMeg clones. Underpowered by design (expected < 15 clones): direction only, no claim.
- **B6c, leave one mouse out.** B5 methylation and expression AUC with each mouse removed.
  **Prediction:** (2) methylation AUC above its null in every leave-one-out set.

### Track B6 result (2026-11-02; two crashed runs before this, no statistics computed in them)

182 cells, 82 (mouse, clone) clones.

| feature set | regions | sister AUC [95% CI] | null 95th |
|---|---|---|---|
| lineage windows (40 genes, TSS +/- 50 kb) | 392 | 0.50 [0.47, 0.53] | 0.54 |
| 50 random gene sets, same windows | ~400 each | median 0.52, 95th pct 0.54 | |
| promoters (TSS +/- 2 kb) | 16,710 | 0.52 [0.47, 0.58] | 0.54 |
| gene bodies | 18,743 | 0.53 [0.48, 0.57] | 0.56 |
| genome-wide 100-kb bins (B5 reference) | 25,704 | **0.63** [0.56, 0.68] | 0.55 |

- **B6a P1 refuted, and the opposite of the hoped-for result:** the clone signal is absent from
  lineage-gene windows, promoters and gene bodies; it exists only in the genome-wide 100-kb bins.
  It is a large-scale, mostly non-genic feature, not a local state at fate genes.
- **B6c P2 holds:** leave-one-mouse-out methylation AUC 0.59-0.63, above its null in all five sets
  (LL719 not separately listed: its cells sit in the others' sets). Expression stays at its null.
- **B6b (direction only):** 11 clones (5 Meg-biased): methylation AUC 0.60, expression 0.70.
  Opposite to the epigenetic-instruction expectation; uninterpretable at n = 11.
- **New confound to rule out before any interpretation: plate.** Sisters are often on the same
  plate (134 same-plate vs 117 cross-plate sister pairs); a plate-level, genome-wide methylation
  batch effect would produce exactly a large-scale, non-genic "clone" signal that is equal in FL
  and BM (E18b) and absent at genes. B5-B6 compared pairs within mouse|tissue, not within plate.

## Track B6d: plate control (pre-registered 2026-11-02)

- Same B5 pipeline (100-kb bins). Sister AUC computed (a) only over pairs on the same plate,
  (b) only over pairs on different plates (sisters and non-sisters alike), each with a clone
  permutation null within the same pair set and a clone bootstrap.
- **Prediction (if the signal is clonal):** cross-plate AUC above its null and within 0.05 of the
  same-plate AUC. **If cross-plate AUC is at its null, the B5-B6 methylation "clone" signal is a
  plate batch effect and the epigenetic-clone-state reading from Camellia is withdrawn.**

### Track B6d result: plate control (2026-11-02)

| readout | same-plate pairs: AUC [CI] (null 95th) | cross-plate pairs: AUC [CI] (null 95th) |
|---|---|---|
| methylation, 100-kb bins | 0.62 [0.55, 0.69] (0.56), 111 sister pairs | **0.63 [0.56, 0.69] (0.58)**, 104 sister pairs |
| expression | 0.52 (0.54) | 0.49 (0.51) |

- **Prediction holds:** the cross-plate AUC is above its null and equal to the same-plate AUC.
  The methylation clone signal is **not a plate batch effect**; the B5 / B5b / E18b readings stand.
- **Revised reading (B6a + B6d).** Clone identity is carried by large-scale, non-genic methylation
  (100-kb bins), absent at promoters, gene bodies and lineage genes, equal in FL and BM, and
  readable across plates. This looks like a heritable, largely neutral epigenetic *record* of
  ancestry (the same kind of signal EPI-Clone uses as a lineage barcode), not a fate-instructive
  state at fate genes. So Camellia shows that a clone-level heritable state exists and is
  invisible in expression; it does not show that this state steers fate. The fate link still
  rests only on EPI-Clone (methylation of multipotent cells predicts myeloid / erythroid bias,
  0.41 vs 0.25), where the predictive regions were not localised.

## Track B3e: which EPI-Clone CpGs carry the fate prediction? (pre-registered 2026-11-02)

- EPI-Clone's own panel annotation (github veltenlab/EPI-clone, infos/cpg_selection.csv): 110
  "Static" CpGs (clone-informative, not cell-state-associated), 200 "Dynamic" (cell-state
  associated), 143 unclassified. Same analysis as B3 (m13: myeloid share of committed
  progenitors from the methylation of the clone's multipotent cells, 5-fold CV ridge,
  composition baseline, clone bootstrap), run separately on Static-only and Dynamic-only CpGs.
- **Readings fixed in advance.** Static >> Dynamic: the heritable record itself carries fate
  information (clone record and fate bias co-inherited). Dynamic >> Static: the prediction comes
  from cell-state methylation of the multipotent cells (closer to "state predicts fate", like
  expression). Both: mixed. Caveat stated now: the Static/Dynamic labels were derived by the
  EPI-Clone authors from these data (clone association / cell-state association), independent of
  our fate target.

### B3e result (2026-11-02)

LARRY mice of M.1-M.3, 181 clones (as B3). CpG counts: 110 Static, 200 Dynamic, 143 NA.

| CpG set | CV r (95% clone bootstrap) | + composition |
|---|---|---|
| all 453 | 0.41 [0.30, 0.52] | 0.39 |
| Static (110) | 0.24 [0.09, 0.36] | 0.25 |
| Dynamic (200) | 0.39 [0.25, 0.52] | 0.40 |

Static minus Dynamic: [-0.32, 0.01].

- **Reading (pre-registered terms): mixed, leaning Dynamic.** Dynamic CpGs alone recover almost
  the whole prediction (0.39 vs 0.41). The difference CI only just reaches zero, so this is not a
  clean "Dynamic >> Static". Static CpGs predict fate above zero on their own (CI excludes 0), so
  the heritable record carries some fate information, but less.
- **Caveat (stated after the look, not tested):** Static has about half as many CpGs as Dynamic.
  Some of the gap could be panel size. A size-matched check (110 random Dynamic CpGs, repeated)
  is the obvious follow-up.
- **Consequence for the model:** fate prediction in EPI-Clone mostly comes from cell-state
  methylation of the multipotent cells. This fits the Camellia result (B6a: the clone record is
  large-scale and not at lineage genes). The record and the fate state should be modelled as
  separate variables, with only a weak coupling between them.

## E18c: neutral record plus separate fate state (pre-registered 2026-11-02)

- **Why.** B6a/B6d say the Camellia clone signal is a large-scale, neutral ancestry record. B3e
  says fate prediction comes mostly from cell-state CpGs. E18 tied LARRY's sister fate sharing
  (rho_L) to the Camellia record through a drift rate, so it assumed record = fate state. That
  link is now unsupported.
- **Model.** Two variables per clone. Record r: stable, heritable, does not affect fate.
  Camellia fixes its sister correlation at rho_cam 0.21 with no drift (E18b rejected constant
  drift). Fate state f: unmeasured; sisters share it with correlation rho_f. LARRY counts depend
  only on f (same simulator as E18 with rho_L = rho_f). So Camellia no longer constrains the
  LARRY fit. rho_f is profiled over {0.5, 0.6, 0.7, 0.8, 0.9, 0.95}. beta, sigma_w and tau_c are
  fitted as in E18 (fit set only; 2 restarts, 3 simulations each, common random numbers).
- **Readings fixed in advance.** (R1) The rho_f range whose cross statistics (full-G, full-M,
  G-M) all fall inside the observed CIs. If it is narrow, LARRY alone pins the fate-state
  sharing. If it is wide or empty, the model needs outside information on the fate state.
  (R2) Compare with E18 (10, 1), the only cell that passed (rho_L 0.86). If the passing range
  includes 0.86 without the Camellia link, then E18's fit never needed Camellia, and the record
  calibration was decorative. (R3) The record-fate coupling stays a stated, unfitted quantity.
  Its only measure is B3e Static r 0.24.
- No held-out look (shard 1 stays unseen; power rule as E18).

### E18c result (2026-11-02)

Fit set only, 2 restarts x 3 simulations, 20 simulations for the cross statistics (mean, sd).
Observed: full-G 0.69 [0.40, 0.92], full-M 0.24 [-0.11, 0.55], G-M 0.51 [0.33, 0.65].

| rho_f | loss | beta | sigma_w | full-G | full-M | G-M | all three in CI |
|---|---|---|---|---|---|---|---|
| 0.50 | 0.148 | 6.2 | 0.12 | 0.37 (0.15) | 0.34 (0.11) | 0.31 (0.10) | no |
| 0.60 | 0.138 | 5.0 | 0.30 | 0.45 (0.16) | 0.39 (0.11) | 0.37 (0.08) | yes |
| 0.70 | 0.105 | 5.9 | 0.45 | 0.56 (0.13) | 0.27 (0.07) | 0.28 (0.06) | no |
| 0.80 | 0.097 | 5.0 | 0.16 | 0.63 (0.13) | 0.45 (0.07) | 0.39 (0.06) | yes |
| 0.90 | 0.092 | 5.5 | 0.16 | 0.73 (0.09) | 0.35 (0.05) | 0.33 (0.06) | at the G-M edge |
| 0.95 | 0.070 | 6.6 | 0.03 | 0.81 (0.07) | 0.37 (0.05) | 0.33 (0.06) | at the G-M edge |

- **R1: wide.** Cells from 0.6 to 0.95 pass or sit on the edge. 0.7 fails, which is not
  monotone: a sign of fit noise (sigma_w jumps to 0.45), not structure. LARRY alone does not pin
  how much fate state sisters share. Fit loss improves steadily with rho_f (best 0.95), so the
  within-condition data mildly favour high sharing.
- **R2: E18's Camellia link was decorative.** The passing range covers E18's rho_L 0.86 without
  any Camellia input. E18's best G-M (0.42) is in line with these fits (0.33-0.39, sd ~0.06)
  given simulation noise. The Camellia calibration did not add predictive information.
- **R3:** record-fate coupling not fitted. Its only measurement stays B3e Static r 0.24.
- **Reading.** The simulator can be run on a two-variable picture: a neutral record (Camellia,
  stable, rho 0.21) and a fate state shared by culture sisters at rho_f ~0.6-0.95, with
  near-deterministic fate given that state (beta ~5-6). The next constraint has to come from data
  on the fate state itself (e.g. EPI-Clone Dynamic CpGs in sisters), not from the record.
  G-M still tops out near the bottom of its CI (max 0.39 of 0.51 observed).

## Track B3f / B3g: size-matched sites and sister fate-state sharing (pre-registered 2026-11-02)

- **B3f (size-matched B3e).** Same as B3e, but Dynamic is cut to 110 CpGs (the Static count):
  200 random draws of 110 Dynamic CpGs, CV r for each. Reading fixed in advance: if Static's r
  (0.24) is below the 5th percentile of the size-matched Dynamic draws, the B3e gap is not panel
  size and cell-state CpGs really carry more fate information. If it is within the middle 90%,
  the B3e gap is panel size, and Static and Dynamic carry similar fate information per site.
- **B3g (sister fate-state sharing).** EPI-Clone M.1-M.3 LARRY cells, multipotent compartment
  (HSC/MPP1-4), clone-labelled. Pairs of cells within the same experiment, mouse batch and cell
  type (so cell-type identity cannot drive similarity); sisters = same LARRY barcode. Feature:
  top-20 PCs of binary methylation, Pearson r between cells. AUC sisters vs non-sisters for
  (a) Dynamic CpGs, (b) Static CpGs, (c) Dynamic size-matched to 110 (20 draws, median), and
  (d) negative |pseudotime difference|. Null: sister labels permuted within group (100x). CI:
  bootstrap over clones (300x).
- **Readings fixed in advance.** Dynamic (size-matched) AUC above its null with a CI excluding
  it: sisters share cell-state methylation, so the fate state is measurable in sisters and
  could give E18c a rho_f. Static > Dynamic: as expected for a record (sanity check). Dynamic at
  null: the fate state is not visibly shared at the methylation level within a cell type, and the
  B3 prediction comes from something set later or from composition within types.
- Caveat stated now: in vivo transplant HSC/MPPs, not culture sisters, so any rho estimate here
  is only a rough guide for LARRY in vitro.

### B3f / B3g results (2026-11-02)

First run crashed in B3g on an empty bootstrap (boot=0 in the size-matched draws; code bug, no
result read). Fixed and rerun; B3f is deterministic and gave identical numbers.
Implementation detail not in the pre-registration: B3g keeps every cell in a clone with >= 2
cells in its group, plus a random 30% of other cells as non-sister background.

**B3f.** Static r 0.24 sits at the **22nd percentile** of 200 size-matched Dynamic draws
(median 0.29, 5-95%: 0.15-0.40). **Reading: the B3e gap is panel size.** Per site, Static and
Dynamic CpGs carry similar fate information. The B3e reading "mostly cell state" is withdrawn.

**B3g.** 2,420 multipotent cells, 357 clone x group sets with sisters, 233k same-group pairs
(14.6k sister pairs).

| Feature | sister AUC (95% clone bootstrap) | null |
|---|---|---|
| Static CpGs (110) | 0.93 [0.92, 0.95] | 0.51 |
| Dynamic CpGs (200) | 0.70 [0.66, 0.74] | 0.51 |
| Dynamic, size-matched 110 (20 draws) | 0.66 (0.63-0.69) | 0.51 |
| pseudotime closeness | 0.49 [0.46, 0.51] | 0.50 |

- **Sanity check passes:** Static > Dynamic, as expected for the sites chosen as the record.
- **Dynamic is well above null.** By the pre-registered reading, sisters share cell-state-class
  methylation within a cell type.
- **But pseudotime is not shared at all.** Sisters are no closer in differentiation position than
  random same-type cells. So the Dynamic-site sharing is not "same point along the
  differentiation path". Two explanations remain and this test cannot tell them apart:
  (i) a shared fate state that is orthogonal to pseudotime; (ii) the clone record also
  marks the "Dynamic" sites (their label comes from association with cell state, which does not
  exclude clonal association).
- **Combined reading (B3e-B3g):** EPI-Clone's site classes do not cleanly separate "record" from
  "state". Fate information is spread across both classes at similar per-site rates. The
  two-variable picture (neutral record + separate fate state) is not refuted, but is no longer
  supported by B3e either. No rho_f for E18c can be read off yet.
- **Follow-up that would separate (i) from (ii):** does Dynamic-site sister similarity survive
  after conditioning on Static-site similarity (partial AUC / residualised similarity)? Also,
  does the Dynamic-site sister similarity predict sister fate agreement?

## Track B3h: shared fate state or record leakage at Dynamic sites? (pre-registered 2026-11-02)

Same cells, pairs and features as B3g / B3 (EPI-Clone M.1-M.3 LARRY, multipotent cells).
- **(a) Sister sharing net of the record.** Bin same-group pairs into 10 deciles of Static-site
  similarity. Within each decile, AUC of Dynamic-site similarity, sisters vs non-sisters;
  report the sister-weighted mean, a clone bootstrap CI (300x) and a within-decile permutation
  null (100x). Size-matched Dynamic (110) as well (20 draws, median).
- **(b) Fate information net of the other class.** B3 design (181 clones, myeloid share). Clone
  features = mean top-20 PCs from each class. Residualise one class's clone features on the
  other's (linear, within each CV training fold, applied to the test fold), then CV ridge as B3.
  Report CV r and clone-bootstrap CI for Dynamic | Static and Static | Dynamic, and r for both
  classes together.
- **Readings fixed in advance.** (a) mean AUC clearly above its null (CI excluding it) and
  (b) Dynamic | Static r with CI excluding 0: Dynamic sites carry a shared, fate-relevant state
  beyond the record, favouring explanation (i) (separate fate state). (a) near null and/or (b)
  Dynamic | Static ~0: the Dynamic sharing is record leakage (ii); one heritable record carries
  both identity and fate information. Mixed: (a) above null but (b) ~0, i.e. shared beyond the
  record but not fate-relevant.
- Caveat stated now: Static similarity measures the record imperfectly (AUC 0.93, not 1), so (a)
  can leave some record signal in the residual. It is biased towards (i); a weak (a) is more
  informative than a strong one.
- **Amendment before the look (2026-11-02).** The smoke test on toys showed the decile
  stratification is biased: with pure leakage and a noisy record measure, the stratified AUC is
  0.58-0.65 (10 bins) and stays ~0.6 at 50-100 bins, while the permutation null is ~0.5. So
  "above null" cannot separate (i) from (ii), and the same holds for residualising in (b).
  Replacement readout: a **record-only benchmark from the data.** Split the 110 Static CpGs at
  random into halves A and B (55 each; 20 splits). Benchmark = A's result conditioned on B, for
  (a) stratified sister AUC (50 bins) and (b) fate CV r of A | B. Test = 55 random Dynamic
  CpGs conditioned on the same B. **Readings:** Dynamic clearly above the benchmark (median of
  draws above the benchmark's 95th percentile) = Dynamic sites carry a shared or fate-relevant
  signal beyond the record (i). At or below the benchmark = leakage alone is enough to explain
  it, (ii) not excluded. This is asymmetric: only "above" is informative. The original (a)/(b)
  numbers are reported as well, as descriptive.

### B3h result (2026-11-02)

Same 181 fate clones; 236k same-group pairs (14.6k sister pairs). 20 random Static half-splits.

| | Dynamic test (55 CpGs \| Static half B) | Record-only benchmark (Static half A \| half B) |
|---|---|---|
| (a) sister AUC, 50-bin stratified | 0.59 (5-95%: 0.56-0.61) | 0.84 (0.82-0.87) |
| (b) fate CV r, residualised | 0.16 (0.06-0.23) | 0.06 (-0.01-0.22) |

Descriptive (original design, full panels): (a) Dynamic | Static deciles 0.61 [0.55, 0.63],
permutation null 0.50; size-matched 0.59. (b) Dynamic | Static r 0.22 [0.06, 0.38];
Static | Dynamic 0.01 [-0.15, 0.17]; both classes together 0.35 [0.21, 0.48].

- **(a) Leakage is enough.** Dynamic-site sister sharing, net of the record, is far *below* the
  record-only benchmark. By the amended rule, sharing beyond the record is not shown. The B3g
  sister sharing at Dynamic sites can be explained by the clone record marking those sites.
- **(b) Not clearly above the benchmark** (Dynamic median 0.16 < benchmark p95 0.22), so by the
  pre-registered rule, fate information beyond the record is not shown either. Leaning that way:
  the median is above the benchmark median, and descriptively the asymmetry is sharp (Dynamic
  keeps 0.22 after removing Static; Static keeps 0.01 after removing Dynamic). Dynamic sites
  hold everything the Static sites know about fate, plus possibly more. Not a test result.
- **Reading.** No evidence for a sister-shared fate state separate from the record. The
  simplest account consistent with B3e-B3h: one heritable record, readable at both site
  classes, carrying the clone-shared part, plus possibly some fate information at cell-state
  sites that sisters do not share. E18c still has no measured rho_f.
- **Tension to keep open.** Camellia said the record looks neutral (not at lineage genes; no
  Meg advantage, 11 clones). EPI-Clone says record-marked sites predict fate (B3, B3f). Different
  panels (genome-wide bins vs a 453-CpG targeted panel chosen for clonal and state information),
  different compartments and fate targets. Not resolved.

## Track B3i: is "methylation predicts fate" a clone-size or mouse confound? (pre-registered 2026-11-02)

Motivated by the literature sweep (LITERATURE_SWEEP_2026-11.md): EPI-Clone finds expanded clones
by their density in static-CpG space, myeloid bias concentrates in expanded clones, and B3's CV
folds were random by clone, not by mouse.
- Same B3 design (181 clones, myeloid share, clone-mean top-20 PCs of the multipotent cells),
  for all / Static / Dynamic CpGs. Mouse = the clone's Experiment|ProcessingBatch (modal).
  Size = log total cells of the clone; also log multipotent cells.
- (1) **Leave-one-mouse-out CV** in place of random folds. (2) **Partial r:** CV prediction vs
  target after regressing both on mouse dummies + the two size terms. (3) Baseline: mouse + size
  alone, CV r (random folds). Clone bootstrap CIs (2000x).
- **Readings fixed in advance.** If Static r (and/or all) falls to a CI including 0 under (1) or
  (2), the B3/B3f "record predicts fate" result was a confound (mouse or clone size), and the
  Camellia vs EPI-Clone tension dissolves: the record is neutral in both. If it survives both with
  CI excluding 0, the record carries fate information beyond size and mouse (tension stands).
  Same reading for Dynamic.

## E19: kin-correlation analysis of fate-state sharing in LARRY in vitro (pre-registered 2026-11-02)

Adopting the kin-correlation idea (Hormoz et al. 2015/2016): the fate-state sharing E18c could
not pin is measurable from relatives at two kinship distances.
- Data: LARRY main in vitro (data/larry), mature Neutrophil and Monocyte cells at days 4 and 6,
  clones with N+M cells in at least two wells. Kinship classes for pairs of a clone's cells:
  **same well, same day** (common ancestor after the day-2 split) and **different wells**
  (common ancestor before the split). For each class, a 2x2 N/M pair table, each clone weighted
  equally (its pair table normalised), and the latent (tetrachoric) correlation r_class with
  thresholds from the pooled N fraction. Clone bootstrap (1000x).
- **Link to the model.** In the E18c simulator, the latent correlation is rho_f * K across
  sister branches and K within a branch (K = beta^2 / total latent variance). So
  **rho_f_hat = r_cross / r_same** is a data estimate of E18c's rho_f, independent of K.
- **Readings fixed in advance.** (R1) rho_f_hat with CI. If it falls inside 0.6-0.95, E18c's
  passing range is confirmed and narrowed by data; outside, E18c is in conflict with LARRY's kin
  structure. (R2) If the CI includes 1, no detectable loss of fate state between the split and
  commitment (state fixed by day 2). (R3) As a check, the same ratio computed on the cytokine
  dataset is not possible (one well per condition), so no cross-dataset check.
- Caveat stated now: main in vitro culture differs from the cytokine-condition experiment E18c was
  fitted on; fate is read at commitment, not at sampling, so no per-generation decay rate is
  claimed.

### B3i result (2026-11-02)

181 clones across 6 mice (27-33 each).

| CpGs | random folds (B3) | leave one mouse out | partial r, mouse + size removed |
|---|---|---|---|
| all | 0.41 [0.30, 0.52] | 0.24 [0.11, 0.37] | 0.23 [0.11, 0.35] |
| Static | 0.24 [0.11, 0.36] | 0.10 [-0.04, 0.23] | 0.01 [-0.12, 0.15] |
| Dynamic | 0.39 [0.25, 0.52] | 0.24 [0.08, 0.39] | 0.25 [0.12, 0.38] |

Baseline, mouse + clone size alone: 0.42 [0.29, 0.53].
- **Static: confound.** Its fate prediction disappears under both controls (CIs include 0). By the
  pre-registered reading, the "record predicts fate" result was mouse and clone size. **The
  Camellia vs EPI-Clone tension dissolves: the heritable record is neutral in both datasets.**
- **Dynamic survives both** (0.24-0.25, CIs exclude 0). Cell-state methylation of a clone's
  multipotent cells carries fate information beyond mouse and clone size. About 40% of the B3
  effect was confound.
- Mouse + size alone predict as well as methylation did in B3 (0.42). Every earlier B3-family
  number (B3, B3e, B3f, B3h-b) is inflated by this and should be read with B3i beside it.

### E19 result (2026-11-02)

Synthetic check before the look: latent model with rho 0.70, K 0.60 recovered as rho_f_hat 0.697,
K ~0.59. LARRY main in vitro, 464 clones with N/M cells in both wells.

| | latent r (95% clone bootstrap) |
|---|---|
| same well, same day | 0.94 [0.92, 0.96] |
| different wells | 0.83 [0.77, 0.87] |
| **rho_f_hat = ratio** | **0.88 [0.82, 0.92]** |

Secondary (cross-well pairs restricted to the same day, 442 clones): 0.87 [0.82, 0.92].
- **R1: confirmed and narrowed.** rho_f_hat lies inside E18c's passing range (0.6-0.95) and cuts
  it to ~0.82-0.92. E18's original cell (rho_L 0.86) is in it.
- **R2: CI excludes 1.** Sister branches lose a detectable but small part of their fate state
  between the day-2 split and commitment.
- **Cross-check, not pre-registered:** the same-well latent r (0.94) is the simulator's K =
  beta^2 / total latent variance. E18c's fits (beta ~5-6, sigma_w small) imply K ~0.88-0.92.
  The data and the fitted model agree that fate is near-deterministic given the branch state.
- **Reading.** LARRY's own kin structure pins the fate-state sharing the model needed. With it,
  E18c predicts G-M ~0.33-0.39 (its 0.8 and 0.9 cells), still near the bottom of the observed
  CI. The remaining G-M gap is not a missing-parameter problem.
- Caveat as pre-registered: main in vitro culture, not the cytokine experiment E18c was fitted on.

## Batch pre-registration 2026-11-03: E20, B1c, E19b, B3j, E21

### E20: is the G-CSF vs M-CSF gap real, with rho_f fixed by data?
- Fit beta, sigma_w, tau_c on the fit set exactly as E18c, with rho_f fixed at E19's 0.88
  (3 restarts x 3 simulations). Then 200 simulations on all clones at the observed depth, scored
  with the same statistics (full-G, full-M, G-M).
- **Readings.** Tail probability p = share of simulations with G-M >= 0.51 (observed point).
  p < 0.05: the gap is real, the model lacks something (triggers E21). p >= 0.05: the observed
  G-M is within the model's sampling spread at this depth; the cross-condition "limit" was a
  sampling ceiling, and that modelling question is closed. Same tail calculation reported for
  full-G and full-M (two-sided).

### B1c: day-2 expression -> fate, with confounds controlled (re-check of track B1)
- B1 design (clones with day-2 cells and >= 3 day-4/6 N+M cells; top-50 expression PCs, ridge).
  Covariates: starting population (LK vs LSK), the clone's modal day-2 library (7 libraries),
  log late N+M cells, log day-2 cells. (1) Leave-one-day-2-library-out CV. (2) Partial r with the
  covariates removed from prediction and target. (3) Covariates alone (random folds).
- **Readings.** B1's r 0.66 holds (CI of (1) and (2) above 0.5) = not confounded. Falls well below
  (CI upper end < 0.5) = largely confound; all downstream uses of "day-2 state explains ~40%" are
  restated.

### E19b: kin correlations in vivo (environment vs clone)
- LARRY in vivo (data/larry_invivo), mature Neu and Mono, days 9 and 16; groups = recipient mouse.
  Same E19 method: latent r within the same mouse and time point vs across mice; ratio with clone
  bootstrap (1000x).
- **Readings.** Ratio well below E19's in vitro 0.88 (CI upper end < 0.82) = the recipient
  environment breaks fate-state sharing beyond what culture branches show (matches E6b). Ratio
  overlapping 0.82-0.92 = the in vivo cross-mouse drop (E6b) is not in the latent fate state but
  elsewhere (e.g. sampling or cell-type mix).

### B3j: which cell-state CpGs carry the fate signal?
- B3 / B3i design (181 clones). Target residualised on mouse + log sizes. For each of the 453
  CpGs, clone-mean methylation over multipotent cells vs residual target, Spearman; BH FDR 0.10.
- **Readings.** Any FDR hits: list them with gene, enhancer annotation, panel selection class and
  chromatin state, and test whether hits are enriched in Dynamic vs Static (Fisher). No hits: the
  signal is diffuse across many sites (what ridge on PCs would pick up); no single-site biology
  claimed. Descriptive only: share of Dynamic vs Static CpGs with EPI-Clone's own clone-association
  p < 0.05 (column pvals_cloneass), as a check on B3h's "record leaks into Dynamic sites".

### E21 (conditional on E20 p < 0.05): condition-specific gain
- E20 model plus a per-condition gain: beta_c = beta * g_c, g_full = 1, other g_c free (4 extra
  parameters), rho_f fixed 0.88, same fit and simulation protocol.
- **Readings.** Gap closed = G-M simulated mean inside the observed CI (0.33-0.65) and p >= 0.05,
  with fit loss no worse than E20's. Otherwise the gap survives a condition-specific gain too.
- Not run if E20 gives p >= 0.05.

### E20 result (2026-11-03)

rho_f fixed 0.88; 3 restarts; best loss 0.058 (lowest of any E18-family fit); beta 8.0, sigma_w
1.05. 200 simulations on all clones at the observed depth:

| statistic | observed | simulated mean (sd) | 95% of simulations | p |
|---|---|---|---|---|
| full-G | 0.69 | 0.70 (0.10) | 0.50-0.89 | 0.90 (two-sided) |
| full-M | 0.24 | 0.32 (0.07) | 0.18-0.44 | 0.17 (two-sided) |
| G-M | 0.51 | 0.25 (0.06) | 0.15-0.34 | **0/200 >= 0.51** |

- **The G-M gap is real** (p < 0.005). It is not a sampling ceiling: at the observed depth the
  model's G-M never reaches 0.51. full-G and full-M are reproduced. **E21 triggered.**
- Note: beta ended at 8.0, the top of the restart range; noted, not interpreted.

### E19b result (2026-11-03)

LARRY in vivo, 252 clones with Neu/Mono cells in >= 2 recipient mice.

| | latent r (95% clone bootstrap) |
|---|---|
| same mouse, same time | 0.22 [0.14, 0.30] |
| different mice | 0.19 [0.10, 0.28] |
| ratio | 0.88 [0.44, 1.53] |

Same-time cross-mouse: ratio 1.02 [0.56, 1.65].
- **Reading (pre-registered terms): the ratio overlaps 0.82-0.92**, so a mouse-specific break of
  the fate state is not shown; the CI is too wide to say more.
- **The striking number is not the ratio but the level** (not pre-registered as a reading): in
  vivo, even cells of one clone in the same mouse agree on N vs M only weakly (latent r 0.22,
  vs 0.94 in vitro). In vivo, the clone sets the N/M choice much less than in culture, and
  cross-mouse agreement is about as high as within-mouse. The E6b cross-mouse drop is
  therefore a general loosening of clone control in vivo, not a recipient-specific effect.

### B3j result (2026-11-03)

181 clones, 451 CpGs tested (2 constant). Target and CpG residualised on mouse + log sizes.
- **3 FDR-10% hits** (q 0.02-0.09): AMPL390597 (Dynamic, intergenic strong enhancer near Synpo,
  panel class MPP_high, rho +0.30), AMPL366042 (Dynamic, intergenic strong enhancer, MPP_high,
  +0.26), AMPL371668 (unclassified, weak enhancer, HSC_high, -0.25). 24 CpGs at p < 0.01 vs 4.5
  expected: the signal is spread over many sites, with a few standing out.
- Hits vs class: 2 Dynamic, 0 Static (Fisher p 0.54; too few hits to test enrichment).
- **Unplanned pattern (descriptive, needs its own test):** in the top 15, every MPP_high-class
  site has positive rho (more methylation, more myeloid; 5/5) and every HSC_high-class site has
  negative rho (6/6). Clones whose multipotent cells carry a more MPP-like methylation profile
  are more myeloid-biased. Composition was not in the B3i/B3j covariates.
- **Check on B3h:** by EPI-Clone's own clone-association test, 98% of Static and 93.5% of Dynamic
  CpGs have p < 0.05. The "Dynamic" sites are almost all clone-associated, which confirms the
  B3h leakage reading directly.

### B1c result (2026-11-03)

571 clones over 7 day-2 libraries (55-116 each), 50% LSK start.

| | r (95% clone bootstrap) |
|---|---|
| random folds (B1) | 0.66 [0.62, 0.70] |
| leave one day-2 library out | 0.67 [0.63, 0.71] |
| partial, covariates removed | 0.60 [0.55, 0.65] |
| covariates alone | 0.34 [0.26, 0.41] |

- **Not confounded** (pre-registered reading: both controlled CIs above 0.5). Starting population,
  library and clone sizes carry some fate information on their own (0.34), but day-2 expression
  predicts well beyond them. "Day-2 state explains ~40%" stands (~36% after covariates).

### E21 result (2026-11-03)

Per-condition gain, rho_f 0.88; best loss 0.083; beta by condition: full 4.5, G 6.6, M 3.7,
Epo 5.0, SCF 5.7; sigma_w 0.01.

| statistic | observed | simulated mean (sd) | 95% | p |
|---|---|---|---|---|
| full-G | 0.69 | 0.68 (0.10) | 0.46-0.88 | 0.98 |
| full-M | 0.24 | 0.43 (0.07) | 0.30-0.56 | 0.02 |
| G-M | 0.51 | 0.33 (0.06) | 0.23-0.45 | 0/200 >= 0.51 |

- **Gap not closed** (pre-registered: needs p >= 0.05 and loss no worse). G-M rises from 0.25 to
  0.33 but stays out of reach, full-M now overshoots, and the fit is worse than E20 (0.083 vs
  0.058).
- Caveat: E21 nests E20 (gains = 1), so a worse loss means Nelder-Mead did not reach E20's optimum
  in the larger space (noisy objective). A better optimiser could narrow it, but the G-M vs full-M
  trade-off (raising G-M pushes full-M too high) looks structural, as in E15-E17.
- **Reading.** The G-CSF vs M-CSF agreement is a real feature the model family cannot produce,
  even with data-fixed fate-state sharing and per-condition gains. Something links G and M
  outcomes of a clone more strongly than its link to the full cocktail.

## Batch pre-registration 2026-11-03 (2): E21b, E22, B3k

### E21b: E21 refit with a warm-started optimiser
- E21's nested model scored worse than E20, an optimiser failure. Refit: start at E20's optimum
  (gains = 1), Nelder-Mead then Powell, 5 simulations per loss with common random numbers. E20's
  optimum re-scored under the same 5-simulation loss as the reference. Same 200-simulation
  prediction protocol.
- **Readings.** Loss must be <= E20's re-scored loss (else the refit failed and is reported as
  such). Gap closed only if G-M p >= 0.05 with full-G and full-M p >= 0.05. Otherwise E21's
  conclusion (gain does not close the gap) is confirmed with a proper fit.

### E22: a second clone trait shared by single-cytokine conditions
- Hypothesis: each clone has a second heritable trait u ~ N(0,1) that shifts N propensity only
  under single cytokines (G, M, Epo, SCF), not under the full cocktail: logit p = beta*zb + gamma*u
  [single-cytokine] + tau_c + noise. u is clone-level (shared across the single-cytokine
  conditions). gamma fitted with beta, sigma_w, tau_c on the within-condition statistics only
  (fit set; rho_f 0.88; warm start from E20 with gamma 0; Nelder-Mead then Powell; 5 sims, common
  random numbers). Cross-condition statistics are predictions.
- **Readings.** Supported if all three cross statistics have p >= 0.05 (G-M included) with loss
  <= E20's re-scored loss. Not supported if G-M stays at p < 0.05, or if reaching G-M breaks
  full-G/full-M. Note stated now: u is a structural guess motivated by the E20/E21 pattern, so a
  pass makes it a candidate, not a finding; it would need its own held-out test (Epo/SCF shard 1
  would be the place).

### B3k: are clones with more MPP-like multipotent methylation more myeloid?
- Index per cell = mean methylation over the panel's MPP_high-class CpGs minus mean over its
  HSC_high-class CpGs (all assay CpGs in those classes, not just B3j's top sites). Cell index
  centred within cell type and batch (removes cell-type identity), then averaged over the
  clone's multipotent cells. Target = myeloid share. Covariates: mouse, log sizes, and the clone's
  multipotent composition (HSC/MPP1, MPP2, MPP3, MPP4 fractions).
- (1) partial Spearman of index vs target given covariates, clone bootstrap CI (2000x);
  (2) the same per mouse (sign count across 6 mice); (3) uncentred index for comparison.
- **Readings.** Positive partial correlation with CI excluding 0 and same sign in >= 5 of 6 mice:
  within cell types, more MPP-like methylation marks myeloid-biased clones. Otherwise the B3j
  pattern is not supported beyond composition. Caveat stated now: the pattern was noticed in
  these data (B3j), so this is a confirmation on a pre-defined whole-class index, not an
  independent replication.

### B3k result (2026-11-03)

181 clones; index over 58 MPP_high and 82 HSC_high assay CpGs.

| index | partial Spearman (95% clone bootstrap) | positive mice |
|---|---|---|
| centred within cell type x batch | 0.19 [0.04, 0.33] | 5 of 6 |
| uncentred | 0.18 [0.03, 0.33] | 3 of 6 |

Per mouse (centred): 0.27, 0.01, -0.04, 0.26, 0.02, 0.16.
- **Meets the pre-registered bar, narrowly.** Within cell types and beyond mouse, clone size and
  multipotent composition, clones whose multipotent cells carry a more MPP-like methylation
  profile are more myeloid-biased. Two of the five positive mice are near zero, so the effect is
  carried mainly by three mice. Same data as the B3j discovery (whole-class index, not the top
  sites), so this is a confirmation, not a replication.
- **Reading.** A first biological handle: myeloid-biased clones' multipotent cells look further
  along the HSC-to-MPP methylation axis than their cell-type label says.

### E21b / E22 results (2026-11-03)

E20's optimum re-scored under the 5-simulation loss: 0.062. Both refits beat it (warm start
worked; the E21 optimiser failure is fixed).

| model | loss | full-G (p) | full-M (p) | G-M sim mean (95%) | G-M p |
|---|---|---|---|---|---|
| E21b per-condition gain | 0.011 | 0.58 (0.28) | 0.39 (0.08) | 0.21 (0.12-0.29) | 0/200 |
| E22 second trait | 0.023 | 0.57 (0.22) | 0.39 (0.09) | 0.21 (0.12-0.29) | 0/200 |

E21b gains: G 1.26, M 1.01, Epo 0.98, SCF 2.21 (beta full 8.2). E22: gamma -0.09, i.e. ~0.
- **E21b: E21's conclusion confirmed with a proper fit.** Per-condition gains fit the
  within-condition data far better, and G-M gets *lower* (0.21), not higher.
- **E22: not supported.** The within-condition data drive the second trait to ~0 (gamma -0.09),
  so it cannot be identified from them, and G-M stays at 0.21. A trait large enough to raise G-M
  would have to be imposed, not fitted.
- **Pattern across E15-E22:** the better a model fits the within-condition statistics, the lower
  its G-M prediction (E20 0.25 at loss 0.058; E21b/E22 0.21 at 0.011-0.023). The observed
  within-condition data and the observed G-M agreement pull in opposite directions in every
  model of this family. Either the G-M statistic carries something the within-condition summary
  statistics cannot express (e.g. which clones survive into both single-cytokine wells), or the
  observed G-M is high by chance beyond what sampling explains (E20 says < 0.5% under the
  model).

## B3l: independent replication of B3k — skipped by the power rule (2026-11-03)

Searched the EPI-Clone Figshare collection. Downloaded M.8 (LARRY transplant from an aged donor,
same 453-CpG panel, identical CpG order) and M.7 (native ageing, two replicates). M.7 carries no
clone labels (EPI-Clone clones would have to be inferred from the same methylation, circular).
M.8 has only HSC/MPP1 as multipotent type and **13 eligible clones** (>= 1 HSC/MPP1 cell, >= 5
GMP + EryP), counted without looking at outcomes. For a true partial r of 0.19, power at n = 13
is ~10%; the expected CI half-width (~0.55) exceeds the effect. **Look skipped (E18 power rule);
M.8 outcomes stay unseen.** Human EPI-Clone data use a different panel without the MPP_high /
HSC_high classes, so they cannot replicate the index directly. B3k stays a single-dataset result.

## E23: does clone size couple to N-share, and does that explain G-M? (pre-registered 2026-11-03)

- **(a) Diagnostic.** Cytokine data, per condition: Spearman of clone N-share vs log(N+M) over
  clones with >= 2 N+M cells, clone bootstrap CI (2000x). The same statistic in 200 E20
  simulations (the model draws fate independently of observed counts, so it should sit near 0).
- **(b) Model E23m, run only if (a) finds a condition whose data CI excludes 0 and whose value lies
  outside the E20 simulated 95% range.** logit p += delta_c * s_kc, s_kc = the clone's observed
  log(N+M) in condition c, standardised within c. delta_c fitted per condition; the per-condition
  share-size correlation is added to the within-condition fit targets (weight 1). rho_f 0.88,
  warm start from E20 (delta 0), Nelder-Mead then Powell, 5 sims, common random numbers;
  200-simulation predictions.
- **Readings.** (a) no coupling beyond the model's range: presence/size structure is not the
  missing piece. (b) supported if G-M p >= 0.05 with full-G and full-M p >= 0.05; not supported
  otherwise. A pass would make size coupling a candidate mechanism (clones that expand more under
  single cytokines skew N/M consistently), still needing a held-out test.

### E23 result (2026-11-03)

(a) Diagnostic: share-size Spearman per condition, data vs E20 simulations (95%):
full -0.06 [-0.13, 0.01] vs [-0.06, 0.08]; **G +0.09 [0.02, 0.18]** vs [-0.09, 0.10];
**M -0.09 [-0.15, -0.02]** vs [-0.01, 0.12] (trigger); Epo, SCF ~0 (wide). Under G-CSF larger
clones are more N, under M-CSF larger clones are more M: clones that expand more follow the
cytokine more.
(b) E23m: loss 0.047 (vs 0.170 at delta = 0 with the added targets); delta G +2.9, M -3.7.
full-G 0.55 (p 0.12), full-M 0.37 (p 0.13), **G-M 0.20, 0/200 >= 0.51. Not supported.** The
opposite-sign size coupling, if anything, pulls G and M apart.

## E24: carry-over of cells already committed before the split (pre-registered 2026-11-03)

- **Observation (design check, not an outcome look).** In the cytokine experiment 16% of day-2
  cells (sampled just before the split, full cocktail) are already annotated Neu (1,058) or Mo
  (1,648). Their sisters at the split are presumably also committed and carry their fate into
  every condition regardless of cytokine (cytokines instruct uncommitted GMPs, Rieger et al.
  2009). The E18-E23 simulator has every cell decide after the split, so it has no carry-over.
  E10 (learned rule) had estimated ~29% committed by day 2.
- **(a)** Per clone with >= 2 day-2 Neu+Mo cells: day-2 committed N-share vs N-share in G and in
  M (Spearman, clone bootstrap). **(b)** G-M r restricted to clones with no day-2 Neu/Mo cells vs
  clones with >= 1 (descriptive; spent statistic). **(c) Model E24m:** E20 plus carry-over: a
  fraction phi_c of each clone's N+M cells in condition c are carried-over committed cells whose
  fate is drawn from the clone's own N-propensity in the full cocktail (latent zb_full with
  tau_full), not the condition's; the rest decide post-split as in E20. phi_c fitted per
  condition (bounded 0-1) with the within-condition targets; warm start from E20 (phi 0);
  Nelder-Mead then Powell; 5 sims, common random numbers; 200-simulation predictions.
- **Readings.** (a) positive with CI excluding 0 in M (where post-split N is rare) supports
  carry-over. (c) supported if all three cross statistics have p >= 0.05 with loss <= E20's
  re-scored loss. If (c) passes, carry-over of pre-committed cells is the leading candidate for the
  G-M gap; it would still need the held-out Epo/SCF shard 1.
