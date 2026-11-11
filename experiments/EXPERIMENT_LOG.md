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

### E24 result (2026-11-03)

(a) Uninformative: only 3 (G) and 4 (M) clones have >= 2 day-2 Neu/Mo cells and enough cells in
the condition. (b) The G-M statistic rests on 78 clones; 77 had no day-2 committed cell sampled,
and their G-M r alone is 0.52 [0.33, 0.67] (the full effect). (c) E24m: loss 0.014 (E20 re-scored
0.062); fitted carry-over phi = 0.05 in full, **0.00 in G, M, Epo, SCF**; full-G 0.57 (p 0.28),
full-M 0.38 (p 0.11), **G-M 0.21, 0/200 >= 0.51. Not supported.** The within-condition data do
not ask for carry-over, and the G-M clones are not the ones with committed day-2 sisters.

### Summary of the G-M gap tests (E20-E24)

Sampling (E20), per-condition gain (E21, E21b), second single-cytokine trait (E22), clone size
coupling (E23) and pre-split commitment carry-over (E24) all fail; G-M stays at 0.20-0.25 vs 0.51.
Literature sweep 2 (LITERATURE_SWEEP_2026-11b.md) points to clone *types* (a mix of
cytokine-insensitive, intrinsically fated clones and plastic clones; two monocyte routes) as the
best-supported untested candidate.

## Batch pre-registration 2026-11-03 (3): E25 clone types, B7 selection of the G-M clones

### E25: a mixture of clone types
- E20 model (rho_f 0.88) plus two clone classes drawn per clone (own rng, so the E20 model is
  nested at pi = 0): **fixed** clones (probability pi_fix) are cytokine-insensitive: in every
  condition they use the full-cocktail offset tau_full instead of tau_c (intrinsically fated,
  Dawson-lab type); **M-only** clones (probability pi_M) make no neutrophils in any condition
  (MDP-route type, Yanez 2017). The rest are plastic as in E20. pi_fix, pi_M fitted (logit
  scale) with beta, sigma_w, tau_c on the within-condition statistics (fit set; warm start from
  E20 with pi ~ 0; Nelder-Mead then Powell; 5 sims, common random numbers); 200-simulation
  predictions of the cross statistics.
- **Readings.** Supported if all three cross statistics have p >= 0.05 (G-M included) with loss
  <= E20's re-scored loss. Partial if G-M rises clearly (simulated mean >= 0.33, the observed CI
  floor) but p < 0.05. Not supported if G-M stays <= 0.25 or the fitted pi are ~0.

### B7: are the 78 G-M clones a selected subset?
- Compare clones with >= 2 N+M cells in both G and M ("both") with clones that reach >= 2 N+M
  cells in only one of the two ("G-only", "M-only"; all with a day-2 sample where available).
  (1) N-share in G: both vs G-only; N-share in M: both vs M-only (Mann-Whitney, medians).
  (2) Same comparison in 200 E20 simulations (presence is fixed to the observed counts, so any
  data-vs-model difference is presence-fate coupling the model lacks). (3) Descriptive: day-2
  cell count, day-2 committed fraction, total size.
- **Readings.** If the data show a presence-fate difference (e.g. "both" clones more N in G or
  less extreme in M) outside the E20 simulated 95% range, presence is coupled to fate and the
  G-M clones are a selected subset: the statistic needs a selection-aware model. If inside, the
  subset is not selected on fate beyond what the model already reproduces.

### B7 result (2026-11-03): the G-M clones are a strongly selected subset

78 clones with >= 2 N+M cells in both G and M; 474 G-only; 931 M-only.

| | in both G and M | in one only | Mann-Whitney p |
|---|---|---|---|
| N-share in G (mean / median) | 0.35 / 0.00 | 0.72 / 1.00 (G-only) | 3e-12 |
| N-share in M (mean / median) | 0.13 / 0.00 | 0.05 / 0.00 (M-only) | 5e-5 |

Median sizes similar (3-4 N+M cells). Data vs 200 E20 simulations (presence fixed to observed):
mean N-share difference in G, both minus G-only: **-0.37** vs model 0.00 [-0.11, 0.10]; in M:
**+0.08** vs model 0.00 [-0.04, 0.04]. Both far outside.
- **Reading (pre-registered): presence is coupled to fate.** Neutrophil-pure clones present
  under G-CSF are mostly absent (or below 2 N+M cells) under M-CSF; the clones seen in both are
  monocyte-leaning bipotent clones, and the ones that do make neutrophils under M-CSF are the ones
  also seen under G-CSF. Cytokines act partly by **selection** of lineage-restricted clones, not
  only by instruction. The G-M statistic is computed on a subset whose membership depends on fate,
  and every simulator since E9 pins membership to the observed counts while drawing fate
  independently of it. **This is the best candidate for the missing piece.**

### E25 result (2026-11-03)

Three starts (pi ~0; fixed-heavy; M-only-heavy) reached similar losses (0.015, 0.017, 0.017;
E20 re-scored 0.062) with very different pi (0.04/0.04, 0.30/0.12, 0.02/0.15): the clone-type
shares are **not identified** by the within-condition statistics. Best: pi_fix 0.04, pi_M 0.04;
full-G 0.55 (p 0.18), full-M 0.35 (p 0.28), **G-M 0.16, 0/200 >= 0.51. Not supported.**
- Like every earlier variant, clone types with membership pinned to the observed counts cannot
  produce the G-M agreement. B7 says why: what matters is *which clones are present*, and that
  is set by fate in the data but fixed in the simulator.

## Proposed next experiments after B7 (not yet pre-registered)

1. **E26: selection-aware simulator (counts simulated, presence emergent).** Per clone: fate
   latent (rho_f 0.88 across conditions), expansion trait (log-normal clone size). In each
   condition, cells choose N/M by instruction (tau_c) and then each lineage expands/survives by a
   condition-specific factor (s_c^N, s_c^M: selection). Observed counts are Poisson thinned at
   each condition's depth, so whether a clone is seen in a condition depends on its fate. Fit on
   within-condition statistics plus per-condition presence (clones with >= 2 N+M) and size
   distribution. Two variants: pure selection (shared tau) vs instruction + selection.
   Predictions: G-M, full-G, full-M and the B7 presence-fate differences (spent, used as
   consistency checks). If it passes, it makes the first model-based prediction worth spending
   the last held-out test on (Epo/SCF shard 1, after a power check).
2. **B8: negative control for B7.** Same presence-fate comparison between the two wells of the
   main in vitro experiment (same cocktail). No selection expected: the "both wells" vs
   "one well" N-share difference should be ~0 there. If not, B7 partly reflects sampling depth.
3. **B9: what marks the "both" clones at day 2?** Day-2 state (SPRING position / expression PCs)
   of clones later seen in both single-cytokine wells vs G-only vs M-only. Prediction under
   selection: "both" clones are bipotent / earlier in the GMP state at day 2; G-only clones are
   already neutrophil-primed. Tests the mechanism, not only the statistic.
4. **Re-read E9-E17 in this light.** The "within-condition fit vs cross-condition agreement" tug
   of war across E15-E25 is what selection on presence would produce. Check whether the E10
   learned rule (the only model with cell-level survival/arrest actions) reproduces the B7
   pattern.
5. **Kin correlations with selection (E19c).** E19's same-well vs cross-well ratio used clones
   present in both wells; repeat with presence-weighting to see whether rho_f 0.88 is biased by
   selection (in vitro wells share a condition, so the bias should be small).
6. **Outside data/collaboration (cannot do locally):** sister-pair time-lapse under G-CSF vs
   M-CSF (Rieger/Schroeder-type imaging) would show selection vs instruction per sister directly.
   CellVoyant does live-cell fate imaging commercially; the Schroeder lab (ETH) does it
   academically.

## Batch pre-registration 2026-11-03 (4): B8, B9, E26

### B8: negative control for B7 (same cocktail)
- LARRY main in vitro, wells 1 and 2 (same cytokine cocktail), N/M cells pooled over days 4 and 6.
  N-share in well 1 for clones with >= 2 N+M in both wells vs clones with >= 2 in well 1 only
  (and the same for well 2). Mean difference with clone-bootstrap CI (2000x).
- **Reading.** |difference| < 0.10 with CI including 0 in both wells: presence is not coupled to
  fate without a cytokine difference, so B7 reflects cytokine selection (and E19's same-cocktail
  kin estimate is not biased by it). Otherwise B7 partly reflects sampling, and B7's reading is
  weakened.

### B9: day-2 state of the selected clones
- Cytokine experiment, clones with day-2 cells, grouped as in B7 (both / G-only / M-only). Per
  clone, over its day-2 cells: share Undifferentiated, share Neu, share Mo, and a priming score
  = mean distance (SPRING) to the day-2 Mo centroid minus distance to the Neu centroid (higher =
  closer to Neu). Kruskal-Wallis across groups, pairwise Mann-Whitney.
- **Prediction under selection.** G-only clones are more neutrophil-primed (higher score, more
  Neu) than "both" clones; M-only clones are more monocyte-primed; "both" clones are the most
  undifferentiated or intermediate. Supported if G-only > both on the priming score (p < 0.05).

### E26: selection-aware simulator (presence emerges from fate)
- Universe: every clone seen anywhere in the cytokine experiment. Per clone: fate latent z, branch
  latent per condition (rho_f 0.88), expansion trait a ~ N(0, sigma_a). In condition c:
  p = sigmoid(beta * zb + tau_c); expected N = exp(mu_c + a) * p * s_c, expected M =
  exp(mu_c + a) * (1 - p); counts Poisson. s_c = relative neutrophil survival/expansion (s_full =
  1). Presence (>= 2 N+M) is therefore set by fate and size.
- Two variants: **E26a pure selection** (one shared tau for all conditions; log s_c free);
  **E26b instruction + selection** (tau_c and log s_c free). Fit on the fit set (Epo/SCF shard 1
  excluded) to the within-condition statistics plus, per condition, the number of present clones
  (relative error) and the mean log size of present clones. Nelder-Mead then Powell from 3
  starts, 5 simulations, common random numbers.
- **Predictions (200 simulations):** full-G, full-M, G-M (p as in E20), and B7's presence-fate
  differences (G and M), all spent data used as consistency gates.
- **Readings.** A variant passes if G-M p >= 0.05, full-G and full-M p >= 0.05, and both B7
  differences fall inside its simulated 95% range. If neither passes, presence-coupled selection
  of this simple form does not explain G-M either.
- **Held-out look (Epo/SCF shard 1), only if a variant passes.** Power rule before the look: the
  passing variant's predicted shard-1 pooled r and E20's must differ by more than the expected
  shard CI half-width (~0.37 from shard 0); otherwise skipped. Look through test_oracle.py,
  reading: observed value inside the passing variant's 95% predicted range.

### B8 / B9 results (2026-11-03)

**B8 (same cocktail, main in vitro wells).** N-share of clones present in both wells vs one well:
well 1 0.58 vs 0.49, diff **+0.09 [0.02, 0.15]** (291 vs 672 clones); well 2 0.57 vs 0.50,
diff **+0.06 [0.00, 0.12]** (291 vs 871).
- By the pre-registered rule (CI must include 0) **B8 fails, so B7 is weakened**: presence and
  fate are coupled a little even without a cytokine difference (clones present in two wells lean
  slightly N). But this coupling is about a quarter of B7's size and of the **opposite sign**
  (B7 under G-CSF: -0.37), so sampling of this kind cannot produce B7's pattern. E19's
  same-cocktail kin estimate may carry a small presence bias.

**B9 (day-2 state of the B7 groups; 39 both, 138 G-only, 282 M-only clones with day-2 cells).**
Neutrophil-priming score (median, SPRING distance to Mo centroid minus to Neu centroid): G-only
772 > both 739 > M-only 611; Kruskal p 6e-19; **G-only vs both p 0.006**, M-only vs both p 0.002.
Day-2 Neu share also differs across groups (Kruskal p 0.0006; G-only 3.1%, both 0%, M-only 0.4%).
- **Supported (pre-registered).** Clones later seen only under G-CSF were already more
  neutrophil-primed at day 2; clones seen only under M-CSF more monocyte-primed; the clones seen
  under both sit in between. The selection is on a fate state that already exists at day 2.

### E26 results (2026-11-03)

**Deviations from pre-registration (made before any result was seen).** Powell stage dropped and
parameters put through bounded transforms; empty conditions (no present clones, Epo/SCF) give a
fixed loss penalty instead of NaN. Two earlier Actions runs returned only the NaN/1e6 penalty and
were discarded as failed runs, not looks. Run on Actions, 3 starts per variant as separate jobs;
best start = lowest loss.

| start | loss | full-G (p) | full-M (p) | G-M (p) | B7 G inside | B7 M inside |
|---|---|---|---|---|---|---|
| a s0 | 9.95 | 0.70 (0.48) | 0.74 (0.00) | 0.71 (0.00) | no (-0.07) | no (0.21) |
| a s1 | 223 | 0.75 (0.00) | 0.64 (0.00) | 0.60 (0.00) | yes (-0.36) | no |
| **a s2 (best)** | **0.34** | 0.63 (0.18) | 0.21 (0.60) | **0.22 (0.000)** | no (-0.28) | no (0.00) |
| b s0 | 12.2 | 0.70 (0.51) | 0.59 (0.00) | 0.50 (0.37) | no (-0.07) | no (0.14) |
| b s1 | 247 | 0.75 (0.00) | 0.36 (0.00) | 0.32 (0.00) | no | no |
| **b s2 (best)** | **0.33** | 0.64 (0.10) | 0.04 (0.00) | **0.06 (0.000)** | no (-0.03) | no (0.00) |

Data: G-M 0.51, full-M 0.24, B7 G -0.37, M +0.08.
- **Fails (both variants).** The best fits reproduce the within-condition data and full-G but put
  G-M at 0.22 and 0.06, and do not reproduce B7. Presence-coupled selection of this simple form
  (neutrophil survival factor per condition + clone expansion trait) does not explain G-M either.
- Same pattern as E21-E25: poorly fitting starts (a s0, b s0) can reach G-M 0.5-0.7 but only by
  breaking full-M or the within statistics. G-M and within-condition fit still trade off.
- Held-out shard 1 stays unseen (no variant passed; no look spent).

## Batch pre-registration 2026-11-03 (5): E19c, E27 (written before either script was run)

### E19c: is rho_f 0.88 biased by which clones are present?
- E19 uses clones present in both in vitro wells; B8 showed presence and fate couple slightly
  (+0.06/+0.09). If presence selects clones, the selected set's sister agreement may differ from
  the whole population's. Clone size is the handle: large clones are present almost regardless of
  fate, small clones only when their fate makes enough sampled N+M cells.
- **Test.** E19 primary estimator (same/cross-well tetrachoric ratio, any day), run separately in
  clone-size tertiles (total N+M cells over all days and wells), 1000 bootstraps each.
  Secondary: clones with >= 3 N+M cells in each well (stricter presence).
- **Reading.** No presence bias if every tertile's rho_f point estimate lies inside E19's 95% CI
  [0.82, 0.92] and the smallest vs largest tertile difference has a bootstrap CI that includes 0.
  A rising trend with size (small clones lower) would mean selection inflates disagreement in
  small clones, i.e. E19 is a lower bound; a falling trend the opposite.

### E27: do the trained learned-rule policies reproduce B7?
- E11, E12 and the converged E16 rerun (r_e16) policies have cell-level divide/arrest actions
  (selection) and were trained without B7. B7 (spent data): under G-CSF, clones also present
  under M-CSF have mean N-share 0.37 lower than G-only clones; under M-CSF, +0.08. E20 predicts ~0.
- **Test.** For each saved policy (10 per family: 5 inherit + 5 resample seeds), 20 simulations
  (seeds 100-119) at the trained clone number; B7 statistics with e25_clone_types.presence_stats.
  Runs on GitHub Actions, one job per family.
- **Reading.** A family reproduces B7 if, in >= 6 of its 10 policies, both data values
  (G -0.37, M +0.08) fall inside that policy's simulated 95% range. Also reported: whether the
  G difference is negative in the family mean (the qualitative sign).
- If a family reproduces B7 but still misses G-M (all did: 0.17-0.30), selection on presence is
  shown to be necessary but not sufficient; if none does, the learned selection schedules are
  not the selection seen in the data. No held-out data are used.

### E19c results (2026-11-03)

| clones | size (N+M cells) | n | rho_f [95% CI] | r same well | r cross well |
|---|---|---|---|---|---|
| small tertile | 2-4 | 84 | 0.95 [0.85, 1.00] | 0.98 | 0.93 |
| middle tertile | 5-11 | 199 | 0.89 [0.81, 0.95] | 0.96 | 0.85 |
| large tertile | 12-143 | 181 | 0.82 [0.72, 0.90] | 0.89 | 0.73 |
| >= 3 N+M in each well | | 185 | 0.83 [0.73, 0.90] | 0.93 | 0.77 |

Large minus small: **-0.13 [-0.25, -0.01]**.
- **By the pre-registered rule there is a size (presence) dependence.** The CI excludes 0, and the
  large tertile's point estimate (0.817) sits just below E19's CI [0.82, 0.92]. The trend is falling:
  small clones agree more across wells.
- **What it means for G-M.** The clones least affected by presence selection (large ones) share
  less across wells (about 0.82), not more. Using 0.82 instead of 0.88 in E20 would lower the
  predicted G-M, so the gap widens. Selection on presence does not hide a higher rho_f.
- Caveat (not tested): large clones contribute more day-4 versus day-6 pairs, so within-clone drift
  over time can also lower their cross-well r. Size and time are confounded here.

### E27 results (2026-11-03)

**0 of 30 policies reproduce B7 (E11 0/10, E12 0/10, r_e16 0/10). Fails.** Family means for the G
difference are -0.02 / -0.04 / -0.03 (data -0.37). For M they are -0.01 / -0.02 / -0.00 (data
+0.08). One policy (E12 inherit seed 0) gets the sign and a third of the size (-0.17
[-0.24, -0.08]). The learned divide/arrest schedules are not the selection seen in the data.

**Post-hoc observation (not pre-registered; supporting only).** Clone overlap between G-CSF and
M-CSF presence is far higher in every model than in the data. Clones present under both, as a
share of clones present under G-CSF:

| | both | G-only | M-only | both / (present under G) |
|---|---|---|---|---|
| data | 78 | 474 | 931 | **0.14** |
| E11 policies | 130-255 | 267-446 | 342-411 | 0.33-0.42 |
| r_e16 policies | 497-781 | 848-1010 | 1076-1248 | 0.37-0.45 |

In the data, a clone seen under G-CSF is rarely seen under M-CSF. Presence across the two
cytokines is nearly independent or anti-correlated: it depends on the clone's fate, not only its
size. This statistic (presence overlap) has never been in any loss. It is a candidate fit target
and the sharpest description yet of what the models miss: in the data, the cytokine decides
whether a clone's cells survive to be sampled, and it does so by the clone's fate.

## E28: founder split across wells + presence overlap (pre-registered 2026-11-03)

- **Motivation.** E27 post hoc: the clones present under both G-CSF and M-CSF, as a share of
  clones present under G-CSF, are 0.14 in the data (78 / 552). Learned policies give 0.33-0.45,
  and the best E26 fits give 0.59-0.62 (computed before this entry, from the E26 parameters). E26
  puts every clone in every condition, scaled by one clone size. In the experiment, a clone's few
  day-2 cells were divided among five wells, so a clone can have no founder in a well. That is
  structural and fate-independent, and no model has had it.
- **Model.** E26b (tau_c and s_c free, fate latent rho_f 0.88, clone expansion trait a), plus
  founders per condition k_c ~ Poisson(nu) independently. Expected counts are scaled by k_c, and a
  clone with k_c = 0 is absent from c. nu is free (shared).
- **Variants.** **E28a**: loss = E26 loss + ((overlap_sim - 0.141) / 0.05)^2.
  **E28b**: the E26 loss only, testing whether the split alone produces the low overlap.
  3 starts each, Nelder-Mead, GitHub Actions, best start = lowest loss.
- **Readings (same gates as E26).** A variant passes if G-M, full-G and full-M p >= 0.05 (as in
  E20) and both B7 differences fall inside its simulated 95% range. Also reported: the overlap
  (in E28a it is fitted, so it is a fit check only; in E28b it is a prediction, inside ±0.05 of
  0.141 or not).
- **What each outcome means.** If E28a passes, the G-M gap comes from presence structure plus
  selection, and the Epo/SCF shard-1 look follows after the power rule. If E28a fits the overlap
  but G-M stays low, the founder split is real but not the missing piece. If E28b reaches about
  0.14 on its own, the overlap anomaly is a design artefact (well splitting), not biology.

### E28 results (2026-11-03)

Run on Actions, 3 starts per variant; best start = lowest loss.

| start | loss | nu | overlap (0.141) | full-G (p) | full-M (p) | G-M (p) | B7 G inside | B7 M inside |
|---|---|---|---|---|---|---|---|---|
| a s0 | 11.7 | 1.19 | 0.12 | 0.72 (0.15) | 0.59 (0.00) | 0.52 (0.79) | no (-0.04) | yes |
| a s1 | 590 | 2.06 | 0.68 | 0.76 (0.00) | 0.38 (0.00) | 0.34 (0.00) | no | no |
| **a s2 (best)** | **2.98** | 0.32 | 0.16 | 0.71 (0.67) | 0.12 (0.05) | **0.15 (0.000)** | no (-0.05) | no (0.00) |
| b s0 | 27.6 | 1.54 | 0.62 | 0.72 (0.05) | 0.60 (0.00) | 0.52 (0.58) | no | yes |
| b s1 | 483 | 2.07 | 0.69 | 0.76 (0.00) | 0.38 (0.00) | 0.34 (0.00) | no | no |
| **b s2 (best)** | **1.03** | 0.29 | **0.22** | 0.70 (0.96) | 0.16 (0.07) | **0.21 (0.010)** | no (-0.04) | no (0.00) |

- **Both variants fail** the G-M and B7 gates.
- **The founder split explains most of the overlap anomaly.** Without the overlap in the loss
  (E28b), fitting the within-condition data alone drives nu to 0.29 founders per well (most
  clones have 0 or 1 founder in a given well). That brings overlap from E26's 0.62 to 0.22. It
  is still 0.08 above the data, outside the pre-registered ±0.05, so splitting is most of the
  anomaly but not all of it. E28a fits it (0.16).
- **G-M gets lower, not higher** (0.15 / 0.21). With about one founder per well, G-M compares
  single sister cells. A shared fate latent with rho_f 0.88 still cannot give 0.51 once each
  side is one founder plus cytokine noise.
- **B7's G-CSF effect (-0.37) is not produced by splitting** (model -0.04/-0.05). The fate-selected
  presence under G-CSF remains unexplained.
- Same trade-off again: the starts that reach G-M ~0.52 (a s0, b s0) break full-M (0.59 vs 0.24).
- Held-out shard 1 stays unseen.

**Reading across E20-E28.** Every model family with one shared fate latent per clone fits the
within-condition data only with G-M at 0.06-0.33. Accounting for presence structure (selection
E26, founder split E28) does not close the gap. What remains: G-CSF and M-CSF outcomes of the
same clone are linked more than either is to the full cocktail, and B7's G-CSF presence
selection is large. A latent that couples the G and M responses specifically, without the full
cocktail, is the next untested structure (for example, a clone-level cytokine-sensitivity
trait).

## E29: ceiling test, shared single-cytokine trait with the cross statistics fitted (pre-registered 2026-11-03)

- **Motivation.** E20-E28: every one-latent model that fits the within-condition data predicts
  G-M 0.06-0.33. E22 tried a second clone trait u shared by the single-cytokine conditions, fitted
  to within-condition data only, and u went to ~0 (not identifiable). A multiplicative
  cytokine-sensitivity trait (logit shift tau_c * exp(kappa v)) is not tested. Its sign is wrong
  by construction (a high-v clone goes more N under G-CSF and more M under M-CSF, which lowers
  G-M), and this is stated before any run. E29 asks the remaining question directly: **can this
  model family fit the within-condition data AND the cross statistics at once, and does B7 then
  follow without being fitted?**
- **Model.** E28 (founder split k_c ~ Poisson(nu), E26b instruction + selection, rho_f 0.88)
  plus u ~ N(0,1) per clone: logit p += gamma * u in G, M, Epo, SCF (not full). gamma in
  [-5, 5], free.
- **Variants.** **E29w**: loss = E28a loss (within + presence + overlap). The E22 identifiability
  question in the E28 setting. **E29x**: E29w loss + sum over full-G, full-M, G-M of
  ((r_sim - r_obs) / 0.1)^2 (spent statistics used as fit targets). 3 starts each, Nelder-Mead,
  GitHub Actions, best start = lowest loss. The within-only part of the loss is reported
  separately.
- **Readings.**
  - E29w: gamma identified if |gamma| > 0.5 in the best start. Otherwise it repeats E22.
  - E29x is a **candidate** only if all of the following hold:
    1. all three cross statistics reach p >= 0.05 (a fit check);
    2. its within-only loss is <= 1.5x E29w's;
    3. both B7 differences fall inside its 95% range (B7 is not fitted, so this is the real test).
  - If 1 holds and 2 fails, the family cannot fit both: a structural negative, the strongest
    statement so far. If 1 and 2 hold but 3 fails, G-M is reachable but the presence selection
    is still missing.
  - A candidate goes to the power rule against E20's shard-1 prediction (difference > ~0.37)
    before any held-out look.

### E29 cold-start results (2026-11-03): optimiser failure, not a reading

| start | loss | within | presence+overlap | cross | gamma | full-M (p) | G-M (p) | B7 G |
|---|---|---|---|---|---|---|---|---|
| w s0 | 15.9 | 0.49 | 15.5 | - | 1.17 | 0.59 (0.00) | 0.54 (0.27) | -0.04 |
| w s2 | 27.0 | 0.46 | 26.6 | - | -2.07 | 0.12 (0.01) | 0.19 (0.00) | -0.06 |
| x s0 | 30.6 | 0.51 | 20.9 | 9.2 | 1.19 | 0.58 (0.00) | 0.54 (0.21) | -0.04 |
| x s2 | 22.7 | 0.47 | 15.2 | 7.0 | -2.06 | 0.12 (0.03) | 0.17 (0.00) | -0.08 |

(s1 starts at loss about 540, not shown.) **E29w nests E28a (gamma = 0) but its best loss is 15.9
against E28a's 2.98.** The fit stalled in presence space, so these starts say nothing about the
model. By the pre-registered gates, criterion 1 fails everywhere (full-M), so this is neither a
candidate nor the structural negative.

**Deviation (before any warm result is seen).** Rerun both variants from 3 warm starts: E28a's
best parameters mapped back through the transforms (checked to reproduce them exactly), with
gamma 0, +1.46 and -1.46 (x = 0, +0.3, -0.3). Outputs are e29{w,x}warm_s{0,1,2}.json. Same gates;
E29w's within-only loss for criterion 2 is taken from the warm runs.

### E29 warm-start results (2026-11-03)

| start | loss | within | gamma | full-G (p) | full-M (p) | G-M (p) | B7 G | B7 M |
|---|---|---|---|---|---|---|---|---|
| w s0 | 1.99 | 0.34 | 0.00 | 0.70 (0.74) | 0.11 (0.09) | 0.12 (0.00) | -0.05 | 0.00 |
| w s1 | 1.85 | 0.37 | 1.46 | 0.65 (0.69) | 0.11 (0.09) | 0.14 (0.00) | -0.06 | 0.00 |
| **w s2 (best)** | **1.66** | 0.36 | -1.44 | 0.66 (0.70) | 0.10 (0.04) | 0.12 (0.00) | -0.06 | 0.00 |
| x s0 | 8.72 | 0.35 | 0.00 | 0.71 (0.71) | 0.11 (0.03) | 0.14 (0.00) | -0.05 | 0.00 |
| **x s1 (best)** | **8.10** | 0.35 | 1.48 | 0.65 (0.68) | 0.11 (0.03) | 0.14 (0.00) | -0.06 | 0.00 |
| x s2 | 8.75 | 0.40 | -1.44 | 0.65 (0.61) | 0.10 (0.04) | 0.12 (0.00) | -0.06 | 0.00 |

- The warm starts beat E28a (1.66 vs 2.98), so the optimiser problem is fixed. gamma stays
  where each start put it, because the loss is flat in gamma.
- **E29w:** gamma is not identified (the loss is flat across 0 and ±1.46). This repeats E22.
- **E29x fails criterion 1.** With the spent cross statistics in the loss, G-M still sits at
  0.14 (p < 0.005) and full-M at 0.11 (p 0.03). B7 is not reproduced.
- **Gamma scan (diagnostic, after the gates; e29_gamma_scan.json).** E29w best fit, gamma from -5
  to +5, 30 simulations each: G-M 0.10-0.15 at every gamma, and within-condition loss
  0.34-0.42 throughout. Even a huge shared single-cytokine trait does not raise G-M.
  **Within this family, G-M is capped around 0.15, whatever the fitted structure.** This is a
  stronger negative than the pre-registered "1 holds, 2 fails": criterion 1 cannot be reached
  at all.
- **Why (fitted parameters, all E26-E29 best fits).** M-CSF is fitted by double suppression of
  neutrophils: instruction tau_M about -3.8 AND survival s_M about 0.025 (E26b: -4.5 and 0.009).
  Under M-CSF a clone then makes almost no N cells. Its N-share is Poisson noise around zero
  and carries no clone signal, so any correlation with G-CSF is capped regardless of the latent
  structure. In the data, clones seen under both cytokines keep a clone-consistent N-share
  under M-CSF (B7: 0.13 vs 0.05).
- **Proposed next diagnostic (B10, not yet pre-registered).** In the data, among the 78 G-M
  clones, how many N cells per clone under M-CSF, and is the M-CSF N-share predicted by the
  G-CSF N-share beyond what Poisson thinning at the fitted s_M allows? If the data's M-CSF N
  cells are too many and too clone-consistent for any s_M that fits the within statistics, the
  missing piece is located: neutrophil output under M-CSF is a clone property, not a uniform
  survival factor.

## B10: is neutrophil output under M-CSF a clone property? (pre-registered 2026-11-03)

- **Motivation.** E29: every fit reproduces M-CSF by suppressing neutrophils twice (tau_M about
  -3.8, s_M about 0.025). That leaves the M-CSF N-share as Poisson noise around zero and caps
  G-M at about 0.15. If the data's M-CSF N cells are more frequent or more clone-consistent than
  that allows, the missing structure is located.
- **Clones.** The 78 "both" clones (>= 2 N+M under G-CSF and under M-CSF, pooled days), the B7
  definition. **Statistics** (data, then the same on 200 simulations of the E29w best fit
  e29wwarm_s2 and of E28a best e28a_s2, the same clone definition per simulation):
  1. fraction of both-clones with >= 1 N cell under M-CSF;
  2. mean N cells per both-clone under M-CSF;
  3. Spearman of G-CSF N-share vs M-CSF N-share among both-clones;
  4. statistic 3 restricted to both-clones with >= 1 N under M-CSF (the clone signal where N
     exists).
- **Readings.**
  - **"Too many"**: statistic 1 or 2 above the model's 97.5th percentile. The fitted M-CSF
    suppression is too strong for the G-M clones (in the data, these clones keep making N cells
    under M-CSF).
  - **"Too consistent"**: statistic 1 and 2 inside the range, but 3 or 4 above the 97.5th
    percentile. N counts are right, but which clones make them follows the G-CSF behaviour.
  - Both point to neutrophil output under M-CSF being a clone property tied to the clone's
    G-CSF output. That is the structure for the next model (E30).
  - If all four statistics are inside the model's range, M-CSF neutrophils are not where G-M
    comes from, and the gradient points elsewhere (the G-CSF side or the full cocktail).
- Spent data only (B7 clones); no held-out data.

### B10 results (2026-11-03)

Among the 78 both-clones, 200 simulations each:

| statistic | data | E29w best (95%) | E28a best (95%) |
|---|---|---|---|
| fraction with >= 1 N under M-CSF | **0.205** | 0.017 (0.00-0.06) | 0.019 (0.00-0.05) |
| mean N cells under M-CSF | **0.36** | 0.02 (0.00-0.07) | 0.02 (0.00-0.07) |
| Spearman G vs M N-share | 0.48 | 0.19 (-0.08-0.32) | 0.17 (-0.03-0.30) |
| same, clones with N under M | 0.53 | finite in 1/200 sims | finite in 0/200 sims |

- **Pre-registered reading: "too many".** The data have about 12x more both-clones with
  neutrophils under M-CSF than the fits. Post hoc (supporting): M-only clones also carry N
  cells under M-CSF (7.3% with >= 1 N, pooled N-share 2.6%) against the fit's 1.7% and 0.4%. So
  the fits under-produce M-CSF neutrophils in all clones, not only in the both-clones.

### Fit audit E26-E29 (post hoc, 2026-11-03): the selection-family fits are inadequate

Prompted by B10. Mean N-share (clones with >= 2 N+M) and number of present clones on the fit set,
data vs best fits (10 simulations; fit_audit_e26_e29.json):

| | full | G-CSF | M-CSF | Epo | SCF |
|---|---|---|---|---|---|
| data | 0.31 / 368 | **0.64** / 277 | 0.05 / 478 | 0.70 / 23 | 0.66 / 39 |
| E26a | 0.51 / 344 | 0.25 / 292 | 0.01 / 501 | 0.42 / 21 | 0.68 / 41 |
| E26b | 0.64 / 414 | 0.38 / 262 | 0.00 / 504 | 0.57 / 26 | 0.56 / 36 |
| E28a | 0.60 / 265 | 0.45 / 189 | 0.01 / 185 | 0.82 / 42 | 0.87 / 97 |
| E28b | 0.61 / 217 | 0.42 / 119 | 0.01 / 274 | 0.74 / 24 | 0.85 / 64 |
| E29w | 0.60 / 237 | 0.44 / 175 | 0.01 / 153 | 0.85 / 35 | 0.86 / 76 |

- **Every E26-E29 best fit has G-CSF less neutrophilic than the full cocktail** (the data: 0.64
  vs 0.31, the opposite), and M-CSF at 0.00-0.01 instead of 0.05. Several miss presence by 2-3x.
  Their losses (0.3-3) looked fine only because squared differences of proportions are small
  next to the relative-error presence terms. The loss was badly scaled, and the optimiser found a
  wrong basin that satisfies presence.
- **Correction.** The E26, E28 and E29 verdicts ("selection / founder split / shared trait
  cannot give G-M"; "G-M capped around 0.15"; "M-CSF double suppression") were measured on
  inadequate fits. **They are withdrawn as statements about the model structures.** They remain
  true only of those fits. The E28 overlap result (founder split lowers overlap) is a property of
  the split itself and stands qualitatively. E20-E25 used a within-only loss with adequate fits
  (E20 loss 0.058) and are not affected. The cause of the B10 gap is therefore probably the
  misfit itself (M-CSF N-share 0.01 vs 0.05), not a missing structure.
- **Process lesson.** No cross-statistic reading without an adequacy gate on the basic levels.
  E20 had such a gate. E26-E29 did not. From now on, every fit must put each condition's mean
  N-share inside its data 95% CI and presence within ±20% before anything else is read.

**Next (E30, to pre-register):** refit the E28/E29 structure with a loss in which every term is
divided by its data standard error (clone bootstrap), using a population optimiser (CMA-ES,
several restarts) instead of Nelder-Mead in 18 noisy dimensions, behind the adequacy gate. Only
then re-read G-M, B7, overlap and B10.

### Fit audit E20-E25 (2026-11-03, post hoc)

Within-condition statistics (mean N-share / variance / mixed fraction) of the best fits vs data,
10 simulations (fit_audit_e20_e25.json):

| | full | G-CSF | M-CSF | Epo | SCF |
|---|---|---|---|---|---|
| data | 0.31/0.179/0.16 | 0.64/0.205/0.13 | 0.05/0.041/0.05 | 0.70/0.212/0.00 | 0.66/0.212/0.05 |
| E20 | 0.39/0.202/0.16 | 0.51/0.212/0.18 | 0.04/0.033/0.05 | 0.55/0.210/0.14 | 0.52/0.216/0.13 |
| E21 | 0.40/0.188/0.24 | 0.54/0.211/0.18 | 0.07/0.044/0.12 | 0.51/0.209/0.16 | 0.57/0.207/0.15 |
| E21b | 0.31/0.183/0.14 | 0.61/0.208/0.14 | 0.05/0.036/0.05 | 0.63/0.196/0.13 | 0.57/0.232/0.05 |
| E22 | 0.31/0.186/0.13 | 0.62/0.204/0.15 | 0.05/0.037/0.05 | 0.58/0.208/0.13 | 0.51/0.212/0.15 |
| E24 | 0.31/0.185/0.14 | 0.62/0.203/0.16 | 0.05/0.035/0.05 | 0.63/0.194/0.14 | 0.59/0.211/0.13 |
| E25 | 0.31/0.184/0.14 | 0.63/0.202/0.15 | 0.04/0.034/0.05 | 0.62/0.195/0.12 | 0.60/0.210/0.11 |

- E21b, E22, E24 and E25 fit the basic levels well (within about 0.03, except the small Epo/SCF
  sets). E20 and E21 are looser (G-CSF 0.51-0.54 vs 0.64). These models condition on the observed
  cells per clone, so presence is matched by construction. **The G-M gap stands on adequate fits:**
  E21b, E22, E24 and E25 give G-M 0.16-0.33 (0.21, 0.21, 0.21, 0.16). E23 is not audited (needs its
  size inputs; it is the same simulator family as E20).

## Batch pre-registration 2026-11-03 (6): E30, B11 (written before either was run)

### E30: presence-aware model refitted properly (SE-scaled loss, population optimiser, adequacy gate)
- **Model.** E29 structure: E26b instruction + selection, founders per well k_c ~ Poisson(nu),
  clone expansion trait, fate latent rho_f 0.88, plus the shared single-cytokine trait gamma u.
  **E30a**: gamma fixed at 0 (= E28). **E30b**: gamma free.
- **Loss.** Every term divided by its data standard error, then squared and summed. The terms
  are: per condition mean N-share, variance and mixed fraction (clone-bootstrap SE, 500
  bootstraps of the fit set); number of present clones (SE sqrt(n)); mean log size of present
  clones (bootstrap SE); and the G/M presence overlap (bootstrap SE). No cross statistics.
  Simulation mean over 5 simulations with common random numbers.
- **Optimiser.** scipy differential evolution on the bounded transform space, x in [-3, 3]^d,
  population 6 x d, up to 120 generations, no polish. 3 seeds per variant (6 GitHub Actions
  jobs). Best = lowest loss among adequate fits.
- **Adequacy gate (before anything else is read).** Every condition's simulated mean N-share
  inside the data's 95% CI (±1.96 SE), and present clones within ±20% of the data, for full,
  G and M. Epo and SCF are reported, not gated (23 and 39 clones).
- **Readings (adequate fits only).** As E26: a pass needs G-M, full-G and full-M p >= 0.05 and
  both B7 differences inside the 95% range. B10's fraction of both-clones with N under M-CSF
  (data 0.205) and the overlap (fitted) are reported.
  - If an adequate fit passes, it is a candidate: power rule, then the shard-1 look.
  - If adequate fits fail G-M, the presence-aware family fails when measured correctly (this
    replaces the withdrawn E26-E29 verdicts).
  - If no fit is adequate, this structure cannot match the basic levels. That is itself a
    finding about the structure.

### B11: are clones that make neutrophils under M-CSF neutrophil-primed at day 2?
- **Clones.** Present under M-CSF (>= 2 N+M, pooled days) with day-2 cells. Group "N under M" =
  >= 1 N cell under M-CSF; otherwise "none".
- **Primary.** B9's day-2 neutrophil-priming score (SPRING distance to the Mo centroid minus
  distance to the Neu centroid, clone mean). Partial Spearman of score vs has-N, controlling for
  log M-CSF clone size (residualise both on log size). Supported if positive with p < 0.05.
- **Secondary.** Mann-Whitney of the score between groups; the same partial test with day-2
  Neu share instead of the score.
- **Reading.** Supported means neutrophil output under M-CSF is set by a fate state already present
  at day 2 (a clone property, the B10 gradient), and E30 should be judged on whether it can
  express that. Not supported means M-CSF neutrophils look random with respect to the day-2
  state.
- **Amendment (before any E30 run).** The SE of each per-condition share term (mean, variance, mixed
  fraction) is floored at 0.25 / sqrt(present clones). Epo's mixed fraction is exactly 0 in the
  data, so its bootstrap SE is about 0 and would dominate the loss.

### B11 results (2026-11-03)

321 clones present under M-CSF with day-2 cells. 27 of them made >= 1 neutrophil under M-CSF.

| day-2 feature | median, N under M | median, none | partial Spearman given log size (p) | Mann-Whitney p |
|---|---|---|---|---|
| neutrophil-priming score | 742 | 610 | **0.24 (1e-5)** | 2e-5 |
| day-2 Neu share | 0 | 0 | 0.18 (0.001) | 0.001 |

- **Supported (pre-registered).** Clones that make neutrophils under M-CSF were already more
  neutrophil-primed at day 2, beyond what their M-CSF clone size explains. Neutrophil output
  under M-CSF is a clone property set before the split. That is the B10 gradient, confirmed
  in the cells' own state.
- **Code fixes made during the run (logged; same conclusion each time).** v1 ranked with
  argsort-of-argsort, which breaks ties by position, and then re-ranked residuals: primary
  0.13 (p 0.017), but the secondary was an impossible 0.71. v2 used average-tie ranks but
  still re-ranked residuals: 0.16 (0.004) / 0.83. v3 is the standard partial Spearman (Pearson
  of size-residualised average ranks): 0.24 (1e-5) / 0.18 (0.001). The primary was positive with
  p < 0.05 in all three, so the reading does not depend on the fix. v3 is the reported value.
- **Together with B7, B9 and B10.** Day-2 neutrophil priming predicts three things: which clones
  appear only under G-CSF (B9), which clones keep making neutrophils under M-CSF (B11), and,
  through those, the clones whose G and M outputs agree (B7, G-M 0.51). The G-M agreement rides
  on a clone state visible at day 2 that keeps a clone's neutrophil output partly independent of
  the cytokine. The fitted models express priming only as a logit shift, which M-CSF's strong
  suppression erases.

### E30 results (2026-11-03)

Differential evolution, 120 generations; every run hit the generation cap, so none is fully
converged. Adequacy gate (full, G, M: mean N-share inside the data 95% CI, present clones ±20%):

| run | loss | adequate | gamma | nu | full-G (p) | full-M (p) | G-M (p) | B7 G (data -0.37) | B7 M (0.08) | B10 (0.205) |
|---|---|---|---|---|---|---|---|---|---|---|
| a s0 | 174 | no (M mean 0.01) | 0 | 4.7 | 0.74 (0.18) | 0.32 (0.71) | 0.36 (0.60) | -0.75 | 0.02 | 0.06 |
| a s1 | 233 | no | 0 | 1.6 | 0.74 (0.24) | 0.15 (0.67) | 0.12 (0.26) | -0.67 | 0.01 | 0.03 |
| **a s2** | **79** | **yes** | 0 | 3.9 | 0.75 (0.22) | **0.59 (0.00)** | **0.72 (0.05)** | -0.64 (out) | 0.14 (in) | **0.27** |
| b s0 | 260 | no (G mean 0.79) | 2.5 | 3.8 | 0.67 (0.67) | 0.57 (0.00) | 0.69 (0.03) | -0.62 | 0.27 | 0.40 |
| b s1 | 477 | no | -4.9 | 4.2 | 0.03 (0.00) | 0.42 (0.02) | 0.06 (0.00) | 0.00 | 0.44 | 0.60 |
| b s2 | 107 | no (G mean 0.71) | 4.9 | 4.7 | 0.61 (0.14) | 0.46 (0.00) | 0.77 (0.00) | -0.67 | 0.20 | 0.31 |

- **One adequate fit (E30a s2, gamma 0).** It matches the basic levels for full, G and M (N-share
  0.29/0.64/0.04 vs 0.31/0.64/0.05; present clones within 10%). Epo is not fitted (mean 1.00 vs
  0.70, 7 vs 23 clones; ungated).
- **Pre-registered reading: no pass.** full-M is 0.59 (data 0.24, p < 0.005) and B7's G difference
  overshoots (-0.64 vs -0.37). The presence-aware family, measured on an adequate fit, fails, but
  **in the opposite direction from E20-E25.**
  - **G-M is no longer too low.** The adequate fit gives 0.72 (data 0.51, p 0.05, borderline
    high), and B10's fraction of both-clones with N under M-CSF is 0.27 (data 0.205; the
    inadequate fits gave 0.02).
  - Presence-coupled selection on an adequate fit can produce strong G-M agreement and the B10
    neutrophils. It then also links the full cocktail to M-CSF too strongly.
- **New anomaly.** The tension has moved. Before: G-M too low on adequate fits without presence
  (E20-E25). Now: full-M too high on an adequate fit with presence. In the data, full and M-CSF
  outcomes agree weakly (0.24) while G and M agree strongly (0.51). Neither family reproduces
  that ordering (G-M > full-M).
- gamma (E30b) did not help: the free-gamma runs were all inadequate (G-CSF N-share too high).
- Converged? No: all runs stopped at 120 generations, and 4 of 6 are far from adequate. A longer
  run is needed before claiming the family's best.
- Held-out shard 1 stays unseen (no pass).

## Batch pre-registration 2026-11-03 (7): E31, B12 (written before either was run)

### E31: E30a converged
- E30 stopped every run at 120 generations. E31 runs E30a (the same model, loss, gate and
  readings; gamma 0) for up to 500 generations. Seeds 0-2 are warm-started (DE x0) from E30a s2,
  the only adequate fit; seeds 3-5 are cold. 6 GitHub Actions jobs. Best = lowest loss among
  adequate fits.
- **Readings as E30.** In addition: if the best E31 fit keeps full-M above the data's range
  (p < 0.05) while G-M passes, the "G-M > full-M ordering" anomaly is confirmed on a converged
  adequate fit.

### B12: what makes full-M low in the data?
- **Facts noted before running.** The full cocktail is almost all day-6 cells (3002 N+M at day 6,
  129 at day 4). G-CSF is split evenly (1312 / 1232). M-CSF is mostly day 6 (864 / 4640). The
  simulators have one time point.
- **(a) Components, data vs the E30a s2 fit (200 simulations).** For each pair (full-M, G-M,
  full-G), among the clones with >= 2 N+M in both conditions: the number of such clones, the
  mean and variance of N-share in each condition, and the pair r. Each data value is flagged if
  outside the model's 95% range.
- **(b) Day split (data only, descriptive).** full-M r with M-CSF day-4 cells only vs day-6
  cells only. G-M r with G-CSF day-4 only vs day-6 only. Each needs >= 15 clones; bootstrap 95%
  CI (2000).
- **Readings.**
  - The full-M components that fall outside E30a's range say where the model over-links full and
    M (for example, too little N-share variance in full, or too many full-M clones).
  - If, in (b), full-M is low only with day-6 M-CSF cells while G-M is high with both G-CSF days,
    the ordering may reflect timing (day-6 M-CSF clones drift away), and a two-time-point model is
    the next structure. If there is no day dependence, timing is not it.
- **Fix before any E31 run.** E30's search space had one dead dimension: 18 for variant a, 19 for b,
  while the model uses 17 / 18 (16 E26b parameters + nu, + gamma). The model ignored that
  coordinate, so E30's results are valid; it only made the search slightly larger. It did
  misplace nu in E31's warm start (smoke-test loss 264 instead of 80.5). Dimensions corrected
  to 17 / 18; the warm start now reproduces E30a s2's loss (80.5).

### B12 results (2026-11-03)

**(a) Components vs the E30a s2 fit (200 simulations; flagged when outside the 95% range).**

| pair | clones in both (model 95%) | N-share, 1st condition | N-share var, 1st | N-share, M or G | r (model 95%) |
|---|---|---|---|---|---|
| full-M | **84** (230-290) below | full 0.14 (0.04-0.08) above | 0.074 above | M 0.043 below | **0.24** (0.43-0.74) below |
| G-M | 78 (78-114) | G 0.35 (0.07-0.19) above | 0.185 above | M 0.13 (in range) | 0.51 (0.51-0.87) |
| full-G | **29** (161-212) below | full 0.40 below | 0.179 below | G 0.37 below | 0.69 (in range) |

- **The largest misfit is how many clones are co-present with the full cocktail:** 3x too
  many in the model for full-M and 6x too many for full-G, while G-M is about right. The
  full-cocktail well and the single-cytokine wells share far fewer N/M-producing clones in the
  data than the model's independent founder placement (k_c ~ Poisson(nu) per well) allows.
- The co-present clones are also a selected set in the data: the full-M clones are more
  neutrophilic in full (0.14 vs 0.04-0.08). B7-type selection applies to the full cocktail too.

**(b) Day split.** full-M: day-4 M-CSF cells give 19 clones (r 0, degenerate); day-6 cells give 80
clones, r 0.25 [-0.12, 0.58]. G-M: G-CSF day 4 gives 53 clones, r 0.40 [0.14, 0.59]; day 6 gives
21 clones, r 0.64 [0.38, 0.87]. No timing effect can be read (the CIs overlap; full has no day-4
cells). **Timing is not supported as the explanation.**

**Post-hoc checks (supporting).** Co-presence among all 5 conditions (>= 2 N+M), data vs
expected under independence (n_a n_b / 3878) vs the E30a fit:
full-G 29 / 105 / 189; full-M 84 / 192 / 264; G-M 78 / 144 / 98; G-SCF 15 / 11 / 58; M-SCF 18 / 21 / 27.
In the data, clones with N/M output in the full cocktail are depleted among those with N/M output
in G-CSF or M-CSF (about 0.3-0.4x of independence). Counted with any cell, overlaps are near or
above independence (full-G 182 vs 105). An "other-fate" explanation (G/M clones making non-N/M
cells in full) is **not** supported: full-cocktail clones co-present with M-CSF have a *lower*
other-fate share in full (0.04 vs 0.31, p 1e-9).

**Reading.** The G-M > full-M ordering comes with a structural fact the models lack: which
clones show N/M output in the full well is anti-associated with N/M presence in the
single-cytokine wells. Candidate mechanisms, all fate-independent at the allocation step:
- **Multinomial founder allocation.** A clone's few day-2 cells are divided among the wells, so
  being in one well makes being in another less likely. E28's independent Poisson draws have no
  such exclusivity.
- **The full well seeded or sampled differently.** It is almost all day 6.

The next model must reproduce the pairwise co-presence table before any cross statistic is read.

## E32: multinomial founder allocation + pairwise co-presence (pre-registered 2026-11-03)

- **Data diagnostic it must explain (retro rule).** B12: N/M co-presence between the full
  cocktail and the single-cytokine wells is 0.3-0.4x of independence in the data, and the E30a
  fit gives 3-6x too much.
- **Why E28/E30 cannot do it.** Independent Poisson founders per well are the Poisson thinning
  of a Poisson total, so presence is independent across wells. Exclusivity needs a total that is
  under-dispersed: a clone with one day-2 founder cell can only be in one well.
- **Model.** E30a (E26b instruction + selection, rho_f 0.88, expansion trait, gamma 0) with
  founders allocated per clone: K = 1 + Poisson(lambda_K) day-2 founders, each placed in well c
  with probability w_c (softmax; full fixed as reference). Each founder's output is as in E30.
  Parameters: 16 + lambda_K + 4 well logits = 21.
- **Loss.** E30's SE-scaled targets plus the 10 pairwise co-presence counts (>= 2 N+M in both;
  SE sqrt(max(n, 4))). Differential evolution, population 6 x d, up to 300 generations, 6 seeds
  (Actions).
- **Adequacy gate.** E30's gate (full, G, M mean N-share in the data CI; present clones ±20%),
  plus full-G, full-M and G-M co-presence each within ±30% of the data (29, 84, 78).
- **Readings (adequate fits only, as E30).** A pass needs G-M, full-G and full-M p >= 0.05
  and both B7 differences inside the 95% range; B10 is reported. The key test is whether getting
  co-presence right also brings full-M down to the data (0.24) while keeping G-M near 0.51.
  - If yes, the ordering anomaly is explained by founder allocation plus selection, and the
    power rule and the shard-1 look follow.
  - If the co-presence fits but full-M stays high, allocation is not the cause of the ordering.

## B13: cytokine receptors and fate regulators at day 2 (pre-registered 2026-11-03)

- **Question.** Is the day-2 selection state (B9, B11) carried by the cytokine receptors
  themselves? G-CSF signals through Csf3r and M-CSF through Csf1r. Downstream fate regulators:
  Cebpa and Cebpe (neutrophil), Spi1/PU.1 and Irf8 (monocyte).
- **Data.** The cytokine experiment's normalised counts (stateFate_cytokinePerturbation, same
  LARRY source), day-2 cells only, clone mean of log1p(normalised expression) over each
  clone's day-2 cells. The same clone groups as B7/B9/B11.
- **Genes.** Primary: Csf3r and Csf1r, plus the receptor balance Csf3r - Csf1r (difference of the
  clone means). Secondary panel: Csf2rb, Il3ra, Kit, Stat3, Cebpa, Cebpb, Cebpe, Gfi1, Spi1,
  Irf8, Klf4, Mafb, Elane, Mpo, Ly86, Ccr2.
- **Predictions (primary, each one-sided at p < 0.05).**
  1. Receptor balance is higher in G-only clones than in M-only clones (Mann-Whitney).
  2. Csf3r is higher in clones that make >= 1 neutrophil under M-CSF than in M-present clones that
     do not: partial Spearman given log M-CSF clone size, the B11 v3 method.
  3. Among clones present under G-CSF, receptor balance correlates positively with the G-CSF
     N-share (Spearman).
- **Secondary.** The same three tests for every panel gene, with BH correction within each test
  (FDR 10%). Also: does the B11 priming-score effect survive after controlling for the receptor
  balance (partial on log size + balance)? If it drops a lot, the receptors carry the priming.
- **Readings.**
  - Primary 1-3 supported: the selection runs through cytokine receptor levels that are set
    per clone before the split. That gives a measurable clone trait for the next model
    (receptor balance as an input instead of a latent).
  - Primaries fail but regulators pass: the state is set downstream of the receptors.
  - Nothing passes: the day-2 priming is a broader transcriptional state not captured by these
    genes.
- Spent data only (the same clones as B7/B9/B11); no held-out data.

### B13 results (2026-11-03)

2222 clones with day-2 cells. Primary (receptor balance = Csf3r - Csf1r, clone mean of
log1p expression):

| test | Csf3r | Csf1r | **balance** |
|---|---|---|---|
| 1. G-only > M-only (one-sided MW p) | 0.53 | 1.00 (opposite: higher in M-only) | **0.006** |
| 2. N under M-CSF, partial rho given size (p) | 0.11 (0.024) | -0.12 | **0.14 (0.005)** |
| 3. G-CSF N-share, rho (p) | 0.13 (0.038) | -0.16 | **0.19 (0.006)** |

- **All three pre-registered primaries are supported.** Clones with more G-CSF receptor relative
  to M-CSF receptor at day 2 are found only under G-CSF more often, keep making neutrophils under
  M-CSF, and are more neutrophilic under G-CSF. Csf1r (higher in M-only clones, lower in
  N-makers) carries as much of it as Csf3r. The effects are modest (rho 0.14-0.19).
- **The receptors carry little of the priming.** The B11 priming-score effect goes from 0.24 to
  0.21 (p 1e-4) after adding the receptor balance. The day-2 state is broader than receptor levels.
- **Panel (BH q < 0.1, two-sided).** The neutrophil programme predicts every outcome, and the
  monocyte programme predicts the reverse.
  - Neutrophil genes, positive. Cebpa: N under M partial 0.28, G N-share 0.27, G-only > M-only,
    all q < 0.002. Gfi1: 0.26 / 0.22. Cebpe, Elane, Mpo: G N-share 0.25-0.31, and higher in G-only
    clones. Kit: 0.22 for N under M.
  - Monocyte genes, negative. Irf8, Spi1/PU.1, Ccr2, Cebpb, Ly86: about -0.12 to -0.20.
  - **The strongest single predictor of neutrophils under M-CSF is Cebpa (0.28)**, above the
    priming score (0.24) and far above the receptors (0.11-0.14).
- **Reading (pre-registered branch).** The primaries passed, but most of the signal is
  downstream: the selection state is the clone's day-2 level of the neutrophil transcription
  programme (Cebpa/Gfi1, then granule genes), with receptor balance a minor part. Measured
  clone trait for the next model: day-2 Cebpa (or a neutrophil-programme score) as an input in
  place of the latent z.

### E32 results (2026-11-03)

300 generations, 6 seeds; none reported converged. **No fit is adequate**, so by the
pre-registered rule there is no reading of the cross statistics. Reported for the record:

| seed | loss | lambda_K | w (full, G, M, Epo, SCF) | present full / G / M (data 368/277/478) | co-presence full-G / full-M / G-M (data 29/84/78) | full-G / full-M / G-M (p) | B7 G |
|---|---|---|---|---|---|---|---|
| 0 | 150 | 2.1 | .09 .30 .27 .07 .28 | 288 / 251 / 426 | 71 / 116 / 53 | 0.75 (.42) / 0.56 (.06) / 0.78 (.05) | -0.55 |
| 2 | 131 | 4.7 | .04 .17 .57 .18 .05 | 243 / 260 / 446 | 66 / 107 / 64 | 0.76 (.35) / 0.55 (.21) / 0.78 (.07) | -0.60 |
| 5 | 177 | 4.5 | .03 .16 .32 .27 .21 | 243 / 217 / 435 | 55 / 104 / 47 | 0.76 (.38) / 0.47 (.28) / 0.76 (.16) | -0.61 |
| 1, 3, 4 | 150-225 | | | | | | |

- **Why it fails the gate.** Multinomial allocation cuts co-presence with the full well
  (full-G from E30a's 189 to 52-71; full-M from 264 to 101-116), but only by giving the full
  well a tiny share of founders (w_full 0.03-0.09). Then the full well has too few present
  clones (243-288 vs 368). The model cannot have both many full-well clones and few clones
  shared with G/M from one shared founder pool.
- **What that implies about the data.** The full-cocktail well's clones are largely a
  *different set* from those in the single-cytokine wells, more than any split of a clone's
  founders can produce. Candidate: the full condition drew from a different or larger pool of
  founders, for example sampled from the bulk culture at day 6 rather than from the same day-2
  split. This is a design question about the experiment, not a fate mechanism. It needs the
  methods text of Weinreb 2020 (supplement not retrieved).
- **Side observation (inadequate fits, so not a reading).** With co-presence closer to the data,
  full-M drops from 0.59 (E30a) to 0.47-0.56. In three seeds all three cross statistics have
  p >= 0.05, but B7 G still overshoots (-0.55 to -0.61 vs -0.37). The direction is consistent
  with B12: the full-M excess in E30a was mostly a co-presence artefact.
- Held-out shard 1 stays unseen.

## B14: committed-like cells per clone at day 2 (pre-registered 2026-11-03)

- **Motivation.** Literature sweep 3: proNeu1 (Kwok 2020) is a neutrophil-only progenitor
  that ignores M-CSF and is expanded by G-CSF. Hypothesis: the G-M sister agreement is carried
  by each clone's committed minority, not by a graded bias of all its cells.
- **Cell classes (day-2 cells; thresholds fixed before looking at outcomes).**
  - Per cell: neutrophil-programme score = mean of z-scored log1p(Gfi1, Cebpe, Elane, Mpo);
    monocyte-programme score = mean of z-scored log1p(Irf8, Ly86, Ccr2, Csf1r). z-scores are
    over all day-2 cells.
  - **N-committed-like:** annotated Neu, or Undifferentiated with neutrophil score >= its 90th
    percentile among Undifferentiated cells and Csf1r = 0.
  - **M-committed-like:** annotated Mo, or Undifferentiated with monocyte score >= its 90th
    percentile and Gfi1 = 0.
  - Per clone: frac_N, frac_M = class share among the clone's day-2 cells.
- **Tests.**
  1. **N under M-CSF:** partial Spearman of frac_N vs has-N given log size (the B11 v3 method).
     Then the joint test: frac_N given log size + clone-mean Cebpa, and Cebpa given log size +
     frac_N.
  2. **Presence:** frac_N higher in G-only than M-only clones, frac_M the reverse
     (one-sided MW).
  3. **G-M agreement (key).** Among both-clones with day-2 cells: Spearman of G-CSF vs M-CSF
     N-share, raw and partial given frac_N and frac_M. Also the same partial given clone-mean
     Cebpa and Irf8 (the graded alternative).
- **Readings.**
  - "Committed minority carries the agreement": test 3's partial r drops by >= 30% given frac_N/M,
    and by more than given Cebpa/Irf8.
  - "Graded programme carries it": the Cebpa/Irf8 partial drops more.
  - Neither drops: the agreement is carried by something these day-2 genes do not capture.
  - n is small (both-clones with day-2 cells, about 39). Report CIs (2000 bootstraps), and read a
    difference only if the drop CI excludes 0.
- **For E33.** If the committed reading wins, E33 = a mixture of instructed bipotent cells and
  cytokine-insensitive committed cells, with the measured frac_N / frac_M per clone as inputs.
  If the graded reading wins, E33 uses clone-mean Cebpa/Irf8 as a measured bias instead of the
  latent z.
- Spent data only.

### B14 results (2026-11-03)

2222 clones with day-2 cells; 2026 N-committed-like and 2730 M-committed-like day-2 cells.

- **Test 1 (N under M-CSF, 321 clones).** frac_N partial 0.18 (p 0.001); Cebpa 0.28 (2e-7). Joint:
  frac_N given Cebpa 0.14 (0.01); **Cebpa given frac_N 0.26 (2e-6)**. The graded Cebpa level
  carries more than the committed fraction, though both contribute.
- **Test 2 (presence).** **frac_N: G-only 0.25 vs M-only 0.05 (p 3e-10); frac_M: M-only 0.16 vs
  G-only 0.04 (p 1e-5).** Committed-like cells strongly mark which single-cytokine well a clone
  is seen in. This is the strongest presence predictor so far.
- **Test 3 (G-M agreement, 39 both-clones with day-2 cells; only 14 have any committed cell).**
  Raw 0.39. Partial given committed fractions 0.30 (drop 0.08, CI -0.06 to 0.33). Partial given
  Cebpa/Irf8 0.28 (drop 0.11, CI -0.04 to 0.31). **Pre-registered reading: "neither"**: both
  drops are about 20-30% and their CIs include 0. Underpowered (n 39).
- **Summary.** Committed cells decide *where* a clone is seen (presence). The graded neutrophil
  programme (Cebpa) decides *how many* neutrophils it makes under M-CSF. Neither alone explains
  the G-M agreement within the 39 measurable clones.

## E33: measured day-2 inputs in the presence-aware model (pre-registered 2026-11-03)

- **Design from B14 (B14 reading "neither", so both measured inputs go in, as nested terms).**
  Base: E30a (E26b instruction + selection, founders k_c ~ Poisson(nu), expansion trait,
  rho_f 0.88, gamma 0; SE-scaled loss, differential evolution, adequacy gate).
  - **Graded bias, measured.** The clone latent becomes z = w * m + sqrt(1 - w^2) * e. Here m is
    the clone's standardised day-2 score (clone-mean Cebpa minus clone-mean Irf8, from
    b14_clone_day2.csv), and w in [0, 1] is fitted. w = 0 is E30a.
  - **Committed cells, measured.** A share kappa * frac_N of each founder's output is neutrophil
    regardless of cytokine, and kappa * frac_M is monocyte regardless (shares capped at 1 in
    total). frac_N and frac_M are the B14 per-clone fractions; kappa in [0, 2] is fitted.
    kappa = 0 is E30a. Committed N still gets the condition's survival factor s_c.
  - Clones without day-2 cells (1656 of 3878) draw (m, frac_N, frac_M) jointly from the
    measured clones.
- **Full-cocktail caveat (E32).** The full well's clone set is not reproducible by founder
  splitting. Full stays in the loss, but its present-clone count and co-presence are
  **not gated**, and full-G / full-M are secondary readings.
- **Gate.** Mean N-share inside the data 95% CI for full, G and M; present clones within ±20% for
  G and M.
- **Readings (adequate fits, best of 6 seeds, 300 generations, Actions).**
  - **Primary pass:** G-M p >= 0.05, and both B7 differences inside the 95% range.
  - Reported: B10, full-G, full-M, the fitted w and kappa.
  - **"Measured inputs identified":** w > 0.2 or kappa > 0.2 in the best adequate fit, with
    the loss below E30a's best (79.2, same scale).
  - A pass with identified inputs is the first model in which measured day-2 state, not a free
    latent, carries the G-M agreement. Power rule, then the shard-1 look.
  - If inputs are identified but G-M or B7 fail, the measured state is relevant but
    insufficient. If w and kappa go to about 0, the day-2 measurements add nothing beyond the
    latent at this level.

### E31 results (2026-11-03)

E30a for 500 generations (still not formally converged), 6 seeds. **All 6 are adequate** (basic
levels and present clones match for full, G and M).

| seed | warm | loss | full-G (p) | full-M (p) | G-M (p) | B7 G (data -0.37) | B7 M in | B10 (0.205) |
|---|---|---|---|---|---|---|---|---|
| 0 | yes | 41.2 | 0.76 (0.10) | 0.60 (0.00) | 0.72 (0.01) | -0.63 | no | 0.29 |
| 1 | yes | 42.1 | 0.75 (0.13) | 0.58 (0.00) | 0.69 (0.06) | -0.66 | yes | 0.22 |
| 2 | yes | 41.3 | 0.76 (0.12) | 0.59 (0.00) | 0.70 (0.02) | -0.66 | yes | 0.25 |
| 3 | no | 90.2 | 0.77 (0.04) | 0.60 (0.00) | 0.72 (0.04) | -0.66 | yes | 0.26 |
| 4 | no | 46.7 | 0.75 (0.12) | 0.62 (0.00) | 0.72 (0.01) | -0.64 | no | 0.26 |
| **5 (best)** | no | **23.8** | 0.75 (0.10) | **0.61 (0.00)** | **0.70 (0.05)** | **-0.66** (out) | no | 0.27 |

- **Pre-registered reading: the ordering anomaly is confirmed on adequate, near-converged fits.**
  Every adequate fit has full-M about 0.6 (data 0.24, p < 0.005), G-M 0.69-0.72 (data 0.51),
  and B7's G difference about -0.65 (data -0.37). It is identical across 6 seeds, so it is a
  property of the model family, not of the optimiser. **E30a fails.** The presence-aware family
  over-links all conditions and over-selects.

### Cytokine design read from the data (2026-11-03)

The supplement (Supp Fig 7) could not be retrieved (PMC placeholder, Science 403). Library names
in the metadata show the design:
- Day 2: six libraries, all "full cocktail" (the pre-split pool).
- G-CSF and M-CSF: three day-4 and two day-6 libraries each. Epo: two day-4 and three day-6.
  SCF: three day-4 only.
- Full cocktail after day 2: one small day-4 library (802 cells, named "all") and four day-6
  libraries ("d6_all_1-4").

The full arm is labelled differently and sampled almost only at day 6, consistent with a
continued bulk culture rather than a well from the same split as the single cytokines. That fits
E32 (its clone set cannot come from the same founder split). **Treat "full" as a separate arm in
future models.** Not confirmed from the methods text.

## Batch pre-registration 2026-11-03 (8): B15, B16 (written before either was run)

### B15: how much of the selection state is readable from day-2 expression?
- **Data.** Cytokine experiment, day-2 cells. Top 2000 variable genes (log1p normalised), 30
  PCs, clone mean. Targets:
  - (a) G-only vs M-only presence (138 vs 282 clones with day-2 cells);
  - (b) >= 1 N under M-CSF among M-present clones (321, 27 positive);
  - (c) G-CSF N-share among G-present clones (177; regression).
- **Models.** L2 logistic regression (or ridge for c), 5-fold CV by clone, repeated 20 times.
  Compared with Cebpa-only and with the B9 priming score alone.
- **Metric.** AUC (a, b) or Pearson r (c) of the out-of-fold predictions, with bootstrap 95% CI
  of the difference to Cebpa-only (2000).
- **Reading.** "More is readable": full-expression AUC/r exceeds Cebpa-only by >= 0.05 with the CI
  of the difference excluding 0. Otherwise "Cebpa captures the readable part".

### B16: does the Cebpa - Irf8 day-2 score replicate in the main in vitro LARRY data?
- **Data.** In vitro LARRY (one cocktail, two wells), day-2 cells
  (day2_normed_triplets.txt.gz). Clones with >= 1 day-2 cell and >= 3 day-4/6 N+M cells.
- **Test.** Partial Spearman (rank residuals) of the clone-mean Cebpa - Irf8 score with the
  clone's day-4/6 N-share, given log(N+M) and log(day-2 cells). Secondary: Cebpa alone, Irf8
  alone, Gfi1 alone.
- **Prediction.** Positive with p < 0.05. Effect reported next to the cytokine data's (B13:
  Cebpa 0.27 and Irf8 -0.20 for the G-CSF N-share).
- **Reading.** Supported means the day-2 neutrophil/monocyte programme predicts fate across
  experiments (a replication of B13 on independent clones). Not supported means B13 is specific
  to the cytokine experiment.

### B16 results (2026-11-03)

Main in vitro LARRY (one cocktail), 571 clones with day-2 cells and >= 3 day-4/6 N+M cells.
Partial Spearman with the clone's N-share, given log(N+M) and log(day-2 cells):

| day-2 clone mean | partial rho | p |
|---|---|---|
| **Cebpa - Irf8** | **0.30** | 2e-13 |
| Cebpa | 0.23 | 4e-8 |
| Irf8 | -0.18 | 2e-5 |
| Gfi1 | **0.34** | 9e-17 |

- **Supported (pre-registered): B13 replicates on independent clones in a different
  experiment.** The day-2 neutrophil programme predicts the clone's later neutrophil share
  with the same signs and similar sizes as in the cytokine experiment (B13: Cebpa 0.27, Irf8
  -0.20 for the G-CSF N-share). Gfi1 is the strongest single gene here (0.34), matching Olsson
  2016's Gfi1-Irf8 axis.
- **Consequence.** The measured day-2 programme is a general clone trait, not a cytokine-experiment
  artefact. It is a usable measured input for models on both datasets (as E33 does).

### B15 results (2026-11-03)

Out-of-fold performance, 5-fold CV by clone, 20 repeats. Brackets give the bootstrap 95% CI of
the difference to Cebpa-only.

| target | n (positives) | full expression (30 PCs) | Cebpa only | priming score | full + Cebpa | reading |
|---|---|---|---|---|---|---|
| (a) G-only vs M-only, AUC | 420 (138) | **0.81** [+0.20, +0.34] | 0.54 | 0.77 [+0.15, +0.31] | 0.81 | **more readable** |
| (b) N under M-CSF, AUC | 321 (27) | 0.76 [-0.06, +0.27] | 0.66 | 0.72 | 0.77 | Cebpa enough (underpowered) |
| (c) G-CSF N-share, r | 177 | 0.36 [-0.01, +0.36] | 0.19 | 0.12 | 0.38 | Cebpa enough (borderline) |

- **(a) Where a clone is seen is well readable from day-2 expression (AUC 0.81), far beyond
  Cebpa (0.54).** The presence-deciding state is broad. The B9 SPRING priming score already
  captures most of it (0.77), and B14's committed fractions are part of it.
- **(b, c) Fate under a given cytokine.** Full expression is numerically better (0.76 vs 0.66;
  r 0.36 vs 0.19), but the pre-registered rule is not met: the CIs include 0, with only 27
  positives in (b). By the rule, Cebpa captures the readable part.
- **Readable ceiling so far.** Presence: AUC about 0.8. Neutrophil output: AUC about 0.75 or
  r about 0.36. Much of the clone's fate state is visible at day 2, but not all (compare B1c:
  about half of the variance unexplained).

### E33 results (2026-11-03)

300 generations, 6 seeds. 5 of 6 adequate. Measured inputs identified in every fit.

| seed | loss | adequate | w (graded) | kappa (committed) | G-M (p) | full-G (p) | full-M (p) | B7 G (data -0.37) | B7 M in | B10 (0.205) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 79.0 | yes | 0.49 | 1.01 | 0.54 (0.67) | 0.81 (0.00) | 0.61 (0.00) | -0.60 | no | 0.26 |
| 1 | 100.4 | yes | 0.44 | 0.91 | 0.51 (0.98) | 0.78 (0.04) | 0.56 (0.00) | -0.54 | no | 0.27 |
| 2 | 97.0 | yes | 0.76 | 0.13 | 0.54 (0.75) | 0.70 (0.67) | 0.53 (0.01) | -0.58 | yes | 0.25 |
| **3 (best)** | **30.5** | yes | 0.59 | 0.94 | **0.49 (0.87)** | 0.80 (0.00) | 0.57 (0.00) | **-0.58** (out) | yes | 0.25 |
| 4 | 109.2 | no | 0.42 | 1.21 | 0.56 | 0.81 | 0.62 | -0.62 | no | 0.24 |
| 5 | 96.0 | yes | 0.71 | 0.18 | 0.38 (0.20) | 0.70 (0.80) | 0.49 (0.02) | -0.53 | yes | 0.24 |

- **Measured inputs identified (pre-registered).** w is 0.42-0.76 and kappa 0.13-1.21 in every
  fit, and the best adequate loss is 30.5 against E30a's 79.2 on the same loss. The fit uses the
  measured day-2 state. Two solutions appear: committed-heavy (kappa about 1, w about 0.5) and
  graded-heavy (w about 0.75, kappa about 0.15).
- **G-M is now right on the data.** All adequate fits give 0.38-0.54, against 0.51
  (p 0.20-0.98). E30a/E31 without measured inputs overshot (0.69-0.72). **This is the first
  presence-aware model whose adequate fits reproduce the G-M agreement**, and it does so through
  measured day-2 Cebpa/Irf8 and committed-cell fractions rather than a free latent.
- **Primary pass: no.** B7's G difference still overshoots in every fit (-0.53 to -0.60 vs
  -0.37). B7 M is inside in three fits. B10 is close (0.24-0.27 vs 0.205).
- **Secondary (full arm, caveat).** full-M stays at 0.49-0.61 (data 0.24). This is consistent
  with the full arm being a separate culture (design note above), which the model still treats
  as a split well.
- Held-out shard 1 stays unseen (no pass). The shard-1 predictions vary from 0.0 to 0.77 across
  seeds, so the Epo/SCF pooled statistic is unstable under this model. Power is poor whatever
  the model, so it is a weak test.

**Where this leaves the G-M question.** Measured day-2 fate state plus cytokine-dependent
survival of neutrophils (s_c) and clone-specific founder presence reproduces the G-CSF vs M-CSF
sister agreement. What remains wrong: (1) the model over-selects under G-CSF (B7 G); (2) it treats
the full arm as a sister well.

## Batch pre-registration 2026-11-03 (9): E34, B17 (written before either was run)

### E34: E33 with the full arm as a separate culture
- **Motivation.** E32 and the library names: the full arm (labelled "all", almost only day 6)
  does not behave like a well from the same founder split. E33 still treats it as one, and
  full-M stays at about 0.55 (data 0.24).
- **Model.** E33 (measured w, kappa; instruction + selection; rho_f 0.88), except for the full
  arm:
  - its own founder rate nu_full (k_full ~ Poisson(nu_full)), independent of the others;
  - its own expansion trait a_full = rho_a * a + sqrt(1 - rho_a^2) * e, with rho_a in [0, 1]
    fitted.
  The fate latent is still shared, because it is the same clone. 21 parameters.
- **Fit.** E33's loss, gate and readings; differential evolution, 500 generations, 6 seeds
  (Actions).
- **Readings.**
  - **Primary pass:** adequate, G-M p >= 0.05, both B7 differences inside the 95% range.
  - **Full-arm reading (new, primary for this question):** in adequate fits, is full-M back
    inside the data range (p >= 0.05)? Supported if full-M p >= 0.05 in the best adequate fit,
    with G-M still p >= 0.05. Then the G-M > full-M ordering is explained by the full arm's
    design, not by fate biology.
  - If full-M stays high, the full arm's separate presence is not enough. The full-M anomaly is
    then a fate effect of the full cocktail (for example, the cocktail dilutes the clone bias
    more than single cytokines do).

### B17: why does the model over-select under G-CSF (B7 G -0.58 vs -0.37)?
- **Comparison.** Data vs E33 best fit (e33_s3), 200 simulations, G-present clones (>= 2 N+M):
  - number of G-only and both clones;
  - mean and SD of G-CSF N-share in each group;
  - the fraction of G-only clones that are all-N;
  - mean log size under G-CSF per group.
  Each component is flagged when the data fall outside the model's 95% range.
- **Reading.** The flagged components say whether the overshoot comes from G-only clones being
  too neutrophilic (survival s_G or committed-N too strong), from both-clones being too
  monocytic, or from presence (group sizes). That says which E33 term to change. Diagnostic
  only; no held-out data.

### B17 results (2026-11-03)

Data vs the E33 best fit (e33_s3), 200 simulations; flagged when outside the model's 95% range.

| component | data | model (95%) | flag |
|---|---|---|---|
| G-only clones | 474 | 410 (381-440) | data above |
| both clones | 78 | 113 (92-135) | data below |
| G-only mean N-share | 0.72 | 0.79 (0.75-0.83) | data below |
| G-only all-N fraction | 0.62 | 0.72 (0.68-0.76) | data below |
| both mean N-share | **0.35** | **0.21** (0.14-0.28) | data above |
| both all-N fraction | 0.24 | 0.15 (0.09-0.22) | data above |
| log size (either group) | | | in range |

- **The overshoot comes from both groups.** G-only clones are too neutrophilic in the model, and
  both-clones too monocytic, so the gap between them is too wide (-0.58 vs -0.37). There are also
  too many both-clones.
- **Cause, read from the fitted parameters: instruction and selection traded off to extremes.**
  - G-CSF: survival s_G = 9.2 with instruction tau_G = -9.0. The instruction pushes toward
    monocytes, and survival then rescues the neutrophils.
  - M-CSF: s_M = 0.03 with tau_M = +2.8. Neutrophil-directed instruction, then neutrophil death.
  - Epo and SCF: the same pattern (tau about -7 to -9, s about 5-6).

  The model reproduces each condition's N-share by selection alone. The instruction terms have
  the **opposite sign** to the literature: G-CSF and M-CSF instruct their own lineages in single
  GMPs (Rieger 2009) and through the PU.1:Cebpa ratio (Dahl 2003). Selection that strong makes
  presence too fate-dependent, which is the B7 overshoot.
- **Implication.** Instruction (tau) and selection (s) are not separately identified by these
  statistics. The fit chose a biologically wrong corner. Next: constrain the instruction signs
  from the literature (tau_G >= tau_full >= tau_M) and see whether B7 falls into range (E35).

### Power check for the held-out look (2026-11-03)

E33's predicted shard-1 statistic (pooled Epo/SCF concordance) by seed: 0.43, 0.00 [0, 0],
0.03, 0.77, 0.00 [0, 0], 0.00. Several fits make the statistic degenerate (too few Epo/SCF clone
pairs, so r = 0). Fits of equal quality predict anything from 0 to 0.77. The observed shard-0 CI
half-width is about 0.37. **The held-out statistic cannot separate these models.** Spending it now
would be uninformative, so the look is skipped by the power rule. It stays unseen for a model
whose shard-1 prediction is stable across seeds.

## E35: E33 with literature-signed instruction (pre-registered 2026-11-03)

- **Motivation (B17).** E33 fits every condition by selection alone, with instruction of the
  wrong sign (tau_G -9, tau_M +2.8). Rieger 2009 and Dahl 2003: G-CSF instructs neutrophil and
  M-CSF monocyte fate.
- **Model.** E33 unchanged, except tau_G = tau_full + 8 * sigmoid(x_G) and
  tau_M = tau_full - 8 * sigmoid(x_M). So tau_G >= tau_full >= tau_M; G-CSF instructs toward
  neutrophils and M-CSF toward monocytes relative to the full cocktail. Epo and SCF taus are
  free. Survival factors are free as before. 19 parameters, 500 generations, 6 seeds (Actions);
  runs in parallel with E34.
- **Readings (E33's gate and readings).**
  - **Primary pass:** adequate, G-M p >= 0.05, both B7 differences inside the 95% range.
  - Also: does B7 G move toward -0.37 relative to E33 (-0.58)? And the loss vs E33's best (30.5).
  - Pass: the over-selection was an identifiability artefact, and literature-signed instruction
    plus measured state plus selection explains G-M and B7.
  - B7 fixed but G-M lost: instruction and selection cannot both be right in this structure.
  - Constraint costs much loss (an inadequate fit): the data prefer selection-dominated fits,
    against the literature, which would itself be notable.

### E34 results (2026-11-03)

500 generations, 6 seeds; 5 adequate.

| seed | loss | adequate | nu_full | rho_a | full-G (p) | full-M (p) | G-M (p) | B7 G | B7 M in | B10 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 91.7 | yes | 2.1 | 0.23 | 0.82 (0.02) | 0.55 (0.09) | 0.59 (0.43) | -0.61 | yes | 0.21 |
| 1 | 89.5 | yes | 1.5 | 0.68 | 0.77 (0.10) | 0.52 (0.03) | 0.44 (0.47) | -0.54 | yes | 0.27 |
| 2 | 30.4 | no | 1.0 | 0.85 | 0.83 (0.00) | 0.59 (0.00) | 0.54 (0.75) | -0.61 | yes | 0.24 |
| 3 | 96.1 | yes | 4.6 | 0.23 | **0.72 (0.70)** | **0.39 (0.30)** | **0.35 (0.07)** | -0.49 | yes | 0.24 |
| 4 | 83.6 | yes | 1.3 | 0.94 | 0.82 (0.00) | 0.54 (0.05) | 0.46 (0.62) | -0.59 | yes | 0.24 |
| **5 (best adequate)** | **51.6** | yes | 4.7 | 0.81 | 0.79 (0.01) | **0.55 (0.00)** | 0.47 (0.67) | -0.55 | yes | 0.26 |

- **Primary pass: no.** B7 G overshoots in every fit (-0.49 to -0.61).
- **Full-arm reading (pre-registered on the best adequate fit): not supported.** Seed 5 keeps
  full-M at 0.55 (p < 0.005). One higher-loss seed (3) has all three cross statistics inside the
  data range, with the ordering G-M 0.35 / full-M 0.39 / full-G 0.72, by decoupling the full
  arm's expansion trait (rho_a 0.23). That shows the structure can produce the ordering, but the
  best fit does not choose it. Not a result.
- **Gains over E33.** B7 M is now inside in all fits, B10 is right (0.21-0.27), and G-M stays on
  the data (0.44-0.59).
- **Convergence.** E34 nests E33 (rho_a = 1, nu_full = nu), but most seeds reach losses (84-96)
  far above E33's 30.5. The 21-dimensional search is not converged, so the seed-to-seed spread
  in full-M is partly optimiser noise.
- **Remaining blocker: B7 G, the over-selection under G-CSF (B17).** E35 (signed instruction) is
  running.

## E36: combined model (pre-registered 2026-11-03, before E35's results are seen)

- **Model.** E34 (E33's measured inputs, full arm with its own founder rate and expansion trait)
  plus:
  - (a) **full-arm co-presence in the loss.** Clones present in both full and G (29 in the
    whole data), and in both full and M (84), counted on the fit set, with SE
    sqrt(max(n, 4)). These are structural statistics (B12), not the cross r we test on.
  - (b) 10 simulations per loss evaluation (was 5).
  - (c) warm starts. Seeds 0-2 start from the best earlier fit mapped into the 21-dim space
    (rho_a 0.99, nu_full = nu); seeds 3-5 are cold. Differential evolution, 500 generations.
- **Decision rule for the instruction-sign constraint, fixed now.** Include E35's constraint
  (tau_G >= tau_full >= tau_M) **if and only if** E35's best adequate fit has a B7 G difference
  closer to the data (-0.37) than E33's best (-0.58), i.e. above -0.58. If included, warm-start
  from that E35 fit; otherwise from e33_s3.
- **Gate.** E33's gate, plus full-G and full-M co-presence within ±30% of the data (29, 84).
- **Readings (best adequate fit).**
  - **Primary pass:** G-M, full-G and full-M all p >= 0.05, and both B7 differences inside the
    95% range.
  - **Stability:** a result counts only if at least 3 of 6 seeds are adequate and agree, with
    every cross statistic's pass/fail the same in each of them.
  - A stable pass is the first model consistent with all the structure. The power check for the
    held-out look follows: a stable shard-1 prediction across the adequate seeds, differing from
    E20's by more than about 0.37.
- Note: the warm start's rho_a is 0.953, not 0.99, because x is clipped to [-3, 3].

### E35 results (2026-11-03)

500 generations, 6 seeds; 4 adequate. Instruction constrained to the literature signs.

| seed | loss | adequate | tau G / full / M | s_G / s_M | G-M (p) | full-G (p) | full-M (p) | B7 G (data -0.37) | B7 M | B10 |
|---|---|---|---|---|---|---|---|---|---|---|
| **0 (best)** | **22.5** | yes | -4.1 / -4.6 / -5.3 | 4.3 / 0.11 | **0.46 (0.33)** | 0.75 (0.23) | 0.58 (0.00) | **-0.30 (in)** | in | 0.18 |
| 1 | 204 | no | | 5.8 / 0.05 | 0.32 (0.00) | | 0.46 | -0.46 (in) | in | 0.11 |
| 2 | 45.5 | yes | -4.7 / -5.3 / -6.6 | 4.4 / 0.15 | 0.44 (0.22) | 0.71 (0.72) | 0.55 (0.00) | -0.22 (out) | in | 0.20 |
| 3 | 70.5 | yes | -4.5 / -5.6 / -6.2 | 4.1 / 0.11 | 0.41 (0.08) | 0.72 (0.55) | 0.53 (0.00) | -0.27 (out) | in | 0.17 |
| 4 | 67.3 | yes | -5.0 / -5.7 / -6.3 | 4.1 / 0.12 | 0.44 (0.23) | 0.72 (0.56) | 0.55 (0.00) | -0.26 (out) | in | 0.19 |
| 5 | 143 | no | | 7.2 / 0.06 | 0.33 (0.00) | | 0.47 | -0.45 (in) | in | 0.11 |

- **Pre-registered primary: PASS in the best adequate fit (seed 0).** It is adequate, G-M 0.46
  (p 0.33), and both B7 differences are inside the 95% range (G -0.30, M in). It is also the lowest
  loss of the whole E33 family (22.5 vs E33's 30.5), despite the constraint. **First model to pass
  the G-M and B7 gates together.**
- **The over-selection was an identifiability artefact.** With the instruction signs fixed by the
  literature, B7 G moves from -0.58 (E33) to -0.22 to -0.30, and the fit improves. The fitted
  biology is **modest instruction** (tau_G about 0.6-1.1 above full, tau_M about 0.6-1.3 below)
  **plus strong selection** (neutrophil survival about 4x under G-CSF, about 0.1x under M-CSF),
  acting on a **measured day-2 state** (w 0.1-0.7, kappa 0.9-1.7). Rieger's instruction and B7's
  selection both hold.
- **Caveats.**
  - Stability: only seed 0 passes. Seeds 2-4 are adequate but undershoot B7 G slightly (-0.22 to
    -0.27, outside the range), so the B7 window is narrow and the pass is not yet seed-stable.
  - full-M stays at 0.53-0.58 (the full arm is not separate in E35; E36 adds that).
  - Epo/SCF survival factors run to the bounds in some seeds (about 240-360, unconstrained and
    ungated).
- **Power note.** Seed 0's shard-1 prediction is 0.52 [0.25, 0.78]. E36 decides whether a
  stable prediction exists across seeds.
- **E36 decision rule applied.** B7 G -0.30 > -0.58, so E36 includes the sign constraint and
  warm-starts from e35_s0.

### E36 results (2026-11-03)

Signed instruction (decision rule applied), 500 generations, 10 simulations per evaluation, 6 seeds
(0-2 warm from e35_s0, 3-5 cold). 5 of 6 pass E33's gate; **0 of 6 pass the full-arm co-presence
gate**. Co-presence is computed from the saved z-scores (sim = obs + z * sqrt(max(obs, 4))) against
the fit set's counts: full-G 16, full-M 33.

| seed | loss | E33 gate | co full-G (data 16) | co full-M (data 33) | full-G (p) | full-M (p) | G-M (p) | B7 G (data -0.37) | B7 M | rho_a | shard-1 pred |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 113.5 | yes | 31 (+95%) | 55 (+67%) | 0.73 (0.49) | 0.59 (0.01) | 0.48 (0.57) | -0.21 (out) | in | 0.09 | 0.70 |
| 1 | 83.5 | yes | 33 (+106%) | 56 (+68%) | 0.72 (0.63) | 0.54 (0.02) | 0.45 (0.40) | -0.23 (out) | in | 0.06 | 0.63 |
| 2 | 81.1 | yes | 30 (+89%) | 51 (+55%) | 0.67 (0.91) | 0.52 (0.04) | 0.43 (0.25) | -0.22 (out) | in | 0.10 | 0.77 |
| 3 | 219.1 | no | 32 (+100%) | 51 (+55%) | 0.69 (0.99) | 0.41 (0.19) | 0.34 (0.00) | -0.45 (in) | in | 0.05 | 0.77 |
| 4 | 161.7 | yes | 30 (+84%) | 51 (+54%) | 0.68 (0.99) | 0.51 (0.02) | 0.43 (0.18) | -0.22 (out) | in | 0.05 | 0.00 (degenerate) |
| 5 | 88.3 | yes | 31 (+92%) | 56 (+71%) | 0.70 (0.86) | 0.46 (0.09) | 0.39 (0.08) | -0.26 (out) | in | 0.06 | 0.77 |

- **Gate: fail in all 6 seeds.** Simulated full-arm clones are shared with G and M about 1.5-2x
  as often as in the data, although co-presence was in the loss. At the same time the full arm has
  too few clones (full_n z about -4.5 in every seed, the largest single loss term). The model
  cannot make a full arm that is both well populated and separate from G/M.
- **Primary: no pass in any seed.** full-M fails (p 0.01-0.04) in 4 of 5 gate-adequate seeds. Seed 5
  passes all three cross statistics but undershoots B7 G (-0.26). B7 G is outside in every
  gate-adequate seed (-0.21 to -0.26); E35's s0 pass (-0.30) did not survive the added full-arm terms.
- **Stability rule: not met** (no result to be stable). The held-out look stays unspent; shard-1
  predictions (0.63-0.77, one degenerate) also fail the power rule.
- **Consistent across seeds:** rho_a collapses to 0.05-0.10 (the full arm's expansion trait
  decouples from G/M), and instruction plus selection stay as in E35 (s_G 3.9-5.2, s_M 0.09-0.17).
- **Reading.** Every model in which the full arm draws its clones from the same day-2 universe
  overshoots full-arm co-presence (E32, E34, E36). Together with B12 (co-presence 0.3-0.4x
  independence), this points to the full arm sampling a **different subset of clones** from the
  G/M arms, for example a separate split of the day-2 pool, rather than to a different fate rule.
  This is an inference from structure; the paper's design (Science Supp Fig 7) is still unread.

## E37: full arm drawn from a separate split of the clone pool (pre-registered 2026-11-04)

- **Data diagnostic it must explain.** Full-arm clones are co-present with G and M at 0.3-0.4x
  independence (B12), and every model drawing all arms from one pool overshoots full-arm
  co-presence by 55-106% while under-filling the full arm (E32, E34, E36; full_n z about -4.5).
- **Model.** E36 (signed instruction, full arm with its own founder rate nu_full and expansion
  trait rho_a, full-arm co-presence in the loss, 10 simulations per evaluation) plus a clone-level
  pool indicator u ~ Bernoulli(phi). A clone with u = 1 seeds the full culture at rate nu_full and
  the G/M/Epo/SCF cultures at rate nu * eps; a clone with u = 0 seeds the full culture at
  nu_full * eps and the others at nu. Two new parameters, phi and eps in (0, 1); 23 dimensions.
  eps = 1 recovers E36 exactly, so the model nests it. Fate, instruction and selection are unchanged.
- **Fit.** Differential evolution, 500 generations, 6 seeds. Seeds 0-2 warm-start from e36s_s2
  (the lowest-loss E36 fit passing E33's gate) with phi 0.5 and eps at the upper clip (0.95);
  seeds 3-5 cold.
- **Gate (as E36).** E33's gate, plus full-G and full-M co-presence within ±30% of the fit set's
  counts (16, 33). Co-presence is recovered from the saved z-scores, sim = obs + z * sqrt(max(obs, 4)),
  as in the E36 analysis.
- **Readings (best gated fit).**
  - **Primary pass:** G-M, full-G and full-M all p >= 0.05, and both B7 differences inside the
    95% range.
  - **Stability:** a result counts only if at least 3 of 6 seeds pass the gate and agree, with every
    cross statistic's pass/fail the same in each.
  - **Mechanism reading:** "separate pool supported" if the gated seeds have eps <= 0.5;
    "not supported" if eps >= 0.8 in them (the leak is near 1 and the indicator does nothing).
  - **Secondary (not gating):** full-arm clone count within ±20% of the data.
- **Power check, if a stable pass:** the held-out look follows only if the shard-1 prediction is
  stable across the gated seeds and differs from E20's by more than about 0.37.

### E37 results (2026-11-04)

500 generations, 6 seeds (0-2 warm from e36s_s2, 3-5 cold). All 6 pass E33's gate; **2 pass the
full co-presence gate (seeds 3 and 5)**. Fit-set co-presence: full-G 16, full-M 33.

| seed | loss | co full-G / full-M | gate | phi | eps | nu_full | full clones (data 368) | full-G (p) | full-M (p) | G-M (p) | B7 G (data -0.37) | B7 M | shard-1 pred |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 (warm) | 81.0 | 26 / 51 | no | 0.46 | 0.72 | 0.40 | 257 | 0.69 (1.00) | 0.53 (0.00) | 0.41 (0.06) | -0.19 (out) | in | 0.74 |
| 1 (warm) | 79.5 | 31 / 53 | no | 0.73 | 0.95 | 0.32 | 273 | 0.69 (0.98) | 0.50 (0.07) | 0.41 (0.12) | -0.22 (out) | in | 0.78 |
| 2 (warm) | 82.6 | 28 / 50 | no | 0.24 | 0.58 | 0.41 | 273 | 0.71 (0.80) | 0.46 (0.05) | 0.35 (0.00) | -0.18 (out) | in | 0.75 |
| 3 (cold) | 92.3 | 19 / 30 | **yes** | 0.22 | **0.08** | 4.76 | 356 | 0.70 (0.81) | 0.48 (0.24) | 0.41 (0.02) | -0.23 (out) | in | 0.03 |
| 4 (cold) | 89.3 | 24 / 39 | no (full-G +50%) | 0.20 | **0.10** | 4.68 | 357 | 0.68 (0.90) | 0.51 (0.07) | 0.42 (0.07) | -0.26 (out) | in | 0.24 |
| **5 (cold, best gated)** | **49.8** | 20 / 30 | **yes** | 0.24 | **0.09** | 4.51 | 361 | 0.70 (0.79) | 0.50 (0.15) | 0.44 (0.24) | **-0.27 (out; range -0.37 to -0.19)** | in | 0.72 |

- **Primary (best gated fit, seed 5): fail, on B7 G only.** All three cross statistics pass
  (p 0.79 / 0.15 / 0.24) and B7 M is inside; B7 G (-0.27) misses its 95% range by about 0.004.
  Caveat on full-M: its simulated mean stays about 0.50 against 0.24; it passes because the
  separate pool leaves few full-arm-co-present clones, so its range widens (0.00-0.79), not because
  the model matches it.
- **Stability: not met** (2 gated seeds, rule needs 3). Held-out look unspent; shard-1 predictions
  are unstable across the separate-pool seeds (0.03, 0.24, 0.72).
- **Mechanism reading: separate pool supported.** Both gated seeds have eps <= 0.5 (0.08, 0.09),
  and so does seed 4. All three cold seeds found the same basin: about 20-24% of clones seed the full
  culture, with about 10% leak each way, and a high full-arm founder rate.
- **Secondary: full arm filled for the first time.** Cold seeds reach 356-361 full-arm clones
  (data 368; E36 about 260-290). The full_n misfit that dominated E36's loss is gone; seed 5 has
  the lowest loss of the E34-E37 line (49.8).
- **The warm starts were a trap.** Seeds 0-2 started at eps 0.95 and stayed in E36's basin (eps
  0.58-0.95), failing the gate as E36 did. Future runs on this model should start cold.
- **Remaining misfit:** B7 G sits at -0.18 to -0.27 in every E36/E37 fit, slightly too little
  selection under G-CSF (E35's lone pass had -0.30).

## E38: E37's model, 18 cold seeds (pre-registered 2026-11-04, after E37's results)

- **Why.** E37's three cold seeds all found a separate-pool basin (eps 0.08-0.10, phi 0.20-0.24),
  two passed the gate, and the best gated fit failed the primary only on B7 G, by 0.004. The warm
  seeds stayed in E36's basin. E38 asks whether the cold-start result is stable.
- **Model.** E37 unchanged (e37_pool.simulate and unpack, 23 dimensions).
- **Fit.** Differential evolution, up to 500 generations, **18 seeds, all cold**. New fitting code
  (cca1cc8): each generation evaluated in parallel on 4 forked workers ("deferred" updating), and
  early stopping when the best loss improves by less than 0.1% over 50 generations. The objective,
  the bounds and the population are unchanged.
- **Gate (as E37).** E33's gate, plus full-G and full-M co-presence within ±30% of the fit set's
  counts (16, 33), recovered from the saved z-scores.
- **Readings.**
  - **Primary pass (best gated fit):** G-M, full-G and full-M all p >= 0.05, and both B7
    differences inside the 95% range.
  - **Stability (rule changed for 18 seeds, fixed now):** a result counts only if at least 9 of
    18 seeds pass the gate, and the gated seeds agree on every primary component's pass/fail in at
    least 2/3 of them.
  - **Mechanism:** separate pool supported if the median eps of gated seeds is <= 0.5; not supported
    if >= 0.8.
  - **Reported, not gating:** the fraction of gated seeds passing the primary; full-arm clone
    count within ±20%; whether full-M's simulated mean is within 0.15 of the data (0.24), since
    E37's full-M passed only through a wide range.
- **Power check, if a stable pass:** the held-out look follows only if the gated seeds' shard-1
  predictions agree (interquartile range < 0.2) and their median differs from E20's by more than
  about 0.37.

### E38 results (2026-11-04)

18 cold seeds, parallel fitting with early stopping (11 stopped early, 249-492 generations).
**9 of 18 pass the gate** (E33's gate plus full-arm co-presence ±30%): seeds 1, 3, 4, 6, 8, 11,
12, 14, 15.

| component (gated seeds) | pass | reading |
|---|---|---|
| G-M p >= 0.05 | 5/9 | **no agreement** (needs >= 6/9 either way); sim 0.36-0.42 vs data 0.51 |
| full-G | 9/9 | agrees, pass |
| full-M | 9/9 | agrees, pass; but simulated mean 0.40-0.52 vs 0.24 in 9/9 (none within 0.15) |
| B7 G inside | 2/9 | **agrees, fail**; sim -0.23 to -0.31 vs data -0.37 |
| B7 M inside | 9/9 | agrees, pass |

- **Primary (best gated fit, seed 1, loss 26.2, the lowest of the E34-E38 line): fail** on G-M
  (0.40, p 0.02) and B7 G (-0.27). One gated seed (8) passes all five components (G-M 0.42 p 0.09,
  B7 G -0.30); 1/9 is not a result.
- **Stability: not met.** The gate count reaches 9/18, but G-M splits 5/4. Stable verdicts: the
  full-arm statistics pass, B7 M passes, B7 G fails.
- **Mechanism: separate pool supported, and stable.** Median eps of gated seeds 0.09; 15 of 18
  seeds land in the same basin (phi 0.19-0.28, eps 0.08-0.12). The full arm's clone count is
  right in 9/9 gated seeds. Selection estimates are also stable across all 18 seeds: neutrophil
  survival s_G 3.8-7.4 (gated 4.1-5.3) under G-CSF, s_M 0.07-0.15 under M-CSF.
- **Power check: fails.** Gated shard-1 predictions have median 0.52 and IQR 0.65 (several
  degenerate at 0). The held-out look stays unspent.
- **What is left.** Three misfits, all stable across seeds: G-M a little low (about 0.40 vs 0.51),
  B7 G a little weak (about -0.27 vs -0.37), and full-M far too high (about 0.48 vs 0.24). E37/E38
  fixed which clones the full culture holds, but not how differently its clones' cells behave. The
  full arm is mostly day-6 libraries while G and M are mostly day-4 (B12), so sampling time is the
  first candidate. Per the rule (no new simulator without a data diagnostic it must explain), the
  next step is a diagnostic: does cross-condition agreement fall with the gap in library day?

## B18: does cross-condition agreement fall with the gap in sampling day? (pre-registered 2026-11-04)

- **Why.** E38 fits G-M and the full arm's clone pool, but full-M stays about 0.48 against 0.24.
  The full arm is mostly day-6 libraries; G and M are split between day 4 and day 6 (B12). If a
  clone's N-share drifts between day 4 and day 6, pairs sampled on different days agree less, which
  could make full-M low with no difference in fate rule.
- **Data.** Cytokine experiment, full / G-CSF / M-CSF only (no Epo or SCF, so the held-out shard is
  not touched), mature N and M counts by library day. Agreement statistic as in OBS: Pearson r of
  clone N-shares, clones with >= 2 N+M cells on each side. 95% CIs by bootstrapping clones (2000).
- **Tests.**
  - **T1 (primary): G-M same-day vs different-day.** Rows for (G4, M4) and (G6, M6) form the
    same-day set; (G4, M6) and (G6, M4) the different-day set. A clone contributes every row it
    qualifies for; the bootstrap resamples clones with all their rows. Statistic: r_same - r_diff.
  - **T2: within-condition day-4 vs day-6 agreement** for G, M and full (how much a clone's N-share
    drifts in two days in the same medium).
  - **T3: full-M and full-G by partner day,** using day-6 full libraries only: r(full6, M6) vs
    r(full6, M4), and the same for G.
- **Readings, fixed now.**
  - **Timing effect supported** if T1's r_same - r_diff > 0 with a 95% CI excluding 0, **and** T3
    shows r(full6, M6) > r(full6, M4).
  - **Timing explains full-M** if, in addition, r(full6, M6) reaches 0.40 or more (G-M's level),
    i.e. day-matched full and M agree like G and M do. If r(full6, M6) stays below 0.30, timing is
    not the explanation and the full culture's cells really differ.
  - **No timing effect** if T1's CI includes 0 and T3 shows no day-matched gain.
  - All clone counts are reported; any comparison with fewer than 30 clones is descriptive only.

### B18 results (2026-11-04)

| comparison | r | 95% CI | clones |
|---|---|---|---|
| **T1 G-M same day** (G4-M4, G6-M6 rows) | 0.50 | | 41 |
| **T1 G-M different day** (G4-M6, G6-M4 rows) | 0.23 | | 44 |
| **T1 difference** | **+0.27** | **-0.12 to 0.74** | |
| G4-M4 / G6-M6 / G4-M6 / G6-M4 | 0.32 / 0.63 / 0.23 / n.a. | | 22 / 20 / 42 / 3 |
| T2 G day 4 vs day 6 | 0.60 | 0.35 to 0.80 | 39 |
| T2 M day 4 vs day 6 | 0.32 | -0.02 to 0.88 | 131 |
| T2 full day 4 vs day 6 | n.a. | | 4 |
| **T3 full6-M6** | **0.04** | **-0.13 to 0.43** | **76** |
| T3 full6-M4 | undefined (no N-share variation among these M4 clones) | | 18 |
| T3 full6-G6 | 0.92 | 0.78 to 0.99 | 12 |
| T3 full6-G4 | 0.15 | -0.33 to 0.91 | 14 |

- **Timing effect: not supported** by the pre-registered rule. Same-day G-M agreement is higher
  than different-day (0.50 vs 0.23), in the predicted direction, but the CI of the difference
  includes 0 (about 40 clones a side). T3 cannot show a day-matched gain for full-M (full6-M4 is
  undefined).
- **Timing does not explain full-M.** Day-matched full6-M6 is 0.04 on 76 clones (upper CI 0.43),
  below the pre-registered 0.30. Even sampled on the same day, full and M-CSF clones barely agree.
- **Descriptive (under 30 clones):** day-matched full6-G6 is 0.92 (12 clones), against 0.15 for
  full6-G4. The full culture agrees closely with G-CSF on the same day, not with M-CSF.
- **A caveat that shapes the next step.** M-CSF libraries contain very few neutrophils (M6: 114 N
  vs 4,526 M cells), so most M-CSF clones have N-share 0. Low full-M may come from **which**
  M-CSF clones carry the rare neutrophils, not from day effects. E38 matches how many M-CSF clones
  have any neutrophil (B10, 0.20 vs 0.205) but still predicts full-M 0.48. In the model, the
  M-CSF clones with neutrophils are the full arm's neutrophil-heavy clones; in the data, they
  apparently are not.

## B19: which M-CSF clones make the rare neutrophils, in data vs model? (pre-registered 2026-11-04)

- **Why.** M-CSF wells hold few neutrophils, so full-M agreement depends on which M-CSF clones
  carry them. E38 matches how many M-CSF clones have any neutrophil (B10) but predicts full-M 0.48
  against 0.24 (B18: 0.04 even day-matched). Hypothesis: in the model the neutrophil-carrying M-CSF
  clones are the full culture's neutrophil-heavy clones; in the data they are not.
- **Data.** Cytokine experiment, full / G-CSF / M-CSF (no Epo/SCF), mature N and M counts pooled
  over days. Clones with >= 2 N+M cells in both arms of a pair. In the M-CSF arm a clone is "N+"
  if it has any neutrophil.
- **Tests.**
  - **T1 (primary), full vs M:** d_full = mean full N-share of N+ M-CSF clones minus that of
    N0 M-CSF clones; also the AUC of full N-share for separating N+ from N0.
  - **T2 (reference), G vs M:** the same with G-CSF N-share in place of full (B7/B11 found the
    G-M link).
  - **Model:** the same statistics on 200 simulations of the best gated E38 fit (e38_s1), the whole
    simulated data set, same filters. Robustness: 50 simulations of each of the other 8 gated E38
    seeds, reporting the range of their model means.
- **Readings, fixed now.**
  - **Model mislocates the rare neutrophils** if the data's d_full is below the 2.5th percentile
    of e38_s1's simulated d_full.
  - **Consistent** if inside the 95% range; then the full-M misfit lies elsewhere.
  - T2 is read the same way and shows whether the problem is specific to the full arm.
  - Data CIs by bootstrapping clones (2000). Fewer than 30 clones in either group: descriptive only.

### B19 results (2026-11-04)

| test | data d (95% CI) | AUC | N+ / N0 clones | model e38_s1 mean [95%] | data percentile | other gated seeds' means |
|---|---|---|---|---|---|---|
| **T1 full vs M** | 0.19 (-0.09 to 0.51) | 0.63 | **8** / 76 | 0.55 [0.09, 0.92] | 7th | 0.51-0.61 |
| T2 G vs M | 0.51 (0.28 to 0.71) | 0.79 | 16 / 62 | 0.53 [0.38, 0.65] | 39th | 0.51-0.54 |

- **Pre-registered reading, T1: consistent** (data at the 7th percentile, inside the model's 95%
  range), and **descriptive only**: just 8 M-CSF clones that are also in the full arm carry any
  neutrophil. The hypothesis is not confirmed.
- **Direction.** In the data, the full culture's N-share barely separates the neutrophil-carrying
  M-CSF clones (0.19); the model separates them strongly (0.55, all 9 gated seeds 0.51-0.61). G-CSF
  N-share separates them equally well in data and model (0.51 vs 0.53). So the misfit, if real, is
  specific to the full arm, but 8 clones cannot show it.
- **Implication.** Full-M rests on a handful of informative clones (B18: 0.04 on 76 clones, most
  with M-CSF N-share 0). The full-M discrepancy is at or below the data's resolution, and no
  diagnostic on this data set can settle it. The G-vs-M structure, which is well powered, matches
  the model (T2 here, G-M and B7 M in E38). The one well-powered misfit left is B7 G
  (about -0.27 vs -0.37, failing in 7/9 gated E38 seeds).

## B20: where does the model's B7 G shortfall come from? (pre-registered 2026-11-04)

- **Why.** B7 G (mean G-CSF N-share of clones also present under M-CSF minus that of G-only
  clones; data -0.37) is the one well-powered misfit left: E38's gated fits give -0.23 to -0.31
  and fail it in 7/9. A model change must be aimed at the part that misfits.
- **Data and model.** Cytokine full / G / M only, pooled over days, presence = >= 2 N+M cells
  (as B7). Model: 200 simulations of e38_s1 and 50 of each other gated E38 seed.
- **Components.**
  - C1: mean G N-share of "both" clones. C2: mean G N-share of G-only clones.
  - C3: number of both clones; C4: number of G-only clones.
  - C5: fraction of G-only clones that are N-pure (share 1); C6: fraction of both clones with
    G N-share 0.
  - C7: B7 G within small G clones (2-3 N+M cells) and C8: within larger clones (>= 4).
  - C9: day-2 state of the selection: mean measured m (standardised Cebpa-Irf8) of both clones minus
    G-only clones, among clones with day-2 measurements (the model uses the same measured m).
- **Readings, fixed now.** For each component, the data's percentile in e38_s1's simulations. A
  component is **misfit** if outside the 2.5-97.5% range, and **robustly misfit** if, in addition,
  at least 7 of the 8 other gated seeds' means lie on the same side of the data. The B7 G shortfall
  is attributed to the robustly misfit components. If none is, the shortfall is diffuse and no
  targeted change is justified.

### B20 results (2026-11-04)

| component | data | e38_s1 mean [95%] | pct | robustly misfit |
|---|---|---|---|---|
| B7 G | -0.373 | -0.266 [-0.354, -0.176] | 1% | yes (8/8 seeds above) |
| **C1 both clones, mean G N-share** | **0.348** | 0.454 [0.381, 0.535] | 1% | **yes** (8/8 above) |
| C2 G-only clones, mean G N-share | 0.721 | 0.720 [0.688, 0.756] | 53% | no |
| **C3 number of both clones** | **78** | 114 [95, 133] | 0% | **yes** (8/8 above) |
| C4 number of G-only clones | 474 | 434 [398, 473] | 97% | yes (0/8 above) |
| C5 G-only clones N-pure | 0.62 | 0.65 [0.61, 0.70] | 12% | no |
| **C6 both clones with G N-share 0** | **0.56** | 0.46 [0.37, 0.54] | 99% | **yes** (0/8 above) |
| C7 B7 G, small G clones (2-3 cells) | -0.41 | -0.25 [-0.35, -0.15] | 0% | yes |
| C8 B7 G, larger clones (>= 4) | -0.23 | -0.16 [-0.30, -0.03] | 18% | no |
| C9 day-2 m, both minus G-only | -0.64 | -0.55 [-0.85, -0.13] | 30% | no |

- **The shortfall is in the clones present under both cytokines, not in the G-only clones.** The
  model's G-only clones match the data (C2, C5). Its "both" clones are too many (114 vs 78) and too
  neutrophilic under G-CSF (0.45 vs 0.35; too few with no neutrophils), so too few G-only clones
  are left (C4). Concentrated in small clones (C7). The day-2 state separating the two groups is
  right (C9).
- **Mechanistic reading.** In the model, M-CSF selection acts only on neutrophil output (s_M
  multiplies N cells), so a neutrophil-leaning clone still shows up under M-CSF through its few
  monocytes. In the data such clones are missing from M-CSF wells altogether: under M-CSF, whole
  neutrophil-leaning clones fail to expand. This fits B13, where day-2 Csf1r vs Csf3r balance
  predicted presence.

## E39: clone-level expansion under M-CSF depends on fate state (pre-registered 2026-11-04)

- **Data diagnostic it must explain (B20).** E38 has too many G-M co-present clones (114 vs 78),
  and they are too neutrophilic under G-CSF (0.45 vs 0.35), while G-only clones fit. This is where
  B7 G's shortfall comes from.
- **Model.** E38 (E37's model, separate clone pool) plus one parameter g_M in (0, 3): under M-CSF
  a clone's expansion is multiplied by exp(-g_M * z), where z is the clone's latent fate state
  (which loads on the measured day-2 Cebpa-Irf8 score). Neutrophil-leaning clones expand less as a
  whole under M-CSF, not only in their neutrophil output. g_M = 0 recovers E38. 24 dimensions.
- **Fit.** As E38: 18 cold seeds, up to 500 generations, parallel workers, early stopping.
- **Gate, primary and stability: as E38.** Gate = E33's gate plus full-G / full-M co-presence
  ±30%. Primary on the best gated fit: G-M, full-G, full-M p >= 0.05 and both B7 differences
  inside. Stability: >= 9/18 gated, and the gated seeds agree on each primary component in >= 2/3.
- **Mechanism readings.**
  - **Clone-level M-CSF selection supported** if the gated seeds' median g_M >= 0.5 **and** B7 G
    passes in >= 2/3 of gated seeds (E38: 2/9).
  - **Not supported** if median g_M < 0.2 (the fit does not use it) or B7 G still fails in >= 2/3.
  - Reported: B20's C1 (both clones' G N-share) and C3 (number of both clones) in the best gated
    fit, against the data (0.35, 78).
- **Power check, if a stable pass:** as E38 (gated shard-1 IQR < 0.2, median differing from E20's
  by more than about 0.37).

### E39 results (2026-11-04)

18 cold seeds. **1 of 18 passes the gate** (seed 0); 17 fail E33's gate, almost all because the
G-CSF arm is too neutrophilic (mean N-share 0.69-0.73 vs data 0.636 ± about 0.04).

- **The fits use the new mechanism strongly:** g_M 0.98-1.45 in 17/18 seeds (one seed 0.27).
  But they trade it against the old one: per-cell M-CSF neutrophil selection weakens (s_M
  0.34-0.88, against about 0.1 in E38), and G-M agreement overshoots (0.62-0.75 vs 0.51, p < 0.05
  in most seeds). B7 G now tends to overshoot (-0.42 to -0.50 in 12/18; E38 undershot at about
  -0.27), and B7 M fails in 10/18.
- **Best (only) gated fit, seed 0:** G-M 0.73 (p 0.00), full-M 0.81 (p 0.01), B7 G -0.45 (inside),
  B7 M outside. Primary: fail. B20 targets: both-clone G N-share 0.34 [0.28, 0.41] (data 0.35, now
  matched) but 146 both clones [122, 169] (data 78; worse than E38's 114), because the weaker s_M
  lets more clones appear under M-CSF.
- **Stability: not met** (1/18 gated). **Mechanism reading:** g_M median 1.31 and B7 G passes in
  the single gated seed, so the pre-registered criterion is formally met on n = 1. That is not a
  result. **Power check:** not applicable.
- **Seed 11**, the one fit that keeps g_M low (0.27) and s_M at 0.13, passes all five primary
  components (G-M 0.41 p 0.06, B7 G -0.32) but fails the gate on the G-CSF mean (0.693). This is the
  E38 solution with a little g_M.
- **Reading.** Clone-level M-CSF selection fixes the composition of the both clones (C1) but the
  fit uses it as a substitute for per-cell selection, which breaks G-M, the G-CSF level and the
  number of both clones. Within this model family, G-M agreement, B7 G and the within-condition
  levels pull against each other, and fits land in one corner or another (E35 s0, E38 s8 and
  E39 s11 are the rare points that satisfy all of them). Adding mechanisms now changes which
  corner the optimiser finds more than it reveals biology. The best-supported model remains E38's.

## E40: profile along G-CSF selection, to map the instruction:selection ridge (pre-registered 2026-11-04)

- **Why.** E35-E39 fits land in different corners, and the literature (Xu et al. 2018) shows
  death-type and differentiation-type rates are often identifiable only as a ratio. If the seed
  scatter is a ridge, the honest result is the range of compatible instruction:selection
  combinations, not one point.
- **Model.** E38's (E37 model: measured day-2 inputs, signed instruction, separate clone pool,
  full-arm co-presence in the loss). Neutrophil survival under G-CSF, s_G, is **fixed** at one of
  six values, 1, 1.5, 2.5, 4, 6.5, 10 (s_G = 1 means no G-CSF selection); every other parameter is
  refitted. The s_G coordinate is left in the search vector but ignored.
- **Fit.** 3 cold seeds per value (18 jobs), up to 500 generations, parallel workers, early
  stopping. No early-stop guard: a profile has no stability rule.
- **Per grid point:** best loss; whether any seed passes E38's gate (E33's gate plus full-arm
  co-presence ±30%); the instruction shift tau_G - tau_full of the best gated seed; the primary
  components (G-M, full-G, full-M, B7 G, B7 M) of the best gated seed.
- **Readings, fixed now.**
  - **Compatible** grid points: at least one gated seed.
  - **Ridge supported** if compatible points span at least a 2.5x range of s_G (three or more
    points) **and** instruction compensates: the Spearman correlation between s_G and
    tau_G - tau_full across compatible points is <= -0.6.
  - **s_G identified** if compatible points cover at most two adjacent values.
  - **Pure instruction under G-CSF excluded** if s_G = 1 is not compatible.
  - Reported: the profile of best loss, and where B7 G and G-M pass along it.

## B21: fluctuation-test analysis of fate memory, no simulator (pre-registered 2026-11-04)

- **Why.** The Luria-Delbrück fluctuation test separates pre-existing from induced change by the
  variance between replicate cultures; its extension to cell states (Singh & Saint-Antoine 2023)
  reads switching timescales off that variance. Our sister wells and replicate libraries are such
  replicates. This gives estimates that do not depend on any of our simulators.
- **Data.** Cytokine experiment, full / G-CSF / M-CSF only (no Epo/SCF), mature N and M cells by
  library. Clones need >= 2 N+M cells in each compartment compared. Libraries of the same condition
  and day are treated as replicate samples (whether they are separate wells is not stated in the
  metadata; caveat). Bootstrap over clones, 1000 resamples.
- **Tests.**
  - **T1, overdispersion:** for each condition x day (libraries pooled), fit a beta-binomial to
    clone N counts by maximum likelihood; rho is the intra-clone correlation. Cells deciding
    independently with a condition-wide probability (pure induction, no heritable state) give
    rho = 0.
  - **T2, replicate libraries:** Pearson r of clone N-share between different libraries of the
    same condition and day (G4, M4, G6, M6, full6), rows pooled over library pairs.
  - **T3, memory across days:** R = r(day 4 vs day 6, same condition) / r(replicate libraries,
    same day), for G and M. Under exponential loss of memory over the two days, the half-life is
    2 * ln 2 / (-ln R) days.
  - **T4, descriptive:** same-day G-M agreement against the replicate-library agreement of G and M.
- **Readings, fixed now.**
  - **Pure induction excluded** in a condition x day if rho's 95% CI lower bound is above 0.05.
  - **Memory stable over days 4-6** in a condition if R's 95% CI includes 1; otherwise the
    half-life and its CI are reported.
  - Comparisons with fewer than 30 clones are descriptive only.

### B21 results (2026-11-04)

| test | G-CSF | M-CSF | full |
|---|---|---|---|
| T1 rho, day 4 (clones) | 0.79 [0.73, 0.85] (330) | 0.86 [0.73, 0.96] (194) | n.a. (11) |
| T1 rho, day 6 (clones) | 0.67 [0.58, 0.75] (238) | 0.62 [0.44, 0.75] (881) | 0.72 [0.68, 0.77] (709) |
| T2 replicate libraries, day 4 | 0.91 [0.82, 0.97] (51) | undefined (18) | |
| T2 replicate libraries, day 6 | 0.85 [0.76, 0.91] (74) | 0.28 [-0.02, 0.58] (305) | 0.84 [0.74, 0.91] (122) |
| T3 day 4 vs day 6 | 0.60 (39); R 0.68 [0.40, 0.92] | 0.32 (131); R uninformative | |
| T4 same-day G-M | G4-M4 0.32 (22), G6-M6 0.63 (20) | | |

- **Pure induction excluded in every condition and day** (rho 0.62-0.86, all CIs far above
  0.05). Cells of a clone do not decide independently with a condition-wide probability:
  N-share is mostly set per clone. Caveat from the fluctuation-test logic: this also holds if
  decisions are induced but made by a few founder cells right after plating and then expanded. It
  excludes cell-by-cell induction late in culture, not induction at the first divisions.
- **Replicate libraries agree closely** in G-CSF (0.85-0.91) and the full cocktail (0.84), about
  the E19 sister level (rho_f 0.88). Under M-CSF they agree weakly (0.28, 305 clones), because
  N-share is near 0 for almost every clone. M-CSF data cannot carry fate-agreement statistics.
- **Fate memory is not perfectly stable from day 4 to day 6 under G-CSF:** R 0.68 [0.40, 0.92],
  CI excludes 1, a half-life of about 3.6 days [1.5, 16]. Over the 2-4 days of the experiment, most
  of the clone's fate is kept. Under M-CSF, R is uninformative.
- **Implication for the models.** Same-day G6-M6 agreement (0.63, 20 clones) approaches the
  replicate ceiling, and G4-M4 (0.32, 22 clones) is lower; descriptive only. A day-dependent memory
  loss of this size is not in any simulator so far (they use one draw per clone, rho_f 0.88).

## B22: do M-CSF-only clones come from monocyte-DC progenitors? (pre-registered 2026-11-04)

- **Why.** GMPs and MDPs give distinct monocytes (Yáñez et al. 2017). If many M-CSF-only clones
  are MDP-type, they make only monocytes under any cytokine, which would lower full-M and inflate
  "selection" with no cytokine effect.
- **Data.** Cytokine experiment, normalised expression. Clones with day-2 cells and >= 2 mature
  cells under M-CSF. **M-only:** no mature cells under G-CSF or the full cocktail.
  **M-shared:** >= 2 mature cells under G-CSF or the full cocktail.
- **Scores (gene sets fixed now).** MDP score = mean z (over day-2 cells) of log1p expression of
  Flt3, Cd74, H2-Aa, H2-Ab1, H2-Eb1, averaged per clone over its day-2 cells. DC-like monocyte score
  = the same five genes over the clone's mature monocytes under M-CSF (days 4 and 6).
- **Tests.**
  - **T1 (primary):** partial Spearman between day-2 MDP score and M-only (1) vs M-shared (0),
    controlling for the Cebpa - Irf8 day-2 score and the clone's log number of M-CSF cells.
  - **T2:** the same with the DC-like score of the clone's M-CSF monocytes.
- **Readings.** **MDP-origin supported** if T1 is positive with p < 0.05 **and** T2 is positive
  with p < 0.05. **Not supported** if T1 has p >= 0.05. Either way, report the fraction of M-only
  clones in the top quartile of the MDP score.

### B22 results (2026-11-04)

246 clones with day-2 cells and >= 2 M-CSF cells: 177 M-only, 69 M-shared.

| test | partial rho | p | clones |
|---|---|---|---|
| **T1 day-2 MDP score vs M-only** (controls: Cebpa - Irf8, log M-CSF size) | **-0.01** | **0.90** | 246 |
| T2 DC-like score of M-CSF monocytes vs M-only | 0.12 | 0.07 | 240 |

- **MDP-origin: not supported** (T1 p >= 0.05). M-only clones do not carry a stronger day-2
  monocyte-DC progenitor signature (mean score -0.10 vs -0.14 in M-shared clones). Unadjusted,
  M-only clones are 81% of the top MDP quartile against 72% overall; the difference disappears once
  the Cebpa - Irf8 axis and clone size are controlled.
- T2 shows a weak trend: M-only clones' monocytes are slightly more DC-like (p 0.07). This is not
  significant, and could follow from their monocytes having matured further in M-CSF alone.
- **Reading.** Clones that appear only under M-CSF are not a separate MDP-type lineage at day 2;
  what separates them is the same Cebpa/Irf8 axis already in the models. The low full-M is not
  explained by an MDP sub-population (with the caveat that a 5-gene score is a coarse MDP marker
  in day-2 LARRY progenitors).

### E40 results (2026-11-04)

| s_G fixed | best loss | gated seeds | instruction tau_G - tau_full (best gated) | G-M (p), best gated | B7 G | G-CSF mean N-share |
|---|---|---|---|---|---|---|
| 1 (no G-CSF selection) | 108.2 | **0/3** | (6.1-7.5 in all seeds) | 0.21-0.29 (0.00) | -0.06 to -0.08 | 0.61-0.64 |
| 1.5 | 119.1 | **0/3** | (5.4-6.9) | 0.27-0.33 (0.00) | -0.06 to -0.16 | 0.63-0.64 |
| 2.5 | 91.8 | 2/3 | 3.70 | 0.30 (0.00) | -0.09 (out) | 0.62-0.64 |
| 4 | 55.4 | 3/3 | 0.58 | 0.43 (0.12) | -0.25 (out) | 0.64-0.66 |
| 6.5 | **37.1** | 2/3 | 0.48 | 0.42 (0.05) | -0.26 (out) | 0.68-0.69 |
| 10 | 128.0 | **0/3** | (0.4-0.6) | 0.41-0.42 | -0.24 to -0.28 | 0.72 (too high) |

- **Ridge: supported, and bounded.** Compatible points are s_G 2.5, 4 and 6.5 (a 2.6x range), and
  instruction compensates exactly: Spearman(s_G, instruction) = -1.0 across them (3.7 logits at
  s_G 2.5, about 0.5 at 4-6.5). Selection and instruction trade off along a ridge, as Xu et al.'s
  identifiability result predicts. But the ridge has ends.
- **Pure instruction under G-CSF excluded.** At s_G = 1 (and 1.5) no seed passes the gate, even
  with very strong instruction (6-7.5 logits). Without a G-CSF neutrophil survival advantage, G-M
  agreement collapses (0.21-0.33 vs 0.51) and the fate-selected presence pattern disappears (B7 G
  about -0.07 vs -0.37). Too much selection is also excluded: at s_G = 10 the G-CSF wells become
  too neutrophilic (0.72 vs 0.636).
- **Where the fit is good.** The loss is lowest at s_G 4-6.5 (37-58), where instruction is small
  (about 0.4-1.1 logits) and G-M passes or nearly passes (p 0.01-0.12). At s_G 2.5, gated fits
  exist but are poor (G-M 0.30-0.36, p 0.00).
- **Primary: no grid point passes.** B7 G is outside its range everywhere (best -0.29); this
  misfit is not a ridge effect.
- **Headline.** Under G-CSF, neutrophils need a survival/expansion advantage of roughly 2.5-6.5x
  (best fits 4-6.5x); the matching instruction is between about 0.5 and 3.7 logits, smaller at the
  better-fitting end. Selection is necessary; instruction alone cannot produce the data.

## E41: fading fate memory between day 4 and day 6 (pre-registered 2026-11-04)

- **Data diagnostic it must explain (B21).** Under G-CSF, a clone's day-4 vs day-6 agreement is
  0.68 of its same-day replicate agreement (CI 0.40-0.92): fate memory fades over the culture. All
  simulators so far give each clone one fate draw per condition, and pool days 4 and 6.
- **Model.** E38's (E37 model, separate clone pool) with day-resolved sampling. Each condition's
  expected cells are split between day 4 and day 6 by the fit set's observed day-4 share of
  mature N+M cells (full 0.041, G 0.504, M 0.155, Epo 0.938, SCF 1.0; fixed, not fitted). Day-4 cells use the clone's
  branch state zb; day-6 cells use zb6 = r46 * zb + sqrt(1 - r46^2) * noise, so r46 in (0, 1) is
  the memory kept from day 4 to day 6. One new parameter; r46 = 1 recovers E38 for every pooled
  statistic. 24 dimensions.
- **New target.** G-CSF day-4 vs day-6 agreement on the fit set: 0.70 (17 clones; SE 0.16 from
  the bootstrap CI). Weakly informative by construction; B21's whole-data value is 0.60.
- **Fit.** 18 seeds, up to 500 generations, parallel workers, early stopping. Seeds 0-8 start
  from E40's gated fits along the ridge (s_G 2.5: e40g2.5_s0, s1, s0; s_G 4: e40g4_s0, s1, s2;
  s_G 6.5: e40g6.5_s0, s2, s0), with r46 starting at 0.9; seeds 9-17 cold. s_G is free.
  **Early-stop guard on:** cancel once more than 9 seeds fail the gate.
- **Gate, primary, stability: as E38.** Gate = E33's gate plus full-G / full-M co-presence ±30%.
  Primary (best gated fit): G-M, full-G, full-M p >= 0.05 and both B7 differences inside.
  Stability: >= 9/18 gated, and gated seeds agree on each primary component in >= 2/3.
- **Readings, fixed now.**
  - **Fading memory relieves B7 G** if B7 G passes in >= 2/3 of gated seeds (E38 2/9; E40 at no
    grid point).
  - **Memory estimate:** median r46 of gated seeds, and the simulated day-4 vs day-6 agreement of
    the best gated fit against the data's 0.70 (fit set).
  - **Ridge consistency:** gated seeds' median s_G inside E40's compatible range 2.5-6.5 means
    consistent; outside means memory substitutes for selection.
  - **Fade used:** median r46 of gated seeds < 0.9; otherwise the fit does not need it.

### E41 results (2026-11-04): inconclusive, methods bug

- **The early-stop guard cancelled the run** after 11 of 18 seeds had finished, 10 of them failing
  the gate (first use of the guard; it worked as designed).
- **But the warm-started seeds did not search.** All 9 (seeds 0-8) stopped at generation 51:
  early stopping fires when the best loss improves by less than 0.1% over 50 generations, and a
  good warm start already is the population's best, so the rule fires before the rest of the
  population can catch up. Their parameters stayed at the starting values (r46 0.90, s_G exactly
  2.5 / 4 / 6.5). Only 2 cold seeds finished (13 and 17: 403 and 335 generations).
- **What can be read.** At E40's ridge fits with r46 0.9, pooled G-M agreement drops to
  0.28-0.38 (E40: 0.36-0.43), and full-G co-presence rises 25-50% above the data: day-6 memory loss
  lowers cross-condition agreement. The two cold seeds settle at r46 0.87 and 0.94. B7 G fails in
  every finished seed (-0.08 to -0.28). Weak evidence that the fit does not want fading memory.
- **Verdict: inconclusive, not a negative.** The guard's cancellation rested mostly on seeds that
  had not searched. Re-run as E41b with the bug fixed.
- **Fix (in e33.main):** early stopping is not checked before generation 200 (PATIENCE_MIN).
  E38-E40 were all cold-started and ran 249-500 generations, so they are unaffected; only warm
  starts were exposed.

## E41b: E41 re-run with the early-stopping fix (pre-registered 2026-11-04)

- Identical to E41 in model, target, seeds, warm starts, gate, primary, stability and readings,
  except that early stopping cannot fire before generation 200. Early-stop guard on (9/18).

### E41b results (2026-11-04): negative, fading memory is not needed and does not fix B7 G

- **Gate:** 9/18 gated, exactly the stability minimum. The warm starts now search (212-500
  generations). The guard logged 8 gated / 7 failed and stopped polling before the last 3 seeds
  finished (guard bug, fixed below); scored offline from all 18 result files.
- **Parameters (gated seeds, all agree):** s_G 4.4-5.8 (median 5.1), s_M 0.12-0.20,
  phi 0.17-0.24, eps 0.084-0.097, r46 0.86-0.95 (median 0.915).
- **Primary fails in every gated seed.** full-G 9/9, full-M 9/9 and B7 M 9/9 pass. **G-M fails
  0/9** (sim about 0.38 vs observed 0.51; E38 split 5/4). **B7 G fails 0/9** (sim -0.12 to -0.25
  vs data -0.37).
- **Readings (pre-registered):**
  - Fading memory relieves B7 G: **no** (0/9 vs the needed 6/9).
  - Fade used: **no**, median r46 0.915 >= 0.9.
  - Ridge consistency: **yes**, median s_G 5.1 inside E40's 2.5-6.5. Memory does not substitute
    for selection.
  - Memory estimate: best gated fit (seed 6, loss 28.5) has r_G46 z -0.10, so simulated day-4 vs
    day-6 agreement is about 0.68 against the data's 0.70.
- **What it means.** Near-full memory (r46 about 0.92) already reproduces B21's day-4 vs day-6
  ratio once the sampling noise of 17 clones is simulated, so B21's 0.68 is not evidence of
  real fading. The extra day-6 noise costs G-M agreement, which now fails everywhere. E41's model
  is discarded; E38/E40 stay the reference. B7 G remains the one unexplained misfit (too many
  co-present G clones that are too neutrophilic, B20).
- **Decision: discard.** Next candidates come from the literature triage: day-2 Gfi1 as a G-CSF
  responsiveness covariate (Sierra 2010), tested first as a data diagnostic (B23).

## B23: does day-2 Gfi1 gate the G-CSF response beyond Cebpa-Irf8? (pre-registered 2026-11-04)

- **Source.** Sierra 2010 (Blood, PMC2869559): Gfi1-null progenitors proliferate under G-CSF but
  make almost no neutrophils (>90% CFU-M), because Gfi1 -> RasGRP1 is needed for G-CSF Erk
  signalling. If day-2 Gfi1 varies between clones, G-CSF should reward low-Gfi1 clones with growth
  but not neutrophils. That is one candidate for B7 G: clones shared between the G and M wells
  are much less neutrophilic under G-CSF than G-only clones (data -0.37; E38-E41 give -0.1 to -0.25).
- **Already known (B13), not re-tested:** day-2 Gfi1 alone predicts G-CSF N-share (rho 0.22).
- **Data.** All clones with day-2 cells. Clone mean log1p normalised expression over day-2 cells
  (B13 method). m = Cebpa - Irf8 (the model's measured input). G-present = at least 2 mature N+M
  cells under G-CSF; shared and G-only as in B7 (e25 groups).
- **Tests (one look each):**
  - **T1 (primary).** G-present clones: partial Spearman of Gfi1 vs G-CSF N-share given m and log
    G-CSF size. Prediction positive, one-sided p < 0.05.
  - **T2 (primary).** B7 G on rank residuals: residualise G-CSF N-share on m (rank regression),
    then compute mean(shared) - mean(G-only) of the residuals; repeat with m + Gfi1. Prediction:
    adding Gfi1 shrinks the m-only residual B7 G by at least one third. Bootstrap CI (2000 clone
    resamples) for the shrink.
  - **T3 (secondary).** G-present clones: partial Spearman of Gfi1 vs log G-CSF size given m.
    Sierra predicts negative (low Gfi1, more proliferation).
  - **T4 (secondary).** T1 and T3 for Rasgrp1 and Csf3r, reported, BH across the four secondaries.
- **Readings, fixed now.**
  - T1 and T2 pass: Gfi1 is a second measured input. Next model = E38 with m2 = Gfi1 on the
    G-CSF selection term, pre-registered separately.
  - T1 passes, T2 fails: Gfi1 adds to fate but not to the B7 G misfit. Note it and do not build.
  - T1 fails: Cebpa-Irf8 already carries the Gfi1 information. Close the Gfi1 lead.

### B23 results (2026-11-04): Gfi1 adds to neutrophil share, does not explain B7 G

177 G-present clones with day-2 cells (39 shared with M-CSF, 138 G-only). Day-2 Gfi1 is
undetected in 63% of them; Rasgrp1 is undetected in all (so its secondary tests are empty).

| test | result | prediction |
|---|---|---|
| T1 Gfi1 vs G-CSF N-share, partial given m and size | rho **0.17**, one-sided p **0.010** | pass |
| T2 B7 G on rank residuals, shrink from adding Gfi1 | **2%** (95% CI -6% to 16%) | fail (needed >= 33%) |
| T3 Gfi1 vs G-CSF size, given m | rho -0.07, p 0.17 | not supported |
| T4 Csf3r vs N-share given m | rho 0.14, p 0.065 (nominal) | - |

- **Reading (pre-registered branch: T1 passes, T2 fails).** Day-2 Gfi1 carries fate information
  that Cebpa-Irf8 does not: Gfi1-high clones are more neutrophilic under G-CSF. But it does not
  separate shared from G-only clones, so it does not explain B7 G. Recorded; **no model built**.
- No support for Sierra's proliferation side (T3): low-Gfi1 clones do not grow more under G-CSF.
- B7 G stays unexplained by every day-2 measure tried (Cebpa-Irf8 in the model, receptors B13,
  MDP B22, Gfi1 B23). Remaining candidate: it is not a day-2 trait. Shared clones may be the ones
  whose M-CSF-well sister cells left a less neutrophilic G-CSF well sample (a sampling or
  co-occurrence effect), which needs a design check rather than a new mechanism.

## B24: is B7 G an artefact? Design check (pre-registered 2026-11-04)

- **Why.** B7 G (G-CSF N-share of clones shared with the M-CSF well minus that of G-only clones;
  fit set -0.38) is the one misfit that survives every model (E38-E41) and every day-2 measure
  (m, receptors B13, MDP B22, Gfi1 B23). The median gap is -1.0: G-only clones are mostly pure N
  under G-CSF, shared clones mostly pure M. One artefact would produce exactly this, with no day-2
  trait involved: **cross-contamination**, where cells or barcodes from a clone that is large in
  the M-CSF well show up as a few "monocytes" in the G-CSF library.
- **Audit already done.** The stored data value (b7_selection.json, -0.373) is computed on all
  clones, including the held-out shard 1. The simulation side uses the fit set (-0.382). The
  values are close, so this does not explain the misfit, but from now on B7 is compared on the
  fit set. No held-out statistic beyond this already-stored descriptive value was seen.
- **Data.** Fit set (split(cnt, None)). G-present / M-present = at least 2 mature N+M cells in that
  well, pooled over days (as B7). Reference simulation: best gated E38 fit (lowest loss), 200
  simulations, same statistics.
- **Tests (one look each):**
  - **D1 dose response.** G-present clones: Spearman of G N-share vs M-well cell count (0 for
    absent), partial given log G size. Data vs simulated 95% interval.
  - **D2 strata.** B7 G within G-well size strata 2, 3-5, 6+. Data vs simulated 95%.
  - **D3 contamination signature.** Share of shared clones whose G sample is small and pure M
    (G size <= 3, no N) while the M well holds at least 5x as many cells. Data vs simulated 95%.
  - **D4 same day vs other day (data only).** For G cells of day d, call a clone shared-same if
    it has at least 2 M-well cells on day d, shared-other if it has them only on the other day.
    B7 G (mean N-share of the day-d G sample) of shared-same and of shared-other minus G-only,
    pooled over d with at least 2 G cells on day d. Contamination needs the same
    library and day; the model has no day dependence for co-presence.
- **Readings, fixed now.**
  - **Contamination-like:** the data falls outside the simulated 95% only in the G-size-2 stratum
    (D2), D3 data above the simulated 95%, and D4 same-day gap at least twice the other-day gap.
    Then B7 G is dropped from the primary pending a cleaned statistic (G samples >= 3 cells).
  - **Biology:** the data falls outside in the 6+ stratum and the D4 gaps are similar. Then B7 G
    stays as a real misfit, pointing to M-CSF-well presence that depends on the clone more strongly
    than the model allows.
  - Anything else: mixed. Report it, and keep B7 G with the caveat.

### B24 results (2026-11-04): mixed. B7 G lives in 2-cell G samples, but it is not same-day contamination

Fit set; reference best gated E38 fit (e38_s1), 200 simulations.

| statistic | data | sim mean [95%] | |
|---|---|---|---|
| B7 G, all | -0.38 | -0.27 [-0.38, -0.13] | edge |
| D2, G size 2 (18 shared) | **-0.47** | -0.22 [-0.40, 0.00] | **outside** |
| D2, G size 3-5 (18 shared) | -0.31 | -0.25 [-0.45, -0.07] | inside |
| D2, G size 6+ (3 shared) | +0.01 | -0.09 [-0.42, 0.07] | inside (3 clones) |
| D3, small pure-M G sample, M well >= 5x | **0.15** (6/39) | 0.02 [0, 0.07] | **outside** |
| D1, G N-share vs M-well size | -0.30 | -0.19 [-0.31, -0.07] | inside |
| shared clones (n) | **39** | 57 [44, 72] | **outside** |
| D4 gap, same day / other day (data) | -0.39 (20) / -0.33 (16) | - | similar |

- **Pre-registered reading: mixed.** The misfit is confined to the smallest G samples (D2) and
  carried by a contamination-like pattern (D3: 6 shared clones whose 2-3 G-CSF cells are all
  monocytes while the clone is at least 5x larger in the M-CSF well; the model makes 0-3). But the
  D4 gap is as large when the M-well cells come from the other day, and same-library
  contamination needs the same day. So the signature is not same-library contamination.
- **What else fits.** A real minority of clones that are strongly M-biased and expand under
  M-CSF but barely survive under G-CSF (a few monocytes left). That is selection against M-biased
  clones under G-CSF, stronger at the extreme than the model's logistic gives. A second,
  independent misfit fits this too: the model makes too many shared clones (57 vs 39). M-biased
  clones are present in the G well too often in the model.
- **Decision.** Per the rule, B7 G stays in the primary with a caveat: it rests on 18 + 18 small
  clones. From now on it is computed on the fit set. A model that drops G-CSF presence of strongly
  M-biased clones (survival, not just fate) is a candidate, pre-registered separately and tested
  on D2, D3 and shared-clone count together.

## B25: is the low M-CSF replicate agreement just sampling? (pre-registered 2026-11-04)

- **Why.** The plan was to down-weight M-CSF agreement statistics because B21's replicate
  libraries agree at 0.28 under M-CSF on day 6 (305 clones), against 0.85-0.91 for G-CSF and the
  full cocktail. But M-CSF day-6 samples are nearly pure monocyte, so N-share has almost no spread
  and binomial sampling alone can push the correlation down. The model already samples cells
  binomially. If sampling explains the 0.28, down-weighting would count the same noise twice.
- **Test (data only, one look).** For each condition and day with at least 2 libraries, use the
  B21 pairing (clones with at least 2 mature cells in both libraries). Null: within each clone,
  pool its cells over its libraries and re-deal N/M labels to the libraries at their observed
  cell counts (hypergeometric), 1000 times. This gives the agreement expected if the libraries
  differ only by sampling. Compare the observed r with the null 95%.
- **Readings, fixed now.**
  - M6 observed r inside the null 95%: sampling explains it. **No down-weighting**; the plan is
    dropped.
  - M6 observed r below the null 2.5%: extra library-level noise under M-CSF. The next model's
    loss inflates the SE of every statistic using the M-CSF well by sqrt(null r / observed r),
    pre-registered with that model.
  - G4, G6 and full6 serve as checks: they should lie inside their own nulls.

### B25 results (2026-11-04): sampling explains the low M-CSF replicate agreement

| condition/day | clones | mean N-share | observed r | sampling null mean [95%] |
|---|---|---|---|---|
| G4 | 51 | 0.72 | 0.91 | 0.82 [0.71, 0.91] |
| G6 | 74 | 0.85 | 0.85 | 0.76 [0.64, 0.86] |
| full6 | 122 | 0.18 | 0.84 | 0.81 [0.75, 0.86] |
| **M6** | 305 | **0.008** | **0.28** | **0.37 [0.11, 0.64]** |
| M4 | 18 | 0 | undefined | - |

- **Reading (pre-registered): sampling explains it. No down-weighting.** M-CSF day-6 samples are
  99% monocytes. With so little spread in N-share, binomial sampling alone puts replicate
  agreement anywhere from 0.1 to 0.6. The model already samples cells binomially, so the M-CSF
  statistics are weighted correctly as they are. The "down-weight M-CSF" plan is dropped.
- Checks pass: G4, G6 and full6 sit inside (at the top of) their own nulls. Replicate libraries
  differ by sampling alone, with no sign of library-level technical noise anywhere.
- Side note for B21's "replicate agreement" numbers: they are mostly sampling ceilings. Read them
  against these nulls, not as biological reproducibility.

### Tooling (2026-11-04)

- `run_sweep.sh` no longer skips a run's result silently when a local file of that name exists.
  It takes the run's file when the local one is missing or empty, and when they differ beyond
  line endings it keeps both (`<file>.run<id>`) and says so. Cause of B23's "0 new result
  files": a smoke test had left an empty `b23_gfi1.json`.

### Correction to B24's audit note (2026-11-04)

B24 said the simulation side of B7 used the fit set. **Wrong.** In E26-E41b (`presence_stats(sim)`
in e26, e28, e29, e30, e32, e33 and everything built on e33) the simulated B7 interval is also
computed on all clones. Data and simulation were compared like for like, so the misfit is not an
artefact of the comparison. The remaining issue is hygiene: B7 used the G-CSF and M-CSF wells of
the held-out shard-1 clones, which also supply half of each held-out (Epo/SCF) pair. It entered
no fitting loss, only the B7 pass/fail checks. `audit_b7_fitset.py` re-scores every saved fit on
the fit set only (results below).

### E42 design probe (2026-11-04): clone dropout under G-CSF abandoned before pre-registration

- **Design probed.** E38 plus G-CSF founder survival q = exp(-h * softplus(z0 - z)), which drops
  monocyte-biased clones from the G-CSF well (h = 0 is E38 exactly; checked bit-identical).
- **Probe.** E38's best gated parameters (e38_s1), z0 = -1, h = 0/1/3/6, 30 simulations each, fit
  set: shared clones 57 / 41 / 24 / 13 (data 39); B7 G -0.28 / -0.27 / -0.25 / -0.17 (data -0.38);
  D3 0.025 / 0.021 / 0.016 / 0.011 (data 0.15).
- **Reading.** Whole-clone dropout fixes the shared-clone count but moves B7 G and D3 the wrong
  way: it removes exactly the M-biased clones from the shared group, while the data keeps them
  present with 2-3 monocytes. Not run. No fit was made, so no look was spent. A mechanism has to
  shrink M-biased clones under G-CSF to a few cells without removing them.
- **Second probe (same day).** Same q applied to G-CSF expansion instead of founders, z0 = -1 / 0,
  h = 1/3/6: B7 G -0.25 to -0.07, D3 0.021 to 0. Same wrong direction: shrinking pushes M-biased
  clones below the 2-cell presence threshold, so they leave the shared group too. **Neither
  "G-CSF punishes M-biased clones" form can make D3.** The data needs clones that stay *present*
  under G-CSF with 2-3 monocytes while being large under M-CSF: a monocyte-committed class that
  expands under M-CSF only. E39's continuous version of that failed (G-M overshoot); a discrete
  class is untested. Probing stops here to avoid tuning against the fit-set statistics; any next
  model is pre-registered before any fit.

### B7 fit-set audit results (2026-11-04): B7 G was weaker than reported; several verdicts change

`audit_b7_fitset.py` re-simulated every saved fit E33-E41b (132 fits, 200 simulations each, same
seeds; the old all-clone path reproduces the stored intervals exactly) and recomputed both B7
checks on the fit set only. Fit-set data: B7 G -0.382 (all clones -0.373), **B7 M 0.044 (all
clones 0.084)**.

| family | gated | B7 G pass, gated: old -> new | B7 M pass, gated | primary pass, gated |
|---|---|---|---|---|
| E33 | 5 | 0 -> 0 | 3 -> 2 | 0 -> 0 |
| E34 | 5 | 0 -> 0 | 5 -> 3 | 0 -> 0 |
| E35 | 4 | 1 -> 2 | 4 -> 4 | 1 -> 2 |
| E37 | 2 | 0 -> 1 | 2 -> 2 | 0 -> 1 |
| **E38** | 9 | **2 -> 5** | 9 -> 9 | **1 -> 3** (s8, s11, s12) |
| E39 | 1 | 1 -> 1 | 0 -> 0 | 0 -> 0 |
| **E40** | 7 | **0 -> 2** (both s_G 4) | 6 -> 7 | 0 -> 0 |
| E41 | 1 | 0 -> 0 | 1 -> 1 | 0 -> 0 |
| E41b | 9 | 0 -> 0 | 9 -> 9 | 0 -> 0 |

- **Why it moves.** Half the clones means a wider simulated interval: E38's B7 G 95% interval is
  0.25 wide on the fit set against 0.18 on all clones, while the data value barely moves. The old
  comparison was like for like and more powerful. But that extra power came from the held-out
  clones, which the protocol keeps out.
- **What changes in the record.**
  - **B7 G is a marginal misfit, not a firm one.** Under the protocol E38 is inside in 5/9 gated
    seeds (stability needs 6/9). The claim "B7 G fails in every model" was wrong on the fit set:
    E38 and E40 at s_G 4 can pass it.
  - E38 primary: 3/9 gated seeds pass (was 1/9). Still not stable (G-M splits 5/4 and B7 G 5/9),
    so E38's verdict, "not stable", stands.
  - E40: B7 G passes at s_G 4 (2 of 3 seeds), fails at 2.5 and 6.5. That slightly favours the
    middle of the ridge.
  - E41b: unchanged (G-M fails in all 9); its discard stands. E39: unchanged.
  - E33/E34: B7 M is harder on the fit set (data 0.044); some former passes fail. Those models
    were already rejected.
- **What does not change.** B24's D2/D3 patterns were computed on the fit set: the excess of
  M-large / G-tiny clones (0.15 vs 0.02) and too many shared clones (57 vs 39) stand.
- **Fix.** `e33_measured.py` now computes both B7 checks on the fit set (data and simulation).
  Every model built on e33 inherits it. The older scripts (e26-e32) are left as they are, with
  their results superseded by this audit.
- **Held-out status.** Shard-1 clones' G-CSF and M-CSF wells entered these descriptive checks.
  The held-out statistic itself (shard-1 Epo/SCF pairs) was never computed or looked at.
  The test stays usable, but this exposure is noted for whoever reads the eventual result.

## E38 under the fixed B7: no re-run needed (2026-11-04)

B7 is not in the fitting loss; it is scored after the fit with 200 simulations on fixed seeds. A
re-run of E38 with the same seeds reaches the same parameters, and the audit already re-scored
them on the fit set with the same simulation seeds. **E38 under the protocol: 9/18 gated, B7 G
5/9, B7 M 9/9, primary 3/9 (s8, s11, s12), G-M 5/4. Not stable.** Recorded as E38's verdict.

## E43: a discrete monocyte-committed clone class (pre-registered 2026-11-04)

- **Data diagnostic it must explain (B24, fit set).** 15% of shared clones are large under M-CSF
  (at least 5x their G-CSF count) with only 2-3 cells under G-CSF, all monocytes; E38 makes 2%
  [0, 7%]. E38 also makes too many shared clones (57 [44, 72] vs 39), and its 2-cell-stratum B7 G
  is -0.22 vs -0.47. Two E42 probes showed that dropping or shrinking M-biased clones under
  G-CSF moves all of these the wrong way.
- **Model.** E38 plus a discrete class. Each clone is monocyte-committed with probability pi_M.
  A committed clone makes only monocytes in every condition, and its expansion under M-CSF is
  multiplied by exp(g_up). Everything else is E38. Two parameters: pi_M in (0, 0.3),
  g_up in (0, 4). Their lower bounds approach E38. 25 dimensions.
- **Fit.** Targets, loss and SE as E38 (B24's statistics are not targets, so they test the model
  out of fit). 18 cold seeds, up to 500 generations, parallel workers, early stopping not before
  generation 200. Early-stop guard on: cancel once more than 9 seeds fail the gate.
- **Direction check before dispatch.** At E38's best gated parameters, set pi_M = 0.1 and
  g_up = 2 for 30 simulations and record D3, the shared count and B7 G. It is reported whatever
  it shows. It is not used to choose values; it only decides whether the run is worth spending.
  If D3 does not rise and the shared count does not fall, E43 is abandoned unrun, like E42.
- **Gate, primary, stability: as E38, with B7 on the fit set.**
- **Readings, fixed now.**
  - **Class used:** median pi_M of gated seeds > 0.02 and median g_up > 0.5.
  - **Explains B24:** for the best gated fit, D2 (G size 2), D3 and the shared count all inside
    the simulated 95% (track_b24 statistics, fit set, 200 simulations).
  - **Better than E38:** stability met (>= 9/18 gated, each primary component in >= 2/3 of
    gated seeds). If met, E43 becomes the reference, and the held-out power check is next.
  - Class used but B24 not explained: the class is absorbing something else. Report it; no
    reference change.
- **Direction check result (before dispatch).** e38_s1 parameters, 30 simulations, fit set.
  pi_M 0 / 0.1 with g_up 0 / 2: D3 0.021 -> **0.114** (data 0.154); shared clones 56 -> 58
  (data 39); B7 G -0.28 -> -0.30 (data -0.38); D2 G-size-2 -0.23 -> -0.24 (data -0.47). D3 rises,
  so the pre-registered rule says run. The shared-clone excess is not addressed at these values.

### E43 results (2026-11-04): negative. The class is used to the cap but breaks the Epo/SCF wells

- **Guard cancelled the run** after 10 finished seeds had all failed the gate (rule: more than 9
  fails). 11 result files were recovered (seeds 0-4, 6, 7, 11, 12, 15, 16); none gated.
- **Class used, maximally.** pi_M 0.25-0.29 in every seed (upper bound 0.3), g_up 0.9-3.0.
  Readings: "class used" yes; "explains B24" and "better than E38" cannot be assessed (no gated
  fit).
- **Why the gate fails.** Gate failures are mostly Epo and SCF (mean and n) and G mean; the
  largest misfits are SCF var (z -5.3), SCF mean (+4.5), Epo var (-4.1), M n (-3 to -7). The
  pre-registered class makes monocytes only **in every condition**, but under Epo and SCF
  nearly every mature cell is a neutrophil, so a quarter of clones committed to monocytes
  everywhere is incompatible with those wells. A design error in the pre-registration (the
  Epo/SCF consequence was not checked); recorded, not patched.
- **What the fit was telling us anyway (descriptive, no gated fit).** With the class, B7 G
  passes in 7 of 11 seeds and full-G/full-M co-presence lands near the data (17-24 / 30-39 vs
  16 / 33). Best seed (s4: loss 72, adequate, fails only the Epo gate) has pi_M 0.26,
  g_up 0.86 and s_G 3.6. The G/M/full wells want a large, persistent, monocyte-leaning
  clone class; the Epo/SCF wells forbid it being monocyte-only there.
- **Decision: discard E43 as specified.** The candidate that remains is a clone class whose
  monocyte commitment applies under the myeloid cytokines (G, M, full) and not under Epo/SCF.
  That is close to a condition-specific fate term, which E38's tau already provides, so it must
  be argued from data before any pre-registration. Not started.

## B26: what do monocyte-committed-looking clones make under Epo and SCF? (pre-registered 2026-11-04)

- **Why.** E43 wanted about 27% of clones committed to monocytes in the G-CSF, M-CSF and full
  wells, but the Epo/SCF wells forbid those clones being monocyte-only there. Two readings
  differ in what comes next: (a) the same clones make neutrophils under Epo/SCF, so commitment
  depends on the condition; (b) the same clones are rarely present under Epo/SCF, so the
  problem is presence, not fate.
- **Data.** Fit set. Myeloid wells = full, G, M. Present = at least 2 mature N+M cells in that
  well, pooled over days.
  - **A (M-pure):** present in at least 2 myeloid wells, with no neutrophil in any of them.
  - **B (mixed or N):** present in at least 2 myeloid wells, with at least one neutrophil.
- **Statistics, each reported for data and for the best gated E38 fit (e38_s1, 200
  simulations, 95% interval).**
  - C0: number of A clones, and A / (A + B).
  - C1: share of A and of B clones present (at least 2 cells) in Epo or SCF, within strata of
    total myeloid size (2-5, 6-15, 16+ cells).
  - C2: mean N-share in Epo/SCF of A clones present there, and of B clones.
- **Readings, fixed now.**
  - **Condition-dependent commitment:** data C2 for A at least 0.5 (they make mostly
    neutrophils under Epo/SCF). Then, if E38 reproduces C2 for A (data inside the simulated 95%)
    and only C0 is off, E38's per-condition fate term already handles it and the issue is how
    many such clones there are. If E38 misses C2 too, a myeloid-only commitment class is
    warranted (pre-registered separately).
  - **Presence problem:** A's presence in Epo/SCF is below half of B's in every size stratum
    with at least 10 A clones. Then the next model needs clone-level survival that differs
    between myeloid and Epo/SCF conditions.
  - Otherwise: mixed. Report it.

### B26 results (2026-11-04): formally "condition-dependent", but it rests on 1 clone; the real finding is that Epo/SCF barely share clones with the myeloid wells

| statistic (fit set) | data | E38 sim mean [95%] |
|---|---|---|
| A: M-pure clones in >= 2 myeloid wells | **38** | 55 [41, 69] (out) |
| B: mixed/N clones in >= 2 myeloid wells | 38 | 49 [36, 63] |
| A clones also present in Epo/SCF | **1** | 1 [0, 3] |
| that clone's Epo/SCF N-share | 0.5 | 0.01 [0, 0] (out) |
| A clones by myeloid size, 2-5 / 6-15 / 16+ | 9 / 24 / **5** | 17 / 37 / **0.8 [0, 3]** |
| present in Epo/SCF, A vs B, 2-5 cells | 0.11 vs 0.25 | 0.02 vs 0.06 |
| present in Epo/SCF, A vs B, 6-15 cells | 0 vs 0.08 | 0.02 vs 0.05 |

- **Pre-registered reading: "condition-dependent; E38 misses it; class warranted".** That is
  formally what the rule gives. **It rests on one clone** (the only A clone seen under Epo/SCF,
  N-share 0.5). The rule set no minimum count, which was a flaw; the reading is uninformative.
- **What the data does say.**
  1. Epo/SCF wells and myeloid wells share very few clones (A 1/38; B a few per stratum), so the
     Epo/SCF data cannot tell whether monocyte commitment depends on the condition.
  2. E43 broke the Epo/SCF statistics because its class also applied to the (mostly separate)
     clones seen only under Epo/SCF. A class acting only in the myeloid wells is not ruled out by
     anything here.
  3. E38's M-pure clones have the wrong size distribution: too many small ones (2-15 cells:
     33 vs 54) and too few large ones (16+: **5 vs 0.8**). This is the same signal as B24's D3:
     large, persistent, monocyte-only clones that E38 cannot make.
- **Decision.** Candidate E44 = E43 with the class acting in the myeloid wells only (full, G, M),
  judged on the same gate and primary plus this table (A size distribution) and B24's D3.
  Pre-registration deferred to the next session, with a minimum-count rule for every reading.
- **Lesson (process).** Every pre-registered reading gets a minimum sample size from now on.

## S2: learned action predictor on LARRY vs E38 (pre-registered 2026-11-04)

- **Question.** Does a flexible predictor that sees a clone's whole day-2 transcriptome, plus the
  cytokine (the "action"), predict the clone's neutrophil share per condition better than E38,
  which sees only three day-2 numbers (m, fn, fm)? If yes, E38 misses fate information in the
  day-2 state. A second question: does a DINO-style encoder trained with **sister cells as views**
  beat plain PCA?
- **Prior art (abstract and discussion read).** LCL (Yang, Wang & Lin, bioRxiv 2024 v2,
  PMC13228636) uses lineage barcodes as natural augmentations: SimCLR-style contrastive learning
  with sister cells as positives and negatives from other lineages. It is tested on LARRY
  (41,093 cells, which subset is not stated) and CellTag for predicting future cell-type
  composition, and beats scVI (LARRY top-200 lineages: KL 0.335 vs 0.549). It does not
  condition on cytokine and is not compared with a mechanistic model. Its stated limitation is
  that small lineages predict poorly. New here: a negative-free (DINO) objective, the cytokine as
  action, and a head-to-head with E38.
- **Units.** Fit-set clones (held-out shard untouched) with at least 1 day-2 cell. For each
  condition (full, G, M primary; Epo, SCF reported), a clone enters if it has at least 1 mature N+M
  cell there. A condition is scored only if at least 50 clones enter it.
- **Encoders (day-2 state of a clone).**
  - (i) **m3:** E38's own inputs (m, fn, fm).
  - (ii) **PCA:** 2000 most variable genes (log1p normalised), PCA-50 fitted on training-fold
    day-2 cells, clone mean.
  - (iii) **DINO-sister:** DINO-style student/teacher MLP (50 -> 128 -> 64 backbone, 64-prototype
    head, teacher EMA 0.996, temperatures 0.1 / 0.04, centre momentum 0.9). Input is the PCA-50 of
    a cell. The two views are two different day-2 cells of the same clone (the same cell if the
    clone has one), each with Gaussian noise of 0.1 x the feature SD. 30 epochs, batch 256,
    Adam 1e-3, trained on training-fold cells only. Embedding = teacher backbone, clone mean.
  - The pretrained foundation-model encoder is dropped. No mouse model runs on this CPU within
    the session, so it is recorded as not done.
- **Predictor.** For each condition, a weighted ridge logistic regression on standardised
  features (cell-weighted mean BCE + 0.01 ||w||^2), LBFGS. Fixed hyperparameters, no tuning.
- **E38 reference.** Best gated E38 fit (e38_s1), 300 simulations. A clone's predicted N-share
  per condition is (sum N + 0.5) / (sum N+M + 1) over simulations. E38's parameters were fitted to
  summary statistics of these same clones, so it gets a mild in-sample advantage. Noted, not
  corrected.
- **Null.** The training-fold pooled N-share of each condition.
- **CV and metric.** 5 folds by clone (seed 0). Score = held-out per-cell binomial deviance,
  pooled over the primary conditions (lower is better). Paired clone bootstrap (2000) of
  differences.
- **Readings, fixed now.**
  - **R1:** best of PCA / DINO-sister minus E38: 95% CI below 0 means E38 misses day-2
    information.
  - **R2:** m3 minus E38: CI below 0 means E38's mechanism loses information it has.
    CI above 0 means the mechanism adds structure beyond a GLM on the same inputs.
  - **R3:** DINO-sister minus PCA: CI below 0 means sister-shared structure carries fate
    information that PCA misses.
  - Anything with fewer than 50 clones in a condition is not read.
- **Amendment before the real run (2026-11-04).** The smoke test (real counts, synthetic
  expression) showed that the pre-registered E38 estimator, which pools cells over simulations,
  is dominated by simulations where a clone expands hugely under selection. That pushes
  per-clone predictions to the extremes (E38 0.38 nats/cell worse than null), although E38's
  average calibration is fine (G clone-mean 0.55 vs 0.57). **Changed to E38's native
  predictive:** for each clone, the mixture likelihood of its observed (k, n), averaged over the
  simulations in which it is present. **Disclosure:** the smoke test used real counts for m3 and
  E38, so **R2 (m3 vs E38) has been seen and is spent**: it will be reported, not read as a test.
  R1 and R3 need the real day-2 expression and have not been seen.
- **Second amendment, same reason, before the real run.** The mixture score still sampled each
  simulation's cells twice (shares from a few simulated cells). Replaced it with E38's latent
  rates: e38_latent() mirrors e37.simulate, and its expected counts match the simulator (G
  neutrophils 1,587 vs 1,614; M 167 vs 169 over 40 runs). The predictive is
  sum_s w_s Binom(k | n, q_s), with q_s = lamN / (lamN + lamM) and w_s proportional to Poisson(n | lamN + lamM).
- **Observation from the spent R2 side (real counts, no expression).** Even with the corrected
  predictive, E38's per-clone fate deviance on the fit set is worse than a constant (G 0.94 vs
  0.63, M 0.52 vs 0.17, full 1.00 vs 0.60 nats/cell). The excess comes from confident misses:
  under M-CSF, pure-monocyte clones of 10-17 cells that E38 expects to be neutrophil-rich
  (+68 to +117 nats each); under G-CSF, pure-N clones it expects to be monocytic. E38's steep
  fate curve (beta about 9) makes near-deterministic per-clone calls. It matches population
  statistics but is **not calibrated per clone**. Reported, not a test (R2 is spent).

### S2 results (2026-11-04): day-2 transcriptome beats a constant; E38 and DINO-sister add nothing proven

1,127 fit-set clones with 1,927 day-2 cells (most clones have **one** day-2 cell, so true sister
views were rare). Held-out (5-fold by clone) per-cell binomial deviance:

| | full (236 clones) | G (175) | M (237) | **primary pooled** |
|---|---|---|---|---|
| null (constant) | 0.533 | 0.641 | 0.146 | 0.357 |
| m3 (E38's 3 inputs, GLM) | 0.523 | 0.587 | 0.125 | 0.331 |
| PCA-50 | 0.436 | 0.632 | 0.113 | 0.314 |
| DINO-sister | 0.433 | 0.642 | 0.108 | **0.313** |
| E38 (latent predictive) | 0.538 | **0.557** | 0.249 | 0.392 |

- **R1 (best learned minus E38): -0.079, 95% CI [-0.183, +0.008]. Not supported**: the CI crosses 0.
- **R2 (m3 minus E38): -0.061 [-0.155, +0.014]. Spent** (seen in the smoke test); reported only.
- **R3 (DINO-sister minus PCA): -0.001 [-0.026, +0.024]. Not supported.** With mostly
  single-cell clones, the sister objective reduces to noise augmentation, so a tie is expected.
- **Versus the constant:** PCA -0.043 [-0.077, -0.007], DINO -0.044 [-0.083, -0.000],
  m3 -0.026 [-0.044, -0.004]. All learned predictors carry real but modest information (about 12%
  of the null deviance). E38 +0.035 [-0.047, +0.134]: no better than a constant per clone.
- **Per condition.** E38 is the best predictor under G-CSF (0.557, where its selection term
  works) and the worst under M-CSF (0.249 vs 0.146: the confident misses on large M-pure
  clones, as in the R2-side observation). The full-cocktail gain comes from the whole
  transcriptome (PCA/DINO 0.43 vs m3 0.52): there, day-2 state beyond Cebpa-Irf8 matters.
- **Reading.** No pre-registered claim passes. Descriptively: (1) the day-2 transcriptome holds
  fate information beyond E38's three inputs in the full cocktail; (2) E38 is miscalibrated per
  clone under M-CSF, in the same direction as B24/B26, which supports E44's target; (3) DINO-style
  pretraining needs more cells per clone than this experiment has (LCL's own stated limitation).

## E44: monocyte-committed class acting only in the myeloid wells (pre-registered 2026-11-04)

- **Data diagnostics it must explain (fit set).** B24 D3: 15% of shared clones are large under
  M-CSF with only 2-3 monocytes under G-CSF (E38: 2%). B26: monocyte-pure clones with 16+ myeloid
  cells, 5 in the data vs 0.8 in E38. S2 side observation: E38 is miscalibrated per clone
  under M-CSF. E43 showed the fit wants such a class (pi_M at its cap) but broke Epo/SCF, because
  the class also acted there. B26 showed Epo/SCF and the myeloid wells share almost no clones,
  so a class confined to the myeloid wells is not contradicted by any data seen so far.
- **Model.** E43 with one change: the class (monocytes only, M-CSF expansion x exp(g_up)) acts
  only under full, G and M. Under Epo/SCF a committed clone behaves like any other. pi_M in
  (0, 0.5) (E43's cap of 0.3 was binding), g_up in (0, 4). 25 dimensions.
- **Every-condition check before registration (the new rule).** At e38_s1's parameters,
  pi_M 0.27, g_up 2, 30 simulations: Epo/SCF targets barely move (Epo mean 0.68 -> 0.71, data
  0.70; SCF mean 0.68 -> 0.69, data 0.66; Epo/SCF n 22/42 -> 20/43, data 23/39); D3 0.02 -> 0.26
  (data 0.15); 16+ monocyte-pure clones 0.9 -> 29 (data 5); shared clones 56 -> 57 (data 39,
  not addressed by this mechanism).
- **Fit.** As E43: 18 cold seeds, 500 generations max, early stopping not before 200, E38
  targets/loss/SE, B7 on the fit set. **Guard on (cancel after more than 9 gate failures).**
  The local guard stops while this PC sleeps; the run itself continues on Actions.
- **Gate, primary, stability: as E38** (gated >= 9/18; each primary component agrees in >= 2/3
  of gated seeds).
- **Readings, fixed now, each with its minimum count.**
  - **Class used:** median pi_M > 0.02 and median g_up > 0.5 over gated seeds (at least 5
    gated seeds, otherwise not read).
  - **Explains the diagnostics:** for the best gated fit, all of the following lie inside the
    simulated 95%: B24 D3 (39 shared clones in the data), B24 D2 2-cell stratum (18 shared),
    B26 count of 16+ monocyte-pure clones (5), and the shared-clone count (39). Read only if a
    gated fit exists.
  - **Better than E38:** stability met. Then E44 becomes the reference and the held-out power
    check comes next.
  - Class used but the diagnostics not explained: report it; no reference change.
  - The S2 per-clone comparison (E44 vs E38 latent predictive) is **not** part of this
    registration. It needs its own.

### E44 results (2026-11-05): negative. 2/18 gated; the class is used heavily, but the gate breaks on Epo and G means

- **Run.** All 18 seeds completed on Actions (run 41921619454). The local watcher died while the
  PC slept, so the guard judged only 7 seeds (all fails) and never cancelled; line artifacts were
  fetched afterwards. Scored with analyze_e44.py (fixed from E43's script plus the B26 16+ count).
- **Gate: 2/18** (s6, s17). Stability not met. Gate failures over 18 seeds: **Epo mean 15**,
  **G mean 14**, SCF mean 9, Epo n 4, SCF n 4, full n 3, M n 3. Although the class does not act
  under Epo/SCF, the shared parameters shift to fit a large class (pi_M 0.31-0.47, median 0.36;
  g_up median 1.6), and the Epo/G means move off.
- **Readings.**
  - Class used: **not read** (2 gated seeds, minimum 5). Descriptively, every seed uses it
    (pi_M >= 0.31).
  - Explains the diagnostics (best gated, s17): **no.** B26 16+ M-pure 5 vs 2.6 [0, 6] inside;
    B7 G inside; but D3 0.15 vs 0.05 [0, 0.11], D2 2-cell -0.47 vs -0.27 [-0.45, -0.08], and
    shared clones 39 vs 66 [52, 81] are all outside.
  - Better than E38: **no** (stability not met).
- **Worth noting (one seed, no claim).** s17 (pi_M 0.31, g_up 1.0, s_G 4.2) is gated and passes
  the whole primary, G-M included. s6 passes 4/5. A moderate class may be compatible with the
  gate; large ones are not, and the optimiser prefers large ones.
- **Decision: discard E44 as specified.** Shared-clone excess (now 66 vs 39) remains the
  misfit no class fixes; it points at presence (how many founders a clone puts into each
  well), not fate. The next step is a data diagnostic on presence, not another class.

## B27: how clones spread across wells (presence), data vs E38 (pre-registered 2026-11-05)

- **Why.** Every class model (E43, E44) leaves the same misfit: too many clones shared between
  G-CSF and M-CSF (E38 57, E44 66, data 39). That is about presence (how a clone's founders are
  split between wells and whether they survive), not fate. The data has to say what is wrong
  before any model is built.
- **Data.** Fit set. A clone is present in a well with at least 2 mature N+M cells (pooled
  days). Myeloid wells: full, G, M. Day-2 size n2 = number of the clone's sampled day-2 cells
  (b14 table; 0 if none). Reference: best gated E38 fit (e38_s1), 200 simulations, same
  statistics, same clone ids (n2 taken from the data for both).
- **Statistics (each: data vs simulated 95%).**
  - **P1 spread.** Among clones present in at least one myeloid well: the fraction present in
    exactly 1, 2 and 3 myeloid wells.
  - **P2 lift.** For G-M, full-G and full-M: co-present count / expected count if presence were
    independent across wells (product of marginal rates x clones present anywhere).
  - **P3 day-2 size.** Presence rate in G and in M by n2 = 0, 1, 2+ (strata with at least 30
    clones only).
  - **P4 marginal presence.** Number of clones present in each myeloid well.
- **Readings, fixed now (minimum: a statistic is read only if its data count is at least 30).**
  - **Too much shared presence:** data G-M lift below the simulated 95%, with P4 G and M counts
    inside. E38's cross-well sharing (shared expansion a, pool u) is too strong. Next model:
    weaker cross-well coupling.
  - **Missing founder heterogeneity:** presence rises with n2 in the data (2+ vs 0 differs by at
    least 0.1 in G or M) while E38's simulated rates do not. Next model: founder rate tied to
    day-2 clone size.
  - **Marginals wrong:** a P4 count outside the simulated 95%. The excess is marginal, not a
    coupling; fix presence per well first.
  - Several can hold; report all. None holding means the shared-clone excess is not localised
    by these statistics.

### B27 results (2026-11-05): G and M wells avoid sharing clones more than E38 allows; M presence too low

| statistic (fit set) | data | E38 sim [95%] |
|---|---|---|
| present in full / G / M | 368 / 277 / **478** | 365 / 271 / **428 [395, 457]** (M out) |
| in 1 / 2 / 3 myeloid wells | **0.927 / 0.067** / 0.006 | 0.891 / 0.105 / 0.004 (1, 2 out) |
| G-M co-present (lift) | **39 (0.31)** | 57 (0.47 [0.38, 0.56]) (out) |
| full-G co-present (lift) | 16 (0.16) | 21 (0.20), inside |
| full-M co-present (lift) | 33 (0.20) | 34 (0.21), inside |
| presence in G by day-2 cells 0 / 1 / 2+ | 0.17 / 0.14 / 0.18 | 0.21 / 0.22 / 0.22 |
| presence in M by day-2 cells 0 / 1 / 2+ | 0.29 / 0.23 / 0.32 | 0.34 / 0.33 / 0.34 |

- **Pre-registered reading: "marginals wrong"** (M presence 478 vs 428 [395, 457]). The "too much
  shared presence" reading did not formally fire, because its condition required the G and M
  marginals to be inside, and M is not.
- **Descriptive (not a pre-registered reading).** The marginal error goes the *other* way from the
  shared excess. E38 has fewer M-present clones yet more G-M shared clones, so raising M
  presence would make the excess worse. The G-M coupling is too strong: the data's lift is 0.31
  vs 0.47. The full-arm lifts are right (E37's separate pool fixed those).
- **No founder-size signal.** Presence does not rise with the number of sampled day-2 cells in
  the data (G 0.17 / 0.14 / 0.18), so sampled day-2 size is not a usable founder proxy.
- **What this points to.** Within the main pool, G and M founders behave as if they compete:
  a clone that seeds the G well is less likely to seed the M well. That is what splitting a
  finite number of sister cells between wells (multinomial allocation) produces, unlike E38's
  independent Poisson founders per well. E32 tried multinomial allocation for all wells and
  failed on the full arm; with E37's separate full pool, allocation within the main pool only
  (G, M, Epo, SCF) is untested. **Candidate E45: E38 with multinomial founder allocation within
  the main pool and a free M-presence rate.** It must be checked against every condition before
  registration.

## E45: limited sister cells split between the main-pool wells (pre-registered 2026-11-05)

- **Data diagnostic it must explain (B27, fit set).** G-M co-presence lift 0.31 in the data vs
  0.47 [0.38, 0.56] in E38 (39 vs 57 clones), while E38 has too *few* M-present clones (428
  [395, 457] vs 478). Full-arm lifts are already right.
- **Why this mechanism.** E38 draws founders independently per well. Splitting a Poisson
  number of founders is mathematically the same thing (Poisson splitting), so competition
  between wells needs a *limited* number of sisters per clone at the split. E45: each clone has
  K = 1 + Poisson(lam_K) sister cells; each goes to G, Epo or SCF with probability r, to M with
  r * w_M, or is lost. r = nu / (1 + lam_K), capped at 0.999 / (3 + w_M); u clones (full pool)
  use nu * eps. The full arm keeps E37's separate pool. lam_K in (0, 20), w_M in (0.25, 4).
  25 dimensions. lam_K large with w_M = 1 recovers E38 (checked: lam_K 20 gives lift 0.48,
  E38 0.47).
- **Every-condition check (e38_s1 parameters, 30 simulations).** lam_K 2 / 0.5 / 0.2 (w_M 1-1.4):
  G-M lift 0.43 / 0.20-0.23 / 0.10 (data 0.31); full-G and full-M lifts unchanged (0.20-0.22);
  at fixed nu, presence in G, M, Epo and SCF falls (G 275 -> 145-198, data 277) because the cap
  limits founders per well, so the fit has to compensate through expansion (mu) and w_M. The
  targets that move off at fixed parameters (G/M/Epo/SCF n) are all in the loss. D3 falls
  (0.02 -> 0.00-0.01; data 0.15): **this mechanism does not address D3**, and that is expected.
- **Fit.** As E44: 18 cold seeds, 500 generations max, early stopping not before 200, E38
  targets/loss/SE, B7 on the fit set. Guard on (more than 9 gate fails). The PC stays awake for
  the run.
- **Gate, primary, stability: as E38.**
- **Readings, fixed now (minimum counts in brackets).**
  - **Competition used:** median lam_K over gated seeds < 5 (at least 5 gated seeds).
  - **Explains B27:** for the best gated fit, the G-M co-present count (39) and lift, and the
    M-present count (478), all inside the simulated 95% (track_b27 statistics, 200 simulations).
  - **Better than E38:** stability met. Then E45 becomes the reference, and the held-out power
    check comes next.
  - **Side readings, reported:** B24 D3 and B26's 16+ M-pure count for the best gated fit. They
    are not expected to be fixed by this mechanism.

### E45 result (run 41998556690, analysed 2026-11-05)

- **Run.** The guard cancelled at 3 gated / 10 failed; 13 of 18 seeds finished (s10, s11, s13,
  s17, s18 cancelled). The watcher's exit 1 is the cancel, not an error.
- **Gate 3/13** (s3, s14, s16; E38 9/18). **Primary 0/3.** All three pass G-M, full-G and
  full-M; B7 G fails in all three (E38 5/9) and B7 M passes in 2/3. **Not stable** (fewer than 9
  gated). **Worse than E38.**
- **Competition used: not readable** (3 gated seeds, at least 5 required). For the record, gated
  lam_K is 0.99 / 1.94 / 16.25 (median 1.94); 7 of the 13 seeds fitted lam_K < 2.5.
- **Explains B27: no, narrowly.** Best gated fit (s14, lam_K 0.99, w_M 1.27): G-M co-present 39
  inside [39, 69], M-present 478 inside [448, 517], lift 0.307 just below [0.313, 0.482]. The lift
  moved most of the way from E38 (0.47) toward the data (0.31), and the M-presence count is now
  right.
- **Side readings.** D3 0.15 vs 0.03 [0, 0.08] and 16+ M-pure clones 5 vs 1.7 [0, 4]: still out,
  as expected. B7 G -0.38 vs -0.24 [-0.38, -0.09] is now just out.
- **Conclusion.** Limited sisters per clone do what B27 asked (fewer shared G/M clones, enough
  M-present clones) at small lam_K, but the fit then loses the gate on most seeds and B7 G on the
  gated ones. As registered, E45 is negative and E38 stays the reference. The trade-off between
  shared presence and the B7 G difference is itself a lead: both are statistics of the same
  G/M sister pairs.

## B28: barcode detection by cell type, and a design check of the day-2 split (pre-registered 2026-11-05)

- **Why.** The critique of 2026-11-04 flagged detection bias: neutrophils carry little mRNA, so
  their LARRY barcode (a GFP transcript) may be captured less often than monocytes'. Every E-series
  statistic counts barcoded cells only, so a per-type capture difference would bias N shares,
  presence and B7 G directly. The Weinreb SI (read 2026-11-05) gives the design: about 6,000
  barcoded founders (20,000 x 30%), about 3 divisions by day 2, 25% of day-2 cells sequenced,
  18.75% to each well. All later cells were GFP+ sorted, so a profiled cell without a barcode is a
  capture failure, not an unbarcoded cell.
- **Data.** data/larry_cytokine metadata (all profiled cells, barcoded or not) and clone matrix.
  Fit-set restriction does not apply: Part A uses no clone identity, and Part B uses counts only.
- **Part A, detection.** For day 4 and 6 cells in full, G and M: the fraction of cells with a
  clone barcode, by annotation (Neu, Mo, and other types), per condition and pooled; 95% CI by
  bootstrap over libraries. Statistic: ratio Neu / Mo detection.
  - **Reading "detection bias":** pooled ratio CI entirely outside [0.8, 1.25], at least 300 cells
    of each type. Then the E series must model per-type capture before any further fate claim.
  - **Reading "no material bias":** pooled ratio CI inside [0.8, 1.25].
  - Otherwise "unresolved". Per-condition ratios are reported (at least 300 cells per type).
- **Part B, design check (calibration, no pass/fail).** Report: barcoded day-2 cells and clones;
  the n2 distribution; the share of clones seen at day 4/6 in full/G/M that were also seen at day 2.
  From that share, solve for the per-cell day-2 capture d assuming K ~ 1 + Poisson(K0 - 1) cells
  per clone at day 2, for K0 = 4, 8, 16. Then report the post-split survival implied by E38's
  fitted founders per well (nu, median over gated seeds): s = nu / (0.1875 K0). Also report the
  libraries and conditions present at days 4 and 6, to settle whether an SCF well exists.
- **Use.** Part B sets fixed design inputs for E46 (pre-registered after B28). Part A decides
  whether E46 needs a capture term.

### B28 result (2026-11-05)

- **Part A, detection: formally "unresolved"; no sign that neutrophils are missed.** Pooled
  Neu/Mo detection ratio 0.91 [0.72, 1.20] (6,035 Neu, 16,619 Mo; 15 libraries), which crosses
  0.8. By condition: full 1.38 [1.19, 1.47] (Neu detected *more*), G 1.12 [0.98, 1.20], M 0.69
  [0.58, 1.05] (only 572 Neu). The pooled ratio below 1 is a mix effect: most Neu are in G wells,
  whose libraries capture barcodes less (Neu 0.40, Mo 0.36) than M wells (Mo 0.54). **Detection
  differs more by library/condition than by cell type.** Per-condition capture is binomial thinning
  of Poisson counts, so the existing per-condition terms (mu, s) already absorb it. **No capture
  term is needed in E46.**
- **Part B, design check.** Day 2: 15,728 cells profiled, 3,776 barcoded (24%), 2,222 clones;
  n2 = 1/2/3/4 in 1,063/845/233/81 clones (none above 4), mean 1.70 when seen. 3,835 clones seen
  later in full/G/M; 34.6% of them were seen at day 2. Implied per-cell day-2 capture d and
  post-split survival (from E38's gated median nu 0.546, 9 seeds):

  | K0 (cells per clone at day 2) | d | survival s = nu / (0.1875 K0) |
  |---|---|---|
  | 4 | 0.42 | 0.73 |
  | 8 | 0.21 | 0.36 |
  | 16 | 0.11 | 0.18 |

  Cross-check at K0 8: 20,000 x 8 x 25% = 40,000 cells sent to sequencing, 15,728 profiled (0.39),
  times barcode capture about 0.5 (Part A) gives d about 0.20, matching 0.21.
- **Conditions present:** full (days 2/4/6), G, M, Epo (4/6), **SCF (day 4 only)**. The SI text
  omits the SCF well; the data has it.
- **Consequence for E45 and B27.** With K0 4-16 and 18.75% per well, founders per well are close
  to independent Poisson (E45's check: lam_K 2 already gives lift 0.43). **Sister competition at
  the split cannot produce the data's low G-M co-presence (0.31) under the actual design.** It must
  arise after the split, from clone-specific outcomes in each well.

## E46: lineage-matched selective expansion (pre-registered 2026-11-05)

- **Data diagnostic it must explain.** B27: G-M co-presence lift 0.31 vs E38 0.47 [0.38, 0.56];
  B28 rules out split competition as the cause. Secondary: B24 D3 (0.15 vs 0.02) and B7 G.
- **Mechanism.** In the G-CSF well, a clone's log expansion gains +gamma * zb; in the M-CSF well
  it gains -gamma * zb (zb = the clone's fate bias in that well, positive = neutrophil-leaning).
  Full, Epo and SCF unchanged. Biologically: each cytokine expands the clones whose progeny depend
  on it, which is clone-level **selection**. One new parameter, gamma in (0, 3):
  gamma = 3 sigmoid(x23). 24 dimensions; everything else as E38 (e37 simulator).
- **Every-condition check (e38_s1 parameters, 30 simulations; scratchpad e46chk).**

  | gamma | G-M lift | n G-M | presence full/G/M/Epo/SCF | B7 G | D3 | N share G / M / full |
  |---|---|---|---|---|---|---|
  | data | 0.31 | 39 | 368/277/478/23/39 | -0.38 | 0.15 | 0.64 / 0.054 / 0.31 |
  | 0 (E38) | 0.48 | 57 | 371/269/422/22/44 | -0.25 | 0.02 | 0.64 / 0.044 / 0.29 |
  | 1 | 0.36 | 43 | 371/280/416/23/41 | -0.28 | 0.06 | 0.74 / 0.021 / 0.29 |
  | 2 | 0.32 | 37 | 371/285/393/21/42 | -0.29 | 0.08 | 0.74 / 0.026 / 0.29 |

  Lift, B7 G and D3 all move toward the data; full, Epo and SCF do not move. At fixed parameters
  the G N-share rises and the M N-share and M presence fall; tau_G, tau_M, s and nu/mu, all fitted,
  can compensate. Checked first and rejected: decorrelating expansion between wells (sigma_a is
  0.19, so it changes nothing: lift 0.47 at every correlation).
- **Fit.** As E45: 18 cold seeds, 500 generations max, E38 targets/loss/SE, B7 on the fit set,
  guard on (more than 9 gate fails), PC kept awake.
- **Gate, primary, stability: as E38.**
- **Readings (minimum counts in brackets).**
  - **Selection used:** median gamma over gated seeds >= 0.5 (at least 5 gated seeds).
  - **Explains B27:** for the best gated fit, G-M co-present count and lift, and M-present count,
    inside the simulated 95% (track_b27 statistics, 200 simulations).
  - **Better than E38:** stability met; E46 becomes the reference and the held-out power check
    comes next.
  - **Instruction vs selection (the project question), reported:** median instruction shifts
    tau_G - tau_full and tau_full - tau_M over gated seeds, E46 vs E38 (at least 5 gated seeds in
    each). If gamma >= 0.5 and both shifts fall by at least a third, report "selection absorbs part
    of the instruction signal". This is descriptive until a stable model exists.
  - **Side readings:** B24 D3 and B26 16+ M-pure count for the best gated fit.

### Tooling (2026-11-05): fit speed-up, results unchanged

Profiled one E46 loss evaluation (10 simulations): 0.26 s, of which 45% was SHA-256 shard hashing
repeated for every clone on every call. Changes: `shard_of` cached (pure function), a one-day
fast path in `pooled`, and the count dictionaries in e37/e46 built from lists instead of numpy
scalars. Now 0.125 s (2.1x). Checked byte-identical against the previous commit on all 28 fit
statistics for 24 simulations (E38 best parameters; E46 at gamma 0, 0.7, 1.5, 2.5) and on the
observed targets. E46's running sweep (run 42012020382) uses the earlier commit; same results.

### E46 result (run 42012020382, analysed 2026-11-05)

- **Run.** Completed, 18/18 seeds (the watcher's exit 1 was not a run failure).
- **Gate 16/18** (E38 9/18; E45 3/13). **Primary 6/16** (E38 3/9): G-M 12/16, full-G 16/16,
  **full-M 7/16**, B7 G 16/16 (E38 5/9), B7 M 13/16.
- **Stability: not met**, only because full-M splits 7 pass / 9 fail (needs 2/3 agreement, i.e.
  at least 11 of 16). Every other component meets it. **As registered, E46 is not "better than
  E38"; E38 stays the reference** and the held-out test stays unspent.
- **Selection used: yes.** gamma is tightly identified: 1.21-1.35 in all 16 gated seeds (median
  1.29). With it, s_G (the uniform N-output boost under G-CSF) drops from about 4.7 (E38) to a
  median 0.63. The selection effect moves from "G-CSF scales neutrophil output for every clone"
  to "each cytokine expands the clones of its own lineage".
- **Instruction vs selection: "not absorbed".** Median instruction shifts, E38 (9 gated) vs E46
  (16 gated): tau_G - tau_full 0.49 vs 0.86; tau_full - tau_M 0.47 vs 4.79. The M shift is
  poorly identified in E46 (0.6-7.5 across seeds), so it is a ridge, not an estimate. No
  instruction claim is made.
- **Explains B27: no, on one of three.** Best gated fit (s11): G-M co-present 39 inside [31, 54],
  lift 0.31 inside [0.27, 0.45], but M-present clones 478 above [406, 465] (E38's old shortfall
  remains).
- **Side readings.** B7 G inside (-0.38 vs [-0.55, -0.26]); B24 D2 inside; B26 16+ M-pure clones
  inside (5 vs [0, 6]); **D3 still out** (0.15 vs [0, 0.08]).
- **Conclusion.** Lineage-matched selective expansion is the first mechanism that fixes the gate
  rate, B7 G and co-presence together, and it is consistently identified. What remains: the
  full-M primary component (unstable across seeds), too few M-present clones, and D3. The M-side
  instruction parameter sits on a ridge. Next: a data diagnostic for full-M before any E47.

## Audit of the primary test (2026-11-05, prompted by Bence: "isn't this test suspicious?")

Checked before any re-scoring:

1. **Not circular.** The primary concordances (G-M, full-G, full-M r) are not loss targets
   (e36.targets: within-condition stats, presence, overlap, full-arm co-presence counts).
2. **Pass/fail rewards vague models.** A component "passes" when p >= 0.05, i.e. when the observed
   value is inside the simulated spread. E46's G-M pass rate rose (E38 p median 0.06 -> E46 0.35)
   with **no change in the simulated centre** (0.42 -> 0.41, data 0.51): the 95% width grew from
   0.22 to 0.40. B7 G is a genuine improvement (centre -0.27 -> -0.41, data -0.38) but also wider
   (0.18 -> 0.27). A normal-approximation CRPS (lower is better; descriptive, not registered):
   E38 sum 0.33, E46 0.49; E46 better on B7 G (0.023 vs 0.080), equal on G-M, worse on full-M
   (0.36 vs 0.15). **"E46 is the best model so far" was overstated** and is withdrawn.
3. **The primary uses held-out shard-1 clones**, the same flaw the B7 audit fixed. OBS (e20_gap:
   G-M 0.51, full-G 0.69, full-M 0.24) and the simulated cross_r are both computed on all clones.
   On the fit set the data values are G-M 0.37, full-G 0.90, full-M 0.43; on shard 1 they are
   0.62 / 0.47 / 0.08. The two halves disagree by 0.25-0.43, so these statistics (16-39 clones
   each) are very noisy, and the full-M "misfit" (0.24) is largely shard 1's 0.08.
4. The "fewer than 10 clones -> r = 0" sentinel in test_stats is rare here (0-2% of simulations).

## B29: re-score the primary on the fit set with a proper score (pre-registered 2026-11-05)

- **What.** For every gated seed of E38 (9) and E46 (16): 200 simulations (rng 1000+s, as in
  e33), the three concordances computed on the **fit set**, and B7 G / B7 M (already fit set).
- **Scores.** (a) The old rule re-done on the fit set: two-sided p >= 0.05 for the three r's,
  B7 inside the 95% range; gate unchanged; stability and primary recomputed exactly as the
  protocol defines them. (b) Empirical CRPS per statistic from the 200 draws
  (mean |X - y| - 0.5 mean |X - X'|), median over gated seeds; (c) 95% widths.
- **Readings (minimum: at least 9 gated seeds per model, as the stability rule already needs).**
  - **"full-M misfit was a held-out artefact"** if E46's fit-set full-M passes in at least 2/3 of
    its gated seeds.
  - **Protocol verdicts** (stable / primary counts) for E38 and E46 are replaced by the fit-set
    versions, as was done for B7.
  - **Model comparison (descriptive):** which model has the lower summed median CRPS over the five
    statistics, and on which statistics.
- **Protocol change from now on (registered here, before E47):** primary concordances are computed
  on the fit set, and model comparisons report CRPS next to pass/fail, so that a wider predictive
  spread cannot count as a better fit.

### B29 result (2026-11-05)

Fit-set data: G-M 0.37, full-G 0.90, full-M 0.43; B7 G -0.38, B7 M 0.04.

| | E38 (9 gated) | E46 (16 gated) |
|---|---|---|
| G-M / full-G / full-M pass | 9 / 9 / 9 | 16 / 16 / 16 |
| B7 G / B7 M inside | 5 / 9 | 16 / 13 |
| primary (all five) | 5/9 | 13/16 |
| **stable (protocol)** | **no** (B7 G 5/9) | **yes, and passing** |
| median 95% width, G-M / full-G / full-M | 0.32 / 0.56 / 0.95 | 0.61 / 0.67 / 1.00 |
| median CRPS G-M / full-G / full-M / B7G / B7M | 0.026 / 0.126 / 0.079 / 0.077 / 0.012 | 0.054 / 0.108 / 0.173 / 0.024 / 0.017 |
| summed median CRPS | **0.320** | 0.376 |

- **"full-M misfit was a held-out artefact": yes** (fit-set full-M passes in 16/16 E46 seeds).
- **Protocol verdict on the fit set: E46 is stable and passing; E38 is not** (it still fails B7 G in
  4 of 9). By E46's registered reading ("stability met: E46 becomes the reference, and the
  held-out power check comes next"), **E46 is now the reference.**
- **But the concordance tests carry almost no information on the fit set.** Their simulated 95%
  ranges span 0.3-1.0 of a scale that runs from about -0.1 to 1; every seed of both models passes
  all three. The only primary component that separates the models is B7 G, where E46 is
  genuinely better (CRPS 0.024 vs 0.077, centre on the data). On the three concordances the proper
  score favours E38 (sharper and closer), so E46's overall CRPS is worse (0.376 vs 0.320).
- **Net:** E46 earns the reference status by the registered rule through a real B7 G improvement;
  the "passes" on the concordances are not evidence for either model. The held-out power check is
  therefore essential before spending shard 1: it must show the shard-1 test can tell E38 and E46
  apart.

### Literature for pinning growth and death rates (2026-11-05)

- **Mossadegh-Keller 2013 (Nature, PMC3679883), read.** M-CSF induces PU.1 in single HSCs
  (SCF + TPO culture, 100 ng/ml M-CSF); time-lapse n = 39 vs 42 cells; PU.1+ cells all arose from
  PU.1- cells, not by division of PU.1+ cells; no proliferation or survival advantage in M-CSF
  (their Supp Fig 5a). Evidence for **instruction at the HSC stage**, before the first division.
  No division times or death rates given.
- **Kimura, Rieger et al. 2009 (Blood, PMC2780307), read.** GM-CSF via STAT5A/B controls GMP
  **proliferation and survival**: in Stat5-null GMPs, generation times are 16-37% longer and death
  rises to 5-24% per generation (controls: negligible death); more than 30 pedigrees per
  experiment, 3 experiments. Only relative times; no G-CSF or M-CSF arms.
- **Not obtained (publisher blocks automated access):** Rieger 2009 Science 325:217
  (10.1126/science.1171461) and Endele 2017 Blood 129:1691 (10.1182/blood-2016-05-714329), the two
  papers most likely to hold division/death rates by cytokine. To be downloaded by hand.
- **So far no absolute rates to pin.** Both mechanisms are documented in the literature at
  different stages: instruction (M-CSF on HSCs) and growth/survival control (GM-CSF on GMPs).

### Rieger 2009 and Endele 2017, read by hand (2026-11-05)

PDFs saved locally in papers/rates/ (not committed: publisher copyright).

- **Rieger et al. 2009 (Science 325:217), read.** Time-lapse of sorted mouse GMPs in only M-CSF or
  only G-CSF (375 and 318 pedigrees, 3 experiments), LysM::GFP as the commitment marker.
  - **88 ± 2% (M-CSF) and 87 ± 6% (G-CSF) of pedigrees had no cell death at all** before becoming
    all-M or all-G.
  - At most 23% (M) and 53% (G) of the sorted GMPs could have been unilineage already; so M-CSF and
    G-CSF **instruct at least 65% and 34%** of bipotent GMPs. About 90% of the same population goes
    to either lineage depending on the cytokine.
  - Both cytokines also speed up commitment (earlier LysM::GFP onset).
  - Cloning efficiency is high in both cytokines (figs S4-S5, in the online supplement, not read).
    No division times or death rates per cytokine in the main text.
- **Endele et al. 2017 (Blood 129:1691), read.** Mechanism (Src signalling carries the CSF-1
  instructive signal). Only rate-like facts: GMPs divide about twice in the first ~24 h, after which
  most are committed; CSF-1R mutants change survival but not average cycle time (data not shown).
  Nothing to pin.
- **What this means for E46 (reading, not a test).**
  - Selective **death** is ruled out as the main mechanism at the GMP stage: at most ~12% of
    pedigrees lose any cell.
  - E46's selection therefore has to be differential **proliferation**. E46's gamma (1.21-1.35)
    gives a G/M expansion ratio of exp(2 gamma) ~ 11-15x for a clone one SD along zb, i.e. about
    3.5-3.9 extra doublings in one well over day 2 to day 4. With a GMP cycle of ~12 h (at most ~4
    doublings in 48 h), that needs the disfavoured well to nearly stop dividing for a typical biased
    clone, which time-lapse with high cloning efficiency in both cytokines would have shown.
  - So E46's gamma is at or beyond the biological ceiling. A plausible bound is about one doubling
    difference per SD (gamma <= ln2/2 = 0.35).
  - Caveats: LARRY founders are LSK/LK, not sorted GMPs, and its expansion runs day 2 to day 4 in
    wells with several cytokines; zb is a per-cell draw. The bound is a prior, not a measurement.
- **Still missing:** Rieger 2009 online supplement (figs S4-S5: cloning efficiency, possibly
  cycle times).

## E47: E46 with selection capped at a plausible strength (pre-registered 2026-11-05)

- **Data diagnostic it must explain.** Same as E46 (B27 co-presence lift 0.31; B7 G), plus the
  Rieger 2009 constraint: GMP pedigrees in single M-CSF or G-CSF are 87-88% death-free with a
  cycle of ~12 h, so selection must act through proliferation, and E46's fitted gamma (1.21-1.35,
  11-15x G/M expansion per SD of bias) is at or beyond what ~4 doublings in 48 h allow.
- **Mechanism.** E46 unchanged (e46.simulate) except gamma = 0.35 sigmoid(x23), in (0, 0.35]: at
  most about one extra doubling per SD of bias between the G and M wells (2 gamma <= ln 2).
  24 dimensions. Script experiments/e47_capsel.py.
- **Every-condition check (30 simulations; scratchpad e47chk).**

  | parameters | lift | n G-M | B7 G | D3 | presence full/G/M/Epo/SCF | N share G / M / full |
  |---|---|---|---|---|---|---|
  | data (fit set) | 0.31 | 39 | -0.38 | 0.15 | 368/277/478/23/39 | 0.67 / 0.040 / 0.28 |
  | E38 best, gamma 0 | 0.48 | 57 | -0.25 | 0.02 | 371/269/422/22/44 | 0.67 / 0.042 / 0.29 |
  | E38 best, gamma 0.175 | 0.45 | 54 | -0.27 | 0.03 | 371/272/427/21/41 | 0.72 / 0.035 / 0.29 |
  | E38 best, gamma 0.35 | 0.43 | 52 | -0.31 | 0.04 | 371/274/426/21/40 | 0.77 / 0.027 / 0.29 |
  | E46 s11, gamma 1.31 | 0.37 | 44 | -0.42 | 0.02 | 376/276/435/25/45 | 0.68 / 0.045 / 0.31 |
  | E46 s11, gamma 0.35 | 0.71 | 51 | -0.12 | 0.00 | 376/182/317/27/46 | 0.31 / 0.18 / 0.31 |

  At the cap, lift, B7 G and D3 move toward the data by about a third of E46's movement; full,
  Epo and SCF do not move. E46's fitted parameters break at the cap (they compensate for strong
  selection), so E47 must be fitted from cold, not started from E46.
- **Fit.** As E46: 18 cold seeds, 500 generations max, E38 targets/loss/SE, guard on (more than 9
  gate fails), PC kept awake.
- **Scoring (B29 protocol).** Gate as before; primary and stability on the fit set
  (b29_rescore.py extended to E47, 200 simulations per gated seed); CRPS reported next to
  pass/fail.
- **Readings (minimum: at least 9 gated seeds for every reading except the B27 one).**
  - **Cap binds:** median gamma over gated seeds >= 0.9 x 0.35 = 0.315 (the data push selection
    to or past the plausible ceiling).
  - **B7 G gain survives the cap:** B7 G inside its range in at least 2/3 of gated seeds **and**
    median B7 G CRPS <= 0.05 (halfway between E46 0.024 and E38 0.077).
  - **Reference decision:** if E47 is stable and passing on the fit set **and** its summed median
    CRPS is <= E46's (0.376), E47 replaces E46 as the reference (same fit, plausible biology).
    If E47 is stable and passing but its CRPS is worse, E46 stays the reference with the flag
    "needs implausibly strong selection". If E47 is not stable and passing, the plausible version
    of selective expansion does not explain the data, and E46's gain is attributed to an
    implausible parameter.
  - **Explains B27 (best gated fit, 200 simulations):** G-M co-present count and lift, and
    M-present count, inside the simulated 95%.
  - **Side readings:** D3; median instruction shifts tau_G - tau_full and tau_full - tau_M
    (descriptive only).

### E47 result (run 42039247503, analysed 2026-11-05)

- **Run.** Completed, 18/18 seeds; the guard did not trigger (9 fails is its limit, not over it).
  The local launcher died before downloading, so results were fetched by hand from the run's
  artifact. Scoring: analyze_e47.py (E46 analysis, parametrised) and `b29_rescore.py e47`
  (fit-set primary + CRPS, 200 simulations per gated seed; results_e18/rescore_e47.json).
- **Gate 9/18** (E46 16/18, E38 9/18): exactly the minimum for every reading.
- **Cap binds: not met, by a hair.** Median gamma 0.312 vs the 0.315 threshold; 8 of 9 gated
  seeds sit at 0.29-0.33 (one at 0.22). The data use nearly all the selection the cap allows.
- **B7 G gain survives the cap: no.** B7 G inside in 9/9 gated seeds, but median CRPS 0.059 > 0.05.
  The simulated centre is -0.29 (E38 -0.27, E46 -0.41, data -0.38; widths 0.25 / 0.25 / 0.27), so
  at a plausible strength selection recovers only ~15% of E46's shift. The 9/9 passes sit near the
  edge of the range.
- **Reference decision (as registered): E47 becomes the reference.** Stable and passing on the
  fit set (all five components 9/9, primary 9/9) and summed median CRPS 0.340 <= E46's 0.376.

  | fit set | E38 | E46 | E47 |
  |---|---|---|---|
  | gated | 9 | 16 | 9 |
  | primary pass | 5 | 13 | 9 |
  | stable and passing | no | yes | yes |
  | CRPS G-M / full-G / full-M | 0.026 / 0.126 / 0.079 | 0.054 / 0.108 / 0.173 | 0.048 / 0.132 / 0.083 |
  | CRPS B7 G / B7 M | 0.077 / 0.012 | **0.024** / 0.017 | 0.059 / 0.018 |
  | summed CRPS | **0.320** | 0.376 | 0.340 |

  E47's win over E46 comes from the concordances, which B29 showed carry little information; it
  still scores worse than E38 overall (0.340 vs 0.320). E38 is excluded only by its 5/9 B7 G.
- **Explains B27: no** (best gated fit s9): G-M count 39 inside [36, 64], lift 0.31 inside
  [0.29, 0.48], M-present 478 above [398, 462].
- **Side readings.** D3 still out (0.15 vs [0, 0.09]); B26 16+ M-pure clones now out (5 vs [0, 4];
  E46 was inside); n shared and D2 inside. Instruction shifts: tau_G - tau_full 0.47,
  tau_full - tau_M 0.62 (E38 0.49 / 0.47): "not absorbed", and the M shift is no longer on a ridge.
- **Conclusion.** Lineage-matched selection at a biologically plausible strength fits about as well
  as E38 and keeps E38's instruction estimates; E46's clear B7 G improvement needs selection about
  four times stronger than ~4 doublings in 48 h allow. Either B7 G is driven by something other
  than selective expansion, or the Rieger GMP constraint does not carry over to LARRY's LSK/LK
  founders. Formally E47 is the reference; in substance the three models are close, and the M
  presence, D3 and B26 misfits are shared and unexplained.

### Rieger 2009 online supplement, read (2026-11-05)

PDF and movies saved locally (papers/rates/rieger2009_som.pdf; not committed). Figures read from
rendered pages; values read off plots, so about +-10%.

- **Fig S4, immediate death** (GMP or both daughters dead within 48 h): no cytokine ~68%,
  **M-CSF ~19%, G-CSF ~19%**. No difference in early death between the two cytokines.
- **Fig S5, population growth** (enzymatic assay, 3 experiments): relative counts ~1.6 (day 0),
  ~7 in both (day 1), **M-CSF ~17.5 and G-CSF ~10** (day 2). Day 0-1: ~2.1 doublings in both
  (cycle ~11 h). Day 1-2: ~1.3 doublings in M-CSF, ~0.6 in G-CSF. Over 48 h: M-CSF ~11x
  (~3.5 doublings), G-CSF ~6x (~2.6 doublings).
- **Fig S12, commitment timing:** ~75% of pedigrees switch on LysM::GFP at the founder cell or its
  daughters (P or F1); 60% within 24 h in M-CSF or G-CSF. Commitment is early, so most of the
  48 h is spent by committed cells.
- **Fig S3:** 71% of sorted GMPs form colonies; of these 21% GM, 23% M, 53% G.
- **What this means for the selection strength (reading, not a test).**
  - The fastest observed growth is ~2 doublings per day (day 0-1), so ~4-4.5 doublings in 48 h is
    a practical ceiling for any clone; the population means are ~3.5 (M-CSF) and ~2.6 (G-CSF).
  - In E46/E47, a clone at +k SD of bias grows exp(gamma k) times the average clone in its
    favoured well. Keeping a +1 SD clone (top ~16%) under the ceiling allows about one extra
    doubling, gamma <= ~0.6; keeping a +2 SD clone (top ~2%) under it gives gamma <= ~0.3.
  - So the plausible range is **gamma ~0.3-0.6**. E47's 0.35 cap sits at the strict end;
    **E46's 1.3 is still about twice the generous bound** (a +1 SD clone would need ~5.4
    doublings in 48 h, a ~9 h cycle sustained through differentiation, while growth in fact slows
    on day 2).
  - Early death does not differ between M-CSF and G-CSF, consistent with selection, if any, acting
    on division rather than death.
  - Caveats as before: sorted GMPs, single cytokines, population-level counts (not per lineage);
    LARRY founders are LSK/LK in mixed media.

## Amendment to the E47 reference decision (2026-11-05, prompted by Bence: "why is E46 not the best model?")

Logged openly; the E47 entry above is left as registered.

- **The E47 rule was flawed.** It compared summed CRPS over five statistics, three of which (the
  concordances) B29 had already shown carry almost no information: their simulated 95% widths are
  0.32-0.61 (G-M), 0.56-0.67 (full-G) and 0.95-1.00 (full-M) on a scale of about -0.1 to 1, and
  the data halves disagree by up to 0.35. E46 lost the sum (0.376 vs 0.340) entirely through
  full-M (0.173 vs 0.083); on every other statistic it was level or better.
- **On the data, E46 is the better model:** fit loss of gated seeds median 62.5 vs 90.7 (best 18.8
  vs 40.4), gate 16/18 vs 9/18, B7 G centred on the data (-0.41 vs -0.29; data -0.38), B26
  inside vs outside. Both miss M presence and D3.
- **Corrected status.** **E46 = best-fitting model.** **E47 = biologically constrained
  alternative** (selection within the Rieger bound; the supplement puts that bound at gamma
  ~0.3-0.6, E46 needs 1.3).
- **Protocol change from now on.** Model comparisons by CRPS use only B7 G + B7 M (widths
  0.13-0.27, and B7 G is the statistic that separates models); concordances are still reported
  with pass/fail but left out of CRPS sums. Under this rule: E46 0.041, E47 0.077, E38 0.089.

## E48: selection capped at the generous plausible bound (pre-registered 2026-11-05)

- **Data diagnostic it must explain.** As E46/E47: B27 co-presence lift 0.31 and B7 G (-0.38),
  under the Rieger-supplement bound gamma ~0.3-0.6. E47 tested the strict end (0.35) and recovered
  ~15% of E46's B7 G shift; E48 tests the generous end.
- **Mechanism.** E46 unchanged (e46.simulate), gamma = 0.6 sigmoid(x23) in (0, 0.6]. 24 dimensions.
  Script experiments/e48_capsel6.py; sweep sweeps/e48_capsel6.txt.
- **Every-condition check (E38 best parameters, 30 simulations; scratchpad e47chk/chk48.py).**

  | gamma | lift | n G-M | B7 G | D3 | presence full/G/M/Epo/SCF | N share G / M / full |
  |---|---|---|---|---|---|---|
  | data | 0.31 | 39 | -0.38 | 0.15 | 368/277/478/23/39 | 0.67 / 0.040 / 0.28 |
  | 0 | 0.48 | 57 | -0.25 | 0.02 | 371/269/422/22/44 | 0.67 / 0.042 / 0.29 |
  | 0.35 | 0.43 | 52 | -0.31 | 0.04 | 371/274/426/21/40 | 0.77 / 0.027 / 0.29 |
  | 0.45 | 0.41 | 50 | -0.29 | 0.05 | 371/277/424/21/41 | 0.79 / 0.022 / 0.29 |
  | 0.6 | 0.39 | 47 | -0.29 | 0.04 | 371/278/417/21/43 | 0.82 / 0.019 / 0.29 |

  Lift keeps moving toward the data; at fixed parameters B7 G saturates near -0.29 and the G
  N-share overshoots, which tau/s must absorb in the fit. Full, Epo and SCF do not move.
- **Fit.** As E47: 18 cold seeds, 500 generations max, E38 targets/loss/SE, guard 9, PC awake.
- **Scoring (amended protocol).** Gate as before; primary and stability on the fit set
  (`b29_rescore.py e46 e48`, 200 simulations per gated seed); CRPS comparison on B7 G + B7 M only.
- **Readings (minimum 9 gated seeds, except B27).**
  - **Cap binds:** median gamma over gated seeds >= 0.54.
  - **Recovers most of E46's gain:** median simulated B7 G centre <= -0.34 (at least half of
    E46's shift from E38's -0.27 toward -0.41) **and** median B7 G CRPS <= 0.05.
  - **Status:**
    - If E48 is stable and passing and its B7 CRPS sum (G + M) <= E46's 0.041: plausible
      selection explains the data as well as E46; **E48 becomes the best model**.
    - If E48 is stable and passing with B7 CRPS sum between 0.041 and E47's 0.077: E46 stays the
      best fit; E48 replaces E47 as the plausible alternative.
    - Otherwise: plausible selective expansion does not explain B7 G; E46's gain rests on an
      implausible strength, and a different mechanism is needed.
  - **Explains B27 (best gated fit, 200 simulations):** G-M count, lift, M-present count inside
    the simulated 95%.
  - **Side readings:** D3, B26 16+ M-pure, instruction shifts (descriptive).

### E48 result (run 42075538007, analysed 2026-11-05)

- **Run.** Completed, 18/18 seeds, results downloaded by the launcher (its exit 1 is the known
  wrapper quirk). Guard not triggered. Scoring: analyze_e48.py, `b29_rescore.py e48`
  (results_e18/rescore_e48.json).
- **Gate 9/18** (E46 16, E47 9). Fit loss of gated seeds median 89.9, best 17.2 (E46 62.5 / 18.8).
- **Cap binds: yes.** Median gamma 0.558 >= 0.54 (8 of 9 at 0.53-0.57).
- **Recovers most of E46's gain: yes.** B7 G centre -0.343 <= -0.34 (data -0.38; E46 -0.41,
  E47 -0.29, E38 -0.27) and median B7 G CRPS 0.026 <= 0.05 (E46 0.024).
- **Status (as registered): E46 stays the best fit; E48 replaces E47 as the plausible
  alternative.** Stable and passing on the fit set (all five components 9/9) with B7 CRPS sum
  0.044, between E46's 0.041 and E47's 0.077. The margin to E46 (0.003) is far below anything
  these 9-16 seeds can resolve: on B7, E46 and E48 are in practice tied.
- **Explains B27: no.** Best gated fit (s0): G-M co-present 39 below [41, 66], lift 0.31 below
  [0.34, 0.52], M-present 478 just above [412, 476]. E46's best fit got count and lift inside.
- **Side readings.** D3 out (0.15 vs [0, 0.07]); B26 16+ M-pure out (5 vs [0, 4]); D2 and
  n shared now out as well. Instruction shifts 0.50 / 0.56 (E38 0.49 / 0.47): "not absorbed".
- **Conclusion.** Selection at the generous end of the Rieger range (one extra doubling per SD
  over 48 h) reproduces E46's B7 G fit, and the data again push gamma to the cap. What it does
  not reproduce is E46's co-presence fit: the lift that motivated E46 stays too high in the best
  plausible fit. So the B7 G evidence for lineage-matched selection is compatible with biology;
  the co-presence evidence needs more than biology allows, or another mechanism. Instruction
  estimates are unchanged across E38, E47 and E48 (~0.5 each way).

## B30: power check for the held-out Epo/SCF shard-1 look (pre-registered 2026-11-05)

- **Why.** The only unspent held-out test is `epo_scf_pooled` shard 1 of 2 (test_oracle.py):
  N-fraction concordance r pooled over every condition pair that includes Epo or SCF, clones in
  shard 1 only (never fitted). Shard 0 (E14) had 24 clones and a 95% CI half-width of ~0.37. The
  standing power rule (E18, E26) says: look only if the candidate models' predicted shard-1 r
  differ by more than the expected CI half-width.
- **What.** Models E38, E46, E47, E48; every gated seed; 50 simulations per seed (rng 3000+s) with
  the model's own simulator on the full clone universe, restricted to Epo/SCF shard 1
  (`split(sim, 1)`), statistic `pooled_pairs_r` (boot 1, only r used). No observed shard-1
  outcome is touched. Script experiments/b30_power.py; output results_e18/b30_power.json.
- **Reported per model:** mean, 2.5-97.5% of the pooled draws, median clone count, share of draws
  hitting the <10-clone sentinel (r = 0), and the simulated half-width (1.96 x SD of draws).
- **Decision rule (registered now).** For each pair among E46, E48 and E38 (E47 reported only):
  the look can discriminate the pair if |mean difference| > max(0.37, the larger of the two
  simulated half-widths). Minimum: at least 9 gated seeds per model (all have 9-16).
  - If no pair passes: **the look is skipped as uninformative**, shard 1 stays sealed, and the
    next step is a sharper held-out statistic (to be pre-registered separately).
  - If a pair passes: the look is proposed (not taken) with a separate pre-registration naming the
    reading for each model.

### B30 result (2026-11-05)

| model | gated | predicted shard-1 r, mean | 95% of draws | half-width | median clones |
|---|---|---|---|---|---|
| E38 | 9 | 0.37 | [0.00, 0.92] | 0.71 | 37 |
| E46 | 16 | 0.39 | [0.00, 0.89] | 0.66 | 48 |
| E47 | 9 | 0.28 | [0.00, 0.90] | 0.67 | 36 |
| E48 | 9 | 0.32 | [0.00, 0.91] | 0.73 | 43 |

- **No pair discriminates.** Mean differences 0.02 (E38-E46), 0.05 (E38-E48), 0.07 (E46-E48)
  against a required 0.71-0.73. The models' predictions differ by about a tenth of the spread of
  any single model's prediction. No draw hit the <10-clone sentinel (the 0.00 lower bounds are
  undefined correlations, e.g. all-N clones, set to 0 by test_stats).
- **Decision (as registered): the look is skipped; shard 1 stays sealed.** For reference, shard 0
  (E14) gave 0.44, inside every model's range.
- **Reading.** The pooled Epo/SCF concordance cannot test what separates these models: they all
  share the same instruction structure (shifts ~0.5) and differ in selection, which acts on clone
  size and G/M presence, not on N-fraction agreement between Epo/SCF and other wells. A useful
  held-out statistic has to target presence/size (B7-type) on shard-1 clones; to be designed and
  pre-registered separately, with its own power check, before shard 1 is spent.

## B31: where does the G-M co-presence deficit live? (pre-registered 2026-11-05)

- **Data diagnostic.** G-M co-presence lift (B27: clones with >= 2 cells in both G and M, over the
  count expected if presence were independent, universe = clones present in any of full/G/M,
  fit set) is 0.31. E46's best fit reproduces it; the plausible E48 best fit leaves it at
  0.34-0.52. Before any E49, locate the deficit, without assuming a mechanism.
- **Statistics (fit set; data and simulations computed identically on pooled days):**
  - (a) **Threshold curve:** G-M lift with presence defined as >= 1, 2, 3 and 5 cells.
  - (b) **Cross-presence by size:** P(present in M | G-clone size bin 2, 3-5, 6+) and
    P(present in G | M-clone size bin 2, 3-5, 6+), presence >= 2 cells.
  - (c) **By measured day-2 state:** G-M lift within the clones with a measured day-2 score m
    (Cebpa - Irf8), separately below and above the median m (expected count computed within each
    half).
  - (d) **Data only, descriptive:** lift for day-matched (G4-M4, G6-M6) vs cross-day (G4-M6,
    G6-M4) presence; the simulators produce one pooled day, so no model comparison.
- **Models.** Best gated fits (lowest loss) of E46 and E48, 200 simulations each (rng 1000+s).
  Script experiments/b31_copresence.py; output results_e18/b31_copresence.json.
- **Minimum counts.** A statistic is read only if its data denominator has >= 30 clones
  ((a) the smaller of G and M presence counts; (b) the bin size; (c) expected co-present count
  >= 10). Otherwise reported as "too few".
- **Readings (each: data inside or outside the model's simulated 95%).**
  - **Small-clone artefact:** E48 misses the lift at >= 2 but not at >= 3 and >= 5 -> the
    deficit lives in 2-cell samples (as B24 found for B7 G), pointing at detection/sampling of
    the smallest clones, not at fate.
  - **Size-linked exclusion:** in (b), the data's P(M | G size) falls with G size and lies below
    E48's range in the 6+ bin (or the mirror for P(G | M size)) -> clones that expand strongly in
    one cytokine are missing from the other, the signature of strong lineage-matched expansion.
  - **Size-independent exclusion:** (b) below E48 in all bins by a similar margin -> a
    presence-level factor (clone-level seeding or survival) unrelated to expansion.
  - **Carried by the measured state:** in (c), the data lift within each half is closer to 1
    than the overall lift, and E48 fits within halves -> the deficit is fate-state structure the
    model under-couples (w); if the data lift stays as low within halves, it is not carried by
    the measured day-2 state.
  - **Day structure (descriptive):** day-matched lift >= 0.2 higher than cross-day suggests
    sampling time contributes.

### B31 result (2026-11-05)

Best gated fits: E46 s11, E48 s0; 200 simulations each. All denominators >= 44 clones except
(c), where expected co-present counts are 23 and 21 (>= 10, read).

| statistic | data | E46 (95%) | E48 (95%) |
|---|---|---|---|
| (a) lift, >= 1 cell | 0.60 | 0.83 [0.77, 0.89] **out** | 0.77 [0.69, 0.84] **out** |
| (a) lift, >= 2 | 0.31 | 0.35 [0.27, 0.45] in | 0.43 [0.34, 0.52] **out** |
| (a) lift, >= 3 | 0.18 | 0.12 [0.04, 0.20] in | 0.21 [0.12, 0.30] in |
| (a) lift, >= 5 | 0.03 | 0.01 [0.00, 0.06] in | 0.04 [0.00, 0.12] in |
| (b) P(M \| G size 2), n 112 | 0.16 | 0.21 [0.14, 0.28] in | 0.26 [0.18, 0.33] **out** |
| (b) P(M \| G 3-5), n 121 | 0.15 | 0.14 in | 0.18 in |
| (b) P(M \| G 6+), n 44 | 0.07 | 0.02 [0.00, 0.08] in | 0.10 in |
| (b) P(G \| M size 2), n 172 | **0.04** | 0.13 [0.08, 0.19] **out** | 0.15 [0.09, 0.21] **out** |
| (b) P(G \| M 3-5), n 160 | 0.10 | 0.11 in | 0.12 in |
| (b) P(G \| M 6+), n 146 | 0.11 | 0.04 [0.01, 0.07] **out** | 0.09 [0.04, 0.15] in |
| (c) lift, m below median | 0.48 | 0.52 in | 0.54 in |
| (c) lift, m above median | 0.39 | 0.43 in | 0.49 in |

(d) Data only: G4-M4 lift 0.86 (13 clones), G4-M6 0.35 (25), G6-M4 0.28 (3), G6-M6 0.14 (7).

- **Small-clone artefact: yes (as registered).** E48 misses the lift at >= 1 and >= 2 cells and
  fits it at >= 3 and >= 5. The whole deficit sits in the smallest clones.
- **Located precisely in (b):** clones with exactly 2 cells under M-CSF are almost never present
  under G-CSF (7 of 172, 4%), against 13-15% in both models; the same, weaker, for 2-cell G clones
  (16% vs E48 26%). Larger clones fit E48 in every bin.
- **Size-linked exclusion: no.** The data do not show big clones missing from the other well more
  than E48 predicts. **E46 gets the overall lift right for the wrong reason:** it under-predicts
  co-presence of large M clones (P(G | M 6+) 0.04 vs data 0.11, out) to compensate for
  over-predicting it in small ones.
- **Carried by the measured state: no reading.** Within halves of the day-2 score both models fit,
  and the data lift is similar in both halves (0.48, 0.39).
- **Day structure (descriptive, small counts):** day-4 G vs day-4 M presence is near independence
  (0.86, 13 clones); everything involving day 6 is low. Not interpretable at these counts.
- **Conclusion.** The co-presence deficit that motivated E46 is a small-clone phenomenon: 2-cell
  clones in one cytokine are private to that well far more often than any fate or selection model
  predicts. That is the signature of something acting on tiny clones only: candidates are
  spurious or error barcodes (low-count, well-private), late-arriving clones, or detection. The
  strong selection in E46 is not needed for it, and in fact distorts large clones. Next: a data
  check of the 2-cell well-private clones (do they appear anywhere else, at day 2 or in the full
  arm; their barcode/UMI quality if available), before any model.

## B32: are the 2-cell well-private clones real? (pre-registered 2026-11-05)

- **Data diagnostic.** B31: clones with exactly 2 N+M cells under M-CSF are present under G-CSF in
  4% of cases (7/172) against 13-15% in E46/E48; 2-cell G clones similarly (16% vs E48 26%).
  Larger clones fit. Question: are these small clones private to their well altogether (spurious
  barcodes, late-arriving or non-clonal events), or only missing from the other cytokine?
- **Not possible:** a barcode-sequence check (one-mismatch neighbours of large clones). The
  public cytokine-experiment files hold a clone x cell matrix, not barcode sequences. Recorded,
  not pursued.
- **Statistics (fit set; presence elsewhere = at least 1 N+M cell, days pooled).**
  - (i) For M-CSF clones of size 2 and of size 3-5 (control), and the mirror for G-CSF clones:
    the share present in each other simulated condition (G or M, full, Epo, SCF) and in any of
    them. Compared with E48's best gated fit (s0, 200 simulations, the same statistic).
  - (ii) Data only: share of those clones with any day-2 cell (n2 > 0), size 2 vs size 3-5,
    with a bootstrap 95% CI of the difference.
  - (iii) Data only: for same-day 2-cell clones, the share whose two cells come from the same
    library, against (a) the share expected if cells fell into libraries in proportion to library
    size and (b) the same-library share of random cell pairs within 3-5-cell clones of the same
    condition and day.
- **Minimum counts.** (i) and (ii) are read only for groups of at least 30 clones; (iii) only
  with at least 30 same-day 2-cell clones per cytokine.
- **Readings.**
  - **Whole-clone private:** for 2-cell M (or G) clones, the data share present in any other
    condition lies below E48's 95% **and** the shortfall shows in at least two of full / Epo / SCF
    (not only the other cytokine). Supported further if (ii) the day-2 share is lower for size 2
    than for 3-5 (CI excludes 0). Implication: a share of small clones is not part of the shared
    clone pool the models assume; presence statistics move to at least 3 cells (B31 showed E48
    fits there), and E46/E48 are re-scored on that basis.
  - **Cytokine-specific absence:** shortfall only in the other cytokine (G for M clones, M for G
    clones), with full / Epo / SCF inside E48's range: a real G-vs-M exclusion of small clones
    that the models miss (biology, e.g. fate-restricted small clones).
  - **Library-local:** (iii) same-library share above both baselines by at least 0.15: small
    clones are disproportionately confined to one library (one well or one capture), supporting
    an artefact or well-local origin. Descriptive.

### B32 result (2026-11-05)

E48 best gated fit s0, 200 simulations. All groups >= 112 clones; 139-205 same-day 2-cell clones.

| clones | n (data / E48) | in G or M | in full | in Epo | in SCF | in any other |
|---|---|---|---|---|---|---|
| M-CSF, 2 cells | 172 / 117 | 0.10 vs 0.28 **out** | 0.11 vs 0.17 **out** | 0.03 vs 0.08 **out** | 0.03 vs 0.10 **out** | **0.21 vs 0.51 out** |
| M-CSF, 3-5 | 160 / 206 | 0.20 vs 0.26 out | 0.16 vs 0.17 in | 0.03 vs 0.07 out | 0.04 vs 0.10 out | 0.38 vs 0.49 out |
| G-CSF, 2 cells | 112 / 110 | 0.23 vs 0.35 **out** | 0.12 vs 0.17 in | 0.06 vs 0.10 in | 0.08 vs 0.13 in | 0.38 vs 0.59 out |
| G-CSF, 3-5 | 121 / 115 | 0.22 vs 0.29 in | 0.14 vs 0.17 in | 0.02 vs 0.12 out | 0.05 vs 0.16 out | 0.35 vs 0.57 out |

- **2-cell M-CSF clones: whole-clone private (as registered).** Only 21% are seen in any other
  well (E48 51%), with the shortfall in full, Epo and SCF as well as G. They are also less often
  seen at day 2: 22% vs 36% for 3-5-cell M clones (difference CI -0.24 to -0.05). And there are
  more of them than the model makes (172 vs 117) while 3-5-cell M clones are fewer (160 vs 206).
  **This is the M-presence misfit (478 vs ~420-460 in every model): the extra M-present clones are
  small and private to the M-CSF well.**
- **2-cell G-CSF clones: cytokine-specific (as registered).** Short only in M-CSF (0.23 vs 0.35);
  full, Epo and SCF inside. No day-2 difference (0.29 vs 0.30).
- **Library-local: no.** Same-library share of 2-cell clones is 0.49 (M-CSF; expected 0.46,
  3-5-cell pairs 0.49) and 0.35 (G-CSF; 0.40, 0.37). The two cells of a small private clone are
  spread over libraries like cells of any clone, so a capture or library artefact is unlikely;
  they look like real cells of real (small) clones.
- **Side finding:** in every group, data clones are seen in Epo and SCF wells about half as often
  as E48 predicts (e.g. G 3-5: Epo 0.02 vs 0.12). The models over-share clones with Epo/SCF.
- **Conclusion.** There is a population of small clones that live only in the M-CSF well: real
  cells, not library artefacts, less often visible at day 2, absent from every other well. They
  are not part of the sister-split pool the models assume. Candidates: clones seeded into the M
  well only (cells with too few sisters at the split), or rare cells that survive and divide only
  in M-CSF (e.g. a monocyte/macrophage precursor fraction that the other cytokines do not
  support). Their excess inflates M presence and deflates G-M co-presence, which is what E46's
  strong selection was compensating for. As registered, presence statistics should move to
  >= 3 cells; B31(a) already shows both E46 and E48 fit the G-M lift at >= 3 and >= 5 cells.
  A full re-fit with >= 3-cell presence targets (or an explicit M-private clone class) is the
  candidate for E49, to be pre-registered separately.

## B33: power check for a selection-sensitive held-out statistic on shard-1 clones (pre-registered 2026-11-05)

- **Why not B7 G.** Bence asked for a power check of B7 G on shard-1 clones. That value is not
  clean: B7 G was computed on all clones (-0.373) and on the fit set (-0.382) in the 2026-11-04
  audit, so the shard-1 value follows almost exactly from the two. The same holds for B7 M and
  for the concordances (B29 printed their shard-1 values). These are excluded as held-out
  statistics.
- **Candidates instead.** The B31 statistics, computed so far only on the fit set (2026-11-05),
  never on all clones or on shard 1, are the ones that separate the selection models:
  (1) G-M lift at >= 2 cells; (2) G-M lift at >= 3 cells; (3) P(present in G | M-clone size 6+);
  (4) P(present in M | G-clone size 6+); (5) P(present in G | M-clone size 2). Computed on
  shard-1 clones only (`split(cnt, 1)`), same definitions as B31 (universe = clones present in any
  of full/G/M on shard 1).
- **What.** E38, E46, E47, E48: every gated seed, 20 simulations each (rng 4000+s), statistic on
  the simulated shard-1 clones. No observed shard-1 value is computed. Script
  experiments/b33_power.py; output results_e18/b33_power.json.
- **Decision rule (as B30).** For each statistic and each pair among E38, E46 and E48: the look
  can discriminate the pair if |mean difference| > the larger of the two simulated half-widths
  (1.96 x SD of draws). Minimum: the statistic's simulated denominator has a median of at least
  30 clones (for (3)-(5), the bin size; for (1)-(2), the smaller presence count).
  - If no statistic discriminates any pair: no shard-1 look is proposed; shard 1 stays unused.
  - If some do: a look is proposed (not taken) in a separate pre-registration listing the
    discriminating statistics and the reading for each model. The look itself needs Bence's go.
- **Caveat recorded now.** Shard-1 clones' G, M and full data were used in earlier all-clone
  statistics (cross r, B7, B12-type co-presence), so even unseen statistics on them are not
  independent of everything seen. A pass makes the look informative, not pristine.

### B33 result (2026-11-05)

Predicted shard-1 means (gated seeds x 20 simulations; all denominators >= 30):

| statistic | E38 | E46 | E47 | E48 |
|---|---|---|---|---|
| G-M lift, >= 2 cells | 0.49 | 0.34 | 0.45 | 0.47 |
| G-M lift, >= 3 cells | 0.28 | 0.13 | 0.23 | 0.25 |
| P(G \| M size 6+) | 0.15 | 0.05 | 0.12 | 0.12 |
| P(M \| G size 6+) | 0.15 | 0.05 | 0.13 | 0.13 |
| P(G \| M size 2) | 0.15 | 0.13 | 0.16 | 0.15 |

- **E38 vs E46: discriminable** on three statistics (lift >= 2: difference 0.14 vs needed 0.10;
  lift >= 3: 0.16 vs 0.11; P(G | M 6+): 0.10 vs 0.07).
- **E46 vs E48: just short** on every statistic (best: lift >= 2, 0.125 vs 0.127; lift >= 3,
  0.122 vs 0.125).
- **E38 vs E48: not discriminable** on any (differences <= 0.03): plausible selection changes
  presence statistics very little.
- **Decision (as registered): a look is informative, for strong selection (E46) against weak or
  none (E38/E48).** It is proposed, not taken; it needs its own pre-registration (statistics:
  lift >= 2, lift >= 3, P(G | M 6+); a reading per model) and Bence's go.
- **Caveat.** On the fit set these statistics already favour E38/E48 over E46 for P(G | M 6+)
  (data 0.11; E46 0.04 out, E48 0.09 in), so the look would be a replication on independent
  clones, with the B33 contamination caveat. It cannot separate plausible selection from none;
  that question stays open.

### Literature: the two retro "read next" papers (2026-11-05)

- **Lin, Zhang, ... Weber, Naik 2024 (bioRxiv 10.1101/2024.03.28.587126), discussion and model
  sections read.** In vivo barcoding of LT-HSC to MDP populations after transplantation. Claims:
  commitment to reduced-potency "tracks" is often early and hard-wired; "common" progenitors are
  often highly restricted; **restricted downstream progenitors (CMPs, CLPs, MDPs) have low potency
  and smaller clones** than upstream cells. The agent-based model folds division and death into one
  effective proliferation rate per cell type (cell counts only), so no usable division or death
  rates. **Bearing on B32:** a fraction of LSK/LK founders that are already monocyte-restricted
  (MDP/cMoP-like) and make small clones is a documented biology, consistent with the small
  M-CSF-private clones. It supports an explicit restricted-clone class in E49.
- **Eisele ... Perie 2022 (eLife 11:e66922), abstract, results and legends read** (discussion and
  methods only via summary, unverified). Ex vivo EPO (16 h) before transplantation: the number of
  lineage-biased clones is unchanged, but biased clones contribute more output. The authors call it
  clonal remodelling and say they cannot separate a transient fate change from selection. No
  proliferation or survival rates. **Bearing:** cytokines can reshape which clones contribute
  without changing who is restricted, but the in vivo setting does not explain the Epo/SCF
  over-sharing in the LARRY wells. Low relevance.

## E49: limited sister split plus plausible selection (pre-registered 2026-11-05)

- **Design check that rejected the first candidate (not registered, not run).** A class of
  monocyte-restricted clones that grow only in the M-CSF well (experiments/e49_restricted.py;
  scratchpad e49chk*.py, from E48's best fit, 30 simulations each). At fixed parameters it barely
  moves the G-M lift (0.43-0.46). With the founder rate rescaled, or lowered so the class supplies
  the M-well count, P(G | M size 2) never fell below 0.10 (data 0.04), and lowering the lift cost
  G, Epo and SCF presence (G 138-192 vs 277). The class cannot produce "small clones are private,
  shared clones are big" on its own. Dropped before any fit.
- **Data diagnostic it must explain.** B31/B32: small clones are private to one well (2-cell M
  clones seen elsewhere 21%, E48 51%), shared clones are larger, G-M lift 0.31 at >= 2 cells.
  E45 (a limited number of sisters split between wells) reproduced the presence structure but
  lost B7 G; E48 (selection, gamma <= 0.6) fits B7 G but leaves small clones too shared.
- **Mechanism.** E45's founder split (K = 1 + Poisson(lam_K) sisters per clone, each sent to G,
  Epo, SCF with probability r or to M with r w_M, or lost) plus E46's lineage-matched selection,
  gamma = 0.6 sigmoid(x25). 26 dimensions. Script experiments/e49_sisters_sel.py; sweep
  sweeps/e49_sisters_sel.txt.
- **Every-condition check (gated E45 fits with gamma added; 30 simulations; scratchpad
  e49chk4.py).**

  | parameters | lift | n G-M | B7 G | D3 | P(G\|M2) | M2 n | M2 seen elsewhere | M2 in Epo | presence full/G/M/Epo/SCF | N share G / M |
  |---|---|---|---|---|---|---|---|---|---|---|
  | data | 0.31 | 39 | -0.38 | 0.15 | 0.04 | 172 | 0.21 | 0.03 | 368/277/478/23/39 | 0.67 / 0.040 |
  | E48 best (reference) | 0.44 | 54 | -0.31 | 0.03 | 0.15 | 115 | 0.52 | 0.08 | 359/267/445/26/41 | 0.68 / 0.051 |
  | E45 s14, gamma 0 | 0.40 | 55 | -0.23 | 0.03 | 0.12 | 117 | 0.36 | 0.03 | 347/287/479/24/48 | 0.69 / 0.047 |
  | E45 s14, gamma 0.3 | 0.36 | 50 | -0.26 | 0.04 | 0.13 | 101 | 0.38 | 0.03 | 347/291/480/23/47 | 0.77 / 0.034 |
  | E45 s14, gamma 0.6 | 0.31 | 42 | -0.27 | 0.06 | 0.13 | 83 | 0.36 | 0.03 | 347/289/474/23/47 | 0.84 / 0.022 |
  | E45 s3, gamma 0.6 | 0.36 | 49 | -0.33 | 0.09 | 0.14 | 83 | 0.44 | 0.01 | 321/296/440/5/35 | 0.80 / 0.020 |
  | E45 s16, gamma 0.6 | 0.43 | 61 | -0.29 | 0.08 | 0.20 | 74 | 0.53 | 0.07 | 388/360/417/20/43 | 0.83 / 0.017 |

  From E45's best fit (s14), selection brings lift, co-present count, M presence and Epo
  over-sharing to the data, and moves "M2 seen elsewhere" and D3 partly toward it. Full presence
  is slightly low and the G N-share overshoots (tau/s are fitted). **Not reached at fixed
  parameters:** P(G | M2) (0.13 vs 0.04) and the number of 2-cell M clones (83 vs 172). These are
  the hard tests.
- **Fit.** 18 cold seeds, 500 generations max, E38 targets/loss/SE, guard 9, launcher as a
  background task, PC awake.
- **Scoring.** Gate as before; fit-set primary, stability and B7 CRPS via `b29_rescore.py e49`;
  analyze_e49.py for B27, B24/B26, the B31/B32 statistics (best gated fit, 200 simulations),
  lam_K and gamma.
- **Readings (minimum 9 gated seeds, except best-fit readings).**
  - **Working model:** E49 replaces E48 if it is stable and passing on the fit set, its B7 CRPS
    sum (G + M) is <= 0.054 (E48 0.044 + 0.01), and its best gated fit explains B27 (G-M count,
    lift and M-present count inside the simulated 95%), which E48 failed.
  - **Explains B32:** best gated fit has "M2 seen elsewhere" and P(G | M2) inside its 95%.
  - **Lift at >= 2 cells inside** (B31(a)), reported.
  - **Sisters used:** median lam_K over gated seeds <= 10 (a genuinely limited split; large lam_K
    approaches E48's independent founders).
  - **Selection:** median gamma reported; "cap binds" if >= 0.54.
  - **Side readings:** D3, B26, Epo/SCF sharing (G 3-5 in Epo/SCF), instruction shifts.
  - If E49 fails the working-model rule, E48 stays the working model and E46 the best fit.

### E49 result (run 42090317872, analysed 2026-11-05)

- **Run.** The guard cancelled it at 16/18 seeds: 6 gated, 10 failed, so the 9-gated minimum could
  no longer be met (watcher exit 1 = cancel). 16 result files downloaded by the launcher.
- **Working model: no (as registered).** 6 gated seeds, below the minimum of 9; not stable.
  E48 stays the working model and E46 the best fit.
- **Best-fit readings (allowed below 9 seeds; best gated s8, loss 76.4 vs E48's 17.2):**
  - **Explains B27: no.** G-M co-present 39 below [45, 71], lift 0.31 below [0.35, 0.51];
    M-present 478 inside [433, 504] (E48 missed this one).
  - **Explains B32: no.** P(G | M2) 0.04 vs 0.13 [0.07, 0.19]; 2-cell M clones seen elsewhere
    0.21 vs 0.49 [0.42, 0.56]; 2-cell M count 172 vs 131 [107, 156]. Lift at >= 2 cells out.
  - B7 G inside (B7 CRPS sum 0.054 over the 6 gated seeds; B7 G 0.028); B26 inside; D3 and D2 out.
    Epo/SCF over-sharing unchanged (G 3-5 in Epo 0.11 vs data 0.02).
- **Sisters used: no.** Median lam_K 13.6 (> 10): when fitted to all targets, the split drifts to
  many sisters, i.e. back toward E48's independent founders, and the presence structure that E45's
  s14 parameters produced is lost. Median gamma 0.55 (cap binds). Instruction shifts 0.42 / 0.63
  (not absorbed).
- **Conclusion.** A limited sister split does not explain the small-clone privacy once the whole
  loss is fitted; the fit prefers independent seeding. Two designs (a restricted M-only class and
  a sister split) have now failed to make small clones private while keeping the other wells
  right. Next: a data diagnostic of the private clones' size distribution and of how private
  vs shared clones differ at day 2, before any further model.

## B34: what are the well-private clones? (pre-registered 2026-11-05)

- **Data diagnostic.** B32: small M-CSF clones are private to that well (2-cell: 21% seen in any
  other well, E48 51%); E49 and a restricted-class design failed to reproduce it. Before another
  model, describe the private clones. B22 (2026-11-04) already found no day-2 MDP signature
  (Flt3, Cd74, MHC-II) in M-only vs M-shared clones; not repeated.
- **Definitions (fit set, N+M cells, days pooled unless stated).** For each M-CSF clone:
  "private" = no N+M cell in full, G, Epo or SCF; "shared" = at least one. Mirror for G-CSF clones.
  Size-matched comparisons use clones of 2-5 cells in the source well.
- **Statistics.**
  - (1) **Privacy by size:** share private for M-clone size bins 1, 2, 3-5, 6-15, 16+, data vs
    E48's best gated fit (s0, 200 simulations, same statistic). Mirror for G.
  - (2) **Day structure (size 2-5):** share of the clone's M-well cells from day 6, private vs
    shared, bootstrap 95% CI of the difference; share of clones seen on both days.
  - (3) **Day 2 (size 2-5):** share with any day-2 cell (repeats B32 at this size), day-2 cell
    count n2 among those seen, and day-2 score m (Cebpa - Irf8), private vs shared (Mann-Whitney).
  - (4) **Composition (size 2-5):** neutrophil share in the M well, private vs shared.
- **Minimum counts.** A comparison is read only with >= 30 clones in each group ((3)'s day-2
  tests: >= 30 private clones with day-2 cells).
- **Readings.**
  - **Privacy is a small-clone effect only:** data private share is above E48's 95% in bins 1-2
    and inside it for 6-15 and 16+.
  - **Late arrival:** private clones' day-6 share exceeds shared clones' by >= 0.15 with the CI
    excluding 0. Late-expanding clones would appear only at the later sampling.
  - **Distinct day-2 state:** private and shared differ in m or n2 at p < 0.05. If neither,
    the private clones are not distinguishable at day 2 by the measured axis.
  - **Composition (descriptive):** a lower neutrophil share in private clones would fit
    monocyte restriction.

### B34 result (2026-11-05)

(1) Share of clones private to their well, by size (E48 best gated fit s0, 200 simulations):

| size | M-CSF: n, data vs E48 | G-CSF: n, data vs E48 |
|---|---|---|
| 1 | 155: 0.61 vs 0.45 [0.37, 0.52] **out** | 181: 0.44 vs 0.41 in |
| 2 | 172: 0.79 vs 0.49 [0.40, 0.57] **out** | 112: 0.62 vs 0.42 [0.32, 0.52] **out** |
| 3-5 | 160: 0.62 vs 0.51 [0.45, 0.58] **out** | 121: 0.65 vs 0.43 [0.34, 0.51] **out** |
| 6-15 | 121: 0.52 vs 0.54 in | 40: 0.75 vs 0.42 [0.29, 0.58] **out** |
| 16+ | 25: 0.52 vs 0.59 in | 4: too few |

(2)-(4), clones of 2-5 cells (M: 235 private / 97 shared; G: 148 / 85):

| | M private | M shared | G private | G shared |
|---|---|---|---|---|
| day-6 share of cells | 0.86 | 0.83 (diff 0.03 [-0.03, 0.10]) | 0.42 | 0.30 (diff 0.13 [0.02, 0.24]) |
| seen on both days | 0.20 | 0.29 | 0.19 | 0.25 |
| neutrophil share in the well | 0.06 | 0.09 | **0.70** | **0.41** |
| any day-2 cell | 0.23 | 0.43 | 0.23 | 0.40 |
| day-2 cells, if seen (n 54/42; 34/34) | 1.50 | 1.74 (p 0.26) | 1.68 | 1.74 (p 0.55) |
| day-2 score m, if seen | -0.49 | -0.14 (p 0.16) | +0.58 | +0.14 (p 0.23) |

- **Privacy is a small-clone effect only: yes for M-CSF** (above E48 at 1-5 cells, inside at
  6-15 and 16+). **Not for G-CSF:** G privacy is above E48 at 2-15 cells and grows with size
  (0.75 at 6-15). The two cytokines differ.
- **Late arrival: no** for M (difference 0.03, CI includes 0); for G the difference is 0.13
  (CI 0.02-0.24), below the registered 0.15. Not read as late arrival.
- **Distinct day-2 state: no.** Private and shared clones do not differ in day-2 cell count or in
  m at p < 0.05. The direction is consistent with fate (M-private more monocyte-primed, G-private
  more neutrophil-primed), but not significant at these counts.
- **Composition:** G-private clones are neutrophil-dominated (0.70 vs 0.41); M-private clones
  slightly more monocytic (0.06 vs 0.09 neutrophils).
- **Common to both:** private clones are about half as often seen at day 2 (0.23 vs 0.40-0.43),
  with the same count when seen.
- **Reading.** Two different things. (a) G-CSF: private clones are neutrophil-committed clones of
  any size that appear only under G-CSF; this is fate-linked presence (the same pattern as B7 G),
  stronger than E48's selection makes. (b) M-CSF: an excess of small private clones with no
  distinguishing day-2 state, less often captured at day 2. Low day-2 representation is shared by
  both and points to clones that were few cells at the split. No model change is registered
  from this; options for the next design: fate-linked presence under G-CSF (stronger at large
  size, e.g. a neutrophil-committed class), and clone-level variation in the number of cells at
  the split.

## B35: does a clone's day-2 abundance predict how widely it is seeded? (pre-registered 2026-11-05)

- **Data diagnostic.** B32/B34: well-private clones are captured at day 2 about half as often as
  shared ones (0.23 vs 0.40-0.43), in both cytokines. If clones differ in how many cells they had
  at the split, clones with more day-2 cells should be present in more wells. E48 has no such
  link: founders are Poisson(nu) per well, the same for every main-pool clone.
- **Statistics (fit set; clones present with >= 1 N+M cell in any of full/G/M/Epo/SCF).** Group
  clones by day-2 cell count n2 = 0, 1, 2+ (b14_clone_day2.csv; clones absent there have n2 = 0).
  Per group: (a) mean number of the five wells the clone is present in; (b) share present in G;
  (c) share present in M; (d) share present in at least two wells. Data vs E48's best gated fit
  (s0, 200 simulations), with n2 attached to simulated clones by clone id.
- **Minimum counts.** Groups with >= 30 clones only.
- **Readings.**
  - **Abundance-linked seeding:** in the data, (a) rises from n2 = 0 to n2 = 2+ by more than the
    upper end of E48's simulated rise (95% of the simulated difference), i.e. the data gradient is
    steeper than the model's. Implication: a measured, clone-level founder rate (from n2) is a
    candidate model term, like the measured fate score m.
  - **No link beyond the model:** the data gradient lies inside E48's simulated range.
  - (b)-(d) reported as supporting detail.

### B35 result (2026-11-05)

| day-2 cells | clones (data / E48) | wells present, mean | in >= 2 wells | in G | in M |
|---|---|---|---|---|---|
| 0 | 947 / 927 | 1.22 vs 1.37 [1.33, 1.41] **out** | 0.19 vs 0.31 **out** | 0.30 vs 0.32 in | 0.42 vs 0.41 in |
| 1 | 347 / 332 | 1.25 vs 1.37 [1.30, 1.44] **out** | 0.20 vs 0.31 **out** | 0.28 vs 0.33 out | 0.38 vs 0.40 in |
| 2+ | 229 / 189 **out** | 1.35 vs 1.37 in | 0.29 vs 0.31 in | 0.35 vs 0.33 in | 0.46 vs 0.41 in |

- **Abundance-linked seeding: yes (as registered).** Data gradient (wells at 2+ minus 0) 0.13,
  above E48's simulated range [-0.09, 0.09]; all groups >= 229 clones.
- The model seeds clones that were rare at day 2 too widely (in >= 2 wells: 0.19-0.20 in the data
  vs 0.31), and it makes too few present clones with 2+ day-2 cells (229 vs 189). Clones that were
  abundant at day 2 fit.
- **Implication.** Clones differ in how many cells they had at the split, and the day-2 count
  measures it. A measured, clone-level founder rate (nu scaled by a function of n2, like the
  measured fate score) is a model term with data behind it; candidate for a later experiment
  (E51), separate from E50's G-only neutrophil class.

## E50: neutrophil-committed clones that grow only under G-CSF (pre-registered 2026-11-05)

- **Data diagnostic it must explain.** B34: G-CSF clones private to that well are
  neutrophil-dominated (0.70 vs 0.41 in shared clones), and privacy grows with size (3-5 cells:
  0.65, E48 0.43; 6-15: 0.75, E48 0.42). B7 G (-0.38; E48 -0.34) is the same pattern seen from the
  shared side. Not targeted: the M-CSF small-clone excess (B32) and the day-2 abundance link (B35,
  candidate E51).
- **Mechanism.** E48 plus class C: a clone is neutrophil-committed with probability
  logistic(logit(phi_C) + b_C z), rising with its latent fate state z (built from the measured
  day-2 score, as in E33). A class-C clone has no founders in the full, M, Epo or SCF wells; in the
  G well it draws Poisson(nu) founders, expands with mu_G + a + d_C, and makes neutrophils only.
  New parameters: phi_C in (0.001, 0.3), b_C in (0, 3), d_C in (-4, 4). 27 dimensions. Script
  experiments/e50_gneu.py; sweep sweeps/e50_gneu.txt.
- **Every-condition check (E48 best s0, nu rescaled by 1/(1 - phi_C); 30 simulations; scratchpad
  e50chk.py).**

  | phi_C, b_C, d_C | lift | n G-M | B7 G | G private 3-5 / 6-15 (n 6-15) | P(G\|M2) | presence full/G/M/Epo/SCF | N share G / M |
  |---|---|---|---|---|---|---|---|
  | data | 0.31 | 39 | -0.38 | 0.65 / 0.75 (40) | 0.04 | 368/277/478/23/39 | 0.67 / 0.040 |
  | ~0 (E48) | 0.44 | 54 | -0.31 | 0.41 / 0.42 (44) | 0.15 | 358/267/450/27/41 | 0.69 / 0.050 |
  | 0.05, 0, 0 | 0.42 | 56 | -0.35 | 0.47 / 0.45 (47) | 0.15 | 341/295/442/26/41 | 0.72 / 0.050 |
  | 0.1, 0, 0 | 0.40 | 58 | -0.35 | 0.52 / 0.47 (50) | 0.17 | 323/325/436/26/41 | 0.74 / 0.050 |
  | 0.1, 1.5, 0 | 0.40 | 56 | **-0.37** | 0.60 / 0.56 (35) | 0.16 | 300/307/442/20/32 | 0.70 / 0.040 |
  | 0.1, 1.5, 1 | 0.34 | 55 | -0.42 | 0.64 / 0.83 (92) | 0.16 | 300/370/440/19/30 | 0.80 / 0.040 |
  | 0.2, 1.5, 1 | 0.32 | 61 | -0.51 | 0.72 / 0.91 (141) | 0.18 | 262/463/440/20/31 | 0.85 / 0.034 |

  The class moves B7 G and G-side privacy to the data; with d_C > 0 it overshoots G presence and
  the 6-15 count, which mu_G and tau (fitted) must absorb. Full presence falls with phi_C (fewer
  pool clones; nu_full is fitted). M-side statistics (P(G | M2), M presence) do not move, as
  expected. Epo/SCF fall slightly at b_C 1.5.
- **Fit.** 18 cold seeds, 500 generations max, E38 targets/loss/SE, guard 9, launcher as a
  background task, PC awake.
- **Scoring.** Gate as before; `b29_rescore.py e50` (fit-set primary, stability, B7 CRPS);
  analyze_e50.py (B27, B24/B26, instruction, and for the best gated fit B34 privacy by size and the
  B31/B32 statistics, 200 simulations).
- **Readings (minimum 9 gated seeds, except best-fit readings).**
  - **Working model:** E50 replaces E48 if it is stable and passing on the fit set, its B7 CRPS sum
    is <= 0.044 (E48), and its best gated fit has G privacy at 3-5 and 6-15 cells inside its 95%
    (E48 fails both).
  - **Class used:** median phi_C >= 0.02 over gated seeds; median b_C and d_C reported.
  - **Selection vs class (descriptive):** median gamma compared with E48's 0.56; if the class takes
    over, gamma should fall.
  - **Reported, not expected to change:** B27 (lift, count, M presence), P(G | M2), D3, B26,
    instruction shifts.
  - If E50 fails the working-model rule, E48 stays the working model.

## Programme note (2026-11-05): goal hierarchy, stopping rule, discoveries register

Set by Bence. Long term: model stem-cell fate in general and eventually embryo morphogenesis.
Current step: fit conditional (perturbation) data, LARRY cytokine experiment first. Short term:
instruct vs select. New rule (program.md): a misfit gets a new model only if fixing it could move
the instruction or selection estimates by more than their spread across standing models;
otherwise it is recorded in DISCOVERIES.md. DISCOVERIES.md started with 12 findings and 5
methodological lessons, each with a literature cross-check status. Applies first to E50's result
(running, run 42101864316): E50 is judged on its pre-registered readings, and then on whether it
moves the instruction shifts or gamma.

## B36: is the Epo/SCF over-sharing a cell-type counting effect? (pre-registered 2026-11-05)

- **Data diagnostic.** B32: data clones present in G or M are seen in the Epo and SCF wells about
  half as often as E48 predicts (e.g. G 3-5-cell clones in Epo 0.02 vs 0.12). Grover 2014: Epo
  pushes multipotent progenitors toward erythroid fate. Our presence counts use only neutrophils
  and monocytes, but Epo wells are 11% N+M (902 of 8,491 day 4-6 cells: erythroid 3,687, basophil
  1,676, undifferentiated 1,081, megakaryocyte 975), and SCF wells 21% N+M (mostly undifferentiated
  4,317 of 6,113). A clone diverted to other lineages would look absent.
- **Statistics (fit set, days pooled).** For clones present with >= 2 N+M cells in G, and
  separately in M: share present in Epo, and in SCF, counting (a) N+M cells only (as the models
  do) and (b) any cell of any type. Compared with E48's best gated fit (s0, 200 simulations; the
  model only produces N+M, so its value is the N+M prediction).
- **Minimum counts.** Source groups have >= 30 clones (G 277, M 478).
- **Readings.**
  - **Counting effect:** the data's any-cell sharing for Epo is inside E48's 95% for both G and M
    sources. The model's N/M presence in Epo then stands for "any cell", and the fix is in how
    Epo/SCF presence is measured, not in the model. Same reading for SCF.
  - **Partial:** the any-cell value closes at least half the gap between the N/M-only data value
    and E48's mean.
  - **No:** less than half the gap closed.
  - Descriptive: cell-type composition of Epo cells in clones shared with G or M.

### B36 result (2026-11-05)

Sources: 277 G clones, 478 M clones (>= 2 N+M cells). E48 best gated s0, 200 simulations.

| sharing | data, N+M only | data, any cell | E48 (95%) | gap closed |
|---|---|---|---|---|
| G clones in Epo | 0.040 | 0.076 | 0.122 [0.088, 0.170] out | 44% |
| M clones in Epo | 0.025 | 0.033 | 0.074 [0.053, 0.096] out | 17% |
| G clones in SCF | 0.079 | 0.199 | 0.153 [0.113, 0.199] **in** | 163% |
| M clones in SCF | 0.048 | 0.140 | 0.097 [0.070, 0.124] out (above) | 189% |

- **Epo: no counting effect (as registered).** Counting every cell type closes 17-44% of the gap.
  The Epo cells of clones shared with G or M are mostly monocytes, basophils, undifferentiated
  cells and neutrophils (25/23/16/13); only 3 erythroid cells. **The erythroid output of the Epo
  wells comes from clones that are not in the G/M wells at all.** Grover-type diversion of shared
  clones is not the explanation; the Epo wells are dominated by a different, erythroid-lineage set
  of clones.
- **SCF: "partial" by the registered rule, in substance a counting effect.** Counting all cell
  types (SCF wells are 71% undifferentiated) brings G-sharing inside E48's range and pushes
  M-sharing above it. The SCF "over-sharing" was an artefact of counting only N+M.
- **Reading.** Epo and SCF fail for different reasons. SCF clones are the same myeloid clones,
  still undifferentiated at days 4-6. Epo clones are largely a separate, erythroid-lineage subset.
  The second pattern resembles the full arm (discovery 7): a well dominated by lineages the G/M
  wells do not support may look like a "separate pool" in N+M-only statistics.
- **Stopping rule:** Epo/SCF statistics are not part of the instruct-vs-select estimate (tau and
  gamma are identified from G, M and full). Recorded in DISCOVERIES; no model change.

### E50 result (run 42101864316, analysed 2026-11-05)

- **Run.** The guard cancelled it at 10/18 seeds: 0 gated, 10 failed (watcher exit 1 = cancel).
- **Working model: no (as registered).** No gated seed, so no stability, B7 or private-clone
  readings. E48 stays the working model.
- **Why the fits fail (descriptive, from the stored z-scores).** 7 of 10 fits are inadequate on the
  loss. The recurring misfit is G_mixed (share of G-CSF clones making both cell types) at
  z -3.8 to -6.0: a neutrophil-only class large enough to make G-private clones also strips the
  G well of mixed clones. Overlap (z +3 to +5) and Epo/SCF mean/variance misfits recur as well.
  The fits that did use the class took phi_C 0.03-0.20 with d_C 1.3-4 (large clones).
- **Instruction in these fits is unstable** (G shift 0.5-7.6, M shift 0.5-7.2), with no gated fit
  to interpret. Not read.
- **Stopping rule (program.md), applied.** E49 and E50 both failed their working-model rules, and
  no gated fit of either moved the instruction or selection estimates beyond the spread of the
  standing plausible models (E38/E47/E48: G shift 0.47-0.50, M shift 0.47-0.62). The private-clone
  features are recorded in DISCOVERIES.md as properties of the data. **LARRY cytokine modelling
  stops here.** Remaining LARRY steps: the held-out look and the write-up
  (WRITEUP_LARRY_CYTOKINE.md).

## B37: the held-out presence look, strong vs plausible selection (pre-registered 2026-11-05)

- **Go-ahead.** Bence, 2026-11-05 ("1 2 3", answering the proposed held-out look).
- **What is looked at, once.** Test `presence_heldout` in test_oracle.py (new, whole test, ledger
  shard 0/1): on Epo/SCF clone shard 1 of 2 (`split(cnt, 1)`, never used in any fit), the three
  B31 statistics B33 found able to separate strong selection from plausible or none:
  (1) G-M lift with presence >= 2 cells, (2) the same at >= 3 cells, (3) P(present in G | M-clone
  size 6+). Definitions exactly as b31_copresence.stats.
- **Predictions (fixed now, from results_e18/b33_power.json: gated seeds x 20 simulations, pooled
  95% of draws).**

  | statistic | E38 | E46 | E47 | E48 |
  |---|---|---|---|---|
  | lift >= 2 | 0.49 [0.38, 0.59] | 0.34 [0.25, 0.45] | 0.45 [0.34, 0.57] | 0.47 [0.34, 0.59] |
  | lift >= 3 | 0.28 [0.19, 0.39] | 0.13 [0.05, 0.22] | 0.23 [0.14, 0.34] | 0.25 [0.14, 0.38] |
  | P(G \| M 6+) | 0.15 [0.08, 0.22] | 0.05 [0.01, 0.10] | 0.12 [0.05, 0.19] | 0.12 [0.05, 0.20] |

- **Readings.** For each model and statistic: observed inside the model's 95% or not.
  - **Strong selection rejected on held-out clones:** E46 outside on at least 2 of 3 and E48
    inside on all 3.
  - **Strong selection favoured:** E46 inside on all 3 and E48 outside on at least 2.
  - **Not decisive:** anything else (including all models inside on all three).
  - E38 vs E48 is not separable by these statistics (B33) and is not read.
- **Caveat (from B33).** Shard-1 clones' G, M and full data entered earlier all-clone statistics,
  so the look is independent of every fit but not of everything ever computed. On the fit set the
  same statistics already favoured E48 over E46 (P(G | M 6+): data 0.11, E46 0.04); the look is a
  replication on clones no fit has used.

### B37 result: the held-out look (2026-11-05, ledger: presence_heldout 0/1, commit 77b8276)

Shard-1 clones (275 / 168 / 152 in the denominators):

| statistic | observed | E38 | E46 | E47 | E48 |
|---|---|---|---|---|---|
| lift >= 2 | **0.29** | out [0.38, 0.59] | **in** [0.25, 0.45] | out [0.34, 0.57] | out [0.34, 0.59] |
| lift >= 3 | **0.18** | out [0.19, 0.39] (just) | in [0.05, 0.22] | in [0.14, 0.34] | in [0.14, 0.38] |
| P(G \| M 6+) | **0.09** | in [0.08, 0.22] | in [0.01, 0.10] | in [0.05, 0.19] | in [0.05, 0.20] |

- **Reading (as registered): not decisive.** E46 is inside on all three, but E48 is outside on
  only one (lift >= 2), short of the "at least 2" needed for "strong selection favoured". Strong
  selection is not rejected on held-out clones either.
- **What replicates.** The low G-M lift with presence at >= 2 cells (fit set 0.31, held-out 0.29)
  is reproduced on independent clones; at >= 3 cells all plausible models fit (0.18). This is the
  small-clone privacy effect of B31/B32, now replicated. On the fit set E46 matched lift >= 2 only
  through compensating errors (B31); on the held-out clones its P(G | M 6+) prediction (0.05) is
  also close to the data (0.09; fit set 0.11).
- **Consequence for the answer.** The held-out data cannot choose between strong (E46) and
  plausible (E48) selection; the case against E46 rests on biology (Rieger bound, about twice the
  plausible gamma) and on the B31 decomposition, not on held-out evidence. The instruction
  estimate differs between them only for M-CSF (E46 on a ridge). Shard 1 is now spent for these
  statistics.

## Programme note (2026-11-05): chapter 2, SIS-seq and SIS-skew

Bence agreed the chapter plan ("Yes, write it"), now in program.md ("Chapter 2 plan"), and asked
for both diagnostics ("go 2 also"). Question: does the LARRY model family generalise to a second
system with a measured clone state and a within-clone perturbation? Every reading names its
level: clone-level selection vs within-clone change (endpoint counts cannot split within-clone
instruction from fate-specific proliferation inside a clone).

## H2: held-out split for chapter 2 (pre-registered 2026-11-05)

- Clone keys: SIS-seq `seq:<GEO sample title>`; SIS-skew `skew:<well>:<barcode ID>` (a barcode is
  a clone only within a well). Shard = `test_oracle.shard_of(key, 3)` (fixed salt). **Shard 0 of 3
  is held out** for the model step (plan step 5) and enters no B-series statistic; shards 1-2 are
  the fit set. One third, not one half, because SIS-seq is small (105 clones with both sister
  wells; about 70 in the fit set).
- Loaders in `sis_data.py` apply the split; no script reads shard 0 except through test_oracle.

## B38: is DC fate fixed in the clone before the split? SIS-seq (pre-registered 2026-11-05)

- **Data diagnostic.** LARRY's main discovery: a clone-level fate state worth 7-8 logits per SD
  drives the outcome (selection acts on it). SIS-seq measures fate in two sister wells per clone
  (GEO GSE119097 metadata: cDC1/cDC2/pDC cells in sis1, sis2) plus each well's filler-cell fate
  (background). If fate is fixed in the clone, sisters agree beyond chance.
- **Units.** Fit-set clones with > 3 cells in both sister wells. Counts rounded to integers.
- **Statistics.**
  - A. Sister distance D_sis = mean total-variation distance between the two wells' fate
    fractions. Baselines: D_rand, sister 1 of clone i vs sister 2 of a random other clone in the
    same batch (2,000 permutations); D_noise, the same distance when both wells are multinomial
    draws, at their observed cell counts, from the clone's pooled fractions (500 draws).
    Heritable share H = (D_rand - D_sis) / (D_rand - D_noise).
  - B. Between-clone SD of fate logits per fate (one vs rest), from the sister covariance
    cov(l1, l2) across clones, l = logit((k + 0.5)/(n + 1)); sigma = sqrt(max(cov, 0)).
  - C. Environment: Spearman correlation, per fate, between the sisters' fraction difference
    (sis1 - sis2) and their wells' background difference (fill1 - fill2).
  - 95% CIs by bootstrap over clones (1,000). Sensitivity: the two large batches (ILSGR0260r,
    JS96; median > 40 cells per well) alone.
- **Minimum sample.** >= 40 fit clones; else "underpowered", no reading.
- **Readings.**
  - **Fate largely fixed in the clone:** H >= 0.7 and CI lower bound > 0.5.
  - **Partly fixed:** CI excludes 0 but not the first reading.
  - **No clone memory:** CI includes 0.
  - **Cross-check with LARRY (B):** sigma >= 2 logits for at least two fates = consistent with a
    strong clone state; all sigma < 1 = in tension.
  - **Well environment matters (C):** any fate's correlation CI excludes 0 (an instruction-like
    effect of the well, at the within-clone level).
- **Synthetic check first** (scratchpad): sisters drawn from the batch's population fractions
  (expect H about 0) and from clone-specific fractions (expect H about 1).

## B39: what does Bcor loss do to sister clones? SIS-skew (pre-registered 2026-11-05)

- **Data diagnostic.** In KO wells, Bcor-KO sisters (Cherry+) make about 5x more cells than their
  WT sisters (Cherry-) and show pDC more often; one-arm clones are commoner than in null wells
  (NEXT_DATASETS check 1). Null wells (W1, W2: both arms WT, Cherry+ = control vector) give the
  baseline for every statistic.
- **Units.** Authors' pipeline (SIS_skew_analysis.rmd): reads to cells with sorted cell numbers
  and the Cherry adjustment, keep barcodes seen in both PCR replicates, sum replicates, clone size
  > 1. W3 dropped (authors: low cell numbers). Fit set only. "Both-arm" = > 3 cells in each arm.
- **Statistics, KO wells (C1-C3) and null wells (W1, W2) separately.**
  - A. Sister disagreement: H as in B38 (Cherry+ vs Cherry- arm of each clone; random pairs within
    well).
  - B. **Decomposition of the population fate shift** (Cherry+ minus Cherry-, both-arm clones),
    per fate: with w = clone share of the arm's cells and p = clone fate fraction,
    f+ - f- = sum w-(p+ - p-) [within-clone change] + sum (w+ - w-) p- [clone reweighting by the
    WT sister's fate: clone-level selection] + sum (w+ - w-)(p+ - p-) [interaction].
  - C. Size ratio vs fate: Spearman correlation of log(cells+ / cells-) with the WT arm's fraction
    of each fate, clones with > 10 cells in the WT arm.
  - D. One-arm clones: share of clones present (> 1 cell) in one arm only, by total size (2-5,
    6-20, 21-100, > 100 cells); in KO wells, KO-only vs WT-only.
  - 95% CIs by bootstrap over clones (1,000).
- **Minimum sample.** >= 150 both-arm fit clones in KO wells and >= 100 in null wells.
- **Readings (each against the null wells).**
  - **Clone-level selection by Bcor:** for some fate, the KO reweighting term's CI excludes 0
    and excludes the null wells' value; or C's KO correlation CI excludes 0 and the null's value.
  - **Within-clone change by Bcor:** for some fate, the KO within-clone term's CI excludes 0 and
    the null's value. (Within-clone instruction or fate-specific proliferation; not split here.)
  - **Sister disagreement:** H_KO's CI below H_null's CI = KO sisters diverge beyond the label.
  - Report each term's share of the KO pDC shift.
- **Synthetic check first:** same as B38 (no-memory and full-memory sisters, no KO effect),
  expecting near-zero terms in B.

### H2 / B38 / B39 results (2026-11-05, commit 3a68531; results_sis/b38.json, b39.json)

**Synthetic checks (scratchpad).** Full-memory sisters: H 0.98-1.00, CIs tight; logit SD 1.5-2.8
(the ceiling given the real clone fractions). No-memory sisters: H's CI includes 0 as it should,
but the point estimate is unstable (denominator D_rand - D_noise near 0): read H only via its CI.
Within-clone term about 0 in both. Batch differences alone give a nonzero logit SD (syn0 all
batches, pDC 1.1), so B is read against the large-batch sensitivity too.

**B38 (SIS-seq, 69 fit clones with both sister wells; large batches 41).**

| | D_sis | D_rand | D_noise | H [95%] | logit SD cDC1 / cDC2 / pDC |
|---|---|---|---|---|---|
| all | 0.17 | 0.47 | 0.05 | **0.71 [0.59, 0.78]** | 1.48 / 2.03 / 2.70 |
| large batches | 0.12 | 0.45 | 0.04 | **0.81 [0.72, 0.86]** | 1.44 / 2.11 / 2.80 |

- **Reading: fate largely fixed in the clone** (H >= 0.7, lower bound > 0.5, in both).
- **Cross-check with LARRY: consistent** (logit SD >= 2 for cDC2 and pDC). The real values sit at
  the full-memory ceiling for these clone fractions.
- **Well environment: no effect** (all three correlations' CIs include 0, e.g. pDC -0.09
  [-0.32, 0.16]).

**B39 (SIS-skew; KO wells 831 fit clones, 399 both-arm; null wells 454, 205): powered.**

| | KO wells | null wells |
|---|---|---|
| H [95%] | 0.37 [0.29, 0.44] | 0.50 [0.39, 0.60] |
| within-clone, cDC1 / cDC2 / pDC | +0.02 / +0.01 / -0.02 (all CIs include 0) | **-0.12 / +0.08 / +0.04** (all exclude 0) |
| reweighting | **-0.05** [-0.10, -0.01] / +0.06 / -0.01 | **-0.06** / **+0.05** / 0.00 |
| interaction | +0.05 / **-0.12** [-0.23, -0.04] / **+0.07** [0.02, 0.11] | **+0.06** / -0.02 / **-0.04** |
| total shift | +0.02 / -0.05 / +0.03 (all include 0) | **-0.13 / +0.12** / +0.01 |
| size ratio vs WT fate (rho) | -0.09 / +0.06 / +0.03 (all include 0) | **-0.18 / +0.23 / -0.18** |
| one-arm share by size 2-5 / 6-20 / 21-100 / >100 | 0.97 / 0.64 / 0.32 / 0.08 | 0.79 / 0.27 / 0.09 / 0.02 |

- **Readings as registered.** Clone-level selection by Bcor: **not met** (the one KO term whose CI
  excludes 0, cDC1 reweighting, equals the null's; no size-ratio correlation). Within-clone change
  by Bcor: **not met** (no KO within-clone CI excludes 0). Sister disagreement: **not met** (H CIs
  overlap).
- **The null wells are not null.** Control-vector (Cherry+) sisters differ from their Cherry-
  sisters: less cDC1, more cDC2 and pDC within clones, and their size ratio tracks fate. Either the
  vector, transduction or sorting changes fate output, or W and C wells differ in more than Bcor
  (W = WT HSPC, C = Cas9 LSK: different starting population and Cas9). Against this baseline the
  KO arms look *different*: the null's label shift is absent in KO wells (KO within-clone CIs
  exclude the null's values for cDC1 and pDC). Not a registered reading.
- **Descriptive, not registered.** (1) In KO wells the pDC gain and cDC2 loss sit in the
  interaction term: clones whose KO arm expands are the ones whose KO arm shifts toward pDC, i.e.
  fate-specific expansion inside clones or instruction in expanding clones (not separable from
  endpoints). (2) Presence: one-arm clones are 2-4x commoner in KO wells at every size; small KO
  clones are mostly WT-only (0.69 at 2-5 cells), large ones more often KO-only (0.07 vs 0.01).
  Bcor loss changes whether a clone's sisters persist at all, the strongest KO signal here.
- **Consequence for the plan.** The SIS-skew conditional comparison has no clean baseline: the
  model step must include an arm (label) effect estimated from W wells and a W-vs-C population
  difference, or restrict to KO-vs-WT arms within C wells with the label effect as a nuisance.

## Note (2026-11-05): what is known about the SIS-skew null wells

Go-ahead: Bence, "go all" (recovery check, null-well investigation, persistence). Tian 2021
(Immunity 54:1338) is paywalled and not in Europe PMC; no preprint of the SIS-skew part was
found. From the authors' own script (SIS_skew_analysis.rmd): W wells = "WT HSPC", Cherry+ =
control vector; C wells = "Cas9 LSK", Cherry+ = Bcor sgRNA, Cherry- = untransduced; the analysis
uses W only as the false-positive baseline for its symmetric SkewID call (KL of forward vs
reverse densities), never as a directional baseline. So the W-arm fate shift found in B39 is not
addressed in the available sources, and W differs from C in starting population (HSPC vs LSK)
and Cas9 as well as the vector. Treated from here as: label effect plus population difference,
both nuisances.

## B40: does Bcor loss select clones by persistence, and by fate? SIS-skew (pre-registered 2026-11-05)

- **Data diagnostic.** B39 D: one-arm clones are 2-4x commoner in KO wells at every size; small KO
  clones mostly lose the KO arm, large ones more often keep only the KO arm.
- **Units.** As B39 (fit set, authors' pipeline, W3 dropped). Present = > 1 cell in an arm.
- **Statistics, KO wells and null wells separately; 95% CIs by bootstrap over clones (1,000).**
  - A. One-arm odds: logistic regression of one-arm on log(total cells); odds ratio KO vs null
    at the pooled median size.
  - B. Direction: share of one-arm clones that are +-only, by the B39 size bins.
  - C. **Persistence by fate (clone-level selection on persistence):** clones with > 10 cells in
    the - arm: Spearman correlation between the - arm's fraction of each fate and the + arm being
    absent (<= 1 cell). Mirror: clones with > 10 cells in the + arm, - arm absence vs the + arm's
    fractions.
- **Minimum sample.** C needs >= 100 clones with - arm > 10 cells in KO wells and >= 60 in null.
- **Readings.**
  - **Bcor selects on persistence by fate:** for some fate, the KO correlation in C (either
    direction) has a CI excluding 0 and excluding the null's value.
  - **Persistence effect without fate link:** A's odds ratio CI excludes 1 and C's reading fails.
- **Caveat.** Arm absence also depends on how many sisters were transduced (Cherry+ founders);
  the transduction rate may differ between W (HSPC) and C (LSK), so A is not attributable to Bcor
  alone. C compares clones within a group and is robust to a group-wide rate.

## R1: can instruction and selection be recovered from SIS-skew statistics? (pre-registered 2026-11-05)

- **Why.** Before any fit (plan step 4): with the real sample size and a label nuisance, do the
  B39/B40 statistics identify (i) KO instruction tau (logit shift toward pDC in the KO arm) and
  (ii) KO selection gamma (KO-arm log-growth per SD of the clone's pDC logit)? LARRY lacked this
  check (E46 vs E48 stayed open).
- **Simulator `r1_skew_sim.py` (new; diagnostics it must reproduce: B39 one-arm shares by size
  and arm fate fractions, null and KO).** Per clone: fate logits z ~ N(mu, diag(sigma^2)) with
  sigma from B38 (1.5, 2.0, 2.7); split size c ~ 1 + Poisson(c0); Cherry+ founders ~ Binomial(c,
  t); each founder line grows to exp(g + noise) cells, g ~ N(m_g, s_g) per clone; arm fate
  fractions softmax(z + arm effects + eps), eps ~ N(0, s_eps) per arm; counts multinomial.
  + arm effects: label delta (logit shift on pDC) and growth lambda_L, both groups; KO wells add
  tau (pDC logit), lambda_K, and gamma x standardised z_pDC on log-growth. Nuisance values
  (mu, c0, t, m_g, s_g, s_eps, lambda_L, lambda_K) are calibrated by hand to the fit-set
  marginals (arm fate fractions, cells per arm, one-arm shares) and recorded before the recovery
  run.
- **Recovery protocol.** Draw (tau, gamma, delta) uniform on [-1.5, 1.5] x [-1, 1] x [-1, 1];
  simulate a null group (tau = gamma = 0) and a KO group at the fit-set sizes (454 and 831
  clones); summary statistics = B39 A-D minus H, plus B40 A-C, per group. 4,000 training and 300
  test simulations; estimator = ridge regression of each parameter on standardised statistics
  (pre-chosen; no tuning on test).
- **Readings.**
  - **Recoverable:** test R^2 >= 0.7 and RMSE <= 0.3 for tau and for gamma.
  - **Confounded:** the correlation of tau-hat errors with true gamma (or gamma-hat with tau)
    exceeds 0.3 in magnitude.
  - **Misspecification:** with the KO-group label shift 0.5 larger than the null's (W vs C
    difference), bias in tau-hat > 0.3 = the null cannot serve as baseline for tau.
- **Minimum.** Fixed by design (simulations); results reported whatever they are.

### B40 and R1 results (2026-11-05; results_sis/b40.json, r1_calibration.json, r1_recovery.json)

**B40 (fit set; KO 831 clones, 352 with - arm > 10 cells; null 454, 171): powered.**

- **A. One-arm odds, KO vs null at the median size: 6.2 [4.1, 11.1].** (Caveat stands: the
  transduction rate may differ between W and C.)
- **B. Direction.** In KO wells the +-only (KO-only) share of one-arm clones rises with size:
  0.30 (2-5 cells), 0.46, 0.61, 0.85 (> 100). Null wells: 0.51, 0.66, 0.60 (few large ones).
  Small KO clones lose their KO arm; large one-arm KO clones exist only as KO.
- **C. Persistence by fate.**

  | rho | KO | null |
  |---|---|---|
  | + absent vs - arm cDC1 / cDC2 / pDC | -0.06 / -0.04 / 0.00 (all include 0) | +0.07 / -0.06 / -0.01 (all include 0) |
  | - absent vs + arm cDC1 / cDC2 / pDC | **-0.17 [-0.25, -0.08]** / -0.03 / +0.07 [-0.02, 0.16] | +0.05 / +0.01 / **-0.19 [-0.32, -0.04]** |

- **Reading: Bcor selects on persistence by fate (met, weakly).** Clones that exist only in the
  KO arm are cDC1-poor (rho -0.17; CI excludes 0 and the null's +0.05). Whether a KO arm
  persists does not depend on the WT sister's fate. So Bcor loss lets non-cDC1 (cDC2/pDC-making)
  clones persist that otherwise do not, consistent with Bcor restraining cDC2/pDC expansion
  (Tian 2021). Effect small; it is one fate in one direction.
- The null wells again show their own fate-linked pattern (pDC-rich control-arm clones more
  often lack a WT arm, -0.19), absent in KO wells: more evidence that W is not a clean baseline.

**R1 calibration (nuisance search, 6,000 random + 6,000 local draws; final loss 30).** Matched
roughly: arm fate fractions (W cDC1-rich, C cDC2-rich, so the fate mix is group-specific),
one-arm shares by size (null 0.79 / 0.27 / 0.09 / 0.02 observed vs 0.79 / 0.31 / 0.17 / 0.08
simulated; KO 0.97 / 0.64 / 0.32 / 0.08 vs 0.93 / 0.58 / 0.30 / 0.14), sister TV (0.27 vs 0.26
null; 0.28 vs 0.22 KO). Missed: the + / - cell ratio runs about 0.6 log units high in both groups,
and the simulated size distribution has too few small clones. Deviation from the registration:
the search was automated, not by hand, and bounds were widened once after the first search
hit them (before any recovery run).

**R1 recovery (4,000 train / 300 test, ridge alpha 1, at fit-set sizes).**

| parameter | test R^2 | RMSE | error vs true others | bias if C label shift is 0.5 larger than W |
|---|---|---|---|---|
| tau (KO instruction, pDC logit) | 0.84 | **0.36** | gamma -0.06, delta **+0.36** | **+0.48** |
| gamma (KO selection, log-growth per SD) | **0.96** | **0.11** | tau 0.01, delta -0.05 | 0.01 |
| delta (label effect) | 0.78 | 0.27 | tau 0.25 | 0.01 |

- **Readings.** gamma: **recoverable**. tau: **not recoverable** by the registered bar (R^2
  passes, RMSE 0.36 > 0.3). Confounded (tau vs gamma): **no** (|r| <= 0.06). Misspecification:
  **tau bias 0.48 > 0.3, so the W wells cannot serve as the baseline for tau**; gamma is immune.
- **Consequence.** A SIS-skew fit can estimate clone-level selection by Bcor reliably, even with
  a label nuisance and a W-vs-C population difference. Within-clone instruction can only be
  bounded, and any W-vs-C label difference passes straight into it. The fit should report gamma
  as the main result and tau with an explicit label-difference sensitivity.

## B41: does the sisters' RNA before the split predict DC fate? SIS-seq (pre-registered 2026-11-05)

- **Go-ahead.** Bence, "go all" (fit, calibration, RNA link). Plan step 6, brought forward: it
  does not depend on the SIS-skew baseline problem.
- **Data diagnostic.** B38: fate is largely fixed in the clone (H 0.71). If so, a molecular state
  should be readable in the sisters sequenced at day 2.5 (GEO GSE119097 gene counts, Entrez IDs,
  mapped with the authors' mart_export.txt).
- **Units.** Fit-set clones (H2) with > 3 cells in both sister wells and >= 1 cell sequenced and
  >= 5,000 counts. Expression: log1p(counts per 10^4); each gene z-scored within batch (removes
  batch level differences). Fate: clone logit per fate (one vs rest), sisters pooled,
  logit((k + 0.5)/(n - k + 0.5)), centred within batch.
- **Pre-chosen programmes (from the literature, not from these data).** pDC: Tcf4 (E2-2), Spib,
  Siglech, Runx2, Ly6d, Bst2. cDC1: Id2, Batf3, Nfil3, Zbtb46. cDC2: Irf4, Klf4, Relb, Notch2.
  Irf8 left out (shared by cDC1 and pDC). Genes found by the authors in these data (e.g. Six5,
  Naaa, Plac8) are excluded to avoid circularity. Score = mean of the set's within-batch z-scores
  (genes detected in >= 10% of clones).
- **Statistics.**
  - A. Spearman rho(programme score, own fate logit) for each fate; 95% CI by bootstrap (1,000).
    Also the 3 x 3 table (each score vs each fate), descriptive.
  - B. Unsupervised: PCA of the 2,000 most variable genes (within-batch z-scores); leave-one-out
    R^2 of a linear regression of each fate logit on PC1-3.
  - C. Ceiling: the share of fate-logit variance that is clone-level, from the sister covariance
    (B38 method): cov(l1, l2) / var(pooled l), per fate.
- **Minimum sample.** >= 40 clones; else underpowered.
- **Readings.**
  - **Pre-chosen programmes predict fate:** A's CI excludes 0 (positive) for >= 2 fates.
  - **Unsupervised state predicts fate:** B's LOO R^2 > 0.1 for at least one fate.
  - **Nothing readable:** neither; then the clone fate state of B38 is not visible in day-2.5
    bulk RNA of a few sisters (or the measurement is too noisy), recorded as such.

### B41 result (2026-11-05; results_sis/b41.json): fate is readable in day-2.5 RNA, not via classic markers

- 65 fit clones (D9JS27 12, ILSGR0260r 22, JS91 15, JS96 16). Genes detected and used: pDC Ly6d,
  Runx2, Tcf4, Spib, Siglech, Bst2; cDC1 Id2, Nfil3 (Batf3, Zbtb46 below 10% detection); cDC2
  Notch2, Klf4, Relb (Irf4 undetected).
- **A. Pre-chosen programmes: not met.** Own-fate rho: pDC 0.10 [-0.16, 0.34], cDC1 0.12
  [-0.11, 0.35], cDC2 0.02 [-0.22, 0.27]. Descriptive: the cDC1 score (Id2, Nfil3) runs against
  pDC fate (rho -0.28), as expected if Id2 antagonises E2-2, but this is not a registered reading.
- **B. Unsupervised state: met.** Leave-one-out R^2 on PC1-3: pDC **0.38**, cDC2 **0.27**, cDC1
  0.06. PCs explain 5.7%, 3.4%, 2.7% of expression variance; the fate signal sits mainly on PC3.
- **C. Ceiling:** clone-level share of fate-logit variance 0.66 (cDC1), 0.84 (cDC2), 0.83 (pDC).
  So the RNA state explains about 45% (pDC) and 32% (cDC2) of the clone-level, heritable part.
- **Descriptive (not registered): what the predictive axis is.** Genes weighted toward pDC fate:
  Satb1, Tox, Lrmp, Arpp21, St7l, Gimap5 (lymphoid-associated); toward cDC2: Csf1r, Grn, H2-Aa,
  Spn, Ctsg, Anxa2 (myeloid-associated). One lymphoid-vs-myeloid priming axis separates pDC from
  cDC2 clones at day 2.5, before any marker of the final type is expressed.
- **Reading overall.** The clone fate state of B38 is partly visible in the sisters' RNA as a
  broad priming programme, not as early expression of lineage-defining factors.

## R2: improved SIS-skew simulator and recovery with every nuisance varied (pre-registered 2026-11-05)

- **Go-ahead.** Bence, "go all" (tighten calibration, rerun recovery, then fit).
- **Diagnostics the simulator must explain.** R1's misses: the + / - cell ratio (about 0.6 log
  units high in both groups) and too few small clones; B39/B40: one-arm shares by size and their
  direction, decomposition terms, size-ratio correlations, persistence-by-fate.
- **Changes from R1 (`r2_skew_sbi.py`, reuses R1's simulator logic and statistics).**
  - Transduction rate per group (t_W for HSPC + vector, t_C for LSK + sgRNA): fixes the cell
    ratio and allows different one-arm rates without Bcor.
  - Founder-line extinction probability q (both groups): makes small clones and lost arms.
  - Effects as vectors over (cDC2, pDC), relative to cDC1: label delta (both groups), KO
    instruction tau (logit shift of the KO arm), KO selection gamma (KO-arm log-growth per SD of
    the clone's standardised cDC2 and pDC logits relative to cDC1).
  - 20 parameters; priors uniform: mu_cDC2_W (-2, 2), mu_pDC_W (-5, 1), mu_cDC2_C (-1, 4),
    mu_pDC_C (-4, 1), c0 (0.5, 6), t_W and t_C (0.1, 0.7), q (0, 0.6), m_g (-0.5, 3), s_g (0.2, 2),
    s_line (0.2, 2), s_eps (0.3, 2), lam_L (-2, 2), lam_K (-1, 4), delta (-1.5, 1.5)^2,
    tau (-2, 2)^2, gamma (-1, 1)^2.
- **Inference (fixed now).** Semi-automatic ABC (Fearnhead and Prangle 2012): 20,000 prior
  simulations at fit-set sizes; summary statistics = R1's 38 per group (76); ridge regression
  (alpha 1) of each parameter on standardised statistics gives the projection; accept the 2% of
  simulations nearest the target in the standardised projected space; local-linear regression
  adjustment (Beaumont 2002); 95% interval = 2.5-97.5 percentiles of the adjusted sample.
- **Identifiability.** tau and a W-vs-C label difference enter the model identically (both shift
  the + arm in KO wells only), so tau is estimated under delta_C = delta_W and is reported as
  "tau + (delta_C - delta_W)".
- **Recovery protocol.** 300 test simulations from the prior, each inferred as above.
- **Readings.** Per parameter (gamma_cDC2, gamma_pDC, tau_cDC2, tau_pDC):
  **recoverable** if RMSE <= 0.3 and the 95% interval covers the truth in 90-99% of tests;
  **miscalibrated** if coverage < 90%. Results reported whatever they are.

## E51: SIS-skew model fit, clone-level selection by Bcor (pre-registered 2026-11-05)

- **Runs only after R2**, on the fit set (H2 shards 1-2), with R2's simulator, priors and
  inference unchanged.
- **Outputs.** Posterior medians and 95% intervals for all 20 parameters; main results gamma
  (clone-level selection) and tau + label difference (within-clone change, bounded).
- **Readings.**
  - **Clone-level selection by Bcor on fate:** a gamma interval excludes 0 (only for parameters
    R2 calls recoverable).
  - **No clone-level fate selection:** both gamma intervals inside [-0.25, 0.25].
  - **Within-clone change (tau + label difference):** an interval excludes 0; reported with the
    caveat that it cannot be split from a W-vs-C label difference.
- **Fit adequacy.** Posterior predictive (200 draws from the accepted sample): the share of the
  76 fit-set statistics inside their 95% predictive ranges; adequate if >= 80%. Misses are listed.
- **Held-out look (plan step 5).** After the fit and before the look, the posterior predictive
  ranges at held-out sizes are written into this log for: KO decomposition terms (9), KO and null
  size-ratio rho (6), one-arm shares by size (8), KO persistence-by-fate rho (6). Test
  `skew_heldout` (new, in test_oracle.py) then computes the same statistics on shard 0, once.
  **Model adequate on held-out clones:** >= 80% of those 29 statistics inside their ranges.

### E51 fit result and held-out predictions (2026-11-05, written before the look)

- **R2 recovery (300 tests):** gamma_cDC2 RMSE 0.24, coverage 0.94; gamma_pDC 0.26, 0.95: both
  **recoverable**. tau_cDC2 RMSE 0.58, tau_pDC 0.72 (coverage 0.95, mean interval width 2.1-2.5):
  **not recoverable** (calibrated but wide).
- **E51 posterior (fit set), median [95%]:** gamma_cDC2 0.05 [-0.38, 0.43], gamma_pDC 0.19
  [-0.24, 0.61]; tau_cDC2 + label diff -0.97 [-1.85, 0.11], tau_pDC + label diff -0.19
  [-1.40, 1.22]; lam_K (KO-arm growth, all clones) **2.28 [1.45, 3.20]**; label delta_cDC2 0.61
  [-0.11, 1.31], delta_pDC 0.05 [-0.92, 1.11]; transduction t_W 0.61 [0.48, 0.70] vs t_C 0.35
  [0.23, 0.50]; line extinction q 0.20 [0.00, 0.43]; full table in results_sis/e51_fit.json.
- **Readings.** Selection: **undetermined** (both gamma intervals include 0 but exceed +-0.25).
  Within-clone change (tau + label difference): **not shown** (both include 0). Fit adequacy:
  **adequate**, 84% of 76 statistics inside (64 of 76).
- **Misses (12):** null: pooled - arm pDC fraction; size-ratio rho cDC1, cDC2; + absent vs - arm
  pDC; size-bin shares 6-20, 21-100. KO: **all three interaction terms**; - absent vs + arm cDC2;
  size-bin shares 2-5, 21-100. The model cannot make clones whose KO arm both expands and shifts
  fate: a within-clone, fate-specific expansion it does not contain.
- **Held-out predictions** (posterior predictive, 400 draws, at the expected held-out sizes
  227 null / 415 KO clones; 95% ranges). Model adequate on held-out clones if
  >= 24 of these 29 are inside:

| statistic | 95% predictive range |
|---|---|
| KO within cDC1 | [-0.063, 0.140] |
| KO within cDC2 | [-0.144, 0.068] |
| KO within pDC | [-0.053, 0.072] |
| KO reweighting cDC1 | [-0.122, 0.079] |
| KO reweighting cDC2 | [-0.120, 0.125] |
| KO reweighting pDC | [-0.055, 0.116] |
| KO interaction cDC1 | [-0.069, 0.059] |
| KO interaction cDC2 | [-0.060, 0.076] |
| KO interaction pDC | [-0.049, 0.040] |
| null size-ratio rho cDC1 | [-0.196, 0.227] |
| null size-ratio rho cDC2 | [-0.233, 0.193] |
| null size-ratio rho pDC | [-0.247, 0.217] |
| KO size-ratio rho cDC1 | [-0.369, 0.218] |
| KO size-ratio rho cDC2 | [-0.263, 0.298] |
| KO size-ratio rho pDC | [-0.209, 0.298] |
| null one-arm 2-5 | [0.658, 0.941] |
| null one-arm 6-20 | [0.240, 0.512] |
| null one-arm 21-100 | [0.042, 0.309] |
| null one-arm >100 | [0.000, 0.300] |
| KO one-arm 2-5 | [0.760, 1.000] |
| KO one-arm 6-20 | [0.373, 0.815] |
| KO one-arm 21-100 | [0.133, 0.398] |
| KO one-arm >100 | [0.028, 0.186] |
| KO +absent vs - cDC1 | [-0.076, 0.209] |
| KO +absent vs - cDC2 | [-0.096, 0.160] |
| KO +absent vs - pDC | [0.071, 0.429] |
| KO -absent vs + cDC1 | [-0.097, 0.143] |
| KO -absent vs + cDC2 | [-0.092, 0.127] |
| KO -absent vs + pDC | [0.041, 0.338] |

### E51 held-out look (2026-11-05, test skew_heldout 0/1, commit 5eab427, taken once)

Held-out shard: 218 null and 461 KO clones (predicted at 227 / 415).

| statistic | held-out | 95% predictive | |
|---|---|---|---|
| KO within cDC1 | -0.085 | [-0.063, 0.140] | **out** |
| KO within cDC2 | 0.023 | [-0.144, 0.068] | in |
| KO within pDC | 0.062 | [-0.053, 0.072] | in |
| KO reweighting cDC1 | 0.019 | [-0.122, 0.079] | in |
| KO reweighting cDC2 | 0.071 | [-0.120, 0.125] | in |
| KO reweighting pDC | -0.091 | [-0.055, 0.116] | **out** |
| KO interaction cDC1 | 0.026 | [-0.069, 0.059] | in |
| KO interaction cDC2 | 0.007 | [-0.060, 0.076] | in |
| KO interaction pDC | -0.033 | [-0.049, 0.040] | in |
| null rho cDC1 | -0.127 | [-0.196, 0.227] | in |
| null rho cDC2 | 0.136 | [-0.233, 0.193] | in |
| null rho pDC | -0.058 | [-0.247, 0.217] | in |
| KO rho cDC1 | -0.054 | [-0.369, 0.218] | in |
| KO rho cDC2 | 0.155 | [-0.263, 0.298] | in |
| KO rho pDC | -0.119 | [-0.209, 0.298] | in |
| null one-arm 2-5 | 0.806 | [0.658, 0.941] | in |
| null one-arm 6-20 | 0.208 | [0.240, 0.512] | **out** |
| null one-arm 21-100 | 0.014 | [0.042, 0.309] | **out** |
| null one-arm >100 | 0.028 | [0.000, 0.300] | in |
| KO one-arm 2-5 | 0.984 | [0.760, 1.000] | in |
| KO one-arm 6-20 | 0.742 | [0.373, 0.815] | in |
| KO one-arm 21-100 | 0.282 | [0.133, 0.398] | in |
| KO one-arm >100 | 0.062 | [0.028, 0.186] | in |
| KO +abs vs - cDC1 | -0.068 | [-0.076, 0.209] | in |
| KO +abs vs - cDC2 | 0.068 | [-0.096, 0.160] | in |
| KO +abs vs - pDC | 0.270 | [0.071, 0.429] | in |
| KO -abs vs + cDC1 | 0.085 | [-0.097, 0.143] | in |
| KO -abs vs + cDC2 | 0.166 | [-0.092, 0.127] | **out** |
| KO -abs vs + pDC | 0.075 | [0.041, 0.338] | in |

- **Reading: model adequate on held-out clones** (24 of 29 inside; bar >= 24, met exactly).
- **Misses (5):** null one-arm share at 6-20 and 21-100 cells (the model makes too many one-arm
  null clones at mid sizes); KO within-clone cDC1 (-0.085, below range); KO reweighting pDC
  (-0.091, below range); KO "- absent vs + arm cDC2" (+0.17, above).
- **Replication of fit-set patterns.** The KO interaction terms that the model missed on the fit
  set are inside on held-out clones (pDC fit +0.066, held-out -0.033): **the expand-and-shift
  pattern (B39) does not replicate.** B40's persistence-by-fate correlation (KO-only clones
  cDC1-poor, fit -0.17) is +0.09 on held-out clones: **does not replicate**; the held-out sign
  is opposite. Both were fit-set patterns of about one CI width.
- **Answer for chapter 2 so far.** Bcor loss expands the KO arm of every clone about 10x
  (lam_K 2.3 [1.5, 3.2]) and changes whether arms persist (one-arm odds 6x, partly a lower
  transduction rate in C wells: t_C 0.35 vs t_W 0.61). Fate-linked clone selection is not
  detected and is not excluded (gamma intervals about +-0.4); within-clone fate change is not
  shown and cannot be separated from the vector-label difference.

## C1: one core model for LARRY and SIS (pre-registered 2026-11-06)

- **Go-ahead.** Bence, "go 1 2 3" (editing-aware fit + B42; shared core model; spatial plan),
  answering RETRO_2026-11-06 decisions 1-4.
- **Why.** Chapter 2 asked whether *the same* model generalises; it used a stand-alone simulator.
  `core_model.py` (new) holds one generative model; each data set is a configuration.
- **Core generative steps (per clone).**
  1. Clone state z in R^d; per-condition (or per-arm) state z_c = sqrt(rho) z + sqrt(1 - rho) noise
     (fate memory). Fate logits = B z_c + tau_c (instruction), K fates, softmax.
  2. Split: either independent Poisson founders per condition with class-specific rates (LARRY:
     wells, separate-pool class) or a partition of c founder cells across arms (SIS:
     transduction).
  3. Each founder line grows to exp(mu_c + a + gamma_c . s(z) + s_line noise), dies with
     probability q; a fraction e of lines in a perturbed arm carry the perturbation (editing
     efficiency; LARRY e = 1).
  4. Output per fate: mean = growth x (free x p_f + committed_f) x exp(sfate_c,f) (sfate =
     fate-specific cell-level expansion; LARRY's s_G), Poisson or multinomial counts.
- **Equivalence test (fixed now).** LARRY config at E48's best gated parameters vs
  `e48_capsel6.simulate`; SIS config (e = 1, sfate = 0) vs `r2_skew_sbi.simulate` at the E51
  posterior median. 300 simulations each; for every summary statistic (LARRY: e36.targets;
  SIS: the 76 R2 statistics), |mean difference| <= 3 Monte Carlo SE (or <= 0.01 absolute).
  **Equivalent** if >= 95% of statistics pass. Only then are R3/E52 run on the core.
- **Not in this step:** refitting LARRY through the core (needs GitHub Actions); proposed next.

## R3 / E52: editing-aware SIS-skew fit (pre-registered 2026-11-06)

- **Diagnostic it must explain.** Tian 2021: Bcor indel efficiency 40%; "major clonal expansion
  in a proportion of Bcor-/Y clones". E51 assumed every KO-arm cell edited.
- **Model = core SIS config.** R2 plus: each Cherry+ founder line in C wells edited with
  probability e, prior uniform (0.15, 0.55) (40% indels is an upper bound on functional loss;
  Bcor is X-linked, so one allele); KO effects (lam_K, gamma, tau) act on edited lines only;
  fate-specific cell-level expansion in edited lines, sfate_pDC and sfate_cDC2 (prior uniform
  (-1.5, 1.5)), the mechanism E51 lacked. 23 parameters; other priors as R2.
- **Inference, recovery and readings:** as R2/E51 (20,000-simulation bank, semi-automatic ABC,
  300 recovery tests; recoverable if RMSE <= 0.3 and coverage 90-99%), for gamma, tau, sfate
  and e. E52 on the fit set: selection reading as E51; **fit adequacy >= 80%**. The SIS-skew
  held-out shard is spent (E51), so E52 has no held-out look; it is judged on the fit set and B42.
- **Stopping-rule check.** If E52's gamma intervals move by more than their E51 width, the
  chapter-2 answer is rewritten from E52; otherwise E51 stands with E52 as sensitivity.

## B42: is Tian's "gain of pDC fate potential" expansion plus sampling? (pre-registered 2026-11-06)

- **Claim tested.** Tian 2021: 78% of SkewID-skewed KO clones gained pDC output (binary).
- **Statistic (fit set, KO wells; null wells as check).** Among clones present (> 1 cell) in
  both arms: G = share with pDC present (> 1 cell) in the + arm and absent in the - arm;
  L = the reverse; Net = G - L.
- **Predictions.** Posterior predictive (400 draws) of G and Net from (i) E52's posterior and
  (ii) E52's posterior with tau and sfate set to 0 (expansion, editing and sampling only).
- **Readings.**
  - **Expansion plus sampling explains it:** observed G and Net inside (ii)'s 95% ranges.
  - **Needs a fate effect:** outside (ii), inside (i).
  - **Unexplained:** outside both.

### C1, R3, E52, B42 results (2026-11-06; results_sis/c1_equiv_*.json, r3_recovery.json, e52_fit.json, b42.json)

**C1: one core model. Equivalent for both data sets.** core_larry vs e48 simulate at E48's
best fit: 28 of 28 targets within 3 MC SE (300 sims each). core_sis (e = 1, sfate = 0) vs R2's
simulator at the E51 median: 76 of 76. LARRY and SIS-skew now run through the same five-step
generative core (core_model.py). Refitting LARRY through the core is still to do (Actions).

**R3 recovery (editing-aware, 23 parameters, 20,000 bank, 300 tests).**

| parameter | RMSE | coverage | mean width | R^2 | reading |
|---|---|---|---|---|---|
| gamma_cDC2 | 0.44 | 0.94 | 1.37 | 0.49 | not recoverable |
| gamma_pDC | 0.44 | 0.91 | 1.42 | 0.41 | not recoverable |
| tau_cDC2 | 0.94 | 0.94 | 3.15 | 0.29 | not recoverable |
| tau_pDC | 1.06 | 0.92 | 3.31 | 0.16 | not recoverable |
| sfate_cDC2 | 0.71 | 0.94 | 2.26 | 0.33 | not recoverable |
| sfate_pDC | 0.78 | 0.90 | 2.40 | 0.25 | not recoverable |
| e (editing) | 0.11 | 0.90 | 0.34 | **0.06** | "recoverable" by the bar, but R^2 0.06: the RMSE equals the prior SD; e is not learned |

- **Reading.** Once editing efficiency and fate-specific expansion are in the model, no fate
  parameter is recoverable at the SIS-skew sample size. Intervals stay calibrated (coverage
  0.90-0.94) but wide. R2's "gamma recoverable" depended on assuming every KO-arm cell edited.

**E52 (fit set).** Fit adequacy **89.5%** (68 of 76; E51 84%); the KO interaction terms that E51
missed now fit (fate-specific expansion of edited lines). Posterior: gamma_cDC2 0.34 [-0.28,
0.89], gamma_pDC 0.10 [-0.54, 0.78]: **undetermined**; tau + label difference: cDC2 -1.18
[-2.00, 0.71], pDC -0.32 [-1.93, 1.45]: not shown; sfate cDC2 -0.07 [-1.25, 1.25], pDC 0.19
[-1.02, 1.50]; e 0.34 [0.19, 0.50] (prior-like); lam_K 3.30 [2.00, 4.00] (edited lines about
27x; at the prior's upper edge); t_C 0.41 vs t_W 0.63.
- **Stopping-rule check.** gamma medians moved 0.29 (cDC2) and 0.09 (pDC), less than E51's
  interval widths (0.81, 0.85): the E51 answer is not overturned. But R3 shows neither fit can
  identify gamma once editing is modelled, so the chapter-2 answer for Bcor becomes: **expansion
  of edited sister lines, yes; any fate-level mechanism (selection or instruction) not
  identifiable from SIS-skew at this size.**

**B42 (fit set, KO wells; null as check).**

| statistic | observed | E52 full 95% | E52 without fate effects (tau = sfate = 0) 95% |
|---|---|---|---|
| KO pDC gain G | 0.213 | [0.080, 0.235] | [0.079, **0.2126**] (observed 0.2126: inside by 0.0001) |
| KO pDC loss L | 0.056 | [0.044, 0.120] | [0.048, 0.135] |
| KO net gain | **0.156** | [-0.020, 0.173] | [-0.047, **0.149**] (outside) |
| null net gain | 0.000 | [-0.120, 0.040] | [-0.109, 0.038] |

- **Reading as registered: "needs a fate effect"** (net gain outside the no-fate range, inside
  the full one). **Marginal:** net gain exceeds the no-fate limit by 0.007, the gain share sits
  on its limit, and the full and no-fate ranges barely differ (fate parameters not identifiable,
  R3). Most of Tian 2021's "gain of pDC fate potential" is reproduced by expansion of edited
  lines plus sampling; a small excess remains, consistent with a modest pDC-directed effect.

## E53: the LARRY answer refitted through the core model (pre-registered 2026-11-06)

- **Go-ahead.** Bence, "go 1" (refit LARRY through the core on GitHub Actions).
- **Why.** C1 showed core_larry reproduces E48's simulator in distribution (28/28 targets). E53
  refits LARRY with the core in place of e46/e48's simulate, everything else unchanged (E48's
  unpack with gamma <= 0.6, targets e36.targets, 10 sims per evaluation, CMA-ES maxiter 500,
  dim 24, 18 seeds s0-s17, `run_sweep.sh sweeps/e53_core.txt --larry --guard 9`). Results to
  results_e18/e53_s*.json; analysis with `analyze_model.py e53 e53_core`.
- **E48's gated ranges (min / median / max over 9 gated seeds), fixed now:** instruction shift
  G 0.38 / 0.50 / 0.70, M 0.40 / 0.56 / 0.92; gamma 0.482 / 0.558 / 0.568; s_G 1.64 / 2.10 /
  2.85; s_M 0.18 / 0.23 / 0.28; beta 6.9 / 7.8 / 9.5.
- **Readings.**
  - **Core reproduces the LARRY answer:** >= 9 gated seeds (stability rule) and the E53 gated
    medians of shift_G, shift_M, gamma, s_G, s_M and beta all inside E48's min-max ranges above.
  - **Partly:** stable, but one or two medians outside; listed and explained.
  - **Not reproduced:** < 9 gated seeds or three or more medians outside: the core differs from
    E48 in a way the C1 equivalence test missed (C1 compared one parameter point).
- No held-out look (LARRY's held-out tests are spent).

### E53 result (2026-11-06; run 42203864786, 18/18 seeds, 40 min; results_e18/e53_analysis.json)

| quantity (gated median) | E53 (core) | E48 gated range | |
|---|---|---|---|
| instruction shift G | 0.52 | 0.38-0.70 (median 0.50) | in |
| instruction shift M | 0.70 | 0.40-0.92 (median 0.56) | in |
| gamma | 0.560 | 0.482-0.568 | in |
| s_G | 2.11 | 1.64-2.85 | in |
| s_M | 0.25 | 0.18-0.28 | in |
| beta | 7.38 | 6.9-9.5 | in |

- Gated seeds 9 of 18 (0, 1, 4, 9, 10, 14, 15, 16, 17): **stable**.
- **Reading: core reproduces the LARRY answer.** LARRY and SIS-skew now come from one model
  code; the chapter-2 question "does the same model family fit both?" is answered **yes** for
  the fit. (What it can identify differs by data set: LARRY identifies instruction and
  selection; SIS-skew with editing modelled identifies neither fate parameter, R3.)

## E54: does the measured RNA axis carry the clone state the sister fates imply? SIS-seq (pre-registered 2026-11-06)

- **Go-ahead.** Bence, "go both in order" (state-RNA link, then label sensitivity). Plan step 6.
- **Model (core SIS-seq configuration, `core_sisseq.py`).** Per clone, fate state z over (cDC1,
  cDC2, pDC); for each fate f, z_f = mu_f(batch) + sigma_f (a_f r_f + sqrt(1 - a_f^2) xi_f), where
  r_f is the clone's measured RNA score for fate f (standardised) and a_f its loading on the
  hidden state (LARRY's analogue: w = 0.95 for the day-2 Cebpa-Irf8 score). Each sister well:
  counts ~ Poisson per fate from softmax(z + s_eps noise) at the observed cell numbers (the core's
  state + sampling steps; no split or growth needed).
- **RNA score, defined on the fit set only.** As B41: log1p(CP10k), genes detected in >= 10%,
  z-scored within batch with fit-set batch means and SDs, PCA of the 2,000 most variable genes
  (fit set), linear regression of each fate's pooled logit on PC1-3. Fit-set r_f = leave-one-out
  predictions (no in-sample optimism); held-out r_f = held-out clones projected with fit-set
  means, SDs, PC loadings and regression weights.
- **Estimator (method of moments, errors-in-variables).** sigma_f^2 = cov(l1, l2) across clones
  (sister logits, l = logit((k + 0.5)/(n - k + 0.5)), centred within batch);
  a_f = cov(r_f, (l1 + l2)/2) / (sd(r_f) sigma_f). 95% CI by bootstrap over clones.
- **Recovery check first (R4).** Core SIS-seq simulator at the fit-set size (65 clones, observed
  well sizes), true a_f uniform on [-0.9, 0.9], 300 tests: recoverable if RMSE <= 0.2 and the
  bootstrap CI covers the truth in >= 90%.
- **Fit-set readings (for fates where R4 says recoverable).**
  - **The RNA axis carries the hidden state:** a_f CI excludes 0 for pDC or cDC2.
  - **Share of the clone state read from RNA:** a_f^2 with CI, compared with LARRY's w^2 = 0.90.
- **Held-out look (test `sisseq_rna_heldout`, H2 shard 0 of SIS-seq, taken once).** Before the
  look, the posterior predictive 95% range of corr(r_f, pooled logit_f) at the held-out size is
  computed from the fit-set estimates (core simulator, 1,000 draws, parameter uncertainty from
  the bootstrap) and written into this log. **Link replicates on held-out clones:** the observed
  correlation is inside its predictive range and > 0, for pDC and cDC2.

## S1: how far can a W-vs-C label difference move the Bcor readings? (pre-registered 2026-11-06)

- **Why.** tau and a W-vs-C label difference are confounded (R2/E51 note); the null wells differ
  from KO wells in population (cKit+ vs LSK) as well as vector (Tian 2021).
- **Runs.** E52 repeated twice with an extra fixed label shift d added to the + arm's cDC2 and
  pDC logits in C wells only, d = +0.5 and d = -0.5 (same priors, bank size, inference); then
  B42 on each posterior.
- **Readings.** tau moves by about -d by construction (reported, not read). **Robust:** gamma
  intervals and the B42 reading unchanged in substance (gamma still undetermined, B42 net gain
  on the same side of the no-fate limit) under both shifts. **Sensitive:** otherwise, listed.

### E54 amendment (2026-11-06, before any reading or held-out look): attenuation-corrected estimator

- **R4 as registered failed for all fates:** cDC1 RMSE 0.25 / coverage 0.54; cDC2 0.17 / 0.68;
  pDC 0.12 / 0.88 (bar: RMSE <= 0.2, coverage >= 0.90). So no fit-set reading is made from the
  raw estimator.
- **Disclosure.** A smoke test printed the raw fit-set estimates (a = 0.37 / 0.58 / 0.66) before
  R4 ran. Fit set only; the held-out shard is untouched.
- **Diagnosis (600 calibration simulations, seed 57).** The raw estimator is attenuated, linear,
  no offset: est = 0.58 x true (cDC1), 0.73 x true (cDC2), 0.88 x true (pDC); residual SD
  0.10-0.12. Cause: one-vs-rest logits of a three-fate softmax state carry only part of each
  fate's state.
- **Amended estimator (fixed now).** a_corr,f = a_raw,f / k_f with k = (0.58, 0.73, 0.88); the
  bootstrap CI is divided by the same k_f. R4 is re-run on the corrected estimator with fresh
  simulations (seed 58) and the registered bar. Readings, predictions and the held-out look then
  proceed exactly as registered, using a_corr (the simulator's own scale).

### E54 fit-set result and held-out predictions (2026-11-06, written before the look)

- **R4 (corrected estimator, 300 fresh tests, seed 58):** cDC1 RMSE 0.20, coverage 0.95; cDC2
  0.15, 0.92; pDC 0.12, 0.91: all **recoverable**. s_eps recalibrated 0.70.
- **Fit set (65 clones), loading a_f [95%] and share a_f^2:**

| fate | a (corrected) | a^2 (share of clone state read from RNA) | sigma (clone logit SD) |
|---|---|---|---|
| cDC1 | 0.65 [0.39, 1.14] | 0.42 [0.15, 1.30] | 1.36 |
| cDC2 | **0.80 [0.61, 1.06]** | 0.64 [0.38, 1.12] | 1.83 |
| pDC | **0.75 [0.66, 0.93]** | 0.57 [0.43, 0.86] | 2.52 |

- **Reading: the RNA axis carries the hidden state** (CI excludes 0 for pDC and cDC2; cDC1 too).
  The day-2.5 RNA score reads about 55-65% of the clone fate state, against 90% for LARRY's day-2
  Cebpa-Irf8 score (w = 0.95). CIs above 1 come from the correction and sampling noise.
- **Held-out predictions** (core simulator, 1,000 draws over bootstrap parameters, expected 32
  held-out clones; corr(r_f, pooled logit_f), 95%): cDC1 [0.03, 0.74]; **cDC2 [0.35, 0.87]**;
  **pDC [0.44, 0.88]**. Link replicates if the held-out correlations for pDC and cDC2 are inside
  these ranges and > 0.

### E54 held-out look (2026-11-06, test sisseq_rna_heldout 0/1, commit 3d01ff9, taken once)

| fate | held-out corr (n = 34) | predicted 95% | |
|---|---|---|---|
| cDC1 | 0.31 | [0.03, 0.74] | in |
| cDC2 | **0.25** | [0.35, 0.87] | **below** |
| pDC | **0.29** | [0.44, 0.88] | **below** |

- **Reading: the link does not replicate at the predicted strength** (pDC and cDC2 below range).
  The sign replicates for all three fates, but at n = 34 none is individually significant
  (r = 0.29 gives p about 0.09). Back-of-envelope, the held-out values imply a loading of about
  0.35-0.45 (share about 15-20% of the clone state), half the fit-set estimate.
- **Why the fit set overstated it.** The RNA score was chosen on the fit set (PCA on fit clones,
  then regression on fate); leave-one-out predictions remove in-sample fitting of the weights but
  not the selection of the PC space from the same clones. The held-out clones are the honest
  estimate.
- **Consequence for the goal.** The day-2.5 RNA of a few sisters carries a weak, positive trace of
  the clone fate state; most of the state that sisters share (B38, H 0.71) is not readable from
  this RNA at this depth. The hidden state stays mostly hidden.

### S1 result (2026-11-06; banks 20,000 each, fits and B42 as E52)

| | E52 (d = 0) | d = +0.5 | d = -0.5 |
|---|---|---|---|
| gamma_cDC2 | 0.34 [-0.28, 0.89] | 0.28 [-0.32, 0.80] | 0.36 [-0.22, 0.96] |
| gamma_pDC | 0.10 [-0.54, 0.78] | 0.27 [-0.38, 1.00] | -0.12 [-0.75, 0.51] |
| tau_cDC2 | -1.18 [-2.0, 0.71] | -1.38 [-2.0, 0.60] | 0.10 [-1.42, 1.95] |
| e (editing) | 0.34 [0.19, 0.50] | 0.40 [0.23, 0.55] | 0.42 [0.24, 0.55] |
| selection reading | undetermined | undetermined | undetermined |
| B42 KO net pDC gain (obs 0.156) vs no-fate 95% upper | 0.149 | 0.182 | 0.165 |
| B42 reading | needs a fate effect | expansion plus sampling | expansion plus sampling |

- **gamma: robust** (undetermined under both shifts).
- **B42: sensitive.** The marginal "needs a fate effect" (#18) flips to "expansion plus sampling
  explains it" under both shifts, so it was within the noise of the model choices, not a finding.
- tau moved asymmetrically (+0.5: -0.2; -0.5: +1.3), not the -d expected by construction; tau sits
  on its prior bound at d = 0 and +0.5, so the bound absorbs the shift. Reported, not read.
- Bug fixed before the B42 runs: `b42()` reused `d` for the data and the shift (TypeError, no
  output); renamed the data `D`. E52's b42.json was produced with d = None and is unaffected.

## E55: do pretrained single-cell RNA models read more of the clone fate state than PCA? (pre-registered 2026-11-06)

- **Go-ahead.** Bence, "Yes, register E55" (after E54 found only a weak RNA trace of the SIS clone
  state).
- **Question.** Is the part of a clone's fate bias that day-2 RNA misses (track B: about 55% in
  LARRY; E54: most of it in SIS-seq) a feature problem that a pretrained model fixes, or absent
  from the RNA? Literature prior: zero-shot Geneformer/scGPT embeddings often do no better than
  highly-variable genes or PCA (Kedzierska et al. 2025 Genome Biology), so the expected answer is
  "no gain".
- **Primary data: LARRY in vitro (Weinreb 2020).** Clones with >= 1 day-2 cell and >= 3 day-4/6
  neutrophil + monocyte cells (571 clones, track B). Target: the clone's later N / (N + M). Clone
  state = mean of its day-2 cells' features.
- **Sealing.** New clone shard for this question: `shard_of("e55:<clone>", 3) == 0` is held out
  (about 190 clones), read once through the oracle (test `larry_fm_heldout`). **Prior exposure,
  disclosed:** track B already used all 571 clones (PCA-50 ridge, 5-fold CV r 0.66) and B16 used
  this data; no foundation-model feature has been computed on it.
- **Features (all label-free, computed per cell).**
  1. **PCA-50 baseline:** log1p of normalised counts, 2,000 most variable genes and PCs fitted on
     fit-shard day-2 cells only; held-out cells projected.
  2. **Marker score:** day-2 Cebpa minus Irf8 (z-scored), as B16.
  3. **Geneformer** (ctheodoris/Geneformer, Apache-2.0, safetensors; default model, plus the
     smaller V1 model if the repo carries it): mouse symbols mapped to one-to-one human orthologs
     (Ensembl BioMart), rank-value tokenisation with Geneformer's gene medians, cell embedding =
     mean over genes of the penultimate layer (the authors' recommended setting).
  4. **scGPT** (tdc/scGPT safetensors, MIT; human gene vocabulary via the same orthologs): cell
     embedding as its authors' `embed_data`. Dropped and logged if it will not run on CPU.
  At most 4 foundation candidates; no fine-tuning (65-571 clones are too few).
- **Prediction model.** Ridge on standardised clone features; lambda chosen by inner 5-fold CV
  inside each outer fold (nested). PCA is refitted inside each outer fold.
- **Choice on the fit shard only.** 5-fold CV r on fit-shard clones for every candidate; the best
  foundation candidate is named in this log before the look.
- **Held-out readings (one look; minimum 150 held-out clones, else descriptive only).** Fit on
  all fit-shard clones, predict held-out clones. delta = r(best foundation) - r(PCA-50), 95% CI by
  paired bootstrap over held-out clones (5,000).
  - **Foundation model reads more of the state:** delta CI above 0.
  - **No gain:** CI includes 0. **PCA better:** CI below 0.
  - **Secondary (same look):** r of PCA-50 + best foundation concatenated vs PCA-50 alone (does it
    add information?); each r reported against the target's sampling-noise ceiling.
  - Resolution: with about 190 clones the paired CI half-width is about 0.08; smaller differences
    cannot be read.
- **Secondary, exploratory: SIS-seq.** Same features for the 65 fit clones (each clone's pooled
  sister profile treated as one cell: outside these models' training distribution), whole
  pipeline inside leave-one-out, against the E54 PCA score. The SIS-seq held-out shard is spent
  (E54) and is not used again.
- **Hygiene.** Weights only as safetensors from the named repos, into `data/fm_weights/` (own
  directory, no code from it executed); ortholog table into `data/orthologs/`; Python reading
  either runs with `-I`. CPU first; GitHub Actions if a run passes about 1 hour. Smoke tests in
  the scratchpad.

### E55 deviations before any fate statistic (2026-11-06)

- **scGPT dropped.** The tdc/scGPT safetensors repo has no gene vocabulary and needs its own model
  code (custom `ScGPTModel`); the official weights live outside Hugging Face. Candidates are the
  two Geneformer models: V1-10M (6 layers, 2,048 tokens) and V2-316M (18 layers, 4,096).
- **Implementation.** Geneformer's tokenisation and embedding were reimplemented from its code
  (read, not run): normalised counts / total x 1e4 / gene median, nonzero genes ranked, V2 adds
  <cls>/<eos>; embedding = mean of gene tokens in the penultimate layer. Dictionaries read with a
  whitelist unpickler. 14,792 of 25,289 mouse genes map one-to-one to Geneformer's vocabulary.
- **Sizes.** 571 clones (374 fit, 197 held-out), 868 day-2 cells; target sampling ceiling R^2 0.96.

### E55 external reference: published fate prediction on the same data (2026-11-06, added before results)

| source | unit and target | method | score |
|---|---|---|---|
| Weinreb 2020 Science, Fig 5 (PMC7608074) | each day-2 cell (early progenitors, CD34-thresholded) vs the N/(N+M) fraction of its clonal relatives | PBA, Waddington-OT, FateID (no clonal training) | Pearson R < 0.26 for all (FateID slightly best) |
| same | same | top 10 most fate-correlated genes | "easily outperformed" the algorithms (no number) |
| same | same | smoothed held-out clonal data (their upper bound) | R = 0.5 ("low overall because of noise in the fate outcomes of single cells") |
| Weinreb 2020, Fig 3 | clone's dominant fate from a day-2 profile, all conditions | logistic regression / MLP | 60% accuracy in vitro (late-time prediction 76%) |
| CoSpar preprint (Wang & Klein 2021 bioRxiv 2021.05.06.443026, Fig 4) | day-2 progenitor N vs M bias vs a *smoothed* ground-truth bias map built from all clonal data (Weinreb's method) | OT/state only before coherence+sparsity; CoSpar state only; CoSpar with day-6 clones only | R 0.51-0.54; **R 0.69**; R 0.68; upper bound (cross-validated ground truth) **R 0.72** |
| this project, track B | clone mean of day-2 cells vs clone N/(N+M), clones with >= 3 later N+M cells | PCA-50 + ridge, supervised, 5-fold CV | r 0.66 (R^2 0.43; ceiling 0.96) |

- **Not directly comparable.** Weinreb scores single cells against all their relatives (noisier
  target: their own ceiling is R 0.5); track B and E55 score clones with >= 3 later cells and
  average 1-2 day-2 cells, and train on clonal labels. The shared message: supervised use of
  day-2 expression beats unsupervised dynamic inference, and a large share of fate stays
  unexplained (Weinreb: "lower bound for the cell-autonomous influence of transcriptional state").
- **CoSpar's R is against a smoothed, state-diffused ground truth**, so state-only methods are
  partly scored against their own smoothing; its 0.69/0.72 are not comparable with raw clone
  fractions either. Numbers are the 2021 preprint's (published version may differ).
- No competition or shared leaderboard exists for this task (checked 2026-11-06); the Arc Virtual
  Cell Challenge 2026 scores perturbation responses, not clone fate.

### E55 fit-shard CV (2026-11-06, 374 clones, 5-fold nested ridge, written before the look)

| features | CV r | CV R^2 |
|---|---|---|
| PCA-50 | 0.64 | 0.35 |
| marker (Cebpa - Irf8) | 0.21 | 0.04 |
| Geneformer V1-10M | 0.44 | 0.19 |
| Geneformer V2-316M | 0.61 | 0.38 |

- **Best foundation candidate (named before the look): gf_V2-316M.** On the fit shard it ties
  PCA-50 (r 0.61 vs 0.64; R^2 0.38 vs 0.35). Target ceiling R^2 0.96. The look follows.

### E55 held-out look (2026-11-06, test larry_fm_heldout 0/1, taken once)

197 held-out clones (above the 150 minimum). Target ceiling R^2 0.96.

| features | held-out r |
|---|---|
| PCA-50 | 0.64 |
| marker (Cebpa - Irf8) | 0.29 |
| Geneformer V2-316M (named best) | 0.58 |
| PCA-50 + V2-316M | 0.60 |

- **delta = r(V2) - r(PCA) = -0.064 [-0.130, -0.0001]** (paired bootstrap, 5,000). CI is below 0:
  by the registered reading, **PCA better**. The upper edge touches 0 and |delta| is under the
  0.08 resolution, so the safe statement is "no gain; if anything worse".
- **Secondary:** concatenation adds nothing (delta -0.047 [-0.106, +0.011]).
- **Reading.** The ~55% of LARRY clone fate bias that day-2 RNA misses is not recovered by a
  pretrained model's zero-shot embedding. Replicates Kedzierska et al. 2025 (zero-shot
  foundation embeddings no better than HVG/PCA) on a clone-fate target. The fit-shard CV
  (0.61 vs 0.64) predicted the held-out ordering.

### E55 SIS-seq exploratory (2026-11-06; 65 fit clones, leave-one-out, nested ridge; held-out shard spent in E54)

| features | cDC1 r | cDC2 r | pDC r |
|---|---|---|---|
| PCA-3 (E54 recipe, inside LOO) | -0.49 | 0.11 | -0.01 |
| Geneformer V1-10M | 0.30 | 0.31 | 0.22 |
| Geneformer V2-316M | 0.37 | **0.74** | **0.95** |

- **Leak checks (all after the result, scratchpad scripts sischeck.py, sischeck2.py):**
  no duplicate profiles; label-shuffle LOO r -0.2 to -0.8 (V1 and V2); within-batch shuffle
  r -0.10, 0.11, 0.28; fate logit vs library size r 0.10, vs genes detected 0.14. Target
  reliability (split-half sister wells): pDC rho 0.89 -> max r 0.97; cDC2 0.75 -> 0.93; cDC1
  0.65 -> 0.89. V2 pDC prediction vs each well separately: 0.89, 0.94 (sister-well rho 0.89).
- **WITHDRAWN (B43 amendment): leak from batch centring incl. the left-out clone; clean V2 = 0.34 / 0.53 / 0.77.**
- **Reading: exploratory hypothesis, not a discovery.** V2 embeddings of the pooled sister
  profile predict pDC and cDC2 fate almost to the target's reliability on the fit clones. No leak
  found, but: n = 65; one look at two models after E54; the PCA-3 baseline here (-0.01 for pDC)
  does not reproduce E54's fit-set LOO (R^2 0.38) because the preprocessing differs (global vs
  per-batch scaling), so "V2 beats PCA" is not established; the only sealed SIS-seq clones were
  spent in E54. Opposite of the LARRY result (#19), where V2 did not beat PCA on 197 held-out
  clones. Confirmation needs clones the choice has not seen.

## B43: what carries the SIS-seq V2 signal? (pre-registered 2026-11-06)

- **Go-ahead.** Bence, "go all 3" (probe the SIS signal; then independent data search; then
  chapter 3).
- **Data.** The 65 SIS-seq fit clones of E55 only (held-out shard spent in E54). Descriptive
  diagnostics; no sealed data, no new claim beyond "hypothesis" without unseen clones.
- **Steps.**
  1. *Baseline reproduced:* E54's own recipe (`core_sisseq.scores("fit")`, per-batch z, PCA
     chosen on all fit clones, LOO only on the regression) and the same with PCA refitted inside
     LOO. Reports r per fate.
  2. *Is the signal in plain RNA?* Gene-selection ridge with selection inside LOO: per fold, top
     k genes (k = 20, 50, 200) by |corr| with the fold's training logits, ridge with nested
     lambda. If pDC r >= 0.8, the signal is in the RNA and V2 is one way of reading it; if
     <= 0.4, V2 adds something plain genes do not.
  3. *Attribution:* correlate the V2 LOO pDC and cDC2 predictions with each gene's batch-centred
     log expression; list the top 30 genes; mark known terminal pDC genes (Siglech, Bst2, Ly6d,
     Ccr9, Cox6a2, Klk1, Ly6c2, Tcf4, Spib, Runx2) and cDC2 genes (Sirpa, Itgam, Irf4, Klf4,
     Esam, Clec4a4). **Reading:** >= 3 terminal pDC genes in the top 30 for pDC = the sisters
     already carry pDC-lineage transcripts at day 2.5 (early commitment visible in RNA), which
     revises #17 from "weak trace" to "trace hidden by the feature choice".
- **Minimum sample.** 65 clones: r resolution about +-0.25 (95%); differences under that are not
  read.

### B43 amendment (2026-11-06, after step 1-3 output, before any reading is logged): leak found

- Shuffle control for step 2 (top-200 genes, within-batch label shuffles): r 0.19-0.72, not
  about 0. **Leak:** features and fate logits were batch-centred over all 65 clones including the
  left-out one, so within a batch both sum to zero and a well-fitting model predicts the
  left-out clone from its batch-mates (batches of 12-22). The same centring was used in the E55
  SIS-seq run (V2 0.95), so that result is presumed inflated.
- **Fix:** batch means (of features and of raw logits) computed from training clones only inside
  each LOO fold; the left-out clone is centred with them. Rerun: PCA-3 (per-batch z inside LOO),
  genes top-50/200, V1, V2; within-batch shuffle controls (3 each) for genes top-200 and V2.
  Readings as registered in B43 apply to the fixed pipeline only. E54's held-out look is not
  affected (its held-out clones were centred with fit-set statistics).

### B43 result (2026-11-06, clean pipeline; 65 SIS-seq fit clones, LOO with batch centring inside folds)

| features | cDC1 r | cDC2 r | pDC r |
|---|---|---|---|
| PCA-3 (top-variance genes) | -0.21 | 0.20 | 0.13 |
| genes top-50 (selected inside LOO) | 0.02 | 0.56 | 0.75 |
| genes top-200 | 0.15 | 0.57 | 0.78 |
| Geneformer V1-10M | 0.15 | 0.23 | 0.18 |
| Geneformer V2-316M | 0.34 | 0.53 | 0.77 |

- Shuffle controls now behave: genes top-200 r -0.48 to +0.38, V2 -0.41 to +0.16 (3 within-batch
  shuffles each; spread about +-0.25, matching the registered +-0.25 resolution).
- **E55 SIS-seq numbers withdrawn:** V2 0.95 / 0.74 were leak-inflated; clean 0.77 / 0.53.
  **V2 adds nothing over plain selected genes** (differences far under the resolution), the same
  answer as on LARRY (#19).
- **Step 2 reading:** pDC 0.78 is between the registered branches (>= 0.8 "in the RNA"; <= 0.4
  "V2 adds something"); read as: a fate signal is in plain RNA for pDC and cDC2, not cDC1, and
  plain genes read it as well as V2.
- **Step 3 reading (attribution on the first, leaky predictions; the gene axis, not the size,
  is what is read):** 0 of 10 terminal pDC genes in the top 30, so the registered "early pDC
  commitment" reading is not met. The genes are a lymphoid-primed vs myeloid-primed progenitor
  axis: toward pDC Dntt, Satb1, Notch1, Lrmp, Gimap9, Arpp21, Hmga2; toward cDC2 Ccr2, Ms4a6c,
  Csf1r, Mpo, Prtn3, Ctsg. Same axis as B41.
- **For #17:** the weak E54 held-out link used top-variance PCA features, which are weak here too
  (pDC 0.13). With genes selected by fate, the fit clones show pDC ~0.77 cleanly; unconfirmed on
  sealed clones (spent). No public independent data set exists (NEXT_DATASETS check 4).

## G1: is a gastruloid clone's A-P propensity a stable clone state, and do mixed clones sort by rank or into fixed domains? (pre-registered 2026-11-06)

- **Go-ahead.** Bence, "yes go with recomm" (register chapter 3's first test; plain features, no
  foundation model).
- **Data.** Raj lab (Ayyappan et al. 2025 bioRxiv 2025.07.12.664536), Dropbox `extracted_data/
  linescans`, downloaded 2026-11-06 to `data/gastruloid_raj/linescans/`. Profiles are 30
  positions along the A-P axis (x = 0-100), intensity z-normalised per gastruloid. Prior
  exposure: file headers, row counts and the per-clone gastruloid counts only; no profile
  statistic computed.
- **G1a: propensity as a clone state (sealed).** `normalized_smoothed_individual_lines.csv`: one
  labelled clone (clones 1, 3, 5, 6, 7) in a bulk chimera, 230 gastruloids in 11 clone x
  replicate cells. Per gastruloid polarity P = mean intensity at x < 50 minus x >= 50 (positive
  = enriched toward x = 0).
  - Sealing: `shard_of("g1:<gastruloid_id>", 3) == 0` held out (about 77 gastruloids), read once
    via the oracle (test `gastruloid_polarity_heldout`).
  - Fit-shard model: P = mu_clone + u_(clone, replicate) + e; method-of-moments variance shares
    (clone, replicate within clone, residual).
  - Predictions (logged before the look): for each clone x replicate cell, the held-out mean
    95% interval mean_fit +- 1.96 sqrt(s_e^2 / n_fit + s_e^2 / n_ho).
  - **Readings.** Clone state: clone share of variance > replicate share and its CI excludes 0.
    Replication: >= 90% of cells with >= 4 held-out gastruloids inside their interval (fewer
    cells than 6 qualifying: descriptive). Minimum 4 held-out gastruloids per cell for its
    reading.
- **G1b: rank sorting vs fixed domains (open data, readings set before computing).** Clone 1 :
  clone 3 mixes at 4:1, 2:1, 1:1, 1:2, 1:4 (`ratios_*_clone1clone3_*.csv`, mean and 95% band per
  position, two dates). Assumption to verify from `extraction_scripts` before computing: red =
  clone 3 (anterior), green = clone 1; f = anterior-clone (clone 3) fraction.
  - Statistic: crossing point x* where mean_red = mean_green (linear interpolation; the first
    crossing from x = 0); uncertainty by drawing curves within their bands (normal, SE =
    half-band / 1.96, 2,000 draws, independent per position: conservative).
  - Models: **rank sorting** (cells sort by state; the anterior clone fills [0, f]; z-normalised
    step profiles cross at x* = 100 f) predicts slope dx*/d(100 f) = 1; **fixed domains**
    (position set by the axis, not by composition) predicts slope 0.
  - **Readings.** Slope CI includes 1 and excludes 0: rank sorting. Includes 0 and excludes 1:
    fixed domains. Both or neither: unresolved; between 0 and 1 excluding both: partial
    proportion regulation. Minimum: >= 4 ratios with a crossing.
- **Next (not this entry).** Clone-memory decay over passages (`clone*_passage_rep*.csv`,
  directionality summary) as a core-model state-decay test.

### G1 amendment (2026-11-06, before any profile statistic)

- **Channels verified** (paper text): "clone 1 (green, posterior propensity) and clone 3 (red,
  anterior propensity)". Authors' notebook `Clone_Ratios.ipynb` (read as text, not run): red =
  Channel 2, green = Channel 0; x = 0-100 "% distance along A-P axis".
- **Bands are +-1 SEM** of the raw mean (then z-scored with the mean profile's mean and SD), not
  95%: draws use SE = half-band. Each ratio file's curve is the z-scored mean of its gastruloids.
- **Batch:** files come from two aggregation dates (04/29/2024: 2:1, 1:2; 11/04/2024: 4:1, 2:1,
  1:1, 1:2, 1:4). The slope is fitted with a date intercept (common slope). An undated
  `ratios_2_1_clone1clone3.csv` is excluded (duplicate of a dated file or unknown batch).
- **Prior disclosed:** the paper states the crossing "was not sensitive to the ratio" with
  batch-to-batch variation (Supp Fig 5); G1b quantifies that claim with a CI against the rank-
  sorting prediction (slope 1).
- **G1a:** `normalized_smoothed_individual_lines.csv` profiles are normalised per gastruloid
  (authors' `normalize_series`); x = 0 is the anterior end in the ratio files; the polarity sign
  is read relative to position index only.

### G1a fit shard and held-out predictions (2026-11-06, written before the look)

- 177 fit gastruloids, 53 held-out (23%, hash chance; registered "about 77" was 1/3 of 230).
- Fit-shard clone means of polarity P (positive = toward x = 0, anterior): clone 1 -0.92, clone 6
  -1.13 (posterior); clone 3 +0.88, clone 5 +1.05, clone 7 +0.73 (anterior). Matches the paper's
  3 anterior / 2 posterior.
- Variance shares: **clone 0.56 [0.08, 0.80]**, replicate 0.07 [0, 0.25], residual 0.37
  (parametric bootstrap, 2,000). Fit-set reading: clone state (clone > replicate, CI excludes 0).
- Held-out intervals per clone x replicate in `results_gastruloid/g1a_fit.json`; 7 cells will
  have >= 4 held-out gastruloids (reading applies: >= 6 qualifying).

### G1a held-out look (2026-11-06, test gastruloid_polarity_heldout 0/1, taken once)

- 53 held-out gastruloids. **7 of 7 qualifying cells (>= 4 gastruloids) inside their intervals**
  (11 of 11 overall). Held-out variance shares: clone 0.69, replicate 0.04, residual 0.26.
- **Reading: replicated.** A clone's A-P propensity is a stable clone property: the clone sets
  most of the variance in polarity, replicate batches add little, and fit-shard means predict
  held-out gastruloids.

### G1b result (2026-11-06; 7 ratio curves, two dates, 2,000 draws)

| clone1:clone3 | f anterior | date | crossing x* (rank sorting predicts 100 f) |
|---|---|---|---|
| 4:1 | 0.20 | 11/04 | 35.5 |
| 2:1 | 0.33 | 11/04 | 34.6 |
| 2:1 | 0.33 | 04/29 | 38.9 |
| 1:1 | 0.50 | 11/04 | 35.6 |
| 1:2 | 0.67 | 11/04 | 39.3 |
| 1:2 | 0.67 | 04/29 | 53.7 |
| 1:4 | 0.80 | 11/04 | 40.7 |

- **Slope 0.17 [-0.40, 0.42]** (date intercepts). CI includes 0, excludes 1: **fixed domains**
  (registered reading). Rank sorting by clone state is rejected; agrees with the paper's claim
  (crossing insensitive to ratio).
- Caveat on the statistic: the "first crossing" in independent per-position draws sometimes
  falls at the noisy anterior edge, so per-ratio CIs reach x = 0 (and the 1:2 04/29 observed
  value lies outside its draw CI). This widens the slope CI (conservative); it still excludes 1.
- **Implication for the core model (chapter 3):** clone state sets *which end* a clone prefers
  (G1a), but domain size is set by the gastruloid, not by starting composition. The local rule
  cannot be "sort by state"; it needs a feedback that sizes domains (growth compensation, as the
  paper's competition data suggest, or fate switching at the boundary).

## G2: do mixed clones elongate by division of labour (anterior + posterior) or by any mixing? (pre-registered 2026-11-06)

- **Go-ahead.** Bence, "cool lets go next steps" (G2 next; chapter 3's second goal: the
  pure-vs-mixed elongation difference).
- **Data.** Raj lab Dropbox `raw_data/Morphology CSVs/Morphologies_1_2_3_4_5_6_7.csv` (171 rows:
  composition, replicate, date, total and elongated gastruloids): bulk, 7 pure clones, clone +
  bulk at 25/50%, all 21 clone pairs at 50:50, one triple. Downloaded 2026-11-06 to
  `data/gastruloid_raj/morph_csv/`.
- **Prior exposure, disclosed:** while checking the format I printed the pooled elongated
  fraction per composition (all rows). No model was fitted. The `morph_08312026` folder holds
  annotations of new subclones (5s2, 6a), not these compositions, so no sealed confirmation set
  exists: G2 is an open-data test with readings fixed here.
- **Propensity labels (from G1a, before G2):** anterior 3, 5, 7; posterior 1, 6; clones 2 and 4
  unlabelled (excluded from G1 by the authors). Complementary pairs: 1+3, 1+5, 1+7, 6+3, 6+5,
  6+7 (6). Same-propensity pairs: 1+6, 3+5, 3+7, 5+7 (4). Pairs with 2 or 4: own indicator,
  not read.
- **Model.** Binomial per row, logit p = sum_i f_i e_i (+ terms), e_i one elongation competence
  per component (clones 1-7, bulk), fitted jointly with:
  - M0 additive: no extra term.
  - M3: + delta_comp [complementary pair] + delta_same [same-propensity pair] + delta_unl
    [pair with 2 or 4] (+ the triple gets delta_unl).
  CIs by cluster bootstrap over dates (2,000).
- **Readings (M3).**
  - **Division of labour:** delta_comp - delta_same CI above 0.
  - **Generic mixing rescue:** delta_comp and delta_same both CI above 0 and their difference CI
    includes 0.
  - **Additive (no mixing effect):** both CIs include 0.
  - Other patterns: described, no claim.
  - Secondary: does M0 already explain clone + bulk chimeras (25/50%)? Observed vs M0 predicted
    per composition, pooled.
- **Minimum.** >= 40 gastruloids in each pair class; otherwise descriptive.
- **Model link.** A division-of-labour reading means elongation needs both an anterior and a
  posterior state population (a tissue-level requirement the simulator must encode); a generic
  rescue means diversity itself helps (e.g. clone-specific defects complemented by any partner).

### G2 result (2026-11-06; 171 rows, 26 dates, 3,597 gastruloids; cluster bootstrap over dates, 2,000)

- Data fix before the fit: one row (Index 80, 100% clone 6) has an empty Elongated count and is
  dropped (the first run returned NaN because of it).
- Gastruloids per class: complementary pairs 487, same-propensity 318, pairs with clone 2/4 or
  the triple 578 (all >= 40).
- M3 (logits, added to the additive competence model): **delta_comp +1.47 [0.96, 1.92]**,
  **delta_same +1.01 [0.42, 1.74]**, delta_unl +0.93 [0.26, 1.93]; **comp - same +0.46 [-0.18,
  1.02]**. NLL 1996 (M0) -> 1923 (M3).
- **Reading: generic mixing rescue** (registered). Any 50:50 mix of two different clones
  elongates more than the clones' own competences predict, about +1 logit, whether the two
  clones prefer opposite ends or the same end. Complementary pairs trend higher (+0.46) but the
  difference is not resolved; division of labour is not shown.
- Pairs observed vs M0: 1+3 0.80 vs 0.60; 1+6 (both posterior) 0.76 vs 0.66; 6+7 0.93 vs 0.78;
  3+7 (both anterior) 0.71 vs 0.73.
- Secondary: clone + bulk chimeras sit near M0 (bulk is itself a mixture), largest misses 50%
  clone 4 (0.79 vs 0.51) and 25% clone 1 (0.58 vs 0.77).
- **Model link:** elongation needs heterogeneity, not specifically an anterior + a posterior
  population. In the simulator, a pure clone should be treated as a deficient tissue that any
  second state population can complement; the A-P division-of-labour mechanism (paper's
  framing) is at most part of it.
- Literature check (paper text, 2026-11-06): the authors report elongation in most pairs
  "regardless of whether" both anterior, both posterior or opposite; G2 agrees and adds the
  effect sizes. Discovery #22: consistent.

## G3: does a gastruloid clone's A-P propensity fade over passages? (pre-registered 2026-11-06)

- **Go-ahead.** Bence, "go 1" (the memory-fade test), before the division-of-labour literature
  sweep and the model.
- **Data.** Raj Dropbox `raw_data/linescans`, per-gastruloid raw scans of clone 1 + clone 3 50:50
  chimeras ("EX") aggregated at passages 1-4 (04/22, 04/24, 04/27, 04/29/2024): 13, 16, 32, 31
  gastruloids. Mapping from the authors' `Clone_Passaging.ipynb` (read, not run): green Channel
  0 = clone 1 (posterior), red Channel 2 = clone 3 (anterior); line tagged "Phase line", values
  ordered from the line's start (taken as anterior, as in the ratio curves).
- **Prior exposure, disclosed:** the authors' summary (`fluorescence_directionality_summary.csv`):
  red anterior in 100 / 81 / 75 / 65% of gastruloids at passages 1-4, green posterior 100 / 63 /
  38 / 58%. Structure and counts of the raw files only; no profile statistic computed.
- **Statistic per gastruloid.** Each channel resampled to 30 points; relative enrichment E_c =
  (mean first half - mean second half) / mean whole line. Separation S = E_red - E_green
  (positive = clone 3 anterior of clone 1; amplitude kept, unlike the z-scored curves).
- **Readings.**
  - **Fades:** OLS slope of S on passage number with CI below 0 (bootstrap over gastruloids within
    passage, 2,000) and S(4)/S(1) CI below 1.
  - **Stable:** ratio CI includes 1.
  - Rate: S(t) = S1 r^(t-1) fitted by least squares on passage means; retention r per passage and
    half-life in passages with bootstrap CI. Derived per-division memory rho = r^(1/g) for g = 3,
    4.5, 6 divisions per passage (mES doubling about 12-16 h, passages 2-3 days; an assumption,
    reported as a range).
  - Check: share of gastruloids with E_red > E_green per passage, against the authors' summary.
- **Caveat registered:** one aggregation per passage, so passage and batch are confounded. G1a
  measured batch (replicate) variance at 0.04-0.07 of polarity variance; a fade far beyond that
  is read as the clone's, not the batch's. Minimum 13 gastruloids per passage (met).

### G3 result (2026-11-06; 92 gastruloids, passages 1-4)

| passage | n | mean S | share clone 3 anterior of clone 1 |
|---|---|---|---|
| 1 | 13 | 0.221 | 1.00 |
| 2 | 16 | 0.178 | 0.94 |
| 3 | 32 | 0.048 | 0.69 |
| 4 | 31 | 0.054 | 0.74 |

- Parsing check passed: per-channel shares reproduce the authors' summary exactly (red anterior
  1.00 / 0.81 / 0.75 / 0.65; green posterior 1.00 / 0.63 / 0.38 / 0.58).
- **Slope -0.060 [-0.076, -0.043] per passage; S(4)/S(1) 0.25 [0.12, 0.39]. Reading: fades.**
- Retention r 0.60 [0.50, 0.68] per passage, half-life 1.4 [1.0, 1.8] passages. Per-division
  memory rho 0.84 [0.79, 0.88] (g = 3), 0.89 [0.86, 0.92] (g = 4.5), 0.92 [0.89, 0.94] (g = 6).
- Shape: the drop is between passages 2 and 3, and passages 3-4 are level (0.048, 0.054), still
  above 0 (69-74% ordered): a residual propensity, not decay to zero. The exponential is a
  summary, not a tested shape.
- Batch confounding: the passage 2 -> 3 drop (0.13) is 1-1.6 within-passage SDs and far above
  G1a's replicate share; read as the clones', with one aggregation per passage as a stated limit.
- **Model link.** The gastruloid clone state is much less stable than LARRY's fate state (#6:
  near-full memory over 2 days in vitro): in pluripotency culture it loses about 40% per passage,
  then holds a residue. For the simulator: clone state needs a memory parameter per context
  (pluripotent drift vs committed progenitor), and the passage at aggregation matters.

## G2b: does "any mixing rescues" (#22) survive stricter nulls? (pre-registered 2026-11-06)

- **Go-ahead.** Bence: "retest this if we build hypos on it". #22 rests on one null (the joint
  additive-logit model fitted to all rows, pairs included). Same data as G2.
- **Why it could be an artefact.** (a) The additive-logit null may be too harsh: averaging two
  clones on the logit scale predicts less than averaging probabilities when clones differ a lot.
  (b) Pure clones and pairs were often scored on different dates; date effects could pose as
  rescue. (c) One clone (e.g. 5, elongating 94% alone; or 4, under 10%) could drive the class
  means.
- **Tests (50:50 two-clone pairs; complementary and same-propensity classes as in G2, plus all
  pairs).**
  1. **Pure-only nulls.** Competences from pure-clone rows only (pooled, Wilson or logit), then
     pair excess over three nulls: N1 mean logit, N2 mean probability, N3 **best partner**
     (max of the two pure probabilities: the strictest; beating it means the mix does better
     than its better member).
  2. **Date-matched.** Pair rows whose two pure clones were scored on the same date (25 of 62
     rows): within-date excess over N2 and N3.
  3. **Leave-one-clone-out.** Repeat test 1 (N2) dropping every pair that contains clone k,
     k = 1..7.
  CIs: cluster bootstrap over dates (2,000), excess = pooled pair rate minus pooled null, per
  class.
- **Readings.**
  - **#22 stands** if, for both complementary and same-propensity pairs, the excess over N2 has
    CI above 0 in test 1 and in the date-matched test 2, and stays above 0 in every
    leave-one-clone-out run.
  - **Strong form** ("the mix beats its better member"): excess over N3 CI above 0 (all pairs).
  - **#22 weakened** if the N2 excess loses CI > 0 in test 2 or in any leave-one-out run (rescue
    is then date- or clone-driven); **withdrawn** if test 1 N2 excess CI includes 0.
  - Complementary vs same difference re-read under N2 (G2's +0.46 was unresolved).
- **Minimum.** Date-matched test needs >= 10 rows and >= 150 gastruloids per class read; else
  pooled over classes only.

### G2b result (2026-11-06; pure-clone rates: 1 0.43, 2 0.57, 3 0.38, 4 0.05, 5 0.93, 6 0.57, 7 0.75)

Excess = pooled pair elongation minus pooled null (probability scale); cluster bootstrap over
dates.

| test | complementary | same-propensity | all pairs |
|---|---|---|---|
| 1 vs N1 (mean logit) | +0.29 [0.17, 0.39] | +0.20 [0.07, 0.34] | +0.24 [0.16, 0.34] |
| 1 vs N2 (mean probability) | **+0.30 [0.18, 0.41]** | **+0.21 [0.09, 0.35]** | +0.24 [0.16, 0.34] |
| 1 vs N3 (better member) | +0.20 [0.08, 0.32] | +0.10 [-0.04, 0.26] | +0.10 [-0.01, 0.21] |
| 2 date-matched vs N2 | +0.30 [0.15, 0.42] (10 rows, 217) | +0.24 [0.13, 0.32] (4 rows, 85: under minimum) | **+0.19 [0.10, 0.31]** (23 rows, 517) |
| 2 date-matched vs N3 | +0.18 [-0.01, 0.35] | +0.13 [0.06, 0.18] (under minimum) | +0.00 [-0.12, 0.20] |

- Leave-one-clone-out (N2): every run keeps comp, same and all above 0 (lowest lower bound:
  same, dropping clone 6, +0.04).
- **comp - same under N2: +0.09 [0.001, 0.17]** (1,978 valid bootstrap draws of 2,000; draws
  missing a class dropped). Under N3: +0.10 [-0.02, 0.22].
- Bug fixed before reading: the first run printed a NaN CI for comp - same (np.percentile over
  draws with missing classes); nanpercentile used.
- **Reading.** **#22 stands**: both classes exceed the probability-average null, the pooled
  date-matched test agrees, and no single clone drives it (the same-propensity date-matched test
  is under its minimum and is read only pooled). **Strong form not shown** for all pairs: mixes
  do not reliably beat their better member (+0.10 [-0.01, 0.21]).
- **New, on the edge:** complementary pairs rescue more than same-propensity pairs (+0.09
  [0.001, 0.17]); and only complementary pairs beat their better member (+0.20 [0.08, 0.32],
  exploratory). So: a generic mixing effect plus a smaller complementarity bonus. G2's logit
  difference (+0.46 [-0.18, 1.02]) points the same way.
- Exception: clone 4 (5% alone) is only partly rescued (2+4 0.48, 4+6 0.28; 4+5 0.73, 4+7 0.69);
  a severe defect is not fully buffered.

## G4: what mechanism explains mixing rescue: buffering, complementation or a T+ window? (pre-registered 2026-11-06)

- **Go-ahead.** Bence, "go 1 2" (G4 tests H1-H3 from LITERATURE_SWEEP_2026-11e; then the model).
- **Data.** Morphology table (G2; all compositions); `T_overlap - Sheet1.csv` (T+ area per
  gastruloid, "overall" and per clone; pure clones 1, 3, 5, 6, 7 n = 3-17, bulk 19, 9 pairs).
  Prior exposure: G2/G2b results; T_overlap structure and counts only.
- **Models of P(elongated) per row** (components: bulk, clones 1-7; f_i = starting fractions):
  - **M0 additive:** logit p = sum f_i e_i (8 parameters).
  - **Mgen (descriptive baseline, = G2 M3):** M0 + delta_comp, delta_same, delta_unl for clone
    pairs (11).
  - **H1 buffering (window on an averaged latent state):** each component has z_i in R^d; the
    aggregate state is Z = sum f_i z_i; logit p = a - |Z|^2 (optimum at 0, unit curvature;
    scale absorbed in z). d = 1 (9 parameters) and d = 2 (17). A mix of two clones on opposite
    sides of the optimum lands nearer it: rescue; same side: none.
  - **H2 complementation (presence OR, needs AND):** each component has capacities c_ik in (0, 1),
    K = 2; the aggregate has C_k = max over components with f_i >= 0.2 of c_ik; logit p = a + b
    sum_k log C_k (18). Strict implication: a mix is never worse than its better member.
  - Fitting: maximum likelihood, 10 random restarts for H1/H2, small ridge (1e-3).
- **Primary test.** Leave-one-composition-out CV over every non-pure, non-bulk composition (pure
  clones and bulk always in training); score = summed held-out binomial log-likelihood.
  - **Reading:** the best model wins if its CV log-likelihood exceeds every other model's by
    >= 2; models within 2 of the best are "not distinguished". H1 and H2 are read against Mgen
    (does a mechanism predict unseen mixtures as well as the descriptive bonus model?).
- **H2 diagnostic.** Pairs whose elongation is below their better pure member (one-sided
  binomial p < 0.05 against the better member's pooled rate): >= 2 such pairs contradict strict
  complementation.
- **H1 emergent check (descriptive).** Fitted d = 1 positions of clones 1, 3, 5, 6, 7 vs G1a
  polarity (Spearman; 5 points). |rho| >= 0.9: the buffering axis is the A-P propensity axis,
  which would explain the complementarity bonus without a separate term.
- **H3 T+ window (descriptive; n too small to test).** (a) pure-clone mean T+ "overall" vs pure
  elongation and vs bulk; (b) pairs: observed T+ vs the mean of their pure clones and vs bulk
  (is T+ pulled toward bulk by mixing?).

### G4 result (2026-11-06; 171 rows; leave-one-composition-out CV over 61 non-pure, non-bulk compositions)

| model | parameters | CV log-lik (held-out) | full-fit log-lik |
|---|---|---|---|
| **Mgen (descriptive bonuses)** | 11 | **-1248.6** | -1923.1 |
| M0 additive | 8 | -1282.8 | -1996.4 |
| H1 buffering, d = 1 | 9 | -1299.4 | -1958.3 |
| H2 complementation, K = 2 | 18 | -1343.5 | -1918.3 |
| H1 buffering, d = 2 | 17 | -1370.4 | -1914.8 |

- **Reading: Mgen wins** (by 34 over M0, 51 over H1 d = 1). Neither mechanism, as formulated,
  predicts unseen mixtures as well as the additive model: H1 and H2 fit the training rows well
  (d = 2 and H2 near Mgen in full fit) but generalise worse (overfit or wrong form). The rescue
  is real (#22) but its mechanism is **not identified** by these data.
- **H2 strict complementation contradicted:** 4+5 (0.73 vs better member 0.94, p 4e-5) and 4+6
  (0.27 vs 0.57, p 2e-5) are below their better member.
- **H1 axis vs G1a polarity:** Spearman -0.6 (5 clones; under the 0.9 bar): the fitted buffering
  axis is not the A-P propensity axis.
- **H3 descriptive.** Pure-clone T+ area: posterior clones high (1: 0.38, 6: 0.44), anterior low
  (3: 0.14, 5: 0.14, 7: 0.10), bulk 0.19. T+ does not track pure elongation (clones 3 and 5:
  same T+, 0.38 vs 0.94 elongated). In pairs T+ moves toward bulk in 6 of 9 (e.g. 1+7 0.12 vs
  pure mean 0.24; 6+7 0.14 vs 0.27; 1+6 0.30 vs 0.41); 1+3 rises (0.36 vs 0.26). n per pair
  2-9 gastruloids: descriptive. Consistent with the proportion regulation of #21 in a second
  readout.
- **For G5 (model):** use the descriptive structure (clone competence + generic mixing bonus +
  complementarity bonus) as a phenomenological rule; the mechanism stays open (literature
  candidates in the sweep; none supported here).

## G5: a gastruloid simulator from clone states plus local rules, and is clone memory lost per division or per passage? (pre-registered 2026-11-06)

- **Go-ahead.** Bence, "go 1 2" then "put 24 as a possible next chapter ... then": build the
  gastruloid model with a descriptive rescue rule (mechanism deferred to candidate chapter 4).
- **Simulator (cell-based, 1D axis; `g5_gastruloid.py`).** Each cell carries an A-P state s.
  Aggregate ~300 cells by composition; growth with **proportion feedback**: final anterior-
  propensity fraction F = (1 - g) f_A + g a* (G1b); cells **sort by rank** of s + sorting noise
  (G1a); clone state means from G1a; memory of s from G3; elongation by the **descriptive rescue
  rule** (G2 Mgen: clone competences + generic and complementary mixing bonuses). Calibration to
  G1a means, G1b crossings, G3 and G2 is a consistency check, not a test (fitted).
- **Out-of-sample test: memory per division vs per passage.** Data: `all_normalized_scans.csv`
  (authors' per-gastruloid z-normalised scans of subclones in chimeras with bulk): head_p1 (3
  subclones, 32 scans), head_p2 (4, 25), tail_p1 (3, 29), tail_p2 (24). Mapping (paper Fig 3C
  text + naming): head = subclones of anterior clone 3, tail = of posterior clone 1; p1/p2 =
  isolated at passage 1 / 2. Polarity P per scan as in G1a; sign-adjusted toward the parent
  (head +, tail -). Not yet computed; the paper's text says p1 subclones "retained the spatial
  propensity of their parent line" and p2 subclones "lacked clear spatial propensities"
  (disclosed prior).
  - **Per-division memory** (G3's retention spread over divisions, rho 0.84-0.92 per division):
    expanding one cell to an aggregation stock (>= 13 divisions, >= 10^4 cells) leaves <= 0.92^13
    = 0.34 of the state: predicts p1 subclone polarity / parent polarity (G1a: clone 1 -0.92,
    clone 3 +0.88) **<= 0.34**.
  - **Per-passage-event memory** (loss at passaging, not at division): predicts the p1 ratio
    **about 1** and the p2 / p1 ratio about G3's r = 0.60 [0.50, 0.68].
  - **Readings.** (1) p1 / parent ratio CI (bootstrap over scans within subclone, 2,000) excludes
    0.34 and includes 1: memory is lost per passage event (per-division rejected); CI includes
    0.34 and excludes 1: per division. (2) p2 / p1 ratio vs [0.50, 0.68]: inside = consistent
    with G3's per-passage rate; below = loss at the subcloning bottleneck exceeds the population
    rate (cells within a lineage diversify). Minimum 20 scans per condition (met).
- The memory rule that wins (1) is the one the simulator uses.

### G5 result (2026-11-06)

**Out-of-sample: memory per division vs per passage** (subclones in chimeras with bulk;
sign-adjusted polarity; scans smoothed as the parent lines, Savitzky-Golay 5/2, as in the authors'
`Clone_Propensity_Passaging.ipynb`; bootstrap over scans within subclone, 2,000).

| | clone 3 subclones (head) | clone 1 subclones (tail) | both |
|---|---|---|---|
| p1 subclone / parent (G1a) | 1.37 [1.03, 1.97] | 1.07 [0.81, 1.42] | **1.21 [1.005, 1.50]** |
| p2 / p1 subclone | 0.61 [0.42, 0.80] | 0.09 [-0.12, 0.31] | **0.37 [0.23, 0.51]** |

- Scans: 93 / 60 / 84 / 85 (head p1, head p2, tail p1, tail p2), 3-4 subclones each.
- **Reading 1: per-division memory rejected** (CI far above 0.34). Formally "other": the CI sits
  just above 1 (registered "per passage" needed it to include 1): single-cell subclones at
  passage 1 are at least as polarised as their parents (a bottleneck may pick strongly biased
  cells). Expansion from one cell (>= 13 divisions) does not erase the state; one passage does.
- **Reading 2:** p2 / p1 0.37 [0.23, 0.51] touches G3's [0.50, 0.68]: consistent (registered
  rule), but clone-specific: clone 3 keeps 0.61, clone 1 loses almost all (0.09).
- **Simulator rule chosen:** memory lost per passage event, retention 0.60 per passage (G3),
  clone-specific rates noted.

**Simulator (`g5_gastruloid.py`, calibrated; consistency checks, not tests).**
- State means (unit within-clone SD, sorting noise 1): clone 1 -0.66, 3 +0.59, 5 +0.83, 6 -0.91,
  7 +0.36 reproduce G1a polarities (-0.97 / 0.92 / 1.15 / -1.19 / 0.62 vs targets -0.98 / 0.90 /
  1.13 / -1.18 / 0.61).
- Proportion feedback g = 0.83, a* = 0.38 (from G1b): simulated 1+3 crossing at 30-36 for clone 1
  shares 0.2-0.8 (observed 35-54).
- Memory: clone 1 polarity in bulk -0.96 / -0.63 / -0.43 / -0.29 over passages 1-4.
- Elongation (descriptive rule): pure clones 0.11-0.91; pairs 1+3 0.78 (obs 0.80), 1+6 0.74
  (0.76), 5+6 0.94 (0.94); misses 3+7 0.81 (0.71) and 4+6 0.50 (0.27), the rule's known limits.
- **Chapter 3 goal status:** the model reproduces clone spatial propensities and the
  pure-vs-mixed elongation difference from clone states plus local rules (sorting, proportion
  feedback, per-passage memory), with elongation by a descriptive rescue rule. The rescue
  mechanism is candidate chapter 4.

### Chapter 3 audit (2026-11-06, Bence: "did you use the holdout? testing can only go through oracle")

- Only **G1a** went through the oracle (ledger: gastruloid_polarity_heldout 0/1). G1b, G2, G2b,
  G3, G4 and G5 were pre-registered readings on **open data with no sealed set**: fit-set-level
  results. Discoveries #21-#25 relabelled "unconfirmed" (they had been "moderate").
- **Rule breach:** `g5_memory.py` (parent polarity) and `g5_gastruloid.py calibrate` (target
  means) called `g1_gastruloid.polarity()`, which returned all G1a gastruloids, the 53 sealed
  ones included, outside the oracle. The G1a look was already spent, so no test result is
  contaminated, but the rule was broken. **Fix:** `polarity()` now drops sealed gastruloids by
  default; only `fit` (counts, for interval widths) and `heldout_stat` (oracle) pass
  `include_heldout=True`. G5 numbers stay as logged (computed before the fix); reruns will
  differ slightly.
- **Remedy:** seal the authors' untouched per-gastruloid raw files as confirmation sets and test
  #20, #23 and #25 through the oracle (next entry). #21 and #22 have no untouched data of the
  same design in this source.

## G6: sealed confirmation of chapter-3 memory and propensity (pre-registered 2026-11-06)

- **Go-ahead.** Bence, after the audit: "if we have backup then its good" (seal untouched files,
  test through the oracle).
- **Sealed sets (whole files, read only by the oracle; structure checked: tags, channels,
  counts; no intensities read).** Raj Dropbox `raw_data/linescans`, in
  `data/gastruloid_raj/raw_ls2/`:
  - **S-P4 (clone + bulk, passage 4):** `clone1+bulk_3+bulk_5+bulk_pass4_12042024.xlsx` (tags
    1+WT 29, 3+WT 25, 5+WT 24) and `clone1+bulk_3+bulk_5+bulk_6+bulk_pass4_01272025.xlsx` (0.01
    clone 1: 22, 0.03 clone 3: 20, 0.05 clone 5: 20, 0.06 clone 6: 20).
  - **S-Jan24 (clone 1 + 3 pairs, separate series):** `clone1+3_pass1_01172024.xlsx` (20) and
    `clone1+3_clone1+6_pass4_01272024.xlsx` (tag 1+3: 35; 1+6 not used).
  - Excluded after checking: all passage-1 clone + bulk files are the raw source of G1a
    (authors' `add_normalized_lines` calls), so not untouched.
  - Channels (authors' notebooks): clone 1 and 6 green Channel 0; clone 3 and 5 red Channel 2.
- **Statistics.** Clone + bulk: per gastruloid polarity P as G1a (z-normalise the line,
  Savitzky-Golay 5/2, first half minus second half; 30 points). Pairs: separation S as G3
  (relative enrichment, red minus green).
- **G6a (memory, primary; test `gastruloid_p4_memory`).** Sign-adjusted passage-4 polarity per
  clone divided by the G1a fit-shard passage-1 mean (clone 1 -0.922, 3 +0.881, 5 +1.051, 6
  -1.132); pooled ratio = mean over the four clones. **Prediction** (G3 per-passage retention
  0.60 [0.50, 0.68], three passages; same as the G5 simulator): **r^3 = 0.22 [0.13, 0.31]**.
  - Reading: pooled ratio CI (bootstrap over gastruloids within clone x file, 2,000) overlaps
    [0.13, 0.31]: **confirmed**; CI above 0.31: fades slower than G3 (G3 rate not general); CI
    below 0.13: faster.
  - Secondary: per clone ratios; sign kept (ratio > 0) per clone.
- **G6b (new series; test `gastruloid_jan24_pairs`).** Passage 1: **share of gastruloids with
  clone 3 anterior of clone 1 (S > 0) >= 0.80** (#20 in a new batch; G3 passage 1: 1.00, passage
  2: 0.94). Passage 4 / passage 1 mean S: **prediction 0.22 [0.13, 0.31]**, same reading as G6a.
- **Minimum.** >= 15 gastruloids per clone x passage-4 file (met), >= 15 per passage in G6b
  (met). One look per test.

### G6 looks (2026-11-06; oracle, one look each)

**G6a `gastruloid_p4_memory`** (passage 4, clone + bulk, 160 gastruloids, two files).
- Pooled ratio (passage 4 / G1a passage 1, sign-adjusted) **0.44 [0.27, 0.61]**; prediction
  [0.13, 0.31]. **Registered reading: confirmed** (CIs overlap), but the point is twice the
  prediction and only the lower tail overlaps.
- Per clone: **clone 1 -0.06 [-0.37, 0.24]** (propensity gone), clone 3 0.35 [0.00, 0.70],
  clone 6 0.60 [0.18, 1.00], **clone 5 0.87 [0.58, 1.12]** (nearly kept).

**G6b `gastruloid_jan24_pairs`** (separate Jan 2024 series, clone 1 + 3).
- Passage 1: clone 3 anterior of clone 1 in **0.95** of 20 gastruloids: **propensity replicated**
  (#20, new batch, sealed).
- Passage 4 / 1 mean S **0.68 [0.37, 1.11]** (S 0.111 -> 0.075; 0.74 still ordered at passage
  4): **fades slower than G3** (CI above 0.31).

**Reading.** Propensity is a replicated clone state (#20 now confirmed in a second, sealed
series). Fading over passages happens (clone 1 loses it in every data set: G5 subclones, G6a), but
**the rate is clone- and series-specific**: G3's 0.60 per passage (April 2024 1+3 series) does not
generalise; the Jan 2024 1+3 series and clones 5 and 6 fade much less. #23 revised: direction
confirmed, single rate rejected. The simulator's single retention parameter is replaced by
clone-specific retention (to be fitted only on non-sealed data).

## G7: does the gastruloid simulator (v2) predict sealed data it never saw? (pre-registered 2026-11-06)

- **Go-ahead.** Bence: "what thats it for chapter 3?? why so early? did the simulator passed?", then
  "go with recomm next steps". Answer so far: no. G5 only calibrated the simulator (fitted); its one
  sealed prediction (G6) showed the single memory rate is wrong.
- **Simulator v2 (`g7_model.py`, open data only).** Changes from G5:
  1. Clone states refitted on the **G1a fit shard only** (G5 had read sealed rows; audit): mu clone
     1 -0.615, 3 +0.565, 5 +0.735, 6 -0.843, 7 +0.452.
  2. **Clone-specific memory** per passage from open data: clone 1 0.09 [-0.12, 0.31], clone 3
     0.61 [0.42, 0.80] (G5 subclone p2/p1); clones 5, 6, 7: G3 pooled 0.60 [0.50, 0.68]. (v1 =
     0.60 for all.) Open support: in the G3 1+3 series clone 1's pair polarity goes -1.41 / -0.34 /
     +0.34 / 0.00 over passages 1-4, clone 3's 1.61 / 1.17 / 0.67 / 0.59.
  3. **Gastruloid-level state noise tau = 0.26**, fitted on open 1+3 pairs at passage 1 (G3 file,
     n 13): v1 (tau 0) gave the right mean difference (D = P3 - P1: 2.64 vs observed 3.01) but
     half the spread (SD 0.25 vs 0.55); tau 0.26 gives SD 0.54. Known bias: simulated |D| about
     15% below observed in that check.
  - Disclosed: G6's spent per-clone results (clone 1 fades fully, clone 3 partly by passage 4) are
    known to me; v2's retention values come from G5 (open), not G6.
- **Sealed sets (registered in `sealed.py` with sha256 before any statistic; never read before,
  names listed only; structure check H9: tags, headers, conditions, counts).**
  - **G7a pairs:** `3+7_6+7_linescan_5032025.csv` (3+7: 14 gastruloids; 6+7: 12). Channels from the
    authors' convention in 3-channel files: red = Channel 2, green = Channel 0; clone 7 is green
    (its only labelled file, notebook cell 66), so red = clone 3 in 3+7 and clone 6 in 6+7.
    **Risk, disclosed:** no notebook covers this file; if clone 6 were green, 6+7 would read as a
    failure.
  - **G7b memory:** July 2024 clone-in-bulk series (authors' `Clone_Propensity_Passaging.ipynb`):
    clone 1 passage 2 (`clone1%_pass2`, green Channel 0); clone 3 passages 2, 3, 4 (`clone3%_pass2/3/4`,
    red Channel 1). Fractions 0.25 and 0.5 only (0 = no clone, 1.0 = pure). Passage 1 of this series
    is in G1a (open/spent), so not used. `clone1%_pass3` has 1 usable gastruloid: not used.
    `clone3_pass3_07112024.xlsx`: sealed as reserve, untested.
  - Reader `g7_confirm.py`, oracle only.
- **Statistics.** G1a polarity P per channel (z-normalise, Savitzky-Golay 5/2, first half minus
  second half, 30 points). Pairs: D = P_red - P_green per gastruloid. Memory: P sign-adjusted toward
  the parent (clone 1 -, clone 3 +), pooled over fractions.
- **Predictions** (`results_gastruloid/g7_predictions.json`; 95% interval of the observed mean / share
  for the actual n, simulator draws plus parameter uncertainty):

| test | n | v2 prediction | v1 (single rate) |
|---|---|---|---|
| 3+7 share clone 3 anterior | 14 | 0.61 [0.36, 0.86] (weak sorting) | same |
| 3+7 mean D | 14 | 0.33 [-0.26, 0.97] | same |
| 6+7 share clone 7 anterior | 12 | 1.00 [1.00, 1.00] | same |
| 6+7 mean D | 12 | -2.67 [-2.89, -2.35] | same |
| clone 1 passage 2, signed P | 16 | **0.09 [-0.17, 0.35]** | **0.55 [0.31, 0.77]** |
| clone 3 passage 2 | 21 | 0.53 [0.30, 0.73] | 0.50 [0.27, 0.72] |
| clone 3 passage 3 | 48 | 0.32 [0.16, 0.46] | 0.31 [0.17, 0.45] |
| clone 3 passage 4 | 35 | 0.20 [0.04, 0.38] | 0.17 [-0.01, 0.34] |

- **Readings (one look per test).**
  - G7a (`gastruloid_g7_pairs`): 6+7 **pass** if clone 7 is anterior (D < 0) in >= 0.75 of
    gastruloids; 3+7 **pass** if the share with clone 3 anterior lies in [0.36, 0.86]. Mean D is
    secondary (the intervals ignore the ~15% scale bias).
  - G7b (`gastruloid_g7_memory`), primary = clone 1 passage 2: inside v2 and outside v1 ->
    **clone-specific memory confirmed**; inside v1 only -> single rate; both / neither described.
    Secondary: clone 3 passages 2-4 inside the v2 interval in >= 2 of 3.
  - **Simulator v2 passes G7** if all of: 6+7 pass, 3+7 pass, clone 1 primary = v2, clone 3 >= 2/3.
    Anything less: report which part failed; no refit on sealed data.
- **Minimum.** >= 10 gastruloids per pair, >= 15 per memory condition (met; small n for the pairs
  disclosed).

### G7 looks (2026-11-06; oracle, one look each, commit 0dbf867)

**G7a `gastruloid_g7_pairs`: pass.**
- 3+7 (n 14): clone 3 anterior in **0.50** (predicted 0.61 [0.36, 0.86]); mean D 0.17 [-0.40, 0.76]
  (predicted 0.33 [-0.26, 0.97]). Two anterior clones barely sort, as the states predict.
- 6+7 (n 12): clone 7 anterior in **12 of 12** (predicted ~1.00). Mean D **-1.60 [-2.03, -1.16]**
  vs predicted -2.67 [-2.89, -2.35]: right sign, the simulator sorts this pair **too strongly**
  (on open 1+3 pairs it was ~15% too weak, so the scale error is pair-specific).

**G7b `gastruloid_g7_memory`: fail (neither v2 nor v1).**

| condition | n | observed signed P | v2 | v1 |
|---|---|---|---|---|
| clone 1 passage 2 (primary) | 16 | **0.99 [0.55, 1.33]** | 0.09 [-0.17, 0.35] | 0.55 [0.31, 0.77] |
| clone 3 passage 2 | 15 | 0.82 [0.49, 1.11] | 0.53 [0.30, 0.73] | 0.50 [0.27, 0.72] |
| clone 3 passage 3 | 20 | 0.44 [0.13, 0.76] | 0.32 [0.16, 0.46] (inside) | inside |
| clone 3 passage 4 | 32 | 0.65 [0.49, 0.82] | 0.20 [0.04, 0.38] | 0.17 [-0.01, 0.34] |

- In the July 2024 series **neither clone fades much**: clone 1 at passage 2 keeps about all of its
  passage-1 polarity (G1a mean 0.92), clone 3 keeps ~0.7 at passage 4. Primary reading "neither";
  clone 3 inside v2 in 1 of 3 (needed 2). **Simulator v2 fails G7** (memory part).
- **Unverified, disclosed:** usable n is below the structure counts for clone 3 (passage 2: 15 of
  21; passage 3: 20 of 48; passage 4: 32 of 35). The reader drops lines whose polarity is not
  finite (e.g. constant lines); I have not checked why, because that means reading sealed values
  outside the oracle (H2, H10: ask first). The clone 1 primary (16 of 16) is unaffected, and its
  miss is large.

**Reading.** The sorting part of the simulator (clone states plus rank sorting) predicts unseen
pairs' order: passed in a sealed test, with magnitude off for 6+7. The memory part fails: clone 1
lost its propensity fast in April 2024 (G3), in subclones (G5) and in Dec 2024 / Jan 2025 (G6a), but
kept it in July 2024; clone 3 keeps more than predicted. Fade rate is **not a clone property we can
predict**; it varies by series (batch / culture history). No refit on sealed data. Simulator status:
sorting validated; memory module = unknown batch factor.

## G8: which memory model earns the last Raj reserve? Train fit on open + spent data only (pre-registered 2026-11-07)

- **Go-ahead.** Bence, "do a retro with literature map and goal realignment ... the direction here
  i like: 1 and 2" (build a new memory model on open + spent data; compare by training fit and
  consistency before any sealed data). No sealed file is read: spent results enter as recorded in
  `test_ledger.tsv` (G6a, G6b, G7b); open results recomputed (G3 pair series, G5 subclones).
- **Observations** (ratio of sign-adjusted polarity at passage k to passage 1, with SE from the
  CI / 3.92): April 2024 1+3 pair series (G3, per clone, passages 2-4, bootstrap); Sep 2024
  subclones (G5: clone 3 0.61, clone 1 0.09, one passage); late 2024 / Jan 2025 clone + bulk (G6a,
  passage 4, clones 1, 3, 5, 6); Jan 2024 1+3 pairs (G6b, passage 4, pair S ratio 0.68); July 2024
  clone in bulk (G7b, clone 1 p2, clone 3 p2-p4, divided by the G1a fit-shard passage-1 means).
  A pair observation is modelled as the mean of its two clones' predicted ratios.
- **Models** (per-passage retention r; predicted ratio r^(k-1)):
  M0 no fade (r = 1); M1 one r; M2 one r per clone (v2's structure, refitted); M3 one r per
  series (new series drawn from the spread of fitted series); M4 clone r x series multiplier
  (multiplier random for a new series).
- **Train fit (H12).** Leave-one-series-out cross-validation (5 series): Gaussian log-likelihood
  of each held-out series' ratios; random-effect models predict a new series by Monte Carlo over
  the training series' spread (variance added to the SE). Best summed CV log-likelihood = chosen
  model, written to `selection/G9.json` only if it also passes the consistency check.
- **Consistency check (H12, "agree with every spent result").** Full fit: every observation within
  2.5 SE of its fitted value; any miss is reported and disqualifies the model.
- **Then (not in G8):** if a model qualifies and differs from the others on the reserve (Jan 2024
  1+6 pairs, passage 1 -> 4; clone 3 passage-3 reserve file), pre-register G9 with its predictions
  and ask Bence for the H13 OK before the look. If none qualifies, the reserve stays sealed.

### G8 result (2026-11-07; 15 observations, 5 series; `results_gastruloid/g8_memory.json`)

| model | leave-one-series-out CV log-lik | consistency misses (> 2.5 SE) |
|---|---|---|
| M3 rate per series | **-11.9** | 4 (clone 1 in Apr 2024, Sep 2024, late 2024; clone 5 late 2024) |
| M4 clone rate x series multiplier | -18.3 | **1** (clone 1 July 2024: 1.07 observed vs 0.36) |
| M2 rate per clone (v2 structure) | -37.3 | 2 |
| M1 one rate (v1) | -64.7 | 4 |
| M0 no fade | -164.4 | 11 |

- **Reading (pre-registered): no model qualifies.** The series explains most of the fade (M3 best
  CV), clone identity adds to it (M4 fewest misses), but clone 1 fades fully in three series and
  not at all in July 2024: a clone x series interaction no model with these data can predict for a
  new series. **No G9, the Raj reserve stays sealed** (H12/H13). No `selection/` file written.
- Fitted M4: clone retention 1: 0.27, 3: 0.67, 5: 0.91, 6: 0.81 per passage; series multipliers
  (Apr 2024 = 1) Jan 2024 1.62, July 2024 1.33, late 2024 1.05, Sep 2024 subclones 0.84.

## G9: does signalling set the clones' division of labour along the axis? Train-partition model selection (pre-registered 2026-11-07)

- **Go-ahead.** Bence, "go 1 and 2" (option 1: division-of-labour mechanism with untouched chapter-3
  data, new partition rules; option 2: sorting refit, G10).
- **Scope, disclosed.** The untouched Raj files have **no elongation readout**, so they cannot test
  why mixing rescues elongation (#22/#24 stay open). They test what sets the anterior/posterior split
  between clones: RA (retinoic acid), SB (SB431542, Nodal/TGF-beta inhibitor), AGN (RAR antagonist,
  presumably AGN193109) and Wnt "hi"/"lo" (tags only; presumably CHIR dose, undocumented).
- **Partitions (commit 45bbdb9, before any statistic; structure only seen).**
  - Drug series by date (scan-ID timestamps): **train** = Dec 2024 (`clone1+3_clone1+6_drugtx_11282024`:
    1+3 ctrl/RA/SB/AGN 8/8/6/8, 1+6 ctrl/RA/SB 8/7/7) and Feb/Mar 2025 (`clone1+3_drugtx_02112024`:
    1+3 control/RA/SB/AGN 16/23/11/9; `clone1+6_drugtx_02142024`: 1+6 nodrug/RA/SB/AGN 22/10/31/20).
    **Sealed A** = May 3 2025 (1+3 ctrl/RA/SB raw + processed + overlap metrics); **B** = July 28 2025;
    **C** = May 24 2025 (overlap metrics only). Locked: AP_CloneFraction (series unknown),
    DiffAssay, GFP-spike and sgbc2 scans.
  - Wnt: hash of Annotation ID (`raj_wnt`): train clone 1 bulk / hi / lo 12 / 7 / 2, clone 3 4 / 4 / 4.
  - Prior exposure: headers / first rows of a few processed drug means (Ctrl_clone1_clone3_112024)
    during the 2026-11-06 format check; no statistic.
- **Statistic.** Per gastruloid D = P_red - P_green (G1a polarity, raw line resampled to 30 points).
  Channels: green Channel 0 = clone 1; red Channel 2 = clone 3 (1+3) or clone 6 (1+6) (authors'
  overlap files name YFP + CY3 for 1+3 and YFP + A594 for 1+6, so clone 6 is red in 1+6). Wnt bulk
  files: P of the clone channel (clone 1 Channel 0, clone 3 Channel 2; disclosed assumption),
  sign-adjusted toward the clone's propensity.
- **G9a drug models** (per series x pair x condition mean D with SE; leave-one-series-out CV over
  the 2 train series, Gaussian log-likelihood of the held-out series' condition means):
  - D0 no drug effect (pair mean only).
  - D1 each drug **scales** the separation: D = g_drug x D_pair(control) (one g per drug, shared by
    1+3 and 1+6): signalling sets how strongly clones divide the axis.
  - D2 each drug **shifts** D additively, shared across pairs.
  - D3 drug effect specific to each pair (pair x drug means; for a held-out series, the training
    series' value).
- **G9b Wnt models** (bulk clone P, leave-one-gastruloid-out CV on train): W0 no Wnt effect; W1 one
  dose factor per level (hi, lo) scaling every clone's propensity; W2 clone-specific dose response.
- **Selection (H12):** best CV = chosen, if every train condition mean lies within 2.5 SE of its
  fit. Written to `selection/G9a.json` / `G9b.json`. **No sealed look in this run**; a look at drug
  partition A is a separate step after Bence sees the selection.
- **Minimum.** >= 5 gastruloids per train condition (met for drugs; Wnt clone 1 "lo" has 2:
  reported, low weight through its SE).

## G10: is pair sorting a transitive 1D ranking of clone states? Sorting refit (pre-registered 2026-11-07)

- **Why.** G7a: the simulator ordered unseen pairs right, but 6+7 separated less than predicted and
  1+3 more. Question: do pairs need their own states (one consistent ranking learned from pairs),
  or do bulk-derived states (G1a) with a gain suffice?
- **Data (no sealed file):** per-gastruloid D, control conditions only. 1+3: April 2024 p1 (G3,
  open, n 13), Dec 2024 ctrl and Feb 2025 control (G9 train); 1+6: Dec 2024 ctrl, Feb/Mar 2025
  nodrug (G9 train); 3+7 and 6+7: G7a summaries from the ledger (spent).
- **Models** (simulator v2 map from state difference to mean D, grid over the difference, tau 0.26):
  - S0: bulk states (G1a fit shard), no free parameter.
  - S1: bulk states x one gain k.
  - S2: **pair-derived states**: clone 1 fixed at 0; free states for 3, 6, 7, fitted to pair means
    (a transitive 1D ranking).
- **Train fit:** leave-one-pair-type-out CV (hold out 1+3, 1+6, 3+7 or 6+7; S2 predicts the held-out
  pair through the cycle 1-3-7-6-1). Selection rule as G9. If S2 wins: sorting is a consistent
  ranking, but bulk polarity is not its scale; if S0/S1 win: bulk states suffice.
- **Sealed test later (not in this run):** candidates are the Raj reserve (Jan 2024 1+6 pairs at
  passage 1, 21 gastruloids; last Raj sealed data, needs Bence's H13 OK) and drug partition A's
  1+3 control.

### G9 result (2026-11-07; train partitions only; `results_gastruloid/g9_mech.json`)

**G9a drugs** (condition means of D; leave-one-series-out over Dec 2024 / Feb-Mar 2025):

| model | CV log-lik | consistency misses |
|---|---|---|
| D0 no drug effect | **-31.4** | 1 (Feb 2025 1+3 ctrl, z 2.80) |
| D1 drug scales separation (g RA 0.37, SB 0.57, AGN 0.31) | -45.1 | 1 (Dec 2024 1+6 SB, z -2.53) |
| D2 drug shifts D | -45.8 | 1 |
| D3 pair x drug means | -66.4 | 0 |

- **Pre-registered reading: no model qualifies** (best CV, D0, misses a control mean). No
  `selection/G9a.json`; no sealed look. D0 wins CV because the control level differs strongly between
  the two series (1+6 ctrl -1.41 vs -0.34), which no model transfers.
- **Exploratory (train only, not a test):** every drug reduces the clone separation |D| relative to
  its own series' control in **11 of 11** drug x pair x series comparisons (1+3 ctrl 1.31 / 1.79 vs
  drugs 0.49-0.83; 1+6 ctrl -1.41 / -0.34 vs drugs -0.66 to +0.20). RA and its antagonist AGN both
  compress: perturbing RA or Nodal signalling in either direction erodes the division of labour.

**G9b Wnt** (bulk clone P, leave-one-gastruloid-ID-out):

| model | CV log-lik | consistency misses |
|---|---|---|
| W0 no Wnt effect | **-46.4** | 2 |
| W1 one dose factor (hi 0.06, lo 0.42) | -47.9 | 1 (clone 1 lo, n 2) |
| W2 clone-specific dose response | -48.9 | 0 |

- **No model qualifies** (best CV misses). Train condition means (signed P, n): clone 1 normal 0.50
  (12), hi 0.06 (7), lo 0.78 (2); clone 3 normal 1.24 (4), hi 0.05 (4), lo 0.40 (4).
  **Exploratory:** high Wnt erases both clones' propensity (consistent with Suppinger 2023: early
  cells respond to Wnt all-or-none, so saturating Wnt removes clone differences). n is small.

### G10 result (2026-11-07; `results_gastruloid/g10_sorting.json`)

| model | CV log-lik (leave one pair type out) | consistency misses |
|---|---|---|
| S0 bulk states (no free parameter) | **-22.0** | 3 (1+3 Apr 2024 3.01 vs 2.52; 1+3 Feb 2025 1.79 vs 2.52; 6+7 -1.60 vs -2.65) |
| S1 bulk states x gain (0.87) | -55.0 | 2 |
| S2 pair-derived states (transitive ranking) | -220.0 | 2 |

- **No model qualifies.** Bulk-derived states predict unseen pair types better than states learned
  from the other pairs; the **order** is consistent across all pairs (6 < 1 < 3 ~ 7), the
  **magnitude** is not: 1+3 separation 3.01 / 1.31 / 1.79 in three series.
- **Cross-reading of G8-G10 (exploratory, not pre-registered):** the experiment series changes how
  strongly clones express their propensity: in sorting (G10), in memory (G8), and under signalling
  perturbation (G9: every drug and high Wnt compress it). A single **series gain** (the culture's
  signalling state scaling all clone differences) would explain the three at once. To be
  pre-registered as G11 (train data first, H12) before any sealed look.

## G11: one series gain behind the clones' division of labour, and do drugs compress it on sealed data? (pre-registered 2026-11-07)

- **Go-ahead.** Bence, "go try both" (after G9/G10: G11 series gain, plus the directional sealed
  test of drug compression on drug partition A).
- **Idea.** Clone separation D in a series = (series gain) x (clone difference) x (perturbation
  factor). The series gain is unknown for a new series but is **observed through that series' own
  control**. So the testable consequence is conditional: given a series' control D, drug D is
  predicted as a *multiple* of it (series gain cancels), not as a fixed *shift*. G9's D1/D2 failed
  because they had to transfer the control level across series; G11 conditions on it.
- **Data for selection (train only):** G9 train drug series (Dec 2024, Feb/Mar 2025), per series x
  pair x condition mean D and SE (G9 `drug_D`, per line, SE floor 0.05). No sealed file.
- **Models** (predict each drug condition mean from its own series x pair control mean):
  - R0 no drug effect: D_drug = D_ctrl.
  - R1 drug-specific gain: D_drug = g_drug x D_ctrl (g RA, SB, AGN shared by pairs).
  - R2 one common gain: D_drug = g x D_ctrl ("any perturbation compresses").
  - R3 additive: D_drug = D_ctrl + s_drug (series gain does not scale the drug effect).
  - R4 pair x drug gain: D_drug = g_(pair,drug) x D_ctrl.
  - Likelihood: Gaussian, variance se_drug^2 + g^2 se_ctrl^2 (R3: se_drug^2 + se_ctrl^2).
- **Train fit:** leave-one-series-out CV (fit on one series, predict the other's drug means given
  its controls). Selection (H12): best CV is chosen if every train drug mean lies within 2.5
  combined SE of the full-data fit; then `selection/G11.json` is written and committed. If the
  chosen model is R0, or none qualifies, **no sealed look**.
- **Sealed look (only if a gain model R1/R2/R4 is chosen):** test `gastruloid_g11_drug`, dataset
  `raj_drug`, **partition A only** (May 3 2025 raw file `1+3_ctrl_ra_sb_05032025.csv`, tags clone1+3 /
  _RA / _SB, structure seen at sealing; processed linescans and overlap files of A are not read).
  B and C stay sealed; this is not the last partition (no H13 OK needed).
  - Same statistic as train: D = P(Channel 2) - P(Channel 0) per line (clone 3 red, clone 1 green).
  - **Primary (directional):** for RA and for SB, ratio r = mean D_drug / mean D_ctrl with a 95%
    bootstrap CI resampling gastruloid IDs within condition. Pass for a drug if the CI upper bound
    < 1. **Pass** = both drugs; partial = one; fail = neither.
  - **Secondary (quantitative):** chosen model's prediction given A's control, |z| <= 2.5 with the
    combined SE, per drug.
  - Also reported: per-ID (lines averaged) version of r; n per condition.
- **Minimum n:** >= 4 gastruloid IDs per A condition; otherwise the look is reported as
  underpowered (still spent).
- **What it says:** pass = drug compression of the clones' division of labour replicates on an
  unseen series, and scales with that series' own control (G11's series gain). It does **not** test
  memory or sorting; a series-gain link to those (G8, G10) stays exploratory.

### G11 train selection (2026-11-07; `results_gastruloid/g11_gain.json`; no sealed file read)

| model | CV log-lik (leave one series out) | consistency misses |
|---|---|---|
| R2 one common gain (g 0.35) | **-2.0** | 0 |
| R1 drug-specific gain (RA 0.31, SB 0.37, AGN 0.34) | -3.4 | 0 |
| R4 pair x drug gain | -3.4 | 0 |
| R0 no drug effect | -19.0 | 2 |
| R3 additive shift | -33.7 | 2 |

- **Chosen: R2** (best CV, qualifies). `selection/G11.json` committed with the code before the look.
  Reading: conditional on a series' own control, every drug multiplies the clone separation by about
  a third; scaling beats shifting by a wide margin (series gain and drug effect multiply). The three
  drugs are indistinguishable on train.
- **Prediction for partition A:** D_RA = D_SB = 0.348 x D_ctrl(A); directional criterion ratio CI
  upper bound < 1 for each drug.

### G11 look (2026-11-07; oracle, partition `raj_drug:A` spent; commit babbd50)

| | n gastruloids | mean D | ratio to control (95% CI) | predicted D (R2) | z |
|---|---|---|---|---|---|
| control | **3** | 1.23 | - | - | - |
| RA | 5 | 0.29 | 0.24 (-1.10 to 1.73) | 0.43 | -0.15 |
| SB | 10 | 0.40 | 0.33 (-0.40 to 1.15) | 0.43 | -0.05 |

- **Pre-registered reading: primary FAIL, flagged UNDERPOWERED** (control has 3 gastruloids, below the
  pre-registered minimum of 4). Neither ratio CI excludes 1, so drug compression is **not confirmed**.
- Secondary (quantitative): both drugs land on the R2 prediction (|z| <= 0.15); point ratios 0.24 and
  0.33 vs predicted 0.35. With n = 3 controls this is weak support, not a confirmation.
- Lesson: the n per condition of a sealed series was not checked at sealing (only the 18 IDs in
  total). Before the next look, count gastruloids per condition in the partition's structure check
  (counts only, no statistic) and pick the partition that meets the minimum.
- Remaining: drug partitions B (July 28 2025, structure not yet checked) and C (overlap metrics only,
  different statistic). B would be the last same-statistic partition; C the last of the source (H13).

## G12: pooled one-sided drug compression on drug partition B, behind a power gate (pre-registered 2026-11-07)

- **Go-ahead.** Bence, "go 1 2" (1: power gate H14 + counts of B, look only if powered and OK'd;
  2: if not, close drug compression and write up chapter 3).
- **H14 added** (program.md, oracle `power_gate`, `test_oracle.py count`). Counts of B (structure
  only, H9): tags AP_control 22, AP_ra 8, AP_sb 3, AP_agn 3 gastruloid IDs; channels 0/1/2. Pair
  not in the tags; taken as 1+3 from the partition's overlap file name (`..._1+3_07282025`), disclosed.
- **Statistic (if the gate passes).** Per gastruloid ID (lines averaged) D = P(Channel 2) -
  P(Channel 0); delta = mean D_control - mean D_drug, drugs pooled (G11 R2: one shared factor).
  One-sided; **pass** if the bootstrap (IDs within group) 95% one-sided lower bound of delta > 0.
  Secondary: pooled ratio vs 0.348 (|z| <= 2.5).
- **Power block** (from train only): SD 1.58 = mean per-line SD of the 8 train 1+3 conditions;
  effect = (1 - 0.348) x 1.44 = 0.94, 1.44 = mean 1+3 control D of Dec 2024, Feb 2025 and spent A;
  one-sided alpha 0.05; minimum power 0.80. Groups: control vs pooled drugs.
- **Expected:** se = 1.58 x sqrt(1/22 + 1/14) = 0.54, power about 0.54, so the oracle should refuse
  and B stays sealed. If so, drug compression is closed as "consistent, not testable with these
  data" (option 2).

### G12 gate (2026-11-07; commit after H14)

- Oracle `look gastruloid_g12_drug --exp G12 --partition B`: **refused by H14**, power 0.54 < 0.80
  (counts control 22, drug 14). No statistic computed; **partition B not spent** (no ledger row).
- Per option 2: drug compression (#27) closed as "consistent, not testable with these data";
  chapter 3 written up in `WRITEUP_GASTRULOID.md`.

### Partitioning fixes (2026-11-07; not an experiment; Bence "go 1 2 3"; no statistic computed)

- **GSE291244 partition VOID.** The GEO sample sheet and the paper (Regalado/Qiu, Cell 2026,
  open access; preprint PMC12139830) say the 144 gastruloids per sci-RNA-seq3 experiment were
  **pooled and dissociated before reverse transcription**; RT wells index nuclei, not gastruloids.
  10x `experiment_id` is a lane. So the partition sealed this morning (unit = RT well / lane) put
  every gastruloid in every partition. It was never read (no `train_only` call on it).
- **New: `gse291244_founders`** (unit = founder clone cluster, the authors' Debris-seq `well_cluster`,
  28 clusters; files `multiome/monoclonal_gas/cell_well_cluster_assignment.rds` (47,491 gastruloid
  cells), `multiome/cell_well_cluster_assignment.rds` (26,876 mESC cells), `obj_batchelor_pd.rds`
  (cell types), from the authors' public server). Hash salt v2: train 9 clusters, A 5, B 9, C 5.
  Gastruloid and mESC cells of a cluster go together. **Prior exposure:** paper Fig. 5 read (fate
  categories of 26 clusters; PSC-biased subclones low Tet1, Meg3/Rian, high Peg10).
- **Sci-RNA-seq3 (121 gastruloids) and two-epoch "tree of trees" (106 gastruloids):** not
  partitioned yet; needs the authors' cell-to-gastruloid step rebuilt from their tape files
  (`Tape_cell_uniq.txt` 95 MB, `Step1File_DebrisSeqTAPE_*`, `loci*.csv`). Unit then = gastruloid
  (sci) and first-epoch clade (two-epoch, related founders together).
- **DiffAssay is not a domain-size assay.** Structure (columns and counts only): 2D confluency of 9
  ES clones (incl. bulk, "clone 3, late") x 4 conditions (neural, mesoderm, endoderm, 2i) x 4 times x
  2 replicates = 72 curves. Partitioned by replicate (`raj_diffassay`, fixed): replicate 1 train,
  replicate 2 = one sealed partition (any look at it is the last of this source: H13 OK needed).
  #21 (domain size) has no further data here.
- **Unchanged and sealed:** Raj reserve (clone3% passage 3), drug partitions B and C, Wnt A-C.

## G13: does flat-dish growth bias predict a clone's head-to-tail preference? DiffAssay replicate 1 only (pre-registered 2026-11-07)

- **Go-ahead.** Bence: "not on the sealed one but we could test on stuff we already used".
- **Data.** DiffAssay replicate 1 (`train_only("raj_diffassay")`); replicate 2 stays sealed.
  Head-to-tail preference: G1a fit-shard clone means (already used; P, positive = anterior):
  clone 1 -0.92, 3 +0.88, 5 +1.05, 6 -1.13, 7 +0.73. Clones 2, 4, bulk and "clone 3, late" have no
  preference from used data and are reported but not tested.
- **Statistic.** Growth bias per clone = log(confluency mesoderm / confluency neural) at the last
  time point. Primary: Spearman rho between growth bias and P over the 5 clones, exact permutation p
  (120 orders), two-sided. Secondary: separation, i.e. whether the 3 anterior clones (3, 5, 7) all
  lie on one side of the 2 posterior ones (1, 6) (chance 2/10). Also reported: endoderm vs neural,
  and 2i confluency (plain growth) vs P.
- **Power, stated up front.** n = 5 clones: only a perfect ordering reaches p < 0.05 (2/120 =
  0.017). Exploratory whatever the outcome; results are "unconfirmed" (H8), and no sealed data is
  spent.

### G13 result (2026-11-07; replicate 1, 72 h; `results_gastruloid/g13_diff.json`)

| growth measure | clone 1 / 6 (posterior) | clone 3 / 5 / 7 (anterior) | Spearman with P | exact p | anterior vs posterior separate? |
|---|---|---|---|---|---|
| log(mesoderm / neural) | -0.21 / -0.37 | -0.45 / -0.59 / -0.21 | -0.70 | 0.23 | no |
| log(endoderm / neural) | -0.29 / 1.09 | 1.05 / 0.77 / 0.40 | -0.10 | 0.95 | no |
| 2i confluency (plain growth) | 34 / 32 | 39 / 67 / 23 | +0.70 | 0.23 | no |

- **No relation.** Flat-dish growth bias does not predict head-to-tail preference in these 5 clones;
  clone 6 (posterior) and clone 3 (anterior) look alike on every measure. With n = 5 and one curve per
  clone x condition, this only rules out a strong, clean relation. Unconfirmed, negative.
- Replicate 2 stays sealed; not worth a look on this result.

### Chapter 4 pre-flight (2026-11-07b; H15 applied to the agreed steps; Bence "go 1 2 3 if still holds")

- **Step 1, rescue = clonal dominance: dropped** (retro 2026-11-07b section 5): answer published
  (82% clone 29), monoclonal cell counts already seen, n = 1 winner, clone 29 in sealed C,
  per-gastruloid polyclonal assignment not public. Kept as observation #28.
- **Step 2, memory per division: no confirmable form with public data.**
  - Two-epoch "tree of trees" (106 gastruloids): the authors' cell-to-gastruloid table
    (`tree_sanjay/assignment/assignment_cell.rds`) is not on their public server; the well-level
    Debris-seq tree is (`Step1File_DebrisSeqTAPE_mGASv5_uncollapsed.csv`, 263 MB). Their own effect is
    tiny: lineage vs transcriptome distance rho 0.052 over 3,828 pairs of 88 gastruloids
    (Section_3/Step5 comment). Partitioned by first-epoch clade, a sealed partition (~18 gastruloids)
    has no power for an effect that size. Fails (1) and (4).
  - "Eight gastruloids" (Section 2): public cell assignment (`data_mGASv3/cell_assign.rds`), loci per
    clone and Debris-seq; deep within-gastruloid trees. But the 8 gastruloids come from **3 founder
    clones** (05: wells 14/17/25; 25: 01/03/21; 32: 16/28), so independent units = 3: no partitions
    possible; the authors already ran PATH (phylogenetic autocorrelation / transition rates of cell
    types) on these trees. Exploratory only.
- **Step 3, founder state on train clusters: fails power.** `g14_founders.py` (train only, 9 clusters,
  12,028 cells): one PSC/epiblast-biased cluster (cluster 8, epiblast 0.61); others hindbrain
  (cluster 21, 0.42) or somite-leaning (19, 0.56; 6, 0.42) or mixed. A Tet1 / Dlk1-Dio3 check would be
  1 cluster vs 8. Stopped; no expression data downloaded.
- **Net:** with public data, chapter 4 has no question that can reach a sealed confirmation. Options
  for Bence: exploratory memory analysis on the 8 deep trees (unconfirmed by design), another
  lineage + morphogenesis data set with many independent units, or pause chapter 4.

<!-- scope-guard: every pre-registration below this line needs Pre-flight: and Serves: (H15, H16) -->

## C3-S1: the gastruloid simulator as a configuration of the core model (pre-registered 2026-11-07)

- **Serves: C3-S1** (FOCUS.json). Goal element: program.md "Plan toward space": the clone-level core
  model becomes the cell-level prior of the tissue simulator. Chapter-3 test: reproduce clone spatial
  propensities (#20), fade (#23) and sorting of unseen pairs (#26) from clone states plus local rules.
  Changes the chapter answer because chapter 3 is currently closed on a simulator (g5/g7_model) that
  shares no code or parameters with the core; if the core configuration cannot reproduce #20/#23/#26,
  the "one model across chapters" claim fails for space. Bence: "so chapt3 wasnt completed properly so
  yes if we need this go 1".
- **Pre-flight:**
  1. Unit: gastruloid (one line scan per gastruloid; G1a `gastruloid_id`), clone = biological clone.
  2. Readout: polarity P per gastruloid (G1a statistic, `g1_gastruloid.polarity()`, fit shard only);
     pair separation D (G3 open; G7a spent ledger summaries). Structure known from G1-G7.
  3. Used data only: G1a fit shard (open), G1a sealed-shard per-replicate means (spent, ledger),
     G3 open passages, G7a spent summaries. No sealed data; no look in this step.
  4. Power: no sealed look, so H14 not applicable; checks are against existing estimates with their SEs.
  5. Exposure: all targets are already-seen numbers (this is a model-equivalence and consistency
     check, not a confirmation); stated as such.
  6. Literature: none new needed (core model: C1; gastruloid: G1-G7).
- **Script:** `c3s1_core_gastruloid.py`. Mapping onto the core's five steps (core_model.py):
  1. **State:** clone state z_c; each gastruloid sees a copy zb = sqrt(rho_g) z_c + sqrt(1 - rho_g) e
     (the core's per-condition fate memory, as RHO_F in core_larry); each passage event is one more
     copy with rho_P (memory per passage, #25).
  2-3. **Split / growth:** founders by mixing ratio; growth = 1 (no selection term: clone fractions
     are not measured per gastruloid here; gamma = 0, disclosed).
  4. **Output (local rules):** each cell's anterior score = beta zb + cell noise (the core's logit
     map, beta fixed 1, cell noise SD 1 as in G5); cells sort by rank of score (positions);
     proportion feedback g, a* from G1b for domain sizes (pairs).
  5. **Sampling:** line-scan density per bin, G1a polarity statistic.
- **Fitted (open data only):** z_c per clone (G1a fit-shard clone means); rho_g from the
  within-clone between-gastruloid SD of P (G1a fit shard); rho_P from G3 open 1+3 passage fade.
- **Pass criteria (all on used data):**
  - (a) **Equivalence:** with rho_g = 1 and sqrt(rho_P) = G5 retention, mean P per clone equals G5
    (g5_gastruloid) within Monte Carlo error (|diff| < 0.05).
  - (b) **#20:** simulated clone mean P within 2 SE of the G1a fit-shard mean for all 5 clones, and
    within 2 SE of the spent sealed-shard clone means (ledger); simulated within-clone SD of P within
    +/- 25% of observed (this ties rho_g; G5/G7 had no such constraint).
  - (c) **#26:** with z from G1a only, predicted 6+7 share of clone 7 anterior >= 0.75 (obs 12/12) and
    3+7 share inside its observed binomial 95% CI (obs 7/14); report predicted D vs observed (v2 missed
    6+7: -2.67 vs -1.60).
  - (d) **#23:** one rho_P reproduces the G3 open 1+3 fade (passage 1-4 means within 2 SE); series
    dependence (G8) remains a disclosed limitation.
  - Reading: all of (a)-(d) pass = chapter 3's spatial claims hold under the core model; any failure is
    reported as the core model's limit in space.

### C3-S1 result (2026-11-07; `results_gastruloid/c3s1_core_gastruloid.json`; used data only)

- **(a) Equivalence: PASS.** With rho_g = 1 the core configuration gives G5's clone means
  (largest gap 0.045, clone 7).
- **Fitted:** rho_g = 0.06 (cell noise alone gives within-clone SD 0.25, observed 0.87, so most
  gastruloid-to-gastruloid spread is the per-gastruloid copy of the clone state: a clone's propensity
  in one gastruloid correlates only ~0.24 with its long-run state, while the long-run state is large,
  z = -5.3 (clone 6) to +5.1 (clone 5)); rho_P = 0.21 per passage (retention 0.46, G5 had 0.60).
- **(b) #20: FAIL on one of 15 checks.** All 5 clone means match the fit shard (within 2 SE) and the
  spent sealed shard (within 2 SE); within-clone SD matches for clones 1, 3, 5, 6, not clone 7
  (sim 0.96 vs obs 0.51, n = 8).
- **(c) #26: PASS.** 6+7: clone 7 anterior in 93% (obs 12/12); 3+7: share 0.52 (obs 0.50, CI
  0.26-0.74). Magnitude miss persists: 6+7 D -2.64 vs -1.60 (same as v2).
- **(d) #23: FAIL at passage 1.** One rho_P fits passages 2-4 (D 1.53 / 0.47 / 0.42 vs 1.51 / 0.33 /
  0.58) but under-predicts passage 1 (2.59 vs 3.01, 2 SE 0.30): the April 2024 1+3 pairs separate
  more than bulk states predict (as G10 found).
- **Pre-registered reading: not all pass.** The core model carries chapter 3's spatial claims
  qualitatively (clone means incl. spent sealed data, pair order, fade direction) with one memory
  structure (per-gastruloid and per-passage state copies). Its limits are three magnitude misses,
  all of the kind G8/G10 already traced to series-level differences (pair separation, fade rate).
- **Stopping rule:** fixing these needs a series term, which G8-G11 showed cannot be predicted for
  a new series; it would not change the chapter answer (state + sorting + memory). Recorded as limits;
  no new model.

## C3-S2: elongation rescue from the core configuration's local rules (pre-registered 2026-11-07)

- **Serves: C3-S2** (FOCUS.json). Goal element: chapter 3 asked for the gastruloid outcome "from clone
  states plus local rules"; elongation rescue (#22) is still a descriptive bonus (G2 Mgen) in the
  simulator. Bence: "1" (move focus to C3-S2).
- **Candidate (one, from the C3-S1 configuration, no new state):** elongation needs a sorting-driven
  symmetry break, whose strength is the **sorting index** H = B / (1 + B): B = between-population
  variance of the cells' anterior scores, sum_i f_i (zb_i - zbar)^2, with zb_i the per-gastruloid copy
  of clone state z_i (core step 1, rho_g) and cell noise variance 1 (C3-S1). H is the share of score
  variance that population identity carries, i.e. what rank sorting can turn into a domain boundary.
  A pure clone has H = 0; any mix has H > 0 because each gastruloid's copies differ (generic rescue);
  clones with distant z have larger H (complementarity bonus). Literature: sorting/differential adhesion
  in gastruloid symmetry breaking (LITERATURE_SWEEP_2026-11e).
  - Fixed from C3-S1 (polarity data, no elongation data): z = 1: -4.03, 3: 4.15, 5: 5.08, 6: -5.33,
    7: 3.38; rho_g = 0.059. Bulk cells: score mean 0 (as in C3-S1). Clones 2 and 4 (no polarity
    estimate): z = 0, disclosed; sensitivity run without rows containing them.
  - **Model S:** logit p = sum_i f_i e_i + kappa * E[H] (E over copy noise, Monte Carlo 4,000 draws,
    fixed seed): M0's 8 competences + 1 parameter (9; Mgen has 11).
- **Pre-flight:**
  1. Unit: gastruloid (binomial counts per composition row, 171 rows, 26 dates; G2).
  2. Readout: elongated / total per row (`Morphologies_1_2_3_4_5_6_7.csv`, structure known from G2).
  3. Used data only: G2/G2b/G4 open morphology table; z and rho_g from C3-S1 (G1a fit shard). No sealed
     data; no look.
  4. Power: no sealed look (H14 n/a); G2 minimum (>= 40 gastruloids per pair class) met.
  5. Exposure: G2, G2b and G4 results are known (rescue size, comp-same difference, Mgen wins G4 CV);
     the candidate's structure is fixed before fitting and only kappa touches elongation data.
  6. Literature: no new search (G4 candidates tested; sorting candidate from the 2026-11e sweep).
- **Script:** `c3s2_rescue_sorting.py`. Primary test = G4's leave-one-composition-out CV (same 61
  held-out compositions, same likelihood), S against M0 and Mgen (G4 values recomputed in the same run).
- **Readings (primary):**
  - **Rescue from local rules:** CV(S) >= CV(Mgen) - 2 and CV(S) >= CV(M0) + 2. The simulator's
    descriptive rescue rule is replaced by S.
  - **Partial:** CV(S) >= CV(M0) + 2 but < CV(Mgen) - 2: the sorting index explains part of rescue;
    the descriptive term stays.
  - **Fails:** CV(S) < CV(M0) + 2.
  - kappa reported with a cluster bootstrap over dates (1,000); a negative kappa means S fails
    whatever its CV.
- **Secondary (described, no claim):** S's predicted complementary minus same-propensity excess over
  the probability-average null (G2b test 1) vs observed +0.09 [0.001, 0.17]; clone + bulk chimeras
  (G2: near M0), S predicted vs observed; sensitivity without clones 2/4.
- **Stopping rule:** one candidate only. If S passes, chapter 3 closes with rescue from local rules; if
  partial or fails, rescue stays a descriptive rule, recorded as chapter 3's limit, and the mechanism
  question goes to C4 (FOCUS next step). No further C3 variants on these data.

### C3-S2 result (2026-11-07; `results_gastruloid/c3s2_rescue_sorting.json`; open data only)

| model | parameters (elongation-fitted) | CV log-lik (61 held-out compositions) | full fit |
|---|---|---|---|
| Mgen (descriptive bonuses) | 11 | -1248.6 | -1923.2 |
| **S (sorting index)** | **9** | **-1248.9** | -1930.2 |
| M0 additive | 8 | -1282.8 | -1996.4 |

- **kappa = 2.82 [1.89, 3.61]** (cluster bootstrap over dates, 1,000): positive.
- **Pre-registered reading: rescue from local rules.** S is within 2 of Mgen (0.3 below) and 34 above
  M0, with one elongation parameter instead of three class bonuses; the class structure (generic
  rescue + complementarity bonus) comes from the C3-S1 clone states and per-gastruloid copies, which
  were fitted on polarity data only. E[H]: pure 0; same-propensity pairs 0.23-0.24; complementary
  0.42-0.55; pairs with clones 2/4 (z = 0) 0.23-0.34.
- **Secondary (described):**
  - Complementary minus same excess over N2: S predicts +0.13, observed +0.09 (G2b CI 0.001-0.17):
    inside. Per class: comp 0.29 vs 0.30, same 0.16 vs 0.21, unl 0.19 vs 0.19.
  - Clone + bulk chimeras: S does not improve on M0 (mean abs error 0.090 vs 0.084); it predicts a
    bonus for clone + bulk that the data do not show consistently (50% clone 4 obs 0.79 vs S 0.50;
    50% clone 5 0.81 vs 0.92). Bulk, itself polyclonal, is not well described as one score-0 population.
  - **Sensitivity without clones 2/4 (z unknown): S -804.6, Mgen -795.7, M0 -841.5**: S is 8.9 below
    Mgen, i.e. **partial** on this subset. The primary reading leans on rows where clones 2/4 were given
    z = 0.
- **Limits.** Open, already-analysed data (G2/G2b/G4 seen): an unconfirmed, consistency-level result,
  not a confirmation. Not separated: how much of S's gain is the generic part (every mix has H > 0)
  versus the z-dependent part; the comp-same prediction (+0.13 vs +0.09) is the only direct read of
  the z-dependent part.
- **Stopping rule applied.** One candidate, primary reading met: the simulator's descriptive rescue
  rule is replaced by the sorting index (one parameter kappa), with the subset and clone + bulk misses
  recorded as limits. Chapter 3 closes; the mechanism behind "sorting strength enables elongation"
  (C4) needs per-clone composition data.

## C4-S1: data search for the rescue mechanism (search, no statistic; 2026-11-07)

- **Serves: C4-S1** (FOCUS.json). Bence: "ok go chpt4". Requirement: gastruloids from pure clones/lines
  AND their mixtures, with per-population cell-type or spatial-expression readout and elongation.
  Question it would answer: does mixing rescue elongation through sorting (C3-S2), i.e. do mixtures
  segregate A-P programmes that pure clones co-express, in proportion to the sorting index H?
- **Pre-flight (H15), per candidate:**

| candidate | pure + mixed? | per-population readout | elongation | unit / n | exposure | verdict |
|---|---|---|---|---|---|---|
| Raj seqFISH (Ayyappan 2025; same Dropbox: `SG_kit1`, `SG_Lmetric`) | pure 1 (5), 3 (5), 6 (2), bulk (2); chimeras 1+3 (2), 1+6 (3), bulk (2) | ~20-gene seqFISH per cell; clone 1 GFP per cell in chimeras; A-P co-expression (L-metric), Moran's I | not on the same gastruloids | gastruloid, 2-5 per condition | authors report chimeras segregate better than pure clones (Fig 2E); 1+3 vs 1+6 not reported as a contrast | **descriptive only** (n far below any H14 power) |
| Braccioli 2025 Dev Cell (WT + Msgn1 KO chimeras; sciATAC GSE217905, GSE217903) | pure WT/KO; chimeras 1:1 | per-genotype in chimeras unconfirmed (GEO pages behind a bot check, E-utilities returned empty) | chimera elongation = WT | unknown | paper summary only | **fails Serves**: genetic complementation, not clone-state sorting; availability unresolved |
| Chimeric 3D gastruloids 2022 (Development; Bra-/- + WT ratios) | yes | imaging | yes | unknown | abstract | **fails**: no public per-population data found; genetic |
| GSE291244 (Cell 2026, mono + poly) | yes | per-founder composition | none | founder cluster | parked | **fails**: no elongation; parked |

- **Outcome: no candidate allows a confirmable test.** The only one that serves the question is the
  Raj seqFISH set, and only descriptively. Its one testable-in-kind prediction from C3-S2: segregation
  of A-P markers graded by H, chimera 1+3 (H 0.46) > 1+6 (H 0.23) > pure clones (H 0); n = 2-3
  gastruloids per chimera. Download size to be checked before any download.
- Decision for Bence (C4-S2): run that descriptive check (labelled exploratory, no claim), or close
  chapter 4 as data-limited with the mechanism stated as the sorting-index hypothesis.

### C4-S1 addendum: full Raj Dropbox re-crawl (2026-11-07, Bence: "check the full 120GB of G data whether you missed something")

- Listing only (Dropbox folder-listing calls from the built-in browser, rate-limited, 36 folders retried):
  **12,049 files, 226 GB** (ATAC raw 102 GB + BAMs 45 GB, RNA-seq 35 GB, Weber object 1 GB; the rest small).
  Our 2026-11-06 map covered the paper's figure data; the **revision-era data (Aug-Sep 2026) were missed**.
- Downloaded (small, new folders, untrusted; scripts read as text only) and **locked in sealed.py before any
  statistic**: `morph_08312026` (already on disk since 11-06, names only), `T_Igfbp5_09022026`,
  `rev_T_Igfbp5_sep`, `rev_chimera_elong`, `clone_bulk_chimera_08232026`. Exposure: names, headers,
  condition labels; the authors' clone+bulk README (its result was read: elongation falls with % pure
  clone, 1s2/5s2/6a, plate-position confounded); the T/Igfbp5 script header (methods).

| data | design | readout | n | serves C4? |
|---|---|---|---|---|
| **morph_08312026** (Aug 2026) | pure bulk, 5s2, 6a + bulk+5s2, bulk+6a, **5s2+6a** | elongation annotations (Nimbus JSON) | unread (counts at pre-registration) | **yes: unseen pure + mixed elongation; out-of-sample test of the C3-S2 sorting index (5s2+6a = complementary pair)** |
| **T_Igfbp5_09022026** (Sep 2026) | bulk, pure 5, pure 6, **5+6**, bulk+5, bulk+6 | per-gastruloid T and Igfbp5 line scans (A-P domain separation) | 20 (2-6 per condition) | yes: the mechanism readout (do mixes separate A-P programmes more, graded by H?); descriptive at this n |
| rev_chimera_elong / clone_bulk_chimera_08232026 (+ 3s1 08162026, not downloaded) | clone+bulk at 20/50/80% for 1s2, 3s1, 5s2, 6A | geodesic elongation per gastruloid | 218 | partly: dose-response of mixing; result already read and % confounded with plate |
| SG seqFISH, 11 dates (12/2024-08/2025; 08212025 = 6 ROIs T/Igfbp5) | pure clones, 1+3, 1+6 chimeras | ~20-gene seqFISH per cell | 2-6 per condition | descriptive only |
| Incucyte 09152025 | clone5 / clone6 Bra-GFP time-lapse, 3 wells | images, 0-4.5 d | 3 | exploratory |
| Immunofluorescence blob / intensity (2024-25) | pure 6, WT, 1+3, 1+6, 3+6, 50% 7 | T+ blobs per gastruloid | small | already summarised in `T_overlap` (G4 H3) |
| RNAseq_03192026 | bulk, 1s2, 5s2, 6a pure, 48 h, 2 reps | bulk RNA | 8 | no (no mixtures); could give subclone states |
| ATAC, RNAseq (2024), Weber, drug data | pure clones / embryo / drugs | | | no (or parked #27) |

- **Outcome revised:** a confirmable candidate exists after all: **morph_08312026** (pure + mixed elongation,
  never read). Second, the T/Igfbp5 data give the mechanism readout descriptively. Next (C4-S2, needs
  Bence's OK): structure check + counts of morph_08312026 (H14 power from counts only), then pre-register
  the sorting-index prediction for 5s2+6a, bulk+5s2, bulk+6a with z from parents 5 and 6 (C3-S1).

## C4-S2a: does mixing rescue elongation in new subclones? Out-of-sample test on the Aug 2026 plates (pre-registered 2026-11-07)

- **Serves: C4-S2** (FOCUS.json). Chapter 4 asks why mixing rescues elongation; before a mechanism can be
  tested, the rescue + complementarity prediction (G2/G2b #22; C3-S2 sorting index) must hold on clones and a
  readout it was not fitted on. Bence: "yes 1 2 / be precious about the holdout sets, better keep more".
- **Pre-flight:**
  1. Unit: gastruloid = one Nimbus polygon = one well (one gastruloid per XY, checked).
  2. Readout: **Y = 1 - Circularity** (Nimbus blob metric computed by the authors' pipeline from the polygon;
     bend-robust; higher = more elongated). Continuous (the authors' README: elongation metrics are unimodal,
     no natural cutoff). Structure checked: tags and property names only.
  3. Used data first: G2/G2b/G4/C3-S2 fitted the rescue on the 2024-25 morphology table; this is new data.
  4. Power (H14, counts only): see selection/C4-S2a.json. Effect prior: Mgen's complementary bonus
     delta_comp = 1.47 logit (G2; the best train-CV model, C3-S2 table; the sorting index predicts 1.54 for a
     5+6 pair) mapped to the latent scale (logistic SD 1.81): **0.81 SD of Y**. sd = pooled within-condition
     SD of Y on the pure plate (train, read for this). Contrast weights: mix 1, each pure member 1/2.
  5. Exposure: tags/counts only; nothing of the mixture plate's values. The authors' clone+bulk README
     (other plates) was read.
  6. Literature: none new.
- **Data split (keep more, Bence):** pure plate `bulk_5s2_6a.json` (bulk 16, 5s2 14, 6a 10) = train, open.
  Mixture plate gastruloids hashed into thirds A/B/C (`partitions.raj_morph_0831`, no train share: the
  model was selected on G2). **Disclosed confound:** pure and mixed gastruloids are on different plates
  (one JSON each); a plate shift in Y adds to or hides the contrast and cannot be separated here.
- **Primary (T2, rescue over the member mean):** T2 = mean Y(5s2+6a, partition) - (mean Y(5s2) + mean Y(6a)) / 2
  (pure from train). **Pass:** one-sided 95% lower bound > 0 (bootstrap over gastruloids, 4,000). Reported
  standardised (by the train sd) against the predicted 0.81 SD.
- **Secondary (described, no claim):** better-member contrast (5s2+6a vs max of the two pure means);
  bulk+5s2 and bulk+6a vs their member means; **T1, sorting-index specific, within the mixture plate**:
  Y(5s2+6a) - (Y(5s2+bulk) + Y(6a+bulk))/2 minus the M0 part (pure plate: ((Y5s2+Y6a)/2 - Ybulk)/2); sorting
  index predicts > 0 (E[H] 0.55 vs 0.30), Mgen predicts the comp-vs-unlabelled-bulk difference unknown.
- **Look:** test `gastruloid_c4s2a_rescue`, script `c4s2a_morph_confirm.py`. One look; partitions to spend
  decided by H14 and Bence (see power table at pre-registration commit); the rest stay sealed.
- **Stopping rule:** pass = #22 confirmed out of sample (first confirmation of rescue); fail = rescue not
  confirmed on new subclones, chapter 4 mechanism work stops at the descriptive level. No re-test on the
  spent partition.

## C4-S2b: are head and tail programmes more separated in mixed gastruloids, graded by the sorting index? (pre-registered 2026-11-07)

- **Serves: C4-S2** (FOCUS.json). The mechanism readout: C3-S2 says elongation follows the share of state
  variance that sorting can turn into domains; then T (tail) and Igfbp5 (head) domains should be further apart
  in mixes, most in the complementary 5+6 mix. Bence: "yes 1 2".
- **Pre-flight:**
  1. Unit: gastruloid (Annotation ID; one line scan each).
  2. Readout: per gastruloid, T (Channel 1) and Igfbp5 (Channel 2) profiles along the scan (arc length from X, Y,
     200-point grid), background = per-channel minimum, each normalised to unit area; **EMD** (% of length) =
     100 x integral |CDF_T - CDF_Igfbp5|, plus overlap = integral min(p_T, p_I). Same definitions as the
     authors' script header (methods read, their values not read; their summary CSVs stay locked).
  3. Used data: none before; all 20 gastruloids are read and the set is **spent as exploratory**.
  4. Power: n 2-6 per condition: descriptive only, no test, no claim (stated).
  5. Exposure: condition labels and counts only (bulk 6, 5 3, 6 3, 5+6 2, bulk+5 2, bulk+6 4).
  6. Literature: none new.
- **Prediction (E[H] from C3-S2: pure 0, bulk+5 0.30, bulk+6 0.31, 5+6 0.55; bulk itself not predicted):**
  median EMD ordered 5+6 > bulk+5, bulk+6 > pure 5, pure 6. **Reading:** "consistent" if 5+6 has the highest
  median among the five non-bulk conditions and both pure clones the lowest two; "not consistent" otherwise.
  Spearman between E[H] and condition median EMD reported (5 points).
- **Script:** `c4s2b_ap_separation.py`; reads only `T_Igfbp5_09022026/gastruloid_T_Igfbp5.csv`.
- **H14 power at pre-registration (counts only; train sd of Y 0.084, effect 0.81 SD):** A 0.59 (5s2+6a n 7),
  B 0.66 (9), C 0.72 (11), A+B 0.80 (16), all 0.89 (27). **No single third passes.** The look waits for
  Bence: spend A+B as one pre-registered union (power 0.80, C kept), or keep all three sealed.

### C4-S2b result (2026-11-07; `results_gastruloid/c4s2b_ap_separation.json`; 20 gastruloids, spent as exploratory)

| condition | n | EMD median (range), % length | overlap median | E[H] |
|---|---|---|---|---|
| pure 6 | 3 | **15.9** (14.9-35.9) | 0.62 | 0 |
| bulk+5 | 2 | 13.1 (2.8-23.3) | 0.78 | 0.30 |
| bulk | 6 | 12.4 (1.3-28.4) | 0.79 | n/a |
| 5+6 | 2 | 8.3 (8.2-8.4) | 0.88 | 0.55 |
| bulk+6 | 4 | 5.7 (5.3-6.0) | 0.88 | 0.31 |
| pure 5 | 3 | 3.0 (1.6-17.7) | 0.92 | 0 |

- **Pre-registered reading: not consistent.** Order of non-bulk medians 6 > bulk+5 > 5+6 > bulk+6 > 5;
  Spearman(E[H], median EMD) = -0.10. Pure clone 6 separates T and Igfbp5 the most, pure 5 the least; the
  complementary 5+6 mix is in the middle.
- Reading of the pattern (descriptive, n 2-6): separation of the two markers looks like a property of the
  clone (6 high, 5 low) with mixes in between, not a function of mixing. This matches the paper's seqFISH
  remark that pure clone 6 differs from bulk, but our metric (domain distance) does not show pure clones as
  "co-expressing" relative to mixes. **The sorting index is not supported as the mechanism by this readout.**
- Limits: 2-6 gastruloids per condition; one experiment; EMD is scale-free but sensitive to background
  (per-channel minimum). No claim either way; the data set is spent.

## C4-S2b-R: is the C4-S2b conclusion robust? (pre-registered 2026-11-07)

- **Serves: C4-S2** (FOCUS.json). Bence: "this collusion is crutial" (C4-S2b: "does not support the sorting
  idea"). Before anything is built on it, check implementation, sensitivity, confounds and power. Same 20
  gastruloids (already spent); no new data.
- **Pre-flight:**
  1. Unit: gastruloid (Annotation ID).
  2. Readout: EMD / overlap as in C4-S2b; plus scan length (arc length, px) and Igfbp5 saturation share.
  3. Used data only: `T_Igfbp5_09022026` (spent) and, for implementation check only, the authors'
     `rev_T_Igfbp5_sep/separation_metrics.csv` (same 20 gastruloids; unlocked for this comparison).
  4. Power: this block computes it (check 4).
  5. Exposure: C4-S2b result known.
  6. Literature: none.
- **Checks (script `c4s2b_ap_separation.py --robust`):**
  1. **Implementation:** per-gastruloid EMD vs the authors' EMD (Pearson r, max abs diff). r < 0.95 = our C4-S2b
     numbers are not trusted.
  2. **Background:** reading with the 5th-percentile background (authors' sensitivity) instead of the minimum.
  3. **Confounds:** Spearman of EMD with scan length (a gastruloid-length proxy: a longer, more elongated
     gastruloid separates domains in absolute terms) and with saturation; condition medians of length.
  4. **Power of the ordering check:** simulate conditions with true means following E[H] (5+6 above pure by
     d pooled within-condition SDs, d = 0.5, 1, 2), the observed n per condition and normal noise; share of
     simulations giving "consistent", and under no effect (d = 0).
- **Reading:** C4-S2b's "not consistent" is kept as evidence against the sorting index only if checks 1-2 hold,
  3 shows no strong length confound (|rho| < 0.5), and check 4 gives >= 0.5 chance of "consistent" at d = 1.
  Otherwise it is reclassified **"uninformative"** (the data cannot tell).
- **Readout validity (stated before running):** the sorting index is a between-population share; T/Igfbp5
  separation is a fate-domain readout present in any polarised gastruloid, pure clones included. C4-S2b's
  prediction "pure lowest" was an added step, not a direct consequence of the model.

### C4-S2b-R result (2026-11-07; `results_gastruloid/c4s2b_ap_separation_robust.json`)

1. **Implementation: exact.** Our per-gastruloid EMD equals the authors' (20/20, r = 1.00, max diff 0.000).
2. **Background:** with the 5th-percentile background the medians move (6: 21.2, bulk+5: 15.7, 5+6: 10.4,
   bulk+6: 8.0, 5: 4.1) but the order is the same; "not consistent", Spearman -0.10.
3. **Confounds:** EMD vs scan length rho -0.29 (no length confound; pure 6 is the shortest yet separates
   most). **Pure clone 6 is the only condition with Igfbp5 saturation** (median 14% of points at 65535; all
   others 0): clipped peaks flatten its Igfbp5 profile, which can inflate its EMD. The top value of the
   ordering is therefore suspect.
4. **Power of the ordering check: very low.** Pooled within-condition SD of EMD 8.5 (% length), n 2-4. Chance
   of "consistent" if the sorting index is true with 5+6 above pure by 1 SD: **0.20** (2 SD: 0.49); with no
   effect: 0.04. So "not consistent" has probability 0.80 under a real 1-SD effect and 0.96 under none: a
   likelihood ratio of 0.84, i.e. **almost no evidence either way.**

- **Pre-registered reading: C4-S2b is reclassified "uninformative".** The robustness rule required >= 0.5
  chance of "consistent" at d = 1; it is 0.20. The C4-S2b sentence "the sorting index is not supported as the
  mechanism by this readout" is **withdrawn**: these 20 gastruloids cannot support or contradict it.
- Readout validity (stated in the pre-registration): domain separation of two fate markers exists in any
  polarised gastruloid, pure clones included, so even a well-powered version would test the sorting index only
  indirectly. A direct test needs per-cell clone identity with positions (seqFISH chimeras) or the elongation
  test C4-S2a.
- Process lesson: I reported a descriptive look with n 2-4 as a conclusion without computing its power first
  (H14 applies in spirit to descriptive checks too). Added to the C4 pre-flight habit: compute the chance of the
  predicted pattern before reading any descriptive result.

## C4-S3: pre-flight of the seqFISH chimeras for a direct sorting readout (search, no statistic; 2026-11-07)

- **Serves: C4-S3** (FOCUS.json). Bence: "2" (keep the Aug 2026 mixture plate sealed; look for per-cell clone
  identity with positions). Sources: Dropbox listing (`DATA_MAP_RAJ.md`) and the authors' plot scripts
  (`plot_scripts/SG (Figure 2 and Supplementary)`, R read as text; their CSV outputs not read). No seqFISH
  data downloaded.
- **Pre-flight:**
  1. Unit: gastruloid (one ROI = one gastruloid section).
  2. Readout: per-cell seqFISH counts (~20 genes) + nuclear centroid; clone identity only for **clone 1
     (GFP)**: each DAPI nucleus is called GFP+ if a GFP-segmented nucleus lies within 5 px (`match_gfp_to_dapi`).
     The partner clone is "GFP-" (not separately labelled).
  3. Chimera ROIs with a GFP segmentation on the Dropbox: 01292025 ROI1 (control), 02142025 ROI2-4 (control)
     + ROI5 (RA), 02172025 ROI2 (control), 05232025 ROI2-4 (RA, drug: parked #27), 08182025 ROI4 + ROI6
     (condition not stated in any script). **Usable controls: 5 gastruloids** (+2 of unknown condition).
     The paper's Fig 2E legend gives the chimera controls as 1+3 (n 2) and 1+6 (n 3), matching the 5.
  4. Power:
     - **Graded sorting-index test** (segregation larger in 1+3, E[H] 0.46, than in 1+6, 0.23): 2 vs 3
       gastruloids; the smallest achievable one-sided rank p is 1/10 = 0.10, so **no test can pass**: power 0.
       Cell-level statistics would be pseudo-replication (cells are not independent units).
     - Pure clones in seqFISH have no clone label, so the sorting index (a between-population share) is 0 by
       construction and cannot be measured there.
     - The one question with any power, "at the same A-P position, do clone 1 and partner cells express
       different fate markers?" (clone-intrinsic vs position-set fate), needs 5/5 gastruloids agreeing
       (sign test p 0.031); power 0.59 even if 90% of gastruloids show the effect. Below 0.8, and it is not a
       test of the sorting index.
  5. Exposure: the authors' scripts (plot code, ROI labels); their outputs (`moran_*`, `coexpr_*`,
     `L_dataframe_clonecombo`) not read.
  6. Literature: the paper's seqFISH section (read 2026-11-07).
- **Verdict: fails pre-flight.** The public seqFISH data cannot test the sorting index directly. No download,
  no statistic.
- **Where chapter 4 stands:** every public route to the mechanism is now checked (C4-S1, C4-S2b-R, C4-S3).
  The only powered test left is C4-S2a (rescue on new subclones, A+B, power 0.80), which confirms the
  rescue, not the mechanism. The mechanism ("mixing rescues because sorting can separate head and tail
  programmes") stays a model-level hypothesis, consistent with G2/C3-S2 and untested beyond them.

## C4-S4: the C4-S2a look on mixture partitions A+B (pre-registered 2026-11-07)

- **Serves: C4-S4** (FOCUS.json). Bence: "1" (answer to "Spend A+B on the rescue test, or close chapter 4 with
  the mechanism untested?"). Amendment to C4-S2a, fixed before the look: the look spends **A and B together
  as one look** (5s2+6a n 16, power 0.80 by H14 from counts); **C stays sealed**. Everything else (readout,
  primary T2, pass rule, secondaries, script `c4s2a_morph_confirm.py`, selection/C4-S2a.json) is unchanged.
- **Pre-flight:** as C4-S2a (unit gastruloid/well; Y = 1 - Circularity; used data first: G2-C3-S2; power 0.80
  for A+B; exposure: pure plate means and sd only; no literature needed).
- Oracle change for this: `--partition AB` spends both partitions in one ledger row (`raj_morph_0831:AB`); a
  later look at A or B is refused; C then is the last partition (H13).

### C4-S4 result (2026-11-07; oracle ledger `raj_morph_0831:AB`; C still sealed)

- Gate: H14 power 0.80 (5s2+6a n 16; pure 5s2 14, 6a 10); selection, sealed-data and pre-registration checks
  passed. One oracle bug fixed before the look (a union was treated as the last partition; commit before the
  look).
- **Primary T2 (5s2+6a vs mean of its pure members, Y = 1 - circularity): +0.021, one-sided 95% lower bound
  -0.037: FAIL.** Standardised +0.25 SD against the predicted +0.81 SD. Rough SE 0.035 (from the bound), so the
  predicted size (0.068) is about 1.4 SE above the estimate: not excluded, not supported.
- Secondary (described): 5s2+6a vs its better member (5s2) -0.045 [-0.128, 0.041]; bulk+5s2 vs members
  +0.017 [-0.069, 0.119]; **bulk+6a vs members +0.077 [0.016, 0.141]**; T1 (sorting-index contrast within the
  mixture plate) -0.026 [-0.107, 0.066].
- Pure plate (train): bulk 0.222, 5s2 0.174, 6a 0.042 (6a forms round gastruloids, as the authors' README
  says).
- **Pre-registered reading: rescue not confirmed on new subclones.** The complementary 5s2+6a mix is not more
  elongated than the average of its members, and not more than 5s2 alone. The only positive contrast is
  bulk+6a (a severely defective clone helped by bulk), which fits "a defect is partly complemented by a
  diverse partner" more than "two complementary clones rescue each other". Disclosed confound: pure and mixed
  gastruloids are on different plates; a plate shift could hide or create T2.
- **Stopping rule applied:** #22's complementary-pair rescue is not confirmed out of sample; chapter 4's
  mechanism work stops at the descriptive level. No re-test on A or B. C stays sealed for any future design
  that can remove the plate confound (same-plate pure and mixed wells).

## C5-S1: chapter 5 (clone memory) pre-flights (search and counts only, no statistic; 2026-11-07)

- **Serves: C5-S1** (FOCUS.json). Bence: "1 and carefule with sealed stuff". Chapter question: is clone memory
  one mechanism (event-triggered loss + stable residue), shared across systems, that predicts the fade of an
  unseen series (the G8 failure)?
- **Pre-flight:** unit / readout / used data / power / exposure / literature are given per candidate below.
- **Sealed memory data inventory:** the only untouched memory file is the **Raj reserve** `clone3_pass3_07112024.xlsx`
  (last Raj sealed file; H13, Bence's OK). G6a/G6b/G7b are spent. Nothing was opened.

| candidate | unit, n | readout | power / verdict |
|---|---|---|---|
| **A. GSE291244 lineage memory** (Cell 2026 two-epoch "tree of trees"; 8 deep trees) | gastruloid (two-epoch, 88-106) / founder (deep trees, 3) | composition vs lineage distance | **fails** (unchanged from the 2026-11-07b pre-flight): cell-to-gastruloid table not public; authors' own effect rho 0.052 over 3,828 pairs, too small for any sealed partition (~18 gastruloids); deep trees = 3 founders. Rebuilding the assignment would not fix the power. |
| **B. Raj ATAC-seq over passages** (`raw_data/ATAC`, sample sheet `extracted_data/ATAC/alignments_clones_gastr_101214.txt`, read) | **clone**: 7 clones + bulk; passages 0, 1, 2, 3; 1-2 replicates; **51 samples** (13 of 64 dropped by the authors) | per-sample accessibility (bigwig 5 GB, or MACS2 peaks; BAMs 69 GB not needed) | **passes structure.** The memory question in chromatin: do clone-specific accessibility differences fade over passages, at the rate the A-P propensity fades (G3 retention 0.60 per passage), with a residue? Never analysed by us; the authors ran per-clone p1 vs p3 contrasts (not read). Power depends on an effect we have no prior for; with clone as unit (n 7) a paired one-sided test needs d >= about 1.2 for 0.8. |
| **C. Shape of memory on used passage data** (G3 1+3 pairs, passages 1-4) | gastruloid, 13-32 per passage, one series | S per passage (g3_fade.json) | **fails alone**: one-rate (r 0.6) vs plateau differs mainly in passage 4 vs 3; difference 0.032 against SE 0.020, power about 0.47; one series, so series effects (G8) cannot be separated. |
| Blood (LARRY, chapters 1-2) | clone | fate memory over 2 days | already fitted (core_larry RHO_F); serves the "shared across systems" comparison as a fixed reference, no new test |

- **Verdict:** only **B** can carry chapter 5. Proposed design for C5-S2 (needs Bence's OK; nothing downloaded):
  1. **Seal by passage before any statistic:** train = passages 0-1 (open), sealed = passages 2 and 3 (two
     partitions, held by sample ID in sealed.py; passage 3 last, H13). This is a forecasting split: memory is
     about the future, so the model is fitted on early passages and must predict later ones.
  2. **Download** the 51 bigwigs (about 5 GB; D: has 445 GB free) into a new folder; lock passage-2 and passage-3
     files by hash on arrival, before reading any value.
  3. **Train-only work:** per clone, clone-specific accessibility (distance to bulk at the same passage, at
     consensus peaks) at passages 0-1; replicate noise; fit one-rate and two-part fade; power for the passage-2
     forecast from train noise and counts (H14). Look only if power >= 0.8.
  4. Cross-modal check (described): does chromatin distinctness track the A-P propensity strength of clones 1, 3,
     5, 6, 7 (C3-S1 z)?
- Guard note: the scope guard blocked reading `results_gastruloid/g3_fade.json` with Python because the name
  matches a script; a false positive (reading results is not running an experiment). To fix: match only
  `<name>.py` invocations.

## C5-S2: one memory law across blood and gastruloid clones? (pre-registered 2026-11-07)

- **Serves: C5-S2** (FOCUS.json). Bence: "1 / and still keep in mind the Raj lab's chromatin (ATAC-seq) data".
  Chapter 5 question: is memory one mechanism across systems? This step writes the law into `core_model.py` and
  checks it against already-used estimates only (no sealed data; no new data).
- **Pre-flight:**
  1. Unit: clone (blood: sister wells of a LARRY clone; gastruloid: subclone / clone x passage).
  2. Readout: published-in-log estimates with CIs: E19 / E19c (LARRY cross-well fate sharing, by clone size),
     E41b (day 4 vs 6), B38 / #13 (SIS-seq heritable share), G5 / #25 (subclone polarity ratios), G3 (passage
     retention), C3-S1 (rho_P, rho_g).
  3. Used data only; nothing recomputed from sealed files.
  4. Power: an interval-consistency check, no new test; resolution limited by the stated CIs.
  5. Exposure: all inputs are known numbers; this is a consistency check, not a confirmation.
  6. Literature: none new (Naik 2013, Nat Commun 2026 HSPC division memory in DISCOVERIES #6).
- **The law (core_model.memory_through):** heritable state z; each division copies it with retention rho_div
  (correlation with the mother, mean shrink sqrt(rho_div)); each event of type e (blood well split, ES passaging)
  copies it with rho_e; the context (well, gastruloid) adds non-heritable noise (rho_ctx), which is not passed on.
- **Checks (interval logic; division counts are assumptions, varied):**
  - **K1, per-division retention shared?** Blood: E19c large-minus-small cross-well sharing (0.82 vs 0.95,
    difference -0.13 [-0.25, -0.01]) read as extra divisions after the split (small clones 2-4 cells, about 1.5
    divisions; large 12-143, about 5.5; difference d = 4, sensitivity 3-6): rho_div = (r_large / r_small)^(1/d).
    Gastruloid: G5 passage-1 subclone / parent polarity 1.21 [1.005, 1.50] after >= 13 divisions:
    sqrt(rho_div)^13 >= lower bound. **Pass** = the two implied rho_div intervals overlap.
  - **K2, passaging retention one value?** G3 r 0.60 [0.50, 0.68] per passage; G5 passage 2 / passage 1 0.37
    [0.23, 0.51]; C3-S1 sqrt(rho_P) 0.46 (point). **Pass** = a common value inside all intervals.
  - **K3, blood split events:** LARRY cross-well 0.88 [0.82, 0.92] vs SIS-seq heritable share 0.71 [0.59, 0.78]
    (different readouts and culture lengths): described only, no pass rule (not comparable without division
    counts for SIS).
- **Reading:** K1 and K2 pass = one law (no per-division loss, event-specific loss) holds on used data; K1 fails =
  per-division memory differs between committed blood progenitors and pluripotent cells (the law needs a
  cell-state-dependent rho_div); K2 fails = passaging loss is series-specific (as G8), the law cannot predict a
  new series' fade without a series covariate.
- **Script:** `c5s2_memory_law.py` (reads numbers recorded in this log and result JSONs of used steps only).
- **Stopping rule:** whatever the reading, the law (with its failures as stated limits) goes into core_model.py;
  the sealed confirmation (C5-S3, Raj reserve) and the chromatin line (C5-S4, Raj ATAC) are separate steps
  needing Bence's OK.

### C5-S2 result (2026-11-07; `results_gastruloid/c5s2_memory_law.json`; used estimates only)

- **Law in the core model:** `core_model.memory_through(z, n_div, events, rho_div, rho_event, rng)`; simulated
  check reproduces E19c's large/small sharing ratio (0.867 vs 0.863) with rho_div 0.964 over 4 divisions.
- **K1, one per-division retention: FAIL.** Blood (LARRY E19c, d = 4 extra divisions): rho_div 0.964 [0.927,
  0.997] (d 3-6: 0.95-0.98, upper bounds 0.996-0.998). Gastruloid subclones (G5): the passage-1 ratio's lower bound
  (1.005) is above 1, so no per-division shrink at all is allowed (rho_div >= 1.00). The intervals do not overlap
  for any d. Caveats: E19c's size gradient is confounded with time (noted in E19c); G5's ratio > 1 may include
  selection of strongly polarised subclones and a parent value from another series.
- **K2, one passaging retention: FAIL.** G3 [0.50, 0.68] and G5 [0.23, 0.51] overlap only at 0.50-0.51; C3-S1's
  0.46 lies outside. Passaging loss differs by series (as G8).
- **K3 (described):** SIS-seq heritable share 0.71 [0.59, 0.78] is below LARRY's 0.88 [0.82, 0.92], the direction a
  per-division loss over SIS's longer culture would give; not testable without SIS division counts.
- **Pre-registered reading:** one memory law does not hold on used data. **Committed blood progenitors lose a
  little state per division (about 3-4%); pluripotent clones lose none while expanding and lose about half at each
  passage, at a series-specific rate.** The law stays in core_model.py with rho_div and rho_event as
  cell-state-specific parameters (stated limit), not one shared value.
- **What it means for the chapter:** memory is not one mechanism across systems in its simplest form. The open
  part is the passaging loss in ES cells: what it is (chromatin?) and what sets its series-specific rate. That is
  exactly what the Raj ATAC data over passages (C5-S4) can address; the Raj reserve (C5-S3) can only confirm the
  passage-3 level of one series.
- Guard fix (2026-11-07): the scope guard now counts a script only when the command runs it (`<name>.py`,
  `import <name>`, `-m <name>`), so reading a result file such as `results_gastruloid/<name>.json` is no longer
  blocked. My first edit wrote literal backspace characters into the regex and briefly disabled blocking; caught by
  a two-way test (result read allowed, off-focus run blocked) and repaired before any other command ran.

## C5-S4: does a clone's chromatin distinctness fade over passages like its A-P propensity? (pre-registered 2026-11-07)

- **Serves: C5-S4** (FOCUS.json). Bence: "1" (chromatin line). Chapter 5 found (C5-S2) that ES clones keep their
  state while expanding and lose about half per passage at a series-specific rate; this step asks whether the
  chromatin of a clone loses its clone-specific part at passaging too, and at what rate.
- **Pre-flight:**
  1. Unit: **clone** (7 clones; bulk as reference only). Samples: clone x passage x replicate (sample sheet
     `alignments_clones_gastr_101214.txt`: 51 samples; passage 0 has 1 replicate for clones 1, 2, 4, 6, 7 and none
     for 3, 5; passages 1-3 mostly 2).
  2. Readout: the authors' MACS2 peaks after duplicate removal (`extracted_data/ATAC/MACS2_output/
     Oct1224_shifted_duprem/*.narrowPeak`, 51 files, 1.7 GB, text). Per sample, the **top 20,000 peaks by signal
     value**, mapped to 500-bp genome bins (depth-robust: a fixed number of peaks per sample). Samples with fewer than
     20,000 peaks: QC fail, excluded (decided from peak counts only).
  3. Used data first: none of this is used; the authors' contrasts (Homer p1 vs p3, DESeq2 AvP / AvBulk) are not
     read.
  4. Power: H14 gate from train-derived noise (below); look only if power >= 0.8.
  5. Exposure: sample sheet only; the paper reports peak numbers (44,540 at passage 1, 41,593 at passage 3) and
     shared/unique shares between anterior and posterior clones at passage 1.
  6. Literature: paper's ATAC section (read 2026-11-07).
- **Partitions (by passage; a forecasting split, fixed now):** passages 0-1 = **train** (open); passage 2 =
  **sealed A**; passage 3 = **sealed B (last; H13: Bence's OK)**. Files of passages 2-3 are locked by sha256 in
  sealed.py on arrival, before any value is read, readable only through the oracle.
- **Statistic.** For clone c at passage p: J_rep(c,p) = Jaccard between its replicates' top-peak bin sets (noise
  ceiling); J_btw(c,p) = mean Jaccard between c's replicates and every other clone's replicates at the same passage.
  **Clone-specific signal S(c,p) = J_rep - J_btw** (how much more a clone resembles itself than other clones).
  Memory fading predicts S(c,p) shrinking with passage.
- **Train work (passages 0-1 only):** S(c,1) for 7 clones; passage 0 single replicates give J_btw only, so the
  train fade estimate uses D_btw(c,p) = 1 - J_btw(c,p) for p = 0, 1 (clones 1, 2, 4, 6, 7). Candidate models for the
  passage-2 forecast, scored on train (H12; selection/C5-S4.json): (M1) S(c,2) = 0.6 x S(c,1) (fades at the
  propensity rate, G3); (M2) no fade, S(c,2) = S(c,1); (M3) the train p0->p1 ratio of D_btw. Chosen = best train
  fit of p1 from p0.
- **Primary test (one look, partition A = passage 2):** R = mean_c S(c,2) / mean_c S(c,1) over clones with >= 2
  QC-passing replicates at both passages. **Fading:** one-sided 95% upper bound of R (bootstrap over clones) < 1.
  Reported against the chosen model's forecast. Power gate: sd of per-clone log change from train (p0->p1 D_btw
  log ratios), effect log(0.6), n = clones with replicates at passage 2 (counts only).
- **Secondary (described):** does S(c,1) rank the clones like their A-P propensity strength |z| (C3-S1; clones 1,
  3, 5, 6, 7)? Spearman, 5 points, no claim.
- **Script:** `c5s4_atac_memory.py` (train functions; sealed reads only in `test(part)` via test_oracle).
- **Stopping rule:** fading confirmed = chromatin carries the passaging loss (mechanism candidate for C5); not
  confirmed = the loss is not visible in peak sets at this resolution; partition B stays sealed either way unless
  Bence OKs it.

### C5-S4 train result (2026-11-07; passages 0-1 only; passages 2-3 locked by sha256 before any read, a84436f)

- Data: 51 duprem narrowPeak files (Dropbox folder zip downloaded by Bence after the scripted download hit a
  captcha page; sizes match the listing exactly). All samples pass QC (>= 20,000 peaks).
- Passage 1: S = J_rep - J_btw > 0 for all 7 clones (0.089-0.269; clone 3 highest, 0.269). Clones are
  chromatin-distinct one passage after cloning.
- Passage 0 -> 1 (clones 1, 2, 4, 6, 7): between-clone distance D_btw fell in all 5 (log ratios -0.11 to -0.36;
  mean ratio 0.83). Caveat: passage 0 has single replicates, so part of the drop may be technical (noisier
  passage-0 libraries) rather than convergence.
- Selection (H12): M3 train ratio (LOO -SSE -0.016) > M2 no fade (-0.068) > M1 G3 rate 0.6 (-0.095). Forecast R = 0.83.
- Secondary: Spearman S(c,1) vs |z| (5 clones) = -0.10. Described only.
- H14 gate (counts only): 6 clones with 2 replicates at passage 2 and at passage 3; power 1.00 with the
  pre-registered sd (0.098, from D_btw log ratios).
- **Look NOT taken.** Validity concern: the gate's sd comes from D_btw, but the test statistic is a ratio of S
  (a small difference, between-clone CV about 0.5 at passage 1). If S at passages 1 and 2 were uncorrelated
  across clones, sd of log R would be about 0.29 and power about 0.55 at effect log 0.6. At the chosen forecast
  (0.83), power would be about 0.1. Waiting for Bence before spending partition A.

### C5-S4 power amendment (2026-11-07, before any look; Bence: "1")

- **Serves: C5-S4.** Why: the pre-registered gate used the noise of log D_btw (sd 0.098), but the test statistic
  is R = mean S(c,2) / mean S(c,1) with a clone bootstrap. S is a small margin, and its between-clone spread is
  much larger (passage-1 CV about 0.5).
- **New noise model (train only, `c5s4_atac_memory.py --power`):** log S(c,1) is taken from the observed passage-1
  values of the 6 testable clones (1, 2, 3, 4, 6, 7). Simulated passage-2 values:
  log S(c,2) = log S(c,1) + log R_true + e, where e ~ N(0, sd_e^2(1 - rho^2))... implemented as a bivariate
  lognormal with the train log-sd for both passages and clone correlation rho. For each scenario, 1,000
  simulated data sets are run through the exact test (2,000-draw clone bootstrap, one-sided 95% upper bound < 1).
- **Gate (decided now):** look at A only if power >= 0.8 at R_true = 0.6 with **rho = 0** (conservative: no
  assumed clone-to-clone persistence of distinctness). Also reported, not gating: power at rho 0.5 and 0.8, at
  R_true 0.83 (the train forecast), and the rho at which power reaches 0.8.
- The selection power block is updated to sd = sd(log R | rho 0) x sqrt(6), so the oracle's analytic gate
  matches the simulation. The old block stays in git history (2bd278b).
- If the gate fails, A stays sealed. C5-S4 is then closed as underpowered at clone level, unless Bence chooses
  otherwise.
- **Power amendment result (train only):** log-sd of S(c,1) over 6 clones = 0.40. Simulated power of the exact test:

| R_true | rho 0 | rho 0.5 | rho 0.8 | rho 0.9 |
|---|---|---|---|---|
| 0.60 | **0.77** | 0.95 | 1.00 | 1.00 |
| 0.83 (train forecast) | 0.29 | 0.39 | 0.61 | 0.83 |

  **Gate fails (0.77 < 0.8 at R 0.6, rho 0). Partition A NOT spent.** The oracle's power block is updated to
  sd 0.573, and the oracle now refuses as well (analytic power 0.70). A and B stay sealed.

## C5-S5: do subclones carry their parent clone's expression signature through a single-cell bottleneck? (pre-registered 2026-11-08)

- **Serves: C5-S5** (FOCUS.json; Bence: "download all the data for possible other candidates and reinvestigate best
  candidate"). Script: `c5s5_subclone_expr.py`.
- **Candidate ranking (why this one):**
  - The Raj store's other memory-carrier candidates are bulk RNA, passage 1 only (VA_1..18), subclone RNA
    (F8MWB5), and Wnt line scans.
  - Only the subclone set tests memory across an event: a single-cell bottleneck plus the passages since.
  - Wnt, Incucyte, seqFISH and IF measure readouts, not carriers, and were not downloaded.
- **Pre-flight:**
  1. Unit: **subclone** (n = 3: 1s2, 5s2, 6a). Parents: clones 1, 5, 6 among 7 candidate clones.
  2. Readout:
     - Test set: within-batch signature = log2 CPM(subclone) - log2 CPM(bulk), mean of 2 replicates each,
       from F8MWB5 (48 h, Mar 2026).
     - Train set: parent signature = log2 TPM(clone) - log2 TPM(bulk reference), passage-1 ESC (kallisto,
       VA_1..18), summed to gene level.
     - Genes: the train-defined panel, the 2,000 most variable genes across the VA clone means (expressed:
       mean TPM >= 10), mapped by gene id.
     - Failed libraries are excluded on QC already read: VA_1 (0.75 M reads) and VA_11 (23% aligned).
     - Bulk reference: VA_2, VA_5, VA_9 (bulk + rainbow).
  3. Used data: none. G5 (subclone polarity kept, ratio 1.21) is phenotype, not expression.
  4. Power: from train only, by split replicates in VA (below).
  5. Exposure: file names, methods text, mapping stats. The authors' PCA and heatmap are locked and unopened.
  6. Batch / state confound: the test is 48 h (likely pre-CHIR aggregates), the train is ESC, and the
     batches differ (2025 vs 2026). Within-batch signatures relative to bulk cancel batch offsets, not
     state changes; a state shift weakens matching and does not create false matches.
- **Statistic:**
  - For each subclone, the Spearman correlation of its signature with each of the 7 clone signatures.
  - rank_s = rank of its parent among the 7 (1 = best).
  - T = sum of the 3 parent ranks.
  - p = P(T <= observed) under independent uniform ranks (exact, 343 cases; minimum p = 1/343 = 0.003).
- **Primary (one look, the last of its set; needs Bence's verbatim OK, H13):** clone memory in expression is
  confirmed if p < 0.05 (T <= 5 of 3..21).
- **Secondary (described):**
  - Does each subclone rank its parent above the parent's propensity-group mates? Posterior: 1, 6; anterior:
    3, 5, 7. That separates clone-specific memory from A-P-state memory.
  - Mean parent correlation vs mean non-parent correlation.
- **Train power (H14 analogue, decided now):**
  - In VA, split replicates for clones 1, 2, 3, 5, 6, 7 (2 good libraries each).
  - Split 1 signatures use VA_5 as bulk; split 2 use mean(VA_2, VA_9). Independent bulk references avoid
    shared-noise matches.
  - Self-match rank of each clone's split-1 signature among the 7 split-2 signatures, with clone 4 (VA_12)
    included as a candidate in split 2.
  - Power = simulated P(p < 0.05) when 3 subclones draw ranks from the empirical self-match rank
    distribution, degraded by a state shift. A pessimistic version uses the train correlation gap halved.
  - **Look only if power >= 0.8 under the halved gap.**
- **Stopping:** confirmed = memory is carried in expression through a bottleneck (C5 mechanism). Not confirmed
  = not visible at 48 h. Underpowered = no look; the set stays locked.

### C5-S5 train result (2026-11-08; VA only; subclone matrix locked and unread, 12b369d)

- Panel: the 2,000 most variable expressed genes. Transcript-to-gene mapping from Ensembl 112 cDNA headers.
- Split-replicate self-match (same batch, independent bulk references): self rank 1 for clones 1, 6, 7; rank 2
  for clone 5; rank 3 for clone 3; rank 7 for clone 2 (its replicates anti-correlate, -0.73; clone 2 split 1
  matches clone 4 best, +0.60). Self-minus-other correlation gap: median 0.41.
- Exact test: T <= 5 needed (p 0.029). Simulated power: **0.41** at the train gap, **0.19** at half the gap.
- **Gate fails (needs 0.8). No look; the subclone matrix stays locked.** Caveat, not used to override: the
  real test averages 2 replicates on each side, so single-replicate splits understate the signal. Even so, the
  unshrunk power is 0.41, and the test also crosses batch and cell state (ESC vs 48 h).
- Reading: clone identity in bulk expression is not reproducible enough at passage 1 (1-2 shallow libraries,
  2 failed) to carry a 3-subclone memory test. Possible clone 2 library issue (rep 1 matches clone 4) noted,
  not acted on.

## C5-S6: does a clone's position on the passage-1 A-P chromatin axis fade over passages? (pre-registered 2026-11-08)

- **Serves: C5-S6** (FOCUS.json; Bence: "1", i.e. pre-register the A-P-axis chromatin test and run its power gate
  on passages 0-1). Script: `c5s6_ap_axis.py`.
- **Why (replaces C5-S4 on the same partitions):** C5-S4 measured how distinct a clone's chromatin is overall.
  The memory in question is A-P propensity, so the sharper readout is signed: where each clone sits on an
  anterior-vs-posterior chromatin axis. If C5-S6 passes its gate, the C5-S4 look is dropped (one look per
  partition).
- **Pre-flight:**
  1. Unit: **clone** (anterior 3, 5, 7; posterior 1, 6; groups from C3-S1 propensity z). Clones 2, 4 have no
     propensity and are not used in the axis or the statistic.
  2. Readout:
     - Per sample: the C5-S4 summit-bin set (top 20,000 peaks by signalValue, 500-bp summit bins; samples with
       fewer than 20,000 peaks fail QC).
     - Axis folds: passage-1 replicate index k = 1, 2. Fold-k axis is built from the passage-1 samples with
       replicate != k (one sample per A-P clone).
     - A-bins = bins present in all 3 anterior samples and absent from both posterior samples; P-bins the
       reverse. No tuning parameter.
     - Score(sample) = fraction of A-bins present - fraction of P-bins present.
     - Passage-1 samples are scored only with the axis of the other fold (cross-fit, so no selection bias).
       Passage-0 and sealed samples are scored with the mean of both fold axes.
     - Clone score = mean over its QC-passing samples. Gap D(p) = mean anterior clone score - mean posterior
       clone score. A global shift at a passage cancels in D.
  3. Used data: passages 0-1 (C5-S4 train). Sealed passage 2 (A) and passage 3 (B) are unread.
  4. Power: train only (below), decided before the look.
  5. Exposure: sample sheet (clone, replicate, passage per file). The authors' p1-vs-p3 Homer and differential
     tables stay unopened.
  6. Technical confound: a passage-wide drop in library quality would shrink D and look like fading. Control at
     the look (pre-registered, described): mean replicate Jaccard over all replicate pairs, including bulk
     (no clone memory), at passage p vs passage 1. A drop of more than 10% flags the result as possibly
     technical.
- **Statistic (partition A, passage 2):**
  - R2 = D(2) / D(1). Inference treats the 5 clones as fixed (the question is whether these clones' A-P
    chromatin difference fades, as G3 was about these clones), with replicate noise only.
  - Var(clone score) = pooled within-clone replicate variance at that passage / n replicates.
    Var(D) = sum over each group of Var(clone)/|group|^2. SE of log R2 by the delta method.
  - **Primary (one look at A): fading confirmed if the one-sided 95% upper bound of R2 < 1.**
  - Secondary (described): R2 vs the G3 forecast 0.60 (inside or outside the 90% CI); per-clone signed
    deviation from the group midpoint at passages 1 and 2; the quality control above.
- **Passage 0 (train, described only):** D(0) on the matched set {7} vs {1, 6} (single replicates), and the
  ratio D(1)/D(0) on the same set.
- **Train power (H14, decided now):**
  - From passage 1: true clone scores = observed cross-fit clone means; replicate sd s = pooled within-clone sd.
  - Simulate passage 1 (2 replicates per clone) and passage 2 with the sheet's replicate counts (clone 5: 1;
    others 2), true passage-2 deviations from the midpoint = R x passage-1 deviations.
  - Power = P(upper bound < 1), 4,000 simulations, at R = 0.6 and 0.83, with s and 1.5 s.
  - **Look at A only if power >= 0.8 at R = 0.6 with 1.5 s** (pessimistic: s comes from 5 df).
- **Partition B (passage 3, last, H13):** not requested now. A pooled slope over passages 1-3 would need its
  own pre-registration and Bence's verbatim OK.
- **Stopping:** confirmed = chromatin carries the fading A-P memory. Not confirmed with power = the A-P chromatin
  difference persists while propensity fades (chromatin is not the fading carrier). Gate fails = no look; A and
  B stay sealed.

### C5-S6 train result (2026-11-08; passages 0-1 only; A and B sealed and unread)

- Axis: 249-342 A-bins and 293-296 P-bins per fold.
- Cross-fitted passage-1 clone scores: clone 3 0.54, clone 5 0.42, clone 7 0.41 (anterior); clone 1 0.13,
  clone 6 0.02 (posterior). D(1) = 0.38, SE 0.069 (replicate sd 0.106, 5 df). The A-P chromatin axis is real
  at passage 1 out of fold (about 5.6 SE).
- Power of the pre-registered ratio test at A: **0.14** at R = 0.6 with 1.5 s (gate); 0.38 at R = 0.6 with s;
  0.17 / 0.09 at R = 0.83. **Gate fails. No look; A and B stay sealed.** Reason: R = D(2)/D(1) carries the
  noise of both gaps over 5 clones with 2 replicates (clone 5 has 1 at passage 2).
- Passage 0, described only (single replicates, matched set {7} vs {1, 6}): D(0) = 0.05 vs D(1) = 0.34 on
  the same set; clone 6 (posterior) scores above clone 7 (anterior) at passage 0. Taken at face value the axis
  is weaker at passage 0 than at passage 1, the opposite of a fade from passage 0. Not used: single
  replicates, possible batch difference (C5-S4 caveat), 3 clones.

### C5-S6 amendment: pooled passages 1-3 fade contrast (pre-registered 2026-11-08, before any power number)

- **Serves: C5-S6.** Bence: "alright next steps cellware" (after the proposal: compute the passages 1-3 pooled
  power on train first). Train-only power; A and B stay sealed.
- **Why:** the ratio test at A alone has power 0.14. Under the G3 fade the gap should be 0.6 x D(1) at passage 2
  and 0.36 x D(1) at passage 3, so a contrast using both sealed passages is much stronger.
- **Statistic (would spend A and B together, one look; B is the last partition, so the look needs Bence's verbatim
  OK, H13):**
  - Same readout, axes and clone scores as C5-S6. D(p) for p = 1, 2, 3 on the 5 A-P clones.
  - L = D(1) - (0.385 D(2) + 0.615 D(3)). Weights are proportional to the G3-expected drop (0.4, 0.64) over
    equal variances, normalised to 1; fixed now.
  - SE(L) from the per-passage SE(D) (pooled within-clone replicate variance per passage, fixed clones).
  - **Primary: fading confirmed if L / SE(L) > 1.645 (one-sided 5%).**
  - Secondary (described): per-passage retention r from a weighted fit of log D on passage (if all D > 0), with
    90% CI vs G3 0.6; the replicate-Jaccard quality control (all replicate pairs incl. bulk) at passages 2 and 3
    vs 1.
- **Gate (same rule as C5-S6):** power >= 0.8 at a true fade of 0.6 per passage with 1.5 s, 4,000 simulations,
  sheet replicate counts (passage 2: clone 5 has 1; passage 3: 2 each). Also reported: s x 1, and fade 0.83.
- **If the gate fails:** no look; chapter 5 chromatin work stops at the train result.

#### C5-S6 amendment result (2026-11-08; train only; A and B sealed and unread)

- Power of L / SE(L) > 1.645: **0.53** at fade 0.6 with 1.5 s (gate); 0.79 with the measured s; 0.34 / 0.21 at
  fade 0.83.
- **Gate fails. No look; A and B stay sealed.** Per the pre-registration, chapter 5 chromatin work stops at the
  train result: the A-P chromatin axis exists at passage 1 (D 0.38, ~5.6 SE out of fold); whether it fades is
  untestable with 5 clones x 2 replicates.

### C5-S6b: a less noisy A-P chromatin readout (pre-registered 2026-11-08, train only)

- **Serves: C5-S6.** Bence: "no we need to investigate more" (after both C5-S6 gates failed on noise).
- **Why:** the gates fail because the replicate noise (s = 0.106) is large against the gap (D = 0.38). The
  C5-S6 readout is binary presence over about 300 strict bins per fold; a continuous, wider readout should
  cut s. This changes the readout only; the tests, weights and gates stay as pre-registered.
- **Pre-flight:** unit clone (3, 5, 7 vs 1, 6); data passage 1 only (cross-fit by replicate fold), passage 0 not
  used for selection; A and B unread; power from train only.
- **Candidates (fixed now):**
  - R0: C5-S6 readout (binary, strict A-only / P-only bins).
  - R1: binary presence, all bins with |fraction anterior - fraction posterior| >= 0.5 in the fold, weighted by
    that difference.
  - R2: continuous. Each sample's peaks (all, not top 20,000) mapped to 500-bp summit bins; value = within-sample
    percentile of signalValue (0 if absent). Fold weight w = mean anterior - mean posterior clone value; score =
    sum(w x) / sum|w| over the 2,000 bins with largest |w|.
  - R3: R2 with 5,000 bins.
- **Selection score:** cross-fitted passage-1 D / SE(D) (largest wins). Then the pooled contrast power is
  recomputed with the winner, same simulation and **same gate (0.8 at fade 0.6 with 1.5 s)**.
- **Optimism guard:** 4 candidates picked on the same 10 samples. Reported: the winner's margin over R0, and the
  gate result. If the winner passes only narrowly (< 0.85), report as marginal and ask Bence before any look.

#### C5-S6b result (2026-11-08; passage 1 only; A and B sealed and unread)

- Cross-fitted D / SE: R0 5.59 (D 0.38, s 0.106); R1 2.43 (D 0.12, s 0.078); R2 0.09; R3 0.07. The continuous
  readouts lose the axis: with one sample per clone per fold, the top-|w| bins are dominated by single-sample
  intensity noise.
- **Winner R0 (the C5-S6 readout); pooled power unchanged, 0.53 at the gate. Gate fails; no look.**
- Diagnostic (described, no rule derived): the replicate noise is spread, not one bad library. Largest splits:
  clone 1 (0.01 vs 0.25) and clone 5 (0.33 vs 0.51). Peak counts range 75k-391k with no clean outlier.
- Reading: the limit is the data (5 clones x 2 replicates), not the readout.

### C5-S6c: public data search for a stronger clone-over-passages set (2026-11-08; search only, nothing downloaded)

- **Serves: C5-S6.** Bence: "Search public data for a stronger set. That means clone-resolved chromatin or
  expression measured over passages, with more clones or replicates."
- **Best design match, data not public:** Linneberg-Agerholm et al. 2026 bioRxiv 10.64898/2026.09.02.748863
  (hESC, lentiviral barcodes, about 981 analysed clones, same-well vs sister-well split at passage 2, scRNA at
  hESC p2/p3, primitive streak, definitive endoderm; multiome at p3/PS/DE, 1,333 clones). Sister agreement
  0.77 vs 0.54 random at p3 (n 343); mesoderm-biased clones carry pre-existing accessibility at mesoderm
  enhancers (LRP6, Wnt). Decay over passages not quantified. Data "deposited to GEO, public at publication":
  no accession yet.
- **Weaker fits:** scRIPT (Cetin et al. 2026, 10.64898/2026.10.14.759042: mESC, barcoding D0 to D7 of
  differentiation, not passages; fate-independent clonal memory enriched in imprinted genes). Mold et al. 2024
  Cell Systems (human T-cell clones, RNA + ATAC, over a year in vivo; GEO not found; a different system). hiPSC
  ClonMapper cardiac preprint 10.64898/2025.12.12.694049 (engineering, not passage-resolved molecular data).
- **Reading:** no public dataset beats the Raj ATAC set for clone-resolved chromatin over passages in
  pluripotent cells. The hESC set would (hundreds of clones, sister splits, multiome) once released.

### C5-S7 pre-flight: unpark G13 DiffAssay as a memory-carrier test? (2026-11-08; no data read)

- Bence: "Unpark DiffAssay". Pre-flight (H15) on what is already recorded (G13, 2026-11-07):
  1. **Readout: correction.** DiffAssay is 2D confluency (cell growth) of 9 lines in neural, mesoderm,
     endoderm and 2i media at 4 times. It has no differentiation or marker readout, so "differentiation tempo"
     (my C5 candidate wording) is not measured; only growth tempo per medium is.
  2. Unit: clone with A-P preference, n = 5 (1, 3, 5, 6, 7). Exact Spearman over 120 orders: only a perfect
     ordering gives p < 0.05.
  3. Used data: replicate 1 already used (G13): growth bias vs A-P preference rho -0.70 / -0.10 / +0.70 by
     measure, no anterior-posterior separation; clone 3 and clone 6 look alike on every measure.
  4. Power: even at the replicate-1 rho of 0.7 taken as true, a perfect ordering in replicate 2 is unlikely;
     a 5-clone single-replicate look cannot reach 0.8.
  5. Sealed: replicate 2 is the only and last partition (H13).
  6. Memory angle: "clone 3, late" (a later passage of clone 3) is the only over-time sample: n = 1, no test.
- **Pre-flight fails. Not unparked; replicate 2 stays sealed.** FOCUS unchanged.

## C6-S1: chapter 6 (cells into space) plan and pre-flight (counts only, no statistic; 2026-11-08)

- **Serves: C6-S1** (FOCUS.json). Bence: "alright now lets go with your first choice for next cpt / 1. Cells into
  space (my pick)." Plan: `CHAPTER6_PLAN.md`.
- **Pre-flight:**
  1. Unit: gastruloid.
  2. Readouts: line-scan polarity (G1a), 1 - circularity (C4-S2a), geodesic aspect ratio (authors' chimera metric).
  3. Used data first: all fit targets are open or spent (G1a fit, G1b, G2, G3, G7/G10, 0831 pure + A+B, chimeras).
  4. Power from counts: S1 Jan 2024 1+6 passage 4 (34 gastruloids): one-sided binomial 0.50 / 0.69 / 0.91 at true
     share 0.66 / 0.70 / 0.76. S2 morph 0831 C (6a+bulk 13, 5s2+6a 11, 5s2+bulk 6): 6a+bulk over member mean 0.39 /
     0.66 / 0.84 at +0.04 / +0.06 / +0.077. Final gate after the model is fixed (H14).
  5. Exposure: the clone+bulk chimera set (175 gastruloids) is exposed (authors' README results read 11-07; one data
     row of `5+bulk.csv` printed by a header check on 11-08). It is relabelled open, for fitting only, never a test.
     S1: tag counts + channel names only. S2: counts only.
  6. Literature: Oriola 2024, Mongera 2018, monoclonal and chimeric gastruloids; rule-3 methods to read before coding.
- **Constraint carried in:** complementary clone-clone rescue (#22) failed out of sample (C4-S4), so the model gets no
  rescue term. Mixing effects must emerge from the local rules.
- **Result:** the plan is written. No statistic has been computed and no sealed values have been read. Next: C6-S2
  simulator build, with its own pre-registered checks.

## C6-S2: 2-D cell-level gastruloid simulator on the core model; build, smoke tests, qualitative checks (pre-registered 2026-11-08)

- **Serves: C6-S2** (FOCUS.json; CHAPTER6_PLAN.md). Bence: "yes and after put it to sleep" (go for the C6-S2 build).
  Script: `c6s2_space.py` (new). It reuses `core_model.py` (state_copy, clone states z and rho_g from C3-S1) and the
  readout helpers in `g5_gastruloid.py` / `g1_gastruloid.py` (polarity from a profile) and `g1b` (boundary
  crossing). No sealed file is read.
- **Pre-flight:**
  1. Unit: simulated gastruloid. Data targets are per-clone or per-condition summaries already computed on open
     or spent data (C3-S1 json, G1b json, G2 json).
  2. Readouts: polarity from a 1-D profile of clone share along the simulated body's long axis (same polarity_of
     as G1a); boundary crossing as in G1b; elongation = 1 - circularity and the geodesic aspect ratio of the
     rasterised outline.
  3. Used data first: yes, summaries only.
  4. Power: not a test (qualitative build checks, as in C3-S1); no claim.
  5. Exposure: as in C6-S1.
  6. Literature: Oriola 2024 (poles from mechanics), Mongera 2018. Rule 3 is a minimal convergent-extension
     stand-in, disclosed as such.
- **Model (fixed before any run):**
  - About 400 off-lattice soft-disc cells in 2-D, starting as a random disc.
  - Each cell carries its clone's state z' (the core's state_copy per gastruloid, rho_g 0.059, z from C3-S1) and a
    second clone trait, competence c (step 4 of the plan; in C6-S2 it is set by hand for synthetic clones only).
  - Rule 1, differential adhesion: neighbour attraction scales with exp(-|s_i - s_j| / w), with
    s = z' + N(0, 1) per cell.
  - Rule 2, proportion feedback: the posterior (P) fraction of cells is q = (1 - g) * mean(sigmoid(-s)) + g * a*,
    with g and a* from G5 (0.83, 0.38). The q lowest-s cells are P.
  - Rule 3, posterior-driven extension: the axis is the line from the A centroid to the P centroid. P cells push
    their P neighbours apart along that axis with strength k * c_i * coherence, where coherence is the separation
    of the A and P centroids over the body radius. Unsorted or one-identity bodies get no consistent axis.
  - Fixed seeds; a set number of steps.
- **Smoke tests (synthetic; must pass before the data checks):**
  - S-a: sorting index (A-P centroid separation) rises with the adhesion contrast (smaller w).
  - S-b: elongation rises with k for a sorted 50:50 mix and stays near round at k = 0.
  - S-c: a body with c = 0 in every cell stays round.
  - S-d: run time under 5 s per gastruloid.
- **Data checks (pre-registered criteria, open/spent summaries):**
  - Q1 (T-a): 50% clone-in-bulk polarity per clone (1, 3, 5, 6, 7), 100 sims each. Pass: sign matches the C3-S1
    fit-shard mean for 5/5 clones, and Spearman rank of the sim means vs fit means >= 0.9.
  - Q2 (T-c): clone 1+3 at the G1b ratios (1:4 to 4:1). Pass: the slope of the boundary crossing vs anterior
    fraction lies inside the G1b CI (g1b_mixing.json).
  - Q3 (T-d/T-e direction, qualitative): a low-competence clone (c 0.2, bulk c 1) at 20/50/80% in bulk.
    Elongation falls monotonically with the clone fraction. Read for its form (linear vs buffered); no claim,
    because the authors' result for this direction is exposed.
- **Stopping rule:** if S-a to S-c fail after the stated model, record the failure and stop C6-S2. Do not add
  rules. If Q1/Q2 fail, the 2-D local rules do not reproduce what the 1-D C3-S1 did, which is a chapter-level
  negative. Fitting competences to G2 is C6-S3, not here.

### C6-S2 result (2026-11-08; synthetic + calibration only; Q1-Q3 NOT run; no data statistic)

- **Formal smoke tests (stated DEFAULT model, `results_gastruloid/c6s2_smoke.json`): all four pass by the letter,
  none in substance.**
  - S-a: polar sorting index 0.099 (w 5) to 0.109 (w 0.5). Homotypic contacts 0.55 to 0.59 (random about 0.52).
    Two clean poles would give about 0.85.
  - S-b: 1 - circularity 0.40 / 0.46 / 0.47 at k 0 / 0.6 / 1.2. This is mostly an outline artefact: the jagged
    raster reads 0.40 for a round body, and the moment elongation stays at 0.02-0.08.
  - S-c: passes (0.39 vs 0.40).
  - S-d: 0.36 s per gastruloid.
- **Calibration probes (disclosed; scratch scripts, synthetic 50:50, states +-2):**
  - Longer runs, stronger adhesion, more motility noise, and heterotypic repulsion plus cohesion were all tried.
  - With strong adhesion (k_adh 8, 4,000 steps) cells **do** sort locally: 0.88 homotypic contacts. They form
    many patches, though, not one A-P axis (polar index 0.06).
  - With rule 3 switched on, the posterior push makes the outline ragged and breaks up the sorting. It does not
    extend the body (moment elongation 0.04).
  - The outline readout needs smoothing: a Gaussian density threshold reads more truly than the disc raster.
- **Reading (pre-registered stopping rule applied): rules 1-3 as stated do not build an A-P axis.** Differential
  adhesion plus proportion feedback gives local domains without symmetry breaking into two poles. This matches
  the literature claim that something beyond adhesion sets the poles (Oriola 2024: signalling sets proportions,
  mechanics sets poles). Q1-Q3 were not run: without poles, polarity and boundary readouts are undefined, so they
  would fail by construction. No rule was added within C6-S2.
- **Next (C6-S3 redesign, its own pre-registration, Bence's go):** one symmetry-breaking local rule. Candidates:
  (a) local posterior self-recruitment with longer-range inhibition (Wnt/T reaction-diffusion on the cell
  graph); (b) a Cellular Potts implementation, where surface tension coarsens domains to two poles. Plus the
  smoothed outline readout. The smoke tests will be re-specified with substantive thresholds (polar index >= 0.5,
  moment elongation rising with k) so they cannot pass trivially.

## C6-S3: two symmetry-breaking rules for the 2-D gastruloid, (A) Wnt/T polarity and (B) Cellular Potts (pre-registered 2026-11-09)

- **Serves: C6-S3** (FOCUS.json). Bence: "test both". C6-S2 showed that adhesion plus proportion feedback gives
  patches, not one A-P axis. Script: `c6s3_break.py` (new; numba for B). It reuses the readouts in
  `c6s2_space.py`, now with the smoothed outline. No sealed file is read.
- **Pre-flight:** as C6-S2 (unit simulated gastruloid; open/spent summaries only; no power, since these are build
  checks; exposure as C6-S1). Literature: wave-pinning polarity (Mori 2008; mass-conserved reaction-diffusion
  gives one domain); Cellular Potts sorting (Graner & Glazier 1992); Oriola 2024.
- **Variant A, Wnt/T polarity on the C6-S2 particle model:**
  - Each cell has a Wnt level u. On the contact graph:
    du/dt = D L u + (b_s + u^2 / (1 + u^2)) * V - u, where V is a global pool (wave-pinning; total Wnt fixed, which
    is also the proportion feedback).
  - b_s = b0 * sigmoid(-s): posterior-leaning cells activate more easily.
  - P identity: u > 0.5 * max(u).
  - Movement: rule 1 adhesion (s-based) plus taxis up the local u gradient, with weight chi * (-s) (posterior
    cells move toward high Wnt, anterior cells away).
  - Rule 3 extension as in C6-S2, with the axis from the A to the P centroid.
- **Variant B, Cellular Potts (2-D lattice, numba):**
  - Each cell is about 20 pixels with an area constraint. Contact energy J_ij = J0 + Jd * |s_i - s_j| between
    cells, J_cm with the medium. Metropolis at temperature T.
  - P identity: the same rank rule as C6-S2 (proportion feedback).
  - Rule 3 as a bias: copies by P cells along the A->P axis, away from the body centre, are favoured with energy
    -k * c * coherence.
- **Substantive smoke tests (synthetic 50:50, states +2/-2, 6 runs each; must pass before any data check):**
  - P1: k = 0 gives polar index >= 0.5 and homotypic contacts >= 0.75.
  - P2: a pure body (one clone, state 0) still forms one P pole (polar index >= 0.5). Pure gastruloids polarise.
  - E1: moment elongation at the chosen k minus at k = 0 >= 0.15.
  - E2: c = 0 in every cell keeps moment elongation within 0.05 of k = 0.
  - R: <= 20 s per gastruloid.
  - Parameters may be calibrated on these synthetic tests only (disclosed). Each variant is tried at most 3
    calibration settings per test; then pass or fail is final.
- **Data checks (each variant that passes all five):** Q1 and Q2 with the C6-S2 criteria; reps 40 (Q1) and 20 per
  ratio (Q2), fewer than C6-S2's 100/40 because of run time. Q3 qualitative, as in C6-S2.
- **Stopping rule:** a variant failing P1 or P2 is dropped. If both fail, chapter 6 records that neither rule builds
  an axis in this implementation, and Bence decides. If both pass Q1/Q2, model selection by CV is C6-S4.

### C6-S3 amendment a (2026-11-09; logged before the re-run)

- **Variant A settings 1-3 were outside the rule's regime.** In the wave-pinning kinetics
  du = (b + u^2/(1+u^2)) V - u, a high-Wnt state exists only when the pool V > 2. Settings 1-3 had mass 1.6, so
  V <= 1.6, and no domain could form. Their smoke failures (polar 0.07-0.16; `c6s3_smoke_A{1,2,3}.json`) say
  nothing about the rule and are recorded as void for A.
- **Diagnostic (static synthetic body, no movement):**
  - D 0.3, mass 2.9: one posterior domain, 39% of cells, polar 0.76, homotypic 0.985.
  - D 0.1 or a different mass gives a spot or an over-large domain (the six combinations are in the run log).
- **Amended variant A:** settings 4-6 (D 0.3, mass 2.9, u0 0.5; chi 1 / 2 / 4). Same smoke tests and pass rules, at
  most these 3 settings, then final. Variant B unchanged (settings 1-3).

### C6-S3 result (2026-11-09; synthetic + open/spent summaries only; no sealed data)

- **Variant B (Cellular Potts) fails P1/P2 in all 3 settings and is dropped.** Polar index 0.11-0.14 (needs
  >= 0.5); homotypic contacts 0.72-0.86; pure bodies do not polarise (0.10-0.13). It sorts locally, like C6-S2, but
  never coarsens into poles within 1,500-4,000 MCS. Longer runs were not tried (the limit was 3 settings).
  `c6s3_smoke_B{1,2,3}.json`.
- **Variant A (Wnt/T wave-pinning + taxis), amended settings 4-6:**
  - Settings 4 and 5 pass all five smoke tests in substance. A4: polar 0.93, homotypic 0.99, pure-body polar 1.12,
    moment elongation 0.05 -> 0.27 with extension on, c = 0 stays round (0.055), 0.9 s per gastruloid.
  - A6 fails E1 (0.035 -> 0.156).
  - **First 2-D model here that builds one A-P axis and elongates from local rules.**
- **Data checks on A4 / A5 (pre-registered criteria):**
  - **Q1 FAIL** (both). Polarity is positive for every clone (A4: 1 +0.81, 3 +1.08, 5 +1.22, 6 +0.70, 7 +1.09);
    posterior clones 1 and 6 have the wrong sign. Sign matches 3/5, Spearman 0.9 / 1.0.
  - **Q2 PASS** (both). Boundary slope -0.21 / -0.14, inside the G1b CI [-0.40, 0.42]: fixed domains, as in the data.
  - **Q3** (qualitative): moment elongation falls monotonically with the low-competence clone fraction (A4 0.30 /
    0.30 / 0.25 / 0.21 / 0.15 at 0/20/50/80/100%). It is roughly linear, not buffered. No claim (direction exposed).
- **Post hoc diagnostic (labelled; not a test; `c6s3_q1_fraction_posthoc.json`):**
  - The all-positive Q1 is a readout artefact. Cells per bin follow body width, and the extended posterior is thin,
    so any label looks anterior. Line scans measure intensity, i.e. local clone fraction.
  - With the fraction readout, polarity is about 0 for every clone (A4: -0.02 / 0.01 / 0.13 / -0.04 / -0.12;
    signs 4/5 by chance, Spearman 0.7; A5 Spearman 0.3).
- **Reading:**
  - Variant A breaks symmetry and elongates, but its clone states barely decide where cells end up. Wnt bias
    b0 = 0.05 and taxis chi 1-2 are too weak a coupling between s and the pole.
  - The C6-S2 readout (density per bin) must be replaced by fraction per bin before any data check. This is now
    fixed in the plan.
- **Next (C6-S4, own pre-registration):** fit the clone-to-pole coupling (b0, chi) to the open G1a fit-shard
  polarities by CV, with the fraction readout. Then check Q2 and Q3 hold out of fit, and only then write
  predictions for the sealed S1 / S2.

## C6-S4: fit clone-to-pole coupling by CV, then fixed-model predictions for S1/S2 (pre-registered 2026-11-09)

- **Serves: C6-S4** (FOCUS.json, 63c6e75). Bence: "Yes, C6-S4". Scripts: `c6s4_fit.py` (new), `c6s3_break.py`,
  `c6s2_space.py`; scratch feasibility `feas_c6s4.py` (synthetic only).
- **Question:** C6-S3 variant A (Wnt/T wave-pinning + s-taxis) builds one axis, but clone state barely decides
  where a clone sits (fraction-readout polarity ~0 for all 5 clones). Can the coupling of clone state to the pole be
  fitted so the model reproduces clone polarities **out of fit** (leave-one-clone-out), without breaking Q2/Q3?
- **Model:** A4 (C6-S3 amended setting: D 0.3, mass 2.9, w_sub 10, dt_u 0.1, k_ext 1.0) with two free coupling
  parameters: **b0** (Wnt production bias, b = b0 * sigmoid(-s)) and **chi** (taxis up the Wnt gradient,
  chi * tanh(-s/2)). Nothing else is free. No rescue term.
- **Step 0, feasibility (synthetic only, before any clone data):** synthetic +-2 mixes over b0 {0.05, 0.3, 1.0} x
  chi {1, 8} x u0 {0.5, 0.1}, 5 reps. It fixes (a) u0 (initial Wnt noise) and (b) the grid bounds: the fit grid
  keeps only points with polar >= 0.5 on the synthetic mix. Reported with the result.
- **Fit grid (after step 0):** b0 {0.05, 0.15, 0.4, 1.0} x chi {1, 4, 16}, trimmed by the step-0 rule; 40
  clone+bulk 50:50 gastruloids per clone per point (n 300 cells, as C6-S3 Q1).
- **Readout:** clone **fraction** per bin along the Wnt-oriented axis (30 bins), z-normalised per gastruloid,
  polarity = mean first 15 bins - mean last 15 (`polarity_of`). Rationale: a midline line scan measures local
  label intensity, i.e. the clone's local share, not its cell count per bin (C6-S3 artefact). Disclosed: this
  readout was chosen after C6-S3's post-hoc look at the same 5 targets.
- **Targets (open, used):** G1a fit-shard mean polarity per clone (C3-S1): 1 -0.922 (n 51), 3 0.881 (54),
  5 1.051 (34), 6 -1.132 (30), 7 0.726 (8). Loss: sum over clones of n-weighted squared error.
- **CV (leave-one-clone-out):** for each held-out clone, pick the grid point minimising loss on the other 4,
  predict the held-out clone. **Pass (all three):** (i) held-out signs 5/5; (ii) held-out MSE < MSE of the
  clone-agnostic null (held-out predicted by the mean of the other 4 targets); (iii) Spearman of held-out
  predictions vs targets >= 0.9. Fail -> the coupling cannot be fitted on this data; no S1 prediction, S1 stays
  sealed untested, and the stopping rule (CHAPTER6_PLAN) is weighed for the polarity half of the chapter.
- **Fix:** if CV passes, the full-data best grid point is the fixed model. **Out of fit:** Q2 crossing slope inside
  the G1b CI [-0.396, 0.423] (20 reps x 5 ratios); Q3 moment elongation falls monotonically 0 -> 100% low clone
  (20 reps); smoke P1 (polar >= 0.5) and E1 (elongation gain >= 0.15) still pass. A failure is reported, not tuned.
- **Predictions (fixed model, written before any sealed look; no sealed file is opened in this step):**
  - **S1 (Jan 2024 1+6 pairs, passage 4, n 34):** 400 simulated 1+6 50:50 gastruloids; prediction = share with
    clone-1 polarity > clone-6 polarity. Clone states carried to passage 4 with `memory_through` (no division
    loss; passage retention r per passage over k passages from the fit-shard passage to passage 4, k read from G1a
    metadata, never from the S1 file). r over {0.46, 0.60, 0.68} (C3-S1, G3, G5 used estimates); the **lowest**
    share across r is the prediction used for H14. H14: one-sided binomial power at alpha 0.05, n 34, must be
    >= 0.8 (share >= ~0.72) for a look to be proposed.
  - **S2 (0831 partition C, 6a+bulk):** predicted 6a+bulk 1 - circularity over the member mean, from the fixed model
    with competence c 6a 0.2, bulk 1.0 (as Q3), expressed as a fraction of the pure bulk - pure 6a gap and
    converted to data units with the spent A+B pure-plate gap. H14 power from the C6-S1 table (0.84 needs
    +0.077). S2 is the last partition: any look also needs Bence's verbatim OK (H13).
  - Passing H14 only makes a look proposable (C6-S5); it is not run in C6-S4.
- **Pre-flight:**
  1. Unit: gastruloid; clones are model inputs.
  2. Readout: fraction line-scan polarity (above); moment elongation / 1 - circularity for Q3 and S2.
  3. Used data first: all targets are open or spent (G1a fit shard, G1b, C3-S1, C5-S2, 0831 A+B).
  4. Power: CV has 5 targets only, so a pass is weak evidence (stated); S1/S2 power is computed from the
     fixed prediction before any look (H14).
  5. Exposure: the 5 targets were seen in C3-S1 and C6-S3 (post-hoc readout); stated as the reason the test is
     out-of-clone CV, not in-sample fit. S1/S2: counts only, unchanged.
  6. Oracle/sealed: no sealed file is read; `test_oracle.py` is not called in this step.
- **Run budget:** about 12 grid points x 200 gastruloids ~ 2,400 sims at ~1 s on 10 workers (~5-10 min), plus
  Q2/Q3/S1/S2 (~1,000 sims). One job at a time on this PC.

### C6-S4 result (2026-11-09; `results_gastruloid/c6s4_fit.json`; open data only, nothing sealed read)

- **Step 0 (synthetic +-2 mixes, before any clone data):** u0 0.1 chosen (keeps one axis at more points than
  0.5: polar 0.80 vs 0.43 at b0 0.05, chi 1). Trim rule (polar >= 0.5) keeps **2 of the 12** grid points:
  (b0 0.05, chi 1) 0.80 and (0.05, 4) 0.57. Every stronger coupling breaks the single pole (polar 0.12-0.24):
  the Wnt domain follows the low-s cells, which sit in adhesion patches, so it splits. In the same runs the share
  of Wnt-high cells that are low-s rises from 0.62 to 0.97. **Pole formation and clone-to-pole coupling trade off
  in this model.**
- **Clone polarities on the 2 surviving points (fraction readout, 40 per clone):** (0.05, 1): 1 -0.19, 3 -0.02,
  5 0.03, 6 -0.06, 7 0.11; (0.05, 4): 1 -0.07, 3 0.12, 5 0.04, 6 -0.07, 7 -0.12. Targets are about +-1.
- **LOCO CV: FAIL.** Held-out signs 3/5 (pass needs 5/5), Spearman 0.6 (needs >= 0.9). Held-out MSE 0.88 is below
  the null's 1.40, but that criterion is weak here: the leave-one-out mean of the other four is always pulled
  against the held-out clone, so the null is worse than 0. All three criteria were needed.
- **Pre-registered consequence:** the clone-to-pole coupling cannot be fitted on this data with this model. No
  fixed model, no Q2/Q3 re-check, **no S1/S2 prediction; S1 and S2 stay sealed and untested.**
- **Reading:** variant A gives cells a shared axis (one Wnt pole, elongation) but cannot also place clones on it
  by their state: weak coupling leaves placement at chance, strong coupling breaks the pole. The 1-D rank-sorting
  model (C3-S1) placed clones well because it imposes the axis; the 2-D model has to build it, and with
  s-dependent adhesion plus s-biased Wnt the two goals conflict.
- **Stopping rule (CHAPTER6_PLAN):** applies to the polarity half (T-a, T-b): the rule set reproduces T-c
  (fixed boundary, C6-S3 Q2) and the direction of T-e (C6-S3 Q3, exposed) but not T-a. Adding a rule for this
  misfit (e.g. decoupling adhesion from s, or a pole-anchored sorting cue) would be a new rule per misfit and needs
  Bence's decision, not a silent extension.

## C6-S5: one new rule, s-independent adhesion (taxis-only clone sorting) (pre-registered 2026-11-09)

- **Serves: C6-S5** (FOCUS.json, a5427a2). Bence: "1" (add one rule rather than close). Scripts: `c6s5_rule.py`
  (new), `c6s4_fit.py` (reused via env overrides), `c6s3_break.py`, `c6s2_space.py`.
- **Single disclosed exception to the stopping rule.** C6-S4 found that s-dependent adhesion makes low-s cells
  clump in patches, so any clone-biased Wnt splits the pole. **New rule:** adhesion is the same for all pairs
  (w -> infinity in exp(-|ds|/w)); clone state acts only through Wnt production bias b0 and taxis chi up the Wnt
  gradient. Everything else as C6-S4 (A4 kinetics, k_ext 1.0, u0 0.1, fraction readout, targets, n 300 cells).
  If CV fails, no further rule is added: chapter 6 closes (C6-S6 write-up + retro).
- **Step 0 (synthetic +-2 mix only):** grid b0 {0, 0.05, 0.15} x chi {1, 4, 16}, 5 reps; keep points with
  polar >= 0.5. Reported with the result.
- **LOCO CV:** identical to C6-S4 (40 per clone per point; pass = held-out signs 5/5 AND Spearman >= 0.9 AND
  MSE < null). Then, only on pass: Q2 (fraction crossing slope in G1b CI), Q3 monotone, smoke P1/E1, S1/S2
  predictions + H14, exactly as `c6s4_fit.main`.
- **Pre-flight:**
  1-3. Unit, readout, used data: as C6-S4 (open G1a fit shard; fraction readout; nothing sealed read).
  4. **Power (new, recorded before the run): S1 cannot be powered by any model consistent with the core.** The
     1-D core model (C3-S1: rho_g 0.059, z1 -4.03, z6 -5.33) predicts a share of clone 1 anterior of clone 6 of
     0.59 with no passage fade (H14 power 0.20 at n 34, k_crit 23) and 0.51-0.53 at passage 4 with retention
     0.46-0.68 (power 0.04-0.06). The C6-S1 power table gave power for assumed shares without asking what the
     core predicts: a pre-flight miss, recorded here. S1 is therefore dropped as a chapter-6 test (stays sealed).
     Only S2 (last partition, H13) remains, and needs a predicted +0.077 for power 0.84.
  5. Exposure: the 5 targets have now been used in C3-S1, C6-S3 (post hoc) and C6-S4; CV is out-of-clone only.
  6. Literature: differential adhesion vs chemotactic sorting in gastruloids/aggregates (Oriola 2024 mechanics
     sets poles; Steinberg DAH) - this step tests the chemotaxis-only alternative.

### C6-S5 result (2026-11-09; `results_gastruloid/c6s5_step0.json`, `c6s5_rule.json`; open data only)

- **Step 0 (synthetic):** with uniform adhesion the pole survives stronger coupling than in C6-S4: 3 of 9 points
  keep one axis (b0 0.05 with chi 1 / 4 / 16: polar 1.21 / 0.91 / 0.54; elongation 0.40 / 0.32 / 0.17). b0 0
  gives no Wnt domain at all (polar 0); b0 0.15 breaks the pole again (0.19-0.37). The trade-off is eased, not
  removed.
- **Clone polarities (fraction readout, 40 per clone):** at (0.05, 1) **all five signs are right in sample**
  (1 -0.11, 3 +0.12, 5 +0.06, 6 -0.09, 7 +0.04; Spearman 0.8), but **about 10x smaller than the data (+-0.7 to
  1.1)**. (0.05, 4) and (0.05, 16) are similar in size with one or two sign errors.
- **LOCO CV: FAIL.** Held-out signs 4/5 (clone 3 wrong: its fold picked chi 16), Spearman 0.8 (< 0.9); MSE 0.80 vs
  null 1.40 (weak criterion, see C6-S4).
- **Reading:** removing state-dependent adhesion turns clone placement from chance (C6-S4) into the right
  direction, so taxis on a self-made pole is a workable mechanism in principle. It is far too weak: clones barely
  separate along the axis, because any coupling strong enough to separate them fragments the pole. Out of sample
  the model is not supported.
- **Pre-registered consequence:** no further rule. No fixed model, no S2 prediction (S1 already dropped for power).
  Chapter 6 closes (C6-S6: write-up + retro), pending Bence's OK to move FOCUS.

## C6-S7: state-first sorting redesign (exploratory under H17) (pre-registered 2026-11-09)

- **Serves: C6-S7** (FOCUS.json, 2d9a72d). Bence: "Yes, C6-S7". Script: `c6s7_sort.py` (new; imports
  `c6s2_space.py`, `c6s3_break.py` readouts and builders, `c6s4_fit.py` fraction readout). Design source:
  `lit/reading_c6_mechanism.md` (de Jong 2024, Guan/Hormoz 2025, McNamara 2024).
- **Label: EXPLORATORY (H17)** unless the S2 power computed below from this model's own prediction is >= 0.8.
  S1 is unpowerable under the core (<= 0.20) and is not a test here.
- **Model (particle, n 300, 1,500 steps, dt 0.05, noise 0.3):**
  1. *Identity from state, no Wnt field:* posterior (P) = the q lowest-s cells, q by G5 proportion feedback
     (`c6s2_space.identities`), s = clone state copy (rho_g 0.059) + cell noise. Fixed from the start.
  2. *Binary sorting:* repulsion below contact (k_rep 4); attraction spring k_adh 3 within 2.0-2.6, weight 1 for
     like pairs and 1/3 for unlike pairs (de Jong J 10 vs 30).
  3. *Long-range posterior attraction (Guan):* each P cell is pulled toward P cells within R 8 with total force
     beta x (unit vector to their centroid).
  4. *Same-type crawling (de Jong best variant):* each cell has a polarity (random start). Every 20 steps it links
     to up to 3 same-type neighbours within 2.6 inside a +-45 degree double cone around its polarity; each link
     pulls both cells together with force lam_F x competence c. Polarity update every 20 steps: 0.99 own + 0.01
     mean of linked neighbours' polarities (Vicsek).
- **Step 0 (synthetic +-2 mixes, 50:50, c = 1; 6 reps per point):** grid beta {0, 0.05, 0.2} x lam_F {0, 0.5, 2}.
  Criteria: **S0-a one pole**: polar index >= 0.5 and multipolar rate <= 20% (multipolar = more than one
  connected P cluster, contact 2.6, holding >= 10% of P cells); **S0-b elongation**: moment elongation >= 0.15
  above the lam_F 0 run at the same beta; **S0-c competence**: same point with c = 0 within 0.05 of lam_F 0;
  **S0-d** runtime <= 10 s per gastruloid. **Chosen setting** = the passing point with the smallest beta, then
  the smallest lam_F (parsimony). Expected from Guan: beta 0 fails S0-a. If no point passes: model fails,
  reported, no data checks.
- **Data checks at the chosen setting (no parameter is fitted to clone data, so these are predictions, not CV):**
  - **Q1 clone placement** (clone+bulk 50:50, 40 per clone, fraction readout as C6-S4): pass = signs 5/5 and
    Spearman >= 0.9 vs the G1a fit-shard means (1 -0.92, 3 0.88, 5 1.05, 6 -1.13, 7 0.73). Magnitude reported
    vs targets (pooled within-clone SD 0.87). Also reported: multipolar rate (benchmark McNamara 0/34).
  - **Q2 fixed boundary** (3 vs 1 at 5 ratios, 20 each): fraction-crossing slope inside the G1b CI
    [-0.396, 0.423].
  - **Q3** (low-competence clone c 0.2, z 0, at 0/20/50/80/100% in bulk, 20 each): moment elongation falls
    monotonically. Exposed target (T-e): described only.
- **H17 / S2 prediction:** 6a+bulk 1 - circularity over the member mean (100 each of pure bulk, pure 6a, 50:50
  mix; c 6a 0.2; z 0 primary, z = clone 6 sensitivity), as a fraction of the pure gap x the spent pure-plate gap
  0.180 (C4-S4: bulk 0.222, 6a 0.042); power one-sided at alpha 0.05 with SE 0.0292 (C6-S1 table). >= 0.8 ->
  S2 look proposable to Bence (H13, last partition); otherwise C6-S7 stays exploratory.
- **Stopping:** no parameter is tuned after the step-0 choice; failures are reported.
- **Pre-flight:**
  1. Unit: gastruloid; clones are model inputs.
  2. Readout: fraction line scan (C6-S4), moment elongation, 1 - circularity (S2, as C4-S2a).
  3. Used data first: only open/spent summaries (G1a fit shard, G1b, C4-S4 pure plate).
  4. Power: H17 block above; S1 dropped (core ceiling 0.20).
  5. Exposure: the 5 Q1 targets were seen and used (C3-S1, C6-S3..S5); here no parameter is fitted to them, which
     is why Q1 is a prediction; still not an unseen test. T-e exposed.
  6. Literature: read in full before code (C6-S6 notes); unverified items listed there (Guan kernel, McNamara
     cadherin LOF).

### C6-S7 result (2026-11-09; `results_gastruloid/c6s7_sort.json`; synthetic only, no clone data used)

- **Step 0: FAIL, no point passes** -> no data checks (pre-registered). Synthetic +-2 mixes, 6 reps each:
  - **Elongation works:** same-type crawling lam_F 2 gives moment elongation 0.51-0.53 (lam_F 0: 0.06), and
    competence 0 stays round (0.05-0.07): S0-b and S0-c pass at lam_F 2 for every beta. lam_F 0.5 is too weak
    (+0.03 to +0.06).
  - **One pole fails everywhere:** polar index 0.08-0.16, multipolar rate 0.83-1.00 at all 9 points (S0-a).
    beta 0 fails as Guan predicts, but so do beta 0.05 and 0.2 with range R 8.
  - Runtime 0.7-2.5 s per gastruloid (S0-d passes).
- **Diagnostic (disclosed, synthetic only, after the failure; `scratchpad/diag_c6s7.py`):** beta 0.2: R 8 -> 40
  and 1,500 -> 5,000 steps leave polar at 0.11-0.12 (multipolar 0.83-1.00). **beta 1.0 with R 40: multipolar 0.00
  but polar 0.10**: posterior cells merge into one cluster in the **centre**, with anterior cells around it
  (core-shell engulfment), not a pole.
- **Reading:** the design merges the posterior cells but places them inside. The cause is my mapping, not the
  literature: I gave both types equal cohesion (like weight 1). Guan's top settings are **asymmetric**: strong
  cohesion in the majority type, moderate in the attracting type, weak between types, with long-range attraction
  in the type that must gather at the surface. With that asymmetry the attracting type forms a surface cap
  (= one pole); with equal cohesion it forms a core. McNamara's live imaging (a Wnt-high ring that pushes out a
  Wnt-low domain) is also a surface-cap route, not a core.
- **Consequence:** C6-S7 fails at step 0 as registered. A corrected mapping (asymmetric cohesion as in Guan)
  is an amendment that needs Bence's OK and its own pre-registration (C6-S7a); nothing has been fitted to
  clone data.

## C6-S7a: amendment, asymmetric cohesion as in Guan/Hormoz (pre-registered 2026-11-09)

- **Serves: C6-S7** (FOCUS.json, 2d9a72d; amendment within the step). Bence: "Yes, C6-S7a" (answer to:
  pre-register amendment C6-S7a with Guan's asymmetric cohesion?). Scripts: `c6s7a_sort.py` (new), `c6s7_sort.py`
  (simulate_S gains per-type-pair cohesion weights; defaults reproduce C6-S7).
- **Why:** C6-S7 step 0 failed; a synthetic diagnostic showed posterior cells merging into a central core with
  equal cohesion. Guan's top settings: strong cohesion in the majority type, moderate in the attracting minority
  type, weak between types, long-range attraction in the minority type. Posterior (P) is the minority here
  (q ~0.39 on synthetic mixes, ~0.38 in clone+bulk), as Guan's outer type (25%).
- **Change (only this):** adhesion weight w_AA 1.0 (anterior-anterior), w_PP and w_AP from the step-0 grid;
  long-range P attraction range R 40 (whole body; Guan's kernel is long-range, C6-S7 R 8 was my choice).
  Everything else as C6-S7, crawling lam_F 2 (the C6-S7 value that elongates).
- **Step 0 (synthetic +-2, 6 reps):** grid w_PP {0.3, 0.6} x w_AP {0.1, 0.33} x beta {0.2, 1.0}. Criteria as
  C6-S7: S0-a polar >= 0.5 and multipolar <= 20%; S0-b elongation >= 0.15 above the same point with lam_F 0;
  S0-c competence 0 within 0.05 of lam_F 0; S0-d <= 10 s. **Choice:** the passing point with the highest polar
  index. No point passes -> the state-first design fails in this particle model; report; no further amendment
  without Bence.
- **Data checks, H17 S2 power, labels:** exactly as pre-registered for C6-S7 (Q1 prediction signs 5/5 and
  Spearman >= 0.9; Q2 slope in G1b CI; Q3 monotone, described; S2 power from the model's prediction; exploratory
  unless >= 0.8).
- **Pre-flight:** unchanged from C6-S7 (items 1-6); exposure adds the C6-S7 synthetic diagnostic (no clone data).

### C6-S7a result (2026-11-09; `results_gastruloid/c6s7a_sort.json`; synthetic only, no clone data used)

- **Step 0: FAIL, no point passes** -> no data checks (pre-registered); no further amendment without Bence.
  - One cluster: beta 1.0 gives multipolar 0.00 at all four cohesion pairs (beta 0.2: 0.17-0.50).
  - **No pole: polar index 0.14-0.28** (best w_PP 0.6, w_AP 0.33, beta 1.0: 0.28; needs 0.5). Asymmetric
    cohesion moves the posterior cluster off centre (C6-S7: 0.10-0.16) but not to a surface cap.
  - Elongation still works (0.28-0.42 vs 0.05-0.11 without crawling; competence 0 stays round), except w_PP 0.3 /
    w_AP 0.33 / beta 0.2 (+0.10).
- **Reading:** the long-range term as I implemented it (unit pull toward the centroid of all posterior cells
  within R) points at the body centre whenever posterior cells are spread around it, so it fights the cap that
  asymmetric cohesion would make. Guan's force is "gravitation-like" (distance-weighted: near cells dominate),
  and its exact form sits in their separate "Biophysical model" file, which C6-S6 could not read. **The kernel was
  my guess, so this tests my guess, not their mechanism.** The state-first design is not refuted; this particle
  implementation of it fails.

## C6-S7bc: C6-S7b Guan exact force law and C6-S7c de Jong Potts port (pre-registered 2026-11-09)

- **Serves: C6-S7** (FOCUS.json; amendments within the step, both exploratory under H17 unless S2 power >= 0.8).
  Bence: "go both 1 2" (answer to: C6-S7b, Guan's force law exactly, or de Jong's Potts?). Scripts:
  `c6s7_checks.py` (new; step-0 job and the C6-S7a data checks copied unchanged, simulator passed by module
  name), `c6s7b_guan.py` (new), `c6s7c_potts.py` (new).
- **Source read (C6-S7b):** Guan et al. "Biophysical model" supplement (bioRxiv 2025.09.27.678924, file 14;
  read in the browser as text, saved nowhere in git). Two-type, pre-differentiated case: cell radius l 1;
  contact force for d < 2l, F_m = (1 - d/(2 l alpha_mn)) * (r_m - r_n)/d (repulsive close, attractive between
  2l alpha and 2l; lower alpha = stronger adhesion), alpha = 0.95 - K_alpha x (0..1), K_alpha 0.175; long-range
  force for d > 2l, F_m = -(beta_mn / d^2) * (r_m - r_n)/d (1/d^2 pull toward sender n), K_beta 0.125;
  overdamped, dt 0.2, T 150 (750 steps), eta 1, position noise kappa_r 0.1 (sqrt(dt kappa_r) N(0,1)). Their gene
  network is replaced by our identity rule (P from clone state via proportion feedback), as in C6-S7.
- **C6-S7b model:** exactly the above in 2-D, N 300, same disc start as C6-S7; P = their attracting minority
  type ("outer"). Grid (step 0): alpha_AA 0.775 (strong majority cohesion, Guan's best), alpha_PP {0.8625, 0.95},
  alpha_AP {0.8625, 0.95}, beta_PP {0.125, 0.5}, receivers {P only (outer-outer), all cells (sender-only form,
  their Eq. 5)} = 16 points. Elongation: same-type crawling from C6-S7 (links to contacts d < 2.2, Vicsek 0.99,
  t_int 20 steps), lam_F {0.05, 0.2} x competence (their forces are ~0.1-0.3, so smaller than C6-S7's 2).
  **Two-stage step 0:** stage 1 runs all 16 points with lam_F 0 (6 reps); points with polar >= 0.5 and
  multipolar <= 20% go to stage 2, which runs each with both lam_F values and applies S0-a..d exactly as C6-S7
  (S0-d <= 10 s). If stage 1 has no such point, step 0 fails at stage 1 (nothing more run).
- **C6-S7c model (de Jong port, numba CPM):** binary types from the same identity rule; J cell-medium 10,
  like-like 10, unlike 30, T 25, Moore (8) neighbourhood for energy and copies, area constraint lambda 5 with
  target area 25 px (de Jong 100; scaled down for run time), no length constraint (simplification, stated);
  1 MCS = lattice-size copy attempts. Crawling: every 20 MCS each cell links to up to 3 same-type contact
  neighbours inside +-45 deg of its polarity (double cone), polarity 0.99 own + 0.01 mean of linked; link energy
  dH = lam_F x c_owner x change in link length (centroid distance; pulling). Grid (step 0): MCS {3000, 10000} x
  lam_F {20, 80}, lam_F 0 arm per MCS. Criteria S0-a..d as C6-S7, except S0-d <= 60 s per run (CPM); multipolar
  uses r 6.5 (1.3 x cell spacing 5, the same 1.3 x spacing rule as the particle r 2.6 / spacing 2).
- **Choice and data checks (both):** the passing point with the highest polar index; then Q1 (prediction: signs
  5/5 and Spearman >= 0.9 vs the G1a fit-shard polarities), Q2 (slope in G1b CI), Q3 (monotone, described), S2
  power from the model's own predicted 6a+bulk effect (H17), exactly as pre-registered for C6-S7. No step-0
  pass -> report; no further amendment without Bence.
- **Pre-flight:** unchanged from C6-S7 (items 1-6): unit = the 5 open clones' fit-shard polarities (exposed, CV
  not claimed); readout = fraction line scan; nothing sealed read; S1 unpowerable (<= 0.20), S2 power computed
  from the model; exposure: all C6-S2..S7a runs used these same 5 clone targets only in Q-checks that did not
  run in S7/S7a (step 0 failed), so no fit to them since C6-S5.

### C6-S7bc result (2026-11-09; `results_gastruloid/c6s7b_guan.json`, `c6s7c_potts.json`; synthetic only, no clone data used)

- First attempt ran both pools at once and ran out of memory (OpenBLAS threads x 20 workers); both rerun one after
  the other with BLAS at 1 thread. Same seeds, so the numbers reproduce the partial first run exactly.
- **C6-S7b (Guan exact force law): step 0 FAILS at stage 1** (no stage 2 run).
  - Polar index 0.03-0.18 at all 16 points (needs 0.5). Best: alpha_PP 0.95, alpha_AP 0.8625, beta 0.5, P-only
    receivers (0.18, multipolar 0).
  - P-only long-range pull keeps one P cluster (multipolar 0 at beta 0.5) but central, as in C6-S7a. Pull on all
    cells (their Eq. 5 form) breaks P into several clusters (multipolar 0.17-1.0) and stretches the body (0.25-0.58)
    without a pole.
  - Run time ~11 s per run with 10 workers (S0-d limit 10 s; moot).
- **C6-S7c (de Jong Potts port): step 0 FAILS.**
  - Polar 0.12-0.26, multipolar 83-100% at all points: types are still in many patches at 10,000 MCS.
  - **Sorting is still coarsening:** with lam_F 0, polar goes 0.12 (3,000 MCS) -> 0.20 (10,000). de Jong ran
    100,000 MCS; this run is not converged.
  - Crawling adds little elongation: +0.13 at most (lam_F 80, 10,000 MCS), short of +0.15; competence 0 differs
    by 0.09 there (S0-c fails).
- **Reading:**
  - Guan's force law, now exact, still gives no cap from a random mixed start in 2-D. Their paper does not report
    random starts (initial condition ambiguous, "outer" type may start outside). Their sims are 3-D, where the
    1/d^2 sum differs. So this does not refute their mechanism, but it rules it out as a drop-in fix here.
  - The Potts result is under-run, not a clean fail; the trend with MCS is the one live lead.
- No clone data used, nothing sealed touched. No further amendment without Bence.

## C6-S7de: C6-S7d long Potts runs and C6-S7e Guan in 3-D or with outer start (pre-registered 2026-11-09)

- **Serves: C6-S7** (amendments within the step; exploratory under H17 unless S2 power >= 0.8). Bence: "1 2"
  (answer to options: 1 run the Potts model 3-10x longer; 2 Guan in 3-D or with posterior cells starting
  outside). Scripts: `c6s7d_potts_long.py` (new; reuses c6s7c_potts.simulate unchanged), `c6s7e_guan3d.py` (new),
  `c6s7_checks.py` (unchanged).
- **C6-S7d:** de Jong port exactly as C6-S7c; MCS {30000, 100000} x lam_F {20, 80}, lam_F 0 arm per MCS, 6 reps.
  Criteria S0-a..d as C6-S7c, except **S0-d <= 300 s per run** (100,000 MCS takes ~150 s; the downstream checks,
  ~1,100 runs, then take ~5 h). Choice and data checks as C6-S7.
- **C6-S7e:** Guan force law exactly as C6-S7b (P-only receivers; their all-cell form split P in C6-S7b), two
  geometries:
  - **3-D:** random mixed start in a ball (radius 1.1 x N^(1/3)); crawling polarity is a 3-D unit vector
    (same Vicsek rule). Readouts on a 2-D view: positions projected on the plane of the P-A centroid axis and
    the largest remaining spread (a gastruloid lying flat); multipolar computed in 3-D (r 2.6).
  - **2-D outer start:** the same disc start, but P cells take the outermost radii (Guan's radial prepattern,
    their "outer" type). Caveat stated in advance: McNamara 2024 sees unbiased Wnt patches before the pole, so
    this start has weak biological support; a pass here needs that caveat in any claim.
  - Grid: alpha_PP {0.8625, 0.95} x alpha_AP {0.8625, 0.95} x beta {0.125, 0.5} x geometry {3-D, outer} = 16
    points; two-stage step 0 and criteria exactly as C6-S7b (S0-d <= 10 s, lam_F {0.05, 0.2} in stage 2).
- **Order:** C6-S7e first, then C6-S7d, one at a time (memory), BLAS 1 thread. If both pass, Q-checks for each.
- **Pre-flight:** unchanged from C6-S7 (items 1-6); no sealed data; clone targets used only in the Q-checks.

### C6-S7de result (2026-11-09; `results_gastruloid/c6s7e_guan3d.json`, `c6s7d_potts_long.json`; synthetic only, no clone data used)

- **Reading first (a9d0e22, committed before these results were looked at):** in the standard protocol,
  T/Bra+ cells do sit on the outside at ~60-72 h (Hashmi 2022 Fig 3C, n 7; Suppinger 2023; Sagy 2019 in
  embryoid bodies), but as a patchy layer, not a full ring; reversed without CHIR (Oriola 2024). So the outer
  start is legitimate for our protocol, not a worst case (lit/reading_c6_outer_start.md).
- **C6-S7e: step 0 FAILS at stage 1** (no stage 2).
  - 3-D, random mix: one P cluster (multipolar 0 at all 8 points), polar 0.20-0.36. Best: alpha_PP 0.95,
    alpha_AP 0.8625, beta 0.125 (0.36). Better than 2-D random (C6-S7b <= 0.18), still no cap.
  - 2-D outer start: polar 0.09-0.26. Weak pull (beta 0.125) breaks the ring into several clusters
    (multipolar 1.0); strong pull (0.5) gives one cluster (multipolar 0) but it is not at one end (<= 0.26).
    (The single smoke run that gave 0.37/0.53 was one seed; the 6-rep means are lower.)
- **C6-S7d: step 0 FAILS.**
  - Without crawling, sorting keeps improving with run length: polar 0.12 (3k MCS), 0.20 (10k), 0.36 (30k),
    **0.45 (100k)**, but multipolar stays 100%.
  - Crawling elongates strongly (0.69-0.91, +0.4 to +0.8 over lam_F 0; competence 0 stays near lam_F 0 at
    lam_F 80) **but lowers polarity to 0.24-0.29**: crawling in both types stretches the body before the
    types finish sorting.
  - Run time 57-270 s per run (within the 300 s limit).
- **Reading:** none of the three mechanisms makes one posterior cap from these starts within the run lengths
  tested. Two leads, neither tested yet:
  1. The Potts lam_F 0 runs reach polar 0.45 yet score 100% multipolar. That is either real (several
     same-type domains along one side) or the r 6.5 centroid-distance criterion splitting one domain. Must be
     checked by contact-based domain counting before reading it either way.
  2. Crawling in both types destroys polarity. de Jong's posterior-only ("yellow-yellow") crawling variant
     also fit their shapes and would not stretch the anterior domain.
- No clone data used, nothing sealed touched. No further amendment without Bence.

## C6-S7f: Potts cluster-count check and posterior-only crawling (pre-registered 2026-11-09)

- **Serves: C6-S7** (amendment within the step; exploratory under H17 unless S2 power >= 0.8). Bence: "1"
  (answer to: C6-S7f, check the Potts cluster count, then posterior-only crawling at 100,000 MCS). Scripts:
  `c6s7f_potts_pcrawl.py` (new), `c6s7c_potts.py` (two additions that leave earlier runs unchanged: optional
  prm "crawl" = "P" restricts links to P-P pairs, default both types; the final lattice and its contact pairs
  are kept in a module variable for readouts), `c6s7_checks.py` (unchanged).
- **Part A, measurement check (synthetic, no clone data):** rerun the C6-S7d lam_F 0, 100,000 MCS arm with
  the same seeds (50100-50105; reproduces polar 0.45). For each run, count P domains two ways: (i) as before
  (centroid distance < 6.5), (ii) **by contact** (P cells sharing a lattice boundary, Moore-free 4-neighbour
  pixel contacts, as b._contacts). A domain is a connected group with >= 10% of P cells, as before. Save one
  picture of the lattice coloured by type (results_gastruloid/c6s7f_lattice.png) for Bence.
  **Decision rule, fixed now:** from C6-S7f on, the Potts multipolar readout is the contact count (cells in a
  CPM have real boundaries; the distance cut was carried over from the particle model). Earlier Potts results
  are not re-scored or re-judged.
- **Part B, posterior-only crawling (de Jong's "yellow-yellow" variant, which also fit their shapes):** links
  only between P cells; everything else as C6-S7d. Grid: 100,000 MCS x lam_F {20, 80}; the lam_F 0 reference is
  Part A. Criteria S0-a..d as C6-S7d (multipolar by contact; S0-d <= 300 s). Choice and data checks as C6-S7
  (Q1 signs 5/5 and Spearman >= 0.9; Q2; Q3; S2 power from the model, H17). No pass -> report; no further
  amendment without Bence.
- **Pre-flight:** unchanged from C6-S7 (items 1-6); no sealed data; the 5 open clone targets are used only if
  step 0 passes. Downstream would take ~6 h (overnight).

### C6-S7f result (2026-11-09; `results_gastruloid/c6s7f_potts_pcrawl.json`, `c6s7f_lattice.png`; synthetic only, no clone data used)

- **Part A (cluster count):** the distance rule was not the problem. By contact, the 6 runs have 3-4 posterior
  domains each (polar 0.31-0.64, mean 0.45, reproduced). **The picture shows why: the aggregate falls apart.**
  Posterior cells leave the anterior body and sit as separate floating clumps, with no A-P contact at all.
- **Cause (read in de Jong's code as text, ca.cpp DeltaH, cell.cpp EnergyDifference, J.dat):** the
  Hamiltonian is the standard one, as in my port. Their J.dat gives medium-cell 10, like 10, unlike 30, so the
  interface tensions are gamma_AP = 30 - 10 = 20 and gamma_Am = gamma_Pm = 10 - 5 = 5. Since gamma_AP >
  gamma_Am + gamma_Pm, **complete separation of the two types is the energy minimum (Steinberg)**. My cells are
  4x smaller (area 25 vs 100), so they sort faster and reach that end state within 100k MCS; their runs
  likely stop short of it (consistent with their "snowman" shapes). Their table is therefore not a gastruloid
  pole model at long times.
- **Part B (posterior-only crawling): step 0 FAILS.** lam_F 20: polar 0.36; lam_F 80: polar 0.49; multipolar
  100% at both; elongation 0.16-0.17, no gain over lam_F 0 (0.11 -> +0.05/+0.06, needs +0.15). Posterior-only
  crawling does not stretch the body, because the posterior clumps are detached.
- **Reading:** chapter 6's Potts arm needs an energy table in the partial-wetting range,
  |gamma_Am - gamma_Pm| < gamma_AP < gamma_Am + gamma_Pm, where two types stay attached and the minority forms
  a cap. None of the Potts runs so far (C6-S3 B, C6-S7c/d/f) were chosen by that rule. No further amendment
  without Bence.

## C6-S7g: Potts with energies in the partial-wetting range (pre-registered 2026-11-09)

- **Serves: C6-S7** (amendment within the step; exploratory under H17 unless S2 power >= 0.8). Bence: "1"
  (answer to: C6-S7g, rerun the Potts model with energies chosen to keep the types attached). Scripts:
  `c6s7g_potts_wet.py` (new), `c6s7c_potts.py` (optional per-type energies J_AA, J_PP, J_Am, J_Pm, J_AP;
  defaults reproduce earlier runs), `c6s7f_potts_pcrawl.py` (contact domain count reused), `c6s7_checks.py`.
- **Rule (Steinberg; tensions gamma_Am = J_Am - J_AA/2, gamma_Pm = J_Pm - J_PP/2, gamma_AP = J_AP -
  (J_AA + J_PP)/2):** two types stay attached as two lobes / a cap when |gamma_Am - gamma_Pm| < gamma_AP <
  gamma_Am + gamma_Pm. All four points satisfy it (J_AA = J_PP = 10, T 25, everything else as C6-S7c):
  - W1: J_Am = J_Pm = 20, J_AP = 20 (gammas 15 / 15 / 10)
  - W2: J_Am = J_Pm = 20, J_AP = 30 (15 / 15 / 20)
  - W3: J_Am = 20, J_Pm = 25, J_AP = 25 (15 / 20 / 15; P less cohesive with medium)
  - W4: J_Am = 25, J_Pm = 20, J_AP = 25 (20 / 15 / 15)
- **Two-stage step 0 (synthetic +-2, 6 reps):**
  - Stage 1, no crawling, MCS {30000, 100000} x W1-W4 (8 points). Pass: polar >= 0.5, multipolar (by contact)
    <= 20%, and **attached**: mean share of cells in the largest contact-connected piece >= 0.95.
  - Stage 2, for stage-1 passes: crawling in both types (de Jong's best; posterior-only did not elongate in
    C6-S7f), lam_F {20, 80}; S0-a..d as C6-S7d (S0-a now also needs attached >= 0.95; S0-d <= 300 s).
  - No stage-1 pass -> step 0 fails; nothing more run.
- **Choice and data checks as C6-S7** (Q1 signs 5/5 and Spearman >= 0.9; Q2; Q3; S2 power from the model, H17).
  Downstream ~6 h if reached.
- **Pre-flight:** unchanged from C6-S7 (items 1-6); no sealed data; clone targets only if step 0 passes.

### C6-S7g result (2026-11-09; `results_gastruloid/c6s7g_potts_wet.json`, `c6s7g_lattice.png`; synthetic only, no clone data used)

- **Step 0 FAILS at stage 1** (no stage 2, no data checks), but on one criterion only.
  - **Attached: 1.00 at all 8 points.** Partial-wetting energies fix the break-up seen in C6-S7f.
  - **Polar reaches the bar at 100k MCS:** W1 0.60, W4 0.58, W2 0.52 (W3 0.36). At 30k MCS: 0.29-0.43.
  - **Multipolar 83-100% everywhere:** the posterior cells form 2-3 separate caps on one attached body (picture),
    not one cap. Caps merge only by travelling around the surface, which is slow.
  - Elongation without crawling 0.12-0.35.
- **Reading:** short-range adhesion gets sorted, attached caps but not a single one, which is exactly Guan's
  finding for adhesion alone (best asymmetry 0.83, no robust single axis) and their fix is long-range
  posterior attraction. In a Potts model that would be an energy term pulling P cells toward other P cells;
  untested. Longer runs may also merge caps (polar rose 0.29-0.43 -> 0.52-0.60 from 30k to 100k). No further
  amendment without Bence.
- **Literature check on detachment (asked by Bence):** no source reports posterior tissue detaching in
  gastruloids; Oriola 2024 heterotypic T+/T- aggregate fusion shows the two tissues fuse (interfacial tension
  below the sum of surface tensions), i.e. the partial-wetting range used here.

## C6-S7h: long-range posterior attraction in the attached Potts model (pre-registered 2026-11-09)

- **Serves: C6-S7** (amendment within the step; exploratory under H17 unless S2 power >= 0.8). Bence: "yes"
  (answer to: pre-register C6-S7h, Guan's long-range posterior attraction in this Potts model, or run longer?).
  Scripts: `c6s7h_potts_lr.py` (new), `c6s7c_potts.py` (kernel gains a per-cell force term; zero force
  reproduces earlier runs), `c6s7g_potts_wet.py` (energy tables and readouts reused), `c6s7f_potts_pcrawl.py`,
  `c6s7_checks.py`.
- **Force (Guan's law, P senders and P receivers only):** every 20 MCS, for each P cell i, F_i = beta x sum over
  P cells j with centroid distance d > 2 r_cell (r_cell = sqrt(25/pi) = 2.8 px) of (c_j - c_i)/d^3 (a 1/d^2 pull
  toward j). In the copy step, a move of a P cell's centroid by dc adds dH = -F_i . dc (standard CPM force
  term). Anterior cells feel nothing.
- **Grid (stage 1, no crawling, 100,000 MCS, 6 reps):** energy tables W1 and W4 (C6-S7g's best) x beta {300,
  1000, 3000} (energy x px; brackets a pull per cell comparable to the crawling link forces, 20-80 per px).
  Pass: polar >= 0.5, multipolar (contact) <= 20%, attached >= 0.95. Stage 2 for passes: crawling in both
  types, lam_F {20, 80}; S0-a..d as C6-S7g (S0-d <= 300 s). Choice and data checks as C6-S7. No stage-1 pass ->
  report; no further amendment without Bence.
- **Pre-flight:** unchanged from C6-S7 (items 1-6); no sealed data; clone targets only if step 0 passes.

### C6-S7h result (2026-11-10; `results_gastruloid/c6s7h_potts_lr.json`, `c6s7h_lattice.png`; clone checks used only open/spent summaries, nothing sealed)

- **Step 0: PASS (first in chapter 6).**
  - **Stage 1, no crawling, 100k MCS, 6 reps:** 5/6 points pass, all attached 1.00 and multipolar 0%. W1: polar
    0.84 / 0.62 / 0.53 at beta 300 / 1000 / 3000. W4: polar 0.93 / 0.44 (fails) / 0.51.
  - **Stage 2:** 4 points pass S0-a..d:
    - W1 b300 lam20 and lam80;
    - W1 b1000 lam80;
    - W4 b3000 lam80.
  - **W4 b300 fails S0-b:** it already elongates without crawling (0.45).
  - **Chosen (highest polar): W1, beta 300, lam_F 80.** Polar 0.88, multipolar 0%, attached 1.00. Elongation
    0.50 vs 0.19 without crawling; competence 0 gives 0.17.
- **Q1 clone placement: signs 5/5, Spearman = 0.90, exactly the pre-registered threshold (>= 0.9).**
  - Only clones 3 and 5 are swapped (sum d^2 = 2, so rho = 1 - 12/120 = 0.9).
  - The script's flag says `pass: false` only because floating point gives 0.8999999999999998. **By the
    registered rule (>= 0.9) this is a pass, at the boundary.** I disclose it here rather than change the flag.
  - Magnitudes are about 1/3 of the targets: sim -0.30 / 0.42 / 0.18 / -0.39 / 0.04 vs target -0.92 / 0.88 /
    1.05 / -1.13 / 0.73 (clones 1 / 3 / 5 / 6 / 7). Clone 7 is near 0.
  - Multipolar 0-15%, polar 0.68-0.73.
  - Still not an unseen test: the targets were exposed and used in C3-S1 and C6-S3..S5.
- **Q2 fixed boundary: FAIL.** The crossing moves with the ratio (4 -> 39 over f_anterior 0.2 -> 0.8). Slope 0.57 is
  outside the G1b CI [-0.40, 0.42]. In the model the A-P boundary follows the proportions; in the data it does not.
- **Q3 competence: FAIL** (no monotone fall).
  - Elongation 0.56 / 0.62 / 0.60 / 0.66 / 0.57 at 0 / 20 / 50 / 80 / 100% low-competence clone.
  - Reading: c 0.2 x lam_F 80 = 16 per link, which is about the lam 20 that already elongates. So elongation
    saturates and competence 0.2 does not slow it.
- **S2 (H17): power 1.00 on the primary state (z 0), but the sign flips on the registered sensitivity (z = clone 6).**
  - **z 0:** mix 0.762 vs bulk 0.738 and 6a 0.703. Fraction of the pure gap is +1.17, i.e. 0.21 in data units.
  - **z clone 6:** -0.91 (-0.16), power about 0.
  - The model's pure gap (bulk - 6a) is only 0.035 against 0.18 in the spent data. The ratio divides two small
    simulated differences (100 gastruloids each), so the prediction is fragile.
  - The script labels it "powered". By the registered rule the S2 look is *proposable* to Bence (H13 last
    partition).
  - **My recommendation: do not spend the last partition on it.** The direction depends on the state
    assumption, and Q2/Q3 fail.
- **Reading:**
  - The attached Potts model with the long-range posterior pull is the first version to build one pole and
    elongate.
  - It orders clone placement correctly, but only at the boundary of the threshold and at 1/3 magnitude.
  - It fails the fixed-boundary and competence checks.
  - Both failures point to the same cause: in the model, proportions and crawling strength set the outcome, not
    the clone's state.
  - No parameter was tuned after the step-0 choice. C6-S7 remains exploratory. Next step needs Bence (C6-S8 or
    further amendment).

## C6-S7i: diagnosis of the C6-S7h Q2 and Q3 failures (pre-registered 2026-11-10)

- **Serves: C6-S7** (diagnosis of its data checks; exploratory). Bence: "yes" (answer to: "pre-register a
  diagnosis of the Q2/Q3 failures, or close C6-S7 and plan C6-S8?"). Scripts: `c6s7i_diag.py` (new),
  `c6s7h_potts_lr.py`, `c6s7g_potts_wet.py`, `c6s7c_potts.py`, `c6s7_checks.py`, `c6s2_space.py`,
  `c6s3_break.py`, `c6s4_fit.py`.
- **Status: diagnosis only, not a test.**
  - The Q2 target (G1b CI) and the Q3 target are already exposed, so nothing here can turn Q2 or Q3 into a pass.
  - Any fix this suggests needs its own pre-registration and Bence's OK. It must then re-pass step 0 and Q1,
    and S2 must be recomputed.
  - Nothing sealed is read. No clone-fit parameter is introduced.
- **D1, Q2 (the boundary follows the proportions; model slope 0.57 vs data CI [-0.40, 0.42]).**
  - **Question:** is the cause the identity rule (posterior = the q lowest-s cells, q from proportion
    feedback), or the mechanics?
  - **D1a, identity-only geometry, no simulation:**
    - Use the same gastruloid builder (clone 3 vs clone 1, f_anterior 0.2 / 1/3 / 0.5 / 2/3 / 0.8, n 300) and
      the same identity rule (g 0.834).
    - Place cells in perfect sorting: anterior cells evenly in [0, 100 n_A/n], posterior cells after them,
      in random order within each type.
    - Use the same readout: z-scored clone-fraction profiles, first crossing, slope against 100 f. 200 reps
      per ratio.
    - **Reading:**
      - slope within 0.2 of 0.57: the identity rule explains Q2.
      - slope within 0.2 of 0 while the simulation gives 0.57: the mechanics or the readout add it.
      - otherwise: mixed.
  - **D1b, same geometry, two identity variants:**
    - (i) **g 0:** q follows composition.
    - (ii) **independent identity:** each cell is posterior with probability sigmoid(-s - theta). theta is set
      per gastruloid so that the expected posterior fraction equals the feedback q, so there is no ranking
      between cells.
    - Report each slope. This is a pointer to a candidate fix, not a fit.
  - **D1c, only if D1a reads "mechanics":** rerun the C6-S7h Q2 simulations (same seeds 92000+, chosen
    setting) and store each clone's posterior fraction and within-type mean axis position per ratio.
    About 30 min.
- **D2, Q3 (no fall in elongation with the low-competence share).**
  - **Hypothesis (saturation):** elongation depends on the product c x lam_F and saturates. At lam_F 80,
    competence 0.2 (product 16) already elongates fully.
  - **Run:** pure gastruloids (state z 0) at W1, beta 300, 100k MCS. Competence c {0, 0.1, 0.2, 0.5, 1} x lam_F
    {20, 80}, 10 reps each (seeds 96000+). Readouts: moment elongation, polar, attached.
  - **Readings:**
    - **Saturation supported** if all three hold:
      1. c 0.2 at lam 80 is within 0.05 of c 1 at lam 80;
      2. elongation rises with c at lam 20 (c 0.2 at least 0.15 below c 1);
      3. points with a similar product (c 0.2 x 80 = 16 vs c 1 x 20 = 20) are within 0.08.
    - Otherwise: not supported, reported as is.
  - **Descriptive only:** the Q3 series at lam_F 20 (W1 b300 lam20, a step-0 passing point; same fractions
    and seeds as C6-S7h Q3, 20 reps). It cannot replace the chosen setting; any switch needs a new
    pre-registration.
- **Cost:** about 1 h on Pool(10) (D2 100 sims + Q3 at lam 20 100 sims; D1a/b are seconds).
- **Pre-flight:**
  1. Unit: gastruloid.
  2. Readouts are unchanged from C6-S7 (fraction line scan and crossing, moment elongation).
  3. Used data first: only spent or open summaries (G1b CI, Q3 description). No sealed file.
  4. Power: not applicable; this is a diagnosis, with no test or claim.
  5. Exposure: the Q2 and Q3 targets are exposed. That is why any fix needs fresh checks.
  6. Literature: G1b implication (fixed domains need a feedback that sizes domains: fate switching at the
     boundary or growth compensation).

## C6-S7j: more repeats of the chosen setting and order of events (pre-registered 2026-11-10)

- **Serves: C6-S7** (robustness of its step-0 choice; exploratory). Bence: "do this in the meanwhile" (answer
  to: "After the checks finish, run more repeats and the order-of-events check?"). Scripts: `c6s7j_repeats.py`
  (new), `c6s7c_potts.py` (optional snapshot recording, `snap` key; off by default, so earlier runs are
  unchanged), `c6s7h_potts_lr.py`, `c6s7g_potts_wet.py`, `c6s7f_potts_pcrawl.py`, `c6s7_checks.py`.
- **Synthetic only.** Same +-2 mixes as step 0. No clone data, nothing sealed. Runs after C6-S7i finishes
  (never two Pool(10) jobs at once).
- **J1, repeats:** chosen setting W1, beta 300, lam_F 80, 100k MCS; 30 new seeds (97000+). Three arms:
  - crawling, c 1;
  - lam_F 0;
  - crawling, c 0.

  Report mean, SD and bootstrap 95% CI of polar, multipolar rate, attached and elongation (and AR =
  1/(1-elong)), plus the share of single runs failing S0-a (polar < 0.5 or multipolar).
  **Reading:**
  - **robust** if the 30-seed means still meet S0-a..c;
  - **fragile** otherwise.
  - The 6-seed values (polar 0.88, elongation 0.50) are compared to the CIs.
- **J2, order of events:** chosen setting, c 1, 10 seeds (97500+). A snapshot every 2,000 MCS records polar,
  contact multipolar and elongation.
  - **Definitions:**
    - t_pole = first snapshot with polar >= 0.5 and a single posterior domain;
    - t_elong = first snapshot with elongation >= start + 0.5 x (final - start).
  - **Expectation from the literature** (written before the run): the pole forms first, then elongation. Pole
    by about 96 h from a homogeneous start (Hashmi 2022); round at 72 h, ellipsoid at 96 h, elongated by
    120 h (Fiuza 2024); elongation 96-120 h (Trani-Bustos 2025).
  - **Reading:**
    - **pole first** if t_pole < t_elong in >= 8/10 runs;
    - **elongation first** if the reverse holds in >= 8/10;
    - otherwise **mixed**.
  - MCS are not calibrated to hours, so only the order is read.
- **Cost:** 100 simulations, about 30 min.
- **Pre-flight:**
  1. Unit: synthetic gastruloid.
  2. Readouts unchanged (polar, contact multipolar, attached, moment elongation).
  3. No data used.
  4. Power: not a test; CIs reported.
  5. Exposure: none.
  6. Literature: order-of-events expectation above, read before the run (lit/reading_c6_shape_benchmark.md and
     lit/reading_c6_outer_start.md).

### C6-S7j result (2026-11-10; `results_gastruloid/c6s7j_repeats.json`, `c6s7j_order.png`; synthetic only; run on GitHub Actions https://github.com/krenusz/cellware/actions/runs/42786181450, 11 jobs, 12 min)

- **J1, 30 seeds, chosen setting (W1, beta 300, lam_F 80): ROBUST by the registered reading (means meet S0-a..c),
  but the 6-seed values were on the lucky side.** Values are mean [95% CI], SD:
  - **crawling, c 1:** polar 0.77 [0.65, 0.88], SD 0.34. Multipolar 10% [0, 20%]. Attached 1.00. Elongation
    0.57 [0.51, 0.63] (AR about 2.6).
  - **lam_F 0:** polar 0.77 [0.59, 0.96], SD 0.54. Multipolar 0%. Elongation 0.16.
  - **crawling, c 0:** elongation 0.16.
  - **S0-b:** +0.41. **S0-c:** 0.003.
  - **Single runs:** one in three crawling runs fails one-pole (polar < 0.5 or more than one cap). The polar index
    is bimodal: runs either form a clear pole (> 0.8) or stay low.
  - **6-seed values vs the CIs:** polar 0.88 sits at the upper end of its CI; elongation 0.50 sits just below.
  - **AR about 2.6** is at the high end of real 120 h medians (1.2-2.2, Blotenburg 2025).
- **J2, order of events, 10 seeds: ELONGATION FIRST (8/10).** This is the opposite of the literature expectation
  registered before the run (pole first).
  - t_elong 4-28k MCS vs t_pole 22-80k MCS.
  - Run 5 never reaches half-gain elongation timing (t_elong none); run 6 never forms a pole (t_pole none).
  - Crawling stretches the aggregate within the first 20k MCS, before the posterior cells have gathered.
  - **Several polar traces are still rising at 100k MCS:** the pole has not settled by the end of the run.
- **Reading:**
  - The step-0 pass holds on average, but about a third of single gastruloids lack a clear pole.
  - In the model, **elongation does not depend on the pole having formed**: all cells crawl along their own
    polarity from the start.
  - In real gastruloids, the pole comes first (by about 96 h) and elongation follows (96-120 h).
  - This independence fits the C6-S7h Q2/Q3 failures: the shape is set by crawling strength, not by the
    pattern or the clone state.
  - Next step needs Bence. Candidate (not registered): elongation driven by posterior cells after the pole forms.

### C6-S7i result (2026-11-10; `results_gastruloid/c6s7i_diag.json`; diagnosis only, synthetic + open summaries, nothing sealed; parts on GitHub Actions https://github.com/krenusz/cellware/actions/runs/42788350289, 30 jobs; the D1c means are identical to the earlier local partial run, so Actions reproduces local)

- **D1a: identity rule NOT the cause of Q2.** With perfect sorting and the feedback rule, the slope is 0.07
  (crossings 58-62). It would read as fixed domains. The full simulation gives 0.57, so **the mechanics or the
  readout add it** (registered reading).
- **D1b:** independent identity gives 0.07; g 0 gives 0.37. Feedback already gives fixed domains in ideal
  geometry, so no identity fix is needed.
- **D1c, simulation at the chosen setting:**
  - Clone 1 posterior share rises 0.50 -> 0.87 and clone 3 rises 0.08 -> 0.27 as clone 3 becomes the
    majority. That is the rank rule, as in D1a.
  - **Mean axis position of anterior cells is about 50 at every ratio.** Perfect sorting would put it at about
    31.
  - So the A and P domains are **poorly aligned with the long (PCA) axis** the readout uses.
  - **Interpretation (not a registered reading):** crawling elongates the body before the pole forms (C6-S7j
    J2). So the long axis is often not the A-P axis, and the line scan along it mixes A and P. The crossing then
    tracks composition: the Q2 failure.
- **D2: SATURATION SUPPORTED** (all three registered criteria). Elongation follows c x lam_F:
  - c 0.2 vs c 1 at lam 80: +0.001;
  - c 1 - c 0.2 at lam 20: +0.40;
  - c 0.2 x 80 vs c 1 x 20: -0.03.
  - lam 20: 0.15 / 0.16 / 0.23 / 0.50 / 0.63 at c 0 / 0.1 / 0.2 / 0.5 / 1.
  - lam 80: 0.16 / 0.45 / 0.59 / 0.39 / 0.59. lam 80, c 0.5 is a noisy dip at 10 reps.
  - **The Q3 failure at the chosen setting is saturation:** competence 0.2 x 80 already elongates fully.
- **Q3 at lam_F 20 (descriptive only; target exposed; cannot replace the chosen setting):**
  - Elongation 0.64 / 0.645 / 0.55 / 0.37 / 0.20 at 0 / 20 / 50 / 80 / 100% low-competence clone.
  - It falls with the low-competence share; the first step (+0.005) is flat.
- **Reading:**
  - Both data-check failures come from the elongation mechanism, not from the clone-state or identity rule.
  - **Q3:** crawling at lam 80 saturates.
  - **Q2:** crawling elongates along a random axis before the pole exists, so the measured axis is not the A-P
    axis.
  - **Candidate fix (not registered; needs Bence):** elongation by posterior cells only after the pole forms,
    at a non-saturating strength. It would have to re-pass step 0 and Q1, then Q2/Q3 (targets exposed, so those
    would be capped), with S2 recomputed.

## C6-S7k: posterior-led elongation after the pole forms (pre-registered 2026-11-10)

- **Serves: C6-S7** (amendment, exploratory under H17). Bence: "1" (answer to: "pre-register C6-S7k, run longer
  only, or close C6-S7?"). Scripts: `c6s7k_ppost.py` (new), `c6s7c_potts.py` (kernel loop gains an optional
  crawling onset `t_on`, default 0, so earlier runs are unchanged), `c6s7g_potts_wet.py`, `c6s7f_potts_pcrawl.py`,
  `c6s7_checks.py`, `c6s2_space.py`, `c6s3_break.py`, `c6s4_fit.py`.
- **Motivation (C6-S7i/j):**
  - With all cells crawling from the start, the body elongates before the pole forms (8/10) and along a random
    axis, so Q2 fails.
  - Crawling at lam 80 also saturates, so competence does not matter and Q3 fails.
  - Biology: the pole forms first (by about 96 h); then the posterior tissue extends (96-120 h).
- **Model:** attached Potts (W1) + Guan pull (beta 300), unchanged.
  - **Crawling only in posterior cells** (links P-P only, as C6-S7f).
  - **It switches on at a fixed developmental time t_on.** This is a local timing cue, not a global pole
    detector.
  - 200,000 MCS (J2: the pole was still forming at 100k).
- **Step 0 (synthetic +-2 mixes, 10 reps per point; GitHub Actions):**
  - **Grid:** t_on {50k, 100k} x lam_F {10, 20, 40, 80}. Controls: lam_F 0 (10 reps) and c 0 at every point
    (10 reps). Snapshots every 4,000 MCS in the c 1 runs.
  - **Criteria, all required:**
    - **S0-a:** mean polar >= 0.5 and multipolar (contact) <= 20%; attached >= 0.95.
    - **S0-b:** elongation >= lam0 + 0.15.
    - **S0-c:** c 0 within 0.05 of lam0.
    - **S0-e (new, biology from the literature, registered before any run):** pole first in >= 7/10 runs
      (t_pole < t_elong, definitions as C6-S7j).
  - **Choice:** the passing point with the smallest lam_F, then the earliest t_on (parsimony, as C6-S7). No
    competence or Q3-type criterion is used in the choice.
  - No pass: report; no further amendment without Bence.
- **Data checks at the chosen point (as C6-S7, same seeds, 200k MCS, chunked on Actions):**
  - **Q1 and Q2:** registered criteria unchanged. Spearman is compared with a 1e-9 tolerance: C6-S7h showed the
    float problem at exactly 0.9. This is fixed before this run.
  - **Q3:** registered criteria unchanged.
  - **Exposure:** the Q2 and Q3 targets are exposed (C6-S7h). A pass counts as consistency, capped in the
    credit ledger.
  - **S2:** computed as C6-S7, plus a new robustness condition. **An S2 look is proposable to Bence only if
    power >= 0.8 under BOTH the z 0 and the z = clone 6 state** (C6-S7h flipped sign). Otherwise exploratory.
    The sealed partition stays closed either way until Bence's verbatim OK (H13).
- **Cost:** step 0, 170 sims at 200k MCS; data checks about 1,000 sims. All on GitHub Actions (standing rule).
- **Pre-flight:**
  1. Unit: gastruloid.
  2. Readouts unchanged.
  3. Used data first: open/spent summaries only (G1a fit shard, G1b, C4-S4 pure plate). No sealed file.
  4. Power: S2 from the model's own prediction, with robustness over the two state assumptions.
  5. Exposure: Q1 targets used before; Q2/Q3 targets exposed. Hence capped credit.
  6. Literature: order of events (Hashmi 2022, Fiuza 2024, Trani-Bustos 2025); posterior-led axis extension
     (Benazeraf 2010 motility gradient). **Risk:** posterior-only crawling did not elongate in C6-S7f, but that
     run had a dissociating body and no long-range pull.

### C6-S7k result (2026-11-10; `results_gastruloid/c6s7k_step0.json`, `c6s7k_step0.png`; synthetic only; GitHub Actions https://github.com/krenusz/cellware/actions/runs/42799531971, 34 jobs)

- **Step 0: FAIL, no point passes.** Per the registration, there are no data checks and no further amendment
  without Bence.
- **Pole: fixed by the longer runs.**
  - At 200k MCS every point has polar 1.15-1.38, multipolar 0% and attached 1.00.
  - The no-crawling control has polar 1.08 and elongation 0.20.
  - Almost all single runs reach a clear pole by 200k. The pole forms at 10-160k: "a third of runs without a
    pole" in C6-S7j was mostly runs stopped too early.
- **Elongation: posterior-only crawling does not elongate (S0-b fails everywhere).**
  - Elongation 0.16-0.28 against 0.20 without crawling (0.35 needed).
  - Competence 0 gives 0.18-0.28 (S0-c passes except t50k lam80).
  - Elongation stays noisy at 0.1-0.4 after onset, with no consistent rise.
- **Pole-first (S0-e):** 1-6/10, which fails. With no real elongation, t_elong only marks noise, so this criterion
  is uninformative here.
- **Reading:**
  - Posterior cells crawling only among themselves pull within the cap. Surface tension of the attached body
    rounds it again, and nothing pushes the tissue along the axis.
  - Elongation in this model needs force across the whole body (C6-S7h: all cells crawl) or an axis-oriented
    mechanism. Posterior-only crawling with random, locally aligned polarity is not enough. This agrees with
    C6-S7f.
  - A robust single pole needs only the long-range pull, the attached energies and enough time (200k MCS).

## C6-S7l: elongation after the pole, all-cell vs axis-oriented crawling, with an axis-alignment criterion (pre-registered 2026-11-10)

- **Serves: C6-S7** (amendment, exploratory under H17). Bence: "1" (answer to: "pre-register C6-S7l, two
  elongation mechanisms in one step 0, or close C6-S7?"). Scripts: `c6s7l_axis.py` (new), `c6s7c_potts.py`
  (optional `orient` polarity cue, default off, so earlier runs are unchanged), `c6s7k_ppost.py` (reused
  helpers), `c6s7g_potts_wet.py`, `c6s7f_potts_pcrawl.py`, `c6s7_checks.py`, `c6s2_space.py`, `c6s3_break.py`,
  `c6s4_fit.py`.
- **Motivation:**
  - C6-S7k: posterior-only crawling does not elongate.
  - C6-S7j/i: crawling from the start elongates before the pole and along a random axis (Q2), and lam 80
    saturates (Q3).
  - C6-S7k: the pole is robust by 200k MCS.
- **Common:** attached Potts W1, Guan pull beta 300, crawling switched on at t_on, 250,000 MCS.
- **Arms:**
  - **(a) all-cell crawling, random Vicsek polarity** (C6-S7h mechanism, delayed): t_on {100k, 150k} x lam_F
    {20, 40}.
  - **(b) posterior-only, mediolateral polarity:** each P cell with anterior contacts sets its polarity
    perpendicular to (own centroid - mean centroid of its anterior contacts). Cells without such contacts keep
    their Vicsek polarity. Crawling links pull along the polarity, so the tissue converges sideways and extends
    along A-P: convergent extension, a local PCP-like cue. t_on 100k x lam_F {20, 40, 80}.
  - **(c) all cells, mediolateral polarity:** the same cue for every cell, using its other-type contacts.
    t_on 100k x lam_F {20, 40}.
- **Step 0:**
  - 10 reps per point, c 1 with snapshots every 5,000 MCS, plus c 0 at every point.
  - lam_F 0 control (250k, 10 reps). GitHub Actions.
- **Criteria, all required:**
  - **S0-a:** polar >= 0.5, multipolar <= 20%, attached >= 0.95.
  - **S0-b:** elongation >= lam0 + 0.15.
  - **S0-c:** c 0 within 0.05 of lam0.
  - **S0-e:** pole first in >= 7/10.
  - **S0-f (new, targets the Q2 cause without Q2 data):** mean |cos| between the body's long axis (PCA of the
    mask) and the pole direction (P centroid - A centroid) >= 0.8.
- **Choice:** smallest lam_F, then arm a < b < c (simplest), then earliest t_on. No competence or Q3-type
  criterion. No pass: report, stop, Bence decides.
- **Data checks at the chosen point:** exactly as C6-S7k stage 2 (Q1 with 1e-9 tolerance; Q2/Q3 exposed, so
  capped credit; S2 proposable to Bence only if power >= 0.8 under both z 0 and z = clone 6; sealed stays
  closed until Bence's verbatim OK, H13). Chunked on Actions.
- **Cost:** 190 sims at 250k MCS for step 0, about 1,000 for the checks. All on Actions.
- **Pre-flight:**
  1. Unit: gastruloid.
  2. Readouts unchanged, plus the axis-alignment readout (synthetic only).
  3. Open/spent summaries only. No sealed file.
  4. Power: S2 robust rule as C6-S7k.
  5. Exposure: as C6-S7k.
  6. Literature: convergent extension by mediolateral intercalation (Keller; PCP) and the order of events
     (Hashmi 2022, Fiuza 2024). **Unverified:** whether gastruloid elongation is mainly convergent extension
     rather than growth (Trani-Bustos 2025 stresses proliferation). Both arms are models, not claims.

### C6-S7l amendment: early-stop rule (2026-11-10, written before any data-check result was looked at)

- Bence: "we can win 40 minutes to think about other possibilities if it fails". No partial result of run
  42812934801 has been opened.
- **Rule:**
  - Evaluate it once all 20 Q1 jobs (40 gastruloids per clone) are done.
  - **If Q1 has <= 3/5 signs right, cancel the remaining jobs.** Without correct clone placement the S2 power is
    meaningless. Report Q1, plus Q2/Q3 if already complete.
  - **Otherwise let the run finish.**
  - Q2 and Q3 are never used to stop; they are reported whatever they show.
- **Withdrawn (same day, still before any look):** Bence: "q1 2 3 answers different questions so maybe we
  shouldnt kill it". Correct: S2 measures mix vs pure shape, which does not depend on clone placement (Q1).
  **The run finishes in full.** Q1 may be reported as soon as its 20 jobs are done (an early read, no decision
  attached).

### C6-S7l result (2026-11-10; `results_gastruloid/c6s7l_step0.json`, `c6s7l_checks.json`, `c6s7l_step0.png`; synthetic + open/spent summaries; nothing sealed; Actions runs 42808970551 (step 0, 38 jobs) and 42812934801 (checks, 100 jobs))

- **Step 0: PASS** (3/9). The chosen point is arm a, t_on 150k, lam_F 20 (registered rule).
  - **Chosen point:** polar 1.13, multipolar 0%, attached 1.00, elongation 0.74 (AR about 3.8), alignment
    0.95, pole first 8/10.
  - **Arm c (mediolateral, all cells) also passes:** alignment 0.97-0.99, pole first 9-10/10.
  - **Arm b (posterior only) fails:** no elongation.
  - **Caveat:** in about 4/10 runs of the chosen point the pole appears only at crawling onset.
  - **Caveat:** AR 3.8 is above typical real 120 h medians (1.2-2.2).
- **Q1: PASS.**
  - Signs 5/5. Spearman 0.90, exactly the threshold again: clones 3 and 5 swapped. Passes with the registered
    1e-9 tolerance.
  - Magnitudes are 43-64% of the targets (clone 7: 32%): sim -0.40 / 0.52 / 0.47 / -0.73 / 0.23 vs -0.92 /
    0.88 / 1.05 / -1.13 / 0.73. In C6-S7h they were about 1/3.
  - Multipolar 0%, polar 0.85-0.94. The targets are not unseen.
- **Q2: PASS.** Crossings 34 / 43 / 45 / 46 / 46 over f_anterior 0.2-0.8; slope 0.18. The G1b CI is [-0.40, 0.42];
  the data slope is 0.17. C6-S7h gave 0.57.
- **Q3: PASS** (monotone fall). Elongation 0.71 / 0.66 / 0.60 / 0.38 / 0.26 at 0 / 20 / 50 / 80 / 100%
  low-competence clone.
- **Q2/Q3 targets were exposed** (C6-S7h). The fix targeted the C6-S7i causes (axis misalignment, saturation),
  not the targets, but these are consistency checks, not tests.
- **S2: NOT POWERED, so EXPLORATORY.**
  - The model's pure gap (bulk - 6a, 1 - circularity) is 0.40; the spent data gap is 0.18. The model now
    separates the pure plates clearly.
  - The mix sits slightly above the pure mean, in the same direction under both state assumptions:
    - **z 0:** +0.10 of the gap, i.e. 0.019 in data units, power 0.16;
    - **z clone 6:** +0.28, i.e. 0.051, power 0.54.
  - **The robust rule fails** (both must be >= 0.8). The sealed partition stays closed.
- **Reading:**
  - The model passes all three open data checks. It does so with the clone-state and identity rule unchanged
    since C6-S7, once elongation starts after the pole at a non-saturating strength.
  - The C6-S7i diagnosis predicted this.
  - There is still no powered test on unseen data: the predicted mix effect is small (0.02-0.05 against SE
    0.029).
  - Per H18 (RETRO_2026-11-10), this is the 12th version; a pass is discounted in the credit ledger.

## C6-S7m: sensitivity of the C6-S7l data checks to arm c, mediolateral all-cell crawling (pre-registered 2026-11-10)

- **Serves: C6-S7** (sensitivity of its chosen setting; exploratory). Bence: "go 2 3" (answer to: "2. read the
  Marangoni preprint; 3. arm c as a sensitivity check"). Scripts: `c6s7m_armc.py` (new, wraps `c6s7l_axis.py`),
  `c6s7l_axis.py` (checks output file name becomes a module variable; logic unchanged), `c6s7c_potts.py`,
  `c6s7g_potts_wet.py`, `c6s7f_potts_pcrawl.py`, `c6s7_checks.py`, `c6s2_space.py`, `c6s3_break.py`, `c6s4_fit.py`.
- **H18:** this is the 13th version within C6-S7. It cites RETRO_2026-11-10. **It is a sensitivity run of an
  already-passing step-0 point, not a new model.** Arm c passed step 0 in C6-S7l (c_t100k_lam20: polar 1.52,
  alignment 0.97, pole first 10/10).
- **Point:** arm c, t_on 100k, lam_F 20. This is the smallest passing lam_F in arm c, fixed now before any run.
- **Checks:** Q1, Q2, Q3 and S2, exactly as C6-S7l: same code, same seeds, 250k MCS, chunked on Actions.
- **Readings:**
  - Report each check next to C6-S7l.
  - **Robust** if Q1, Q2 and Q3 keep their C6-S7l verdicts.
  - **S2:** report power under both states. **Arm c cannot replace the registered choice for a sealed look.** If
    arm c alone reaches robust power >= 0.8, that is reported as selection-dependent, and any S2 proposal to Bence
    must say so (H13 last partition, Bence's verbatim OK).
- **Pre-flight:**
  1. Unit: gastruloid.
  2. Readouts unchanged.
  3. Open/spent summaries only. No sealed file.
  4. Power: as C6-S7l.
  5. Exposure: Q1-Q3 targets seen; C6-S7l results seen. Hence sensitivity only.
  6. Literature: convergent extension (Keller; PCP).

### C6-S7m result (2026-11-10; `results_gastruloid/c6s7m_checks.json`; sensitivity only; Actions run 42830280687, 100 jobs; nothing sealed)

Arm c (mediolateral polarity, all cells, t_on 100k, lam_F 20) vs C6-S7l's chosen arm a:

| check | C6-S7l (arm a) | C6-S7m (arm c) |
|---|---|---|
| Q1 | pass: 5/5, rho 0.90, magnitude 43-64% | **pass:** 5/5, rho 0.90 (3/5 swapped again), magnitude **57-84%** (clone 7: 46%); polar 1.49-1.57, multipolar 0% |
| Q2 | pass: slope 0.18, crossings 34-46 smooth | **pass:** slope 0.31, inside CI [-0.40, 0.42], but crossings **erratic** (34 / 76 / 55 / 73 / 60) |
| Q3 | pass: 0.71 -> 0.26, monotone | **FAIL (registered strict monotone):** 0.81 / 0.815 / 0.68 / 0.65 / 0.56. The first step rises by 0.008 and the overall fall is shallow: 100% low-competence still elongates 0.56 |
| S2 power (z 0 / z clone 6) | 0.16 / 0.54 | 0.29 / **0.81**, not robust (z 0 < 0.8) |

- **Registered reading: NOT ROBUST.** Q3 changes verdict.
  - Q1 holds and improves.
  - Q2 holds but is noisy.
  - Q3 depends on the arm. Mediolateral crawling elongates more strongly (0.81 at 0%), and competence 0.2 no
    longer slows it much: closer to saturation again.
- **S2:** arm c reaches 0.81 under the clone-6 state only.
  - That is selection-dependent and fails the both-states rule, so it is **not usable** for a sealed proposal (as
    registered).
- **Consequence:**
  - C6-S7l's Q3 pass depends on the elongation mechanism and its strength, not only on the clone state.
  - C6-S7l's Q1/Q2 consistency holds across both arms.
  - No arm gives a robustly powered S2.

## C6-S7 closed (2026-11-10)

Bence: "Yes, close and plan C6-S8". Closed as **exploratory**: summary and C6-S8 plan in CHAPTER6_PLAN.md,
credit in CREDIT_C6.md, retro in RETRO_2026-11-10.md. Sealed sets untouched (S1, S2 partition C, all others).

## C6-S8a: power inventory for an unseen test at the C6-S7l model (pre-registered 2026-11-10)

- **Serves: C6-S8** (FOCUS.json). Bence: "Go C6-S8a". Scripts: `c6s8a_power.py` (new), `c6s7l_axis.py`,
  `c6s7_checks.py`, `core_model.py`, `c6s2_space.py`, `c6s3_break.py`, `c6s4_fit.py`, `c6s7c_potts.py`,
  `c6s7g_potts_wet.py`, `c6s7f_potts_pcrawl.py`.
- **Synthetic only.** No data statistic, no sealed file opened. GitHub Actions.
- **Models:**
  - arm a = C6-S7l chosen (t_on 150k, lam_F 20);
  - arm c = C6-S7m (mediolateral, t_on 100k, lam_F 20);
  - both 250k MCS.
- **S1 (sealed Jan 2024 clone 1+6 pairs, passage 4, n 34):**
  - **Statistic (as C6-S4):** share of 50:50 clone 1+6 gastruloids with clone-1 polarity > clone-6 polarity.
    States are carried to passage 4 with `core_model.memory_through` (code copied from c6s4_fit.job_s1).
  - **Retention:** r 0.46 only. That is the lowest of C6-S4's {0.46, 0.60, 0.68}, which gives the lowest share
    (registered rule: the lowest share is the prediction; monotonicity assumed and disclosed).
  - 200 gastruloids per arm (binomial SE about 0.035).
  - **Power:** one-sided binomial at alpha 0.05, n 34 (needs share >= about 0.72).
- **C (3s1 + bulk chimera; partly exposed):**
  - **What the data is:** the same data as the clone 3.1 rows of the authors' locked revision table
    (`rev_chimera_elong/data/geo_metrics.csv`).
  - **Exposure:** the authors' README (read in C4-S1) says elongation falls with % pure clone, so **the direction
    is known**. The % is plate-confounded (C4-S1).
  - **Counts read from label columns only:** 3.1 at 20 / 50 / 80% = 9 / 16 / 18 gastruloids, with no pure
    plates. No value column was read.
  - **Statistic:** OLS slope of per-gastruloid 1 - circularity on 3s1 fraction (0.2 / 0.5 / 0.8 at those counts),
    one-sided for a negative slope.
  - **Model prediction:** 20 gastruloids per fraction, (3s1 fraction f, z, c3) + (bulk 1 - f).
    - **c3 (fixed now, from OPEN pure-clone morphology scores):** same-date clone 3 / bulk elongated-fraction
      ratio **0.42** (4 dates); sensitivity 0.50 (all dates).
    - **State z:** 0 or Z["3"].
    - **Conversion to data units:** x (0.180 / the arm's model pure gap). The pure gaps are 0.401 (arm a,
      C6-S7l S2) and 0.352 (arm c, C6-S7m).
  - **Noise:** per-gastruloid SD 0.084 (C4-S4: +0.021 = 0.25 SD), so slope SE = 0.084 / sqrt(sum (f - fbar)^2)
    = 0.056 per unit fraction.
  - **Disclosed:** the elongated-fraction to competence mapping is untested; for 6a it would give 0.6, where we
    use 0.2.
- **Gate:** robustly powered only if power >= 0.8 at **every** combination: both arms; for C, both z and both c3;
  for S1, r 0.46.
  - A powered S1 becomes C6-S8b (Bence's verbatim OK; oracle only).
  - A powered C is still only **partial** evidence (direction exposed, plate confound) and is labelled so.
  - If nothing is powered, chapter 6 closes as exploratory with a write-up.
- **Cost:** S1 400 sims + C 480 sims at 250k MCS, chunks of 10 on Actions.
- **Pre-flight:**
  1. Unit: gastruloid.
  2. Readouts: polarity line scan (S1); 1 - circularity (C).
  3. Only open and spent summaries are used as inputs (open morphology CSV, C4-S4 spent SD, C6-S7l/m model gaps).
     Locked file: label counts only.
  4. Power: this step.
  5. Exposure: S1 sealed and unseen; C direction exposed.
  6. Literature: none new.

### C6-S8a result (2026-11-10; `results_gastruloid/c6s8a_power.json`; synthetic only; Actions run 42852610362, 88 jobs; nothing sealed opened)

- **S1 (sealed 1+6 pairs, passage 4, n 34): NOT POWERED.**
  - The predicted share of gastruloids with clone 1 anterior to clone 6 is **0.45 in both arms** (r 0.46,
    200 each), i.e. chance. Power 0.006 (23/34 needed).
  - After 3 passages of memory decay, clones 1 and 6 (both posterior-biased) have nearly the same state, so the
    model predicts no ordering.
- **C (3s1 + bulk; partly exposed): NOT ROBUSTLY POWERED.** Power **0.34-0.80** across the 8 combinations
  (arm x state x c3). The predicted slope in data units is -0.07 to -0.14 per unit fraction, against SE 0.056.
  Only arm a with z = clone 3 and c3 0.42 reaches 0.80.
- **S2:** not powered (C6-S7l / C6-S7m).
- **Gate: no candidate is robustly powered.** Per the C6-S8 plan, **chapter 6 closes as exploratory** with a
  write-up.
- Public data search (`lit/data_search_c6s8.md`): no outside dataset with clone-labelled gastruloids and
  per-gastruloid spatial data.
- All sealed sets stay untouched and remain available for a sharper model or new data: S1, S2 partition C, the
  revision chimera set.

## C6-S8c: S1 power with clone-specific memory from spent G6a (pre-registered 2026-11-10)

- **Serves: C6-S8.** Bence: "go both" (answer to: "1. re-run S1 with clone-specific memory; 2. a non-clone
  unseen readout from the Raj 2026 atlas"). Scripts: `c6s8c_s1mem.py` (new), `c6s8a_power.py` (job reused),
  `c6s7l_axis.py`, `c6s7_checks.py`, `c6s2_space.py`, `c6s3_break.py`, `c6s4_fit.py`, `c6s7c_potts.py`,
  `c6s7g_potts_wet.py`, `c6s7f_potts_pcrawl.py`.
- **Correction disclosed:**
  - C6-S8a carried states to passage 4 with one shared retention, as in C6-S4.
  - The spent G6a look (2026-11-06) had already rejected a single rate. At passage 4 / passage 1, clone 1 was
    -0.06 [-0.37, 0.24] (propensity gone) and clone 6 0.60 [0.18, 1.00]. G6 said to replace the single rate with
    clone-specific retention fitted on non-sealed data.
  - This step does that. Sealed S1 itself is not read.
- **Statistic (unchanged from C6-S4/C6-S8a):** share of 50:50 clone 1+6 gastruloids with clone-1 polarity >
  clone-6 polarity. One-sided binomial power at alpha 0.05, n 34 (needs 23/34, share >= about 0.72).
- **Passage-4 clone mean states (clone mean z scaled; cells as in `c6s3_break.gastruloid`):**
  - **V1 (G6a point estimates):** z1 x (-0.06), z6 x 0.60.
  - **V2 (G6a CI ends least favourable to ordering):** z1 x 0.24, z6 x 0.18.
  - **V3 (series sensitivity: G6b Jan 2024 pooled pair retention, applied uniformly):** z1 x 0.68, z6 x 0.68.
- **Models and size:**
  - arm a (C6-S7l chosen) and arm c (C6-S7m) for V1 and V2; arm a only for V3.
  - 120 gastruloids per combination (binomial SE about 0.045). New seeds 121000+. Chunks of 8, NPROC 4,
    GitHub Actions.
- **Gate:** S1 becomes proposable (C6-S8b: its own pre-registration, oracle only, Bence's verbatim OK) only if
  power >= 0.8 under **V1 and V2 in both arms**.
  - V3 is reported. If V3 fails, any proposal must state that S1's series may fade uniformly, in which case the
    model predicts no ordering.
  - Otherwise chapter 6 closes as exploratory (C6-S8a reading stands).
- **Pre-flight:**
  1. Unit: gastruloid.
  2. Readout: polarity line scan.
  3. Used data: spent G6a/G6b summaries only. S1 sealed, unread.
  4. Power: this step.
  5. Exposure: G6a/G6b results were seen (calibration); S1 unseen.
  6. Literature: none new.

### C6-S8c result (2026-11-10)

- Actions run https://github.com/krenusz/cellware/actions/runs/42875656181 (the first run, 42874409962, failed on
  CR characters in the sweep file before any compute).
- Share of clone 1 anterior of clone 6, n 120 sims each (power at n 34, k_crit 23):

  | combo | share | power |
  |---|---|---|
  | arm a, V1 (G6a point) | 0.675 | 0.57 |
  | arm c, V1 | 0.717 | 0.77 |
  | arm a, V2 (G6a CI ends) | 0.533 | 0.07 |
  | arm c, V2 | 0.575 | 0.15 |
  | arm a, V3 (uniform 0.68) | 0.533 | 0.07 |

- **Gate failed** (min power over V1/V2 0.07). S1 is not proposable. Clone-specific memory lifts the share from
  0.45 (C6-S8a) to 0.68-0.72 at the G6a point estimates, but even there power stays below 0.8. At the
  uncertainty ends and under uniform retention it is near chance.
- Reading: the S1 ordering prediction rests on clone 1 losing its memory faster than clone 6. G6a cannot pin
  that down (CIs overlap), so S1 cannot test the model robustly. C6-S8a's conclusion stands; S1 stays sealed.

## C6-S8d: seeding-size effect on elongation, Triandafillou 2026 atlas (pre-registered 2026-11-10)

- **Serves: C6-S8** (an unseen readout for the C6-S7l model). Bence: "1 + 2 together" (answer to: pre-register
  atlas test 1 = axis alignment and test 2 = 100 vs 300 cells). Scripts: `c6s8d_size.py` (new), `c6s7l_axis.py`,
  `c6s7_checks.py`, `c6s3_break.py`, `c6s2_space.py`, `c6s7c_potts.py`, `c6s7g_potts_wet.py`,
  `c6s7f_potts_pcrawl.py`, `c6s8a_power.py`.
- **Data:** `data/ext_triandafillou/members/` (Zenodo 10.5281/zenodo.22233771, CC BY 4.0), sha256-locked in
  `SHA256SUMS` before any statistic.
  - **Read so far:** headers, row counts, JSON structure, tag vocabularies and tag counts. No value has been read.
  - Notebook outputs are not opened. Only source lines matching "A_area|P_area|AP axis" were grepped
    (2026-11-10).
- **Test 1 (axis alignment) is NOT pre-registered here. Disclosed reason:**
  - In the 25 seqFISH sections with `areas.json`, the A_area and P_area are hand-drawn polygons.
  - The authors' code places cells along a hand-drawn AP line rescaled at a T landmark. That suggests the A/P
    areas follow the AP line, which itself follows the shape.
  - If so, alignment between the long axis and the P-A direction is close to circular.
  - The model runs below still record `align`, P share and multipolarity, so an alternative test 1 can be
    pre-registered later (Bence's call) without new runs.
- **Test 2: data readout.**
  - All 6 brightfield plates at 120 h:
    - 300 cells: 2024-01-19 (n 88) and 2024-09-01 (n 86), n 174;
    - 100 cells: 2025-04-04 wtp2 A/B and wtp3 A/B (93, 88, 84, 90), n 355.
  - Each `gastruloid` polygon in the `*.json` files is rasterised at 1 px with skimage `polygon` and passed
    through `c6s3_break.shape_from_mask`.
  - **Primary: `moment_elong`** (1 - sqrt(ev_min/ev_max); scale-free, the same function as the model).
    Secondary: `one_minus_circ`.
  - No exclusions. Polygons with fewer than 3 vertices are dropped (none expected).
- **Statistic:** one-sided Welch t on `moment_elong`, mean(100 cells) - mean(300 cells), in the direction the model
  predicts, alpha 0.05, gastruloid as the unit.
  - Secondary (descriptive): the 6 plate means. Whether all four 100-cell plates fall on the predicted side of
    both 300-cell plates. A 2 vs 4 rank test cannot reach 0.05 (min p 1/15).
- **Model prediction:**
  - **Model:** arm a (C6-S7l chosen: W1 + beta 300 + all-cell crawling from 150k, lam_F 20, 250k MCS) is primary.
    Arm c (C6-S7m) is the sensitivity check.
  - **Cells:** wild-type bulk, all cells competence 1, state z 0 (`parts = [("bulk", 1.0, None, 1.0)]`).
  - **Size:** N 300 (the model's default, matching 300 seeded cells) vs N 100. Every per-cell parameter and the MCS
    count are unchanged.
  - **Disclosed assumption:** model cell number is proportional to seeded cells, and 250k MCS = 120 h at both
    sizes.
  - 64 gastruloids per (arm, N), 256 sims, seeds 131000+, 8 per chunk on GitHub Actions.
  - Also recorded per sim: `one_minus_circ`, `align`, P cell fraction, multipolar.
- **Power (H17, from the model's own prediction):**
  - d = Delta_model / SD_model, where SD_model is the pooled per-gastruloid SD of `moment_elong` from the sims.
  - Power = one-sided Welch at n 355 / 174.
- **Gate for proposing the one look (C6-S8e, Bence's verbatim OK, a single scripted read):**
  - power >= 0.8 in **both** arms, and both arms predict the **same sign**.
  - If the arms disagree, or either has power < 0.8, there is no test, and chapter 6 closes as exploratory.
- **Scoring if the look happens:**
  - Pass = significant in the predicted direction.
  - Also report the data Delta against the model's 90% interval for Delta (bootstrap over sims).
- **Disclosed limits:**
  - Size is fully confounded with date and passage (2024 vs 2025 plates), so a pass is consistent with the model,
    not causal.
  - The direction of size effects on elongation appears in the literature (Fiuza 2024 size series, read
    earlier), so the direction is partly exposed. Only the magnitude is unseen.
  - E14 wild-type vs the Raj clone lines: different backgrounds.
  - Counts for Form/Unseen only, not the clone claim.
- **Pre-flight:**
  1. Unit: gastruloid.
  2. Readout: polygon `moment_elong`.
  3. Used data: none (synthetic first).
  4. Power: this step.
  5. Exposure: atlas values unseen; direction partly exposed (literature).
  6. Literature: Fiuza 2024 size series (already read).

### C6-S8d result (2026-11-10)

- Actions run https://github.com/krenusz/cellware/actions/runs/42888098598. 64 sims per (arm, N). No atlas value
  has been read.

  | arm | moment_elong N 300 | N 100 | Delta (100 - 300) [90% CI] | d | power |
  |---|---|---|---|---|---|
  | a (all-cell random crawling) | 0.710 | 0.487 | -0.223 [-0.245, -0.201] | -2.9 | 1.00 |
  | c (mediolateral crawling) | 0.716 | 0.809 | +0.093 [+0.054, +0.132] | +0.7 | 1.00 |

- **Gate failed: the arms predict opposite signs.** Smaller gastruloids elongate less under random crawling and
  more under mediolateral crawling. The model's size prediction depends on the crawling mode, which the data
  so far cannot fix, so there is no test. The atlas stays sealed (values unread).
- Side readings (model only): alignment drops at N 100 under arm a (0.58 vs 0.82) and stays high under arm c.
  P fraction is 0.398 at both sizes; no multipolar runs.
- Under the C6-S8 plan, chapter 6 now closes as exploratory with a write-up.

## C7-S2: is blood fate a heritable clone state in CellTag-multi? Sister wells at day 5 (pre-registered 2026-11-11)

- **Serves: C7-S2** (CHAPTER7_PLAN.md). Before asking *where* the clone fate state is carried (C7-S3), test
  that one exists here. Bence: "Go C7-S2". Scripts: `c7s2_sister.py` (new), `sis_data.py` (functions
  `heritable_share`, `clone_logit_sd`, `bootstrap`, `synthetic`, reused unchanged), `partitions.py`,
  `test_oracle.py`.
- **Data:**
  - `data/ext_celltag_multi/hsc.rna_atac.r1_2_master_v2.csv` (sha256-locked), partition `celltag_lsk`
    (unit = clone).
  - Day-5 **RNA** cells only. The well comes from the barcode prefix: `d5r1-` = sister well 1, `d5r2-` = sister
    well 2. Day-5 ATAC has no well label, so it is not used.
  - The fate label is `cell_type2` of each day-5 cell.
- **Fate groups (fixed now, from the level names only):**

  | group | `cell_type2` levels |
  |---|---|
  | Neu | Neutro |
  | Mono | Mono |
  | BEM | Baso/Eos/Mast |
  | EMk | Ery/Meg + MEP |
  | DC | Ccr7_DC + pDC |
  | Lym | Lym |
  | Prog | MPP + MPP/GMP |

  - Cells labelled `Unk_1`/`Unk_2`/`Unk_3`/`uns` are dropped before counting.
  - **Sensitivity:** differentiated groups only (Prog dropped, 6 groups).
- **Units:** clones with >= 4 counted day-5 RNA cells in **both** wells (B38 used > 3).
- **Statistics (exactly as B38, `sis_data` functions):**
  - **A.** H = (D_rand - D_sis) / (D_rand - D_noise). Total-variation distance; D_rand comes from random
    well-1 / well-2 pairs (one group: one experiment); D_noise from multinomial redraws from the clone's pooled
    fractions.
  - **B.** Between-clone logit SD per group from the sister covariance.
  - 95% CIs by bootstrap over clones (1,000).
- **Synthetic check first (train only):** syn0, no memory (expect the H CI to include 0); syn1, full memory
  (expect H about 1).
- **Train (`train_only`):**
  - all statistics plus the sensitivity analysis;
  - H12 selection file `selection/C7-S2.json`: two candidate models of well 2 given well 1, scored by per-cell
    log-likelihood on train clones in both directions:
    - **memory**: p = (k_other + pi) / (n_other + 1), pi the population fractions;
    - **no memory**: p = pi.
    The chosen model is the better one.
  - **H14 power block**, one-sided alpha 0.025 (the 95% lower bound) for H > 0.5, from the train H and its
    bootstrap SE: sd = SE x sqrt(n_train), effect = H_train - 0.5.
- **Stopping rule (from the plan):** if the train H CI includes 0, the chapter stops, and there is no look.
- **Sealed look:** `test_oracle.py look celltag_c7s2 --exp C7-S2 --partition A`. One look, partition A only;
  B and C stay sealed for C7-S3.
- **Readings (on A):**
  - **Fate largely fixed in the clone:** H >= 0.7 and lower bound > 0.5.
  - **Partly fixed:** lower bound > 0 but not the first reading.
  - **No clone memory:** CI includes 0. The chapter stops.
  - **Cross-check with LARRY / SIS:** logit SD >= 2 for at least two groups = a strong clone state, as in #1/#13.
- **Pre-flight:**
  1. **Unit:** clone; the two wells are its sisters.
  2. **Readout:** fate composition from the authors' labels (their label transfer and clustering; not ours).
  3. **Used data first:** none; this data is new to us. Expected size from SIS: H 0.71 [0.59, 0.78].
  4. **Power:** from train, H14 block above; the oracle refuses below 0.8.
  5. **Exposure:**
     - level names of the fate columns only; no counts;
     - the paper's results and figures are unread;
     - the abstract mentions "lineage priming" (disclosed in the plan).
  6. **Literature:** Weinreb 2020 (sister fate agreement in LARRY) and Tian 2021 (SIS) give the same design.
- **Known caveat:** the two wells were split at day 2.5, after tagging during days 0-2. A clone tagged late
  shares fewer divisions before the split, which would lower H, not raise it.

### C7-S2 train result (2026-11-11; `results_celltag/c7s2_train*.json`; train partition only, nothing sealed read)

- **Synthetic checks pass:** syn0 (no memory) H -0.36 [-1.99, 0.44], CI includes 0; syn1 (full memory)
  H 1.02 [0.98, 1.07].
- **Train, 144 clones with >= 4 cells in both wells:** D_sis 0.29, D_rand 0.53, D_noise 0.17, **H 0.66
  [0.58, 0.75]**.
  - Logit SD: Mono 1.50, Prog 1.38, Neu 0.71, EMk 0.62, Lym 0.42, BEM 0.25, DC 0.12.
  - Differentiated-only (73 clones): H 0.53 [0.27, 0.71].
- **Stopping rule not triggered** (the train CI excludes 0). Selection: memory -0.972 vs no memory -1.073
  per-cell log-likelihood, so memory is chosen.
- **Power block (`selection/C7-S2.json`):** sd 0.505, effect 0.164 (H > 0.5), alpha 0.025 one-sided.
- **Counts amendment (before any look):** the oracle count is the structural count scaled down by the train
  ratio of eligible clones after and before the fate-label exclusions. Otherwise it would overstate n and power.
- **Reading so far:** a clone fate state exists in this culture, weaker than in LARRY/SIS (no group reaches
  logit SD 2). The sealed look decides "fixed" vs "partly fixed".

### C7-S2 look refused by the power gate (2026-11-11)

- `test_oracle.py look celltag_c7s2 --exp C7-S2 --partition A`: counts 67 clones (conservative scaling), power
  **0.76 < 0.8**. **Refused; partition A NOT spent** (H14). No sealed value was read.
- C7-S2 therefore stands on train only: H 0.66 [0.58, 0.75], so a clone fate state exists (stopping rule not
  triggered). "Fixed vs partly fixed" is unconfirmed.
- Options, for Bence:
  1. **No sealed look:** keep A, B and C for C7-S3, the chapter's main test.
  2. A union A+B look (power about 0.97), which leaves only C for C7-S3.
  3. Pre-register a weaker claim (H > 0) on A.

## C7-S2a: what limits H and its power? Train-only diagnostic of the C7-S2 readout (pre-registered 2026-11-11)

- **Serves: C7-S2.** The look was refused at power 0.76 with H 0.66, versus SIS 0.71 and LARRY 0.88. Bence:
  "explore what we could miss that we cant produce the sis or larry numbers; check for possibilities how to
  enhance the train exp without touching sealed data". Scripts: `c7s2a_diag.py` (new), `c7s2_sister.py`,
  `sis_data.py`, `partitions.py`.
- **Train only** (`train_only("celltag_lsk")`). No sealed partition is touched, and `counts()` is not called.
- **Known before running (structure / metadata only):**
  - **GEO:** all four day-5 scATAC lanes are **replicate 1** (GSM6675426-29 titles). Day-5 ATAC cells can
    therefore only add to well 1.
  - **LARRY's 0.88 is not an H.** It is the cross-well correlation of the neutrophil fraction N/(N+M) (K3; the in
    vitro cross-well r was 0.74). SIS's 0.71 is the same H as ours. **The like-for-like comparison is computed
    here.**
- **Grid** (every cell reported; each with n, H, 95% CI, bootstrap SE, projected power at the partition size):
  1. **Minimum cells per well:** 2, 3, 4 (pre-registered C7-S2), 6, 8.
  2. **Grouping:**
     - G7 (C7-S2);
     - G6 differentiated only;
     - G4 coarse: Neu, Mono, Prog, Other = EMk + BEM + DC + Lym. This reduces sparse categories.
  3. **Clone quality:** all vs `edge.den` >= train median. Edge density is the authors' clone-graph density;
     low density can mean merged false clones, which would lower H.
  4. **Well 1 cells:** RNA only vs RNA + day-5 ATAC (authors' transferred labels).
  5. **LARRY-comparable statistic:** Pearson r across wells of N/(N+M), clones with >= k N+M cells in both
     wells (k = 2, 4), with bootstrap CI.
- **Projected power:** the C7-S2 formula, with partition size = train n x (partition / train clone ratio
  1348 / 2720), H > 0.5 one-sided alpha 0.025.
- **What this step may and may not do:**
  - It describes which factors lower H or its precision (noise per well, sparse fates, clone-call quality,
    modality).
  - If a variant would be used for a sealed look, it becomes **C7-S2b with its own pre-registration**, and
    the number of configurations tried here (5 x 3 x 2 x 2 = 60, plus 2) is disclosed as a forking path.
  - **No sealed look in this step.**
- **Pre-flight:**
  1. **Unit:** clone.
  2. **Readout:** as C7-S2.
  3. **Used data:** C7-S2 train results (spent).
  4. **Power:** projected per variant (this step).
  5. **Exposure:** train only; GEO sample titles read.
  6. **Literature:** none new.

### C7-S2a amendment (2026-11-11, before reading anything new): synthetic validity at small wells

- Synthetic syn0/syn1 checks (as C7-S2) are repeated at min cells 2 and 3 (G7, RNA only, train), to test whether
  H is biased when wells hold 2-3 cells, because D_noise is large there.

### C7-S2a result (2026-11-11; `results_celltag/c7s2a_diag.json`, `c7s2a_synthetic_small.json`; train only, nothing sealed)

- **The limit is the cell threshold, not the signal.** At >= 4 cells per well (C7-S2) only 144 train clones
  qualify.

  | min cells | n | H [95%] | projected power at partition size |
  |---|---|---|---|
  | 2 | 342 | 0.75 [0.70, 0.81] | 1.00 |
  | 3 | 210 | 0.71 [0.64, 0.77] | 0.99 |
  | 4 | 144 | 0.66 [0.57, 0.74] | 0.77 |
  | 6 | 72 | 0.66 [0.51, 0.78] | 0.35 |
  | 8 | 44 | 0.67 [0.47, 0.86] | 0.24 |

  (G7, RNA only.)
- **Small-well validity (amendment):**
  - syn0 gives H -0.05 [-0.26, 0.22] at 2 cells and -0.07 [-0.62, 0.39] at 3.
  - syn1 gives 0.92 [0.88, 0.96] and 0.94 [0.88, 0.98].
  - So H is not inflated at small wells, and if anything it is slightly conservative.
- **H is higher in the smallest clones** (0.75 at >= 2 vs 0.66 at >= 4). This is consistent with memory lost per
  division after the split (larger clones = more divisions), as LARRY's K1 (rho_div 0.964). It is a hypothesis,
  not tested here.
- **Other factors:**
  - Coarse grouping G4 is about G7 (+0.01 to +0.02).
  - Day-5 ATAC cells added to well 1 give +10% n and +0.01 to +0.02 H.
  - The `edge.den` >= median filter leaves 22-31 clones, which is unusable. Edge density is mostly 1 for small
    clones, so the median splits off small clones.
  - Dropping progenitors (G6) lowers H (0.52-0.71). Part of the clone state is **how far a clone has
    differentiated** (Prog logit SD 1.38).
- **LARRY-comparable statistic:** cross-well r of N/(N+M) = 0.51 [0.33, 0.65] at >= 2 N+M cells (n 206), and
  0.45 [0.14, 0.71] at >= 4 (n 67). That is below LARRY (0.88 in the cytokine split; in vitro 0.74). This r has
  no noise correction, and here few clones have many N+M cells. The neutrophil/monocyte axis looks weaker in
  this culture than in LARRY's, but this comparison is not noise-matched.
- **Forking path:** 60 + 2 train configurations were run. Any sealed look on a changed threshold needs its own
  pre-registration (C7-S2b) and states this count.

### C7-S2 closed on train (2026-11-11)

- Bence: "1" (answer to: close C7-S2 on train and keep A for C7-S3, B for replication, C in reserve).
- **No sealed look for C7-S2. Partitions A, B and C of celltag_lsk stay unspent for C7-S3 / C7-S4.**
- **Status: precondition met on train, not confirmed on sealed data.**
  - Pre-registered setting: H 0.66 [0.57, 0.74], n 144.
  - At >= 2 cells per well (C7-S2a): H 0.75 [0.70, 0.81], n 342.
- The chapter continues with C7-S3.

## C7-S3: does early chromatin carry a clone's fate better than early RNA? CellTag-multi LSK (pre-registered 2026-11-11)

- **Serves: C7-S3**, the chapter 7 question (CHAPTER7_PLAN.md). Bence: "1" (answer to: close C7-S2 on train,
  pre-register C7-S3 with A for the look, B for replication, C in reserve). Scripts: `c7s3_carrier.py` (new),
  `c7s2_sister.py` (constants), `partitions.py`, `test_oracle.py`.
- **Data:**
  - Clone table `hsc.rna_atac.r1_2_master_v2.csv` (locked), partition `celltag_lsk`.
  - Day-2.5 count matrices from GEO: RNA GSM6681127 `d2_5_filtered_feature_bc_matrix.h5`; ATAC GSM6675425
    `d2_5_filtered_peak_bc_matrix.h5`. They go into `data/ext_celltag_multi/geo/` and are sha256-locked on
    arrival. Clone-table barcode `d2_5-rna-<BC>-1` maps to matrix barcode `<BC>-1` (same for atac).
- **Units:** clones with >= 1 day-2.5 sibling in the modality and >= 2 counted day-5 fate cells. Two groups:
  RNA-sibling clones and ATAC-sibling clones; clones with both siblings are in both.
- **Fate target:** the clone's day-5 fate counts in the C7-S2 groups G7 (Neu, Mono, BEM, EMk, DC, Lym, Prog).
  Both wells, both day-5 assays (authors' labels); Unk_1-3 / uns dropped.
- **Early features (fixed now):**
  - **RNA:**
    - library-size normalise to 1e4, log1p;
    - 2,000 highly variable genes (Seurat-v3-style dispersion on the fitting cells);
    - z-score, PCA 30.
  - **ATAC:**
    - binarise peaks; TF-IDF;
    - truncated SVD 31 components, dropping component 1 (depth) = 30.
  - **Fitting cells for both:** day-2.5 cells of train clones plus day-2.5 cells in no clone. **Cells of sealed
    clones are never used to fit** HVG / PCA / LSI; they are only projected inside the oracle test.
  - **Clone feature:** mean over its day-2.5 siblings of that modality.
- **Model:**
  - Multinomial logistic regression (sklearn, L2), one per modality.
  - Rows = clones, target = fate cells (clone rows repeated per fate cell, with weights so each clone counts
    min(n_fate, 20) in total).
  - C chosen from {0.01, 0.1, 1, 10} by 5-fold CV on train clones, per modality.
- **Score:** per clone, the mean per-cell log2-likelihood of its day-5 fate cells under the model, minus the
  same under the population baseline (train fate frequencies). This is **information gain G, in bits per cell.**
  A modality's score is the weighted mean of G over its clones (weight min(n_fate, 20)).
- **Primary statistic:** Delta = G_ATAC - G_RNA. Two-sided, 95% CI by bootstrap over clones (1,000; a clone in
  both groups is resampled once).
- **Readings:**
  - **Chromatin carries more:** Delta > 0, CI excludes 0.
  - **RNA carries more:** Delta < 0, CI excludes 0.
  - **No difference detectable:** otherwise. Also report each G with its CI; G > 0 means early state predicts
    fate at all.
- **Secondary (descriptive):**
  - **paired clones** (both siblings): the within-clone Delta;
  - **stratified by number of early siblings** (1 vs >= 2), because ATAC detection differs;
  - **G6** (progenitor cells dropped).
- **Train first:** 5-fold CV estimates of G_RNA, G_ATAC and Delta on train clones (folds by clone), with per-clone
  SDs.
  - **H12 selection file:** candidates = the two modalities' train-CV G; chosen = the best. The look still
    reports the pre-registered Delta.
  - **H14 power block:** groups {rna, atac} with the partition counts; sd = pooled per-clone SD of G; effect =
    |Delta_train|; two-sided alpha 0.05; min 0.8.
  - **If the power is < 0.8: no look.** The train result is reported and chapter 7 waits for C (paired multiome).
- **Sealed look:** `test_oracle.py look celltag_c7s3 --exp C7-S3 --partition A`. The model and features are
  refitted on all train clones, then applied to A. **B stays for replication** (its own pre-registration).
  **C is reserve** (last; H13).
- **Pre-flight:**
  1. **Unit:** clone.
  2. **Readout:** fate composition from the authors' labels; early features from GEO matrices.
  3. **Used data first:** C7-S2 / C7-S2a (train).
  4. **Power:** from train (block above).
  5. **Exposure:**
     - paper methods read; **results and Fig. 2 (their RNA vs ATAC fate prediction) unread**;
     - the abstract says lineage priming is characterised;
     - our prior is C5-S6 (chromatin separates ES clone groups) and #17 / #19 (RNA weak).
  6. **Literature:** Weinreb 2020 (RNA state predicts only part of fate), Jindal 2024 (methods).
- **Known caveats:**
  - RNA and ATAC siblings are different cells of the clone, not one cell measured twice.
  - Detection depth and clone size differ by modality (stratified report).
  - 30 components for each modality is a fixed, equal budget, not tuned.

### C7-S3 amendment before any run (2026-11-11)

- The G6 secondary (progenitors dropped) is **not implemented**. Only the G7 primary and the paired / stratified
  secondaries are computed.
- **HVG implementation:** "Seurat-v3-style" is a quadratic fit of log variance on log mean of library-normalised
  counts, with standardised clipped variance, top 2,000.
- **CV detail:** the train-CV featurisers are fit on all train-clone cells, unsupervised and without fate. Only
  the classifier is cross-validated.

### C7-S3 amendment 2 (2026-11-11, train only, before any sealed look): feature scaling bug

- **The first train run had a bug:**
  - ATAC gain was about 0.0015 bits/cell at **every** C (0.01-10), with per-clone SD 0.04.
  - Cause: TF-IDF/LSI components are about 1e-3 in scale, so L2 regularisation zeroed them. RNA PCs are about
    1 in scale.
  - That run (RNA 0.56, ATAC 0.0015) is **void** as a modality comparison.
- **Fix:** clone features of both modalities are z-scored with the fitting (train) clones' mean and SD before the
  classifier. Nothing else changes.
- **Also noted:** RNA chose C = 0.01, the grid edge. The grid stays as pre-registered, and the edge choice is
  reported.

### C7-S3 amendment 3 (2026-11-11, train only): pipeline validity check before any look

- **Question:** can each modality's day-2.5 single-cell features predict the cell's **own day-2.5 label**
  (`cell_type2` of day-2 train cells, the authors' annotation and label transfer)? This involves no clone fate.
- **Method:** 5-fold CV logistic regression (C 1) on per-cell features (z-scored), balanced accuracy vs chance.
- **Readings:**
  - If ATAC is near chance while RNA is not, the ATAC pipeline (single-cell LSI) cannot see cell state at all,
    and C7-S3's "no ATAC signal" is a measurement limit, not a biological reading.
  - If ATAC predicts its own labels well, the C7-S3 train result stands.

### C7-S3 validity result + amendment 4 (2026-11-11, train only)

- **Validity (amendment 3):** on own day-2.5 labels, RNA balanced accuracy 0.62 (11 classes, chance 0.09, 971
  cells). **ATAC 0.13 (8 classes, chance 0.125, 522 cells): blind.**
  - The C7-S3 train comparison (RNA 0.54 vs ATAC 0.004 bits/cell) is a measurement artefact and is **void as a
    carrier reading**.
  - **No sealed look.**
- **Amendment 4: ATAC featurisation is tuned on the fate-free validity task only.**
  - The criterion is own-label balanced accuracy. Fate is never used to choose.
  - **Variants:**
    - (a) Signac-style log1p(TF x IDF x 1e4) LSI;
    - (b) (a) on the 50,000 most frequently accessible peaks;
    - (c) (b) plus kNN smoothing (k 20, cosine in LSI space, over all day-2.5 ATAC cells), as the authors
      smoothed before prediction;
    - (d) RNA with the same kNN smoothing, so both modalities get equal treatment.
  - **The chosen ATAC variant:** the highest validity, provided it reaches >= 0.4 balanced accuracy. If none
    does, C7-S3 cannot be run on this data with our tools, and the authors' chromVAR / ArchR outputs (Zenodo)
    are the fallback; that needs Bence.
  - If smoothing is chosen for ATAC, RNA gets the same smoothing for the fate comparison.

### C7-S3 amendment 4 result (2026-11-11; `results_celltag/c7s3_validity_variants.json`; fate-free)

- **Own-label balanced accuracy:**
  - ATAC: (a) Signac LSI 0.76, (b) top-50k 0.76, (c) top-50k + kNN20 **0.78**.
  - RNA: 0.61 base, (d) kNN20 0.67.
- The blind first pipeline lacked the log1p(TF x IDF x 1e4) step.
- **Chosen by the pre-declared rule:**
  - (c) for ATAC; the same kNN20 smoothing for RNA.
  - Smoothing runs over all day-2.5 cells of the modality, in a space fitted on train-clone and non-clonal cells
    only (unsupervised; no fate).
- The code is changed accordingly. The C7-S3 train CV is re-run with the selection file regenerated.
- **H18:** this is the 4th amendment in C7-S3, so a **retro is mandatory before the sealed look**.

### C7-S3 train result with the fixed pipeline (2026-11-11; `results_celltag/c7s3_train.json`; train only, nothing sealed)

- **Information gain on day-5 clone fate from day-2.5 siblings (5-fold CV, bits per fate cell):**

  | | G | 95% CI | clones |
  |---|---|---|---|
  | RNA | 0.54 | [0.44, 0.65] | 434 |
  | ATAC | 0.31 | [0.22, 0.43] | 237 |
  | **Delta (ATAC - RNA)** | **-0.23** | **[-0.36, -0.10]** | |
  | paired (69 clones) | -0.31 | | |

- **Both modalities carry clone fate at day 2.5, and on train RNA carries more than chromatin.** This is the
  opposite of the chapter's working hypothesis (C5-S6).
- Both chose C 0.01 (the grid edge).
- Stratified:
  - ATAC with one early sibling 0.36 (195 clones), with two or more 0.10 (42 clones);
  - RNA 0.54 / 0.55.
- **Power for the sealed Delta test (H14, from train):**
  - partition A: effective n about 137 RNA / 75 ATAC, **power about 0.34**;
  - union A+B: about 0.6.
  - **The Delta contrast cannot reach 0.8 on the sealed partitions. No look.**
- **Status:** H18 retro due (4 amendments in C7-S3) before any further step.

## C7-S5: early chromatin vs early RNA in a second system, CellTag-multi reprogramming, TRAIN ONLY (pre-registered 2026-11-11)

- **Serves: C7-S5** (CHAPTER7_PLAN.md: the same question in a second system). Bence: "2" (answer to: run the same
  comparison on the reprogramming data, its own partitions, on train first, to see whether RNA also wins in
  another system). Scripts: `c7s5_reprog.py` (new), `c7s3_carrier.py` (Featurizer, model and score functions
  reused unchanged), `partitions.py`.
- **H18 status:** this is a new step, not an amendment to C7-S3. The C7-S3 retro is still due before any further
  C7-S3 step. Nothing sealed is read here; `celltag_reprog` partitions A/B/C stay unspent.
- **Data:**
  - Clone table `b4.rna_atac.r1_2_master_rev1_annotated.csv` (locked), partition `celltag_reprog`, unit
    `clone_id`.
  - Day-3 matrices from GEO, one file per lane: RNA GSE216518 (9 lanes, GSM6675617-25, `B4D3-RNA-r*-*`);
    ATAC GSE217119 (13 lanes, GSM6705338-50, `B4D3-ATAC-r*-*`). Into `data/ext_celltag_multi/geo_reprog/`,
    sha256-locked on arrival. Clone-table barcode `B4D3-<MOD>-<lane>-<BC>-1` maps to `<BC>-1` in that lane's file.
- **Units:** clones with >= 1 day-3 sibling in the modality and >= 2 counted day-12/21 fate cells.
- **Fate target:** the clone's day-12 + day-21 cells in the authors' 3 classes (reprogramming, transition,
  dead-end), both assays.
- **Early features:** the C7-S3 pipeline as fixed after amendment 4 (RNA: 1e4 normalise, log1p, 2,000 HVG,
  z-score, PCA 30; ATAC: binarise, top 50k features, Signac log1p(TF x IDF x 1e4), SVD 31 drop component 1;
  both: kNN 20 cosine smoothing, clone mean, z-score with fit-clone stats). Changes forced by this dataset,
  fixed now:
  1. **ATAC lanes have their own peak sets.** Peaks are mapped to a common genome tiling of 5 kb bins (a peak
     counts in the bin holding its midpoint); the top-50k step then runs on bins.
  2. **Cell universe:** all clonal day-3 cells plus up to 20,000 non-clonal day-3 cells per modality, sampled
     with seed 75 (memory: about 16 GB RAM). Fitting cells = train-clone cells + those non-clonal cells; sealed
     clones' cells are not in the universe at all in this train-only step.
  3. **No batch correction** across lanes (the same as C7-S3's single lane would need none). Lane is a caveat.
- **Model and score:** as C7-S3 (multinomial logistic, weight min(n_fate, 20), information gain G in bits per
  fate cell over the population baseline). C grid {0.001, 0.01, 0.1, 1, 10}: 0.001 is added because both C7-S3
  modalities chose the grid edge 0.01. 5-fold CV by clone on train.
- **Statistics (train, descriptive):** G_RNA and G_ATAC with bootstrap 95% CIs; Delta = G_ATAC - G_RNA with
  CI (clone resampling as C7-S3); paired-clone Delta; stratified by number of early siblings.
- **Fate-free validity check, run and reported before the fate result is read:** day-3 cells have no cell-type
  labels, so C7-S3's own-label check is not available. Instead: **sibling nearness**, i.e. for train clones
  with >= 2 day-3 cells in the modality, the AUC of the cosine similarity of sibling pairs vs random cross-clone
  pairs, in the smoothed feature space. It uses clone membership, not fate. The modality is called **blind** if
  the AUC < 0.55. A blind modality's G is reported but not interpreted.
- **Readings (train only, no sealed claim):**
  - **RNA also carries more:** Delta < 0, CI excludes 0 (replicates the LSK direction on train).
  - **Chromatin carries more:** Delta > 0, CI excludes 0 (the systems disagree).
  - **No difference detectable:** otherwise; expected if ATAC stays at ~40 train clones.
- **Power block (reported, decides whether any later sealed look on `celltag_reprog` is worth pre-registering):**
  effective n on A from the structural count x train ratios, as in C7-S3 `counts()`; power of Delta two-sided
  alpha 0.05 and of G > 0 per modality one-sided alpha 0.025. **No look in this step whatever the power.**
- **Pre-flight:**
  1. **Unit:** clone.
  2. **Readout:** authors' fate classes; early features from GEO lane matrices.
  3. **Used data first:** C7-S3 train (LSK) gave the direction and the pipeline.
  4. **Power:** block above; train-only, so no look depends on it.
  5. **Exposure:** paper methods read; results / figures on reprogramming fate prediction unread; the LSK train
     result (RNA > ATAC) is known and is the prior.
  6. **Literature:** Biddy 2018 / Kong 2020 (CellTag reprogramming: early dead-end vs reprogramming bias),
     Jindal 2024 (methods).
- **Known caveats:** different cells per modality; ATAC has ~1,249 clonal day-3 cells over 13 lanes; lanes are
  uncorrected batches; 5 kb tiling is coarser than peaks.

## C7-S4: carrier coverage, how much of a clone's heritable fate state does the early snapshot capture? LSK, TRAIN ONLY (pre-registered 2026-11-11)

- **Serves: C7-S4** (chapter 7: where is a clone's fate state carried before differentiation?). Bence: "Yes,
  pre-register" (answer to: pre-register C7-S4 as carrier coverage, train only, then run it; framing in
  `lit/reading_c7_prep.md`). Scripts: `c7s4_coverage.py` (new), `c7s3_carrier.py` (clone_features, fit_model,
  wmean; unchanged), `c7s2_sister.py` (CSV, GROUPS), `partitions.py`.
- **Train only.** No sealed row is read. `celltag_lsk` A/B/C stay sealed, and **no look follows this step**,
  whatever the result. Structural train counts (lit/reading_c7_prep.md) show no sealed contrast is powerable.
- **Units:** train clones with >= 2 counted day-5 RNA fate cells (G7) in **each** sister well (r1, r2) **and**
  >= 1 day-2.5 sibling in the modality. The structural counts were 140 (RNA) and 62 (ATAC) before fate-label
  exclusions.
- **Target cells:** the clone's day-5 **RNA** cells in r1 and r2. ATAC day-5 cells carry no well, so they are
  not used here. Both scores below are evaluated on exactly this cell set.
- **Baseline:** fate frequencies of day-5 RNA cells over the training fold's clones.
- **G_sister (ceiling):**
  - Each well's cells are predicted from the sister well's composition, shrunk to the baseline:
    p = (k_other + a * base) / (n_other + a). This is done in both directions.
  - G_sister = summed log2 gain over the base for all r1 + r2 cells / their number.
  - a is chosen from {0.5, 1, 2, 5, 10, 20} by 5-fold CV over the unit clones.
- **G_early:**
  - The C7-S3 model (multinomial logistic, weight min(n, 20), fixed C7-S3 featurisation from amendment 4,
    features fitted on train cells), per modality.
  - 5-fold CV by clone over all train clones with an early sibling and >= 2 counted fate cells (all wells, both
    assays, as in C7-S3). A clone's model never sees that clone.
  - C is chosen from {0.001, 0.01, 0.1, 1, 10} by CV.
  - G_early per unit clone is evaluated on its r1 + r2 RNA cells.
- **Primary statistic per modality:** coverage = wmean(G_early) / wmean(G_sister) over the unit clones,
  weight min(n_r1 + n_r2, 20). 95% CI by bootstrap over clones (1,000), with the ratio recomputed each time.
- **Readings (per modality; train, descriptive):**
  - **The snapshot carries most of the state:** CI lower bound > 0.5.
  - **The snapshot carries a minority:** CI upper bound < 0.5.
  - **Unresolved:** otherwise.
- **Sanity (null calibration, reported before the readings):** G_sister with sister wells re-paired across
  random clones (10 permutations). It must be <= 0 (mean). If it is > 0.02 bits, the ceiling is biased and
  coverage is not interpreted.
- **Secondary (descriptive):**
  1. Coverage at >= 4 RNA cells per well, which gives a less noisy ceiling.
  2. G_sister and G_early separately, with CIs.
  3. **Caveat check:** G_sister is a noisy-ceiling *estimate*. Small wells understate it, so coverage can
     exceed 1. The >= 4 secondary shows the direction of that bias.
- **Pre-flight:**
  1. **Unit:** clone.
  2. **Readout:** authors' fate labels; early features from the fixed C7-S3 pipeline (no new featurisation, so
     no new validity gate; the C7-S3 own-label validity 0.78 ATAC / 0.67 RNA holds).
  3. **Used data first:** C7-S2 (H 0.66-0.75) and C7-S3 train.
  4. **Power:** train-only, descriptive; there is no look.
  5. **Exposure:** C7-S2 / C7-S3 train results known; Safi 2022 abstract read.
  6. **Literature:** Weinreb 2020 (RNA state predicts only part of clonal fate), Safi 2022.
- **Pre-registration tick-list (new rule):** the script outputs all of the following, and nothing promised is
  missing:
  - a and C choices;
  - the null calibration;
  - G_sister, G_early (RNA, ATAC) with CIs;
  - coverage per modality with CI and reading;
  - the >= 4-cell secondary.
- **Known caveats:**
  - RNA and ATAC coverages use different clone sets (62 vs 140), so they are not a paired contrast.
  - Sister wells are technical replicates (same condition).
  - The wells were split at day 2.5, when the early siblings were profiled. So G_sister measures the state the
    clone already shared at day 2.5, the same time point as the snapshot. Later decisions are not shared
    across wells, except through inherited state. That makes coverage a same-time comparison: snapshot vs
    the clone's heritable state.

### C7-S5 train result (2026-11-11; `results_celltag/c7s5_train.json`, `c7s5_validity.json`; train only, nothing sealed)

*(Restored 2026-11-11: this entry and the next two were first written with a shell heredoc whose stdin was
overridden, so nothing was appended, although the commit messages 3d93eb7, 58dc2c1 and b675d70 describe them.
The numbers are unchanged from the result files.)*

- **Data:**
  - 22 day-3 lane matrices from GEO (9 RNA, 13 ATAC), sha256-locked in
    `data/ext_celltag_multi/geo_reprog/SHA256SUMS`.
  - Universe: RNA 118,641 cells (6,975 train-clonal + 20,000 non-clonal); ATAC 60,626 cells (507 train-clonal +
    20,000 non-clonal), 202,665 5-kb bins before the top-50k step.
- **Validity (fate-free, read before the fate result):** sibling-nearness AUC RNA **0.87** (13,769 pairs), ATAC
  **0.75** (97 pairs). Neither is blind.
- **Fate information** (bits per fate cell, 3 classes; train base reprogramming 0.19, transition 0.42, dead-end
  0.39):

  | early sibling (day 3) | G | 95% CI | clones | C |
  |---|---|---|---|---|
  | RNA | **0.27** | [0.18, 0.35] | 227 | 0.01 |
  | chromatin | **0.29** | [0.16, 0.43] | 61 | 0.01 |
  | chromatin minus RNA | **+0.03** | [-0.11, +0.18] | | |

  - Paired clones (45): Delta -0.003.
  - Stratified: RNA 1 early 0.33 (115) / >= 2 early 0.21 (112); ATAC 1 early 0.27 (47) / >= 2 early 0.40 (14).
  - C 0.01 is interior on the extended grid.
- **Reading: no difference detectable.** The LSK direction (RNA > chromatin) does not replicate on
  reprogramming train.
- **Power on A (descriptive, no look):** G_RNA > 0 0.97; G_ATAC > 0 0.59; Delta 0.04.
  - **Disclosed deviation:** A's effective n was scaled by the partition clone ratio, not by structural counts
    of sealed rows, so no sealed row was read.
- **Disclosure:** C7-S5 ran on my reading of Bence's "2" as "reprogramming on train". The next-step box's
  second button was "Wait for hESC data", so the reply was ambiguous and I should have asked.

## C7-S3 and C7-S5 closed on train (2026-11-11)

- Bence: "Close C7-S3 and C7-S5 on train and keep every sealed set."
- Both closed as exploratory (RETRO_2026-11-11, H18). There is no sealed look; `celltag_lsk` and
  `celltag_reprog` A/B/C are all unspent.
- **Train reading:** clone fate is readable before differentiation from both early RNA and early chromatin in
  both systems.
  - Chromatin is not privileged.
  - Blood: RNA 0.54 vs chromatin 0.31, Delta -0.23 [-0.36, -0.10].
  - Reprogramming: 0.27 vs 0.29, Delta +0.03 [-0.11, +0.18].
- The RNA-vs-chromatin question waits for same-cell data.

### C7-S4 result (2026-11-11; `results_celltag/c7s4_coverage.json`; train only, nothing sealed, no look)

- **Tick-list:** all pre-registered outputs are present.
- **Null calibration passes:** re-paired sister wells give G_sister -0.18 (RNA units) and -0.51 (ATAC units).

  | units | n | G_sister | G_early | coverage | 95% CI | pre-registered reading |
  |---|---|---|---|---|---|---|
  | RNA, >= 2 cells per well | 116 | 0.29 [0.17, 0.46] | 0.40 [0.26, 0.57] | **1.36** | [0.95, 2.19] | snapshot carries most |
  | RNA, >= 4 | 53 | 0.18 [0.07, 0.31] | 0.24 [0.10, 0.38] | 1.33 | [0.47, 3.44] | unresolved |
  | ATAC, >= 2 | 48 | 0.60 [0.27, 1.03] | 0.28 [0.09, 0.45] | **0.46** | [0.13, 1.08] | unresolved |
  | ATAC, >= 4 | 20 | 0.30 [0.09, 0.49] | 0.22 [0.05, 0.38] | 0.73 | [0.23, 1.59] | unresolved |

  C 0.01 for both; a mostly 5 (RNA) and 1-2 (ATAC).
- **Reading:**
  - **The sister-well "ceiling" is not a ceiling with these wells.** One or two early RNA siblings, through a
    model pooled over clones, predict a clone's day-5 fate composition at least as well as its sister well's own
    fate cells do.
  - The >= 4 secondary did not raise G_sister, so the shortfall is not only small-well noise.
  - The honest reading for RNA is the weak form: **the day-2.5 RNA snapshot holds at least as much clone-fate
    information as a sister well's outcome.** "Carries most of the heritable state" can't be claimed, because
    no valid ceiling was measured.
  - ATAC is unresolved; it is a different clone set (G_sister 0.60).
- **Status:** C7-S4 done (train, descriptive). All `celltag_lsk` partitions are unspent.

## C7-S6: human HSPC replication: does early RNA carry clone fate, and how much of the clone's serial state does it cover? cHSPCTrack, TRAIN ONLY (pre-registered 2026-11-11)

- **Serves: C7-S6** (chapter 7: where is a clone's fate state carried before differentiation?). It replicates
  C7-S3 (RNA side) and C7-S4 (coverage) in a third system, with the true serial reference that C7-S4 lacked.
  Bence: "1" (answer to: pre-register the cHSPCTrack replication, RNA only, train). Scripts: `c7s6_hspc.py`
  (new), `c7s3_carrier.py` (fit_model, wmean, WCAP), `c7s5_reprog.py` (ChunkedFeaturizer), `partitions.py`.
- **Train only.** Partition `chspctrack` (sealed 2026-11-11). Sealed clones' cells are **not loaded at all**
  (stricter than C7-S3). No look follows this step.
- **Prior exposure:**
  - obs column names;
  - category names of `top_level_cell_type`, `cell_type`, `condition`;
  - structural train counts below;
  - the Blood 2026 abstract.
  - No fate column has been tabulated.
- **Data:** `data/ext_chspctrack/cells_with_clone_info.h5ad` (X = raw UMI counts), human CD34+ single-cell
  clones, serial half-well sampling d3-d12, cytokine condition per clone.
- **Fate groups G8** (from `top_level_cell_type`): Neu (Neutrophils), Mono (Monocyte), Ery (Ery), MK (MK),
  Baso (Basophils), Eos (Eosinophil), Mast (Mast), Prog (GMP, MEBEMP, MEP, BEMP, BEP).
- **Times:**
  - early = days 3-5;
  - mid = day 7;
  - late (target) = day 10.
- **Units:**
  - **U1:** train clones with >= 1 early cell and >= 2 day-10 cells. 265 structurally (condition I 166, III 53,
    II 20, unknown 26).
  - **U2 (coverage):** U1 clones that also have >= 2 day-7 cells. 113 structurally.
- **Baseline:** day-10 G8 frequencies **per condition** over the training fold's clones. Unknown counts as its
  own condition, so cytokine bias is not credited to the clone state.
- **Early features:** the C7-S3 RNA pipeline.
  - 1e4 normalise, log1p, 2,000 HVG (chunked computation, identical result), z-score, PCA 30.
  - kNN 20 cosine smoothing over the universe.
  - Clone mean over its early cells, z-scored.
  - Universe and fit cells: early cells of train clones plus all early cells without a clone.
- **G_early:**
  - Multinomial logistic on [early RNA features + condition one-hot], weight min(n, 20).
  - C from {0.001, 0.01, 0.1, 1, 10} by 5-fold CV (folds by clone hash).
  - Per-cell log2 gain of the CV-predicted distribution over the condition baseline, on the clone's day-10
    cells.
- **G_serial (reference):**
  - The same model class on [day-7 G8 fractions + log n_day7 + condition one-hot] predicting the day-10 cells.
    It is a learned mapping across clones, like G_early. Same C grid and folds.
- **Statistics:**
  1. **Replication:** wmean G_early over U1 with a bootstrap 95% CI.
     - **"Early RNA carries clone fate (human HSPC)"** if CI lower > 0;
     - **"not detected"** otherwise.
  2. **Coverage** = wmean G_early / wmean G_serial over U2, with a bootstrap 95% CI (ratio recomputed).
     - **Most** if CI lower > 0.5;
     - **minority** if CI upper < 0.5;
     - else **unresolved**.
- **Validity gate (fate-free, run and reported first):** sibling-nearness AUC of early cells (train clones
  with >= 2 early cells, sibling vs random cross-clone cosine, smoothed space). If < 0.55, the RNA pipeline is
  blind and G_early is not interpreted.
- **Null calibration:**
  - early features permuted across U1 clones within condition: G_early must be <= 0.02;
  - day-7 compositions permuted within condition: G_serial must be <= 0.02;
  - 5 permutations each.
  - If a null fails, the corresponding statistic is not interpreted.
- **Plan-stage power:**
  - descriptive, train only;
  - a sealed partition holds about half of train (about 130 U1 / 55 U2);
  - **no look in this step.**
- **Pre-flight:**
  1. **Unit:** clone.
  2. **Readout:** authors' cell-type labels at day 10; early RNA from raw UMIs.
  3. **Used data first:** C7-S3 / C7-S4 train results.
  4. **Power:** above.
  5. **Exposure:** listed above.
  6. **Literature:** cHSPCTrack (Blood 2026) abstract, Weinreb 2020.
- **Tick-list:** the script outputs
  - validity AUC;
  - C per model;
  - both nulls;
  - G_early (U1) with CI and reading;
  - G_early and G_serial (U2) with CIs;
  - coverage with CI and reading;
  - unit counts after fate-label filtering.
- **Known caveats:**
  - Day-7 and day-10 cells come from the same well (shared environment), and day 7 is closer in time than the
    early sample. So G_serial is an upper reference, not a pure heritable ceiling, and coverage is
    conservative.
  - RNA only: no chromatin.
  - Most clones are condition I.

### C7-S6 first run (pre-registered design) and amendment 1 (2026-11-11, train only, before re-running)

- **Pre-registered run** (`results_hspc/c7s6_train_v1_confounded.json`):
  - validity AUC 0.88 (not blind);
  - units U1 265, serial pool 177, U2 113;
  - G_early (U1) **0.72** [0.55, 0.90], C 0.1, null -0.07 (ok);
  - G_serial (pool) **-0.005** (null -0.009);
  - on U2: G_early 0.047 [0.018, 0.064], G_serial 0.004 [0.001, 0.008], coverage 12.0 [3.9, 42.0].
- **Not interpretable as a clone state: a confound in the design, found because of the result.**
  - The 0.72 vs 0.047 gap between U1 and its U2 subset prompted a structural check (experiment x condition x
    early day; no fate column).
  - Condition I clones come from **two experiments** with **different early days**: exp 1 early cells are all
    day 3 (358 cells); exp 7 early cells are all day 5 (934).
  - Early RNA therefore identifies experiment and sampling day. A condition-only baseline credits any
    experiment-level difference in day-10 composition to "clone state".
  - U2 is entirely exp 1 / condition I, where that confound can't act.
- **The disclosure (H18 spirit):** the check was triggered by a fate result. The fix, though, is a standard
  design correction (stratify by batch), not a tuning toward a target, and it is fixed before re-running.
- **Amendment 1 (fixed now):**
  - The baseline and the model's one-hot use **stratum = experiment x condition** instead of condition.
  - Everything else is unchanged.
  - The pre-registered readings are applied to the amended run. The v1 numbers stay logged and are not
    interpreted.

### C7-S6 result after amendment 1 (2026-11-11; `results_hspc/c7s6_train.json`; train only, nothing sealed, no look)

- **Tick-list:** all pre-registered outputs are present.
- **Validity:** sibling AUC 0.88 (not blind).
- **Nulls:** G_early null -0.08, G_serial null -0.01, both pass.
- **Units:** U1 265, serial pool 177, U2 113.

| statistic | value | 95% CI | pre-registered reading |
|---|---|---|---|
| G_early, U1 (265 clones) | **0.66** bits/cell (C 0.1) | [0.49, 0.84] | early RNA carries clone fate (human HSPC) |
| G_early, U2 (113; all exp 1 / condition I, early = day 3) | **-0.02** | [-0.05, -0.00] | |
| G_serial, U2 (day-7 composition -> day 10) | **0.004** | [0.001, 0.008] | |
| coverage, U2 | -4.4 | [-20.6, -0.04] | formally "minority"; **not interpretable** (denominator ~0) |

- **Reading:**
  1. **After stratifying by experiment x condition, early RNA still carries clone fate on U1** (0.66 [0.49,
     0.84]).
     - But the signal is **not uniform:** in exp 1 (early sample at day 3, all U2) it is about zero.
     - By subtraction, it comes from exp 7 (early sample at day 5, about 130 clones).
     - A day-5 sample is closer to fate, and its cells may already show their lineage, so this is "fate visible
       early" rather than a hidden state. This split was not pre-registered, so it is descriptive.
  2. **Within exp 1 / condition I, neither early RNA nor the day-7 composition predicts the day-10 composition
     beyond the stratum baseline** (0.004 bits). This fits the authors' abstract ("single HSPCs stochastically
     yield multiple lineages"). Between-clone variation in day-10 composition there looks close to sampling
     noise.
  3. **So the serial reference isn't usable as a ceiling here either.** The coverage ratio is undefined in
     practice.
- **Comparison with chapter 7 so far:**
  - Early RNA carries clone fate in mouse LSK (0.54), reprogramming (0.27) and human HSPC (0.66, exp 7). In
    human HSPC sampled at day 3 it carries none.
  - This is the first system where an early snapshot shows **no** clone-fate signal. That may be the sampling
    day, or clones in that culture may be close to fate-unbiased.
- **Status:** C7-S6 done (train, descriptive, 1 amendment). `chspctrack` A/B/C are unspent.

## C7-S7: chapter 7 confirmatory looks: clone fate state is heritable and readable before differentiation (pre-registered 2026-11-11)

- **Serves: C7-S7** (closes chapter 7). Bence: "3" (answer to: adopt H19, close chapter 7 with powered
  confirmatory looks, then choose chapter 8 under H19). Scripts: `c7s7_confirm.py` (new; selection/power on
  train, sealed statistics via `test_oracle`), reusing unchanged `c7s2_sister.py`, `c7s3_carrier.py`,
  `c7s5_reprog.py`, `c7s6_hspc.py`, plus `partitions.py` and `test_oracle.py`.
- **Claim under test:** a clone's fate state is heritable, and readable from early RNA (and chromatin, in LSK)
  before differentiation, on **unseen clones**.
  - This is the chapter's achievable confirmed result.
  - The RNA-vs-chromatin *comparison* is not tested here (unpowerable; it waits for same-cell data).
- **One look per dataset, on partition A only.** B and C stay sealed in every dataset.
- **Five tests in a fixed sequence**, each one-sided alpha 0.025. A test counts as confirmed only if every
  earlier one is confirmed. This controls family-wise error with no power loss for the early tests.
  1. **LSK H** (C7-S7a): heritable fate share across sister wells, clones with >= 2 G7 RNA cells per well.
     **H > 0.5**: one-sided lower bound H - z(0.975) x bootstrap SE > 0.5. Train (C7-S2a, min 2 cells): H 0.75
     [0.70, 0.81], n 342. C7-S2's 4-cell threshold was underpowered (0.76); 2 cells is the threshold C7-S2a
     showed to be valid (synthetic syn0 about -0.05, syn1 0.92-0.94).
  2. **LSK G_rna > 0** (C7-S7a): the C7-S3 fixed pipeline; model and features fitted on train, applied to A;
     one-sided 2.5% bootstrap lower bound > 0.
  3. **Reprogramming G_rna > 0** (C7-S7b): the C7-S5 pipeline, RNA only; same rule.
  4. **Human HSPC G_early > 0** (C7-S7c): the C7-S6 pipeline after amendment 1 (stratum = experiment x
     condition), U1 units; same rule.
  5. **LSK G_atac > 0** (C7-S7a): the C7-S3 fixed ATAC pipeline; same rule. It comes last as the weakest
     (train power about 0.83).
- **Selection (H12) and power (H14):**
  - `python c7s7_confirm.py select` reruns the train CV and writes `selection/C7-S7a|b|c.json`: candidates =
    the C grid scores, chosen = best C. The power block uses the train per-clone SD and effect = train G.
  - The oracle's power gate computes power from partition-A counts (structure only) and refuses below 0.8.
    For C7-S7a the gate uses the weakest test (G_atac); the H and G_rna powers are recorded in the file.
- **Disclosure, found while writing this:** C7-S5's train universe treated other partitions' clonal day-3
  cells as non-clonal. Those cells could enter the unsupervised featurizer fit and the kNN smoothing as part of
  the 20,000 non-clonal sample. That's expression only, no fate, but it contradicts the C7-S5 text ("sealed
  clones' cells are not in the universe").
  - C7-S7b's train selection reruns C7-S5's `features()` unchanged, so the same caveat applies to the train
    CV.
  - The sealed test (`test_reprog`) excludes all other partitions' clonal cells explicitly.
  - The LSK pipeline smooths over all day-2.5 cells, sealed included (as logged in C7-S3 amendment 4).
- **Readings:**
  - Each test is **confirmed** or **not confirmed**.
  - Chapter 7 conclusion:
    - **"Clone fate state is heritable and readable before differentiation on unseen clones"** if tests 1-4
      are confirmed;
    - plus **"also from early chromatin (LSK)"** if test 5 is confirmed.
  - Any failure is reported as-is, and the sequence stops there for confirmatory claims.
- **Pre-flight:**
  1. **Unit:** clone (all three datasets partitioned by clone).
  2. **Readout:** authors' fate labels; early features from the pipelines fixed in C7-S3/S5/S6.
  3. **Used data first:** C7-S2a, C7-S3, C7-S5, C7-S6 train.
  4. **Power:** from `select`, gated by the oracle.
  5. **Exposure:** all train results known; A partitions are unread.
  6. **Literature:** as C7-S3/S5/S6.

## C7-S7a: LSK confirmatory look (pre-registered 2026-11-11)

- **Serves: C7-S7.** Tests 1, 2 and 5 above. `test_oracle.py look celltag_c7s7_lsk --exp C7-S7a --partition
  A`. Script `c7s7_confirm.py` (test_lsk, counts_lsk).
- **Pre-flight:** see C7-S7.

## C7-S7b: reprogramming confirmatory look (pre-registered 2026-11-11)

- **Serves: C7-S7.** Test 3. `test_oracle.py look celltag_c7s7_reprog --exp C7-S7b --partition A`. Script
  `c7s7_confirm.py` (test_reprog, counts_reprog).
- **Pre-flight:** see C7-S7.

## C7-S7c: human HSPC confirmatory look (pre-registered 2026-11-11)

- **Serves: C7-S7.** Test 4. `test_oracle.py look hspc_c7s7 --exp C7-S7c --partition A`. Script
  `c7s7_confirm.py` (test_hspc, counts_hspc).
- **Pre-flight:** see C7-S7.

### C7-S7 train selection and power (2026-11-11; `selection/C7-S7a|b|c.json`; train only)

| test (fixed order) | train value | chosen C | power on A (train-ratio estimate) |
|---|---|---|---|
| 1. LSK H, >= 2 cells | 0.75 [0.70, 0.81], n 342 | - | 1.00 |
| 2. LSK G_rna | 0.56 bits, n 434 | 0.01 | 1.00 |
| 3. reprogramming G_rna | 0.24, n 227 | 0.01 | 0.93 |
| 4. HSPC G_early (stratified) | 0.66, n 265 | 0.1 | 0.99 |
| 5. LSK G_atac | 0.28, n 237 | 0.01 | 0.93 |

- The oracle power gates recompute power from partition-A structural counts before each look.
- The LSK C grid has no 0.001 (C7-S3 grid). 0.01 is its edge, as in C7-S3.

### C7-S7 result: sealed looks on partition A (2026-11-11; `test_ledger.tsv`, commit 5e69507)

- **Spent:** celltag_lsk:A, celltag_reprog:A, chspctrack:A. **B and C remain sealed in all three.**
- **Oracle power gates (partition-A counts):** LSK ATAC 83 effective units, power 0.95; reprogramming RNA 56,
  power 0.93; HSPC 76, power 0.99.

| # (fixed order) | test | A estimate | one-sided 97.5% lower bound | threshold | result |
|---|---|---|---|---|---|
| 1 | LSK heritable fate share H (>= 2 cells per well), n 158 | **0.75** (95% CI 0.65-0.82) | 0.66 | > 0.5 | **confirmed** |
| 2 | LSK early RNA -> clone fate, n 196 | **0.50** bits/cell | 0.33 | > 0 | **confirmed** |
| 3 | reprogramming early RNA -> clone fate, n 113 | **0.35** | 0.21 | > 0 | **confirmed** |
| 4 | human HSPC early RNA -> clone fate (stratified), n 122 | **0.40** | 0.13 | > 0 | **confirmed** |
| 5 | LSK early chromatin -> clone fate, n 137 | **0.24** | 0.10 | > 0 | **confirmed** |

- **Train vs A:**
  - H 0.75 -> 0.75;
  - LSK RNA 0.56 -> 0.50;
  - reprogramming 0.24 -> 0.35;
  - HSPC 0.66 -> 0.40;
  - LSK ATAC 0.28 -> 0.24.
  - No sign flips. The HSPC estimate shrank, consistent with the train heterogeneity between experiments
    (C7-S6).
- **Pre-registered conclusion (all 5 confirmed in sequence):** **a clone's fate state is heritable, and is
  readable before differentiation, on unseen clones**:
  - from early RNA in three systems (mouse LSK, mouse fibroblast reprogramming, human HSPC);
  - and from early chromatin in LSK.
- **Not claimed:**
  - which layer carries more. RNA was ahead in LSK and equal in reprogramming, both on train; the comparison
    was never powered;
  - any statement about the HSPC day-3 experiment alone (descriptive, C7-S6);
  - a ceiling or coverage (C7-S4/S6: no valid ceiling in these data).

## C8-S0: H19 power scan for chapter 8 candidates, TRAIN ONLY (pre-registered 2026-11-11)

- **Serves: C8-S0** (choosing chapter 8 under H19). Bence: "1" (answer to: build the chapter 8 shortlist under
  H19). Script: `c8s0_scan.py` (new; reuses `c7s3_carrier.py`, `c7s5_reprog.py`, `c7s6_hspc.py`,
  `c7s7_confirm.py` helpers unchanged).
- **Purpose:** compute, for each candidate, the train effect, its SD and the plan-stage power on partition B,
  so the shortlist shows the three H19 checks with numbers.
  - Train only.
  - B's size comes from the train-to-B clone ratio; no sealed row is read.
  - No candidate becomes a chapter without Bence's choice and its own pre-registration.
- **Candidates (all with data in hand, B and C unspent):**
  - **K1, instruction vs clone state (human HSPC, cHSPCTrack exp 7, early = day 5, fate = day 10):**
    - Per clone: g_cond = information from the cytokine condition alone (condition baseline vs the exp-7
      pooled baseline) and g_state = information from early RNA beyond condition (the C7-S6 stratified model).
      Both are bits per day-10 cell, from 5-fold CV.
    - Test statistic: mean paired difference g_state - g_cond (two-sided).
    - The project's original question (chapters 1-2: instruction vs selection), now in human cells.
  - **K2, early predictor of reprogramming success (celltag_reprog, RNA):**
    - CV-predicted probability of "reprogramming" from early RNA (binomial logistic, C grid as C7-S5).
    - Test statistic: Spearman rho with the clone's observed reprogramming fraction among day-12/21 cells
      (clones with >= 2 fate cells). One-sided rho > 0.
  - **K3, early predictor of neutrophil vs monocyte bias (celltag_lsk, RNA):**
    - Units: clones with >= 2 Neu + Mono day-5 cells.
    - CV-predicted Neu probability from early RNA vs the observed Neu fraction. One-sided Spearman rho > 0.
    - The classic LARRY (Weinreb 2020) question, on CellTag-multi clones.
- **Power:**
  - K1: normal approximation on the paired difference with the effective n of B.
  - K2 and K3: Fisher z, one-sided alpha 0.025, n_B = n_train x clone ratio (LSK 1365/2720, reprog
    1647/3208, HSPC about 0.5).
- **H19 pass rule:** power >= 0.8 on B, and a path of <= 3 steps (one train step + one look).
- **Pre-flight:**
  1. **Unit:** clone.
  2. **Readout:** authors' fate labels; early features from the pipelines fixed in C7-S3/S5/S6.
  3. **Used data first:** C7 train and A results.
  4. **Power:** this step.
  5. **Exposure:** C7 train and A known; B/C unread.
  6. **Literature:** Weinreb 2020 (Neu/Mono early-state prediction), Biddy 2018 (reprogramming success
     clones), cHSPCTrack 2026 abstract (cytokines bias, stochastic diversification).

### C8-S0 result: H19 power scan, train only (2026-11-11; `results_c8/c8s0_scan.json`, log `results_celltag/c8s0_scan.log`)

- **No sealed row read.** B size = train count x train-to-B clone ratio.
- **Run note:** the first run stopped at an assertion before computing anything. Train has two experiments with
  day-5 early cells (exp 6 and exp 7); K1 was then pinned to exp 7 as pre-registered. Exp 6 is not used.

| candidate | train n | train effect | n on B (est.) | power on B at train effect | power if the effect shrinks as C7 train -> A did | steps | H19 |
|---|---|---|---|---|---|---|---|
| K1 instruction vs clone state (HSPC exp 7) | 132 | g_state - g_cond = 0.97 bits (G_cond 0.21, G_state 1.17), SD 2.0 | neff 40 | 0.86 | 0.46 (HSPC shrank 0.66 -> 0.40) | 2 | passes at face value only; fragile |
| K2 early RNA -> reprogramming success | 227 | Spearman rho 0.66 | 117 | 1.00 | about 1.00 (reprog grew on A) | 2 | passes |
| K3 early RNA -> Neu vs Mono bias (LSK) | 324 | Spearman rho 0.46 | 163 | 1.00 | about 1.00 (rho 0.30 would still give 0.97) | 2 | passes |

- **Caveats:**
  - K2 and K3 are close to the confirmed C7-S7 tests 2-3 (same features, binary fate split). They are robust
    because they narrow a confirmed result.
  - K1 is the only new question, but its power rests on an optimistic train estimate. The C grid was chosen on
    train, the SD is large, and 26 of 132 clones have an "unknown" condition.
- **Steps for each candidate:** train selection + one B look = 2 steps (<= 3).

## C8-S1: chapter 8 train selection for T1 (K3) and T2 (K1), TRAIN ONLY (pre-registered 2026-11-11)

- **Serves: C8-S1** (chapter 8, CHAPTER8_PLAN.md). Bence: "1" (answer to the C8-S0 shortlist; list item 1
  and button 1 both read "K3, then K1"). Script: `c8s1_confirm.py` (new). It reuses `c7s3_carrier.py`,
  `c7s6_hspc.py`, `c7s7_confirm.py` and `c8s0_scan.py` helpers unchanged.
- **This step:** train only. Writes `selection/C8-S1a.json` (T1) and `selection/C8-S1b.json` (T2) with
  candidates, the chosen C (best train CV, H12) and the power block (H14). It also adds oracle entries
  `celltag_c8s1_lsk` and `hspc_c8s1` with test and count functions. No sealed row is read.
- **The B look (C8-S2) is a separate step. It needs Bence's verbatim OK for each partition** (celltag_lsk:B,
  then chspctrack:B only if T1 is confirmed).
- **T1 (K3), celltag_lsk:**
  - **Units:** clones with >= 2 day-5 Neu + Mono cells and day-2.5 RNA cells.
  - **Features:** C7-S3 RNA pipeline. The featurizer is fit on train clones + non-clonal cells
    (`c3.clone_features('rna', train+held, train)`), exactly as C7-S7a.
  - **Model:** binomial logistic (Neu vs Mono counts, clone weight min(n, 20)) fit on all train units with the
    C chosen by 5-fold train CV (grid 0.001-10, criterion CV information gain, as C8-S0).
  - **Statistic:** Spearman rho between predicted P(Neu) and observed Neu fraction on B units.
  - **Confirmed if** the one-sided 97.5% bootstrap lower bound (2,000 clone resamples) is > 0.
  - **Power block (Fisher z):** effect = atanh(train CV rho), sd = sqrt(1.06), counts = B units - 3.
- **T2 (K1), chspctrack exp 7 only:**
  - **Units:** exp-7 clones with >= 2 day-10 G8 cells and >= 1 early (day <= 5) cell.
  - **Features:** C7-S7c pipeline (`_hspc_features`, featurizer fit on train early + non-clonal cells, z-score
    by train clones) + condition one-hot (exp-7 strata, "unknown" kept as its own stratum).
  - **Model:** multinomial logistic fit on all train exp-7 units with the C chosen by train CV.
  - **Per B clone:**
    - g_state = sum k (log2 P_model - log2 B_cond) / n;
    - g_cond = sum k (log2 B_cond - log2 B_pool) / n;
    - B_cond and B_pool are train exp-7 day-10 frequencies (+0.5) for the clone's condition and pooled.
  - **Statistic:** weighted mean of d = g_state - g_cond (weights min(n, 20)).
  - **Confirmed if** the one-sided 97.5% bootstrap lower bound > 0, which means early clone state carries more
    fate information than the condition.
  - **Power block:** effect = train weighted mean d, sd = SD(d), counts = B structural units x train effective
    ratio.
  - **Also reported, not tested:** G_cond and G_state on B with CIs.
- **Fixed sequence:** T1, then T2. A failed T1 stops the sequence and chspctrack:B stays sealed.
- **Pre-flight:**
  1. **Unit:** clone.
  2. **Readout:** authors' fate labels (LSK `cell_type2` groups Neu/Mono; HSPC G8). Early features from
     pipelines already validated (sibling AUC 0.87 LSK RNA / 0.88 HSPC).
  3. **Used data first:** C8-S0 train scan (done).
  4. **Power:** plan-stage from C8-S0. T1 1.00. T2 0.86 at the train effect, 0.46 if it shrinks like C7-S7.
     The oracle re-checks from B counts.
  5. **Exposure:** train and A known (A spent in C7). B and C unread.
  6. **Literature:** Weinreb 2020 (Neu/Mono early prediction in LARRY), cHSPCTrack abstract (cytokines bias
     fate; stochastic diversification).
- **Tick-list (checked against code before the run):**
  - selection files hold >= 2 candidates and chosen = best;
  - no sealed_rows call in the select_ functions;
  - test_ functions fit only on train;
  - bootstrap seeds fixed;
  - the exp-7 filter is literal "7".
