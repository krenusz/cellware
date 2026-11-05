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
