# Experiments

## Collective vs individual memory (re-run of the 2023 Tufts interim experiment)

`collective_vs_individual.py` trains 4 PPO agents that share one depleting petri dish
(16 dishes in parallel per seed). In the individual condition each agent learns from its own
trajectories; in the collective condition every agent learns from the pooled trajectories of
all agents. 5 seeds per condition, 300 PPO iterations each, CPU, about 5 minutes per run.

```bash
uv run python experiments/collective_vs_individual.py --condition collective --seed 0 --iterations 300
uv sync --extra viz && uv run python experiments/analyze.py
```

Results (2026-10-31, run on commit with this file): [results/summary.md](results/summary.md),
[results/curves.png](results/curves.png).

### What the run shows

- **Ceiling effect.** The map holds 790 reward points in total. Both conditions reach about
  780 per dish (99%) within 50 iterations, so final return cannot separate the conditions.
  The statistically significant 4.7-point gap in the individual condition's favour is real
  but tiny (0.6%).
- **Collective learns faster.** Pooled memory reaches the ceiling at about iteration 30
  versus about 45 for individual learning (see the return curves).
- **Collective agents converge to the same policy.** Policy divergence between agents is about
  0.0009 (collective) versus 0.18 (individual); the action log-prob spread is lower too. This
  matches the 2023 "harmony" observation, but part of it is built in: collective agents are
  trained on identical data.
- **Unequal sharing in both.** Returns differ strongly across agents in both conditions
  (CV about 1.05): whoever reaches the rich cells first takes them. Memory sharing did not make
  the split fairer.

## Harder dish: regrowth, cooperative cells, visible neighbours

Same setup, but resources regrow (2% chance per step per depleted cell), the four richest cells
(50/60/70/90) pay only when two or more agents cover them in the same step (value split), and
agents see other agents in their 3x3 view.

```bash
uv run python experiments/collective_vs_individual.py --condition collective --seed 0 \
    --iterations 300 --regrow_rate 0.02 --coop_threshold 50 --see_others --out experiments/results_hard
uv run python experiments/analyze.py --results experiments/results_hard
```

Results: [results_hard/summary.md](results_hard/summary.md),
[results_hard/curves.png](results_hard/curves.png).

- **No ceiling now, and still no difference in total harvest.** About 1,820 per dish in both
  conditions (the difference's 95% CI includes zero).
- **Two different social organisations emerge.**
  - *Individual* agents split the dish: territory overlap 0.46, policies keep diverging, and
    rewards are unequal (CV 0.54, with large seed-to-seed spread).
  - *Collective* agents crowd the same areas: overlap 0.73, near-identical policies, and a
    fairer split (CV 0.40, very consistent across seeds).
  - Both reach the same total, so specialisation and sharing are equally efficient here, but
    they distribute reward differently.
- **Cooperation is learned in both.** About 6 joint payouts per dish per episode on the rich
  cells, reached earlier with shared memory (~iteration 70 vs ~150).
- **Shared memory again learns faster** (return plateau at ~100 iterations vs ~200).

## Stem-cell tissue: first knockout study

`train_tissue.py` grows tissues from one stem cell on a 12x12 lattice (`cellware.tissue`). All
cells share one PPO policy and the tissue-level reward; the target is 60 cells with
10% stem / 60% A / 30% B. Knockouts remove contact signalling, the morphogen, gap-junction
coupling, or memory inheritance. 5 seeds per condition, 300 iterations, about 2 minutes per run.

```bash
uv run python experiments/train_tissue.py --seed 0 [--no_contact|--no_morphogen|--no_gap|--no_memory]
uv run python experiments/analyze_tissue.py
```

Results: [results_tissue/summary.md](results_tissue/summary.md),
[results_tissue/curves.png](results_tissue/curves.png).

- **The cells learn the target.** The full model reaches 65 cells at 6% stem / 62% A / 33% B
  (score -0.32 from -1.40), within about 150 iterations.
- **No knockout changes the outcome.** Scores are equal across all five conditions; only small
  shifts in size and stem share are significant. Removing memory inheritance does not lower
  sister concordance (0.66 in both), so lineage memory is not being used.
- **Why: a confound in the observation.** Every cell sees the global tissue size (and the
  episode time). That broadcast signal is enough to regulate growth and fate proportions
  without any local communication, so the signalling channels are redundant. Real cells have no
  such global readout.
- **No spatial pattern** (segregation ~0), as none is rewarded.

Next version: remove the global size input, so size and proportions must be sensed locally,
and add a spatial target (e.g. A core, B rim, a French-flag-like pattern) that cannot be met
without signalling. Then rerun the knockouts.

## Stem-cell tissue: local signals only, spatial target (second knockout study)

Global size and time removed from the observation (`global_readout=False`, the default now;
the first study corresponds to `--global_readout`), and an A-core / B-rim pattern added to
the score (`--spatial`). 5 conditions x 5 seeds x 400 iterations, run one at a time.

Results: [results_tissue_spatial/summary.md](results_tissue_spatial/summary.md),
[results_tissue_spatial/curves.png](results_tissue_spatial/curves.png).

- **Training collapses in many runs.** Tissues first grow to 20-55 cells, then the policy
  learns to stop dividing: in 2 of 5 `full` runs and 4 of 5 `nomorphogen` runs the tissue ends
  as the single founding cell, committed to A. That scores -1.78, worse than the growing
  tissues (best scores -1.1 to -1.4), so it is a local optimum, not a solution.
- **The knockout comparison is therefore not interpretable yet.** Runs without contact,
  gap-junction coupling or memory inheritance happened to collapse less and so score better
  than `full`. That says the training is unstable, not that these channels are harmful.
- Sister concordance and segregation are undefined in collapsed tissues (no committed pairs).

Before rerunning the knockouts, training has to be made stable, for example by: not letting the
founder commit until the tissue has grown, a larger entropy bonus or annealing, terminal or
potential-based reward with a learned baseline, and longer training with more seeds.

### Limits and next steps

- The task saturates. Next runs need a harder or scarcer environment (regrowing resources,
  larger maps, rewards that need two agents at once) so that coordination can matter.
- Individual agents diverge, but it is not yet shown whether that is a division of labour
  (different territories) or noise. Next: measure spatial overlap of agents' visits.
- 5 seeds and one map. Treat the effect sizes as preliminary.
