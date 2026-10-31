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

### Limits and next steps

- The task saturates. Next runs need a harder or scarcer environment (regrowing resources,
  larger maps, rewards that need two agents at once) so that coordination can matter.
- Individual agents diverge, but it is not yet shown whether that is a division of labour
  (different territories) or noise. Next: measure spatial overlap of agents' visits.
- 5 seeds and one map. Treat the effect sizes as preliminary.
