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

### Limits and next steps

- The task saturates. Next runs need a harder or scarcer environment (regrowing resources,
  larger maps, rewards that need two agents at once) so that coordination can matter.
- Individual agents diverge, but it is not yet shown whether that is a division of labour
  (different territories) or noise. Next: measure spatial overlap of agents' visits.
- 5 seeds and one map. Treat the effect sizes as preliminary.
