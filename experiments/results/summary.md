# Collective vs individual memory: results

Seeds per condition: individual 5, collective 5. Values are means over the last 30 iterations of each run, then mean ± sd across seeds.

| Metric | Individual | Collective | Difference (coll − ind) | 95% bootstrap CI |
|---|---|---|---|---|
| Mean return per agent | 196 ± 1.4 | 194.8 ± 0.12 | -1.163 * | [-2.194, -0.1535] |
| Total return per dish | 784 ± 5.4 | 779.3 ± 0.48 | -4.652 * | [-8.775, -0.6142] |
| CV of returns across agents | 1.049 ± 0.089 | 1.052 ± 0.022 | +0.002774 | [-0.0675, +0.07565] |
| Std of action log-probs across agents | 0.5609 ± 0.03 | 0.482 ± 0.032 | -0.07891 * | [-0.1125, -0.04476] |
| Policy divergence between agents (JS) | 0.1837 ± 0.089 | 0.0008698 ± 0.00021 | -0.1828 * | [-0.2536, -0.112] |

\* the 95% CI excludes zero.
