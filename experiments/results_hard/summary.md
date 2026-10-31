# Collective vs individual memory: results

Seeds per condition: individual 5, collective 5. Values are means over the last 30 iterations of each run, then mean ± sd across seeds.

| Metric | Individual | Collective | Difference (coll − ind) | 95% bootstrap CI |
|---|---|---|---|---|
| Mean return per agent | 457.3 ± 4.7 | 454.2 ± 2.6 | -3.055 | [-7.06, +1.283] |
| Total return per dish | 1829 ± 19 | 1817 ± 10 | -12.22 | [-28.24, +5.13] |
| CV of returns across agents | 0.5406 ± 0.15 | 0.401 ± 0.0063 | -0.1397 * | [-0.2501, -0.02767] |
| Std of action log-probs across agents | 0.557 ± 0.0069 | 0.5039 ± 0.003 | -0.05311 * | [-0.05899, -0.04703] |
| Policy divergence between agents (JS) | 0.1489 ± 0.058 | 0.0008483 ± 4.8e-05 | -0.1481 * | [-0.1915, -0.1046] |
| Territory overlap between agents | 0.4586 ± 0.14 | 0.7304 ± 0.0088 | +0.2718 * | [+0.1668, +0.3747] |
| Cooperative payouts per dish | 5.931 ± 0.055 | 5.965 ± 0.045 | +0.03417 | [-0.01834, +0.09042] |

\* the 95% CI excludes zero.
