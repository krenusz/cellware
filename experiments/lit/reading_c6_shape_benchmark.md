# C6-S7: real gastruloid shape benchmark and ingredient table (2026-11-09)

Bence: "go 1 2". Descriptive only. No new simulation, no clone data, nothing sealed. Written while the C6-S7h
downstream checks (Q1-Q3, S2) were still running, and before any of their output was looked at.

## 1. Real gastruloid shape at 120 h

Our readout is `moment_elong = 1 - sqrt(ev_min / ev_max)` of the 2-D mask covariance, i.e. 1 - minor/major
axis of the equivalent ellipse. So **aspect ratio AR = 1 / (1 - moment_elong)**: 0.19 -> 1.23, 0.50 -> 2.0,
0.91 -> 11.

| source | protocol | measure | 120 h value |
|---|---|---|---|
| Blotenburg et al. 2025 PLoS One 20:e0317309 (PMC11940422), Fig 1B, D | standard: 300 cells (B6 600), CHIR 48-72 h | MOrgAna ellipse major/minor on brightfield masks | per-condition medians **about 1.2-2.2**, single gastruloids 1.0-4.0; TR line 1.2-1.5, IB10 1.4-2.2, B6 1.3-1.8. **Read off the plot by eye, not stated in the text** |
| Fiuza et al. 2024 bioRxiv 10.1101/2024.12.12.628151 | 75-300 cells, CHIR 48-72 h | 2-D mask AR | round at 72 h, ellipsoid at 96 h, elongated by 120 h; values only in plots (Fig S1D, 2A), not read |
| Trani-Bustos et al. 2025 bioRxiv 10.1101/2025.10.27.684710 | 300 cells | AR | elongation 96-120 h; values only in plots, not read |
| Science Advances adv7790 (size dependence) | N0 100-1200 | ellipse AR at 120 h, 144 h | HTTP 403, not read |
| Human EMLO thresholds (search summary [S]) | human | length/width | spherical < 2, short-elongated 2-3.5, long > 3.5 (human only; not used) |

**Verdict:** the chosen C6-S7h setting (AR about 2.0) sits **at the upper end of typical 120 h medians, inside the
real range**. It is not over-stretched. The older de Jong all-cell crawling runs (C6-S7d, AR about 6-11) were
unrealistically long. The no-crawling runs (AR about 1.2) look like 96 h or poorly elongating lines.

**Caveats:**
- There is one quantified source, and its values were read by eye.
- Model time (MCS) is not calibrated to hours.
- Cell lines vary by about 2x in AR, so this benchmark cannot separate "realistic" from "slightly too long".

## 2. What each ingredient does (Potts, 100k MCS, 6 seeds each, from logged results)

| step | adhesion energies | long-range P pull | crawling | polar | runs with >1 P cap | elong (AR) | attached |
|---|---|---|---|---|---|---|---|
| C6-S7d | de Jong (dissociating) | none | none | 0.45 | 100% | 0.11 (1.1) | falls apart (C6-S7f) |
| C6-S7d | de Jong | none | all cells, lam 80 | 0.27 | 100% | 0.91 (11) | not measured |
| C6-S7f | de Jong | none | P only, lam 80 | 0.49 | 100% | 0.17 (1.2) | not measured |
| C6-S7g | W1 partial wetting | none | none | 0.60 | 83% | 0.19 (1.2) | 1.00 |
| C6-S7h | W1 | beta 300 | none | 0.84 | 0% | 0.19 (1.2) | 1.00 |
| C6-S7h | W1 | beta 300 | all cells, lam 80 | **0.88** | **0%** | **0.50 (2.0)** | 1.00 |
| C6-S7h | W1 | beta 300 | all cells, lam 80, clone signal off (c 0) | - | - | 0.17 (1.2) | - |

**Reading:**
- **Partial-wetting adhesion keeps the body in one piece.** It does not give one pole: caps stay at 2-3.
- **The long-range posterior pull merges the caps into one pole** (multipolar 83% -> 0%) without changing shape.
- **Crawling elongates,** and only when the clone signal is on (0.50 vs 0.17).
- **Without the pull, crawling destroys polarity** (C6-S7d: 0.27).
- Each ingredient has one job. **Only all three together pass step 0.**
- Side note: W4 with beta 300 elongates without crawling (0.45). So the pull can also stretch the body, and that
  is why W4 failed the "crawling must add elongation" criterion.
