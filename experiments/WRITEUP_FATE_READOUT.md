# Chapter 8 write-up: a clone's specific fate is readable from its early RNA (closed 2026-11-11, confirmed)

## Question

Chapter 7 confirmed that a clone's overall fate state is heritable and readable early. Chapter 8 asks a
sharper question with practical value: can early RNA predict **a specific fate outcome** of a clone that the
model has never seen? It was chosen under H19, with data in hand, power >= 0.8 on a sealed partition, and
<= 3 steps (C8-S0 scan).

## Results (fixed sequence, one-sided alpha 0.025 each; sealed partition B)

| test | system | units (B) | train | B result | verdict |
|---|---|---|---|---|---|
| T1 (K3) | mouse LSK, day 2.5 RNA -> day 5 neutrophil vs monocyte split | 153 clones | rho 0.46 | **rho 0.45**, 95% CI 0.32-0.57 | **confirmed** |
| T2 (K1) | human HSPC exp 7: does early clone state beat the cytokine condition? | 32 effective | +0.97 bits/cell | not run: oracle power gate 0.78 < 0.8 | **untested**, B sealed |
| T3 (K2) | fibroblast reprogramming, day 3 RNA -> share of clone that reprograms (day 12/21) | 105 clones | rho 0.66 | **rho 0.59**, 95% CI 0.46-0.69 | **confirmed** |

## What we can say

- **Readable early in two systems.** A clone's early transcriptional state, measured days before its
  descendants commit, predicts on unseen clones:
  - how its blood descendants split between neutrophils and monocytes;
  - how many of its fibroblast descendants reprogram.
- **Stable from train to unseen clones** (0.46 -> 0.45 and 0.66 -> 0.59), unlike the HSPC shrinkage in C7.
- **Both narrow the chapter-7 result** (fate state readable early) into concrete outcomes. They are not
  independent discoveries.

## What we cannot say

- **Instruction vs clone state in human HSPC (T2).** It was never tested. Exp 7's B partition has too few
  clones for 0.8 power.
- **Mechanism, or which genes carry the signal.** The models are linear on 30 PCs. Nothing was interpreted.
- **Generalisation beyond these cultures.** One lab protocol per system.

## Partitions after chapter 8

| dataset | spent | sealed |
|---|---|---|
| celltag_lsk | A (C7), B (T1) | C |
| celltag_reprog | A (C7), B (T3) | C |
| chspctrack | A (C7) | B, C |

## Process note

- **T2's plan-stage power (0.86) was wrong.** It took B's size from the dataset-wide clone ratio, but T2 used
  only exp 7. The oracle's structural count caught it before any outcome was computed (0.78).
- **H19 is amended** to require the subset's own structural count (RETRO_2026-11-11b.md).
- **Steps:** scan, selection, looks, plus one amendment (T3 replaces T2). That is within H19's spirit, but it
  needed a mid-chapter swap.

Files: CHAPTER8_PLAN.md, EXPERIMENT_LOG C8-S0 ... C8-S2, `c8s0_scan.py`, `c8s1_confirm.py`,
`selection/C8-S1a|b|c.json`, `test_ledger.tsv`.
