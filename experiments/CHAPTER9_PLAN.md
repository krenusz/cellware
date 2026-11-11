# Chapter 9 plan: does a human blood clone's early state predict whether it makes myeloid cells? (2026-11-11)

Bence: "1" (answer to the C9-S0 shortlist; list item 1 and button 1 both = "erythroid vs myeloid becomes
chapter 9"). Chosen under H19 (amended) from the C9-S0 scan.

## Question

In human CD34+ HSPC (cHSPCTrack), almost every clone makes erythroid/megakaryocyte cells (mean share 0.97).
Does a clone's early RNA state predict how much of it goes neutrophil/monocyte instead? The test compares
clones **within the same experiment and cytokine condition**, so the culture recipe cannot drive it. This is
the human counterpart of chapter 8's mouse neutrophil/monocyte result.

## Test (one sealed look)

| | value |
|---|---|
| data | chspctrack, exps 1 and 7 (the only ones with early cells and day-10 fates) |
| partition | B (A spent in C7; C stays sealed) |
| statistic | Spearman rho between predicted and observed erythroid/MK share, both centred within experiment x condition |
| confirmed if | one-sided 97.5% bootstrap lower bound > 0 |
| train (scan) | rho 0.57, n 179 |
| power on B | 1.00 at the scan estimate (79 effective units); about 0.87 if rho shrinks to 0.35 |

- **Steps:**
  - C9-S0 scan (done);
  - C9-S1 pre-registration + train selection (C chosen by CV information gain, not by the statistic);
  - C9-S2 one look on B, with Bence's OK.
- **Risks stated up front:**
  - The outcome is skewed: the signal is mostly which clones make any myeloid cells.
  - Exp 1 contributed nothing in the state-vs-condition scan. It may dilute this test too. The per-experiment
    values are reported, and the test stays pooled as scanned.
