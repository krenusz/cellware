# Chapter 8 plan: how early, and how much more than the instruction, does a clone's state set its fate? (2026-11-11)

Bence: "1" (answer to the C8-S0 shortlist: "K3, then K1", which was both list item 1 and button 1).
Chosen under H19 from the C8-S0 power scan (EXPERIMENT_LOG C8-S0 result).

## Tests (one fixed sequence, one-sided alpha 0.025 each; T2 is tested only if T1 is confirmed)

| # | question | data | sealed partition | train effect | power on B (train effect) | power if it shrinks like C7 |
|---|---|---|---|---|---|---|
| T1 (K3) | Does early (day 2.5) RNA predict a clone's neutrophil vs monocyte bias on unseen clones? | celltag_lsk | B | Spearman rho 0.46, n 324 | 1.00 | about 1.00 |
| T2 (K1) | In human HSPC (exp 7), does early clone state carry more information about day-10 fate than the cytokine condition does? | chspctrack | B | 0.97 bits/cell more (state 1.17 vs condition 0.21), neff 40 on B | 0.86 | 0.46 |

- **Steps:** C8-S1 = train selection (models, C, power blocks; selection files). C8-S2 = one B look per
  dataset through test_oracle.py, which needs Bence's OK. Two steps, within H19's limit of 3.
- **Risk stated up front:** T1 is near-certain and narrows a confirmed chapter-7 result. T2 is the new
  question and may not confirm if its effect shrinks. Under the fixed sequence it adds no extra error budget.
- **Sealed after chapter 8:** C in celltag_lsk and chspctrack; B and C in celltag_reprog (K2 not used).
- **Not in scope:** exp 6 (also day-5 early cells), K2, and the RNA-vs-chromatin question (parked with
  chapter 7).

## Amendment 1 (2026-11-11): T3 replaces the refused T2

- **T2 (K1) was refused by the oracle power gate:** chspctrack:B has 32 effective exp-7 units, power 0.78 < 0.8.
  It was not spent and T2 stays untested. The plan-stage estimate had used the dataset-wide clone ratio.
- **Bence: "go with your recomm".** T3 = K2, early RNA -> reprogramming success, on celltag_reprog:B.
  Train rho 0.66, n 227, plan-stage power about 1.00. Pre-registered as C8-S1c.
- **Fixed sequence now:** T1 (confirmed) -> T3.
