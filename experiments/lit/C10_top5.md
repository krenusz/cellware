# C10 H20 literature read: drug instruction vs selection (2026-11-13)

The source paper, Dai 2024 (ReSisTrace), was read in full on 2026-11-12. These five were read by a subagent and
checked against the question: has anyone split resistance into genetic-subclone selection, non-genetic pre-state
selection and drug instruction in one quantified model with held-out confirmation?

| paper | design | quantifies the induced share? | held-out? | read |
|---|---|---|---|---|
| Shaffer 2017 Nature (vemurafenib, melanoma) | RNA FISH + fluctuation | no: rare pre-state, then reprogrammed by the drug (qualitative) | no | abstract |
| Goyal 2023 Nature (FateMap) | barcodes + scRNA, twin split | no: resistant fate type mostly set before drug; dose switches it; no recurrent mutations | 80/20 classifier, FISH | full text ~45% |
| Russo 2022 Nat Genet (CRC, EGFR/BRAF) | two-step fluctuation + population model | yes, but **genetic only** (drug raises mutation rate 7-50x); no non-genetic pre-state | no | abstract |
| Eyler 2020 Genome Biol (GBM, dasatinib) | barcodes + scRNA, replicate splits | no: names all three routes (CNV jackpots, shared lineage, Notch persisters), no model | partial replication | full text ~80% |
| Whiting 2025 bioRxiv (CRC, 5-FU) | barcodes + ABC on counts | pre-existing fraction vs switching rate; no genetic/non-genetic split | simulation recovery | abstract |

**Minimal published mechanism:**
- Luria-Delbruck fluctuation logic (jackpot variance means pre-existing resistance; Kessler & Levine 2013);
- two-state switching with drug-dependent rates (Pisco 2013; cited from memory);
- the three-term barcode-free version closest to ours is Russo 2022.

**Verdict:** the gap holds at this depth. Nobody fits the three terms jointly on sister-split lineages with a held-out
replicate. **Limits:** three of the five are abstract-only, and the search was not exhaustive.

**For C10:**
- Our instruction term must be distinguishable from the switching-rate models above. Pre-sister state predicting
  post state = selection or memory; a residual shift shared across resistant families within one post library =
  instruction, confounded with time in culture. The **control arm** (same split and re-plating, no drug) is the
  baseline for that shift.
- The genetic term needs subclone calls (paper: subclone A from CNV inference), on train only.
