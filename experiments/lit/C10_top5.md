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

## Full reads (2026-11-13): Russo 2022, Whiting 2025, and ReSisTrace's control arm

- **Russo 2022 (full text, supplement not read):**
  - Model: a two-step Luria-Delbruck branching process, plus a persister-transition model (lambda, Dp, f0).
  - The only non-genetic pre-state term is the population fraction f0, which is fit at 0.
  - No barcodes, no family-level states, no held-out data.
- **Whiting 2025 (about 50% of the full text read):**
  - ABC on population counts and barcode diversity: rho, mu, delta, psi, then sigma and alpha.
  - Single-cell data are used only for follow-up. Genetic and non-genetic routes are not separated, and there is
    no per-family induced change.
- **Dai 2024, how the control arm is used:**
  - for protocol mimicry, for control-subtracted pre-signatures, and as a comparator for clone size, diversity and
    subclone proportions;
  - **no paired post-family vs pre-sister state comparison, and no within-family post spread**;
  - "induced" appears only in a separate, non-lineage drug-vs-DMSO experiment;
  - subclones were called with inferCNV v1.7.1 (subclusters, Leiden 0.5).
- **Time per arm (inferred mapping, confirm in Supp. Data 11):** carboplatin 3 + 10 days, olaparib 7 + 6 days, NK
  1 + 7 days, control 13 days of culture. **NK is about 5 days shorter**, so a drug-vs-control spread contrast
  confounds NK with drift time.
- **Verdict:** S (family-level post spread beyond the pre-sister prediction, drug vs control) is not done in any of
  these. Caveat: control "survivors" are selected too, so F_control must use the same family matching.
