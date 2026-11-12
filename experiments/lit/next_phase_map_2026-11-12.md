# Next-phase map: public data where the literature leaves our questions open (2026-11-12)

Cellware stays PAUSED. This is a map, not a plan. Nothing here was downloaded or tested.

**Method:**
- GEO E-utilities, 3 term sets (sister/split, cytokine/perturbation, multiome), 2022-2026: 165 series.
- Shortlisted 12 with clonal barcodes **and** a condition split, sister design or multiome.
- Read each linked paper's abstract (Europe PMC), or the GEO summary if no paper is indexed.
- For each, asked whether it already does our four analyses:
  1. instruction vs selection, quantified;
  2. sister heritability H;
  3. early-state readout;
  4. sealed held-out confirmation.

**Limits:**
- Abstract level only.
- Unit counts and power are **not** checked yet. That needs counts-only metadata (H19 step 1).

## Ranked candidates

| rank | data | design | source paper does | gap we would fill | fit to our tools |
|---|---|---|---|---|---|
| 1 | **GSE223003** ReSisTrace, ovarian cancer (20 samples) | sister cells split before olaparib / carboplatin / NK cells | predicts pre-resistant states from sister transcriptomes (Nat Commun 2024) | **does the drug instruct resistance, or select a primed state, and by how much?** Our LARRY-style instruction/selection model, plus H per drug, on a sealed split | high: the same sister-split design as SIS-seq; three arms give within-dataset replication |
| 2 | **GSE318348** LARRY AML stem cells, Npm1/Flt3 (20 samples, no paper indexed) | state-fate, in vivo vs ex vivo, therapy response, CROP-seq | clonally heritable programs predict engraftment and therapy response | instruction vs selection under therapy; whether the readout holds on unseen clones | high: same LARRY pipeline as chapter 1 |
| 3 | **GSE326529** HSC in aged vs young niche (12 samples; Research Square 2026) | shared sister clones across hetero- and homochronic transplants | sister variability, not state, predicts age-associated clonal selection | does the aged niche **instruct** or only **select**? Quantified split, sealed confirmation | medium: few samples; the paper already argues selection |
| 4 | GSE305751 multiome lineage under transient vs persistent stress (42 samples, no paper indexed) | multiome lineage, two treatment regimes | models expansion and intra-clonal heterogeneity (primed clones) | instruction term and sealed test only. **The selection modelling overlaps ours** | medium-low value: largely pre-empted |
| 5 | GSE300510 skin, viral resistance lineage memory (14 samples) | multicolour lineage, infection | heritability of susceptibility (siblings, cousins) | H with a calibrated null; instruction by infection | low: colour lineage, not barcodes; small unit count likely |
| 6 | GSE348275 ARID1A clonal multiomics; GSE342607 DevMap embryo; GSE266609 progenitor workflow; GSE312424 spinal cord; GSE220949 synthetic embryos; GSE307390 platelet-biased HSC; GSE264015 alpha vs beta cells | lineage maps, mostly without a condition split | unpublished or descriptive | early readout only (chapter 7-8 style). Little room for the instruction question | low novelty: replication-type (H20 would flag) |

## Where the real gap is

- **The instruction vs selection question is open in every treated-clone dataset found.**
  - Papers either predict pre-existing (primed) states, which implies selection, or model expansion only.
  - None fits both terms in one model with a plausible-selection bound, as our LARRY analysis did.
- **The sister-split drug design (#1) is the best match.**
  - It gives three drugs as internal replicates.
  - It is a direct transfer of the chapter 1 + chapter 2 machinery.
- **Readout-only questions (#6) would just replicate known results**, the value problem from the retro.

## Before any of this becomes a chapter (on resume, Bence's call)

1. **H19 counts only:** download metadata (barcode tables, no expression), and count clones per arm and sisters per clone for #1 and #2.
2. **H20 value check:** full-text read of ReSisTrace (open access) to confirm that it does not quantify instruction.
3. **Seal before any statistic:** partition train/A/B/C at the clone level before loading outcomes.
