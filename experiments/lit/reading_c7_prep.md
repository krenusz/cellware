# C7-S4 prep (2026-11-11): Safi 2022 reading, GSE266524 pre-flight, C7-S4 framing

Bence: "Yes, prep C7-S4". No data statistic was computed and nothing sealed was read.

## 1. Safi et al. 2022, Cell Rep (10.1016/j.celrep.2022.110798; preprint 10.1101/2020.04.29.069062)

- **Access:** abstract only (Europe PMC).
  - The Cell Press PDF returned 403.
  - The bioRxiv PDF returned 429.
  - Europe PMC has no preprint full text (500).
  - No bypass attempted. The Lund repository record lists no file.
- **What they did:** scATAC of LSK cells, the same population as our CellTag-multi LSK. A signal-processing
  scan over 571 TF-motif accessibility profiles finds a transition population, LSK Flt3-int CD9-high. Its
  chromatin shows **stem-like and lineage-affiliated programs at the same time** (lympho-myeloid and
  megakaryocyte-erythroid together). The cells are multipotent in vitro and in vivo, with no self-renewal.
- **No clonal tracking:** the paper uses population snapshots plus functional assays of sorted cells.
- **Bearing on C7-S3:** "chromatin primes before lineage restriction" in LSK means **multi**-lineage priming:
  several programs open at once. Such a chromatin state is less fate-specific than a committed RNA program.
  This fits our train result: in LSK, early chromatin carried less clone-fate information than RNA (0.31 vs
  0.54 bits per cell). Their claim is about timing (chromatin opens early); ours is about specificity per clone.
  The two are compatible. It is a post-hoc fit, not a test.
- **For C7-S4:** if chromatin is multi-primed, a model that reads it as **a mix of fates** (composition) should
  do better than one reading a single dominant fate. Our G score already scores composition (per-cell
  log-likelihood), so nothing to change.

## 2. GSE266524 / GSE266608 pre-flight (GEO records only; methods-level, no data downloaded)

| series | content | clones? | same-cell RNA + ATAC? |
|---|---|---|---|
| GSE266524 (TEA-seq) | 4 runs, bone marrow LSK (HSC-MPP x2) and Lin- Kit+ CD34+ (MultiLin x2); RNA + ATAC + 112 ADT | **no** (genotype WT, treatment none, no CellTag) | yes |
| GSE266608 (CITE-seq + CellTag) | CITE-seq panels; **CellTag_Sca+/Sca- x2: RNA only** | yes (CellTag) | **no** |

- **Verdict: fails.** No sample has both clone barcodes and same-cell chromatin. It can't carry the paired
  RNA-vs-chromatin test, so it drops off the watch list.
- **The one possible use** would be a non-clonal same-cell reference for checking an ATAC featurisation. It's
  not needed: C7-S3 / S5 already have fate-free validity gates.

## 3. C7-S4 framing (proposal; nothing pre-registered)

**Plan line (C7-S1):** "does the core model's latent clone state z line up with the early chromatin axis better
than with RNA?"
- After C7-S3/S5 this adds little: G already measures how well each early modality predicts the clone's fate
  composition, which is what z encodes.
- A z-alignment test would be the same comparison with an extra fitted layer. It also has the same
  paired-clone power problem.

**Proposed instead: carrier coverage**, i.e. how much of the heritable clone state the early snapshot captures.
- **Ceiling:** C7-S2 found heritable fate across sister wells (H 0.66-0.75). The information one well's fate
  composition gives about the sister well's fate, in the same bits per fate cell as G, is the most that any
  early carrier could reach for that clone: **G_sister**.
- **Coverage = G_early / G_sister.** It is computed per modality on the clones that have both an early sibling
  and sister wells.
  - **Near 1:** the early snapshot carries the clone's state.
  - **Well below 1:** most of the heritable state is carried somewhere the day-2.5 snapshot misses, or set after
    day 2.5 yet still shared by sisters.
- **This answers the chapter question directly** ("where is the state carried"). It connects C7-S2 to C7-S3 and
  is what the core model's memory_through predicts: information decays with divisions between snapshot and fate.
- **Data:** LSK only. Reprogramming has no sister wells.
- **Plan-stage power (new rule):** train only, descriptive, no look.
  - **Structural count on TRAIN** (day/assay/well columns only; no fate read): clones with sister wells of
    >= 2 RNA cells each = 440. Of these, 140 have an early RNA sibling, 62 an early ATAC sibling and 24 both.
    Fate-label exclusions will lower these slightly.
  - **Hence:**
    - RNA coverage on train is estimable;
    - ATAC coverage is rough;
    - **RNA+ATAC combined (24 clones) is too small, so it is dropped.**
  - A on its own would hold about half of train (about 70 / 30 / 12), so **no sealed contrast is powerable**.
    C7-S4 would be pre-registered as train-only, descriptive.
- **Pre-flight items:**
  1. **Unit:** clone.
  2. **Readout:** authors' fate labels, early features from the fixed C7-S3 pipeline (no new featurisation, so
     no new validity gate needed).
  3. **Used data first:** C7-S2 / C7-S3 train.
  4. **Power:** structural count first.
  5. **Exposure:** C7-S2 and C7-S3 train results known.
  6. **Literature:** Safi 2022 (above), Weinreb 2020 (RNA state explains only part of fate).
