# Data search for chapter 7 alternatives to the hESC sister-well multiome (2026-11-11)

Bence: "search for alternatives thoroughly". Only metadata was read: GEO records, abstracts, and
data-availability sections. No dataset was downloaded and no statistic was computed. No contact with anyone.

**Need:** clone identity, plus the early state of the clone's cells in more than one molecular layer (ideally
the same cell), plus a later fate readout per clone. Optionally sister/serial splits for a real ceiling.

**Searched:**
- Europe PMC: 6 structured queries, 2021-2026, about 1,660 hits screened by title;
- GEO E-utilities: 9 term sets (DARLIN, Camellia-seq, ReDeeM, EPI-Clone, LARRY + multiome, lineage barcode +
  multiome + differentiation, clone + ATAC 2025-26);
- Zenodo and GitHub (lab repositories);
- web search, standard and extended.

**Exposure note:** an extended web search summary quoted the hESC preprint's headline: fate bias is present
before differentiation cues, characterised by chromatin accessibility at lineage-specific elements. That is now
known prior information for any C7-C1 on that data. It is recorded here.

## Tier 1: clones + same-cell multi-layer state (public)

| study | design | layers (same cell) | fate per clone | access | verdict |
|---|---|---|---|---|---|
| **Chen et al. 2025 Nat Methods**, epimutation lineage tracing (GEO SuperSeries GSE262580; LK multi-omic sub-series GSE262218/219/220) | mouse Lin- Kit+ Sca1- progenitors from one mouse, LARRY-barcoded, **6 days in vitro pan-myeloid differentiation**, modified Camellia-seq | **LARRY barcode + RNA + DNA methylation (+ accessibility from the GpC channel, to be confirmed)** in each cell | yes: clones hold both progenitors and differentiating cells at day 6 (cell type from RNA) | public, processed per sub-series | **best public candidate.** It is plate-based, so clone counts may be small. **Next: methods-only pre-flight of cell and clone numbers** |
| **DARLIN, Li et al. 2023 Cell** (GSE222486; scRNA GSE222246, methylation/accessibility GSE222328) | inducible CRISPR barcode mouse; Camellia-seq on HSCs at AGM, fetal liver and adult bone marrow | barcode + RNA + methylation + accessibility | **no:** HSCs only. It measures clone memory (sibling similarity), not fate | processed state public; **lineage barcodes only as raw SRA reads** (would need re-calling with their public pipeline) | relevant to "carrier of clone state". Its published finding (memory in methylation, not RNA or accessibility) is already known to us. Secondary candidate |
| **ReDeeM, Weng et al. 2024 Nature** (GSE219xxx, GSE261078) | human HSPCs in vivo, natural mtDNA clones, snapshot | RNA + ATAC + mtDNA | snapshot only (clone members of different types) | public | **weak:** the clone calls are disputed (Lareau et al. 2024 preprint: most shared variants rest on one molecule). No time split |

## Tier 2: clones + fate with sister/serial splits, RNA only (public); a ceiling or coverage replication, not the carrier test

| study | design | access | verdict |
|---|---|---|---|
| **cHSPCTrack, Blood 2026** (GSE334867) | human CD34+ single cells, CellTag, cultured per well under cytokine combinations; **half of each well harvested at several time points (serial sister halves)**, 10x scRNA | processed public (raw withheld, human) | good for a **true sister ceiling and coverage** replication (C7-S4 found no valid ceiling in LSK). RNA only |
| **Clonal memory of division, Nat Commun 2026** (GSE300913 + Zenodo) | human cord-blood HSPCs, sister cells tracked by live imaging; CITE-seq / SMART-seq3 | public | sister-level fate memory; RNA only |
| hiPSC cardiac barcoding, bioRxiv 2025 (10.64898/2025.12.12.694049) | hiPSC clones -> cardiomyocyte outcome | data statement unchecked | pluripotent differentiation; RNA / outcome only |
| Xue 2026 Adv Sci hematopoietic organoids (PRJNA855311) | hPSC -> hematopoietic; LARRY + mtDNA; scRNA + spatial | raw reads only | pluripotent differentiation; RNA only; heavy re-processing |

## Fails

- **GSE266524 / GSE266608:** TEA-seq has no barcodes; the CellTag runs are RNA only.
- **GSE306002**, motor neuron multiome: no lineage barcodes.
- **GSE264015**, SC-islet alpha/beta: reporter lineage labelling, not clones.
- **EPI-Clone, GSE282971:** methylation clones + protein, no RNA.
- **GALILEO 2024, ARID1A 2025, B-ALL GSE302493:** cancer clones, outside the chapter question.

## Still private (watch)

- hESC endoderm sister wells + multiome (bioRxiv 10.64898/2026.09.02.748863);
- scRIPT (E-MTAB-17351);
- DNA Typewriter embryos (GSE324482).

## What this changes

- **A same-cell, clone-resolved, differentiation dataset with fate exists publicly now:** Chen 2025, LK
  Camellia-seq.
- Its second layer is DNA methylation (plus possibly accessibility), not 10x ATAC. That widens the chapter
  question from "chromatin vs RNA" to "which layer carries the clone's fate: RNA, accessibility or
  methylation". The DARLIN result (memory in methylation) makes that a sharp, testable prior.
- **Proposed next step (Bence decides):**
  1. Methods-only pre-flight of GSE262218/219/220: cells, clones with >= 2 progenitor + >= 2 differentiated
     cells, layers per cell; read from GEO sample counts and the processed-file headers only.
  2. Then download, sha-lock, partition by clone, and pre-register with plan-stage power.
- **In parallel:** cHSPCTrack (GSE334867) can give the true sister ceiling that C7-S4 lacked.
