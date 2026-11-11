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

## Pre-flight and downloads (2026-11-11, Bence: "test all the data and download accordingly")

Structure only: column names and cells per clone / day / well / condition. **No cell-type, fate, pseudotime or
score column was tabulated.** Each download went into its own new folder under `data/` (gitignored) and was
sha256-locked.

| dataset | test | result | action |
|---|---|---|---|
| **cHSPCTrack** (Blood 2026; public S3 `chspctrack`, no sign-in) | clones with cells early and late | 67,332 cells (54,217 with a clone); **13,700 clones**; days 3-12; 8 cytokine conditions; 1 well per clone. Clones with >= 2 cells at both days: d3+d7 254, d3+d10 221, d5+d8 259, d5+d10 273, **d7+d10 419**, d5+d12 167, d8+d12 166. **Passes** (RNA only) | downloaded `cells_with_clone_info.h5ad` (2.6 GB), `metacells.h5ad` (1.1 GB), `cell_type_colors.csv` -> `data/ext_chspctrack/` (SHA256SUMS). Pickles not downloaded (whitelist rule). **Partitioned by clone: `partitions.chspctrack`** (salt cellware-chspctrack-v1, sealed 2026-11-11) |
| **MethylTree LK** (Chen 2025; GSE262218-20; figshare 27288630) | same-cell layers + clone + early/late | same cell: LARRY clone + RNA + DNA methylation. But **549 cells, 75 clones (33 with >= 4 cells), one day-6 snapshot**, no early/late split. **Fails** (size: about 13 train clones; design) | zip (8.0 GB, sha256 1fec590c...) -> `data/ext_methyltree/`. LK RNA+LARRY h5ad, sample sheet and methylation matrix extracted to `extract/` (SHA256SUMS). **Locked, not partitioned, no statistic** |
| MethylTree H9 (GSE262216) | fate, RNA | about 51 H9 hESC cells, scBS-seq methylation only, undifferentiated. **Fails** | not downloaded |
| MethylTree 293T (GSE259241) | fate, RNA | about 58 HEK293T cells, methylation only, a cell line. **Fails** | not downloaded |
| DARLIN Camellia (GSE222486) | fate per clone | HSCs only; barcodes as raw reads only. **Fails** | not downloaded |
| ReDeeM (GSE219xxx) | reliable clones | clone calls disputed. **Fails** | not downloaded |
| Clonal memory of division (GSE300913, Zenodo) | fate per sister clone | sister RNA as raw SMART-seq3 reads only; no per-clone fate table. **Fails** | not downloaded |
| hiPSC cardiac barcoding (bioRxiv 2025) | data | no data statement found | not pursued |

- GSE26258 (as typed by Bence) is a 2010 *S. aureus* microarray. It was read as the MethylTree SuperSeries
  GSE262580, whose three parts (LK, H9, 293T) all fail.
- **Usable now:** cHSPCTrack, for an RNA-only C7 replication in human HSPCs:
  - (a) early-sample RNA -> the clone's later fate;
  - (b) a **true serial ceiling**: fate composition at d7 predicting d10 within the same clone.
- **Still no public dataset with clones + same-cell chromatin + early/late fate.** The hESC release is the only
  route.
