# Data search 2026-11-14 (Cellware paused)

Bence: "pause cellware, and do a retro / after look around for more data".

**Method:**
- Four extended web searches covering:
  - haematopoiesis multiome lineage;
  - drug-resistance barcoding with replicates;
  - clonal memory with sister cells;
  - new LARRY / CellTag-multi data.
- Nine candidate pages, read for design only: accession, lineage method, replicate units per arm, and clones observed at two or more timepoints.
- Nothing was downloaded, and no statistic was computed.
- Each candidate is scored against the chapter gate (H19 data in hand, H22 units per arm, H20 already published, H11 enough units for a 40/20/20/20 split).

## Candidates

| # | Dataset | Data public? | Unit / design | Gate verdict |
|---|---|---|---|---|
| 1 | **LSC expansion cultures + LARRY** (Rodriguez-Fraticelli lab, bioRxiv 10.64898/2026.02.24.707683) | **Yes: Zenodo 10.5281/zenodo.18444428, CC-BY-4.0**, count matrices for D20 ex vivo, D20 in vivo, D40 ex vivo, D40 in vivo control/treated (~2.3 GB listed). GEO GSE318348 status unread (NCBI bot check). | Clone. 1,375 clones with >=3 cells at D20 ex vivo; the same clones are followed into D40 culture and transplant. Chemo arm is 4 vs 4 mice, but only ~13 clones per mouse. | **Only candidate that may pass.** Clone-level question (D20 state -> D40 / in-vivo expansion) passes H22 (no between-condition contrast). **Unknown:** how many clones are seen at both D20 and D40/in vivo (sets C's count). **H20 risk:** the authors already report heritable clonal programs shaping engraftment; our increment would be held-out clone prediction (as chapter 8). Chemo contrast fails H19 (about 50 clones per arm). |
| 1b | **PEtracer mouse embryo lineage trees** (Colgan ... Weissman, Cell 2026 189:7154, doi 10.1016/j.cell.2026.09.050; found via gap map #18) | **Stated public:** GEO GSE342607; processed data Zenodo 10.5281/zenodo.19892784; code Zenodo 10.5281/zenodo.22833553. Based on a model summary of PMC13722109; not verified on GEO (bot check). | 16 chimeric embryos, E7.5-E10.0 at half-day steps, triplicates per stage; 1.79 M cells; 1.45 M donor cells in reconstructed trees (prime-editing recorder, 35 cassettes) | **Large, public, tree-resolved.** Between-embryo contrasts have 3 per stage (H22 borderline). Clade-level questions have huge n. **H20 risk high:** the paper is a comprehensive fate-restriction map, so our question must be one they do not report. Size likely tens of GB. Not the DNA Typewriter set (GSE324482 stays on watch). |
| 2 | CloneSweeper (WM989 melanoma, dabrafenib + trametinib; 10.64898/2026.01.30.700779) | No (GEO "will be deposited") | Lineage; 2 resistance experiments; 21 lineages > 200 cells in both | Fails H19 (no data). Watch. |
| 3 | MeRLin metastasis (10.64898/2026.04.09.717571) | No (upon publication, reviewer access only) | Mouse; scRNA on 1 mouse | Fails H19 and H22. |
| 4 | PseuMO-Tag ARID1A (10.1101/2025.05.24.655549) | No accession found | Clone; scRNA and scATAC in **separate** cells (pseudo-multiome) | Fails H19; not same-cell, so it does not meet the ch7 reopen condition. |
| 5 | scRIPT mESC -> haemato-endothelial (10.64898/2026.10.14.759042; E-MTAB-17351) | Accession named in the paper; returned 403 on 2026-11-14 (watch list) | Clone; **barcodes not traced across timepoints**; D5/D7 splits are "for capacity, not replicates" | Even when released: no clone at two timepoints, so it supports within-timepoint fate coupling only. Low value. |
| 6 | Mito + LARRY human haematopoietic organoids (Adv Sci 2026, 10.1002/advs.202518084) | Page 403; unread | 2,453 clones from 4,659 cells (mostly singletons) | Too few multi-cell clones; not pursued. |
| 7 | MeRLin melanoma persisters (Mol Cancer 2026) | Unread (paywall redirect) | Unknown | Not pursued. |
| 8 | AML persisters, watermelon + multiome (ScienceDirect) | Unread (403) | Unknown | Not pursued without OA text. |
| 9 | Idling persisters (eLife RP 111996) | Unread (bot check) | Unknown | Not pursued. |

The watch list (from earlier today) is unchanged: GSE324482 private, E-MTAB-17351 403, and no data accessions for the hESC endoderm or DeepTrack preprints.

## What candidate 1 would need before any proposal (no data read)
1. **Bence OK to download** the Zenodo count matrices (24 files, listed as ~2.3 GB; the page also says 67.8 GB of "data volume", likely downloads/views). Without that OK, nothing is fetched.
2. **Format check only (H9):** confirm that LARRY barcodes are in the matrices. Feature files are 766 kB vs 291 kB for the non-LARRY sets, which suggests extra features. Count clones present at D20 and at D40/in vivo. **Seal by clone hash before any statistic (H1, H11).**
3. **H20 read:** methods plus Figures 2-4 legends, stopping at outcome panels for any question we would test.
4. **H19 power:** from the sealed partition's own clone count, with the effect taken from chapter 8 T1 (rho 0.45, LSK) as the named prior, shrunk.

## Not found
- No new public same-cell RNA + ATAC clone dataset (the ch5 and ch7 reopen condition).
- No drug lineage dataset with >= 3 libraries per arm (the ch10 reopen condition).
- No second CellTag-multi LSK dataset (which would have rescued C9-S2's unit count).
