# Raj lab gastruloid data store: full schema (crawled 2026-11-07)

Source: public Dropbox "Paper" folder of Ayyappan, Triandafillou, Sarma, Raj (bioRxiv 2025.07.12.664536),
link in the preprint's data availability (`dropbox.com/scl/fo/yklfs2p4s9dg8uyb6w8nt/...`, rlkey in
NEXT_DATASETS_2026-11.md). Listed with the web app's `list_shared_link_folder_entries` call from the built-in browser
(rate limited: retry with backoff). **12,076 files, 226 GB.** Local copies live under
`data/gastruloid_raj/` (gitignored). Status: **spent** = used by a sealed look; **open** = analysed (results
unconfirmed); **locked** = in sealed.py, no reader yet; **not downloaded**.

## Top level

| folder | files | size | content |
|---|---|---|---|
| `Figures/` | 32 | 0.5 GB | figure PDFs/AI (Fig 1-6 + supplements), 2 supplementary movies |
| `extracted_data/` | 10,313 | 80.7 GB | processed outputs (mostly ATAC) |
| `extraction_scripts/` | 27 | 0.1 GB | authors' notebooks (read as text only) |
| `plot_scripts/` | 105 | 5.5 MB | R plotting + **revision analyses** (2026) with small CSVs |
| `raw_data/` | 1,599 | 144.9 GB | raw tables, annotations, images, sequencing |

## raw_data/ (design -> our use)

| path | files / size | what | our status |
|---|---|---|---|
| `ATAC/ATAC_Oct12_2024` | 192 / 102 GB | bam, bai, bw per sample (pure clones, passages 1/3) | not downloaded; pure clones only |
| `RNAseq/DefaultProject/VA_1..18` | 61 / 35 GB | fastq.gz, 18 bulk RNA samples (3 anterior + 2 posterior clones etc.); `studydesign.tsv`, RunManifest | not downloaded |
| `RNAseq_03192026/F8MWB5_results` | 24 / 43 MB | bulk RNA 48 h: bulk, clone1s2, clone5s2, clone6a x 2 reps; `expression-matrix.tsv` 10.8 MB | not downloaded; pure only (could give subclone states) |
| `Morphology CSVs` | 2 | `Morphologies_1_2_3_4_5_6_7.csv` (171 rows, all clone pairs) + `T_overlap - Sheet1.csv` | **open** (G2, G2b, G4, C3-S2) |
| `morph_08312026` | 2 / 0.2 MB | Nimbus JSON: pure bulk/5s2/6a; bulk+5s2, bulk+6a, **5s2+6a** elongation annotations (Aug 2026) | **locked** (C4-S2 test) |
| `clone_bulk_chimera_08232026` | 6 / 0.7 MB | {1s2, 5, 6a}+bulk at 20/50/80%, csv + polygon JSON | **locked** |
| `clone_bulk_chimera_3s1_08162026` | 2 / 0.3 MB | 3s1+bulk, csv + JSON | not downloaded |
| `T_Igfbp5_09022026` | 1 / 4.4 MB | `gastruloid_T_Igfbp5.csv`: per-gastruloid X, Y, Ch0 brightfield, Ch1 T (YFP), Ch2 Igfbp5 (Cy5) line scans; 20 gastruloids: bulk 6, 5 3, 6 3, 5+6 2, bulk+5 2, bulk+6 4 | **locked** (C4-S2 descriptive) |
| `linescans` | 42 / 16 MB | per-gastruloid raw xlsx/csv behind the curves (passages, ratios, drugs, subclones, clone+bulk) | open / sealed / spent per file (sealed.py, G1-G12) |
| `Line Scans/07162025, 07282025, 08052025` | 4 / 5 MB | GFP-spike (2), drug 07282025 (sealed partition B), sgbc2 50% | locked / sealed |
| `Nimbus JSON/<date>` (22 dates, 09/2024-08/2025) | 227 / 8 MB | per-gastruloid morphology annotations (elongated / not) behind the morphology CSVs; 11172024: 1+3, 1+6, 3+6, clone 6, WT | not downloaded (source of the CSV) |
| `Morphologies/` | 57 / 273 MB | 07282025 drug nd2 images (40), 08072025 drug dose JSONs per clone, 07112025 rainbow, Nimbus DrugTx | not downloaded; drug sets parked (#27) |
| `Immunofluorescence Blob Metrics/<date>` | 54 / 0.1 MB | T+ blob metrics per gastruloid (pure 6, WT, 1+3, 1+6, 3+6, 50% 7) | summarised in `T_overlap` (G4 H3) |
| `Immunofluorescence Intensity Measurements/04192025, 04202025, 05032025` | 25 / 2 MB | per-well DAPI/YFP/Cy3/A594 intensities | not downloaded |
| `SG/<date>` (11 dates 12/2024-08/2025) + `Clone Combination Analysis 01292025` + `reprocessed` | 581 / 1.4 GB | seqFISH (Spatial Genomics): per ROI transcripts csv, decode hdf5, cell x gene h5ad, nuclei segmentations; pure clones, 1+3 / 1+6 chimeras; 08212025 = 6 ROIs T/Igfbp5 | not downloaded; descriptive only (2-6 per condition) |
| `Patch Analysis` | 70 / 1.5 GB | 1+3 RA/SB nd2 images (05032025, 05242025) | drug, parked |
| `Incucyte_raw_test` | 222 / 3.4 GB | time-lapse tif: clone5/clone6 + Bra-GFP 72 h (3 wells, 09152025); bulk/clone1s1/bulk+clone7s1 (08272025) | not downloaded; exploratory |
| `Differentiation Assay - 10212024` | 3 | confluency txt | partitioned (raj_diffassay), parked |
| `Bootstrap Analysis/rainbow_distribution` | 18 | rainbow aggregate JSONs (120 h) | = extracted_data/bootstrapping_rainbow |
| `Weber_reanalysis` | 1 / 1.07 GB | `seurat_object.rds` (embryo LoxCode scRNA) | not relevant |

## extracted_data/

| path | files / size | what | status |
|---|---|---|---|
| `ATAC/Sorted_BAMs_Oct12_2024_duprem` | 51 / 69 GB | dedup BAMs | not downloaded |
| `ATAC/MACS2_output` (3 variants) | 459 / 11.4 GB | narrowPeak/xls/bed | not downloaded |
| `ATAC/Homer_Output` (22 contrasts) | 9,590 / 0.16 GB | motif results (anterior vs posterior, p1 vs p3, per clone) | not downloaded |
| `ATAC/BED Files`, `Homer_Peaklists` | 38 | peak lists | not downloaded |
| `RNAseq/Counts/VA_1..18` | 54 / 118 MB | per-sample h5 / abundance tsv (kallisto) | not downloaded |
| `linescans` | 49 / 1.2 MB | z-scored profiles, clone+bulk per-gastruloid lines, subclones, directionality summary | **open/spent** (G1, G3, G5) |
| `IOU` (+07282025 pngs) | 46 / 31 MB | clone overlap metrics per gastruloid (1+3, 1+6; drugs) | sealed drug partitions A/B/C, AP_CloneFraction locked |
| `Differentiation Assay` | 1 | DiffAssay_Oct212024.csv | partitioned, parked |
| `bootstrapping_rainbow` | 18 | rainbow distributions | on disk (`rainbow/`) |

## plot_scripts/ revision analyses (2026; small CSVs = the authors' own results)

| path | key files | status |
|---|---|---|
| `T_Igfbp5_separation (revisions)` | `separation_analysis.R` (EMD / overlap / centroid distance of T vs Igfbp5 profiles), `separation_metrics.csv`, `shape_features.csv`, `structure_vs_level.csv`, `correlation_ci.csv`, `profile_correlation.csv` (20 rows each) | **locked** (authors' values unread) |
| `chimera_elongation (revisions)` | README (results read), `compute_geo_metrics.py`, `data/geo_metrics.csv` (218 gastruloids: clone 1.2/3.1/5.2/6A x 20/50/80%, geodesic aspect, circularity, solidity, curl) | **locked**; result known (elongation falls with % pure clone; plate-confounded) |
| `SG (Figure 2 and Supplementary)/SG_moran_coexpr` | Moran's I and coexpression per gastruloid CSVs | not downloaded |
| others (Fig 1-6) | R scripts | not needed |

## Local layout (data/gastruloid_raj/)

`morph_csv/`, `linescans/`, `raw_linescans/`, `raw_ls2/`, `iou/`, `diff_assay/`, `rainbow/`, `scripts/` (authors'
notebooks), `morph_08312026/`, `T_Igfbp5_09022026/`, `rev_T_Igfbp5_sep/`, `rev_chimera_elong/`,
`clone_bulk_chimera_08232026/`; `*_zip/` = the downloaded archives.
