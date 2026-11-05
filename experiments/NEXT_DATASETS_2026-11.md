# Next conditional data sets: shortlist (2026-11-05)

Goal step: after LARRY's cytokine experiment, fit a second data set where **the same clones are
split across conditions**, then move toward tissue and embryo fate (program.md). Found by web
search on 2026-11-05; data access was read from search summaries and is **not yet verified**
(no files downloaded).

## Criteria

1. Sister cells or clones split across conditions (the conditional design LARRY gave us).
2. Single-cell readout of fate plus a clonal barcode; public counts.
3. A fate question the current model family can pose (instruct vs select, or fate bias).
4. A step toward the long-term goal (other lineages, then multicellular patterning).

## Shortlist (my ranking)

| rank | data set | design | why | access (unverified) |
|---|---|---|---|---|
| 1 | **SIS-seq / SIS-skew**, Tian et al. 2021, Immunity ("Clonal multi-omics reveals Bcor as a negative regulator of emergency dendritic cell development") | Sibling cells of one clone split between conditions: wild-type vs Bcor-deficient siblings grown in parallel, in Flt3L-driven DC development; one sibling profiled, others followed for fate | The closest match to LARRY's design, in a new lineage tree (cDC1, cDC2, pDC). It asks expansion vs fate potential, the same instruct-vs-select question, with a genetic rather than cytokine perturbation | GEO SuperSeries GSE153676 (SubSeries GSE153675); processed data and scripts on Mendeley Data, doi 10.17632/vj9tskkvks.1 |
| 2 | **FateMap**, Goyal et al. 2023, Nature ("Diverse clonal fates emerge upon drug treatment of homogeneous cancer cells") | Barcoded melanoma cells expanded, then split across drug doses and drugs; resistant fate types read by scRNA-seq | A strong conditional design. Resistant fates are largely predetermined but switch with dose and drug: selection vs instruction again, outside stem cells | GEO GSE233622; Goyal lab dataset pages |
| 3 | **Monoclonal gastruloids with DNA Typewriter lineage recording** (GEO GSE291244, 2025) | Gastruloids grown from single mouse ES cells; scRNA-seq plus recorded lineage trees of founder cells; related founders give similar cell-type compositions | The first step toward the long-term embryo-morphogenesis goal: fate bias inherited across sister clades in a self-organising embryo model. Not a perturbation design | GEO GSE291244 |

Also seen, lower priority: ReSisTrace (Dai et al. 2024, Nat Commun; sister-cell priming of drug
resistance, deposition not confirmed); scDynaBar (CRISPR molecular clock, includes gastruloids,
Figshare); ClonMapper (CLL, ENA PRJNA721170).

## Recommendation

Start with **SIS-skew (1)**. It reuses almost everything built for LARRY: the sibling split,
clone-level presence and size statistics, selection vs instruction, held-out discipline. It
changes the lineage tree and the perturbation type. Check first, without downloading the whole
series, that clone-level sibling counts per condition are in the processed Mendeley files. Then
pre-register a data diagnostic (B-series) before any model, as for LARRY.

**Gastruloids (3)** are the natural follow-on for the long-term goal. They link back to the
original 2023-24 multi-agent tissue simulator in this repo.

## Sources

- Tian et al. 2021, WEHI publication page: https://publications.wehi.edu.au/Publications/19066
- SIS-seq data and analysis (Mendeley): https://data.mendeley.com/datasets/vj9tskkvks
- GEO GSE153676 via OmicsDI: https://omicsdi.org/dataset/geo/GSE153676
- FateMap, Goyal et al. 2023: https://www.nature.com/articles/s41586-023-06342-8.pdf ; GSE233622: https://omicsdi.org/dataset/geo/GSE233622
- Monoclonal gastruloids, GSE291244: https://www.omicsdi.org/dataset/geo/GSE291244
- ReSisTrace, Dai et al. 2024: https://www.nature.com/articles/s41467-024-45478-7.pdf
- scDynaBar (Figshare): https://figshare.com/articles/dataset/_b_scDynaBar_b_Using_CRISPR_barcoding_as_a_molecular_clock_to_capture_dynamic_processes_at_single-cell_resolution/27627303
