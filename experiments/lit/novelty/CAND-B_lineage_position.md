# CAND-B novelty check: lineage history predicts measured position beyond cell state (H25, 2026-11-18)

The morphogenesis gap (G1 in the C13-S1 map) tested where position is measured, outside whole embryos. Not a FOCUS step. Bence: "go fine check A and B".

- Searched: web (lineage predicts position beyond cell type; in situ lineage tracing; whole-embryo spatial lineage), 2026-11-18; fetch agents read LineageMap (bioRxiv 10.64898/2026.01.19.700383) and baseMEMOIR (Chadly et al., bioRxiv 10.1101/2024.01.03.573434; abstract verbatim, the rest partly from the agent's summary). The Spatio-DARLIN article was paywalled; its abstract came from GEO GSE296896/GSE296897 records.
- Source paper: baseMEMOIR (colony data), Spatio-DARLIN (intestine). No partitions sealed; normal reads, no per-cell results taken.

## Data facts (metadata only)
- baseMEMOIR: mouse ESC colonies in 2D culture, 3 days; about 7 usable colonies, about 200 cells, 30-50 cells per tree; 12 imaged genes; x-y centroids. Caltech records pmpby-gpj05 (raw images) and 327t7-ke088 (analysis and supplements): public, open, CC-BY.
- Spatio-DARLIN: adult mouse small intestine sections; lineage barcodes in about a quarter of cells; thousands of multi-cell clones per section; GEO GSE296896/7.
- Whole embryos: no public lineage data with measured position (Colgan 2026 positions are inferred from expression).

## Planned findings
- Finding: in ESC colonies, lineage distance predicts measured spatial distance beyond cell state.
  - Verdict: partly published
  - Papers: baseMEMOIR reports spatial distance growing with lineage distance and clade-position-state links; MOLLUSC 2024 (Bioinformatics) models displacement at division on mESC data; LineageMap 2026 couples lineage and space (method). The "beyond cell state" control was not found.
- Finding: in adult tissue, clones predict position.
  - Verdict: published
  - Papers: Spatio-DARLIN (clones along the crypt-villus axis); classic intestinal clone ribbons.
- Finding: in whole embryos, clone history predicts measured position.
  - Verdict: not found (no data)
  - Papers: Colgan 2026 reports A-P heritability on inferred positions.

## Feasibility
- baseMEMOIR fails our gates at S0: about 7 colonies means about 1 colony per sealed partition (H24, H19), 12 genes is a thin "state" for a beyond-state control, and 200 cells is too few for H21.
- Spatio-DARLIN: adult intestine, where the lineage-position link is textbook; the beyond-state version is a small refinement.

## What is left that is ours
Nothing that public data can carry today. The real gap stays open in whole embryos and needs data that does not exist publicly. **No chapter here.** Watch for: whole-embryo in situ lineage (imaging-based recorders in embryos or gastruloids).
