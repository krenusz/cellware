# Chapter 14 write-up: a cell's expression state beyond its type predicts the fate of its lineage relatives in mouse embryos (draft 2026-11-16, confirmed on one sealed partition)

## Question
Claim H-b: in a whole-embryo lineage tree, does a progenitor's transcriptional state *beyond its cell type* predict the lineage composition of its nearest tree relatives? The test must hold in embryos the model never saw.

Data: the public PEtracer mouse embryo set (GSE342607; Zenodo lean set plus per-embryo GEO count matrices). It has 16 embryos, E7.5-E10.0. They are partitioned by embryo before any statistic:
- train: 7 embryos
- A: 5 embryos
- B: 3 embryos
- C: 1 embryo

## Readout
- For each cell, take the smallest ancestor with at least 5 relatives of a different type, skipping clades with more than 300 leaves.
- For each type, take the two dominant relative lineages (L1, L2), each at least 20% of relatives.
- y = 1 if relatives lean to L1 rather than L2. Cells need at least 3 informative relatives; ties are dropped.
- A type enters only with at least 200 cells and at least 30 clusters per side (H21).
- Base model: lineage, germ layer, log counts, detection rate, log clade size, ancestor depth, stage.
- The state model P adds 30 PCs of 2,000 highly variable genes. HVG, z-scoring and PCA are fit on train only.
- Statistic: within-type cell-weighted delta AUC (P minus base), per embryo.

## Steps
| Step | Data | Result |
|---|---|---|
| C14-S0 gate | train, author annotations + UMAP | LOEO delta AUC 0.081; 5 of 6 embryos positive |
| C14-S1a selection | train | subtype alone +0.016; UMAP + cycle phase +0.076; both +0.081; smoke tests pass (after a power-index bug fix, amendment 1) |
| C14-S1b selection | train, expression matrices | P (expression PCs) +0.090, 6 of 6 embryos positive; shuffled 0.002, planted 0.138; P chosen |
| C14-S2 fit | train (Actions) | frozen model; held-out train embryo smoke +0.102 |
| **C14-S2 look** | **sealed A, one look** | **mean embryo delta AUC 0.084, z = 3.0 (> 1.96): CONFIRMED**; 4 of 5 embryos positive; pooled 0.121 over 279k cells, 40.6k clusters, 36 types |
| C14-S3 confound | train (Actions) | gain beyond regional scores 0.061 of 0.074 (82% retained), 6 of 6 embryos positive |

Per A embryo: E8.0-R2 -0.006, E8.5-R1 +0.043, E9.0-R2 +0.115, E9.5-R2 +0.137, E9.5-R3 +0.131. The pre-stated prediction (at least 4 of 5 positive, mean 0.05-0.10) was met.

## What we can say
- In held-out mouse embryos, a cell's expression state adds information beyond its annotated type, lineage, germ layer, depth and clade size about which lineage its closest relatives belong to. This is confirmed on one sealed partition of 5 embryos. It was replicated on 2026-11-27 in a second recorder (DNA Typewriter, one embryo, held-out founder clades A+B; CAND-A, below).
- The gain is not explained by coarse body-axis position. Adding the anterior-posterior, dorsal-ventral and neural AP scores to both models keeps 82% of it on train.
- The gain is carried most strongly by endothelial, somatic-mesoderm, limb-field and heart-field types. It rises with stage.

## What we cannot say
- **Mechanism.** The result is predictive, not causal. It does not separate inherited state (memory) from a shared local environment.
- **Fine-scale position.** The regional scores are inferred from expression. The DV score is missing for 91% of readout cells, and local niches below that resolution are untested. No public whole-embryo set has lineage together with measured position; this is the gap that closed C13.
- **Early embryos.** The earliest A embryo (E8.0) shows no gain, and train E8.0 gains are small. The effect may be specific to E8.5 onwards.
- **Replication.** On PEtracer, B and C are unspent and n = 5 embryos for the confirmation. A second-recorder replication was added on 2026-11-27 (CAND-A, below). It covers one embryo only.
- **Strength wording.** No "strong" claims; one partition.

## Replication in a second recorder (CAND-A, 2026-11-27)
- **Data:** the DNA Typewriter mouse embryo (shendurelab/dtt-mouse v1.1.0; one E13.5 embryo, a different recorder, lab and stage).
  - Unit: founder clade of the dated tree (maximal clade with <= 20,000 leaves).
  - Partitions: train 38 / A 19 / B 18 / C 18, sealed 2026-11-20.
- **Readout and model:**
  - C14's readout, unchanged. Type = the finest released label (132 levels).
  - Base: own lineage, log clade size, ancestor depth and date.
  - State: base + the five author UMAP columns. There is no expression matrix, so this differs from C14's 30 PCs.
- **Train (CAND-A-P1):** founder-grouped CV gain 0.118 [0.086, 0.149], 30 of 33 founders positive.
- **Boundary check (CAND-A-P2, train):** cells far from any type boundary keep 0.887 of the gain. Isolated boundary mislabels do not carry it.
- **Sealed look (CAND-A-L1, one union look on A+B, 34 usable founders):** mean founder delta AUC 0.091, z 4.87, 29 of 34 positive. **CONFIRMED.** The prediction (0.06-0.12) was met.
- **What it adds:** the C14 claim holds in a second recorder, lab and stage, in held-out founder clades. That is replication, not a new finding. **What it does not add:**
  - It is one embryo, so the unit is the clade within the embryo, not independent embryos.
  - The UMAP was computed by the authors on all cells, sealed ones included.
  - There are no counts columns, so quality is uncontrolled.
  - Shared niche and coherently mislabelled groups are not excluded.
- dtt_founder C stays sealed (power about 0.72 alone).

## Exploratory: what carries the gain (C14-S4, train only, no claim)
- In mesoderm types (endothelial, somatic mesoderm, second heart field, trunk neural crest), one shared axis separates cells by their relatives' lineage. Cells with paraxial relatives carry paraxial/somite genes (Tcf15, Meox1, Pax3). Cells with lateral-plate relatives carry lateral-plate and posterior genes (Gata6, Pitx1, Prrx1, posterior Hox, Cdx).
- In neural types, telencephalon cells with neural-crest relatives carry border genes (Msx1, Gli3). Floor plate cells with neural relatives carry axial-progenitor genes (T, Fgf8, Cdx2).
- The gene lists are stable when one embryo is left out (top-50 overlap 0.65-0.86). The gain is spread across many PCs.
- Boundary check (C14-S5, train only, no claim): cells far from any type boundary (kNN type purity >= 0.9, 68% of cells) keep 94% of the gain (0.069 of 0.074, 5 of 6 embryos). Mixed cells between types do not explain it. A shared local environment is still not excluded.
- **Reading:** a cell keeps a trace of its lineage-of-origin program beyond its annotated type. This fits inherited state, but also mixed cells near type boundaries. The posterior Hox genes are partly axial position.

## Value against the literature
- Lineage-coupled expression memory is well documented in cell lines and in vitro or organoid systems: Minkina et al. 2022 (HEK293 lineage plus expression); Eisele et al. 2024, GEMLI (lineage inferred from memory genes in cell lines, organoids and crypts). Neither tests prediction of relatives' fate beyond cell type, and neither evaluates on held-out individuals or uses whole-embryo trees.
- Closest neighbour (read 2026-11-17, gap map #25): Gowri et al. 2026, arXiv 2610.40447, "lineage inpainting". It uses the same embryo atlas and a tree-aware generative model that predicts a cell's state dynamics from its relatives' states. It does not test state beyond annotated type against relatives' fate. It does not hold out embryos (only clone trees), and it fits no mechanistic model. C14's claim is therefore not pre-empted. Its per-embryo and per-clone results were not read, because its held-out trees may come from our sealed embryos.
- The PEtracer authors' own analyses and two re-analysis abstracts (topic level only) were not used beyond the type and lineage labels.
- What this chapter adds: a held-out-embryo test of state beyond type predicting relatives' fate in an in vivo whole-embryo tree, with a regional confound check. Gap map: `lit/gap_2026-11-15.md`.
- **Correction (2026-11-18 audit, lit/novelty_audit_2026-11-18.md and lit/source_claims_petracer_2026-11-18.md):** the "not pre-empted" reading above predates the claims-level read of the source paper. The source (doi 10.1016/j.cell.2026.09.050) reports heritable fate programs and origin traces, including endothelium from several mesodermal origins. **C14 is partly published.** The biology is known; what is ours is the held-out test standard.
- **Refresh 2026-11-27 (OpenAlex, six claim-shaped queries run by the Haiku lane, all hits returned unfiltered; the orchestrator read the abstracts):**
  - *Coverage, pass A (from 2026-10-15):* returned / found = 25 / 1,065, 25 / 35, 6 / 6, 3 / 3, 4 / 4, 2 / 2. That is 52 unique records, all with abstracts, all keyword-scanned and the relevant ones read in full.
  - *Pass B (no date filter):* the top 15 by relevance for three queries; read at title level, plus the abstracts of the 2026 hits.
  - **Direct in-kind sources** (cells keep a lineage-of-origin expression trace beyond type, in mouse embryos):
    - the PEtracer source paper (doi 10.1016/j.cell.2026.09.050), which reports origin traces and endothelium from several mesodermal origins;
    - doi 10.64898/2026.11.11.765433 (preprint), whole-embryo barcoding at gastrulation: endothelial cells from several mesodermal origins differ transcriptionally by clonal lineage. This is the same pattern as C14-S4.
  - **Likely direct, resting on one abstract sentence:** doi 10.64898/2026.10.06.756974 (preprint), DNA Typewriter recording in mouse embryos. It infers "lineage-associated transcriptional programs" from the trees. Only its abstract was read; it may describe the dtt embryo, so it is not read further without Bence's OK.
  - **Adjacent:**
    - doi 10.48550/arxiv.2610.40447, lineage inpainting (does not pre-empt; above);
    - doi 10.64898/2026.10.19.760045, memory genes used to infer relatedness from state (the reverse direction);
    - doi 10.64898/2026.10.17.759846, clone-disjoint evaluation of fate prediction from early state, in blood, with no "beyond type" contrast. **So held-out fate prediction as such is not ours.**
  - No record states a held-out test of state beyond annotated type against relatives' lineage in embryo trees.
- **Verdict (2026-11-27): partly published, unchanged** (the registry label).
  - *Published:* the biology. Embryo cells keep a lineage-of-origin expression trace beyond their annotated type.
  - *What remains ours, narrowly:* in vivo whole-embryo trees; relatives' lineage as the outcome; an explicit contrast beyond the annotated type; sealed held-out units under a frozen pre-registered model; and confirmation in two recorders (C14 on PEtracer A; CAND-A on DNA Typewriter A+B).
  - That is a stronger testing standard for known biology, not a discovery.

## Exposure
- Two re-analysis abstracts of this dataset, read at topic level.
- C13 headings and figure titles.
- The authors' cell-type, lineage and germ-layer labels are used in the readout and base model. Their UMAP and subtypes are not in P.
- The authors' result tables for the sealed embryos were not read.
- 2026-11-27 novelty refresh: abstracts only, of the DOIs listed under Value against the literature. PEtracer B and C and dtt_founder C stay sealed.

## Partitions after chapter 14
- train: used.
- A: spent (C14-S2, 88f4654).
- B (E7.5-R2, E7.5-R3, E9.0-R3): sealed. Simulated power of the frozen rule (C14-S5): 0.60 region-adjusted if the E7.5 embryos give no readout (as train E7.5-R1 did not), so B would be a one-embryo look. B is underpowered and stays sealed.
- C (E10.0-R1): sealed, last reserve (H13).

## Process note
- Compute ran locally twice (S0, S1a) against the standing GitHub Actions rule; S1b onward ran on Actions.
- The first A run crashed out of memory before any statistic. It was logged as a technical re-run, and A was not spent by the crash.
- Four practice lines were adopted from RETRO_2026-11-15 (2026-11-16).

## Reopen if
- A public whole-embryo set with lineage plus measured position appears. That would allow testing fine-scale co-location, and claim H-a.
- Bence wants a B look. It would be pre-registered against the region-adjusted model, after sim_power on B's stages.
- A train-only mechanism step is wanted: which genes or programs carry the PCs' signal, starting with endothelial types.
