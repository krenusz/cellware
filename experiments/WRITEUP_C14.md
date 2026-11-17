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
- In held-out mouse embryos, a cell's expression state adds information beyond its annotated type, lineage, germ layer, depth and clade size about which lineage its closest relatives belong to. This is confirmed on one sealed partition of 5 embryos (H8: one confirmation, not replicated).
- The gain is not explained by coarse body-axis position. Adding the anterior-posterior, dorsal-ventral and neural AP scores to both models keeps 82% of it on train.
- The gain is carried most strongly by endothelial, somatic-mesoderm, limb-field and heart-field types. It rises with stage.

## What we cannot say
- **Mechanism.** The result is predictive, not causal. It does not separate inherited state (memory) from a shared local environment.
- **Fine-scale position.** The regional scores are inferred from expression. The DV score is missing for 91% of readout cells, and local niches below that resolution are untested. No public whole-embryo set has lineage together with measured position; this is the gap that closed C13.
- **Early embryos.** The earliest A embryo (E8.0) shows no gain, and train E8.0 gains are small. The effect may be specific to E8.5 onwards.
- **Replication.** B and C are unspent. n = 5 embryos for the confirmation.
- **Strength wording.** No "strong" claims; one partition.

## Exploratory: what carries the gain (C14-S4, train only, no claim)
- In mesoderm types (endothelial, somatic mesoderm, second heart field, trunk neural crest), one shared axis separates cells by their relatives' lineage. Cells with paraxial relatives carry paraxial/somite genes (Tcf15, Meox1, Pax3). Cells with lateral-plate relatives carry lateral-plate and posterior genes (Gata6, Pitx1, Prrx1, posterior Hox, Cdx).
- In neural types, telencephalon cells with neural-crest relatives carry border genes (Msx1, Gli3). Floor plate cells with neural relatives carry axial-progenitor genes (T, Fgf8, Cdx2).
- The gene lists are stable when one embryo is left out (top-50 overlap 0.65-0.86). The gain is spread across many PCs.
- **Reading:** a cell keeps a trace of its lineage-of-origin program beyond its annotated type. This fits inherited state, but also mixed cells near type boundaries. The posterior Hox genes are partly axial position.

## Value against the literature
- Lineage-coupled expression memory is well documented in cell lines and in vitro or organoid systems: Minkina et al. 2022 (HEK293 lineage plus expression); Eisele et al. 2024, GEMLI (lineage inferred from memory genes in cell lines, organoids and crypts). Neither tests prediction of relatives' fate beyond cell type, and neither evaluates on held-out individuals or uses whole-embryo trees.
- Closest neighbour (read 2026-11-17, gap map #25): Gowri et al. 2026, arXiv 2610.40447, "lineage inpainting". It uses the same embryo atlas and a tree-aware generative model that predicts a cell's state dynamics from its relatives' states. It does not test state beyond annotated type against relatives' fate. It does not hold out embryos (only clone trees), and it fits no mechanistic model. C14's claim is therefore not pre-empted. Its per-embryo and per-clone results were not read, because its held-out trees may come from our sealed embryos.
- The PEtracer authors' own analyses and two re-analysis abstracts (topic level only) were not used beyond the type and lineage labels.
- What this chapter adds: a held-out-embryo test of state beyond type predicting relatives' fate in an in vivo whole-embryo tree, with a regional confound check. Gap map: `lit/gap_2026-11-15.md`.

## Exposure
- Two re-analysis abstracts of this dataset, read at topic level.
- C13 headings and figure titles.
- The authors' cell-type, lineage and germ-layer labels are used in the readout and base model. Their UMAP and subtypes are not in P.
- The authors' result tables for the sealed embryos were not read.

## Partitions after chapter 14
- train: used.
- A: spent (C14-S2, 88f4654).
- B (E7.5-R2, E7.5-R3, E9.0-R3): sealed. Two of its three embryos are E7.5, where the readout is thin and the effect small, so a B replication would be underpowered for the E8.5+ effect. Power must be recomputed (sim_power) before any look.
- C (E10.0-R1): sealed, last reserve (H13).

## Process note
- Compute ran locally twice (S0, S1a) against the standing GitHub Actions rule; S1b onward ran on Actions.
- The first A run crashed out of memory before any statistic. It was logged as a technical re-run, and A was not spent by the crash.
- Four practice lines were adopted from RETRO_2026-11-15 (2026-11-16).

## Reopen if
- A public whole-embryo set with lineage plus measured position appears. That would allow testing fine-scale co-location, and claim H-a.
- Bence wants a B look. It would be pre-registered against the region-adjusted model, after sim_power on B's stages.
- A train-only mechanism step is wanted: which genes or programs carry the PCs' signal, starting with endothelial types.
