# Literature sweep C13: lineage trees, space and embryo morphogenesis (2026-11-15)

Bence 2026-11-15: "do a literature review around this and citation map, what this could answer that hasnt been proved yet?"
Serves: C13-S1 (H20 reading, second part). Opus orchestrated; OpenAlex metadata (titles, abstracts, citation edges) only, no full texts.

## Method
- Seeds (11): source paper Cell 2026 (10.1016/j.cell.2026.09.050, abstract **not** read), PEtracer Science 2025 (10.1126/science.adx3800),
  Chan 2019 (10.1038/s41586-019-1184-5), Kalhor 2018 (10.1126/science.aat9804), Bowling 2020 (10.1016/j.cell.2020.04.048),
  asynchronous fate restriction preprint 2026 (10.64898/2026.10.06.756974), spatiotemporal lineage Nat Genet 2026 (10.1038/s41588-026-02739-z),
  intMEMOIR 2021 (10.1126/science.abb3099), virtual lineages 2026 (10.64898/2026.10.19.760045), LineageMap 2026 (10.64898/2026.01.19.700383).
  (An NMP clonal-analysis DOI resolved to an unrelated paper and was dropped.)
- Graph: references and citers of every seed, 1,673 works and 2,382 edges. Each work was scored on five keyword groups: lineage, space, embryo, morphogenesis, inference.
- **Exposure (H20):**
  - 11 works cite the source paper; their titles were read.
  - Two abstracts re-analyse what is very likely this embryo atlas:
    - lineage inpainting (arXiv 2610.40447): reports convergence events and a counterfactual that matched a single-cell Tbx5 knockout measurement;
    - scPFA (10.64898/2026.09.21.753290): reports heritable expression programs.
  - Both are topic-level outcome exposure across all 16 embryos. No numbers were reported.

## Map by theme (key nodes)
| Theme | Papers | What is established |
|---|---|---|
| Embryo lineage recorders, dissociated cells | Kalhor 2018; Chan 2019; Bowling 2020; DNA Typewriter mouse 2026 (10.1126/science.ael0508, one embryo, 1.28M cells); source Cell 2026 (16 embryos); DeepTrack 2026 (10.64898/2026.08.29.748052); asynchronous restriction 2026 | Sibling cells share type above chance. There is a dated hierarchy of type couplings. Fate bias appears before gastrulation and restriction is asynchronous, with founder clades differing in timing. Clonal dominance arises at gastrulation (one embryo). NMP regulators come from clone-resolved multi-omics. |
| Analysis of trees | Cassiopeia 2020; ConvexML 2025; LAML-Pro 2026; pycea 2026; quantitative fate mapping 2022 (10.1016/j.cell.2022.10.028); lineage inpainting 2026; scPFA 2026 | Tree reconstruction, branch timing, ancestral states, generative prediction from lineage context, heritable programs |
| Lineage with **measured** position | MEMOIR 2017; intMEMOIR 2021; baseMEMOIR 2024 (ESC colonies); PEtracer + MERFISH 2025 (non-embryonic in vivo growth model); Nat Genet 2026; brain clones + spatial 2022 (10.1038/s41593-022-01011-x); Polytope 2024; SPACE-seq 2026 | Spatial trees exist in colonies, adult tissue and non-embryonic growth models. Phylogeographic models of displacement at division exist (10.1093/bioinformatics/btae221; LineageMap). |
| Classical clone dispersal in the mouse | Lawson 1991 (10.1242/dev.113.3.891); Gardner & Cockroft 1998 (10.1242/dev.125.13.2397); heart field origins 2021 (10.1371/journal.pbio.3001200) | Epiblast clones lose coherence before gastrulation and intermingle into organogenesis. Streak position at exit predicts heart-field fate. |
| Axial progenitors in models | Embryoid NMP recording 2024 (10.1016/j.devcel.2024.03.024) | In trunk-like embryoids, NMPs mature over time towards neural fate, with broad per-progenitor fate spread. |

## Gaps (not shown anywhere in the mapped set)
1. **Whole-embryo lineage with measured position.** No mapped work has an evolving recorder plus in situ position across a whole mammalian embryo during axis elongation. Spatial trees exist only in colonies, adult tissue or non-embryonic growth models. That is the data our morphogenesis question needs, and no public set has it yet (watch list).
2. **Out-of-sample replication of embryo fate rules.** Whole-organism results are one embryo (DNA Typewriter mouse) or reported without pre-registered held-out embryos. Whether sibling-type sharing, restriction timing or founder dominance replicate in independent embryos is not shown as a sealed test. (The source has 16 embryos; whether it tests this is unknown, since its results were not read.)
3. **State beyond type predicting clade fate in vivo.** Cellware confirmed this in culture (chapters 7, 8, 10). Lineage inpainting predicts states from lineage context but has no held-out-embryo confirmation, and does not test state beyond cell type.
4. **In vivo test of the embryoid NMP maturation claim within an embryo** (tree depth as time, not stage arms). Overlap risk: source headings name bipotent axial progenitors, and DeepTrack studies NMP commitment.

## What the C13 set can and cannot answer
- **Cannot:** morphogenesis or position. Its spatial scores are expression-derived (C13-S1 part 1).
- **Can, in principle:** gaps 2, 3 and 4, all fate questions.
  - Units are embryos: train 7, A 5, B 3, C 1.
  - Power is unknown until H17 simulation on train.
  - Gap 3 fits Cellware's confirmed arc best and has the lowest overlap with the source headings.
- SG_kit1 (gastruloid spatial, clone 1 vs rest) already failed the chapter gate on design counts (C11, 2026-11-14), so it is no fallback for gap 1.
