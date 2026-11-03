# Literature sweep 3, 2026-11-03: cytokines, receptors and the day-2 neutrophil programme

Working notes. Question: given B7/B9/B11/B13, is "cytokines select clones by a pre-existing
neutrophil (Cebpa/Gfi1) vs monocyte (Irf8/PU.1) programme" already known? What does it change?
Checked from abstracts and search summaries, not full texts, unless noted.

## Already known (our B13 replicates it)

- **Gfi1 vs Irf8 are counteracting networks at the neutrophil/monocyte choice** (Olsson et al.
  2016, Nature 537:698). Single-cell RNA-seq of myeloid progenitors found mixed-lineage
  intermediates. Losing both Irf8 and Gfi1 traps cells in that state. Our B13 signs agree: Gfi1
  is positive for neutrophil outcomes, Irf8 negative.
- **The PU.1 : C/EBPalpha ratio sets neutrophil vs macrophage fate, and G-CSF supports the
  neutrophil side** (Dahl et al. 2003, Nat Immunol 4:1029). A high C/EBPalpha:PU.1 ratio gives
  granulocytes, and a low ratio gives macrophages. Our B13: Cebpa is the strongest positive
  predictor (0.28) and Spi1/PU.1 is negative.
- **Early fate biases are detectable in LARRY with the right method.** CoSpar (Wang et al.
  2022, Nat Biotechnol; NeurIPS workshop) infers early fate bias from transcriptome plus
  lineage and predicts transcription factors and receptors involved in the choice. It is not
  confirmed here whether they used the cytokine experiment.

## Directly relevant and new to us

- **proNeu1: a unipotent neutrophil progenitor inside the GMP compartment** (Kwok et al. 2020,
  Immunity). Its key properties, from the abstract:
  - it makes only neutrophils, **even when cultured with CSF-1 (M-CSF)**;
  - **G-CSF increases its abundance.**

  This is almost exactly our B10/B11 pattern. Clones that keep making neutrophils under M-CSF
  would be clones that carry proNeu1-like, already committed cells at day 2. G-CSF expanding
  proNeu1 is selection by expansion, the mechanism behind B7's G-only clones. Best candidate
  identity for our "selection state".
- **Instruction is real in bipotent GMPs** (Rieger et al. 2009, Science; reanalysed by
  Niederberger et al. 2015, Bioinformatics). Continuous imaging found that G-CSF and M-CSF
  instruct lineage choice in single GMPs, of which about 90% were bipotent. The reanalysis
  supports instruction, not selection, as the cause. M-CSF instructs PU.1 in HSCs without a
  survival or proliferation advantage (Mossadegh-Keller et al. 2013, Nature).
  **Reconciliation with B7-B13:** LARRY day-2 cells are a mixed population. Most cells are
  bipotent and get instructed (the shift in levels between dishes). A pre-committed minority
  (proNeu1-like, Cebpa/Gfi1-high; or Irf8-high monocyte-committed) ignores the cytokine. The G-M
  sister agreement comes from that committed minority. Instruction and selection are not
  exclusive. Our data say both act, and the clone-level agreement rides on the committed part.
- **Two monocyte routes** (Yáñez et al. 2017 Immunity; Weinreb 2020). GMP-derived
  "neutrophil-like" monocytes vs MDP-derived monocytes. Already in sweep 2; it gives the
  monocyte-side counterpart of proNeu1 (MDP-type clones that cannot make neutrophils).

## Not found

- No reanalysis of the Weinreb cytokine-split sister clones (G-CSF vs M-CSF) testing selection
  vs instruction. The G-M agreement and its presence structure (B7, B12) look unaddressed.
- No study tying Csf3r/Csf1r expression heterogeneity to clonal selection. Our B13 receptor
  result (modest, rho 0.14-0.19) is not contradicted, and not reported elsewhere as far as this
  sweep shows.

## What this changes in the plan

1. **The model's clone trait should be a committed fraction, not only a bias.** E25 (clone types,
   fitted latent fractions) failed, but it guessed the types. Kwok gives a measurable one:
   day-2 proNeu-like cells per clone (Gfi1/Elane/Mpo high, Csf1r low), and the monocyte
   counterpart (Irf8 high). Next diagnostic B14: the per-clone fraction of committed-like day-2
   cells. Does it explain B10/B11 and the G-M agreement better than Cebpa level alone?
2. **Model E33 (after B14):** a mixture of instructed bipotent cells and cytokine-insensitive
   committed cells, with the committed fraction measured per clone at day 2 (not fitted).
   It must explain B7, B10, B11 and G-M, behind the adequacy gate.

## Sources

- Olsson et al. 2016, Nature 537:698 — https://pubmed.ncbi.nlm.nih.gov/27580035/
- Kwok et al. 2020, Immunity — https://www.cell.com/immunity/fulltext/S1074-7613(20)30235-1
- Dahl et al. 2003, Nat Immunol 4:1029 — https://mobile.jci.org/references/scholar/38491/B23
- Rieger et al. 2009, Science 325:217 — https://mobile.jci.org/references/scholar/28583/B2
- Niederberger et al. 2015, Bioinformatics — https://doc.rero.ch/record/301414?ln=en
- Mossadegh-Keller et al. 2013, Nature 497:239 — https://pmc.ncbi.nlm.nih.gov/articles/PMC3679883
- CoSpar, Wang et al. — https://www.biorxiv.org/content/10.1101/2021.05.06.443026.full.pdf
- Weinreb et al. 2020, Science — https://pmc.ncbi.nlm.nih.gov/articles/PMC7608074
