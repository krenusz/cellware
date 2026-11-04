# Literature sweep 4, 2026-11-04: outside the usual haematology papers

Earlier sweeps covered the instruction literature (Rieger 2009, Mossadegh-Keller 2013), the
Cebpa/PU.1/Irf8/Gfi1 network and LARRY methods. This one looked elsewhere: microbiology,
statistics of branching processes, lineage-tracing method papers, sister-split designs. Searched
2026-11-04. Items marked "abstract/search only" were not read in full.

## 1. The fluctuation test, generalised to cell states

- **Singh & Saint-Antoine 2023,** "Probing transient memory of cellular states using single-cell
  lineages", *Front. Microbiol.* (PMC9942930). Review, search summary only. The Luria-Delbrück
  fluctuation test (pre-existing vs induced resistance) is extended to reversible switching
  between cell states. Colony-to-colony variance in composition gives the switching timescale from
  a single end-point measurement: fast switching gives uniform colonies, slow switching makes
  composition follow the founder.
- **Why it matters for us.** Our design is a fluctuation test. Sister wells of one clone under
  different cytokines are replicate "colonies". High G-M agreement (0.51) plus selection is the
  "pre-existing" signature; pure instruction would erase between-clone variance. We have been
  fitting full simulators to get at this; the fluctuation framework gives a closed-form route to
  the memory (switching) timescale from the variance of N-share across sister wells and across days
  (B18's day-4 vs day-6 agreement: G 0.60, M 0.32). Candidate diagnostic: **B21**.
- Classic background: the 2017 Bayesian reanalysis of Luria-Delbrück (arXiv 1701.05627) finds the
  data favour "pre-existing" but cannot exclude a Lamarckian (induced) share. That is our E35
  result in another field: selection dominant, instruction not excluded.

## 2. Identifiability: why our fits keep landing in corners

- **Xu, Koelle, Guttorp, Wu, Dunbar, Abkowitz & Minin, arXiv 1610.07550 (v2 2018).** Barcoded
  macaque haematopoiesis as a multi-type branching process, fitted to correlations of barcode
  counts. Differentiation and death rates are often identifiable **only up to a ratio**, because a
  correlation-based objective is scale-free; they fix death rates from published lifespans. They
  used 250-2,000 random restarts per fit and bootstrap intervals.
- **Raue et al. / sloppiness vs identifiability** (PubMed 27789352): sloppy models can still be
  identifiable; designs should target practical identifiability, not sloppiness.
- **Why it matters for us.** Our loss uses means, variances and correlations of N-share and clone
  counts, so selection (s, survival) and instruction (tau) are partly exchangeable, the trade-off
  we saw from E33 to E39. The seed-to-seed scatter is probably a **ridge**, not optimiser noise.
  Two remedies from this literature:
  - anchor one scale from outside, such as published neutrophil vs monocyte survival under
    M-CSF, or the measured day-2 committed fractions we already use;
  - map the ridge instead of point-fitting: profile likelihood along s_G / tau_G, or
    simulation-based inference (ABC or neural posterior estimation; an ABC comparison in a
    progenitor-clone study, UCL rsos.202231, found ABC best). That would turn "1/6 seeds pass"
    into a stated range of compatible instruction:selection ratios.

## 3. Proliferation bias in clonal datasets

- **Bonham-Carter & Schiebinger 2024,** "Cellular proliferation biases clonal lineage tracing and
  trajectory inference", *Bioinformatics* 40(8) btae483. Read in full. Restricting to clones seen
  at two or more time points over-represents fast-growing cells:
  P(in multi-time clone) = r1 b (1 - (1 - r2 b)^g). Single-time clones are unbiased; the bias grows
  as sampling falls. They show it in simulation and a human T-lineage time course, not LARRY. No
  validated correction. Code: github.com/rbonhamcarter/simulate-clones.
- **Why it matters for us.** Our "both G and M" clones are their "multi-time clones" with
  conditions in place of times. B7 is this bias, made fate-dependent by the cytokine. Their
  formula is a forward model for how much presence selection to expect from growth differences
  alone, and their point that single-condition clones are the unbiased reference supports how
  B7 and B20 are built. It also warns that CoSpar-style fate maps from LARRY multi-time clones
  carry this bias, which matters for anyone reusing Weinreb's fate probabilities.

## 4. Sister-split designs

- **SIS-seq (Naik lab, WEHI; bioRxiv 10.1101/403113).** Single progenitors divide, and the sister
  cells are assayed separately: some for fate in culture, some by scRNA-seq. Most clones keep their
  lineage bias across duplicate sister wells (105 clones; search summary only, the PDF was rate-limited).
  A CRISPR screen of 500 predicted genes found about 30 that programme dendritic-cell fate.
  Independent support for "heritable fate state readable from sister transcriptomes" (our B9, B11,
  B13-B16), in DC rather than N/M.
- **Naik et al. 2013 Nature** (in vivo barcoding of LMPPs): siblings share fate, which the authors
  called "heritable lineage imprinting" earlier than thought. The same idea at an earlier fork.

## 5. Mechanism papers on CSF signalling (search summaries)

- GMP inhibitor study: PI3K and SFK inhibitors cut GMP clonogenicity but not total monocyte
  output. Survival/proliferation signalling and lineage output are pharmacologically separable,
  which fits the selection-plus-modest-instruction split in E35/E38.
- M-CSF but not G-CSF drives strong, sustained Erk1/2 activation in Lin- marrow cells; signal
  duration is proposed to shape commitment. A possible molecular basis for the residual
  instruction term.
- CSF1R reporter absent from Lin- HSCs (argues against direct M-CSF instruction at the HSC stage);
  Csf3r is detected in about 30% of HSC1/HPC1 cells and over 70% of later progenitors.
- **Yáñez et al. 2017 Immunity:** GMPs and MDPs produce distinct monocytes ("neutrophil-like" vs
  "DC-like"). Out-of-box angle: M-CSF wells may contain MDP-type clones that make only monocytes
  under any cytokine. That would lower full-M without any cytokine effect, and is testable on day-2
  states of M-only clones (an MDP signature). Candidate diagnostic: **B22**.

## 6. Tools

- **TraceSim** (NeurIPS 2026 poster): a generative simulator and benchmark for joint scRNA-seq +
  lineage tracing. Could validate our pipeline on data with known instruction:selection ratios.
- Bayesian Luria-Delbrück model selection (arXiv 1701.05627) as a template for the
  instruction-vs-selection model comparison.

## What changes for the project

1. Treat seed instability as a possible identifiability ridge and test that directly (profile
   along s_G, or SBI), instead of adding mechanisms.
2. Fluctuation-style variance analysis (B21) gives a simulator-free estimate of fate memory.
3. The MDP-clone hypothesis (B22) is a new, data-checkable explanation for low full-M.

## Sources

- [Singh & Saint-Antoine 2023, Front. Microbiol.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9942930/)
- [Frontiers 2022 fluctuation-test review](https://www.frontiersin.org/journals/microbiology/articles/10.3389/fmicb.2022.1050516/full)
- [Bayesian Luria-Delbrück reanalysis, arXiv 1701.05627](https://pdf.arxiv.org/pdf/1701.05627)
- [Xu et al., arXiv 1610.07550](https://arxiv.org/abs/1610.07550)
- [Sloppiness vs identifiability, PubMed 27789352](https://pubmed.ncbi.nlm.nih.gov/27789352/)
- [ABC for progenitor clones, UCL rsos.202231](https://discovery-pp.ucl.ac.uk/10127667/1/rsos.202231.pdf)
- [Bonham-Carter & Schiebinger 2024, Bioinformatics](https://pmc.ncbi.nlm.nih.gov/articles/PMC11316616/)
- [SIS-seq preprint, bioRxiv 10.1101/403113](https://www.biorxiv.org/content/10.1101/403113v1)
- [Naik et al. 2013 Nature, WEHI record](https://publications.wehi.edu.au/Publications/11523)
- [Myeloid CSF review, MDPI IJMS 21(7):2362](https://mdpi.com/1422-0067/21/7/2362)
- [CSF signalling papers: PMC5364337](https://pmc.ncbi.nlm.nih.gov/articles/PMC5364337), [PMC3918448](https://pmc.ncbi.nlm.nih.gov/articles/PMC3918448), [PMC4591835](https://pmc.ncbi.nlm.nih.gov/articles/PMC4591835), [PMC8168498](https://pmc.ncbi.nlm.nih.gov/articles/PMC8168498)
- [Monocyte ontogeny review (Yáñez 2017 cited), PMC10873193](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10873193/)
- [TraceSim, NeurIPS 2026](https://neurips.cc/virtual/2026/poster/139529)
