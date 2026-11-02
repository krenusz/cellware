# Literature sweep, 2026-11-02: is any of this already solved?

Working notes. Checked against what the programme found up to B3h.

## Already known (we replicated it, not new)

- **Clones carry a heritable methylation record; expression and accessibility carry little.**
  Camellia-seq paper (Li et al., Cell 2023): methylation shows strong clonal memory, expression
  and accessibility weak; ~279 clone-varying regions "unrelated to specific genes or functions"
  (secondary summary of the Cell version; the v1 preprint has no clonal-methylation analysis).
  Our B5 (methylation 0.63 vs expression 0.51) and B6a (signal only in large-scale bins, not at
  lineage genes) reproduce this.
- **Why the record is large-scale and neutral.** EPI-Clone (Scherer et al., Nature 2025): the
  clone-informative ("static") CpGs sit preferentially in heterochromatin (Fisher P 2e-5) and in
  late-replicating domains (P 0.001); state-associated ("dynamic") CpGs sit in enhancers. That is
  the textbook location of stochastic epimutation, which accumulates neutrally with divisions. Our
  B6a "100-kb bins only" result is the same thing seen from the other side.
- **Fate bias is heritable and shared by relatives.** Naik et al. 2013 (sibling LMPPs often share
  fate: imprinting), Tak et al. eLife 2021 (fate decisions "largely inherited" from the common
  ancestor; families homogeneous, population heterogeneous), EPI-Clone Fig. 3k (lineage bias
  stable through transplant). Our E6/E9 consistency results agree.
- **The fate-determining variable is hidden from scRNA-seq.** Weinreb 2020 says this itself:
  sisters predict fate better than the day-2 transcriptome; "missing heritability" attributed to
  unmeasured heritable state. Our track B (day-2 r 0.66, ~55% unexplained) re-measures this. A
  2025 preprint ("The hidden predictors of human haematopoietic clonal fate", HaemaTONIC organoid
  + barcoding + multi-omics) attacks the same question in human cells.

## Explains our B3e-B3h puzzle

- **EPI-Clone's "dynamic" label never excluded clonal association.** Static = CpGs with no
  significant association with any surface protein (KS test, Bonferroni) and 25-90% mean
  methylation. Dynamic = associated with a surface protein. Nothing tested dynamic CpGs against
  clone identity. So our B3h finding (Dynamic-site sister sharing = record leakage) is what the
  definition allows. It is not a contradiction.
- **The authors never claim static CpGs predict fate.** They use static CpGs only to *identify*
  clones, then read fate from the clone's cell-type composition.
- **Probable source of "record predicts fate" (B3, B3f) — a confound we have not tested.**
  EPI-Clone finds expanded clones by their *density* in static-CpG space, and myeloid bias is
  concentrated in expanded (HSC-expanded) clones. So clone-mean static features can encode
  clone size / expansion, and expansion correlates with bias. Our B3 CV folds were random by
  clone, not by mouse, so mouse/batch effects were not separated either. **The Camellia vs
  EPI-Clone "tension" may be a clone-size or mouse confound, not biology.**

## Methods we should have used

- **Kin correlation analysis** (Hormoz, Desprat, Shraiman PNAS 2015; Hormoz et al. Cell Systems
  2016; code released by Caltech). Infers heritable state dynamics from correlations between
  relatives at different kinship distances. E18/E18c's "rho from sisters with drift" is an ad hoc
  version of this. KCA gives a principled likelihood and is the standard framing for "how much
  state do sisters share and how fast is it lost".
- **Division-count input for Camellia:** the Camellia HSCs were barcoded at E10.0 (when HSCs form
  in the AGM) and sampled ~9 weeks later in adult bone marrow (v1 preprint, Fig. 6B). That pins
  the start point E18 had to assume.

## Not found (still open as far as this sweep shows)

- No paper on the G-CSF vs M-CSF sister agreement limit, or on cross-condition clone correlations
  being underestimated at low sampling depth. The ~0.3-0.4 vs 0.51 gap remains our own question.
- No paper that tests whether the neutral record itself carries fate information net of clone
  size and individual.

## Sources

- Scherer et al. 2025, Nature 643:478-487, doi 10.1038/s41586-025-09041-8 (OA PDF:
  https://edoc.mdc-berlin.de/id/eprint/25376/1/25376oa.pdf) — read in full for methods.
- Li et al. 2023, Cell 186:5183 (v1 preprint https://www.biorxiv.org/content/10.1101/2023.01.29.526062v1.full)
  — preprint read; clonal-methylation claims from a secondary summary of the Cell version.
- Weinreb et al. 2020, Science, doi 10.1126/science.aaw3381.
- Naik et al. 2013, Nature 496:229, doi 10.1038/nature12013.
- Tak et al. 2021, eLife 10:e60624.
- Hormoz et al. 2015 PNAS (arXiv 1409.7211); Hormoz et al. 2016 Cell Systems.
- "The hidden predictors of human haematopoietic clonal fate", bioRxiv 10.1101/2025.07.28.667115.
- Rodriguez-Fraticelli et al. 2020, Nature 583:585 (heritable intrinsic HSC signature, TCF15).
