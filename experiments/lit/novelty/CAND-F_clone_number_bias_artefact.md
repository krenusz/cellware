# CAND-F novelty check: is "more engrafting clones -> more myeloid bias" a sampling artefact? (H25, 2026-11-22)

Novelty-first search (Bence: "discover other novelty gaps"). Not a FOCUS step. Chapter 1-2 type: a published headline inference tested against a model the source did not fit.

- Searched: Europe PMC full text of the paper (PMC13278965: the LARRY re-analysis passage and its methods), its citers in Europe PMC (3 listed: two primary papers, one review; none re-analyses the claim), GEO GSE299000 metadata, 2026-11-22.
- Source paper: Rodriguez-Correa et al., Nat Cell Biol 2026 (doi 10.1038/s41556-026-01958-0). Not the source of a sealed set, so a normal read. It re-analyses GSE134242 (our sealed rf2020_clone source) and GSE299000.

## The claim
Across recipient mice (bulk transplant of ~1,000 LARRY-barcoded HSCs each), the number of engrafting clones correlates strongly and positively with the average megakaryocyte/myeloid bias of the clones; read as support for an extrinsic competition model ("lineage bias is extrinsic"). Methods: clones with >= 2 cells kept; clone contributions normalised within each cell-type annotation before the bias metric; one correlation across recipients.

## Planned findings
- Finding: under fixed, intrinsic clone biases with no competition, sampling alone (cells sampled per recipient, the >= 2-cell filter, small-clone noise, per-annotation normalisation) produces a positive clone-number vs bias correlation across recipients of the published size.
  - Verdict: not found
  - Papers: the source itself (no null model for this correlation stated); Rodriguez-Fraticelli 2020 (potency intrinsic across split recipients). No re-analysis or rebuttal among its citers.
  - Classic search: owed (Actions map with held-out recall); Europe PMC citer screen done (3 citers, none relevant).
- Finding: the correlation, re-computed on recipients not used by the source's argument, holds or vanishes after a sampling-matched null.
  - Verdict: not found
  - Papers: as above.

## Units
- rf2020_clone (GSE134242): 9 recipients in 5 sample sets (sealed by clone; a recipient-level statistic would read sealed clones, so only structural counts per recipient are usable: cells and clones per recipient).
- GSE299000: public superseries (developmental HSC fate mapping, public 2026-03-30); its transplant sub-series and recipient count not yet listed.
- The first deliverable is design-level (a null-model recovery study, like chapter 2's #14), which needs structural per-recipient counts, not a sealed look.

## Status
Novelty: passes at citer level (classic search owed). Units: design-level study feasible from structural counts; a replication needs the GSE299000 recipient count. Power: not pretested.

## What would be ours
A sampling null showing whether a 2026 headline argument for extrinsic lineage bias survives; if it does not, a correction of a published inference (our strongest genre so far).

## Second novelty pass (2026-11-22)
- Classic search on Actions (results_lit/novelty_CAND-F2.json; run 45065297120): 5 phrasings plus the source's 48 references ranked by citations.
- **Recall check: PASS** for Dykstra 2007 (held out, found in the references); Rodriguez-Fraticelli 2020 not in the top 20 references (it is cited; the cut is by citations).
- Found: stochastic competition models of chimeras (2004), a minimal intrinsic/extrinsic commitment model (2009), the source itself. No null model for the clone-number vs bias correlation and no re-analysis of it. Verdict: not found (holds).
