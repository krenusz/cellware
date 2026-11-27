# CAND-A novelty check: cross-recorder test of the C14 claim (H25, 2026-11-18)

Candidate from lit/gap_outcomes_2026-11-18.md. Not a FOCUS step yet; rename to the step name if a chapter opens on it. Bence: "go fine check A and B".

- Searched: web (cross-data-set and cross-recorder fate-model transfer; DNA Typewriter mouse data availability), 2026-11-18; a fetch agent read the DNA Typewriter preprint v2 (bioRxiv 10.64898/2026.07.29.741625; Science 2026, doi 10.1126/science.ael0508): abstract, Results headings and figure titles; about 80% verbatim, the rest from the agent's summary.
- Source paper: the DNA Typewriter preprint is the data source. No partition of it is sealed yet, so this was a normal read; no per-cell results were taken. The petracer_embryo source (Colgan 2026) was read at claims level earlier today (source_claims_petracer_2026-11-18.md).

## Data facts (metadata only, nothing downloaded)
- **One embryo** (embryo #3), E13.5 only, zygote-rooted time-calibrated tree, 1,281,141 tips (655,701 backbone).
- Lineage and expression from the same nuclei (sci-RNA-seq3, single-nucleus). 138 cell-type labels transferred from the authors' E12.75-E14.25 atlas.
- GEO GSE341627 (public, 18 records); code on Zenodo 10.5281/zenodo.22801271 (shendurelab/dtt-mouse, open); Newick trees via the NextCell browser. Raw size not checked.

## Planned findings
- Finding: in an independent recorder, lab and stage, a cell's expression state beyond its annotated type predicts the type composition of its lineage relatives (the C14 claim, re-tested).
  - Verdict: partly published
  - Papers: Colgan 2026 (heritable fate programs, origin traces; PEtracer embryos); the DNA Typewriter paper itself shows siblings share type far above chance and predicts masked tips' labels from cousins (type from relatives, not state beyond type). "State beyond type predicts relatives' fate" in this embryo: not found.
- Finding: a model frozen on the PEtracer train embryos (E7.5-E10.0, whole cells) transfers to the DNA Typewriter embryo (E13.5, nuclei, other annotation).
  - Verdict: not found
  - Papers: no cross-recorder transfer found. But the two sets share no stage, assay or annotation, so a failed transfer would mostly measure platform and stage mismatch. Not a meaningful test as designed.

## Feasibility
- H22/H24: one embryo, so the unit is the clade within the embryo (e.g. clades rooted at gastrulation), not independent embryos. Sealing by clade is possible and probably powered, but it is a within-embryo test with no between-embryo replication.
- H21: 138 types and over 1M tips; the readout gate would pass on size.

## What is left that is ours
A within-embryo, held-out-clade re-test of C14 in an independent recorder. **That is a stronger confirmation of a partly published claim, not a discovery.** Its value is as supporting evidence if C14 is written up; it does not answer an open biological question. Do not open a chapter on it for discovery.

## Refresh 2026-11-27 (after the CAND-A-L1 look)
- OpenAlex claim-shaped queries (Haiku lane, all hits returned unfiltered; abstracts read by the orchestrator). In-kind sources for the C14 biology: new doi 10.64898/2026.11.11.765433 and doi 10.64898/2026.10.19.760045; known but not cited on 2026-11-18: doi 10.64898/2026.10.06.756974 (abstract level since 2026-11-17, a different tree deposit) and doi 10.48550/arxiv.2610.40447.
- CAND-A-L1 confirmed the C14 claim on dtt_founder A+B (T 0.091, z 4.87). Verdict unchanged: stronger confirmation of a partly published claim, not a discovery. See WRITEUP_C14.md, Value against the literature.
