# C17 source papers, claims level (H20/H25, 2026-11-18)

Bence: "1" (claims-level read before any test). Read by web-fetch subagents (Opus 5.5): first ~40k characters verbatim, the rest as a model summary. Claims, headings and conclusions only: no numbers, panels, tables or supplements. Both data sets will be sealed before any statistic, so this is recorded as accepted exposure (claims level).

## 1. Kinetics-based model of haematopoiesis (Rodriguez-Correa et al., Nat Cell Biol 2026; PMC13278965; corrected 2026-10-16)
- **Claims:** single-HSC transplants plus clonal-progeny phenotyping order HSCs on one hierarchical trajectory; lineage biases are transitory states, not stable ones; **apparent biases arise from cell-extrinsic feedback and clonal competition** (slow vs fast engrafting clones filling mature compartments to a size limit). Kinetic position and potency are **intrinsic and heritable** (daughter-HSC secondary transplants).
- **Model:** phenomenological chimerism curves, then a compartmental ODE (three stem subpopulations, progenitor chain, mature compartment; Bayesian fit) and a feedback simulation with a progenitor carrying capacity. Flow-level, not molecular.
- **Evidence for "extrinsic":** inverse relation between clone output and co-transplanted support marrow; **re-analysis of published LARRY barcoding data** (more engrafting clones per recipient goes with more myeloid bias; clones compared across pooled recipients); lymphoid-deficient hosts change slow clones' skew.
- **Not done:** the **same clone followed in two recipients** (identical sister clones in different hosts); instruction vs selection is not framed (their models are selective in kind); clone-state memory is not modelled.
- **Open questions they state:** the molecular basis of restricted output in slow HSCs; mapping to native haematopoiesis; whether slow clones are dormant HSCs; ageing selection.

## 2. Cortical TF regulatory networks (Ding et al., Nature 2026; bioRxiv PMC12485932; Nature version PMC12999477 not read)
- **Claims (clonal part, summary level):** CRISPRi of single TFs in barcoded radial-glia clones **shifts clonal fate biases** (for example, more interneuron-biased clones with one knockdown, more excitatory-neuron-biased clones with another, fewer dual-fate clones with a third), read as lineage plasticity behind the population screen's output changes.
- **Not done (as far as read):** fate choice is not separated from clone growth or survival (clone-size distributions sit in extended data); no sister-similarity or inherited-axis analysis; no comparison with other systems' lineage data.
- **Open questions they state:** chromatin-level mechanisms in radial glia; regulators of protracted human maturation; interneuron subtypes; migration roles.

## 3. Rodriguez-Fraticelli et al. 2020 (Nature 583:585-589, doi 10.1038/s41586-020-2503-6; source of GSE134242)
- Read 2026-11-18 from Bence's PDF copy: abstract, figure and extended-data titles, and sentences stating conclusions (no numbers, panels or tables). Accepted exposure, claims level.
- **Design (stated):** expressible LARRY barcoding of adult HSCs, lineage plus transcriptome during transplant reconstitution; for the engraftment programme, half of the barcoded LT-HSC compartment was sampled and the other half **randomly split into two equal parts and transplanted into two secondary recipients**; an "equipotent HSC null model" is used as a comparison.
- **Claims:** many engrafted clones have low progeny output, contribute biasedly to the Mk lineage and carry a quiescence-linked signature; **long-term potency is an intrinsic and heritable property** that propagates through transplantation, with a clone-intrinsic transcriptional programme; an in vivo CRISPR screen finds TCF15 required and sufficient for HSC quiescence and self-renewal.
- **Figure titles:** simultaneous lineage and transcriptome sequencing; a clonal molecular signature of serial repopulation capacity; in vivo CRISPR screening of regulators of HSC output; TCF15 marks functional LT-HSCs.
- **Not claimed (as read):** whether a clone's **lineage bias** (not its potency) agrees between the two recipients it was split into, or a model separating intrinsic and extrinsic components of lineage bias.

## Consequences for C17 (to go into the S1-S3 pre-registration)
| question | verdict | what is ours |
|---|---|---|
| Is HSC lineage bias intrinsic or extrinsic, tested with the same clones in two recipients? | partly published, and the two sources pull opposite ways: potency intrinsic and heritable across split recipients (2020), lineage bias extrinsic (2026, other designs) | a direct test between them: within-clone **lineage-bias** agreement across the two recipients a clone was split into, against a clone-shuffled null, separated from potency; also a held-out system for the joint model |
| Do TF knockdowns instruct fate or select clones (fate choice vs clone growth)? | not found (shifts in clone fate bias are published; the split is not) | the chapter 1-2 instruction-vs-selection model on a new system, clone = unit, guide = clone-level treatment |
| Which TFs move cells along an inherited, fate-predictive axis? | not found | the perturbation x lineage join |
