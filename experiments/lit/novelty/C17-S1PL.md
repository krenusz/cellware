# C17-S1PL novelty check (H25, 2026-11-20): pluripotent-class leads (whole-embryo recorder, trunk-like structures, neuromesodermal organoids)

Bence 2026-11-20: "go but also do a full novelty check", "1 i mean" (= S0 gates for the pluripotent leads).

- Searched: three passes on 2026-11-20 (Opus 5.5; lookups of the leads by the Haiku lane):
  1. **Deep pass** (scratchpad script modelled on C17-S1T's; results_lit/novelty_C17-S1PL_deep.json): OpenAlex, relevance-ranked title+abstract search, 50 per phrasing, 16 field phrasings + 4 project phrasings; forward citations (topic-filtered) of 9 key papers; full reference lists of Phillips 2019, BMC Biol 2024, MemorySeq and the DNA Typewriter paper. 951 unique works, 755 after the neutral filter. **Recall gate 9/9** (Phillips 2019, BMC Biol 2024, MemorySeq, kin correlation 2016, Memory from variability 2026, clonally heritable expression 2024, quantitative fate mapping 2022, DNA Typewriter 2026, the sealed embryo source). Stated limit: by search phrasing alone, recall is 2/9; the other 7 were found through citation links. A re-run with 10 more field phrasings could not run (the OpenAlex keyless daily budget ran out, HTTP 429).
  2. **Classic search** (results_lit/novelty_C17-S1PL.json): Europe PMC (includes PubMed), no date limit, sorted by citations, title+abstract, 7 phrasings in the field's own words; 126 titles screened. Source references (results_lit/novelty_C17-S1PL_refs.json): the DNA Typewriter paper's 115 references ranked by citations, top 20 screened (tree methods, aligners, atlases, the nematode lineage classics; nothing on memory).
  3. **Screening:** 186 deep-pass titles with a topic term and all 126 classic titles screened; 14 abstracts read (Europe PMC).
- Source paper: none of the three lead sets is sealed or used. Their source papers were read at **claims level only** (abstracts), so that a later hold-out on these sets stays clean: DNA Typewriter mouse (Science 2026, doi 10.1126/science.ael0508; preprint 10.64898/2026.07.29.741625); DuTracer dual-nuclease recorder (Cell Rep 2025, doi 10.1016/j.celrep.2024.115105); the trunk-like-structure tracer study (GSE220949; "Reconstructing regulative progenitor behaviors in synthetic embryos"; paper record not found, GEO title only). The sealed embryo source (Colgan 2026) was not reopened.

## Planned findings
- Finding: a third pluripotent-class system gives sister-clade fate agreement that is far above chance, rises with developmental time, and is mostly commitment (as C17-S1P / C17-S1T on the sealed embryo set).
  - Verdict: published
  - Papers: DNA Typewriter mouse 2026 (doi 10.1126/science.ael0508): tree siblings share cell type far above chance; heterotypic siblings mark terminal differentiation; a dated hierarchy of cell-type couplings from clade co-occurrence. FateVec preprint 2026 (doi 10.64898/2026.10.06.756974): the same recorder with tetraploid complementation; loss of potential is progressive and asynchronous, also in a stem-cell-derived embryo model. DuTracer 2025: lineage relationships of cell types in embryoid bodies and neuromesodermal organoids. Colgan 2026 (claims level). Quantitative fate mapping 2022.
  - Classic search: not required (published); the passes above cover it.
- Finding: on a time-calibrated whole-embryo tree, where the time since a split spans a wide range, sister agreement still decays with time since the split after commitment at the split is fitted; per-division memory is separable from commitment (the C17-S1T nested test, which failed on the sealed set's narrow k range of 2.4-3.6 edges).
  - Verdict: partly published
  - Papers: FateVec 2026 (fate-bias dynamics on these trees; no per-division memory parameter in the abstract); scPFA 2026 preprint (doi 10.64898/2026.09.21.753290; latent expression programs evolving along lineages, heritable programs, not fate); PATH 2024 (heritability vs plasticity from phylogenetic correlation, assumes equilibrium); kin correlation 2016; quantitative fate mapping 2022 (commitment times, no memory); BMC Biol 2024 (memory falls with differentiation over 1-2 divisions). Not found: per-division fate memory estimated separately from commitment, by nested models, on a dated tree.
  - Classic search: `python -I experiments/novelty_search.py C17-S1PL --engine europepmc --no-refs --q "transcriptional memory cell divisions" --q "epigenetic memory differentiation" --q "heritable gene expression single cells" --q "sister cells fate decisions" --q "lineage restriction mouse embryo" --q "fate commitment lineage tracing" --q "memory loss per division"`, plus source references `--source 10.1126/science.ael0508` (OpenAlex, before the rate limit). Results in results_lit/novelty_C17-S1PL.json and _refs.json; 126 titles and the top 20 references screened; the 10 most relevant abstracts read (listed under Papers here and below). Found: memory and commitment studied apart, not separated on one tree.
- Finding: memory loss is tied to events (passage, commitment, reprogramming) rather than to division count, across systems (the reading from C5-S2 + C17-S1T).
  - Verdict: partly published
  - Papers:
    - Bacteria (Microcolony-seq, Cell 2025, doi 10.1016/j.cell.2025.08.001): growth to stationary phase erases inherited state, so an event erases memory.
    - Reprogrammed stem cells (Nature 2010, doi 10.1038/nature09342): somatic memory **survives** reprogramming, at low passage. Cultured fibroblasts (PLoS Genet 2016, doi 10.1371/journal.pgen.1005819): site-of-origin memory survives many generations in culture. Not every event erases memory, which is a counter-example to a simple event law.
    - Human cells (Nature 2006, doi 10.1038/nature05316): protein-level memory decays over about 1-3 generations, i.e. on a division timescale.
    - T cells (PNAS 2022, doi 10.1073/pnas.2116260119): heritable division speed within a clone over up to 10 divisions.
    - Theory (Memory from variability 2026): memory length is trait-specific.
    - BMC Biol 2024: memory lost with differentiation.

    Not found: a test, inside one mammalian system, of event-tied versus division-tied memory loss for fate, or a comparison of the two across systems.
  - Classic search: same Europe PMC run and screening as above; "memory loss per division" returned 8 hits, none on fate.
- Finding: cell-state memory compared across classes (blood, pluripotent, neural) on one scale.
  - Verdict: partly published
  - Papers: as in lit/novelty/C17-S1T.md (memory measured in several systems within single studies; no cross-system parameter comparison). Nothing new in today's passes.
  - Classic search: same run; the project phrasings ("cross-system comparison cell state memory", "memory per division blood embryo organoid") returned no such comparison in the deep pass.

## What is left that is ours
- Not finding 1. A value from these sets would be an input to the table, as C17-S1P was. The DNA Typewriter paper reports sibling agreement itself.
- The C17-S1T nested test (per-division memory vs commitment) on a dated tree with a wide range of time since split. The sealed embryo set could not separate the two on its narrow k range; a single 1.28M-cell dated tree could. This tree has one embryo, so the test unit would be clades, not embryos.
- The events-vs-divisions comparison inside and across systems.

Value judgement: the leads matter mainly for finding 2, i.e. as a deeper tree for the stage-aware memory model, not as another "agreement is high" value.

New data leads found by the search (not acted on; watch list):
- monoclonal gastruloids with lineage recording (2025 preprint, doi 10.1101/2025.05.23.655664): pluripotent class, heritable modes of early development;
- FateVec's stem-cell-derived embryo-model trees (same recorder).
