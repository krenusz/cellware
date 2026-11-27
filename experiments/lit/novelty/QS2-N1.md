# QS2-N1 novelty check (H25, 2026-11-26)

- Searched: OpenAlex classic search (novelty_search.py functions, run by the Haiku lookup lane from a scratchpad driver, so no results_lit file was written), 2026-11-26. Queries: "clonal dispersal mouse gastrulation", "sister cells spatial coherence", "clone mixing epiblast", "lineage tracing mesoderm migration". Top 25 titles screened per query; the 10 most related abstracts checked (3 had no abstract in OpenAlex, and Europe PMC was blocked by TLS and 403). The source's references were ranked.
- Source paper: doi 10.1038/s44318-025-00441-0 (the source of mouse_gast_tree, which is sealed). Claims level only, read earlier (DS-EMB-N; partitions.mouse_gast_tree prior_exposure): section headings and sentences on sisters, generations and fate restriction. QE2's row records that the source publishes sister trajectory similarity.

## Planned findings
- Finding: in tracked gastrulating mouse embryos, sister cells end closer together than non-relatives matched on birth frame and birth separation (clonal coherence beyond starting position).
  - Verdict: partly published (in kind)
  - Papers:
    - The source reports sister trajectory similarity (claims level), close in kind.
    - Classic clonal analyses of mouse gastrulation report dispersal and intermingling of clones (e.g. doi 10.1242/dev.125.13.2397, doi 10.1242/dev.124.9.1631, doi 10.1242/dev.122.5.1523). They are qualitative, without start-matched controls.
  - Classic search: recall passed (the classic clonal-dispersal papers rank 1). None of the abstracts read states the start-matched comparison. Three related items were unreadable (doi 10.1038/ncb3024, doi 10.1016/s1534-5807(04)00133-9, doi 10.1016/j.mod.2008.06.007).

## What is left that is ours
A held-out, start-matched quantification of sister coherence, as a test. The source states sister trajectory similarity descriptively, so this is "in kind": held for Bence (RETRO_2026-11-26 decision 1). No step opens on it.

Closed 2026-11-27 under decision 1 (Bence: "Only if source differs"): the same-data check is the source's claims-level statement of sister trajectory similarity (above), so the claim is reported in kind on the same data.
