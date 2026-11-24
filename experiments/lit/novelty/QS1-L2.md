# QS1-L2 novelty check (H25, 2026-11-24)

Pre-gate check for a question from the QS1 search (Bence: "Yes, question search"). Not a FOCUS step. It was run after gate 2 had failed, to tell whether a train-only step is worth offering. It does not change the gate verdict.

- Searched: Europe PMC (title and abstract, sorted by citations), 5 phrasings, 2026-11-24. Hits: 0, 19, 1, 2 and 1. Every title was screened and 2 abstracts were read (below). OpenAlex was not used. The source references were not run.
- Source paper: the lsc_larry source preprint (bioRxiv 10.64898/2026.02.24.707683). Its partitions are sealed. Nothing more was read for this step: methods and the Fig 2 legend were read in C12-S1, and that is logged exposure. Its claims about the D40 ex vivo sample are **unchecked**.

## Planned findings
- Finding: a clone's persistence at D40 ex vivo predicts the presence of its sister cells in vivo, beyond D20 ex vivo size and state (same clone, two environments).
  - Verdict: partly published (in kind)
  - Papers:
    - Sakurai et al. Nature 2023 (doi 10.1038/s41586-023-05739-9): split-clone transplantation is used to support ex vivo expansion of single stem-cell clones. It is the same kind of design, used to show expansion, not concordance beyond state.
    - Blood Adv 2024 (doi 10.1182/bloodadvances.2024013932): culture conditions change clonal fitness and engraftment, traced by barcode. It compares conditions, not one clone across two environments.
    - Background: the C12 source shows that state relates to in vivo presence. C12 itself (published in kind) is the state part of this question.
  - Classic search: **weak, not a valid negative.** Three of the five phrasings put a wildcard inside a quoted phrase, a form Europe PMC does not match. The first returned 0 hits, and there was no recall gate. In this search no paper tested one clone's in vitro vs in vivo success beyond its state, but that is not established. Queries (TITLE_ABS expands to TITLE OR ABSTRACT; since 1900):
    1. (TITLE_ABS:barcod* OR TITLE_ABS:"lineage tracing") AND TITLE_ABS:"in vitro" AND TITLE_ABS:"in vivo" AND TITLE_ABS:clon* AND (TITLE_ABS:engraftment OR TITLE_ABS:repopulat*) AND (TITLE_ABS:concordan* OR TITLE_ABS:predict*)
    2. TITLE_ABS:"same clone*" AND TITLE_ABS:"in vitro" AND TITLE_ABS:"in vivo"
    3. (TITLE_ABS:"ex vivo" OR TITLE_ABS:culture) AND TITLE_ABS:clonal AND TITLE_ABS:barcod* AND TITLE_ABS:transplant* AND TITLE_ABS:fitness
    4. TITLE_ABS:"clonal fitness" AND TITLE_ABS:"in vitro" AND TITLE_ABS:"in vivo" AND TITLE_ABS:barcod*
    5. (TITLE_ABS:"split clone*" OR TITLE_ABS:"sister clone*" OR TITLE_ABS:"clone splitting") AND (TITLE_ABS:transplant* OR TITLE_ABS:engraft*)

## What is left that is ours
Possibly the part beyond state: whether in vitro clone fitness carries to in vivo fitness once state and sampled size are adjusted. Two problems come first.
- **Confound:** true clone abundance drives both readouts, and sampled size measures it with noise.
- **Source claims unchecked:** the source's own D40 ex vivo claims have not been read.

## QS1-L2-P1 update (2026-11-24, run https://github.com/krenusz/cellware/actions/runs/45446043968; `results_lit/novelty_QS1-L2.json`)
- **Replaces the weak search above.** Five frozen plain-word phrasings on OpenAlex plus the source references (sweeps/qs1_l2_novelty.txt, pre-registered in 0f579b5).
- **Recall gate: PASSED.** Wilkinson 2019 (10.1038/s41586-019-1244-x) is rank 1 for "clonal expansion ex vivo engraftment" and rank 14 in the source references. Sakurai 2023 is absent.
- **Filter: 55 of 119 screened slots dropped by the neutral-wording filter** (12, 8, 7, 10 and 18 per phrasing; 18 of 19 on "in vitro clonal fitness engraftment"). Drops in the reference list cannot be computed. Queries were not reworded. So the literature check cannot return "not found" on this set: the best verdict is "partly".
- **Limits:** the two field-word phrasings returned mostly off-field titles.
- **Abstracts read (9; the cap was 10):** 10.1038/s41586-019-1244-x, 10.1038/s41586-023-05739-9, 10.1016/j.omtm.2021.02.009, 10.1016/j.stemcr.2019.06.001, 10.1182/blood-2009-06-229757, 10.7554/elife.66922, 10.1182/blood-2010-08-303222, 10.1038/nm.2695, 10.1016/j.stem.2023.06.002 (and its preprint version), 10.64898/2026.10.21.760611. The fetch of 10.1182/blood.v100.13.4391 failed and it was not read.
- **Citers of Wilkinson 2019:** 374 titles listed, screened with a keyword filter only (clon, single-cell-derived, split, paired daughter, heterogene); titles without those words were not looked at. None of the 22 matches was in kind on its abstract (a gene-edited clone expansion study and its preprint, a clone-competition platform). The 2026 paired-daughter preprint (10.64898/2026.07.24.739718) abstract was not read.
- **Result of check 1:** no paper among those read states a relation between one clone's culture behaviour and its own transplant outcome. That is not a clean "not found", for the filter reason above. Verdict stays **partly**.
- **Check 2, source read: abstract only.** Both bioRxiv full-text URLs returned HTTP 520 (no Retry-After header, so no wait was applied) and Europe PMC full text returned 503. The source abstract (read via the preprint-server API) states that clonal programs are linked with ex vivo self-renewal, in vivo initiation and treatment response. That is the state-mediated version of L2's question, already published in kind (C12). It does not state a direct persistence-to-presence relation. Without the full text, the source's D40 claims remain **unchecked**.
- **Decision (pre-registered rule): L2 is not cleared.** The train-only estimate behind the size-only null is not offered. QS1-L2 stays parked. Reopen only on a full-text claims read of the source that states nothing about it.
