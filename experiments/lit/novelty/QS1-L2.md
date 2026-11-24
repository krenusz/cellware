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
