# CAND-D novelty check: do cytokines instruct or select human blood progenitor clones? (H25, 2026-11-22)

Novelty-first search after C12 closed (Bence: "search for something we can discover and novelty"). Not a FOCUS step. Chapter 1-2 type: a source's headline interpretation tested against a model it did not fit.

- Searched: Europe PMC, no date limit, by citations, title and abstract, 5 phrasings (results_lit/novelty_CAND-D.json), 2026-11-22; 13 titles screened; 2 abstracts read. Citers of the source in Europe PMC: 0. Limit: a short AND-term search; the known instruction paper (Rieger 2009) did not come back, so recall is weak. A full map on Actions is still owed.
- Source paper: cHSPCTrack, Blood 2026 (doi 10.1182/blood.2026033834), data set chspctrack (A spent in C7; B, C sealed). Claims level only: the abstract.

## Planned findings
- Finding: under a model that separates fate choice from clone growth and survival, the cytokine effect on human progenitor clones is mainly selection of pre-existing clone fate states (or mainly instruction), using serial half-well samples of the same clone.
  - Verdict: partly published
  - Papers: the source abstract says single progenitors diversify stochastically and that cytokine inputs "modulate differentiation probabilities by introducing biases toward preferred fates", which frames the effect as directed; it does not state a test against selection. Endele et al. 2014 (doi 10.1016/j.yexcr.2014.07.011): review; instruction exists (time-lapse work), the debate is framed. Our chapter 1 did this for mouse cytokine LARRY wells (pure instruction rejected, instruction about 0.5 logit).
  - Classic search: `python -I experiments/novelty_search.py CAND-D --engine europepmc --no-refs --q "cytokine instructive selective lineage" --q "cytokine instruction hematopoietic lineage choice" --q "cytokines select hematopoietic progenitors" --q "clonal tracking cytokine differentiation human" --q "stochastic diversification cytokine clones"`. Found: no human clone-level instruction-vs-selection test.

## Units (structural counts only, 2026-11-22; no cell type, score or pseudotime read)
- 13,700 clones, one condition per clone (one well), 8 experiments with 0-7 donors each; conditions are partly confounded with experiment (experiments 1-2 have condition I only).
- Clones with >= 2 cells early (day <= 5) and late (day >= 8), per partition: condition I 130 / 64 / 59 / 59 (train / A / B / C); III 111 / 56 / 46 / 49; II 46 / 23 / 27 / 19; IV-VI <= 18 train. A spent, so B or C would carry a look.
- Within-experiment contrasts (experiments 6 and 7 hold I, II and III together) are smaller still.
- **Units: borderline.** Only the I vs III (and perhaps II) contrast has tens of serial clones per sealed partition.

## Status
- Novelty: passes at a weak-recall level (needs the Actions map with a held-out recall check).
- Units: borderline. Power: NOT pretested. No claim that it works.

## What would be ours
A human, clone-level separation of instruction from selection for cytokine effects, with the same clone seen early and late, challenging or supporting the source's "cytokine-induced" reading.
