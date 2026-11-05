# Discoveries register

Every finding about the data or the biology, with its evidence (experiment ids in
EXPERIMENT_LOG.md), its strength, and a literature cross-check. Status of the check:
**consistent**, **in tension**, or **not yet checked**. Methodological lessons are at the end.
Started 2026-11-05; entries before that date are collected from the log and FINDINGS_2026-11.md.

Data set for entries 1-12: Weinreb et al. 2020, LARRY cytokine-perturbation experiment
(day-2 sisters split into full cocktail, G-CSF, M-CSF, Epo, SCF; read out days 4 and 6).

## Fate decisions

| # | discovery | evidence | strength | literature | check |
|---|---|---|---|---|---|
| 1 | Cytokines act mainly by **selection on a fate state that already exists at day 2**; G-M co-present clones are monocyte-leaning under G-CSF. | B7, B9, B11, B13-B16, B21 | strong | Weinreb 2020 (fate bias readable at day 2; Supp Fig 7g, lineage-matched expansion, survival "cannot be ruled out") | consistent |
| 2 | **Pure instruction cannot produce the data** under G-CSF (no fit with selection switched off). | E40 | strong within the model family | Rieger 2009: instruction in >= 34% (G) / 65% (M) of sorted GMPs, so instruction exists, but GMPs are a later, sorted stage | consistent (both act) |
| 3 | **Instruction is modest and stable**: about 0.5 logits each way under G-CSF and M-CSF, almost unchanged across E38, E47, E48, E49 whatever selection is added. | E38-E49 | moderate | Rieger 2009, Mossadegh-Keller 2013 (M-CSF instructs HSCs), Endele 2017 (CSF-1/Src instructs) | consistent |
| 4 | G-CSF gives neutrophil-leaning clones a **2.5-6.5x survival/expansion advantage**, trading off against instruction along a ridge. | E40 | moderate | Kimura 2009 (GM-CSF/STAT5 controls GMP proliferation and survival) | consistent (qualitative) |
| 5 | **Lineage-matched selection at a plausible strength** (gamma ~0.3-0.6: at most about one extra doubling per SD of bias over 48 h) fits B7 G; the fit pushes gamma to whatever cap is set. E46's unbounded 1.3 is about twice the plausible bound. | E46, E47, E48 | moderate | Rieger 2009 supplement: ~19% early death in both M-CSF and G-CSF, ~2 doublings on day 1, slower on day 2 | consistent (bound derived from it) |
| 6 | **Fate memory is near-full over 2 days**; the apparent fading (B21) is sampling noise. | E41b, B21 | moderate | Naik 2013 (heritable lineage imprinting in sister progenitors); Nat Commun 2026 (clonal memory of division in human HSPCs, >= 2 divisions) | consistent |

## Clonal structure of the experiment

| # | discovery | evidence | strength | literature | check |
|---|---|---|---|---|---|
| 7 | The **full-cocktail culture was seeded from a largely separate set of clones** (about 22% of clones mainly there). | B12, E37, E38 | moderate (fitted structure) | Weinreb 2020 SI 8.1 and Supp Fig 7: day-2 cells split evenly into four wells, Mix included, so not a separate seeding; alternative: LSK-like cells expand mainly in the full cocktail (SCF, TPO) | **in tension** |
| 8 | **Detection does not differ by cell type** (neutrophil/monocyte barcode capture ratio 0.91 [0.72, 1.20]); sister competition at the split cannot explain co-presence. | B28 | moderate | Weinreb 2020 SI (design, capture) | consistent |
| 9 | **Small M-CSF clones are private to the M-CSF well**: 2-cell clones seen in any other well 21% (model 51%); real cells spread over libraries, not an artefact; half as often seen at day 2. They explain the M-presence and low G-M co-presence misfits. | B31, B32, B34; replicated on held-out clones (B37: lift at >= 2 cells 0.29 vs fit set 0.31) | strong (replicated) | Lin/Naik 2024: restricted downstream progenitors (MDPs, CMPs) make small clones | consistent (candidate explanation, untested) |
| 10 | **G-CSF-private clones are neutrophil-committed at every size** (70% neutrophils vs 41% in shared clones; privacy grows with size). | B34 (E50 tried a G-only neutrophil class: 0/10 gated, it strips mixed G clones) | moderate | Weinreb 2020 Supp Fig 7g (neutrophil-only clones expand more in G-CSF) | consistent |
| 11 | **A clone's day-2 abundance predicts how many wells it reaches**, beyond independent seeding (gradient 0.13 vs model [-0.09, 0.09]). Clones differ in how many cells they had at the split. | B35 | moderate | no direct reference found; follows from splitting each clone's day-2 cells into wells (Weinreb SI: day-2 sample measures clone size at the split) | expected from design |
| 12 | Every model **over-shares clones with the Epo and SCF wells** (about 2x). For SCF it is a counting artefact: SCF wells are 71% undifferentiated, and counting all cell types fixes it. For Epo it is real: the erythroid cells come from clones absent from the G/M wells; clones shared with G/M make myeloid/basophil cells in Epo. | B32, B36 | moderate | Grover 2014 (Epo pushes multipotent progenitors to erythroid fate): diversion of shared clones tested and not supported; an erythroid-lineage clone subset fits the data | SCF explained; Epo in tension with diversion |

## Methodological lessons (how not to fool ourselves)

| # | lesson | evidence |
|---|---|---|
| M1 | Pass/fail at p >= 0.05 rewards wide predictions; compare models with a proper score (CRPS) and only over informative statistics. | B29, E47 amendment |
| M2 | Statistics computed on all clones leak the held-out shard; compute on the fit set. | B7 audit, B29 |
| M3 | A good fit to an aggregate statistic can come from compensating errors; decompose it (e.g. by clone size) before crediting it. | B31 (E46 co-presence) |
| M4 | A held-out test must be powered for the question: the Epo/SCF concordance cannot see selection. | B30, B33 |
| M5 | Check every model change against every condition before registering it; two designs were rejected this way before any fit. | E49 restricted class, E43 |

## Cross-check log

- 2026-11-05 (RETRO_2026-11-05b): entries 6, 7, 11, 12 cross-checked; 7 moved to "in tension". B36 tested 12's candidate: SCF explained, Epo not.
- 2026-11-05 (B37): held-out look replicated discovery 9's small-clone co-presence deficit; strong vs plausible selection not decided.
