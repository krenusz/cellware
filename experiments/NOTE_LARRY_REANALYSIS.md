# Reanalysis note: LARRY cytokine experiment (Weinreb et al. 2020) (internal draft, 2026-11-12)

Not a paper. Internal note. Publishing needs Bence's decision and a check of his employment contract. No
contact with the original authors (standing rule).

## Claims

1. **Pure instruction cannot reproduce the cytokine data.** With selection switched off, no gated fit exists
   under G-CSF (E40).
2. **Instruction is modest when selection is held to plausible strength.** Shift about 0.5 logit each way (G
   0.50, M 0.56; gated range 0.38-0.92), stable across four model variants (E47, E48, E49, E53/core).
3. **Small M-CSF clones are private to the M-CSF well.** 2-cell clones are seen in any other well 21% of the
   time, against 51% under the model. Real cells spread over libraries, so it is not an artefact. **It
   replicated on held-out clones (B37).**

## Evidence and strength

| claim | data | evidence | confirmation |
|---|---|---|---|
| 1 | fit set, profile over G-CSF selection (E40) | no fit with selection off; loss lowest at a 4-6.5x neutrophil advantage | within the model family |
| 2 | gated seeds, E47/E48/E49/E53 | 0.42-0.70 (G), 0.40-0.92 (M) | within the model family. Sealed cross-well looks (E6, E9/E10, E14) tested the model, not this parameter |
| 3 | fit set, then sealed held-out clones (B37) | lift 0.29 at >= 2 cells | **held-out, one look** |

## What we cannot say

- **Claim 2 depends on the selection bound.**
  - The bound (gamma <= 0.6, at most about one extra doubling per SD of bias over 48 h) comes from Rieger
    2009's death and doubling rates.
  - With selection unbounded (E46, gamma 1.3), the M-CSF instruction is not identified (0.6-7.5).
  - On the E40 ridge, instruction runs 0.5-3.7 logits.
  - So: "about 0.5 logit **if selection is physiologically bounded**".
- **Mechanism:** survival vs proliferation is not separable with these data.
- **Generalisation:** these are in-vitro LSK cultures only.

## Novelty check (2026-11-12)

- **Full texts of 283 open-access papers citing Weinreb 2020** (Europe PMC, 305 indexed citers).
  - None estimates instruction vs selection on the LARRY cytokine arm.
  - None reports well-private clones.
  - Matches were trajectory benchmarks, data reuse and unrelated "instruct" wording.
- **Not covered:** about 480 citers OpenAlex lists but Europe PMC does not index (paywalled, non-PMC), and
  preprints outside PMC.
- **Closest prior work:**
  - Weinreb 2020 itself: fate bias at day 2; survival "cannot be ruled out"; Supp Fig 7g.
  - Rieger 2009: instruction in sorted GMPs.
  - Lin/Naik 2024: restricted progenitors make small clones, which is consistent with claim 3.

- **All citers screened by title and abstract (2026-11-12):** 712 citers (Semantic Scholar), 622 with
  abstracts.
  - Two pattern matches, neither a quantification.
  - One is a 2020 review arguing cytokines instruct (Brown et al., IJMS 2020, doi 10.3390/ijms21072362): the
    opposing view, to cite.
  - The other is a 2026 preprint on ageing clonal selection.

## Before any outside use

1. ~~Re-check the non-indexed citers~~ (done, see above).
2. **Selection-prior sensitivity, mostly done from existing runs (2026-11-12):**

   | selection cap (gamma) | run | instruction G / M (logit, gated median) |
   |---|---|---|
   | none in the model (E38: uniform boost, no lineage-matched term) | E38 | 0.49 / 0.47 |
   | <= 0.35 (strict, Rieger) | E47 | 0.47 / 0.62 |
   | <= 0.6 (generous, Rieger) | E48 | 0.50 / 0.56 (range 0.38-0.92) |
   | <= 0.6, core code | E53 | 0.52 / 0.70 |
   | unbounded (fit 1.3, about 4x the plausible bound) | E46 | 0.86 / not identified (0.6-7.5) |

   - **Robust at about 0.5 across every physiologically plausible cap.** It breaks only when selection is
     allowed about four times stronger than doubling rates permit.
   - **Optional, not run:** a cap of about 0.9 would locate where the estimate starts to drift. That needs a new
     pre-registered GitHub Actions sweep (18 seeds, about 40 min).
3. A figure set:
   - presence lift with model bands (claim 3);
   - the instruction/selection ridge with the plausible band shaded (claims 1-2).

Sources: EXPERIMENT_LOG (E40, E46-E49, E53, B31, B34, B37), DISCOVERIES #2, #3, #9; lit/novelty_map_2026-11-12.md.
