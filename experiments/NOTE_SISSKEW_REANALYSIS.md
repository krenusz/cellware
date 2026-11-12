# Reanalysis note: SIS-skew Bcor experiment (Tian et al. 2021, Immunity) (internal draft, 2026-11-12)

Not a paper. Internal note. Publishing needs Bence's decision and a check of his employment contract. No
contact with the original authors (standing rule).

## Claims

1. **A Bcor fate effect is not identifiable from SIS-skew.**
   - The model accounts for 40% editing efficiency (the paper's own figure), fate-specific expansion and
     sampling.
   - Under it, no fate parameter is recoverable at the study's size (R3: R^2 0.16-0.49, RMSE 0.44-1.06, over
     300 simulated datasets).
   - The paper's "gain of pDC fate potential" is matched by expansion plus sampling within 0.007 of the
     no-fate limit (B42).
   - That margin disappears under a ±0.5 control-label shift (S1).
2. **Bcor loss acts on persistence and expansion.**
   - One-arm clones are 2-4x commoner in KO wells at every size.
   - Uniform KO-arm expansion fits; no fate term is needed.
3. **The control ("null") wells are not null.** Control-vector sisters shift within clones: cDC1 -0.12,
   cDC2 +0.08, pDC +0.04. All CIs exclude 0, 454 clones. Their size ratio tracks fate.

## Evidence and strength

| claim | basis | confirmation |
|---|---|---|
| 1 | recovery study (identifiability) + fit-set test B42 + sensitivity S1 | a design-level result: it does not need a held-out set, but rests on our simulator |
| 2 | fit set, KO wells, 831 clones (399 with both arms) | fit set. The related "non-cDC1 persistence" pattern did **not** replicate on held-out clones (E51), so it is excluded |
| 3 | fit set, null wells 454 clones (205 with both arms) | fit set only, strong CIs; never tested on held-out clones |

## What we cannot say

- **That Bcor has no fate effect.** Only that this design cannot detect one.
- **The cause of claim 3.** Candidates: a vector, transduction or sorting effect, or the W (WT cKit+) vs C
  (Cas9 LSK) population difference. The data cannot separate them.
- **Whether other assays in Tian 2021** (beyond SIS-skew) support the Bcor fate claim. We did not reanalyse
  them.

## Novelty check (2026-11-12)

- **All 12 papers citing Tian 2021 in Europe PMC, read in full text.** None re-examines the Bcor fate claim
  or the control wells.
- One 2023 review restates it as established ("Bcor as a negative regulator of emergency dendritic cell
  development"), so the claim currently stands unchallenged.
- **Tian 2021 itself** reports no change in control-well barcode number or expansion and 1% skewed clones, but
  has no within-clone directional test. Claim 3 is finer than their test, not a direct contradiction.

## Before any outside use

1. Check Tian 2021's supplement for per-clone editing data. It might make claim 1 testable instead of "not
   identifiable".
2. Simulate claim 3's alternatives (vector effect vs population difference) and state which data would
   separate them.
3. A figure set:
   - recovery R^2 per parameter (claim 1);
   - within-clone shift in KO vs null wells (claim 3).

Sources: EXPERIMENT_LOG (B39, B40, R2/E51, R3/E52, B42, S1), DISCOVERIES #14, #15, #16, #18;
lit/novelty_map_2026-11-12.md.
