# Why does mixing rescue elongation? Chapter 4 (closed, data-limited)

Working write-up, 2026-11-07. Not a paper. Steps C3-S2 and C4-S1 to C4-S4 are pre-registered in EXPERIMENT_LOG.md;
the data store is mapped in DATA_MAP_RAJ.md; the retro is RETRO_2026-11-07c.md. Chapters 1-3: WRITEUP_LARRY_CYTOKINE.md,
WRITEUP_SIS.md, WRITEUP_GASTRULOID.md.

## Question

Chapter 3 found that gastruloids made from one mouse ES clone often fail to elongate, and that mixing two clones
rescues elongation (#22, about +1 logit, with a smaller extra gain when one clone prefers the head and the other the
tail). Chapter 4 asked why.

## What we did

| step | what | result |
|---|---|---|
| C3-S2 | Rescue from the core model's local rules: a **sorting index** (share of a gastruloid's cell-state variance carried by which clone a cell came from) | fits the 2024-25 table as well as the 3-bonus descriptive rule with one parameter (CV -1248.9 vs -1248.6; no mixing effect -1282.8); weaker without clones 2/4; open data |
| C4-S1 | Data search, then a full crawl of the Raj data store (12,076 files, 226 GB) | found the authors' Aug-Sep 2026 revision data; locked before any statistic |
| C4-S2b (+R) | Head/tail marker (T vs Igfbp5) domain separation by composition, 20 gastruloids | **uninformative**: the ordering check had power 0.20; pure clone 6 saturated; a first "not supported" was withdrawn |
| C4-S3 | seqFISH chimeras for per-cell clone identity with position | fails pre-flight: 5 labelled control gastruloids (1+3 x2, 1+6 x3), smallest achievable p 0.10; not downloaded |
| **C4-S4** | **Sealed out-of-sample test** on new subclones (Aug 2026 plates): 5s2+6a vs the mean of its pure members, Y = 1 - circularity, mixture partitions A+B, power 0.80 | **fail**: +0.021 (+0.25 SD vs predicted +0.81), one-sided lower bound -0.037; only bulk+6a beat its members (+0.077 [0.016, 0.141]) |

## What this chapter established

1. **Rescue did not replicate on new clones.** On the one powered sealed test, a head-preferring plus tail-preferring
   pair of new subclones was not more elongated than its members' average, and not more than its better member. #22
   stays **unconfirmed**: it fits the 2024-25 table (G2, G2b, C3-S2) only.
2. **What did show:** a severely defective clone (6a, round alone) improved with bulk cells. This fits "a diverse
   partner partly makes up for a defect" (G2b already saw clone 4 only partly rescued) better than "two
   complementary clones rescue each other".
3. **The mechanism is not testable with public data.** The sorting index is a model-level hypothesis, consistent with
   the 2024-25 table and untested beyond it. Spatial readouts exist only for 2-6 gastruloids per condition.

## Limits

- Pure and mixed gastruloids of the Aug 2026 experiment were on different plates; a plate shift could hide or create the
  contrast. Partition C (11 gastruloids of 5s2+6a) stays **sealed** for a design with pure and mixed wells on one plate.
- The readout changed (continuous circularity from outlines, not a human elongated/not call); the effect prior was
  mapped from logits through a logistic latent scale (an assumption).
- Subclone states (5s2, 6a) were taken from their parents (5, 6); memory fades over passages (#23), so the pair may be
  less complementary than its parents.

## Data still sealed

Mixture partition C (raj_morph_0831); clone + bulk 20/50/80% chimeras (geo_metrics, clone_bulk_chimera_08232026; the
authors' result was read); the authors' T/Igfbp5 summaries other than separation_metrics; all chapter 3 reserves
(WRITEUP_GASTRULOID.md, "Data still sealed").
