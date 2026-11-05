# Weinreb 2020 Supplementary Information: what it says about the cytokine experiment

Source: NIHMS1633606 Supplementary Information (PMC7608074), saved by Bence 2026-11-05 to
papers/weinreb2020_supp/weinreb2020_SI.pdf (gitignored). Read: Supp Methods 2.4-2.7, 8.1.

## Design (Supp Methods 8.1, with 2.5 for the culture)

- Input: Lin-Kit+Sca1+ (LSK) 33% + Lin-Kit+Sca1- (LK) 66%, enriched from native 10/90.
- 20,000 cells plated, barcoded by lentivirus (barcoding efficiency about 30%, so roughly 6,000
  barcoded founders), in the pan-myeloid cocktail: IL-3, FLT3-L, IL-11, IL-5, EPO, TPO, SCF, IL-6.
- Day 2: 25% of cells profiled (inDrops); the other 75% split evenly into four wells.
- **Conditions listed: full cocktail, M-CSF 20 ng/mL, G-CSF 20 ng/mL, EPO 3 U/mL (all in StemSpan).**
  The text numbers two of them "(iii)" and lists **no SCF well**, but our data (GSE140802) has a
  fifth, SCF, condition. Unresolved; Supp Fig 7 may show the final design.
- Harvest at day 4 and day 6; the fraction taken at each is not stated for this experiment
  (the main state-fate experiment took 30% at day 4).
- Proliferation about 1-2 cycles per day.

## Detection (2.5, 2.7, from the main in vitro experiment; assumed to apply)

- Later time points: cells detached with 2.5% trypsin for 5 minutes, then FACS-sorted for GFP+.
  So adherent macrophages are detached on purpose, though recovery is not measured; dead cells
  are excluded by the sort.
- Extensive anti-loss steps (washes, decoy copy for counting).

## What this pins in our models

1. **Split probabilities are known.** Each day-2 cell goes to sampling with p = 0.25 and to each
   well with p = 0.1875: a multinomial split with no "lost" bin at the split. E45's free r and
   w_M were not design-consistent: w_M should be 1, and losses belong to post-split survival.
2. **Founders per well.** With about 8 cells per clone at day 2 (about 3 divisions), the design
   gives about 1.5 founders per well before death. E38 fits nu about 0.55, so about 2/3 of
   founders, or the clones' day-2 size, is lost somewhere: a testable number, not a free one.
3. **The full-only pool (phi) has a candidate identity: LSK cells.** E38 fits phi about 0.22
   (14/18 seeds 0.19-0.28) against a 33% LSK share at plating. Stem-like cells that expand only
   in the full cocktail (it has SCF and TPO) would behave as this pool.
4. **Day-2 sampling is a measurement of K.** The 25% sample gives each clone's day-2 count (n2),
   which constrains clone size at the split directly.

## Supplementary Figure 7 (figures PDF, saved 2026-11-05 as papers/weinreb2020_supp/weinreb2020_suppfigs.pdf)

- Design panel: day 2 sample plus four wells (Mix, EPO, M-CSF, G-CSF), sampled at days 4 and 6.
  **SCF is again absent**, although the data has three day-4 SCF libraries. Unresolved; we keep
  treating SCF as a fifth well, as the data do.
- Counts: 65,075 cells; 25,749 in multi-cell clones; 5,257 clones; 1,601 state-fate clones;
  958 multi-condition clones (all match our files).
- Panel f, clones seen in both G-CSF and M-CSF: monocyte-only in both 65%, neutrophil-only in both
  10%, monocytes in M-CSF and neutrophils in G-CSF ("fate switching") 11%.
- Panel g: **monocyte-only clones expand more in M-CSF than G-CSF (73% vs 10% of clones), and
  neutrophil-only clones the reverse (68% vs 7%).** Authors' conclusion: cytokines act by regulating
  proliferation or survival of committed cells, plus possibly biasing uncommitted cells; "we cannot
  rule out that these data result from changes in survival alone".
- **Relation to E46.** E46 (lineage-matched selective expansion) was pre-registered before this
  figure was read. Panel g is the same data seen descriptively (all clones, held-out shard
  included), so it supports the mechanism's plausibility but is **not independent evidence** and
  is not a test of E46.

## Weinreb PNAS papers (abstract, discussion and section titles read 2026-11-05)

- **Fundamental limits on dynamic inference from single-cell snapshots** (PNAS 2018,
  10.1073/pnas.1714723115). Population balance analysis (PBA). A snapshot fixes dynamics only if
  the cell density *and* the rates of entry, exit, proliferation and death across states are known;
  different dynamics can give the same snapshot. Assumes memoryless dynamics and warns that hidden
  variables (chromatin and the like) break that. **Lesson for us:** at the level of end counts, a
  change in fate choice (instruction) and a change in growth or death by state (selection) are the
  same kind of confound. Separating them needs measured or pinned growth/death rates, which is why
  the instruction vs selection reading in E46 stays descriptive until such rates are anchored
  (candidate: literature division and death rates of GMP-stage cells under G-CSF / M-CSF).
- **Lineage reconstruction from clonal correlations** (PNAS 2020, 10.1073/pnas.2000238117).
  CLiNC: barcode co-occurrence between cell types (second moments of a branching process) encodes
  the hierarchy. It excludes cell death and feedback, and shows co-occurrence depends on sampling
  depth and on mature-cell division/death kinetics. Barcode analysis is "hypothesis-generating".
  **Lesson for us:** B27's co-presence lift is such a second moment, so it is sensitive to
  expansion and death after the split, which agrees with B28's conclusion that the low G-M
  co-presence arises after the split (the E46 hypothesis).
