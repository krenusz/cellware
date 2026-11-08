# Chapter 5: clone memory (parked 2026-11-08, data-limited)

Question: what carries a stem-cell clone's fate memory, and how fast does it fade?
Bence: "close (park) cpt5".

## Model (the chapter's lasting output)

`core_model.memory_through(z, n_div, events, rho_div, rho_event, rng)`: a clone's heritable state is copied with
retention `rho_div` per division and `rho_event[e]` per event (passage, split). Non-heritable context noise (a
well, a gastruloid) is separate and is not passed on.

| system | per division | per passage / split | source |
|---|---|---|---|
| blood progenitors (LARRY) | 0.964 [0.927, 0.997] | - | C5-S2 (E19c) |
| ES clones, A-P propensity | about 1 (no loss while expanding) | 0.50-0.68 (G3), 0.23-0.51 (G5), 0.46 (C3-S1) | C5-S2 |

- One shared law fails (C5-S2, K1 and K2): the parameters are cell-state and series specific.
- ES memory acts like a ratchet: kept through divisions, about half lost at each passage. What passaging
  resets is not identified (hypothesis only: dissociation / replating).

## Carrier search

| step | candidate | result |
|---|---|---|
| C5-S1 | pre-flights | GSE291244 fails (founder state, not memory); Raj ATAC passes structure (7 clones x passages 0-3) |
| C5-S4 | chromatin distinctness | all 7 clones distinct at passage 1; distinctness ratio test power 0.77 < 0.8; no look |
| C5-S5 | expression through a single-cell bottleneck | passage-1 bulk RNA self-match too weak; power 0.41 (0.19 at half gap); subclone RNA stays locked |
| C5-S6 | **A-P chromatin axis** | **real at passage 1: anterior 3/5/7 vs posterior 1/6, gap 0.38, about 5.6 SE out of fold.** Fade test power 0.14 (passage 2), 0.53 (passages 1-3 pooled); no look |
| C5-S6b | less noisy readouts | none better; the limit is 5 clones x 2 replicates |
| C5-S6c | public data | best match (hESC, ~980 clones, sister wells, multiome; Linneberg-Agerholm 2026) not public yet |
| C5-S7 | DiffAssay | growth only, not differentiation; n = 5, replicate 1 null; pre-flight fails |

Supporting outside evidence (not our test): in hESCs, mesoderm-biased clones carry pre-existing accessibility at
mesoderm enhancers (LRP6, Wnt) before differentiation (Linneberg-Agerholm 2026 preprint). Chromatin is the
best-supported carrier; its fade rate is untested.

## What stays sealed (all unread)

- Raj ATAC passage 2 (A) and passage 3 (B, last): pre-registered C5-S6 tests ready (`c5s6_ap_axis.py`
  `test(part)`, `test_pooled()`); the B look needs Bence's verbatim OK.
- F8MWB5 subclone RNA matrix (C5-S5 `test()`), authors' PCA and heatmap.
- DiffAssay replicate 2; Raj reserve (C5-S3).

## Re-open when

- The hESC sister-well multiome data are released (GEO at publication): test fade of clone chromatin signatures
  over p2 -> p3 -> differentiation with hundreds of clones.
- Or any set with >= 10 clones x >= 3 passages x >= 2 replicates of chromatin or expression. Then the C5-S6
  design (cross-fitted A-P axis, gap ratio) carries over unchanged.
