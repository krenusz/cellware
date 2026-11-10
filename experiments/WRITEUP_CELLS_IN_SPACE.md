# Can clone-state cells build a gastruloid? Chapter 6 (closed, exploratory)

Working write-up, 2026-11-10. Not a paper.
- Steps C6-S1 to C6-S8d are pre-registered in EXPERIMENT_LOG.md.
- Plan: CHAPTER6_PLAN.md. Credit ledger: CREDIT_C6.md. Retro: RETRO_2026-11-10.md.
- Atlas check: lit/data_triandafillou_schema.md.
- Earlier chapters: WRITEUP_LARRY_CYTOKINE.md, WRITEUP_SIS.md, WRITEUP_GASTRULOID.md, WRITEUP_RESCUE.md,
  WRITEUP_MEMORY.md.

Bence: "Close and write up" (2026-11-10).

## Question

Can cells that each carry the core model's clone state, with local rules only, build a gastruloid's A-P domains
and its elongation? Can that model then predict a readout it has not seen?

## Answer in one paragraph

**Partly, and the prediction part is untestable with the data we have.**
- After 12 model versions, a 2-D cellular Potts model gives one attached body with one posterior pole and
  realistic elongation:
  - partial-wetting adhesion;
  - a long-range pull among posterior cells (Guan/Hormoz);
  - all-cell crawling that starts after the pole forms.
- It also passes every open clone check, with the clone rule from chapters 1-3 unchanged:
  - Q1, clone order: 5/5 signs, rho 0.90;
  - Q2, fixed boundary under mixing: slope 0.18 vs data 0.17;
  - Q3, elongation falling with the low-competence share.
- No unseen test reached power 0.8 robustly. Every candidate depends on a quantity the open data cannot pin
  down:
  - the S1 clone-pair test depends on how fast each clone loses its memory;
  - the atlas size test depends on how cells orient their crawling;
  - the atlas alignment test is circular by construction.
- All sealed sets stay sealed. Chapter credit: about 3.25/10, of which about 1 is carried over from chapters 1-3.

## What we did

| step | what | result |
|---|---|---|
| C6-S2 | off-lattice particles with adhesion, proportion feedback and posterior push | sorts locally into patches, never one axis; elongation is an outline artefact |
| C6-S3 | (A) Wnt/T wave-pinning + taxis, (B) Potts | A: first single axis + elongation, but **Q1 fails** (every clone posterior-positive); B dropped |
| C6-S4 | fit clone-to-pole coupling by leave-one-clone-out CV | **fail**: strong coupling fragments the pole; pole and clone coupling trade off |
| C6-S5 | state-independent adhesion (taxis only) | right signs in sample, 10x too small; CV fail |
| C6-S7 to S7g | "state-first" sorting: Guan cohesion, Guan exact force, de Jong Potts port, 3-D, outer start, partial wetting | step 0 fails throughout: central clusters, detached clumps (complete-wetting energies), or 2-3 caps on one body |
| **C6-S7h** | + long-range posterior pull as a Potts force | **step 0 passes** (polar 0.88, one pole, attached); Q1 at the bar; Q2/Q3 fail |
| C6-S7i, S7j | diagnosis + 30 repeats | Q2/Q3 failures come from crawling (axis misaligned with the pole, saturation); elongation came before the pole in 8/10, opposite to biology |
| C6-S7k | crawling only by posterior cells, after onset | pole robust, no elongation |
| **C6-S7l** | all-cell crawling from 150k MCS + an alignment criterion (arm a) | **passes step 0, Q1, Q2, Q3**; S2 power 0.16 / 0.54 |
| C6-S7m | arm c (mediolateral crawling) as a sensitivity check | Q1/Q2 hold, **Q3 flips**; S2 0.29 / 0.81 (not robust) |
| C6-S8a | power inventory | S1 share 0.45 (power 0.006); 3s1 chimera C 0.34-0.80; no outside clone dataset |
| C6-S8c | S1 with clone-specific memory (spent G6a) | power 0.57 / 0.77 at the G6a point estimates, 0.07 / 0.15 at the CI ends: **not powered** |
| C6-S8d | atlas: 100 vs 300 seeded cells (529 gastruloids) | **arms predict opposite signs** (a -0.22, c +0.09): no test |
| C6-S8 check | atlas seqFISH axis alignment | **circular**: gastruloids picked for being uniaxial; axis drawn along the shape |

## What this chapter established

1. **A single posterior pole needs more than local adhesion.**
   - Short-range differential adhesion sorts cells into patches or several caps, never one cap. This holds in
     particles and in Potts, in 2-D and 3-D, and from random or outer starts.
   - Complete-wetting energies (the de Jong table) make posterior tissue detach.
   - A long-range posterior pull (Guan/Hormoz) on a partially wetting, attached body gives one pole. This agrees
     with Guan's conclusion that adhesion alone gives no robust single axis.
2. **Order of events matters for elongation.**
   - If crawling starts before the pole, the body stretches along a random axis. That misaligns shape and pole,
     and it also produced the Q2/Q3 failures.
   - Delaying crawling until after the pole forms, and requiring alignment, fixes both. It matches the biology
     (pole at about 72-96 h, elongation after).
3. **The clone rule carries over unchanged.**
   - Clone states from chapters 1-3 place clones in the right order (Q1 5/5) at about half the observed
     magnitude. No clone-specific parameter is fitted in space.
4. **Mechanism-dependence is the main limit.**
   - Two crawling modes that both pass the open form checks disagree on Q3 (C6-S7m) and on the size effect
     (C6-S8d).
   - The model is under-determined: open data does not choose how cells crawl.

## Why there is no unseen test

| candidate | blocker | what would unblock it |
|---|---|---|
| S1: Jan 2024 clone 1+6 pairs, passage 4 (sealed, n 34) | the prediction hinges on clone-specific memory loss; G6a CIs overlap (clone 1 -0.06 [-0.37, 0.24], clone 6 0.60 [0.18, 1.00]) | clone-specific propensity measured over passages 1-4 (more gastruloids per clone per passage) |
| S2: partition C | power 0.16-0.81 depending on arm | a larger partition, or the crawl mode fixed |
| C: 3s1 + bulk chimeras | power 0.34-0.80; direction exposed; % plate-confounded | same-plate design |
| atlas 100 vs 300 cells | arm a and arm c predict opposite signs | live imaging of cell crawling orientation (random vs mediolateral) in elongating gastruloids |
| atlas seqFISH alignment | axis drawn along the shape on uniaxial-selected gastruloids | unselected gastruloids with an expression-defined axis |

The two measurements that would make chapter 6 testable:
1. **Clone-specific memory loss across passages.**
2. **Crawling orientation during elongation.**
Neither exists in public data (lit/data_search_c6s8.md, 2026-11-10).

## Limits and disclosures

- **12 model versions** (H18 retro on 2026-11-10). Q2/Q3 targets were seen before the C6-S7l fix, so those
  passes are capped. Q1 passed exactly at its threshold (rho 0.90).
- **Borrowed mechanisms:** the long-range pull is Guan/Hormoz's. The Potts machinery is de Jong's, with the
  energies changed.
- **Model calibration:** aspect ratio about 3.8 vs real 1.2-2.2 at 120 h (Blotenburg 2025). Shape magnitudes
  were converted to data units by the pure-gap ratio (0.180 / model gap).
- **2-D lattice, 300 cells, one time mapping** (250k MCS = 120 h, assumed).
- **My mistake, corrected:** C6-S8a used one shared memory rate for both clones, although G6a had already
  rejected that. C6-S8c redid it with clone-specific rates; the conclusion did not change.
- The atlas brightfield set is sha256-locked and unread (values). It stays available for a future test once
  the crawl mode is known.

## State of the sealed sets (unchanged by this chapter)

- **S1:** sealed, unread.
- **S2 partition C:** sealed.
- **Revision chimera files:** locked; only label counts read.
- **Triandafillou atlas:** locked, schema only.

## Credit

About 3.25/10 (CREDIT_C6.md):

| component | points |
|---|---|
| Form | 2 |
| Clone order | 0.75 |
| Mechanism | 0.5 |
| Unseen | 0 |

The 4 "unseen" points cannot be earned with current data.
