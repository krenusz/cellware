# Chapter 6 plan: cells into space (2026-11-08)

Bence: "alright now lets go with your first choice for next cpt / 1. Cells into space (my pick)."

## Question

Can cells that each carry the core model's clone state (`core_model.py`), with local rules only, build a
gastruloid's A-P domains **and** its elongation? Can that model then predict an unseen spatial readout?

This closes the gap the 11-07 goal check named: "elongation rescue is still a descriptive rule". C3-S1 and C3-S2
put the core model into a 1-D rank-sorting simulator with a population-level elongation rule (the sorting
index H). Chapter 6 replaces both with cells in 2-D space.

**Serves:** the long-term goal (cells with local rules build a patterned tissue); program.md "Plan toward space".

## What the model must explain (targets; all open or spent data)

| # | fact | status | data |
|---|---|---|---|
| T-a | clones have stable A-P propensities (polarity per clone) | confirmed, sealed G1a/G6 (#20) | linescans |
| T-b | pairs order by clone state (6 < 1 < 3 ~ 7) | sealed pass G7 (#26, moderate) | linescans, G10 |
| T-c | the boundary is fixed under mixing ratios 4:1 to 1:4 (proportion feedback) | open, unconfirmed (#21) | G1b linescans |
| T-d | elongation differs by clone; 6/6a gastruloids are round | open + spent | G2 table (3,597), 0831 pure plate |
| T-e | elongation falls with the fraction of a pure clone in clone+bulk chimeras (20/50/80%), similar slope in 1.2, 5.2, 6A | **exposed**: authors' README results read; plate-confounded | `rev_chimera_elong` (175 gastruloids) |
| T-f | complementary clone-clone rescue (#22) | **not confirmed** out of sample (C4-S4: +0.25 SD vs 0.81) | must NOT be built in |

T-e and T-f together show what to aim at. The one supported mixing effect is that a defective clone gets
partial help from a diverse partner (bulk). It is not complementarity between two clones. The model has to get
this right without being tuned to the exposed numbers. T-e is used as a fit target and is labelled exposed.

## Model (C6-S2, to build)

- **Cells:** a 2-D cellular Potts lattice (or off-lattice particles, whichever is faster on this PC), about
  500-1,000 cells. Each cell has a clone label and an A-P score s = beta * z_clone' + cell noise. Here z_clone'
  is the core's state copy per gastruloid (`state_copy`, rho_g) after passages (`memory_through`).
- **Local rules (candidates; each one is a switch, compared by CV on open data):**
  1. *Differential adhesion by s:* sorting into A and P domains (gives T-a, T-b).
  2. *Proportion feedback:* a cell's posterior identity depends on the local posterior fraction, which sets the
     domain size (T-c, the G5/C3-S1 rule moved to the cell level).
  3. *Posterior-driven extension:* posterior-domain cells intercalate along the axis that points away from the
     anterior domain (convergent extension; Oriola 2024: mechanics sets the poles). Elongation then needs
     **both** domains to be present and sorted. Literature candidates to compare: posterior growth zone, and a
     fluidity gradient (Mongera 2018).
- **Readouts computed exactly as in the data:** linescan polarity along the midline (T-a to T-c), circularity and
  geodesic aspect ratio from the outline (T-d, T-e).
- **No rescue term.** If elongation in mixes comes out, it must emerge from rules 1-3.

## Steps

1. **C6-S1 (this step):** plan + pre-flight. No statistic is computed.
2. **C6-S2:** build the simulator on `core_model.py`. Smoke-test on synthetic clones, then check T-a to T-e on
   open/spent data with criteria pre-registered before the run. One-job PC limit; sweeps on Actions.
3. **C6-S3:** model selection over the rule switches by leave-one-condition-out CV on open data. Fix the model and
   write its predictions for the sealed sets.
4. **C6-S4:** sealed look(s), only if H12 + H14 + H15 pass and Bence OKs the partition.
5. **Stopping rule:** if no rule set reproduces T-d and T-e qualitatively on open data, close the chapter with
   that negative result. Do not add rules for each misfit.

## Sealed test candidates (counts only; nothing read)

| set | n | what the model predicts | rough power (H14; the model fixes the effect later) |
|---|---|---|---|
| **S1: Jan 2024 clone 1+6 pairs, passage 4** (`clone1+3_clone1+6_pass4_01272024.xlsx`, tag 1+6; 1+3 spent in G6b) | 34 gastruloids | share of gastruloids with clone 1 anterior to clone 6, after 4 passages | one-sided binomial: 0.50 at share 0.66, 0.69 at 0.70, 0.91 at 0.76. Passes only if the model predicts >= ~0.72 |
| **S2: morph 0831 partition C** (last partition, H13: Bence's verbatim OK) | 6a+bulk 13, 5s2+6a 11, 5s2+bulk 6 | 6a+bulk elongation (1 - circularity) over its member mean | 0.39 at +0.04, 0.66 at +0.06, 0.84 at +0.077 (the spent A+B estimate, inflated by the winner's curse). Plate confound as in C4-S2a |
| Raj reserve `clone3_pass3_07112024.xlsx` | - | memory; off-question for chapter 6 | not proposed |

A new public set with same-plate pure and mixed gastruloids plus outlines would be a better elongation test.
If the S1/S2 power gates fail, a data search restricted to that requirement comes before any look.

## Pre-flight (H15)

1. **Unit:** gastruloid (one outline, one line scan). Clones enter only as model inputs.
2. **Readout:** polarity from line scans, as in G1a (T-a to T-c, S1); 1 - circularity (S2, as in C4-S2a); geodesic
   aspect ratio (T-e, the authors' bend-robust metric).
3. **Used data first:** every fit target is open or spent data (G1a fit shard, G1b, G2, G3, G7/G10, 0831 pure
   plate + A+B, chimera set).
4. **Power:** see the table. Neither sealed set is powered unless the fixed model predicts a large effect. This is
   decided by H14 after C6-S3, not now.
5. **Exposure:** the chimera README (results per clone at 20/80%) was read on 11-07. A header check on 11-08
   printed one data row of `5+bulk.csv` (one gastruloid's blob metrics). The chimera set is therefore
   **open/exposed** and used for fitting only, never as a test. S1: tag counts and channel names only. S2: counts
   per condition only.
6. **Literature:** Oriola 2024 (communication sets proportions, mechanics sets poles); Mongera 2018 (fluidity
   gradient, zebrafish axis); Regalado/Shendure monoclonal gastruloids; chimeric gastruloids 2022 (cell-non-
   autonomous compensation). Read the methods for rule 3 before coding it.
