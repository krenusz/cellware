# Imaging collaboration brief (draft, 2026-11-03)

Working draft for an outside partner. Nothing has been sent.

## The question in one paragraph

In LARRY lineage data (Weinreb et al. 2020), sister cells split into G-CSF versus M-CSF culture at
day 2 still agree on their neutrophil/monocyte balance (r = 0.51 [0.33, 0.65], 78 clones). Every
model we fitted, from equations, learned rules, clone types and selection-aware simulators (E9-E26),
predicts 0.06-0.33. The clones seen under both cytokines are a fate-selected subset (B7):
neutrophil-pure clones under G-CSF mostly vanish under M-CSF. That selection already shows in the
day-2 transcriptome (B9). Snapshot sequencing cannot say whether a sister cell is instructed (it
changes fate) or selected (it dies or stops dividing). Time-lapse imaging can.

## The experiment we would ask for

- Mouse GMP-stage progenitors (or the LARRY day-0 Lin-Kit+Sca1+ population), filmed from
  first division.
- Split sisters at first or second division into G-CSF versus M-CSF wells (micro-wells or
  sister-tracking chambers; Rieger et al. 2009 / Schroeder-type continuous single-cell imaging).
- Per cell: division times, death, and end-point fate (CD11b/Ly6G/CD115 or reporter lines such as
  Lyz2-GFP / PU.1-eYFP), plus brightfield for label-free fate prediction (Buggenthin et al. 2017).
- About 100-150 sister pairs per condition pair would give a sister-agreement CI of about ±0.15.
  That is narrow enough to compare 0.51 with the models' 0.2-0.3.

## What it would decide

| Readout | Instruction (models E9-E25) | Selection (B7/B9 reading) |
|---|---|---|
| Death or arrest of N-primed sister in M-CSF | rare | common |
| Fate switch of a surviving sister | common | rare |
| Sister agreement among survivors only | ~0.2-0.3 | ~0.5 |

## What we bring

- A fitted simulator family and pre-registered predictions made before seeing the imaging data
  (one prediction per readout, written before analysis).
- Analysis code for lineage-tree kin correlations (E19-style) and clone-level selection tests.
- Everything is public in krenusz/cellware (branch modernize). Notes are in WRITEUP_E5_E25.md and
  EXPERIMENT_LOG.md.

## Candidate partners (to check before contacting)

1. **Schroeder lab, ETH Zurich (D-BSSE, Basel).** Long-term single-cell imaging of
   haematopoietic cells. Rieger 2009 (cytokines instruct lineage choice) is the closest published
   design. Best academic fit.
2. **Marr lab, Helmholtz Munich.** Label-free fate prediction from brightfield time-lapse
   (Buggenthin 2017). A computational partner, likely with access to existing movies.
3. **CellVoyant (Bristol).** Commercial live-cell fate imaging (FateView/FateCast). Bence
   interviewed there in January 2024. Possible route for a pilot. Their focus is iPSC quality,
   not haematopoiesis.
4. **AI4CellFate authors (Crick/Oxford, 2025 preprint).** Fate from early frames of live-cell
   imaging.
5. **Dawson lab (Peter MacCallum, Melbourne).** SPLINTR clone splitting across environments.
   They report cell-intrinsic versus environment-dependent clone subsets, the closest
   barcode-based result.

Cheapest first step: ask labs 1 and 2 whether existing sister-pair movies under G-CSF or M-CSF
(Rieger 2009 data) can be re-analysed for death and arrest by sister fate. That needs no new
experiment.

## Before any outreach

- Employment contract: side business needs written approval and the IP clause is broad. An
  unpaid academic collaboration on public code is probably fine but should be checked first.
- No email is drafted or sent until Bence says so.
