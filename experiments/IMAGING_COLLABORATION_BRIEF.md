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

## Outreach plan (elaborated 2026-11-03)

**Step 0: contract (mostly settled).** Cellware is covered by the signed Side-Project Agreement
(effective 13 Nov 2024). Bence keeps the IP in synthetic biology, provided no employer resources
or confidential information are used. An unpaid academic collaboration on public Cellware code
fits that. A paid or commercial arrangement (e.g. a CellVoyant pilot) may count as "other
business" under Cl. 5.1 and would need written approval first. Use a personal email and personal
hardware.

**Step 1: the free ask (re-analysis of existing movies).** Target: the Schroeder lab (ETH Zurich,
D-BSSE). Rieger et al. 2009 and later work filmed single GMPs and their progeny under G-CSF and
M-CSF with continuous tracking. The question to ask: do their archived sister-pair trees allow
scoring death, arrest and fate per sister by condition? If yes, the analysis is ours, and their
cost is a data share. Send one short email, attaching the one-page question and this table.

**Step 2: the computational partner.** The Marr lab (Helmholtz Munich) co-authored the
brightfield fate-prediction work (Buggenthin 2017) on Schroeder-type movies. They may hold
movies plus tracking already processed. Contact in parallel with step 1, with the same ask.

**Step 3: the new experiment (only if 1-2 give nothing).** A split-sister G-CSF vs M-CSF
time-lapse from LARRY-like progenitors, 100-150 sister pairs per condition pair. That is a
wet-lab project, so it needs a host lab. Offer pre-registered predictions plus analysis.

**Step 4: the industry angle (optional).** CellVoyant (Bristol). Bence interviewed there in
January 2024, and their platform does live-cell fate prediction. Haematopoiesis is not their
focus, so pitch it as a validation dataset for FateCast-type models. Contract check first
(step 0, commercial).

**What we send.** A one-page question, the readout table above, a link to the repo, the
pre-registered predictions per readout, and a clear "we do the analysis".

**What would make it compelling.** Our numbers (G-M 0.51 vs models <= 0.33, B7 selection
-0.37), and a single readout that decides: sister agreement among survivors only.

**Status.** Nothing drafted to anyone and nothing sent. An email is drafted only when Bence says
so. Contact names and addresses are to be checked on the lab pages at that point, not guessed.
