# Literature sweep 2, 2026-11-03: what is the missing piece?

Working notes. Question: why does every model in the E15-E24 family fit the within-condition data
but underpredict G-CSF vs M-CSF sister agreement (model 0.20-0.25, observed 0.51 [0.33, 0.65],
E20 p < 0.005)? And what did we miss?

## What the original authors say about this experiment

- Weinreb et al. 2020 (Science; PMC7608074) describe the cytokine experiment in one sentence of
  the main text: when split across cytokine conditions, sister cells showed **consistent shifts
  of clone size and observed cell fate** (Supp. Fig. 7e-g), plus "such environmental plasticity
  acts at sub-clonal resolution". 958 clones were sampled between conditions. The supplement
  (with per-condition numbers) could not be retrieved; the condition names come from the public
  data. So the original paper itself links clone *size* and fate across conditions; we only
  tested that in E23 (a coupling exists: under G-CSF larger clones are more N, under M-CSF more
  M, but modelling it did not raise G-M).
- **Design facts from the data that matter** (checked this session): the G-M statistic rests on
  **78 clones**; full cocktail has almost no day-4 cells (one library, 802 cells), G and M have
  day-4 and day-6 libraries; 16% of day-2 cells are already Neu/Mo before the split, but 77 of the
  78 G-M clones had no committed cell sampled at day 2 (their G-M r alone is 0.52).

## Candidate missing pieces, ranked

1. **Clone types, not one continuous bias** (strongest literature support; not yet tested).
   - Dawson lab clone-splitting (SPLINTR barcodes; FASEB 2023, Lorne 2024 abstracts): sister
     clones put into different microenvironments split into **subsets whose fate is
     cell-intrinsic and subsets whose fate depends on the environment**.
   - Two monocyte routes (Yáñez et al. 2017 Immunity): GMP-derived "neutrophil-like" monocytes vs
     MDP-derived monocytes; a later barcoding study supports both routes. MDP-type clones cannot
     make neutrophils in any cytokine.
   - Naik et al. 2013: many progenitors are already lineage-imprinted.
   - Our simulator draws one Gaussian bias per clone and lets every clone respond to cytokines the
     same way. A mixture (a fraction of clones fixed/cytokine-insensitive, the rest plastic)
     raises cross-condition agreement among the fixed clones without inflating within-condition
     variance the same way. This is directly testable on our data (E25).
2. **Cytokine instruction vs selection** (Rieger et al. 2009 Science; Mossadegh-Keller et al.
   2013 Nature: M-CSF instructs PU.1 in single HSCs, no survival advantage by video microscopy).
   Our model treats cytokines as a logit shift (instruction) on all cells. If part of the effect
   is selection of pre-primed clones (emergency-DC work: expansion of pre-primed clones), clone
   presence/size and fate are coupled. E23 tested a size coupling; it did not close the gap.
3. **Pre-split commitment carry-over** (E24). Tested: not supported (fitted carry-over 0 in G and M; G-M 0.21).
4. **Barcode artefacts.** LARRY barcode homoplasy/collisions are a known issue (Advanced Science
   2026, mitochondrial lineage tracing in human haematopoietic organoids: LARRY limits include
   homoplasy, restricted diversity, detection sensitivity). Collisions merge unrelated clones and
   should *lower* agreement, so they are unlikely to explain a too-high G-M; ambient/index-hopping
   barcode reads across libraries were not checked.
5. **Non-Markov kin structure.** The cousin-mother inequality in lineage trees (Sandler et al.
   2015; Chakrabarti 2018; Hughes, Barr & Thomas eLife 2022) shows single hidden-variable
   inheritance models fail; oscillator-gated (circadian/cell-cycle phase) fate control produces
   correlations that do not decay monotonically with kinship. Our E19 ratio assumes a simple
   Markov decay.

## What we missed (process)

- We fitted only summary statistics of each condition. Every model fits them; the G-M
  disagreement is information those summaries do not carry. The original authors pointed at
  clone size; the literature points at clone *types*.
- We never asked whether the 78 G-M clones are a special subset (clones present with >= 2 N+M
  cells in both single-cytokine wells). Selection on presence is plausible under selection-type
  cytokine effects.
- Imaging-based work answers the question we only infer: Buggenthin et al. 2017 (Nature Methods)
  predict GM vs MegE choice from brightfield time-lapse up to three generations before markers
  (AUC 0.87); Rieger 2009 and Hoppe 2016 follow single GMPs and sisters directly. Sister-pair
  time-lapse under G-CSF vs M-CSF would test clone types directly; scRNA snapshots cannot.

## Startups and groups with a similar goal

- **CellVoyant** (Bristol, University of Bristol spin-out, founded 2021): AI + live-cell imaging
  to predict stem-cell behaviour (FateView, FateCast; phase-contrast prediction of iPSC quality
  attributes with Pluristyx). £7.6m seed, Jan 2024 (Octopus Ventures lead; Horizons, Verve, Air
  Street). Bence interviewed there in January 2024 (coding session 22 Jan 2024, not progressed;
  his RL presentation was shared with the team).
- Imaging/phenomics at scale: **Recursion** (phenomics OS), **insitro** (stem-cell disease
  models, CellPaint-POSH).
- Cell-state / virtual-cell: **Cellarity** (cell-state drug discovery; CLY-124), **NewLimit**
  (TF-combination rejuvenation, ~$247m raised), **Xaira** (X-Cell virtual cell model), **Tahoe**
  (perturbation datasets), **BioMap** (won Arc Virtual Cell Challenge 2025). Arc Institute's 2026
  challenge is zero-shot; organisers say models do not yet consistently beat naive baselines.
- Academic tools in the same space: AI4CellFate (Crick/Oxford, 2025 preprint; fate from live-cell
  imaging, often from the first frame), RegVelo (Cell 2026; dynamics + GRNs).
- Not verified this session: Cellino, Deepcell, bit.bio, Mogrify, clock.bio, Shift Bioscience
  (no current news found).

## Sources

- Weinreb et al. 2020, Science, doi 10.1126/science.aaw3381 (PMC7608074).
- Rieger et al. 2009, Science 325:217 ("Hematopoietic cytokines can instruct lineage choice").
- Mossadegh-Keller et al. 2013, Nature 497:239 (M-CSF instructs myeloid fate in single HSCs).
- Yáñez et al. 2017, Immunity 47:890 (GMP and MDP produce distinct monocytes).
- Naik et al. 2013, Nature 496:229.
- Dawson lab abstracts: FASEB 2023 (doi 10.1096/HM2023.S01.P01); Lorne Cancer 2024; Lorne Genome 2024.
- Buggenthin et al. 2017, Nature Methods 14:403, doi 10.1038/nmeth.4182.
- Sandler et al. 2015 Nature; Hughes, Barr & Thomas 2022 eLife 11:e80927.
- Single-cell mitochondrial lineage tracing in human haematopoietic organoids, Advanced Science 2026,
  doi 10.1002/advs.202518084.
- CellVoyant: University of Bristol news, Jan 2024 (https://www.bristol.ac.uk/news/2024/january/cellvoyant-funding.html).
- Arc Virtual Cell Challenge 2026: https://arcinstitute.org/news/virtual-cell-challenge-2026.
- AI4CellFate: https://www.biorxiv.org/content/10.1101/2025.05.12.653464v1.
